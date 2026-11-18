"""ECS execution backend for the runner listener.

The listener keeps its existing API contract: it still reports a `job_name`
(here, the ECS task ARN), still POSTs `job-status`, still PUTs `log-stream`, and
still answers `cancel_job`. Only the thing behind that contract changes -- an
ECS ``RunTask`` instead of a Kubernetes Job.

Kubernetes remains the default. This module is imported lazily by the listener
only when ``runners.execution.backend == "ecs"``, so a Kubernetes deployment's
import graph and runtime behaviour are unchanged.

All AWS calls go through aiobotocore directly (the repository already uses it
for S3/KMS) with bounded retry and backoff. A definitive 4xx is final; only
connection errors and the documented throttling/transient codes are retried.
"""

from __future__ import annotations

import asyncio
from typing import Any

from aiobotocore.session import get_session
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError

from terrapod.config import EcsExecutionConfig
from terrapod.logging_config import get_logger
from terrapod.runner.ecs_task import task_identifier

logger = get_logger(__name__)

#: Sent on every RunTask and used to scope ListTasks. Without it a listener
#: would count unrelated tasks in the same cluster as occupying its pool.
STARTED_BY = "terrapod-runner"

_MAX_ATTEMPTS = 3
_BACKOFF_BASE_SECONDS = 0.25
_RETRYABLE_CODES = frozenset(
    {
        "Throttling",
        "ThrottlingException",
        "ThrottledException",
        "RequestThrottled",
        "RequestThrottledException",
        "TooManyRequestsException",
        "ServiceUnavailable",
        "ServiceUnavailableException",
        "InternalFailure",
        "InternalServerError",
        "ConnectionError",
        "RequestTimeout",
        "RequestTimeoutException",
    }
)
_TERMINAL_STATES = frozenset({"STOPPED", "FAILED"})
_RUNNING_STATES = frozenset(
    {"PROVISIONING", "PENDING", "ACTIVATING", "RUNNING", "DEACTIVATING", "STOPPING"}
)


class EcsExecutionError(RuntimeError):
    """An ECS/CloudWatch call failed after bounded retries."""

    def __init__(self, message: str, *, code: str = "") -> None:
        super().__init__(message)
        self.code = code


class EcsExecutionBackend:
    """Launch and observe runner Tasks in ECS."""

    def __init__(self, config: EcsExecutionConfig) -> None:
        self._config = config
        self._session = get_session()
        self._clients: dict[str, Any] = {}
        self._contexts: dict[str, Any] = {}

    async def _client(self, service: str) -> Any:
        if service not in self._clients:
            # botocore's own retries are OFF: this module owns the bounded
            # backoff so the policy is visible and testable in one place.
            context = self._session.create_client(
                service,
                region_name=self._config.region or None,
                config=Config(
                    retries={"max_attempts": 1, "mode": "standard"},
                    connect_timeout=10,
                    read_timeout=60,
                ),
            )
            self._clients[service] = await context.__aenter__()
            self._contexts[service] = context
        return self._clients[service]

    async def close(self) -> None:
        """Close every lazily created client (called by the listener's shutdown)."""
        contexts = list(self._contexts.values())
        self._contexts.clear()
        self._clients.clear()
        for context in contexts:
            await context.__aexit__(None, None, None)

    async def _call(self, service: str, operation: str, **kwargs: Any) -> dict[str, Any]:
        last_error: Exception | None = None
        for attempt in range(_MAX_ATTEMPTS):
            try:
                client = await self._client(service)
                return await getattr(client, operation)(**kwargs)
            except ClientError as e:
                code = str(e.response.get("Error", {}).get("Code", ""))
                last_error = e
                if code not in _RETRYABLE_CODES or attempt == _MAX_ATTEMPTS - 1:
                    raise EcsExecutionError(
                        f"{service}.{operation} failed: {e}", code=code
                    ) from e
            except (BotoCoreError, OSError) as e:
                last_error = e
                if attempt == _MAX_ATTEMPTS - 1:
                    raise EcsExecutionError(f"{service}.{operation} failed: {e}") from e
            await asyncio.sleep(_BACKOFF_BASE_SECONDS * (2**attempt))
        # Kept for type checkers: every loop path either returns or raises.
        raise EcsExecutionError(f"{service}.{operation} failed: {last_error}")

    async def run_task(self, kwargs: dict[str, Any]) -> str:
        """Start one runner Task and return its ARN."""
        request = dict(kwargs)
        request.setdefault("startedBy", STARTED_BY)
        response = await self._call("ecs", "run_task", **request)
        failures = response.get("failures") or []
        tasks = response.get("tasks") or []
        if not tasks:
            detail = "; ".join(str(f.get("reason", f)) for f in failures) or "no task returned"
            raise EcsExecutionError(f"ECS RunTask did not start a task: {detail}")
        return str(tasks[0]["taskArn"])

    async def describe_task(self, task_arn: str) -> dict[str, Any] | None:
        """Describe one Task, or None when ECS no longer knows about it."""
        response = await self._call(
            "ecs", "describe_tasks", cluster=self._config.cluster, tasks=[task_arn]
        )
        for failure in response.get("failures") or []:
            if "NOT_FOUND" in str(failure.get("reason", "")):
                return None
        tasks = response.get("tasks") or []
        return tasks[0] if tasks else None

    async def task_status(self, task_arn: str) -> dict[str, Any]:
        """Translate ECS task state into the listener→API job-status shape."""
        task = await self.describe_task(task_arn)
        if task is None:
            return {"status": "deleted", "terminal": True}

        last_status = str(task.get("lastStatus") or "").upper()

        if last_status in _RUNNING_STATES or (last_status not in _TERMINAL_STATES):
            # A task whose desiredStatus is STOPPED is still running until ECS
            # reports STOPPED. The API must not mark the run finished early.
            return {"status": "running", "terminal": False}

        exit_code: int | None = None
        reason = str(task.get("stoppedReason") or "")
        containers = task.get("containers") or []
        container = next(
            (c for c in containers if c.get("name") == self._config.container_name),
            containers[0] if containers else {},
        )
        raw_exit = container.get("exitCode")
        if isinstance(raw_exit, int):
            exit_code = raw_exit
        container_reason = str(container.get("reason") or "")
        if container_reason:
            reason = container_reason

        result: dict[str, Any] = {"terminal": True}
        if exit_code == 0:
            result["status"] = "succeeded"
        elif exit_code is not None:
            result["status"] = "failed"
            result["exit_code"] = exit_code
        elif "cancel" in reason.lower() or "stopped by" in reason.lower():
            result["status"] = "deleted"
        else:
            result["status"] = "failed"
        if reason:
            result["reason"] = reason
        return result

    async def list_running(self) -> set[str]:
        """ARNs of this pool's non-terminal Tasks, or raise on failure.

        The listener treats a raised error as "occupancy unknown" and fails
        closed, exactly as it does for a failed Kubernetes list.
        """
        arns: set[str] = set()
        token: str | None = None
        while True:
            kwargs: dict[str, Any] = {
                "cluster": self._config.cluster,
                "desiredStatus": "RUNNING",
                "startedBy": STARTED_BY,
                "maxResults": 100,
            }
            if token:
                kwargs["nextToken"] = token
            response = await self._call("ecs", "list_tasks", **kwargs)
            arns.update(str(arn) for arn in response.get("taskArns") or [])
            token = response.get("nextToken")
            if not token:
                return arns

    async def logs(self, task_arn: str, tail_lines: int = 500) -> str:
        """Read the runner container's CloudWatch stream as plain text."""
        if not self._config.log_group:
            return ""
        stream = (
            f"{self._config.log_stream_prefix}/{self._config.container_name}/"
            f"{task_identifier(task_arn)}"
        )
        limit = max(1, min(tail_lines, 1000))
        events: list[dict[str, Any]] = []
        token: str | None = None
        previous: str | None = None
        for _ in range(10):
            kwargs: dict[str, Any] = {
                "logGroupName": self._config.log_group,
                "logStreamNames": [stream],
                "limit": limit,
            }
            if token:
                kwargs["nextToken"] = token
            else:
                kwargs["startFromHead"] = False
            try:
                response = await self._call("logs", "filter_log_events", **kwargs)
            except EcsExecutionError as e:
                if e.code == "ResourceNotFoundException":
                    return ""  # stream not created yet — the ordinary startup case
                raise
            page = response.get("events") or []
            events = [*page, *events]
            if len(events) >= limit:
                break
            token = response.get("nextBackwardToken")
            if not token or token == previous:
                break
            previous = token

        ordered = sorted(events, key=lambda e: (e.get("timestamp", 0), e.get("ingestionTime", 0)))
        lines = [_message(event) for event in ordered[-limit:]]
        text = "\n".join(line for line in lines if line)
        return text[-256 * 1024 :]

    async def stop_task(self, task_arn: str, reason: str = "Canceled by Terrapod listener") -> None:
        """Stop a Task for a canceled run."""
        await self._call(
            "ecs",
            "stop_task",
            cluster=self._config.cluster,
            task=task_arn,
            reason=reason,
        )


def _message(event: dict[str, Any]) -> str:
    """Decode one CloudWatch event's message, tolerating binary payloads."""
    raw = event.get("message", "")
    if isinstance(raw, bytes):
        return raw.decode("utf-8", errors="replace")
    return str(raw)
