"""ECS/CloudWatch backend behaviour used by the listener (#ECS).

The listener↔API contract (job-launched, job-status, log-stream, cancel) does
not change when runs move from Kubernetes Jobs to ECS Tasks. These tests pin the
translation at that boundary, plus the retry and fail-closed rules that keep a
transient AWS blip from dropping a run or over-admitting the pool.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest
from botocore.exceptions import BotoCoreError, ClientError

from terrapod.config import EcsExecutionConfig
from terrapod.runner.ecs_backend import (
    STARTED_BY,
    EcsExecutionBackend,
    EcsExecutionError,
)


def _config(**overrides: object) -> EcsExecutionConfig:
    payload: dict[str, object] = {
        "cluster": "runner-cluster",
        "task_definition": "terrapod-runner:1",
        "region": "eu-west-1",
        "subnets": ["subnet-a"],
        "log_group": "/terrapod/runner",
    }
    payload.update(overrides)
    return EcsExecutionConfig(**payload)


@pytest.fixture
def backend() -> EcsExecutionBackend:
    # The SDK session is created lazily and never used in these tests: `_call`
    # is replaced at the client boundary below. Keeping construction inert also
    # proves no AWS credential lookup happens before the ECS backend is chosen.
    with patch("terrapod.runner.ecs_backend.get_session", MagicMock()):
        return EcsExecutionBackend(_config())


def _client_error(code: str) -> ClientError:
    return ClientError({"Error": {"Code": code, "Message": "test error"}}, "TestOperation")


class TestBoundedRetries:
    @pytest.mark.asyncio
    async def test_throttling_is_retried_with_exponential_backoff(
        self, backend: EcsExecutionBackend
    ) -> None:
        client = SimpleNamespace(
            run_task=AsyncMock(side_effect=[_client_error("Throttling"), {"ok": True}])
        )
        backend._client = AsyncMock(return_value=client)
        with patch("terrapod.runner.ecs_backend.asyncio.sleep", new_callable=AsyncMock) as sleep:
            result = await backend._call("ecs", "run_task", answer=42)

        assert result == {"ok": True}
        assert client.run_task.await_count == 2
        assert sleep.await_args_list == [call(0.25)]

    @pytest.mark.asyncio
    async def test_definitive_four_xx_is_never_retried(
        self, backend: EcsExecutionBackend
    ) -> None:
        client = SimpleNamespace(run_task=AsyncMock(side_effect=_client_error("ValidationError")))
        backend._client = AsyncMock(return_value=client)
        with patch("terrapod.runner.ecs_backend.asyncio.sleep", new_callable=AsyncMock) as sleep:
            with pytest.raises(EcsExecutionError) as exc:
                await backend._call("ecs", "run_task")

        assert exc.value.code == "ValidationError"
        assert client.run_task.await_count == 1
        sleep.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_connection_errors_stop_after_three_attempts(
        self, backend: EcsExecutionBackend
    ) -> None:
        client = SimpleNamespace(
            run_task=AsyncMock(side_effect=[BotoCoreError(), BotoCoreError(), BotoCoreError()])
        )
        backend._client = AsyncMock(return_value=client)
        with patch("terrapod.runner.ecs_backend.asyncio.sleep", new_callable=AsyncMock) as sleep:
            with pytest.raises(EcsExecutionError):
                await backend._call("ecs", "run_task")

        assert client.run_task.await_count == 3
        assert sleep.await_args_list == [call(0.25), call(0.5)]


class TestTaskLifecycle:
    @pytest.mark.asyncio
    async def test_run_task_marks_every_task_as_terrapod_managed(
        self, backend: EcsExecutionBackend
    ) -> None:
        arn = "arn:aws:ecs:eu-west-1:000000000000:task/runner-cluster/abc123"
        backend._call = AsyncMock(return_value={"tasks": [{"taskArn": arn}]})

        assert await backend.run_task({"cluster": "runner-cluster"}) == arn
        kwargs = backend._call.await_args.kwargs
        assert kwargs["cluster"] == "runner-cluster"
        assert kwargs["startedBy"] == STARTED_BY

    @pytest.mark.asyncio
    async def test_run_task_reports_ecs_failures_instead_of_losing_the_run(
        self, backend: EcsExecutionBackend
    ) -> None:
        backend._call = AsyncMock(
            return_value={"tasks": [], "failures": [{"reason": "no capacity"}]}
        )

        with pytest.raises(EcsExecutionError, match="no capacity"):
            await backend.run_task({})

    @pytest.mark.asyncio
    async def test_describe_task_returns_none_when_ecs_has_forgotten_it(
        self, backend: EcsExecutionBackend
    ) -> None:
        backend._call = AsyncMock(
            return_value={"failures": [{"reason": "referenced task was not found"}]}
        )

        assert await backend.describe_task("arn:missing") is None
        backend._call.assert_awaited_once_with(
            "ecs", "describe_tasks", cluster="runner-cluster", tasks=["arn:missing"]
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("task", "expected"),
        [
            (None, {"status": "deleted", "terminal": True}),
            (
                {"lastStatus": "RUNNING", "desiredStatus": "STOPPED"},
                {"status": "running", "terminal": False},
            ),
            ({"lastStatus": "PROVISIONING"}, {"status": "running", "terminal": False}),
            (
                {
                    "lastStatus": "STOPPED",
                    "containers": [{"name": "runner", "exitCode": 0}],
                },
                {"terminal": True, "status": "succeeded"},
            ),
            (
                {
                    "lastStatus": "STOPPED",
                    "stoppedReason": "essential container exited",
                    "containers": [
                        {"name": "runner", "exitCode": 1, "reason": "terraform failed"}
                    ],
                },
                {
                    "terminal": True,
                    "status": "failed",
                    "exit_code": 1,
                    "reason": "terraform failed",
                },
            ),
            (
                {"lastStatus": "STOPPED", "stoppedReason": "Task stopped by user"},
                {"terminal": True, "status": "deleted", "reason": "Task stopped by user"},
            ),
            (
                {"lastStatus": "STOPPED"},
                {"terminal": True, "status": "failed"},
            ),
        ],
    )
    async def test_ecs_state_maps_to_the_existing_job_status_contract(
        self,
        backend: EcsExecutionBackend,
        task: dict[str, object] | None,
        expected: dict[str, object],
    ) -> None:
        backend.describe_task = AsyncMock(return_value=task)

        assert await backend.task_status("arn:task") == expected

    @pytest.mark.asyncio
    async def test_list_running_pages_through_only_this_pools_tasks(
        self, backend: EcsExecutionBackend
    ) -> None:
        backend._call = AsyncMock(
            side_effect=[
                {"taskArns": ["arn:one"], "nextToken": "page-2"},
                {"taskArns": ["arn:two"]},
            ]
        )

        assert await backend.list_running() == {"arn:one", "arn:two"}
        assert backend._call.await_args_list == [
            call(
                "ecs",
                "list_tasks",
                cluster="runner-cluster",
                desiredStatus="RUNNING",
                startedBy=STARTED_BY,
                maxResults=100,
            ),
            call(
                "ecs",
                "list_tasks",
                cluster="runner-cluster",
                desiredStatus="RUNNING",
                startedBy=STARTED_BY,
                maxResults=100,
                nextToken="page-2",
            ),
        ]

    @pytest.mark.asyncio
    async def test_list_running_raises_so_the_listener_fails_closed(
        self, backend: EcsExecutionBackend
    ) -> None:
        backend._call = AsyncMock(side_effect=EcsExecutionError("ECS unavailable"))

        with pytest.raises(EcsExecutionError):
            await backend.list_running()

    @pytest.mark.asyncio
    async def test_stop_task_uses_the_configured_cluster(
        self, backend: EcsExecutionBackend
    ) -> None:
        backend._call = AsyncMock()

        await backend.stop_task("arn:task")

        backend._call.assert_awaited_once_with(
            "ecs",
            "stop_task",
            cluster="runner-cluster",
            task="arn:task",
            reason="Canceled by Terrapod listener",
        )


class TestLogsAndCleanup:
    @pytest.mark.asyncio
    async def test_logs_read_the_runner_stream_backwards_in_order(
        self, backend: EcsExecutionBackend
    ) -> None:
        backend._call = AsyncMock(
            side_effect=[
                {
                    "events": [{"timestamp": 1, "message": "first"}],
                    "nextBackwardToken": "cursor",
                },
                {
                    "events": [{"timestamp": 2, "message": b"second"}],
                    "nextBackwardToken": "cursor",
                },
            ]
        )

        task_arn = "arn:aws:ecs:eu-west-1:000000000000:task/runner-cluster/abc123"
        assert await backend.logs(task_arn, tail_lines=10) == "first\nsecond"
        assert backend._call.await_args_list == [
            call(
                "logs",
                "filter_log_events",
                logGroupName="/terrapod/runner",
                logStreamNames=["terrapod-runner/runner/abc123"],
                limit=10,
                startFromHead=False,
            ),
            call(
                "logs",
                "filter_log_events",
                logGroupName="/terrapod/runner",
                logStreamNames=["terrapod-runner/runner/abc123"],
                limit=10,
                nextToken="cursor",
            ),
        ]

    @pytest.mark.asyncio
    async def test_a_not_yet_created_stream_is_not_an_error(
        self, backend: EcsExecutionBackend
    ) -> None:
        backend._call = AsyncMock(
            side_effect=EcsExecutionError("missing", code="ResourceNotFoundException")
        )

        assert await backend.logs("arn:task") == ""

    @pytest.mark.asyncio
    async def test_no_log_group_means_no_cloudwatch_call(
        self, backend: EcsExecutionBackend
    ) -> None:
        backend._config = _config(log_group="")
        backend._call = AsyncMock()

        assert await backend.logs("arn:task") == ""
        backend._call.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_close_exits_every_lazily_created_client(self) -> None:
        with patch("terrapod.runner.ecs_backend.get_session", MagicMock()):
            backend = EcsExecutionBackend(_config())
        contexts = [SimpleNamespace(__aexit__=AsyncMock()), SimpleNamespace(__aexit__=AsyncMock())]
        backend._contexts = {"ecs": contexts[0], "logs": contexts[1]}
        backend._clients = {"ecs": object(), "logs": object()}

        await asyncio.wait_for(backend.close(), timeout=1)

        for context in contexts:
            context.__aexit__.assert_awaited_once_with(None, None, None)
        assert backend._contexts == {}
        assert backend._clients == {}
