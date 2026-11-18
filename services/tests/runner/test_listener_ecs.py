"""Listener branch selection for the opt-in ECS execution backend (#ECS).

Kubernetes stays the default and its code path is deliberately untouched. These
tests exercise the ECS branches through the real listener methods: occupancy
reads fail closed, the ECS Task ARN is reported as the opaque `job_name`, and
status/logs/cancel keep the existing listener→API wire contract.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from terrapod.config import EcsExecutionConfig, RunnerExecutionConfig
from terrapod.runner.listener import RunnerListener


def _client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(base_url="http://test", transport=httpx.MockTransport(handler))


def _listener(client: httpx.AsyncClient, backend: str = "kubernetes") -> RunnerListener:
    listener = RunnerListener()
    listener._http_client = client
    listener.identity = SimpleNamespace(
        listener_id="11111111-1111-1111-1111-111111111111",
        name="listener",
        api_url="http://test",
        certificate_pem="",
    )
    listener._execution_backend = backend
    listener._ecs_backend = None
    return listener


def _ecs_config() -> EcsExecutionConfig:
    return EcsExecutionConfig(
        cluster="runner-cluster",
        task_definition="terrapod-runner:1",
        subnets=["subnet-a"],
    )


class TestLazyBackendSelection:
    @pytest.mark.asyncio
    async def test_the_ecs_client_is_created_once_and_only_on_demand(self) -> None:
        async with _client(lambda request: httpx.Response(200)) as client:
            listener = _listener(client)
            fake_backend = object()
            with patch(
                "terrapod.runner.ecs_backend.EcsExecutionBackend", return_value=fake_backend
            ) as constructor:
                assert await listener._get_ecs_backend() is fake_backend
                assert await listener._get_ecs_backend() is fake_backend

            constructor.assert_called_once_with(listener.runner_config.execution.ecs)


class TestCapacityAdmission:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("occupying", "expected_claims"),
        [(["arn:one", "arn:two"], 0), (["arn:one"], 1)],
    )
    async def test_ecs_task_count_gates_claims_like_kubernetes_jobs(
        self, occupying: list[str], expected_claims: int
    ) -> None:
        calls = {"next": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path.endswith("/runs/next"):
                calls["next"] += 1
                return httpx.Response(204)
            return httpx.Response(200, json={"ok": True})

        async with _client(handler) as client:
            listener = _listener(client, backend="ecs")
            listener._max_concurrent = 2
            listener._active_launches = 0
            backend = SimpleNamespace(list_running=AsyncMock(return_value=set(occupying)))
            listener._get_ecs_backend = AsyncMock(return_value=backend)

            await listener._handle_run_available()

        assert calls["next"] == expected_claims

    @pytest.mark.asyncio
    async def test_an_ecs_listing_error_fails_closed(self) -> None:
        calls = {"next": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path.endswith("/runs/next"):
                calls["next"] += 1
                return httpx.Response(204)
            return httpx.Response(200, json={"ok": True})

        async with _client(handler) as client:
            listener = _listener(client, backend="ecs")
            listener._max_concurrent = 10
            listener._active_launches = 0
            backend = SimpleNamespace(list_running=AsyncMock(side_effect=RuntimeError("ECS down")))
            listener._get_ecs_backend = AsyncMock(return_value=backend)

            await listener._handle_run_available()

        assert calls["next"] == 0


class TestLaunchPayload:
    def test_reserved_keys_from_run_variables_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="reserved ECS payload key"):
            RunnerListener._ecs_secret_data(
                runner_token="token",
                terraform_vars=[],
                env_vars=[{"key": "token", "value": "collision"}],
                execution_hooks=None,
                git_auth=None,
                ansible_auth=None,
                vault_file_values=None,
                ca_pem="",
            )

    def test_token_and_ca_are_added_to_the_same_payload(self) -> None:
        data = RunnerListener._ecs_secret_data(
            runner_token="runner-token",
            terraform_vars=[{"key": "region", "value": "eu-west-1"}],
            env_vars=[{"key": "TF_VAR_color", "value": "blue"}],
            execution_hooks=None,
            git_auth=None,
            ansible_auth=None,
            vault_file_values={"vault-file-0": "secret"},
            ca_pem="CUSTOM CA",
        )

        assert data["token"] == "runner-token"
        assert data["ca-extra.pem"] == "CUSTOM CA"
        assert data["TF_VAR_color"] == "blue"
        assert data["vault-file-0"] == "secret"
        assert "terraform.tfvars.json" in data

    @pytest.mark.asyncio
    async def test_launch_reports_the_task_arn_and_cluster(self) -> None:
        async with _client(lambda request: httpx.Response(200)) as client:
            listener = _listener(client, backend="ecs")
            listener.runner_config.execution = RunnerExecutionConfig(backend="ecs", ecs=_ecs_config())
            backend = SimpleNamespace(
                run_task=AsyncMock(return_value="arn:aws:ecs:eu-west-1:000000000000:task/abc")
            )
            listener._get_ecs_backend = AsyncMock(return_value=backend)
            listener._report_job_launched = AsyncMock()
            listener._report_launch_failed = AsyncMock()

            with patch(
                "terrapod.runner.ecs_task.run_task_kwargs", return_value={"cluster": "x"}
            ) as build:
                arn = await listener._launch_ecs_run(
                    run_id="01ABC",
                    phase="plan",
                    spec={"spec": {}},
                    secret_data={"token": "t"},
                    vault_mounts=[],
                )

            assert arn == "arn:aws:ecs:eu-west-1:000000000000:task/abc"
            build.assert_called_once()
            backend.run_task.assert_awaited_once_with({"cluster": "x"})
            listener._report_job_launched.assert_awaited_once_with(
                "01ABC", arn, "runner-cluster"
            )
            listener._report_launch_failed.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_a_run_requiring_an_ansible_execution_environment_is_refused(self) -> None:
        async with _client(lambda request: httpx.Response(200)) as client:
            listener = _listener(client, backend="ecs")
            listener._report_launch_failed = AsyncMock()
            listener._get_ecs_backend = MagicMock()

            assert (
                await listener._launch_run("01ABC", {"phase": "plan", "ee-image": "ee:latest"})
                == ""
            )
            listener._report_launch_failed.assert_awaited_once_with(
                "01ABC", "Ansible execution environments are not supported by the ECS backend"
            )
            listener._get_ecs_backend.assert_not_called()


class TestStatusLogsAndCancel:
    @pytest.mark.asyncio
    async def test_ecs_task_state_is_posted_in_the_job_status_body(self) -> None:
        requests: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return httpx.Response(200, json={"ok": True})

        async with _client(handler) as client:
            listener = _listener(client, backend="ecs")
            backend = SimpleNamespace(
                task_status=AsyncMock(
                    return_value={
                        "status": "failed",
                        "terminal": True,
                        "exit_code": 137,
                        "reason": "OOMKilled",
                    }
                )
            )
            listener._get_ecs_backend = AsyncMock(return_value=backend)

            await listener._handle_check_job_status(
                {
                    "run_id": "01ABC",
                    "job_name": "arn:task",
                    "phase": "apply",
                    "job_namespace": "runner-cluster",
                }
            )

        assert requests[0].url.path.endswith("/runs/run-01ABC/job-status")
        assert json.loads(requests[0].content) == {
            "status": "failed",
            "terminal": True,
            "phase": "apply",
            "exit_code": 137,
            "reason": "OOMKilled",
        }
        backend.task_status.assert_awaited_once_with("arn:task")

    @pytest.mark.asyncio
    async def test_cloudwatch_output_is_put_to_the_log_stream_endpoint(self) -> None:
        requests: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return httpx.Response(200, json={"ok": True})

        async with _client(handler) as client:
            listener = _listener(client, backend="ecs")
            backend = SimpleNamespace(logs=AsyncMock(return_value="line one\nline two"))
            listener._get_ecs_backend = AsyncMock(return_value=backend)

            await listener._handle_stream_logs(
                {
                    "run_id": "01ABC",
                    "job_name": "arn:task",
                    "phase": "plan",
                    "tail_lines": 12,
                }
            )

        assert requests[0].method == "PUT"
        assert requests[0].url.path.endswith("/runs/run-01ABC/log-stream")
        assert requests[0].url.params["phase"] == "plan"
        assert requests[0].content == b"line one\nline two"
        backend.logs.assert_awaited_once_with("arn:task", 12)

    @pytest.mark.asyncio
    async def test_cancel_stops_the_ecs_task(self) -> None:
        async with _client(lambda request: httpx.Response(200)) as client:
            listener = _listener(client, backend="ecs")
            backend = SimpleNamespace(stop_task=AsyncMock())
            listener._get_ecs_backend = AsyncMock(return_value=backend)

            await listener._handle_cancel_job({"job_name": "arn:task"})

            backend.stop_task.assert_awaited_once_with("arn:task")
