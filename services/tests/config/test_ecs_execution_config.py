"""Validation for the opt-in execution backend config (#ECS).

The Kubernetes path is the default and must stay byte-for-byte inert when no
ECS values are set. Once an operator selects ECS, a half-configured block would
otherwise queue every run until the launch timeout, so the three required keys
are checked when `RunnerConfig` is built rather than at the first launch.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from terrapod.config import EcsExecutionConfig, RunnerConfig, RunnerExecutionConfig


def _ecs_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "cluster": "runner-cluster",
        "task_definition": "terrapod-runner:1",
        "subnets": ["subnet-abc"],
    }
    payload.update(overrides)
    return payload


def test_kubernetes_is_the_default_and_an_empty_ecs_block_is_inert() -> None:
    config = RunnerExecutionConfig()

    assert config.backend == "kubernetes"
    assert config.ecs == EcsExecutionConfig()
    assert config.ecs.cluster == ""
    assert config.ecs.task_definition == ""
    assert config.ecs.subnets == []


def test_ecs_selection_accepts_the_minimum_required_config() -> None:
    config = RunnerExecutionConfig(backend="ecs", ecs=_ecs_payload())

    assert config.backend == "ecs"
    assert config.ecs.cluster == "runner-cluster"
    assert config.ecs.launch_type == "FARGATE"
    assert config.ecs.assign_public_ip is False


@pytest.mark.parametrize("missing", ["cluster", "task_definition", "subnets"])
def test_ecs_selection_fails_at_startup_when_a_required_key_is_missing(missing: str) -> None:
    payload = _ecs_payload()
    del payload[missing]

    with pytest.raises(ValidationError, match=f"execution\\.ecs\\.{missing}"):
        RunnerExecutionConfig(backend="ecs", ecs=payload)


def test_an_empty_subnet_list_is_not_a_valid_ecs_config() -> None:
    with pytest.raises(ValidationError, match="execution\\.ecs\\.subnets"):
        RunnerExecutionConfig(backend="ecs", ecs=_ecs_payload(subnets=[]))


def test_an_unknown_backend_is_refused() -> None:
    with pytest.raises(ValidationError, match=r"backend\s+Input should be"):
        RunnerExecutionConfig(backend="nomad")  # type: ignore[arg-type]


def test_runner_config_layers_nested_execution_settings_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """The Helm ConfigMap renders nested YAML; env overrides use the same paths."""
    monkeypatch.setenv("TERRAPOD_EXECUTION__BACKEND", "ecs")
    monkeypatch.setenv("TERRAPOD_EXECUTION__ECS__CLUSTER", "runner-cluster")
    monkeypatch.setenv("TERRAPOD_EXECUTION__ECS__TASK_DEFINITION", "terrapod-runner:1")
    monkeypatch.setenv("TERRAPOD_EXECUTION__ECS__SUBNETS", '["subnet-abc"]')
    monkeypatch.setenv("TERRAPOD_EXECUTION__ECS__ASSIGN_PUBLIC_IP", "true")

    config = RunnerConfig()

    assert config.execution.backend == "ecs"
    assert config.execution.ecs.cluster == "runner-cluster"
    assert config.execution.ecs.task_definition == "terrapod-runner:1"
    assert config.execution.ecs.subnets == ["subnet-abc"]
    assert config.execution.ecs.assign_public_ip is True
