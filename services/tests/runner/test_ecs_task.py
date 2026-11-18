"""Translate a listener Job spec into an ECS ``RunTask`` request (#ECS).

The listener keeps building the canonical Kubernetes Job spec through the engine
strategy. Under the opt-in ECS backend that spec is only a description of the
run: these tests pin the environment/file channels and the resource translation,
because a silently dropped Secret is a run that starts and never reports back.
"""

from __future__ import annotations

import base64
import json
from typing import Any

import pytest

from terrapod.config import EcsExecutionConfig
from terrapod.runner.ecs_task import (
    environment,
    files,
    run_task_kwargs,
    task_identifier,
)


def _spec(
    *,
    env: list[dict[str, Any]] | None = None,
    cpu: str = "500m",
    memory: str = "1Gi",
    image: str = "runner:latest",
) -> dict[str, Any]:
    return {
        "spec": {
            "template": {
                "spec": {
                    "containers": [
                        {
                            "name": "runner",
                            "image": image,
                            "env": env if env is not None else [],
                            "resources": {"requests": {"cpu": cpu, "memory": memory}},
                        }
                    ]
                }
            }
        }
    }


def _config(**overrides: object) -> EcsExecutionConfig:
    payload: dict[str, Any] = {
        "cluster": "runner-cluster",
        "task_definition": "terrapod-runner:1",
        "subnets": ["subnet-a", "subnet-b"],
    }
    payload.update(overrides)
    return EcsExecutionConfig(**payload)


class TestEnvironmentTranslation:
    def test_direct_values_and_secret_references_are_resolved(self) -> None:
        spec = _spec(
            env=[
                {"name": "TP_API_URL", "value": "http://api"},
                {
                    "name": "TP_TOKEN",
                    "valueFrom": {"secretKeyRef": {"name": "auth", "key": "token"}},
                },
            ]
        )

        assert environment(spec, {"token": "secret-token"}) == [
            {"name": "TP_API_URL", "value": "http://api"},
            {"name": "TP_TOKEN", "value": "secret-token"},
        ]

    def test_a_missing_secret_value_fails_before_the_task_starts(self) -> None:
        spec = _spec(
            env=[
                {
                    "name": "TP_TOKEN",
                    "valueFrom": {"secretKeyRef": {"name": "auth", "key": "token"}},
                }
            ]
        )

        with pytest.raises(ValueError, match="TP_TOKEN"):
            environment(spec, {})

    def test_null_values_are_skipped_rather_than_sent_as_literal_none(self) -> None:
        spec = _spec(env=[{"name": "TP_OPTIONAL", "value": None}])

        assert environment(spec, {}) == []


class TestFileTranslation:
    def test_secret_keys_are_mapped_to_the_paths_the_runner_reads(self) -> None:
        spec = _spec(
            env=[
                {
                    "name": "TP_API_KEY",
                    "valueFrom": {"secretKeyRef": {"name": "vars", "key": "api-key"}},
                }
            ]
        )
        secret_data = {
            "api-key": "env-value",
            "token": "runner-token",
            "terraform.tfvars.json": "[]",
            "execution-hooks.json": "[]",
            "git-auth.json": "[]",
            "ansible-auth.json": "[]",
            "ca-extra.pem": "custom-ca",
            "vault-file-0": "vault-secret",
        }

        assert files(spec, secret_data, [{"name": "app.conf", "secret_key": "vault-file-0"}]) == {
            "/var/run/terrapod/vars/terraform.tfvars.json": "[]",
            "/var/run/terrapod/vars/execution-hooks.json": "[]",
            "/var/run/terrapod/vars/git-auth.json": "[]",
            "/var/run/terrapod/vars/ansible-auth.json": "[]",
            "/var/run/terrapod/files/app.conf": "vault-secret",
        }

    def test_an_unmapped_key_is_rejected_instead_of_dropped(self) -> None:
        with pytest.raises(ValueError, match="unmapped runner file key"):
            files(_spec(), {"new-channel": "payload"}, [])


class TestRunTaskTranslation:
    def test_the_full_request_carries_the_runner_payload(self) -> None:
        config = _config(
            ephemeral_storage_gib=64,
            task_role_arn="arn:aws:iam::000000000000:role/runner",
            execution_role_arn="arn:aws:iam::000000000000:role/execution",
            assign_public_ip=True,
            security_groups=["sg-1"],
            tags={"team": "platform", "terrapod-managed": "custom"},
        )
        spec = _spec(
            env=[
                {
                    "name": "TP_TOKEN",
                    "valueFrom": {"secretKeyRef": {"name": "auth", "key": "token"}},
                }
            ]
        )
        secret_data = {
            "token": "secret-token",
            "terraform.tfvars.json": '[{"key":"region","value":"eu-west-1"}]',
            "ca-extra.pem": "CUSTOM CA",
        }

        request = run_task_kwargs(
            spec=spec,
            config=config,
            secret_data=secret_data,
            vault_mounts=[],
            run_id="01ABC",
            phase="apply",
        )

        assert request["cluster"] == "runner-cluster"
        assert request["taskDefinition"] == "terrapod-runner:1"
        assert request["launchType"] == "FARGATE"
        assert request["count"] == 1
        assert request["networkConfiguration"] == {
            "awsvpcConfiguration": {
                "subnets": ["subnet-a", "subnet-b"],
                "securityGroups": ["sg-1"],
                "assignPublicIp": "ENABLED",
            }
        }
        override = request["overrides"]["containerOverrides"][0]
        assert override["name"] == "runner"
        assert override["image"] == "runner:latest"
        assert override["entryPoint"] == ["python"]
        assert override["command"] == ["-m", "terrapod.runner.ecs_entrypoint"]
        assert override["cpu"] == 512
        assert override["memory"] == 1024
        assert {"name": "TP_TOKEN", "value": "secret-token"} in override["environment"]
        ecs_files = next(e for e in override["environment"] if e["name"] == "TP_ECS_FILES")
        payload = json.loads(ecs_files["value"])
        decoded = {
            path: base64.b64decode(encoded).decode() for path, encoded in payload.items()
        }
        assert decoded == {
            "/var/run/terrapod/vars/terraform.tfvars.json": (
                '[{"key":"region","value":"eu-west-1"}]'
            )
        }
        ca = next(e for e in override["environment"] if e["name"] == "TP_ECS_CA_EXTRA")
        assert base64.b64decode(ca["value"]).decode() == "CUSTOM CA"
        assert request["overrides"]["ephemeralStorage"] == {"sizeInGiB": 64}
        assert request["overrides"]["taskRoleArn"] == config.task_role_arn
        assert request["overrides"]["executionRoleArn"] == config.execution_role_arn
        assert request["tags"] == [
            {"key": "team", "value": "platform"},
            {"key": "terrapod-managed", "value": "custom"},
            {"key": "terrapod-run-id", "value": "01ABC"},
            {"key": "terrapod-phase", "value": "apply"},
        ]

    def test_defaults_keep_the_task_minimal(self) -> None:
        request = run_task_kwargs(
            spec=_spec(cpu="", memory="", image=""),
            config=_config(),
            secret_data={},
            vault_mounts=[],
            run_id="01ABC",
            phase="plan",
        )

        override = request["overrides"]["containerOverrides"][0]
        assert override["cpu"] == 512
        assert override["memory"] == 1024
        assert "image" not in override
        assert "ephemeralStorage" not in request["overrides"]
        assert request["networkConfiguration"]["awsvpcConfiguration"]["assignPublicIp"] == (
            "DISABLED"
        )
        assert {"key": "terrapod-managed", "value": "true"} in request["tags"]

    @pytest.mark.parametrize(
        ("cpu", "expected"),
        [("250m", 256), ("1", 1024), ("0.5", 512), (None, 512)],
    )
    def test_cpu_quantities_become_ecs_units(self, cpu: str | None, expected: int) -> None:
        request = run_task_kwargs(
            spec=_spec(cpu=cpu or "", memory="1Gi"),
            config=_config(),
            secret_data={},
            vault_mounts=[],
            run_id="01ABC",
            phase="plan",
        )
        assert request["overrides"]["containerOverrides"][0]["cpu"] == expected

    @pytest.mark.parametrize(
        ("memory", "expected"),
        [("512Mi", 512), ("2Gi", 2048), ("1073741824", 1024), (None, 1024)],
    )
    def test_memory_quantities_become_mib(self, memory: str | None, expected: int) -> None:
        request = run_task_kwargs(
            spec=_spec(cpu="500m", memory=memory or ""),
            config=_config(),
            secret_data={},
            vault_mounts=[],
            run_id="01ABC",
            phase="plan",
        )
        assert request["overrides"]["containerOverrides"][0]["memory"] == expected

    @pytest.mark.parametrize(
        ("overrides", "message"),
        [
            ({"cluster": ""}, "cluster"),
            ({"task_definition": ""}, "task_definition"),
            ({"subnets": []}, "at least one subnet"),
        ],
    )
    def test_incomplete_config_is_refused(self, overrides: dict[str, Any], message: str) -> None:
        with pytest.raises(ValueError, match=message):
            run_task_kwargs(
                spec=_spec(),
                config=_config(**overrides),
                secret_data={},
                vault_mounts=[],
                run_id="01ABC",
                phase="plan",
            )


def test_task_identifier_handles_both_arn_shapes() -> None:
    assert (
        task_identifier(
            "arn:aws:ecs:eu-west-1:000000000000:task/runner-cluster/0123456789abcdef"
        )
        == "0123456789abcdef"
    )
    assert task_identifier("0123456789abcdef") == "0123456789abcdef"
