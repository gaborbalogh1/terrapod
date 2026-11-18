"""Translate a runner Job spec into ECS ``RunTask`` overrides.

The listener keeps building the canonical Job spec through the engine strategy.
Under the ECS execution backend that spec is used as a *description of the run*
(image, environment, variables, resources) rather than as a Kubernetes object:
this module maps it onto container overrides for the same runner image. The
runner process itself is unchanged — it still reads ``TP_*`` env vars and talks
to the API over HTTP.

Kubernetes-only scheduling fields (service accounts, node selectors, tolerations,
projected volumes) have no ECS equivalent and are deliberately not translated.
Configuration that is delivered as a mounted Secret on Kubernetes is written to
the same filesystem paths by :mod:`terrapod.runner.ecs_entrypoint`.
"""

from __future__ import annotations

import base64
import json
import re
from typing import Any

from terrapod.config import EcsExecutionConfig

#: Keys of the per-run vars Secret that are files, not environment variables.
_VARS_FILES = frozenset(
    {"terraform.tfvars.json", "execution-hooks.json", "git-auth.json", "ansible-auth.json"}
)
#: The per-run auth Secret key holding the runner token.
_AUTH_TOKEN_KEY = "token"
_CA_EXTRA_KEY = "ca-extra.pem"
_VARS_DIR = "/var/run/terrapod/vars"
_FILES_DIR = "/var/run/terrapod/files"

#: ECS CPU units (1024 = one vCPU) and memory (MiB) when the Job spec has no
#: explicit resource request. These mirror a modest Fargate runner.
_DEFAULT_CPU_UNITS = 512
_DEFAULT_MEMORY_MIB = 1024


def _cpu_units(value: Any, default: int) -> int:
    """Convert a Kubernetes CPU quantity to ECS CPU units."""
    if value in (None, ""):
        return default
    text = str(value)
    if text.endswith("m"):
        return max(1, round(int(text[:-1]) / 1000 * 1024))
    return max(1, round(float(text) * 1024))


def _memory_mib(value: Any, default: int) -> int:
    """Convert a Kubernetes memory quantity to whole MiB."""
    if value in (None, ""):
        return default
    text = str(value)
    multipliers = {"Ki": 1 / 1024, "Mi": 1, "Gi": 1024, "Ti": 1024 * 1024}
    for suffix, factor in multipliers.items():
        if text.endswith(suffix):
            return max(1, round(float(text[: -len(suffix)]) * factor))
    if text.endswith(("K", "M", "G", "T")):
        factor = {"K": 1 / 1024 / 1000, "M": 1 / 1000, "G": 1000, "T": 1_000_000}[text[-1]]
        return max(1, round(float(text[:-1]) * factor))
    return max(1, round(int(text) / (1024 * 1024)))


def _container(spec: dict) -> dict:
    """The first (runner) container from a Kubernetes Job spec."""
    pod = spec.get("spec", {}).get("template", {}).get("spec", {})
    containers = pod.get("containers") or []
    if not containers:
        raise ValueError("Job spec has no containers to translate")
    return containers[0]


def environment(spec: dict, secret_data: dict[str, str]) -> list[dict[str, str]]:
    """Resolve the Job spec's env into ECS ``name``/``value`` pairs.

    ``secretKeyRef`` values are resolved from the values the listener already
    collected for the per-run Secrets. A missing key is a hard failure: running
    the image with an empty token would produce a run that never reports back.
    """
    resolved: list[dict[str, str]] = []
    for item in _container(spec).get("env") or []:
        name = item.get("name")
        if not name:
            continue
        if "value" in item:
            value = item["value"]
        else:
            ref = (item.get("valueFrom") or {}).get("secretKeyRef") or {}
            key = ref.get("key", "")
            if key not in secret_data:
                raise ValueError(f"missing ECS secret value for environment variable {name} ({key})")
            value = secret_data[key]
        if value is None:
            continue
        resolved.append({"name": str(name), "value": str(value)})
    return resolved


def files(spec: dict, secret_data: dict[str, str], vault_mounts: list[dict]) -> dict[str, str]:
    """Map Secret keys to the filesystem paths the runner reads.

    Vault files keep their ``vault-file-N`` Secret keys and are named by the
    ``vault_mounts`` entry that refers to them. Any other unrecognised key is
    rejected rather than silently dropped — that is how a new file channel
    would be forgotten.
    """
    # These are Secret *keys* consumed as environment variables. Compare them
    # with the Secret keys in `secret_data`, not with the environment variable
    # names — a `secretKeyRef` may deliberately use a different name for both.
    secret_ref_keys = {
        ((item.get("valueFrom") or {}).get("secretKeyRef") or {}).get("key", "")
        for item in _container(spec).get("env") or []
        if "valueFrom" in item
    }
    vault_by_key = {m["secret_key"]: m["name"] for m in vault_mounts}
    result: dict[str, str] = {}
    for key, value in secret_data.items():
        if key in secret_ref_keys or key == _AUTH_TOKEN_KEY:
            continue
        if key in _VARS_FILES:
            result[f"{_VARS_DIR}/{key}"] = value
        elif key == _CA_EXTRA_KEY:
            continue  # handled by ecs_entrypoint as a merged CA bundle
        elif key in vault_by_key:
            result[f"{_FILES_DIR}/{vault_by_key[key]}"] = value
        else:
            raise ValueError(f"unmapped runner file key in ECS run payload: {key}")
    return result


def run_task_kwargs(
    *,
    spec: dict,
    config: EcsExecutionConfig,
    secret_data: dict[str, str],
    vault_mounts: list[dict],
    run_id: str,
    phase: str,
) -> dict[str, Any]:
    """Build the ``RunTask`` request for one phase of a run."""
    if not config.cluster or not config.task_definition:
        raise ValueError("ECS execution requires cluster and task_definition")
    if not config.subnets:
        raise ValueError("ECS execution requires at least one subnet")

    container = _container(spec)
    resources = container.get("resources") or {}
    requests = resources.get("requests") or {}
    limits = resources.get("limits") or {}
    cpu = _cpu_units(requests.get("cpu", limits.get("cpu")), _DEFAULT_CPU_UNITS)
    memory = _memory_mib(requests.get("memory", limits.get("memory")), _DEFAULT_MEMORY_MIB)

    env = environment(spec, secret_data)
    file_payload = files(spec, secret_data, vault_mounts)
    env.append(
        {
            "name": "TP_ECS_FILES",
            "value": json.dumps(
                {
                    path: base64.b64encode(content.encode()).decode()
                    for path, content in file_payload.items()
                },
                separators=(",", ":"),
            ),
        }
    )
    if _CA_EXTRA_KEY in secret_data:
        env.append(
            {
                "name": "TP_ECS_CA_EXTRA",
                "value": base64.b64encode(secret_data[_CA_EXTRA_KEY].encode()).decode(),
            }
        )

    image = container.get("image", "")
    override: dict[str, Any] = {
        "name": config.container_name,
        "environment": env,
        # Override both: the task definition ships the normal runner entrypoint,
        # and only the ECS wrapper should materialise files before handing over.
        "entryPoint": ["python"],
        "command": ["-m", "terrapod.runner.ecs_entrypoint"],
        "cpu": cpu,
        "memory": memory,
    }
    if image:
        override["image"] = image

    task_override: dict[str, Any] = {"containerOverrides": [override]}
    if config.ephemeral_storage_gib:
        task_override["ephemeralStorage"] = {"sizeInGiB": config.ephemeral_storage_gib}
    if config.task_role_arn:
        task_override["taskRoleArn"] = config.task_role_arn
    if config.execution_role_arn:
        task_override["executionRoleArn"] = config.execution_role_arn

    kwargs: dict[str, Any] = {
        "cluster": config.cluster,
        "taskDefinition": config.task_definition,
        "launchType": config.launch_type,
        "count": 1,
        "networkConfiguration": {
            "awsvpcConfiguration": {
                "subnets": config.subnets,
                "securityGroups": config.security_groups,
                "assignPublicIp": "ENABLED" if config.assign_public_ip else "DISABLED",
            }
        },
        "overrides": task_override,
    }
    tags = dict(config.tags)
    tags.setdefault("terrapod-managed", "true")
    tags.setdefault("terrapod-run-id", run_id)
    tags["terrapod-phase"] = phase
    kwargs["tags"] = [{"key": k, "value": str(v)} for k, v in tags.items() if k and v is not None]
    return kwargs


def task_identifier(task_arn: str) -> str:
    """Short task id used as the CloudWatch log stream suffix."""
    return re.split(r"[/:]", task_arn)[-1]
