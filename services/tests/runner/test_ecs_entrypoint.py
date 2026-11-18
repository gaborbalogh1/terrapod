"""The ECS wrapper materialises the same files Kubernetes would mount (#ECS).

The runner process is unchanged. ECS has no per-run Secret volume here, so the
listener ships base64 payloads in environment variables and this wrapper writes
them to the exact paths the ordinary runner reads. A path escape would let a
payload overwrite the image or this entrypoint, so those checks are part of the
contract rather than incidental validation.
"""

from __future__ import annotations

import base64
import json
import os
import stat
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from terrapod.runner import ecs_entrypoint
from terrapod.runner import job_entrypoint


@pytest.fixture
def roots(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    vars_root = tmp_path / "vars"
    files_root = tmp_path / "files"
    monkeypatch.setattr(ecs_entrypoint, "ALLOWED_ROOTS", (vars_root, files_root))
    return vars_root, files_root


class TestPayloadPaths:
    def test_absolute_paths_under_the_two_mount_roots_are_allowed(self, roots: tuple[Path, Path]) -> None:
        vars_root, files_root = roots

        assert ecs_entrypoint._safe_path(str(vars_root / "terraform.tfvars.json")) == (
            vars_root / "terraform.tfvars.json"
        )
        assert ecs_entrypoint._safe_path(str(files_root / "nested" / "app.conf")) == (
            files_root / "nested" / "app.conf"
        )

    @pytest.mark.parametrize("raw", ["relative/file", "/etc/passwd", "/var/run/terrapod/other/x"])
    def test_paths_outside_the_mount_roots_are_refused(self, raw: str) -> None:
        with pytest.raises(ValueError, match="path"):
            ecs_entrypoint._safe_path(raw)

    def test_a_traversal_path_cannot_escape_the_mount_root(self, roots: tuple[Path, Path]) -> None:
        vars_root, _ = roots

        with pytest.raises(ValueError, match="outside"):
            ecs_entrypoint._safe_path(str(vars_root / ".." / "escape"))


class TestFileMaterialization:
    def test_payloads_are_written_with_private_permissions(
        self, roots: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        vars_root, files_root = roots
        payload = {
            str(vars_root / "terraform.tfvars.json"): base64.b64encode(b'[]').decode(),
            str(files_root / "secret"): base64.b64encode(b"vault-value").decode(),
        }
        monkeypatch.setenv("TP_ECS_FILES", json.dumps(payload))

        written = ecs_entrypoint.materialize_files()

        assert written == [vars_root / "terraform.tfvars.json", files_root / "secret"]
        assert (vars_root / "terraform.tfvars.json").read_bytes() == b"[]"
        assert (files_root / "secret").read_bytes() == b"vault-value"
        for path in written:
            assert stat.S_IMODE(path.stat().st_mode) == 0o600

    def test_no_payload_is_a_noop(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("TP_ECS_FILES", raising=False)

        assert ecs_entrypoint.materialize_files() == []

    @pytest.mark.parametrize("raw", ["not json", '["not", "an", "object"]'])
    def test_malformed_payloads_fail_closed(
        self, raw: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("TP_ECS_FILES", raw)

        with pytest.raises(ValueError, match="TP_ECS_FILES"):
            ecs_entrypoint.materialize_files()

    def test_invalid_base64_is_refused(
        self, roots: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        vars_root, _ = roots
        monkeypatch.setenv(
            "TP_ECS_FILES", json.dumps({str(vars_root / "tfvars"): "%%% not base64"})
        )

        with pytest.raises(ValueError, match="not valid base64"):
            ecs_entrypoint.materialize_files()


class TestCaMerge:
    def test_a_custom_ca_is_appended_to_certifi_and_exported(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        roots = tmp_path / "roots.crt"
        roots.write_text("SYSTEM ROOT\n", encoding="utf-8")
        ca_dir = tmp_path / "ca"
        ca_file = ca_dir / "ca-bundle.crt"
        monkeypatch.setitem(sys.modules, "certifi", SimpleNamespace(where=lambda: str(roots)))
        monkeypatch.setattr(ecs_entrypoint, "CA_DIR", ca_dir)
        monkeypatch.setattr(ecs_entrypoint, "CA_FILE", ca_file)
        monkeypatch.setenv("TP_ECS_CA_EXTRA", base64.b64encode(b"CUSTOM CA\n").decode())

        assert ecs_entrypoint.merge_ca_bundle() == ca_file
        assert ca_file.read_text(encoding="utf-8") == "SYSTEM ROOT\nCUSTOM CA\n"
        for name in ecs_entrypoint.CA_ENV_NAMES:
            assert os.environ.get(name) == str(ca_file)

    def test_no_custom_ca_leaves_the_image_roots_alone(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("TP_ECS_CA_EXTRA", raising=False)

        assert ecs_entrypoint.merge_ca_bundle() is None

    def test_invalid_ca_payload_is_refused(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TP_ECS_CA_EXTRA", "not base64")

        with pytest.raises(ValueError, match="TP_ECS_CA_EXTRA"):
            ecs_entrypoint.merge_ca_bundle()


def test_main_materialises_payloads_then_hands_over_to_the_unchanged_runner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    materialize = MagicMock(return_value=[])
    merge = MagicMock(return_value=None)
    runner_main = MagicMock(return_value=7)
    monkeypatch.setattr(ecs_entrypoint, "materialize_files", materialize)
    monkeypatch.setattr(ecs_entrypoint, "merge_ca_bundle", merge)
    monkeypatch.setattr(job_entrypoint, "main", runner_main)

    assert ecs_entrypoint.main(["--phase", "plan"]) == 7

    materialize.assert_called_once_with()
    merge.assert_called_once_with()
    runner_main.assert_called_once_with(["--phase", "plan"])
