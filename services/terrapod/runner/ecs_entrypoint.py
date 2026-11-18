"""Wrapper entrypoint for runner images launched as ECS Tasks.

Kubernetes delivers per-run variables and credentials as mounted Secrets, and
merges a custom CA with the image's system roots in an init container. ECS has
no equivalent Secrets volume mount on this task definition, so the listener
ships the same payload as base64 in ``TP_ECS_FILES`` and an optional custom CA
in ``TP_ECS_CA_EXTRA``. This module materialises those payloads at the exact
paths the unchanged runner entrypoint already reads, then hands over to
:func:`terrapod.runner.job_entrypoint.main`.

Nothing about the runner's own logic changes: it still reads ``TP_*`` env vars,
talks to the API over HTTP and uploads its logs and artifacts itself.
"""

from __future__ import annotations

import base64
import binascii
import json
import os
import sys
from pathlib import Path

#: Only these roots are writable. A payload that escaped them could overwrite
#: the runner image or the entrypoint itself.
ALLOWED_ROOTS = (
    Path("/var/run/terrapod/vars"),
    Path("/var/run/terrapod/files"),
)
CA_DIR = Path("/etc/terrapod-ca")
CA_FILE = CA_DIR / "ca-bundle.crt"
CA_ENV_NAMES = ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE", "GIT_SSL_CAINFO")


def _safe_path(raw: str) -> Path:
    """Validate a payload path against the two per-run mount roots."""
    path = Path(raw)
    if not path.is_absolute():
        raise ValueError(f"ECS file payload path must be absolute: {raw!r}")
    resolved = path.resolve(strict=False)
    if not any(resolved.is_relative_to(root) for root in ALLOWED_ROOTS):
        raise ValueError(f"ECS file payload path is outside the per-run mount roots: {raw!r}")
    return resolved


def _write(path: Path, data: bytes) -> None:
    """Write one payload file with 0600 permissions, refusing symlinks."""
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(path, flags, 0o600)
    with os.fdopen(fd, "wb") as handle:
        handle.write(data)


def materialize_files() -> list[Path]:
    """Decode ``TP_ECS_FILES`` onto the filesystem and return the paths written."""
    raw = os.environ.get("TP_ECS_FILES", "")
    if not raw:
        return []
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError("TP_ECS_FILES is not valid JSON") from e
    if not isinstance(payload, dict):
        raise ValueError("TP_ECS_FILES must be a JSON object")

    written: list[Path] = []
    for raw_path, encoded in payload.items():
        path = _safe_path(str(raw_path))
        try:
            data = base64.b64decode(str(encoded), validate=True)
        except (binascii.Error, ValueError) as e:
            raise ValueError(f"ECS file payload for {raw_path!r} is not valid base64") from e
        _write(path, data)
        written.append(path)
    return written


def merge_ca_bundle() -> Path | None:
    """Append a custom CA to certifi's roots and export the standard CA env vars.

    This is the ECS equivalent of the Kubernetes init container. The runner's
    Job spec already points SSL_CERT_FILE (and friends) at ``CA_FILE``; this
    function creates that file so those variables are useful.
    """
    raw = os.environ.get("TP_ECS_CA_EXTRA", "")
    if not raw:
        return None
    try:
        extra = base64.b64decode(raw, validate=True).decode("utf-8").strip()
    except (binascii.Error, UnicodeDecodeError, ValueError) as e:
        raise ValueError("TP_ECS_CA_EXTRA is not valid base64-encoded UTF-8") from e

    import certifi

    roots = Path(certifi.where()).read_text(encoding="utf-8").strip()
    CA_DIR.mkdir(parents=True, exist_ok=True, mode=0o700)
    CA_FILE.write_text(f"{roots}\n{extra}\n", encoding="utf-8")
    for name in CA_ENV_NAMES:
        os.environ[name] = str(CA_FILE)
    return CA_FILE


def main(argv: list[str] | None = None) -> int:
    """Materialise the ECS payload, then run the ordinary runner entrypoint."""
    materialize_files()
    merge_ca_bundle()
    from terrapod.runner.job_entrypoint import main as runner_main

    return runner_main(argv if argv is not None else sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
