"""Guard the lazy ECS imports that the ordinary image walkers skip (#ECS).

Both images deliberately copy selective files. The listener's import walker
ignores function-local imports, which is where the ECS backend is loaded from,
and the runner's walker starts at ``job_entrypoint.py``, which never mentions
the ECS wrapper — ``ecs_task.py`` names it in a generated ``python -m`` command.
Either omission would pass every existing test and then fail only in the image,
so the lazy paths get their own guards here.
"""

from __future__ import annotations

import ast
import pathlib
import re


def _root() -> pathlib.Path:
    here = pathlib.Path(__file__).resolve()
    for candidate in here.parents:
        if (candidate / "docker" / "Dockerfile.listener").is_file() and (
            candidate / "docker" / "Dockerfile.runner"
        ).is_file():
            return candidate
    raise AssertionError("could not find the docker/ fixtures in either checkout or test image")


ROOT = _root()
SOURCE = next(
    (
        p
        for p in (ROOT / "services" / "terrapod", ROOT / "terrapod")
        if (p / "runner" / "listener.py").is_file()
    ),
    None,
)


def _requirement(name: str) -> pathlib.Path:
    for candidate in (
        ROOT / "services" / f"requirements-{name}.txt",
        ROOT / f"requirements-{name}.txt",
    ):
        if candidate.is_file():
            return candidate
    raise AssertionError(f"requirements-{name}.txt is missing from the image fixtures")


def _manifest(name: str) -> pathlib.Path:
    for candidate in (
        ROOT / "services" / f"pyproject-{name}.toml",
        ROOT / f"pyproject-{name}.toml",
    ):
        if candidate.is_file():
            return candidate
    raise AssertionError(f"pyproject-{name}.toml is missing from the image fixtures")


def _reachable_runner_modules(start: pathlib.Path) -> set[str]:
    """Follow every `terrapod.runner.*` import, including lazy ones."""
    assert SOURCE is not None, "the terrapod source fixture is missing"
    runner = SOURCE / "runner"
    seen = {start.stem}
    queue = [start]
    while queue:
        path = queue.pop()
        if not path.is_file():
            continue
        tree = ast.parse(path.read_text(), filename=str(path))
        imports: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                if node.module.startswith("terrapod.runner"):
                    tail = node.module.removeprefix("terrapod.runner").lstrip(".")
                    imports.append(tail)
                    imports += [f"{tail}.{alias.name}".strip(".") for alias in node.names]
            elif isinstance(node, ast.Import):
                imports += [
                    alias.name.removeprefix("terrapod.runner.")
                    for alias in node.names
                    if alias.name.startswith("terrapod.runner.")
                ]
        for module in imports:
            rel = module.replace(".", "/")
            if rel in seen:
                continue
            candidates = (runner / f"{rel}.py", runner / rel / "__init__.py")
            target = next((p for p in candidates if p.is_file()), None)
            if target is not None:
                seen.add(rel)
                queue.append(target)
    return seen


class TestListenerImageShipsItsLazyEcsBackend:
    def test_the_lazy_import_and_its_module_chain_are_in_the_image(self) -> None:
        assert SOURCE is not None, "the terrapod source fixture is missing"
        listener = (SOURCE / "runner" / "listener.py").read_text()
        assert "from terrapod.runner.ecs_backend import EcsExecutionBackend" in listener

        reachable = _reachable_runner_modules(SOURCE / "runner" / "ecs_backend.py")
        assert {"ecs_backend", "ecs_task"} <= reachable

        dockerfile = (ROOT / "docker" / "Dockerfile.listener").read_text()
        assert "COPY services/terrapod/runner/ ./terrapod/runner/" in dockerfile
        # The ECS modules import these directly; the listener image copies them
        # explicitly rather than through the runner/ directory.
        for module in ("config.py", "logging_config.py"):
            assert f"COPY services/terrapod/{module} ./terrapod/{module}" in dockerfile

    def test_the_hash_pinned_listener_lock_includes_aiobotocore_but_not_boto3(self) -> None:
        manifest = _manifest("listener").read_text()
        lock = _requirement("listener").read_text()

        assert re.search(r'^\s*"aiobotocore>=3\.10,<4\.0"', manifest, re.MULTILINE)
        assert re.search(r"^aiobotocore==[\d.]+ ", lock, re.MULTILINE)
        assert re.search(r"^botocore==[\d.]+ ", lock, re.MULTILINE)
        assert not re.search(r"^boto3==", lock, re.MULTILINE)


class TestRunnerImageShipsTheEcsWrapper:
    def test_the_generated_python_m_module_is_copied_and_rebuilt(self) -> None:
        assert SOURCE is not None, "the terrapod source fixture is missing"
        ecs_task = (SOURCE / "runner" / "ecs_task.py").read_text()
        assert '"-m", "terrapod.runner.ecs_entrypoint"' in ecs_task

        dockerfile = (ROOT / "docker" / "Dockerfile.runner").read_text()
        tiltfile = (ROOT / "Tiltfile").read_text()
        for module in sorted(_reachable_runner_modules(SOURCE / "runner" / "ecs_entrypoint.py")):
            # Directory packages such as `phases/` have their own COPY guard in
            # test_runner_image_completeness.py; this guard is for named modules.
            if "/" in module or not (SOURCE / "runner" / f"{module}.py").is_file():
                continue
            assert f"COPY services/terrapod/runner/{module}.py" in dockerfile, (
                f"lazy ECS wrapper module {module} is missing from the runner image"
            )
            assert f"services/terrapod/runner/{module}.py" in tiltfile, (
                f"lazy ECS wrapper module {module} is missing from the Tiltfile build deps"
            )

    def test_certifi_is_pinned_for_the_custom_ca_merge(self) -> None:
        assert re.search(r"^certifi==[\d.]+ ", _requirement("runner").read_text(), re.MULTILINE)
