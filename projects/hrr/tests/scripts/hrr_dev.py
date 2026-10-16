#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Build and run the HRR suites in a local Docker container.

The host needs Docker. CMake, Ninja, a ROCm toolchain and rocJitsu are inside
the hrr-dev image, which this script builds the first time it is missing.
`emulator` runs the integration drivers under the rocJitsu configs in
ci.yaml. `gpu` runs them on the machine's GPUs. The host unit binary has no
device code, so both commands run it once inside the container and do not
launch it under rocJitsu or on a GPU.

    python3 projects/hrr/tests/scripts/hrr_dev.py emulator
    python3 projects/hrr/tests/scripts/hrr_dev.py gpu
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import hrr_ci_configure

IMAGE = "hrr-dev"
CONTAINER = "hrr-dev"
REPO = Path(__file__).resolve().parents[4]

# Family keys whose rocJitsu target is not the key itself.
_ARCH = {
    "gfx94x": "gfx942",
    "gfx103x": "gfx1030",
    "gfx110x": "gfx1100",
    "gfx120x": "gfx1201",
    "gfx125x": "gfx1250",
}


@dataclass(frozen=True)
class DockerState:
    image: bool
    container: str | None  # None, "running", or "exited"


def arch_for(family: str) -> str:
    return _ARCH.get(family, family)


def emulator_jobs(config_path: Path | None = None) -> list[dict]:
    config = hrr_ci_configure.load_config(
        config_path or hrr_ci_configure.DEFAULT_CONFIG
    )
    return [
        {
            "family": family,
            "arch": arch_for(family),
            "config": config.simulator.configs[family],
        }
        for family in config.simulator.default_families
    ]


def docker_build_command(repo: Path) -> list[str]:
    return [
        "docker",
        "build",
        "-t",
        IMAGE,
        "-f",
        str(repo / "projects/hrr/tests/docker/Dockerfile"),
        str(repo),
    ]


def docker_run_command(repo: Path, backend: str) -> list[str]:
    command = [
        "docker",
        "run",
        "-d",
        "--name",
        CONTAINER,
        "-v",
        f"{repo}:{repo}",
        "-w",
        str(repo),
        "--user",
        f"{os.getuid()}:{os.getgid()}",
    ]
    if backend == "gpu":
        command += [
            "--device",
            "/dev/kfd",
            "--device",
            "/dev/dri",
            "--group-add",
            "video",
        ]
    command += [IMAGE, "sleep", "infinity"]
    return command


def plan_docker(repo: Path, backend: str, state: DockerState) -> list[list[str]]:
    """Commands that make the hrr-dev container ready. An existing container
    is started, not recreated."""
    commands = []
    if not state.image:
        commands.append(docker_build_command(repo))
    if state.container is None:
        commands.append(docker_run_command(repo, backend))
    elif state.container != "running":
        commands.append(["docker", "start", CONTAINER])
    return commands


def inspect_docker(runner=subprocess.run) -> DockerState:
    image = runner(
        ["docker", "image", "inspect", IMAGE],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    container = runner(
        ["docker", "inspect", "-f", "{{.State.Status}}", CONTAINER],
        capture_output=True,
        text=True,
    )
    status = container.stdout.strip() if container.returncode == 0 else None
    return DockerState(image.returncode == 0, status or None)


def container_script(repo: Path, backend: str) -> str:
    """Bash run inside the container. One rocJitsu family failing still runs
    the other."""
    dist = f"{repo}/build/hrr-dev/dist"
    work = f"{repo}/build/hrr-dev"
    headers = f"{repo}/projects/hip/include;{repo}/projects/clr/hipamd/include"
    lines = [
        "set -euo pipefail",
        f'export ROCM_PATH="{dist}"',
        "python3 projects/hrr/tests/scripts/hrr_capture_build.py fetch-base "
        f'--dist "{dist}" --build "{work}" --base-json "{work}/base.json"',
        "cmake -S projects/hrr -B build/hrr-dev/unit -G Ninja "
        "-DCMAKE_BUILD_TYPE=Release "
        f'-DROCM_PATH="{dist}" -DCMAKE_PREFIX_PATH="{dist}" '
        f'"-DHRR_HIP_INCLUDE_DIRS={headers}" '
        "-DHRR_BUILD_PLAYBACK=ON -DHRR_BUILD_TESTS=ON -DHRR_BUILD_INTEGRATION_TESTS=OFF",
        "cmake --build build/hrr-dev/unit -j",
        "python3 projects/hrr/tests/scripts/hrr_capture_build.py build "
        f'--dist "{dist}" --build "{work}"',
        "set +e",
        "status=0",
        "python3 projects/hrr/tests/scripts/run_catch2.py "
        "--xml build/hrr-dev/unit.xml --config projects/hrr/tests/config/unit.yaml "
        "--arch none -- build/hrr-dev/unit/tests/unit/hrr-unit-tests || status=1",
        "if [ -x build/hrr-dev/unit/tests/unit/hrr-handler-tests ]; then "
        "python3 projects/hrr/tests/scripts/run_catch2.py "
        "--xml build/hrr-dev/handler.xml --config projects/hrr/tests/config/handler.yaml "
        "--arch none -- build/hrr-dev/unit/tests/unit/hrr-handler-tests || status=1; fi",
    ]
    binary = f"{work}/hrr/tests/integration/hrr-integration-tests"
    if backend == "emulator":
        for job in emulator_jobs():
            launcher = f"rocjitsu --daemon --config {repo}/{job['config']} --"
            lines.append(
                "python3 projects/hrr/tests/scripts/run_catch2.py "
                f"--xml build/hrr-dev/integration-{job['family']}.xml --timeout 1200 "
                "--config projects/hrr/tests/config/integration.yaml "
                f"--os linux --arch {job['arch']} --launcher \"{launcher}\" "
                f"-- {binary} || status=1"
            )
    else:
        lines.append(
            "python3 projects/hrr/tests/scripts/run_catch2.py "
            "--xml build/hrr-dev/integration-gpu.xml --timeout 1200 "
            "--config projects/hrr/tests/config/integration.yaml "
            f"--os linux --arch auto -- {binary} || status=1"
        )
    lines.append("exit $status")
    return "\n".join(lines) + "\n"


def exec_command(repo: Path, backend: str) -> list[str]:
    command = ["docker", "exec", "-w", str(repo)]
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if token:
        command += ["-e", f"GH_TOKEN={token}"]
    command += [CONTAINER, "bash", "-lc", container_script(repo, backend)]
    return command


def prepare(repo: Path, backend: str, runner=subprocess.run) -> None:
    for command in plan_docker(repo, backend, inspect_docker(runner)):
        subprocess.run(command, check=True)
    subprocess.run(exec_command(repo, backend), check=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("backend", choices=("emulator", "gpu"))
    parser.add_argument("--repo", type=Path, default=REPO)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        prepare(args.repo.resolve(), args.backend)
    except subprocess.CalledProcessError as exc:
        return exc.returncode or 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
