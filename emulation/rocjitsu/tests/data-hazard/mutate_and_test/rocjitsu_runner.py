# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Starting kernels under rocJitsu with the ``data_hazard`` plugin enabled.

This is the rocJitsu half of the harness: the launcher binary and the simulator
config that has to match the target the kernels are compiled for. Other drivers
plug in their own :class:`~mutate_and_test.pipeline.KernelRunner`.
"""

import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def _rocjitsu_root() -> Path:
    """Return the rocjitsu source root (``tests/data-hazard/mutate_and_test`` → root)."""
    return Path(__file__).resolve().parent.parent.parent.parent


class RocjitsuRunner:
    """
    Runs kernels under rocJitsu with the ``data_hazard`` plugin enabled.

    Plugin selection and the report destination are part of the simulator
    configuration rather than the environment, so a fresh config file is written
    for every run and handed to the launcher as
    ``rocjitsu --config <file> -- <exe>``.
    """

    def __init__(
        self,
        launcher: Path,
        base_config: Path,
        workdir: Path,
        *,
        verbose: bool = False,
    ) -> None:
        self.launcher = launcher.resolve()
        self.base_config = base_config.resolve()
        self.config_dir = workdir / "rocjitsu_configs"
        self.verbose = verbose

        if not self.launcher.is_file() or not os.access(self.launcher, os.X_OK):
            raise FileNotFoundError(f"rocjitsu launcher not found at {self.launcher}")
        if not self.base_config.is_file():
            raise FileNotFoundError(f"rocjitsu config not found at {self.base_config}")
        self.config_dir.mkdir(parents=True, exist_ok=True)

    def command(self, exe: Path, report_path: Path) -> Tuple[List[str], Dict[str, str]]:
        """Build the launcher argv that runs *exe* reporting into *report_path*."""
        config_path = self.config_dir / f"{report_path.stem}.json"
        config = json.loads(self.base_config.read_text())
        plugins = config.setdefault("plugins", {})
        plugins.setdefault("data_hazard", {})["report_path"] = str(report_path)
        config_path.write_text(json.dumps(config, indent=2) + "\n")

        argv = [str(self.launcher), "--config", str(config_path), "--", str(exe)]
        if self.verbose:
            print(f"  rocjitsu: {' '.join(argv)}")
        return argv, {}


def gfx_target_version(arch: str) -> Optional[int]:
    """Encode a gfx target name the way the configs do: ``gfx1250`` -> 120500."""
    m = re.fullmatch(r"gfx(\d+)([0-9a-f])([0-9a-f])", arch)
    if not m:
        return None
    return int(m.group(1)) * 10000 + int(m.group(2), 16) * 100 + int(m.group(3), 16)


def default_config_for_arch(arch: str) -> Optional[Path]:
    """Pick the plain (non-kmd, non-guest) config that simulates *arch*."""
    config_dir = _rocjitsu_root() / "configs"
    wanted = gfx_target_version(arch)
    if wanted is None:
        return None
    candidates = [
        p
        for p in sorted(config_dir.glob("*.json"))
        if "kmd" not in p.stem and not p.stem.startswith("guest_")
    ]
    for path in candidates:
        try:
            device = json.loads(path.read_text())["vm"]["gpu"]["device"]
        except (OSError, ValueError, KeyError):
            continue
        if device.get("gfx_target_version") == wanted:
            return path
    return None


def build_rocjitsu_runner(
    workdir: Path,
    *,
    arch: str,
    launcher: Optional[Path] = None,
    config: Optional[Path] = None,
    build_dir: Optional[Path] = None,
    verbose: bool = False,
) -> RocjitsuRunner:
    """
    Construct a :class:`RocjitsuRunner` from explicit paths or the environment.

    Resolution order for each path is the explicit argument, then the matching
    environment variable (``ROCJITSU_LAUNCHER`` / ``ROCJITSU_CONFIG`` /
    ``ROCJITSU_BUILD_DIR``), then the default location inside the build tree.
    The config must simulate the same target the kernels are compiled for: a
    kernel that the simulated device cannot run reports no hazards at all,
    which otherwise looks like a clean run rather than a misconfiguration.
    """
    build = (
        build_dir
        or Path(os.environ.get("ROCJITSU_BUILD_DIR", _rocjitsu_root() / "build"))
    ).resolve()

    resolved_launcher = launcher or Path(
        os.environ.get(
            "ROCJITSU_LAUNCHER", str(build / "tools" / "rocjitsu" / "rocjitsu")
        )
    )
    env_config = os.environ.get("ROCJITSU_CONFIG")
    resolved_config = config or (
        Path(env_config) if env_config else default_config_for_arch(arch)
    )
    if resolved_config is None:
        raise FileNotFoundError(
            f"no config in {_rocjitsu_root() / 'configs'} simulates {arch}; "
            "pass --rocjitsu-config"
        )

    wanted = gfx_target_version(arch)
    try:
        device = json.loads(Path(resolved_config).read_text())["vm"]["gpu"]["device"]
        found = device.get("gfx_target_version")
    except (OSError, ValueError, KeyError):
        found = None
    if wanted is not None and found is not None and found != wanted:
        raise FileNotFoundError(
            f"config {resolved_config} simulates gfx_target_version {found}, but kernels "
            f"are compiled for {arch} ({wanted}); the mismatch reports zero hazards"
        )

    return RocjitsuRunner(resolved_launcher, resolved_config, workdir, verbose=verbose)
