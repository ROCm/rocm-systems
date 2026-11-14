# Copyright Advanced Micro Devices, Inc. SPDX-License-Identifier: MIT

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

import pytest

SCRIPTS = Path(__file__).resolve().parents[1]
DATA = Path(__file__).resolve().parent / "data"
EXPECTED = DATA / "expected"
ROCM_TRACE_INCLUDE = SCRIPTS.parent / "include"
REPO = SCRIPTS.parents[2]

sys.path.insert(0, str(SCRIPTS))

from rocm_trace_gen import cli  # noqa: E402
from rocm_trace_gen.config import load  # noqa: E402
from rocm_trace_gen.emit import generate  # noqa: E402
from rocm_trace_gen.parse import parse_tables  # noqa: E402


def generate_files(config_path: Path, clang_args: Optional[List[str]] = None) -> dict:
    """Generate in memory: {file name: contents}."""
    config = load(config_path)
    tables = parse_tables(
        config.includes,
        [(t.name, t.size_member) for t in config.tables],
        clang_args if clang_args is not None else [f"-I{DATA}"],
    )
    return generate(config, tables)


def run_cli(*args: str) -> int:
    return cli.main(list(args))


def resource_dir() -> Optional[str]:
    """clang's resource directory, for parsing headers that need stddef.h."""
    env = os.environ.get("ROCM_TRACE_GEN_RESOURCE_DIR")
    if env:
        return env
    for clang in (shutil.which("clang"), "/opt/rocm/llvm/bin/clang"):
        if clang and Path(clang).exists():
            out = subprocess.run(
                [clang, "-print-resource-dir"],
                capture_output=True,
                text=True,
                check=False,
            )
            path = out.stdout.strip()
            if out.returncode == 0 and (Path(path) / "include" / "stddef.h").exists():
                return path
    return None


def tracelogging_include() -> Optional[Path]:
    """Directory holding tracelogging/TraceLoggingProvider.h (LTTng flavour)."""
    candidates = [
        os.environ.get("ROCM_TRACE_GEN_TRACELOGGING_INCLUDE"),
        "/usr/include",
        "/usr/local/include",
    ]
    for c in candidates:
        if c and (Path(c) / "tracelogging" / "TraceLoggingProvider.h").exists():
            return Path(c)
    return None


@pytest.fixture
def fake_config() -> Path:
    return DATA / "fake.yaml"
