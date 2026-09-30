# MIT License
#
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

"""Session setup for this skill's behavioral evaluations (amd/skillscope hooks).

The evaluations trace kernels with rocprofv3 --att and read the traces with
scripts/att_mine.py, so the agent needs what the skill's Prerequisites list: the
rocprof-trace-decoder library and its Python package. Unless the environment already
provides both, setup_session builds the decoder once per session from the source tree the
skill came from, the way resources/capture.md does, and exports the variables the skill
names. The agent inherits this process's environment.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

from skillscope import sources

SKILL = Path(__file__).resolve().parents[1].name
DECODER_SOURCE = Path(
    "projects/rocprof-trace-decoder"
)  # within the rocm-systems checkout
# Loads the Python package and the library the way att_mine.py does.
PROBE = "import elftools, rocprof_trace_decoder as d; d.Decoder().close()"


def _python() -> str:
    """The interpreter the agent runs as python3."""
    return shutil.which("python3") or "python3"


def _run(cmd: list[str], what: str) -> None:
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True)
    except FileNotFoundError:
        raise SystemExit(
            f"{SKILL} hooks: {what} needs {cmd[0]}, which is not installed"
        ) from None
    if proc.returncode:
        raise SystemExit(
            f"{SKILL} hooks: {what} failed:\n{(proc.stdout + proc.stderr)[-2000:]}"
        )


def _provided() -> bool:
    """Whether rocprofv3 and att_mine.py can already load a decoder."""
    rocm_lib = Path(os.environ.get("ROCM_PATH", "/opt/rocm")) / "lib"
    found = os.environ.get("ROCPROF_ATT_LIBRARY_PATH") or any(
        rocm_lib.glob("librocprof-trace-decoder.so*")
    )
    return (
        bool(found)
        and subprocess.run([_python(), "-c", PROBE], capture_output=True).returncode == 0
    )


def setup_session(cache_dir):
    if _provided():
        print(f"[{SKILL} hooks] using the decoder this environment provides", flush=True)
        return {}
    cache_dir = Path(cache_dir)
    source = sources.resolve(SKILL, cache_dir)
    src = source.path / DECODER_SOURCE
    if not (src / "CMakeLists.txt").is_file():
        raise SystemExit(
            f"{SKILL} hooks: {src} is not the rocprof-trace-decoder source; set "
            "SKILL_SOURCE_DIR to a rocm-systems checkout"
        )
    build = cache_dir / "decoder-build"
    lib = build / "lib" / "librocprof-trace-decoder.so"
    if not lib.is_file():
        _run(["cmake", "-S", str(src), "-B", str(build)], "configuring the decoder")
        _run(["cmake", "--build", str(build), "-j"], "building the decoder")
    paths = [str(src / "python")]
    if subprocess.run(
        [_python(), "-c", "import elftools"], capture_output=True
    ).returncode:
        deps = cache_dir / "python-deps"
        _run(
            [
                _python(),
                "-m",
                "pip",
                "install",
                "--quiet",
                "--target",
                str(deps),
                "pyelftools>=0.31",
            ],
            "installing pyelftools",
        )
        paths.append(str(deps))
    if os.environ.get("PYTHONPATH"):
        paths.append(os.environ["PYTHONPATH"])
    os.environ["ROCPROF_ATT_LIBRARY_PATH"] = str(lib.parent)
    os.environ["ROCPROF_TRACE_DECODER_LIB"] = str(lib)
    os.environ["PYTHONPATH"] = os.pathsep.join(paths)
    _run([_python(), "-c", PROBE], "loading the decoder it built")
    print(f"[{SKILL} hooks] decoder built from {source.origin}: {lib}", flush=True)
    return {}
