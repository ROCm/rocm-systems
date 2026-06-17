"""Red-team: compiler-flag smuggling rejected by knowledge/compiler_flags.yaml allowlist."""

from pathlib import Path
from unittest import mock

import pytest

from perfxpert.tools import compile_runner


SMUGGLED_FLAGS = [
    "-Xlinker",
    "-Wl,-rpath,/evil",
    "-Wa,--compress-debug-sections",
    "-Xpreprocessor",
    "-Wl,--wrap,write",
    "--sysroot=/evil",
    "-fplugin=/evil.so",
]


@pytest.mark.parametrize("bad", SMUGGLED_FLAGS)
def test_unknown_compiler_flag_rejected(bad, tmp_path: Path):
    (tmp_path / "src.cpp").write_text("int main(){return 0;}\n")
    with pytest.raises(compile_runner.CompileFlagError):
        compile_runner.build(
            project_root=tmp_path,
            source_rel="src.cpp",
            flags=["-O2", bad],
        )


def test_allowlisted_flags_still_work(tmp_path: Path, monkeypatch):
    """Regression: smuggling rejection must not break legitimate flags."""
    (tmp_path / "src.cpp").write_text("int main(){return 0;}\n")
    monkeypatch.setattr(
        "perfxpert.tools.compile_runner.subprocess.run",
        mock.MagicMock(return_value=mock.MagicMock(
            returncode=0, stdout=b"", stderr=b""
        )),
    )
    monkeypatch.setattr("perfxpert.tools.compile_runner.require_tool", mock.MagicMock())
    result = compile_runner.build(
        project_root=tmp_path,
        source_rel="src.cpp",
        flags=["-O3", "--offload-arch=gfx942", "-ffast-math"],
    )
    assert result["returncode"] == 0
