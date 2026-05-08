"""Red-team: rocprofv3 flag smuggling rejected by allowlist."""

from pathlib import Path

import pytest

from perfxpert.tools import profile_runner


SMUGGLED_FLAGS = [
    "--hip-api-trace",           # deprecated / not real
    "--kernel-names",            # value-taking flag rocprofv3 does not accept
    "--totally-fake",
    "--exec",                    # plausible-looking but not in rocprofv3
    "--dump-to",                 # plausible-looking
    "--allow-shell",             # obviously fake but emphasizes allowlist discipline
]


@pytest.mark.parametrize("bad", SMUGGLED_FLAGS)
def test_unknown_rocprofv3_flag_rejected(bad, tmp_path: Path):
    with pytest.raises(profile_runner.RocprofFlagError):
        profile_runner.run(
            argv=["rocprofv3", "--sys-trace", bad, "--", "./app"],
            cwd=tmp_path,
        )


def test_known_flags_still_work(tmp_path: Path, monkeypatch):
    from unittest import mock
    monkeypatch.setattr(
        "perfxpert.tools.profile_runner.subprocess.run",
        mock.MagicMock(return_value=mock.MagicMock(returncode=0, stdout=b"", stderr=b"")),
    )
    monkeypatch.setattr("perfxpert.tools.profile_runner.require_tool", mock.MagicMock())
    result = profile_runner.run(
        argv=[
            "rocprofv3", "--sys-trace", "--kernel-trace", "--stats",
            "-d", "out", "-o", "results", "--", "./app",
        ],
        cwd=tmp_path,
    )
    assert result["returncode"] == 0
