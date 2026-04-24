"""Red-team: path traversal via ../ + symlink escape."""

from pathlib import Path

import pytest

from perfxpert.tools import patch_mgr
from perfxpert.tools._safety import PathConfinementError


def _symlink_or_skip(link: Path, target: Path) -> None:
    try:
        link.symlink_to(target)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"symlink creation unavailable on this host: {exc}")


TRAVERSAL_PAYLOADS = [
    "../../../etc/passwd",
    "../outside",
    "/etc/passwd",
    "/etc/shadow",
    "/root/.ssh/authorized_keys",
    "../../Library/Application Support/keys",
]


@pytest.mark.parametrize("payload", TRAVERSAL_PAYLOADS)
def test_patch_apply_rejects_traversal(payload, tmp_path: Path):
    (tmp_path / "ok.cpp").write_text("// hi\n")
    with pytest.raises(PathConfinementError):
        patch_mgr.apply(tmp_path, payload, "evil\n")


def test_symlink_escape_rejected(tmp_path: Path):
    outside = tmp_path.parent / "outside_target.txt"
    outside.write_text("target\n")
    inside = tmp_path / "link.cpp"
    _symlink_or_skip(inside, outside)

    with pytest.raises(PathConfinementError):
        patch_mgr.apply(tmp_path, "link.cpp", "malicious\n")


def test_symlink_chain_escape_rejected(tmp_path: Path):
    """A-chains that eventually escape must also be rejected."""
    outside = tmp_path.parent / "outside_final.txt"
    outside.write_text("target\n")
    middle = tmp_path.parent / "outside_middle"
    _symlink_or_skip(middle, outside)
    inside = tmp_path / "chain.cpp"
    _symlink_or_skip(inside, middle)

    with pytest.raises(PathConfinementError):
        patch_mgr.apply(tmp_path, "chain.cpp", "malicious\n")
