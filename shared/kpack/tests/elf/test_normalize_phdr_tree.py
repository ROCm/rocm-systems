"""Tests for the post-strip PHDR normalization tool (rocm_kpack.tools.normalize_phdr_tree).

Guards the packaging regression: the kpack split pins the relocated PHDR to
p_vaddr == p_offset, but `strip` (run by rpm/deb packaging) removes sections
ahead of the trailing table and shifts its file offset, re-breaking it. The tool
re-applies the normalization and (with --check) gates against it.
"""

import shutil
import subprocess
from pathlib import Path

import pytest

from rocm_kpack.elf import ElfSurgery, PT_LOAD
from rocm_kpack.elf.kpack_transform import kpack_offload_binary
from rocm_kpack.tools.normalize_phdr_tree import main as normalize_main


def _phdr_covering_load(path: Path):
    surgery = ElfSurgery.load(path)
    e_phoff = surgery.ehdr.e_phoff
    covering = [
        ph
        for _, ph in surgery.iter_program_headers()
        if ph.p_type == PT_LOAD and ph.p_offset <= e_phoff < ph.p_offset + ph.p_filesz
    ]
    assert covering, "no PT_LOAD covers the program header table"
    return covering[0]


def _strip_tool():
    for tool in ("llvm-strip-20", "llvm-strip", "strip"):
        if shutil.which(tool):
            return tool
    return None


@pytest.mark.skipif(_strip_tool() is None, reason="no strip tool available")
def test_strip_breaks_phdr_and_normalize_restores(test_assets_dir: Path, tmp_path: Path):
    inp = test_assets_dir / "bundled_binaries/linux/cov5/test_kernel_single.exe"
    out = tmp_path / "out.exe"
    kpack_offload_binary(
        input_path=inp,
        output_path=out,
        kpack_search_paths=["test.kpack"],
        kernel_name="test_kernel",
    )

    # Split-time normalization: relocated PHDR has p_vaddr == p_offset.
    ph = _phdr_covering_load(out)
    assert ph.p_vaddr == ph.p_offset, "kpack offload should leave p_vaddr == p_offset"

    # strip (as rpm/deb packaging does) re-breaks it.
    subprocess.run([_strip_tool(), str(out)], check=True)
    ph = _phdr_covering_load(out)
    assert ph.p_vaddr != ph.p_offset, "strip is expected to re-break p_vaddr == p_offset"

    # --check flags the non-compliant binary (non-zero exit).
    assert normalize_main(["--check", str(out)]) == 1

    # Apply restores p_vaddr == p_offset.
    assert normalize_main([str(out)]) == 0
    ph = _phdr_covering_load(out)
    assert ph.p_vaddr == ph.p_offset, "normalize should restore p_vaddr == p_offset after strip"

    # Now compliant and idempotent.
    assert normalize_main(["--check", str(out)]) == 0
