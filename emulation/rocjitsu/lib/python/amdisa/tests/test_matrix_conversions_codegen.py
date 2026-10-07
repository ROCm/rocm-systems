# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Exercise matrix-header regeneration through the public CLI."""

import os
from pathlib import Path
import subprocess
import sys

import pytest

_REPO = Path(__file__).resolve().parents[6]
_ROCJITSU = _REPO / 'emulation' / 'rocjitsu'
_MRISA = Path(
    os.environ.get('MRISA_PATH', _REPO / 'shared' / 'machine-readable-isa' / 'isa')
).resolve()


@pytest.mark.parametrize(
    ('output_option', 'existing_header'),
    [('--dbt-output', False), ('--isa-output', True)],
)
def test_dbt_cli_regenerates_matrix_header(tmp_path, output_option, existing_header):
    output = tmp_path / 'generated'
    header = output / 'matrix_conversions.h'
    if existing_header:
        output.mkdir()
        header.write_text('// stale output must be replaced\n')

    result = subprocess.run(
        [
            sys.executable,
            '-m',
            'amdisa',
            '--gen-dbt',
            output_option,
            str(output),
            f'cdna4:{_MRISA / "amdgpu_isa_cdna4.xml"}',
            f'rdna4:{_MRISA / "amdgpu_isa_rdna4.xml"}',
        ],
        cwd=tmp_path,
        env={**os.environ, 'PYTHONPATH': str(_ROCJITSU / 'lib' / 'python')},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert header.is_file(), result.stdout + result.stderr

    checked_in = (
        _ROCJITSU / 'lib/rocjitsu/src/rocjitsu/code/dbt/generated/matrix_conversions.h'
    )
    # The CLI emits unformatted C++; the whole-tree helper runs clang-format.
    # The committed header is the compatibility oracle for the catalog's
    # complete contents, including the derived lane permutation values.
    assert ''.join(header.read_text().split()) == ''.join(
        checked_in.read_text().split()
    )
