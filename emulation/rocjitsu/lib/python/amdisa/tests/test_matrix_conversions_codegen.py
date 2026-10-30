# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Test matrix-header CLI regeneration, output helpers, and standalone compilation."""

import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from amdisa.codegen.dbt.rule_gen import (
    emit_matrix_conversions_header,
    generate_catalog_matrix_rules,
    generate_matrix_conversions,
)

_REPO = Path(__file__).resolve().parents[6]
_ROCJITSU = _REPO / 'emulation' / 'rocjitsu'
_MRISA = Path(
    os.environ.get('MRISA_PATH', _REPO / 'shared' / 'machine-readable-isa' / 'isa')
).resolve()


@pytest.mark.parametrize('output_option', ['--dbt-output', '--isa-output'])
@pytest.mark.parametrize('existing_header', [False, True], ids=['missing', 'stale'])
def test_dbt_cli_regenerates_matrix_header(tmp_path, output_option, existing_header):
    shutil.copyfile(_ROCJITSU / '.clang-format', tmp_path / '.clang-format')
    output = tmp_path / 'generated'
    header = output / 'matrix_conversions.h'
    if existing_header:
        output.mkdir()
        header.write_text('// stale output must be replaced\n', encoding='utf-8')

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
    # The committed header is the compatibility oracle for the catalog's
    # complete contents, including the derived lane permutation values.
    if shutil.which('clang-format') is not None:
        assert header.read_bytes() == checked_in.read_bytes()
    else:
        # The output helper leaves formatting to the caller without clang-format.
        assert ''.join(header.read_text(encoding='utf-8').split()) == ''.join(
            checked_in.read_text(encoding='utf-8').split()
        )


def test_matrix_output_creates_directory_without_formatter(tmp_path, monkeypatch):
    monkeypatch.setenv('PATH', '')
    output = tmp_path / 'nested' / 'generated'
    header = generate_matrix_conversions(output)
    assert header == output / 'matrix_conversions.h'
    assert header.read_text(encoding='utf-8') == emit_matrix_conversions_header(
        generate_catalog_matrix_rules()
    )


def test_matrix_output_matches_tracked_formatting(tmp_path):
    if shutil.which('clang-format') is None:
        pytest.skip('clang-format is unavailable')
    shutil.copyfile(_ROCJITSU / '.clang-format', tmp_path / '.clang-format')
    output = tmp_path / 'nested' / 'generated'
    header = generate_matrix_conversions(output)
    checked_in = (
        _ROCJITSU / 'lib/rocjitsu/src/rocjitsu/code/dbt/generated/matrix_conversions.h'
    )
    assert header.read_bytes() == checked_in.read_bytes()
    header.write_text('// stale output must be replaced\n', encoding='utf-8')
    generate_matrix_conversions(output)
    assert header.read_bytes() == checked_in.read_bytes()


def test_emitted_matrix_header_compiles_standalone(tmp_path):
    compiler = shutil.which('clang++') or shutil.which('c++')
    if compiler is None:
        pytest.skip('C++ compiler is unavailable')
    header = tmp_path / 'matrix_conversions.h'
    header.write_text(
        emit_matrix_conversions_header(generate_catalog_matrix_rules()),
        encoding='utf-8',
    )
    source = tmp_path / 'matrix_conversions.cpp'
    source.write_text(
        '#include "matrix_conversions.h"\n'
        'static_assert(rocjitsu::kMatrixConversionCount == 4);\n',
        encoding='utf-8',
    )
    result = subprocess.run(
        [compiler, '-std=c++20', '-pedantic-errors', '-fsyntax-only', str(source)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
