# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Focused generators must share the consuming core's operation identities."""

import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

import amdisa
from amdisa.codegen.instruction_metadata import read_operation_names

_ROCJITSU = Path(__file__).resolve().parents[4]


def _run(command, **kwargs):
    result = subprocess.run(
        command, capture_output=True, text=True, timeout=180, **kwargs
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result


def _core_header():
    return (
        _ROCJITSU
        / 'lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/generated/shared/operations.h'
    )


def _core_alias(tmp_path):
    # The core is also reached through instruction.h. Both spellings must name
    # the same enum, without requiring a particular filename or include prefix.
    header = tmp_path / 'core headers' / 'selected-catalog.h'
    header.parent.mkdir()
    header.symlink_to(_core_header())
    return header


def _xml(arch):
    root = Path(
        os.environ.get(
            'MRISA_PATH', _ROCJITSU.parents[1] / 'shared/machine-readable-isa/isa'
        )
    )
    return f'{arch}:{root / ("amdgpu_isa_" + arch + ".xml")}'


def _compiler_flags():
    compiler = shutil.which('clang++') or shutil.which('c++')
    if compiler is None:
        pytest.skip('C++ compiler is unavailable')
    runtime = Path(
        os.environ.get(
            'RJ_RUNTIME_API_HEADERS_SOURCE_DIR',
            _ROCJITSU.parents[1] / 'runtimes/api-headers',
        )
    )
    includes = [
        _ROCJITSU / 'lib/rocjitsu/src',
        _ROCJITSU / 'lib/rocjitsu/include',
        runtime / 'include',
        *(_ROCJITSU / f'lib/{name}/include' for name in ('arith', 'util', 'simdojo')),
    ]
    return [
        compiler,
        '-std=c++20',
        *(flag for root in includes for flag in ('-I', str(root))),
    ]


@pytest.mark.parametrize(
    'explicit_catalog', [False, True], ids=['checkout', 'explicit']
)
def test_combined_generation_compiles_actual_dbt_consumer(tmp_path, explicit_catalog):
    include = tmp_path / 'include'
    output = include / 'isa'
    dbt = tmp_path / 'dbt'
    core_header = _core_alias(tmp_path)
    _run(
        [
            sys.executable,
            '-m',
            'amdisa',
            _xml('cdna4'),
            _xml('rdna4'),
            '--include-root',
            str(include),
            '--isa-output',
            str(output),
            '--dbt-output',
            str(dbt),
            *(['--core-operations', str(core_header)] if explicit_catalog else []),
        ],
        env={
            **os.environ,
            'PYTHONPATH': str(Path(amdisa.__file__).resolve().parents[1]),
        },
    )
    assert not (output / 'shared/operations.h').exists()
    if explicit_catalog:
        for header in (
            dbt / 'matrix_conversions.h',
            output / 'shared/async_inline_operations.h',
        ):
            assert (
                f'#include "{core_header.resolve().as_posix()}"' in header.read_text()
            )
    # Overlay the generated DBT header at the spelling used by the real consumer.
    overlay = include / 'rocjitsu/code/dbt/generated'
    overlay.parent.mkdir(parents=True)
    overlay.symlink_to(dbt, target_is_directory=True)
    consumer = (
        _ROCJITSU / 'lib/rocjitsu/src/rocjitsu/code/dbt/semantic/cdna4_to_rdna4.cpp'
    )
    _run(
        [
            _compiler_flags()[0],
            '-I',
            str(include),
            *_compiler_flags()[1:],
            '-include',
            str(output / 'shared/async_inline_operations.h'),
            '-fsyntax-only',
            str(consumer),
        ]
    )


def test_installed_generator_decodes_wait_with_core_catalog(tmp_path):
    package = tmp_path / 'installed'
    package_source = tmp_path / 'package-source'
    shutil.copytree(
        Path(amdisa.__file__).resolve().parents[1],
        package_source,
        ignore=shutil.ignore_patterns(
            '__pycache__', '*.egg-info', 'build', '.pytest_cache'
        ),
    )
    _run(
        [
            sys.executable,
            '-m',
            'pip',
            'install',
            '--no-deps',
            '--no-build-isolation',
            '--target',
            str(package),
            str(package_source),
        ]
    )
    env = {**os.environ, 'PYTHONPATH': str(package)}
    result = _run(
        [sys.executable, '-c', 'import amdisa; print(amdisa.__file__)'],
        cwd=tmp_path,
        env=env,
    )
    assert Path(result.stdout.strip()).is_relative_to(package)
    output = tmp_path / 'isa'
    command = [
        sys.executable,
        '-m',
        'amdisa',
        _xml('cdna1'),
        '--gen-isas',
        '--isa-output',
        str(output),
    ]
    missing = subprocess.run(
        command, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=180
    )
    assert missing.returncode != 0
    assert '--core-operations' in missing.stderr
    _run(
        [*command, '--core-operations', str(_core_alias(tmp_path))],
        cwd=tmp_path,
        env=env,
    )
    source = tmp_path / 'wait.cpp'
    source.write_text('''
#include "cdna1/decode_factories.h"
#include "cdna1/machine_insts.h"
#include "rocjitsu/isa/arch/amdgpu/shared/wait_counter_policy.h"
#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/code/analysis/waitcheck/state.h"
#include <cassert>
int main() {
  using namespace rocjitsu;
  const cdna1::MachineInst opcode = 0xbf8c0f70;
  auto decoded = cdna1::detail::decodeSWaitcntSopp(&opcode, {});
  assert(decoded.succeeded());
  const auto &inst = *decoded.value();
  assert(inst.is_waitcnt());
  assert(inst.amdgpu_operation() == amdgpu::Operation::S_WAITCNT);
  auto fields = amdgpu::WaitCounterPolicy::explicit_wait_fields(inst, rocjitsu::IsaFamily::Cdna1);
  assert(fields.succeeded() && fields.value().has_value());
  assert((*fields.value())[static_cast<size_t>(amdgpu::WaitCounterKind::Load)] == 0);
  assert(!(*fields.value())[static_cast<size_t>(amdgpu::WaitCounterKind::Ds)]);
  assert(!(*fields.value())[static_cast<size_t>(amdgpu::WaitCounterKind::Exp)]);
  waitcheck_detail::PendingState state;
  auto &loads = state.pending[static_cast<size_t>(amdgpu::WaitCounterKind::Load)];
  loads.emplace_back();
  assert(waitcheck_detail::WaitcheckStateOps::apply_waitcnt(
      state, inst, rocjitsu::IsaFamily::Cdna1).succeeded());
  assert(loads.empty());
}
''')
    core = _ROCJITSU / 'lib/rocjitsu/src/rocjitsu'
    executable = tmp_path / 'wait'
    _run(
        [
            *_compiler_flags(),
            '-I',
            str(output),
            '-ffunction-sections',
            '-fdata-sections',
            str(source),
            str(output / 'cdna1/sopp.cpp'),
            str(output / 'cdna1/encodings.cpp'),
            str(output / 'cdna1/operand.cpp'),
            str(core / 'isa/execution_backend.cpp'),
            str(core / 'isa/operand.cpp'),
            str(core / 'isa/arch/amdgpu/shared/wait_counter_policy.cpp'),
            str(core / 'code/analysis/waitcheck/state.cpp'),
            str(core / 'code/analysis/waitcheck/target.cpp'),
            '-Wl,--gc-sections',
            '-o',
            str(executable),
        ]
    )
    _run([str(executable)])


@pytest.mark.parametrize(
    'body', ['', 'Unknown, S_WAITCNT = 4, Count,', 'Unknown, S_WAITCNT, S_NOP, Count,']
)
def test_invalid_core_catalog_is_rejected(tmp_path, body):
    header = tmp_path / 'operations.h'
    header.write_text(f'enum class Operation : uint16_t {{{body}}};')
    with pytest.raises(ValueError, match='invalid generated core operation catalog'):
        read_operation_names(header)


def test_generator_api_uses_selected_core_catalog(tmp_path):
    from amdisa.codegen._generator import CodeGenerator
    from amdisa.codegen.config import CodegenConfig

    header = tmp_path / 'operations.h'
    header.write_text('enum class Operation : uint16_t { Unknown, S_WAITCNT, Count, };')
    generator = object.__new__(CodeGenerator)
    generator.config = CodegenConfig(core_operations_header=str(header))
    assert (
        generator._operation_expression('S_WAITCNT') == 'amdgpu::Operation::S_WAITCNT'
    )
    assert generator._operation_expression('V_NEW_F32') == 'amdgpu::Operation::Unknown'
