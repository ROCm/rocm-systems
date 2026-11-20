# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Contracts for canonical identity and orthogonal decoded instruction facts."""

from types import SimpleNamespace

import pytest

from amdisa.codegen._generator import CodeGenerator
from amdisa.codegen.instruction_metadata import (
    canonical_operation_name,
    classification_flags,
    emit_operations,
    memory_classification,
    operation_expression,
    operation_names,
)


def _spec(names, *, arch='rdna4', slots=()):
    return SimpleNamespace(
        arch_name=arch,
        cpp_namespace=arch,
        inst_encodings=[
            SimpleNamespace(insts=[SimpleNamespace(name=name) for name in names])
        ],
        profile=SimpleNamespace(vopd_slot_ops=slots, legacy_flat_lds=False),
    )


def test_operation_catalog_canonicalizes_forms_and_includes_resolved_segments(tmp_path):
    first = _spec(['V_MOV_B32', 'V_MOV_B32_E32', 'FLAT_LOAD_DWORD'])
    second = _spec(
        ['GLOBAL_LOAD_DWORD', 'S_NOP'],
        slots=[SimpleNamespace(mnemonic='v_dual_cndmask_b32')],
    )
    expected = [
        'FLAT_LOAD_DWORD',
        'GLOBAL_LOAD_DWORD',
        'SCRATCH_LOAD_DWORD',
        'S_NOP',
        'V_DUAL_CNDMASK_B32',
        'V_MFMA_F32_16X16X16_BF16',
        'V_MFMA_F32_16X16X16_F16',
        'V_MFMA_F32_16X16X32_F16',
        'V_MFMA_I32_16X16X32_I8',
        'V_MOV_B32',
    ]
    specs = [('first', first, None), ('second', second, None)]
    assert operation_names(specs) == expected
    assert emit_operations(str(tmp_path), specs) == expected
    initial = (tmp_path / 'shared/operations.h').read_bytes()
    emit_operations(str(tmp_path), list(reversed(specs)))
    assert (tmp_path / 'shared/operations.h').read_bytes() == initial


def test_custom_operation_is_unknown_only_when_base_cannot_represent_it():
    names = frozenset({'V_MOV_B32'})
    assert (
        operation_expression('v_mov_b32_e32', names) == 'amdgpu::Operation::V_MOV_B32'
    )
    assert operation_expression('V_CUSTOM_ADD', names) == 'amdgpu::Operation::Unknown'
    assert canonical_operation_name('V_CVT_F32_F16') == 'V_CVT_F32_F16'


@pytest.mark.parametrize(
    ('name', 'sem_class', 'addressing', 'kind'),
    [
        ('S_MOV_B32', 'scalar_mov', 'None', 'None'),
        ('S_LOAD_DWORD', 'smem_load', 'Scalar', 'Load'),
        ('S_SCRATCH_STORE_B32', 'smem_store', 'Scratch', 'Store'),
        ('S_ATC_PROBE', None, 'Scalar', 'Probe'),
        ('S_MEMTIME', None, 'Scalar', 'Load'),
        ('BUFFER_ATOMIC_ADD', 'buffer_atomic', 'Buffer', 'Atomic'),
        ('TBUFFER_STORE_FORMAT_X', 'tbuffer_store', 'TypedBuffer', 'Store'),
        ('IMAGE_SAMPLE_D', 'image_sample', 'Image', 'Sample'),
        ('IMAGE_BVH_INTERSECT_RAY', None, 'Image', 'Bvh'),
        ('DS_ADD_RTN_U32', 'ds_atomic', 'Lds', 'Atomic'),
        ('DS_GWS_BARRIER', 'ds_gws', 'Gds', 'Other'),
        ('DS_DIRECT_LOAD', 'lds_direct_load', 'DirectLds', 'Load'),
        ('GLOBAL_LOAD_ASYNC_TO_LDS_B128', None, 'Global', 'Load'),
        ('TENSOR_STORE_FROM_LDS', None, 'Tensor', 'Store'),
        ('S_DCACHE_INV', None, 'Scalar', 'Cache'),
        ('S_PREFETCH_INST', None, 'Scalar', 'Prefetch'),
    ],
)
def test_memory_categories_separate_addressing_from_effect(
    name, sem_class, addressing, kind
):
    assert memory_classification(name, sem_class) == (addressing, kind)


@pytest.mark.parametrize(
    ('name', 'expected'),
    [
        ('S_MOVRELS_B32', {'RELATIVE_SGPR_SOURCE'}),
        ('S_MOVRELD_B32', {'RELATIVE_SGPR_DESTINATION'}),
        ('S_MOVRELSD_B32', {'RELATIVE_SGPR_SOURCE', 'RELATIVE_SGPR_DESTINATION'}),
        ('V_SWAPREL_B32', {'RELATIVE_VGPR_SOURCE', 'RELATIVE_VGPR_DESTINATION'}),
        ('V_INTERP_P2_F32', {'INTERPOLATION'}),
        ('V_WMMA_SCALE16_F32_32X16X128_F4', {'MATRIX_OP'}),
        ('S_GET_PC_I64', {'READS_PC'}),
        ('S_PREFETCH_INST_PC_REL', {'READS_PC'}),
        ('S_PREFETCH_DATA_PC_REL', {'READS_PC'}),
        ('GLOBAL_LOAD_ASYNC_TO_LDS_B128', {'MEMORY_TO_LDS'}),
        ('TENSOR_STORE_FROM_LDS', {'MEMORY_FROM_LDS'}),
    ],
)
def test_shared_register_and_transfer_properties(name, expected):
    assert set(classification_flags(name, None)) == expected


def test_decoded_segment_does_not_confuse_scratch_vaddr_with_lds_transfer():
    codegen = object.__new__(CodeGenerator)
    codegen.isa_spec = _spec([])
    statements = '\n'.join(
        codegen._instruction_metadata_stmts(
            SimpleNamespace(name='FLAT_LOAD_DWORD'),
            SimpleNamespace(semantic_class='flat_load'),
            {'seg', 'lds'},
        )
    )
    assert 'inst_.seg == 1 ? amdgpu::Operation::SCRATCH_LOAD_DWORD' in statements
    assert 'inst_.seg == 2 ? amdgpu::Operation::GLOBAL_LOAD_DWORD' in statements
    assert 'amdgpu::MemoryAddressing::Flat' in statements
    assert 'MEMORY_TO_LDS' not in statements
    assert 'mnemonic' not in statements


def test_legacy_flat_lds_transfer_reuses_the_execution_predicate():
    codegen = object.__new__(CodeGenerator)
    codegen.isa_spec = _spec([])
    codegen.isa_spec.profile.legacy_flat_lds = True
    codegen.isa_spec.profile.legacy_flat_d16_lds = False
    statements = '\n'.join(
        codegen._instruction_metadata_stmts(
            SimpleNamespace(name='FLAT_LOAD_DWORD'),
            SimpleNamespace(
                semantic_class='flat_load',
                num_elems=1,
                elem_size=4,
                d16_hi=False,
                d16_lo=False,
            ),
            {'seg', 'lds'},
        )
    )
    assert (
        'if (inst_.lds && (inst_.seg == 1 || inst_.seg == 2)) flags_ |= MEMORY_TO_LDS;'
        in statements
    )


def test_gds_bit_changes_addressing_without_changing_atomic_identity():
    codegen = object.__new__(CodeGenerator)
    codegen.isa_spec = _spec([])
    statements = '\n'.join(
        codegen._instruction_metadata_stmts(
            SimpleNamespace(name='DS_ADD_RTN_U32'),
            SimpleNamespace(semantic_class='ds_atomic'),
            {'gds'},
        )
    )
    assert 'set_amdgpu_operation(amdgpu::Operation::DS_ADD_RTN_U32);' in statements
    assert 'inst_.gds ? amdgpu::MemoryAddressing::Gds' in statements
    assert 'amdgpu::MemoryKind::Atomic' in statements


@pytest.mark.parametrize(
    'name', ['BUFFER_LOAD_FORMAT_D16_XY', 'BUFFER_LOAD_FORMAT_D16_XYZW']
)
def test_full_dword_packed_tuples_retain_d16_data_layout(name):
    # Both forms fill complete DWORDs: neither a partial-byte mask nor a
    # preservation-only read identifies this data layout.
    sem = SimpleNamespace(semantic_class='buffer_load_format_d16', elem_size=2)
    assert 'D16_MEMORY_DATA' in classification_flags(name, sem)
    assert 'D16_MEMORY_DATA' not in classification_flags('BUFFER_LOAD_DWORD', None)
