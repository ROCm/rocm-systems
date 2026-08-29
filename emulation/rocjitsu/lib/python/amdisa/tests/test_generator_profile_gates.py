# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import xml.etree.ElementTree as elem_tree
import os
import re
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from amdisa.__main__ import (
    _collect_shared_execute_body_variants,
    _run,
    _unshared_execute_keys_from_variants,
)
from amdisa.codegen import CodeGenerator
from amdisa.codegen.config import CodegenConfig
from amdisa.codegen.execute import ExecuteContext
from amdisa.codegen.execute.vector_special import (
    gen_cvt_fp8,
    gen_cvt_scalef32,
    gen_vector_mad_64_32,
    gen_vector_div_scale,
    gen_vector_movrel,
    gen_vector_cvt_pk,
)
from amdisa.codegen.execute.vector_alu import gen_vector_unary
from amdisa.codegen.execute.matrix import gen_mfma as emit_mfma
from amdisa.codegen.execute.vector_cmp import (
    gen_vector_add_co,
    gen_vector_cmp,
    gen_vector_cmp_class,
    gen_vector_cmpx,
)
from amdisa.codegen.execute.simd_codegen import simd_probe_line
from amdisa.cross_isa import CrossIsaAnalyzer
from amdisa.gpuisa import Instruction, Operand
from amdisa.isa_profile import (
    Cdna1Profile,
    Cdna2Profile,
    Cdna4Profile,
    CdnaProfile,
    Cdna5Profile,
    DppOpcodeRule,
    MatrixLayout,
    Rdna1Profile,
    Rdna2Profile,
    Rdna3_5Profile,
    Rdna3Profile,
    Rdna4Profile,
    SwmmacLayout,
)
from amdisa.parser import Parser
from amdisa.semantics import (
    InstructionSemantics,
    derive_all_semantics,
    derive_semantics,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[6]


def _mrisa_dir() -> Path:
    default = _repo_root() / 'shared' / 'machine-readable-isa' / 'isa'
    return Path(os.environ.get('MRISA_PATH', default))


def _gen_mfma(
    inst: Instruction,
    arch_name: str,
    profile=None,
    enc_field_names: set[str] | None = None,
    op_sel_hi_2_expr: str = 'inst_.op_sel_hi_2',
) -> str:
    profiles = {
        'cdna1': Cdna1Profile,
        'cdna2': Cdna2Profile,
        'cdna3': CdnaProfile,
        'cdna4': CdnaProfile,
        'rdna3': Rdna3Profile,
        'rdna3_5': Rdna3_5Profile,
        'rdna4': Rdna4Profile,
        'cdna5': Cdna5Profile,
    }
    profile = profile or profiles[arch_name]()
    return emit_mfma(
        ExecuteContext(
            inst=inst,
            sem=InstructionSemantics(inst.name, 'mfma'),
            dst_ops=['vdst'],
            src_ops=['src0', 'src1', 'src2'],
            profile=profile,
            enc_name=inst.enc_name,
            is_vop3=True,
            has_abs=False,
            arch_name=arch_name,
            enc_field_names=set() if enc_field_names is None else enc_field_names,
            op_sel_hi_2_expr=op_sel_hi_2_expr,
        )
    )


def _matrix_profile(
    *,
    uses_vgpr_msb_indexing: bool = False,
    swmmac_layout: SwmmacLayout = SwmmacLayout.NONE,
    matrix_layout: MatrixLayout = MatrixLayout.MFMA_ACCUMULATOR,
    wave_size: int = 64,
    wave_size_max: int | None = None,
    supports_gpr_idx: bool = False,
) -> SimpleNamespace:
    return SimpleNamespace(
        uses_vgpr_msb_indexing=uses_vgpr_msb_indexing,
        swmmac_layout=swmmac_layout,
        matrix_layout=matrix_layout,
        wave_size=wave_size,
        wave_size_max=wave_size if wave_size_max is None else wave_size_max,
        supports_gpr_idx=supports_gpr_idx,
    )


def _matrix_profile_for_arch(
    arch_name: str,
    *,
    uses_vgpr_msb_indexing: bool,
) -> SimpleNamespace:
    if arch_name == 'cdna5':
        return _matrix_profile(
            uses_vgpr_msb_indexing=uses_vgpr_msb_indexing,
            swmmac_layout=SwmmacLayout.FIXED_WAVE,
            matrix_layout=MatrixLayout.WMMA_SPLIT_K,
            wave_size=32,
            wave_size_max=32,
        )
    if arch_name == 'rdna4':
        return _matrix_profile(
            uses_vgpr_msb_indexing=uses_vgpr_msb_indexing,
            swmmac_layout=SwmmacLayout.RUNTIME_WAVE,
            matrix_layout=MatrixLayout.WMMA_SPLIT_K,
            wave_size=32,
            wave_size_max=64,
        )
    if arch_name in ('rdna3', 'rdna3_5'):
        return _matrix_profile(
            uses_vgpr_msb_indexing=uses_vgpr_msb_indexing,
            matrix_layout=MatrixLayout.WMMA_REPLICATED_HALFWAVE,
            wave_size=32,
            wave_size_max=64,
        )
    return _matrix_profile(uses_vgpr_msb_indexing=uses_vgpr_msb_indexing)


@pytest.fixture
def rocjitsu_source_root() -> Path:
    return Path(__file__).resolve().parents[4]


@pytest.fixture
def amdgpu_root(rocjitsu_source_root: Path) -> Path:
    return (
        rocjitsu_source_root
        / 'lib'
        / 'rocjitsu'
        / 'src'
        / 'rocjitsu'
        / 'isa'
        / 'arch'
        / 'amdgpu'
    )


@pytest.fixture
def amdgpu_generated_root(amdgpu_root: Path) -> Path:
    return amdgpu_root / 'generated'


@pytest.fixture
def execute_shared_path(amdgpu_generated_root: Path) -> Path:
    return amdgpu_generated_root / 'shared' / 'execute_shared.h'


@pytest.fixture
def gfx1250_generated_root(amdgpu_generated_root: Path) -> Path:
    profile = Cdna5Profile()
    assert profile.generated_arch_name is not None
    return amdgpu_generated_root / _generated_dir_name(profile.generated_arch_name)


def _generated_dir_name(arch_name: str) -> str:
    profile = Cdna5Profile()
    if arch_name in ('gfx1250', profile.generated_arch_name):
        assert profile.generated_dir_name is not None
        return profile.generated_dir_name
    return arch_name


def _profile_for_arch(arch_name: str):
    profile_types = {
        'cdna1': Cdna1Profile,
        'cdna2': Cdna2Profile,
        'cdna3': CdnaProfile,
        'cdna4': Cdna4Profile,
        'rdna1': Rdna1Profile,
        'rdna2': Rdna2Profile,
        'rdna3': Rdna3Profile,
        'rdna3_5': Rdna3_5Profile,
        'rdna4': Rdna4Profile,
        'cdna5': Cdna5Profile,
        'gfx1250': Cdna5Profile,
    }
    return profile_types[arch_name]()


@pytest.mark.parametrize(
    'arch_name',
    (
        'cdna1',
        'cdna2',
        'cdna3',
        'cdna4',
        'rdna1',
        'rdna2',
        'rdna3',
        'rdna3_5',
        'rdna4',
    ),
)
def test_shared_scalar_pair_selector_contract_matches_in_tree_isas(
    arch_name: str,
) -> None:
    spec = Parser(
        str(_mrisa_dir() / f'amdgpu_isa_{arch_name}.xml'),
        _profile_for_arch(arch_name),
    ).parse()

    CodeGenerator(spec, '')._validate_shared_scalar_pair_selector_contract()


def test_shared_scalar_pair_selector_contract_rejects_value_drift() -> None:
    arch_name = 'cdna4'
    spec = Parser(
        str(_mrisa_dir() / f'amdgpu_isa_{arch_name}.xml'),
        _profile_for_arch(arch_name),
    ).parse()
    selector = next(
        item for item in spec.opnd_selectors if item.operand_type == 'OPR_SSRC'
    )
    selector.op_sel_vals = [
        (name, '105' if name == 'OPR_SSRC_VCC_LO' else value)
        for name, value in selector.op_sel_vals
    ]

    with pytest.raises(ValueError, match=r'OPR_SSRC_VCC_LO=106'):
        CodeGenerator(spec, '')._validate_shared_scalar_pair_selector_contract()


def gen_mfma(
    inst: Instruction,
    dst: list[str],
    src: list[str],
    arch_name: str,
    *,
    supports_gpr_idx: bool | None = None,
    enc_field_names: set[str] | None = None,
    op_sel_hi_2_expr: str = 'inst_.op_sel_hi_2',
) -> str:
    """Call the matrix emitter with the selected ISA profile capability."""
    profile = _profile_for_arch(arch_name)
    if supports_gpr_idx is not None:
        profile = SimpleNamespace(
            uses_vgpr_msb_indexing=profile.uses_vgpr_msb_indexing,
            swmmac_layout=profile.swmmac_layout,
            matrix_layout=profile.matrix_layout,
            wave_size=profile.wave_size,
            wave_size_max=profile.wave_size_max,
            supports_gpr_idx=supports_gpr_idx,
        )
    return _gen_mfma(
        inst,
        arch_name,
        profile,
        enc_field_names=enc_field_names,
        op_sel_hi_2_expr=op_sel_hi_2_expr,
    )


@pytest.fixture
def rdna4_generated_root(amdgpu_generated_root: Path) -> Path:
    return amdgpu_generated_root / 'rdna4'


@pytest.fixture
def cdna4_generated_root(amdgpu_generated_root: Path) -> Path:
    return amdgpu_generated_root / 'cdna4'


def _shared_execute_body(execute_shared: str, name: str, next_name: str) -> str:
    start = execute_shared.index(f'inline void execute_{name}')
    end = execute_shared.index(f'inline void execute_{next_name}', start)
    return execute_shared[start:end]


def _generated_method_body(cpp: str, class_name: str, next_class_name: str) -> str:
    start = cpp.index(f'void {class_name}::execute_impl')
    end_markers = (
        f'{next_class_name}::{next_class_name}',
        f'void {next_class_name}::execute_impl',
    )
    end = min(
        cpp.index(marker, start) for marker in end_markers if marker in cpp[start:]
    )
    return cpp[start:end]


def _generated_function_body(cpp: str, signature: str) -> str:
    start = cpp.index(signature)
    opening = cpp.index('{', start)
    depth = 0
    for offset, char in enumerate(cpp[opening:], start=opening):
        if char == '{':
            depth += 1
        elif char == '}':
            depth -= 1
            if depth == 0:
                return cpp[start : offset + 1]
    raise AssertionError(f'unterminated generated function: {signature}')


def _generated_class_body(header: str, class_name: str) -> str:
    start = header.index(f'class {class_name} ')
    end = header.index('\n};', start)
    return header[start : end + 3]


def _generated_inline_method_body(header: str, class_name: str, signature: str) -> str:
    return _generated_function_body(
        _generated_class_body(header, class_name), signature
    )


def test_gfx1250_addtid_uses_m0_byte_base_addresses(
    gfx1250_generated_root: Path,
) -> None:
    vds = (gfx1250_generated_root / 'vds_exec.cpp').read_text()
    store_body = _generated_method_body(
        vds, 'DsStoreAddtidB32Vds', 'DsLoadAddtidB32Vds'
    )
    load_body = _generated_method_body(vds, 'DsLoadAddtidB32Vds', 'DsPermuteB32Vds')
    for body in (store_body, load_body):
        assert 'uint32_t lane_offset = (m0 + lane * 4U) & 0xFFFFFU;' in body
        assert 'lane_offset + offset + wf.lds_base()' in body
        assert 'ds_stride_bytes' not in body


def test_gfx1250_returning_two_address_exchange_b32_uses_dual_atomic_state(
    gfx1250_generated_root: Path,
) -> None:
    vds = (gfx1250_generated_root / 'vds_exec.cpp').read_text()
    body = _generated_method_body(
        vds,
        'DsStorexchg2addrRtnB32Vds',
        'DsStorexchg2addrStride64RtnB32Vds',
    )
    assert 'd->wf_size = wf.wf_size();' in body
    assert 'd->ds2_active = true;' in body
    assert 'd->ds2_per_lane_addr[lane]' in body
    assert 'd->ds2_store_data' in body
    assert 'd->ds2_dst_reg_base' in body


def test_gfx1250_profile_scopes_special_ds_semantics() -> None:
    names = (
        'DS_STORE_ADDTID_B32',
        'DS_STOREXCHG_2ADDR_RTN_B32',
        'DS_STOREXCHG_2ADDR_RTN_B64',
        'DS_STOREXCHG_2ADDR_STRIDE64_RTN_B32',
        'DS_STOREXCHG_2ADDR_STRIDE64_RTN_B64',
    )

    def semantics_for(profile):
        spec = SimpleNamespace(
            profile=profile,
            inst_encodings=[
                SimpleNamespace(
                    enc_name='ENC_VDS',
                    insts=[SimpleNamespace(name=name) for name in names],
                )
            ],
        )
        return derive_all_semantics(spec)

    gfx1250 = semantics_for(Cdna5Profile())
    assert Cdna5Profile().ds_addtid_uses_m0_byte_base
    assert gfx1250['DS_STORE_ADDTID_B32'].semantic_class == 'ds_write_addtid'
    for name in names[1:]:
        assert gfx1250[name].semantic_class == 'ds_atomic2'
        assert gfx1250[name].operation == 'swap'

    rdna4 = semantics_for(Rdna4Profile())
    assert not Rdna4Profile().ds_addtid_uses_m0_byte_base
    assert rdna4['DS_STORE_ADDTID_B32'].semantic_class == 'ds_write'
    for name in names[1:]:
        assert rdna4[name].semantic_class == 'ds_atomic'


@pytest.mark.parametrize(
    ('profile', 'expected_kind'),
    [
        (Cdna4Profile(), 3),
        (Rdna4Profile(), 6),
        (Cdna5Profile(), 7),
    ],
)
def test_b8_transpose_aliases_share_profile_routing(profile, expected_kind) -> None:
    names = ('DS_LOAD_TR8_B64', 'DS_LOAD_B64_TR_B8', 'DS_LOAD_TR_B64')
    spec = SimpleNamespace(
        profile=profile,
        inst_encodings=[
            SimpleNamespace(
                enc_name='ENC_VDS',
                insts=[SimpleNamespace(name=name) for name in names],
            )
        ],
    )

    semantics = derive_all_semantics(spec)

    assert {semantics[name].transpose_kind for name in names} == {expected_kind}


@pytest.mark.parametrize(
    ('profile', 'expected_kind'),
    [
        (Cdna4Profile(), 6),
        (Rdna4Profile(), 6),
        (Cdna5Profile(), 6),
    ],
)
def test_global_b8_transpose_aliases_share_profile_routing(
    profile, expected_kind
) -> None:
    names = (
        'GLOBAL_LOAD_TR8_B64',
        'GLOBAL_LOAD_B64_TR_B8',
        'GLOBAL_LOAD_TR_B64',
    )
    spec = SimpleNamespace(
        profile=profile,
        inst_encodings=[
            SimpleNamespace(
                enc_name='ENC_VGLOBAL',
                insts=[SimpleNamespace(name=name) for name in names],
            )
        ],
    )

    semantics = derive_all_semantics(spec)

    assert profile.global_b8_transpose_kind == expected_kind
    assert {semantics[name].transpose_kind for name in names} == {expected_kind}


def test_global_b8_transpose_kind_can_be_overridden_by_profile() -> None:
    class GlobalB8Kind3Profile(Cdna5Profile):
        @property
        def global_b8_transpose_kind(self) -> int:
            return 3

    sem = derive_semantics('GLOBAL_LOAD_TR8_B64', 'ENC_VGLOBAL', GlobalB8Kind3Profile())

    assert sem is not None
    assert sem.transpose_kind == 3


def test_gfx1250_ds_b8_transpose_uses_profile_routing(
    gfx1250_generated_root: Path,
) -> None:
    vds = (gfx1250_generated_root / 'vds_exec.cpp').read_text()
    body = _generated_method_body(vds, 'DsLoadTr8B64Vds', 'DsLoadB96Vds')
    assert 'd->transpose = 7;' in body


@pytest.mark.parametrize(
    ('name', 'elem_size', 'scale', 'dst_dwords'),
    [
        ('DS_STOREXCHG_2ADDR_RTN_B32', 4, '4U', 1),
        ('DS_STOREXCHG_2ADDR_STRIDE64_RTN_B32', 4, '256U', 1),
        ('DS_STOREXCHG_2ADDR_RTN_B64', 8, '8U', 2),
        ('DS_STOREXCHG_2ADDR_STRIDE64_RTN_B64', 8, '512U', 2),
    ],
)
def test_gfx1250_dual_atomic_generator_covers_each_variant(
    name: str, elem_size: int, scale: str, dst_dwords: int
) -> None:
    codegen = object.__new__(CodeGenerator)
    codegen._vgpr_base_expr = lambda operand, **_kwargs: operand
    codegen._append_wait_counter_type = lambda lines, _semantic_class: lines.append(
        '  d->wait_counter_type = amdgpu::WaitCounterType::DSCNT;'
    )
    sem = InstructionSemantics(
        name, 'ds_atomic2', operation='swap', elem_size=elem_size, num_elems=1
    )
    body = codegen._gen_ds_atomic2([], [], sem)
    assert f'inst_.offset0) * {scale}' in body
    assert f'inst_.offset1) * {scale}' in body
    assert f'd->ds2_dst_reg_base = vdst + {dst_dwords};' in body
    assert 'd->ds2_active = true;' in body
    assert 'd->ds2_store_data' in body


def _generated_constructor_body(cpp: str, class_name: str) -> str:
    start = cpp.index(f'{class_name}::{class_name}(')
    end = cpp.index('\n\n', start)
    return cpp[start:end]


def _generated_bool_method_body(cpp: str, class_name: str, method: str) -> str:
    start = cpp.index(f'bool {class_name}::{method}() const')
    end = cpp.index('\n\n', start)
    return cpp[start:end]


def _generated_decode_body(cpp: str, class_name: str) -> str:
    # Generated files declare every decoder before defining it.  Select the
    # definition, including when several numbered model shards are combined.
    start = cpp.rindex(f'DecodeResult decode{class_name}(')
    end = cpp.index('\n}\n} // namespace detail', start)
    return cpp[start : end + 2]


def _generated_split_model_source(arch_root: Path, stem: str) -> str:
    """Read every numbered model shard for one logical generated source."""
    paths = sorted(arch_root.glob(f'{stem}.cpp')) + sorted(
        arch_root.glob(f'{stem}_[0-9]*.cpp')
    )
    assert paths
    return '\n'.join(path.read_text() for path in paths)


def _execution_source_path(path: Path, profile) -> Path:
    """Return the source containing execution bodies for a generated file."""
    if not profile.split_execution_sources:
        return path

    stem = path.stem
    if stem.rsplit('_', 1)[-1].isdigit():
        stem = stem.rsplit('_', 1)[0]
    if stem.startswith('vop3_'):
        stem = stem.replace('vop3_', 'vop3_exec_', 1)
    elif stem.startswith('vopc_'):
        stem = stem.replace('vopc_', 'vopc_exec_', 1)
    else:
        stem += '_exec'
    return path.with_name(f'{stem}.cpp')


def test_gfx1250_vop3_generated_definition_inventory_is_exact(
    amdgpu_generated_root: Path,
):
    arch_root = amdgpu_generated_root / _generated_dir_name('gfx1250')
    header = (arch_root / 'vop3.h').read_text()
    declared = set(re.findall(r'^class (\w+) : public Vop3', header, re.MULTILINE))
    vop3_sources = sorted(arch_root.glob('vop3*.cpp'))
    model_sources = '\n'.join(
        path.read_text()
        for path in vop3_sources
        if (path.stem == 'vop3' or path.stem.startswith('vop3_'))
        and not path.stem.startswith('vop3_exec')
    )
    execute_sources = '\n'.join(
        path.read_text() for path in vop3_sources if path.stem.startswith('vop3_exec')
    )
    constructors = set(
        re.findall(r'^(\w+)::\1\(const MachineInst \*inst', model_sources, re.MULTILINE)
    )
    execute_definitions = set(
        re.findall(r'^void (\w+)::execute_impl\(', execute_sources, re.MULTILINE)
    )

    assert declared == constructors, (
        f'missing constructors: {sorted(declared - constructors)}; '
        f'orphan constructors: {sorted(constructors - declared)}'
    )
    assert declared == execute_definitions, (
        f'missing execute definitions: {sorted(declared - execute_definitions)}; '
        f'orphan execute definitions: {sorted(execute_definitions - declared)}'
    )


def test_gfx1250_model_sources_do_not_include_execution_headers(
    amdgpu_root: Path,
    gfx1250_generated_root: Path,
):
    forbidden_includes = (
        '#include "rocjitsu/vm/',
        '#include "rocjitsu/isa/arch/amdgpu/cdna5/addr_calc.h"',
        '#include "rocjitsu/isa/arch/amdgpu/cdna5/mma_exec.h"',
        '#include "rocjitsu/isa/arch/amdgpu/shared/dpp_sdwa_ops.h"',
        '#include "rocjitsu/isa/arch/amdgpu/generated/shared/execute_shared.h"',
        '#include "rocjitsu/isa/arch/amdgpu/shared/simd_glue.h"',
        '#include "rocjitsu/isa/arch/amdgpu/shared/tensor_dma.h"',
    )
    model_sources = [
        path
        for path in sorted(gfx1250_generated_root.glob('*.cpp'))
        if '_exec' not in path.stem and path.name != 'addr_calc.cpp'
    ]

    assert model_sources
    for path in model_sources:
        source = path.read_text()
        for forbidden in forbidden_includes:
            assert forbidden not in source, f'{path.name} includes {forbidden}'

    for header_name in ('operand.h', 'encodings.h'):
        header = (gfx1250_generated_root / header_name).read_text()
        assert '#include "rocjitsu/vm/' not in header
        assert 'shared/dpp_sdwa_ops.h' not in header

    isa_header = (amdgpu_root / 'cdna5' / 'isa.h').read_text()
    assert '#include "rocjitsu/vm/' not in isa_header
    assert 'shared/dpp_sdwa_ops.h' not in isa_header


def test_generated_execution_helper_includes_are_scoped(
    amdgpu_generated_root: Path,
    gfx1250_generated_root: Path,
    execute_shared_path: Path,
):
    fp_include = '#include "rocjitsu/isa/arch/amdgpu/shared/fp_mode.h"'

    for path in (
        gfx1250_generated_root / 'smem_exec.cpp',
        amdgpu_generated_root / 'cdna4' / 'smem_exec.cpp',
    ):
        source = path.read_text()
        assert fp_include not in source, path

    assert fp_include in (gfx1250_generated_root / 'vop2_exec.cpp').read_text()

    shared = execute_shared_path.read_text()
    assert fp_include in shared


def test_gfx1250_model_include_graph_does_not_reach_vm(
    rocjitsu_source_root: Path,
    gfx1250_generated_root: Path,
):
    include_roots = (
        rocjitsu_source_root / 'lib' / 'rocjitsu' / 'src',
        rocjitsu_source_root / 'lib' / 'rocjitsu' / 'include',
        rocjitsu_source_root / 'lib' / 'util' / 'include',
        rocjitsu_source_root / 'lib' / 'simdojo' / 'include',
    )
    model_sources = [
        path
        for path in sorted(gfx1250_generated_root.glob('*.cpp'))
        if '_exec' not in path.stem and path.name != 'addr_calc.cpp'
    ]
    pending = [(path, [path]) for path in model_sources]
    visited = set()

    while pending:
        path, chain = pending.pop()
        if path in visited:
            continue
        visited.add(path)

        for line in path.read_text().splitlines():
            stripped = line.strip()
            if not stripped.startswith('#include "'):
                continue
            include = stripped.removeprefix('#include "').split('"', 1)[0]
            assert not include.startswith(('rocjitsu/vm/', 'simdojo/')), (
                'model include graph reaches execution header: '
                + ' -> '.join(item.name for item in chain)
                + f' -> {include}'
            )
            resolved = next(
                (
                    root / include
                    for root in include_roots
                    if (root / include).is_file()
                ),
                None,
            )
            if resolved is not None:
                pending.append((resolved, [*chain, resolved]))


def _parse_cdna_specs(*names: str):
    specs = []
    for name in names:
        spec = Parser(
            str(_mrisa_dir() / f'amdgpu_isa_{name}.xml'), CdnaProfile()
        ).parse()
        sem = derive_all_semantics(spec)
        specs.append((name, spec, sem))
    return specs


def test_parser_separates_logical_arch_directory_and_cpp_namespace():
    class IdentityOverrideRdna4Profile(Rdna4Profile):
        @property
        def generated_dir_name(self) -> str:
            return 'test_generated_rdna4'

        @property
        def cpp_namespace(self) -> str:
            return 'test_rdna4_namespace'

    isa_xml = _mrisa_dir() / 'amdgpu_isa_rdna4.xml'
    spec = Parser(str(isa_xml), IdentityOverrideRdna4Profile()).isa_spec

    assert spec.arch_name == 'rdna4'
    assert spec.generated_dir_name == 'test_generated_rdna4'
    assert spec.cpp_namespace == 'test_rdna4_namespace'


@pytest.mark.parametrize(
    'isa_name,profile_type',
    [
        ('cdna1', Cdna1Profile),
        ('cdna2', Cdna2Profile),
        ('cdna3', CdnaProfile),
        ('cdna4', Cdna4Profile),
        ('rdna1', Rdna1Profile),
        ('rdna2', Rdna2Profile),
        ('rdna3', Rdna3Profile),
        ('rdna3_5', Rdna3_5Profile),
        ('rdna4', Rdna4Profile),
        ('cdna5', Cdna5Profile),
    ],
)
def test_generated_scc_accesses_match_mrisa(isa_name, profile_type):
    isa_xml = _mrisa_dir() / f'amdgpu_isa_{isa_name}.xml'
    if not isa_xml.is_file():
        pytest.skip('MR ISA XML not available')

    parser = Parser(str(isa_xml), profile_type())
    spec = parser.parse()
    semantics = derive_all_semantics(spec)
    generator = CodeGenerator(spec, '', semantics)
    mrisa_scc_accesses = parser.implicit_operand_accesses('OPR_SSRC_SPECIAL_SCC')
    active_keys = {
        (inst.name, inst.enc_name) for enc in spec.inst_encodings for inst in enc.insts
    }
    assert set(mrisa_scc_accesses) == active_keys
    mismatches = []
    expected_exclusions = {
        'rdna4': {'S_ALLOC_VGPR', 'S_BARRIER_SIGNAL_ISFIRST'},
        'cdna5': {'S_ALLOC_VGPR'},
    }
    checked_exclusions = set()
    checked = 0
    seen = set()

    for enc in spec.inst_encodings:
        for inst in enc.insts:
            key = (inst.name, inst.enc_name)
            if key in seen or key not in mrisa_scc_accesses:
                continue
            seen.add(key)

            mrisa_reads_scc, mrisa_writes_scc = mrisa_scc_accesses[key]
            sem = semantics.instructions.get(inst.name)
            if inst.name in expected_exclusions.get(isa_name, set()):
                assert (
                    mrisa_writes_scc
                ), f'{isa_name}: stale SCC exclusion for {inst.name}'
                assert sem is not None and sem.semantic_class == 'true_nop', (
                    f'{isa_name}: implemented SCC exclusion must join the contract: '
                    f'{inst.name}'
                )
                checked_exclusions.add(inst.name)
                continue

            checked += 1
            if mrisa_reads_scc and sem is None:
                mismatches.append(
                    f'{inst.name}/{inst.enc_name}: MR ISA reads SCC but has no '
                    'derived semantics'
                )
                continue

            derived_writes_scc = sem is not None and sem.sets_scc not in (None, 'none')
            if derived_writes_scc != mrisa_writes_scc:
                mismatches.append(
                    f'{inst.name}/{inst.enc_name}: derived '
                    f'sets_scc={None if sem is None else sem.sets_scc!r}, '
                    f'MR ISA writes SCC={mrisa_writes_scc}'
                )
                continue

            if sem is not None:
                body = generator._gen_execute_body(inst, sem, enc.enc_name)
                body_reads_scc = 'read_scc()' in body
                body_writes_scc = 'write_scc(' in body
                if body_reads_scc != mrisa_reads_scc:
                    mismatches.append(
                        f'{inst.name}/{inst.enc_name}: generated body reads '
                        f'SCC={body_reads_scc}, MR ISA reads SCC={mrisa_reads_scc}'
                    )
                if body_writes_scc != mrisa_writes_scc:
                    mismatches.append(
                        f'{inst.name}/{inst.enc_name}: generated body writes '
                        f'SCC={body_writes_scc}, MR ISA writes SCC={mrisa_writes_scc}'
                    )
                if (
                    sem.semantic_class == 'scalar_addk'
                    and 'signed_add_overflows' not in body
                ):
                    mismatches.append(
                        f'{inst.name}/{inst.enc_name}: ADDK does not use signed overflow'
                    )

    assert checked > 0, f'{isa_name}: no instruction SCC contracts checked'
    assert checked_exclusions == expected_exclusions.get(isa_name, set())
    assert not mismatches, f'{isa_name}:\n' + '\n'.join(mismatches)


def test_implicit_operand_accesses_covers_filters_merging_and_compat_insts():
    parser = object.__new__(Parser)
    parser.insts_node = elem_tree.fromstring('''
        <Instructions>
          <Instruction>
            <InstructionName>READ</InstructionName>
            <InstructionEncodings>
              <InstructionEncoding>
                <EncodingName>ENC_READ</EncodingName>
                <EncodingCondition>default</EncodingCondition>
                <Operands>
                  <Operand Input="true" Output="false" IsImplicit="true">
                    <OperandType>OPR_SCC</OperandType>
                  </Operand>
                </Operands>
              </InstructionEncoding>
            </InstructionEncodings>
          </Instruction>
          <Instruction>
            <InstructionName>WRITE</InstructionName>
            <InstructionEncodings>
              <InstructionEncoding>
                <EncodingName>ENC_WRITE</EncodingName>
                <EncodingCondition>default</EncodingCondition>
                <Operands>
                  <Operand Input="false" Output="true" IsImplicit="true">
                    <OperandType>OPR_SCC</OperandType>
                  </Operand>
                </Operands>
              </InstructionEncoding>
            </InstructionEncodings>
          </Instruction>
          <Instruction>
            <InstructionName>MERGED</InstructionName>
            <InstructionEncodings>
              <InstructionEncoding>
                <EncodingName>ENC_DUP</EncodingName>
                <EncodingCondition>default</EncodingCondition>
                <Operands>
                  <Operand Input="true" Output="false" IsImplicit="true">
                    <OperandType>OPR_SCC</OperandType>
                  </Operand>
                </Operands>
              </InstructionEncoding>
              <InstructionEncoding>
                <EncodingName>ENC_DUP</EncodingName>
                <EncodingCondition>default</EncodingCondition>
                <Operands>
                  <Operand Input="false" Output="true" IsImplicit="true">
                    <OperandType>OPR_SCC</OperandType>
                  </Operand>
                </Operands>
              </InstructionEncoding>
            </InstructionEncodings>
          </Instruction>
          <Instruction>
            <InstructionName>EXPLICIT</InstructionName>
            <InstructionEncodings>
              <InstructionEncoding>
                <EncodingName>ENC_EXPLICIT</EncodingName>
                <EncodingCondition>default</EncodingCondition>
                <Operands>
                  <Operand Input="true" Output="true" IsImplicit="false">
                    <OperandType>OPR_SCC</OperandType>
                  </Operand>
                </Operands>
              </InstructionEncoding>
            </InstructionEncodings>
          </Instruction>
          <Instruction>
            <InstructionName>FILTERED</InstructionName>
            <InstructionEncodings>
              <InstructionEncoding>
                <EncodingName>ENC_SKIP</EncodingName>
                <EncodingCondition>default</EncodingCondition>
                <Operands />
              </InstructionEncoding>
              <InstructionEncoding>
                <EncodingName>ENC_COND</EncodingName>
                <EncodingCondition>skip_me</EncodingCondition>
                <Operands />
              </InstructionEncoding>
            </InstructionEncodings>
          </Instruction>
        </Instructions>
        ''')
    parser.profile = SimpleNamespace(
        skip_encodings={'ENC_SKIP'},
        skip_inst_encoding=lambda _name, condition: condition == 'skip_me',
    )
    active = [
        ('READ', 'ENC_READ'),
        ('WRITE', 'ENC_WRITE'),
        ('MERGED', 'ENC_DUP'),
        ('EXPLICIT', 'ENC_EXPLICIT'),
        ('INJECTED', 'ENC_INJECTED'),
    ]
    parser.isa_spec = SimpleNamespace(
        inst_encodings=[
            SimpleNamespace(insts=[SimpleNamespace(name=name, enc_name=encoding)])
            for name, encoding in active
        ]
    )

    assert parser.implicit_operand_accesses('OPR_SCC') == {
        ('READ', 'ENC_READ'): (True, False),
        ('WRITE', 'ENC_WRITE'): (False, True),
        ('MERGED', 'ENC_DUP'): (True, True),
        ('EXPLICIT', 'ENC_EXPLICIT'): (False, False),
        ('INJECTED', 'ENC_INJECTED'): (False, False),
    }


def _execute_impl_body(source: str, signature: str, next_ctor: str) -> str:
    start = source.index(signature)
    end = source.index(next_ctor, start)
    return source[start:end]


def test_simm64_literals_require_operand_type():
    codegen = object.__new__(CodeGenerator)
    codegen.isa_spec = SimpleNamespace(operand_types=['OPR_SIMM32'])
    assert not codegen._supports_simm64_literal_operands()

    codegen.isa_spec = SimpleNamespace(operand_types=['OPR_SIMM32', 'OPR_SIMM64'])
    assert codegen._supports_simm64_literal_operands()


def test_generated_literal_fixups_separate_declared_and_dynamic_true16(
    amdgpu_generated_root: Path,
):
    cdna3_vop2 = (amdgpu_generated_root / 'cdna3' / 'vop2.cpp').read_text()
    madak_start = cdna3_vop2.index('VMadakF16Vop2::VMadakF16Vop2')
    madak_end = cdna3_vop2.index('void VMadakF16Vop2::implicit_uses', madak_start)
    madak_ctor = cdna3_vop2[madak_start:madak_end]
    assert 'Vop2InstLiteralMachineInst *>(inst)->simm32 & 0xFFFFu' in madak_ctor

    rdna4_vop3 = (amdgpu_generated_root / 'rdna4' / 'vop3.cpp').read_text()
    and_start = rdna4_vop3.index('VAndB16Vop3::VAndB16Vop3')
    and_end = rdna4_vop3.index('void VAndB16Vop3::implicit_uses', and_start)
    and_ctor = rdna4_vop3[and_start:and_end]
    assert (
        'static_cast<int>(reinterpret_cast<const Vop3InstLiteralMachineInst *>(inst)->simm32)'
        in and_ctor
    )
    assert '((amdgpu::vop3_opsel(inst_) >> 0) & 1u) * 16u' in and_ctor
    assert '((amdgpu::vop3_opsel(inst_) >> 1) & 1u) * 16u' in and_ctor

    rdna4_operand = (amdgpu_generated_root / 'rdna4' / 'operand.cpp').read_text()
    assert 'if (has_literal16_display_)' in rdna4_operand
    assert 'static_cast<uint32_t>(encoding_value_)' in rdna4_operand


def test_vop_dpp8_support_is_detected_from_machine_inst_structs():
    codegen = object.__new__(CodeGenerator)
    codegen.isa_spec = SimpleNamespace(
        inst_encodings=[
            SimpleNamespace(fmt_enc_name='Vop1Inst'),
            SimpleNamespace(fmt_enc_name='Vop2VopDpp16'),
        ]
    )
    assert not codegen._supports_vop_dpp8()

    codegen.isa_spec = SimpleNamespace(
        inst_encodings=[
            SimpleNamespace(fmt_enc_name='Vop1Inst'),
            SimpleNamespace(fmt_enc_name='Vop1VopDpp8'),
        ]
    )
    assert codegen._supports_vop_dpp8()


def test_machine_inst_struct_has_field_detects_dpp16_fi():
    codegen = object.__new__(CodeGenerator)
    codegen.isa_spec = SimpleNamespace(
        inst_encodings=[
            SimpleNamespace(
                fmt_enc_name='Vop1VopDpp8',
                ucode_fields=[
                    SimpleNamespace(name='src0'),
                    SimpleNamespace(name='lane_sel_0'),
                ],
            ),
            SimpleNamespace(
                fmt_enc_name='Vop1VopDpp16',
                ucode_fields=[
                    SimpleNamespace(name='src0'),
                    SimpleNamespace(name='fi'),
                ],
            ),
        ]
    )

    assert not codegen._machine_inst_struct_has_field('Vop1VopDpp8MachineInst', 'fi')
    assert codegen._machine_inst_struct_has_field('Vop1VopDpp16MachineInst', 'fi')


def test_vop3_sdst_dpp_support_is_detected_from_machine_inst_structs():
    codegen = object.__new__(CodeGenerator)
    codegen.isa_spec = SimpleNamespace(
        inst_encodings=[
            SimpleNamespace(fmt_enc_name='Vop1VopDpp16', enc_name='VOP1_VOP_DPP16'),
            SimpleNamespace(
                fmt_enc_name='Vop3SdstEncVopDpp16', enc_name='VOP3_SDST_ENC_VOP_DPP16'
            ),
            SimpleNamespace(
                fmt_enc_name='Vop3SdstEncVopDpp8', enc_name='VOP3_SDST_ENC_VOP_DPP8'
            ),
        ]
    )

    assert codegen._vop_dpp_struct_names('VOP3_SDST_ENC') == (
        'Vop3SdstEncVopDpp16MachineInst',
        'Vop3SdstEncVopDpp8MachineInst',
    )
    assert codegen._supports_vop_dpp_encoding('VOP3_SDST_ENC')


def test_vopc_dpp16_uses_vopc_layout_when_available():
    codegen = object.__new__(CodeGenerator)
    codegen.isa_spec = SimpleNamespace(
        inst_encodings=[
            SimpleNamespace(fmt_enc_name='Vop1VopDpp16', enc_name='VOP1_VOP_DPP16'),
            SimpleNamespace(fmt_enc_name='VopcVopDpp16', enc_name='VOPC_VOP_DPP16'),
            SimpleNamespace(fmt_enc_name='VopcVopDpp8', enc_name='VOPC_VOP_DPP8'),
        ]
    )

    assert codegen._vop_dpp_struct_names('ENC_VOPC') == (
        'VopcVopDpp16MachineInst',
        'VopcVopDpp8MachineInst',
    )


def test_vopc_dpp_falls_back_to_vop1_layout_for_legacy_cdna():
    codegen = object.__new__(CodeGenerator)
    codegen.isa_spec = SimpleNamespace(
        inst_encodings=[
            SimpleNamespace(fmt_enc_name='Vop1VopDpp', enc_name='VOP1_VOP_DPP'),
        ]
    )

    assert codegen._vop_dpp_struct_names('ENC_VOPC') == (
        'Vop1VopDppMachineInst',
        None,
    )


def test_vopc_dpp_requires_vopc_dpp16_on_rdna():
    codegen = object.__new__(CodeGenerator)
    codegen.isa_spec = SimpleNamespace(
        inst_encodings=[
            SimpleNamespace(enc_name='VOP1_VOP_DPP16'),
            SimpleNamespace(enc_name='VOP2_VOP_DPP16'),
        ]
    )
    assert not codegen._supports_dpp_for_encoding('ENC_VOPC')
    assert codegen._supports_dpp_for_encoding('ENC_VOP1')

    codegen.isa_spec = SimpleNamespace(
        inst_encodings=[
            SimpleNamespace(enc_name='VOP1_VOP_DPP16'),
            SimpleNamespace(enc_name='VOP2_VOP_DPP16'),
            SimpleNamespace(enc_name='VOPC_VOP_DPP16'),
        ]
    )
    assert codegen._supports_dpp_for_encoding('ENC_VOPC')

    codegen.isa_spec = SimpleNamespace(
        inst_encodings=[
            SimpleNamespace(enc_name='VOP1_VOP_DPP'),
            SimpleNamespace(enc_name='VOP2_VOP_DPP'),
        ]
    )
    assert codegen._supports_dpp_for_encoding('ENC_VOPC')


def test_rdna4_parser_injects_s_waitcnt_compat():
    spec = Parser(
        str(_mrisa_dir() / 'amdgpu_isa_rdna4.xml'),
        Rdna4Profile(),
    ).parse()

    sopp = spec.encoding_map['ENC_SOPP']
    waitcnt = next(
        inst for inst in sopp.insts if inst.name == 'S_WAITCNT' and inst.opcode == 9
    )
    assert waitcnt.available_encodings == frozenset({'ENC_SOPP'})
    assert 'OPR_WAITCNT' in spec.operand_types

    dt_ptr = sopp.primary_dt_ptrs[9]
    dte = spec.primary_decode_table[dt_ptr]
    assert dte.sub_decode_funcs[9] == 'decodeSWaitcntSopp'


def test_gfx1250_generated_inventory_omits_legacy_s_waitcnt(
    gfx1250_generated_root: Path,
):
    opcodes = (gfx1250_generated_root / 'opcodes.h').read_text()
    decoder = (gfx1250_generated_root / 'decoder.cpp').read_text()

    assert 'kSWaitcntSopp' not in opcodes
    table_start = decoder.index('DecoderImpl::sub_decode_sopp = {')
    table_end = decoder.index('\n};', table_start)
    decode_targets = re.findall(
        r'&(detail|DecoderImpl)::([A-Za-z0-9_]+)',
        decoder[table_start:table_end],
    )
    assert decode_targets[9] == ('DecoderImpl', 'decodeInvalid')


def _fake_sopp_parser(arch_name: str, *, sub_decode_funcs: list[str | None] | None):
    parser = object.__new__(Parser)
    parser.profile = {
        'cdna5': Cdna5Profile(),
        'rdna3': Rdna3Profile(),
        'rdna4': Rdna4Profile(),
    }[arch_name]
    sopp = SimpleNamespace(
        insts=[
            Instruction('S_WAIT_ALU', 'ENC_SOPP', 8, []),
            Instruction('S_WAIT_IDLE', 'ENC_SOPP', 10, []),
        ],
        primary_dt_ptrs=[-1] * 10,
    )
    sopp.primary_dt_ptrs[9] = 0
    dte = SimpleNamespace(
        sub_decode_funcs=sub_decode_funcs,
        decode_func=None,
        inst_name=None,
    )
    encoding_map = {'ENC_SOPP': sopp}
    if arch_name == 'cdna5':
        encoding_map['ENC_VOP1'] = SimpleNamespace(
            insts=[Instruction('V_PERMLANE64_B32', 'ENC_VOP1', 103, [])],
            primary_dt_ptrs=[0] * 104,
        )
    parser.isa_spec = SimpleNamespace(
        arch_name=arch_name,
        encoding_map=encoding_map,
        primary_decode_table=[dte],
        operand_types=[],
    )
    return parser, sopp, dte


def _s_waitcnt_state(sopp, dte):
    return (
        list(sopp.insts),
        list(sopp.primary_dt_ptrs),
        None if dte.sub_decode_funcs is None else list(dte.sub_decode_funcs),
        dte.decode_func,
        dte.inst_name,
    )


def test_rdna4_parser_injects_s_waitcnt_compat_once_in_opcode_order():
    parser, sopp, dte = _fake_sopp_parser('rdna4', sub_decode_funcs=[None] * 16)

    parser._inject_compat_insts()
    parser._inject_compat_insts()

    assert [(inst.name, inst.opcode) for inst in sopp.insts] == [
        ('S_WAIT_ALU', 8),
        ('S_WAITCNT', 9),
        ('S_WAIT_IDLE', 10),
    ]
    assert (
        sum(inst.name == 'S_WAITCNT' and inst.opcode == 9 for inst in sopp.insts) == 1
    )
    assert dte.sub_decode_funcs[9] == 'decodeSWaitcntSopp'


def test_rdna4_parser_selects_its_compatibility_slot_by_instruction_name():
    parser, sopp, dte = _fake_sopp_parser('rdna4', sub_decode_funcs=[None] * 16)
    parser.profile = SimpleNamespace(
        compatibility_instruction_slots={
            ('ENC_OTHER', 4): 'OTHER_COMPAT',
            ('ENC_SOPP', 9): 'S_WAITCNT',
        }
    )

    parser._inject_s_waitcnt_compat()

    assert any(inst.name == 'S_WAITCNT' and inst.opcode == 9 for inst in sopp.insts)
    assert dte.sub_decode_funcs[9] == 'decodeSWaitcntSopp'


def test_gfx1250_parser_does_not_inject_legacy_s_waitcnt():
    parser, sopp, dte = _fake_sopp_parser('cdna5', sub_decode_funcs=[None] * 16)

    parser._inject_s_waitcnt_compat()

    assert [(inst.name, inst.opcode) for inst in sopp.insts] == [
        ('S_WAIT_ALU', 8),
        ('S_WAIT_IDLE', 10),
    ]
    assert dte.sub_decode_funcs[9] is None


def test_gfx1250_parser_injects_permlane64_compat_once():
    parser = object.__new__(Parser)
    parser.profile = Cdna5Profile()
    vop1 = SimpleNamespace(
        insts=[
            Instruction('V_SWAP_B16', 'ENC_VOP1', 102, []),
            Instruction('V_NOT_B16', 'ENC_VOP1', 104, []),
        ],
        primary_dt_ptrs=[-1] * 128,
    )
    vop1.primary_dt_ptrs[102] = 0
    vop1.primary_dt_ptrs[104] = 0
    dte = SimpleNamespace(sub_decode_funcs=['decodeInvalid'] * 128, decode_func=None)
    parser.isa_spec = SimpleNamespace(
        arch_name='cdna5',
        encoding_map={'ENC_VOP1': vop1},
        primary_decode_table=[dte],
    )

    parser._inject_cdna5_permlane64_compat()
    parser._inject_cdna5_permlane64_compat()

    assert [(inst.name, inst.opcode) for inst in vop1.insts] == [
        ('V_SWAP_B16', 102),
        ('V_PERMLANE64_B32', 103),
        ('V_NOT_B16', 104),
    ]
    permlane = vop1.insts[1]
    assert permlane.available_encodings == frozenset({'ENC_VOP1'})
    assert [operand.operand_type for operand in permlane.operands] == [
        'OPR_VGPR',
        'OPR_SRC_VGPR',
    ]
    assert dte.sub_decode_funcs[103] == 'decodeVPermlane64B32Vop1'
    assert vop1.primary_dt_ptrs[103] == 0


def test_gfx1250_parser_permlane64_requires_an_unambiguous_adjacent_route():
    parser = object.__new__(Parser)
    parser.profile = Cdna5Profile()
    vop1 = SimpleNamespace(
        insts=[Instruction('V_SWAP_B16', 'ENC_VOP1', 102, [])],
        primary_dt_ptrs=[-1] * 128,
    )
    # A unique route elsewhere in the table is not evidence for opcode 103.
    vop1.primary_dt_ptrs[12] = 0
    vop1.primary_dt_ptrs[102] = 1
    vop1.primary_dt_ptrs[104] = 2
    parser.isa_spec = SimpleNamespace(
        arch_name='cdna5',
        encoding_map={'ENC_VOP1': vop1},
        primary_decode_table=[
            SimpleNamespace(sub_decode_funcs=[None] * 128, decode_func=None)
            for _ in range(3)
        ],
    )

    with pytest.raises(ValueError, match='exactly one adjacent ENC_VOP1 decode route'):
        parser._inject_cdna5_permlane64_compat()

    assert not any(inst.name == 'V_PERMLANE64_B32' for inst in vop1.insts)
    assert vop1.primary_dt_ptrs[103] == -1


def test_gfx1250_parser_permlane64_rejects_opcode_collision_before_mutation():
    parser = object.__new__(Parser)
    parser.profile = Cdna5Profile()
    collision = Instruction('V_OTHER_B32', 'ENC_VOP1', 103, [])
    vop1 = SimpleNamespace(
        insts=[collision],
        primary_dt_ptrs=[0] * 128,
    )
    parser.isa_spec = SimpleNamespace(
        arch_name='cdna5',
        encoding_map={'ENC_VOP1': vop1},
        primary_decode_table=[
            SimpleNamespace(sub_decode_funcs=[None] * 128, decode_func=None)
        ],
    )

    with pytest.raises(
        ValueError, match='opcode 103 is already occupied by V_OTHER_B32'
    ):
        parser._inject_cdna5_permlane64_compat()

    assert vop1.insts == [collision]
    assert vop1.primary_dt_ptrs[103] == 0


def test_gfx1250_parser_permlane64_rejects_invalid_decode_table_index_before_mutation():
    parser = object.__new__(Parser)
    parser.profile = Cdna5Profile()
    vop1 = SimpleNamespace(
        insts=[Instruction('V_SWAP_B16', 'ENC_VOP1', 102, [])],
        primary_dt_ptrs=[-1] * 128,
    )
    vop1.primary_dt_ptrs[102] = 4
    vop1.primary_dt_ptrs[104] = 4
    parser.isa_spec = SimpleNamespace(
        arch_name='cdna5',
        encoding_map={'ENC_VOP1': vop1},
        primary_decode_table=[
            SimpleNamespace(sub_decode_funcs=[None] * 128, decode_func=None)
        ],
    )

    with pytest.raises(ValueError, match='invalid primary decode-table index 4'):
        parser._inject_cdna5_permlane64_compat()

    assert not any(inst.opcode == 103 for inst in vop1.insts)
    assert vop1.primary_dt_ptrs[103] == -1


@pytest.mark.parametrize(
    ('decode_entry', 'message'),
    [
        (
            SimpleNamespace(
                sub_decode_funcs=[None] * 103,
                decode_func=None,
                inst_name=None,
            ),
            'outside the selected subdecode table',
        ),
        (
            SimpleNamespace(
                sub_decode_funcs=[None] * 103 + ['decodeOther'],
                decode_func=None,
                inst_name=None,
            ),
            'subdecode slot is already occupied by decodeOther',
        ),
        (
            SimpleNamespace(
                sub_decode_funcs=None,
                decode_func='decodeOther',
                inst_name='V_OTHER_B32',
            ),
            'terminal decode entry is already occupied',
        ),
    ],
)
def test_gfx1250_parser_permlane64_rejects_terminal_conflicts_before_mutation(
    decode_entry, message
):
    parser = object.__new__(Parser)
    parser.profile = Cdna5Profile()
    neighbor = Instruction('V_SWAP_B16', 'ENC_VOP1', 102, [])
    vop1 = SimpleNamespace(
        insts=[neighbor],
        primary_dt_ptrs=[-1] * 128,
    )
    vop1.primary_dt_ptrs[102] = 0
    parser.isa_spec = SimpleNamespace(
        arch_name='cdna5',
        encoding_map={'ENC_VOP1': vop1},
        primary_decode_table=[decode_entry],
    )

    with pytest.raises(ValueError, match=message):
        parser._inject_cdna5_permlane64_compat()

    assert vop1.insts == [neighbor]
    assert vop1.primary_dt_ptrs[103] == -1


@pytest.mark.parametrize(
    ('encoding_map', 'message'),
    [
        ({}, 'requires ENC_VOP1'),
        (
            {'ENC_VOP1': SimpleNamespace(insts=[], primary_dt_ptrs=None)},
            'requires an ENC_VOP1 primary decode route table',
        ),
        (
            {'ENC_VOP1': SimpleNamespace(insts=[], primary_dt_ptrs=[-1] * 103)},
            'does not contain V_PERMLANE64_B32 opcode 103',
        ),
    ],
)
def test_gfx1250_parser_permlane64_rejects_missing_route_invariants(
    encoding_map, message
):
    parser = object.__new__(Parser)
    parser.profile = Cdna5Profile()
    parser.isa_spec = SimpleNamespace(
        arch_name='cdna5',
        encoding_map=encoding_map,
        primary_decode_table=[],
    )

    with pytest.raises(ValueError, match=message):
        parser._inject_cdna5_permlane64_compat()


def test_s_waitcnt_compat_injection_can_patch_direct_decode_entry():
    parser, sopp, dte = _fake_sopp_parser('rdna4', sub_decode_funcs=None)

    parser._inject_compat_insts()

    assert any(inst.name == 'S_WAITCNT' and inst.opcode == 9 for inst in sopp.insts)
    assert dte.decode_func == 'decodeSWaitcntSopp'
    assert dte.inst_name == 'SWaitcntSopp'


def test_rdna4_s_waitcnt_rejects_opcode_collision_before_mutation():
    parser, sopp, dte = _fake_sopp_parser('rdna4', sub_decode_funcs=[None] * 16)
    sopp.insts.insert(1, Instruction('S_OTHER', 'ENC_SOPP', 9, []))
    before = _s_waitcnt_state(sopp, dte)

    with pytest.raises(ValueError, match='opcode 9 is already occupied by S_OTHER'):
        parser._inject_s_waitcnt_compat()

    assert _s_waitcnt_state(sopp, dte) == before


@pytest.mark.parametrize(
    ('encoding_map', 'message'),
    [
        ({}, 'requires ENC_SOPP'),
        (
            {'ENC_SOPP': SimpleNamespace(insts=[], primary_dt_ptrs=None)},
            'requires an ENC_SOPP primary decode route table',
        ),
        (
            {'ENC_SOPP': SimpleNamespace(insts=[], primary_dt_ptrs=[-1] * 9)},
            'does not contain S_WAITCNT opcode 9',
        ),
    ],
)
def test_rdna4_s_waitcnt_rejects_missing_route_invariants(encoding_map, message):
    parser = object.__new__(Parser)
    parser.profile = Rdna4Profile()
    parser.isa_spec = SimpleNamespace(
        arch_name='rdna4',
        encoding_map=encoding_map,
        primary_decode_table=[],
    )

    with pytest.raises(ValueError, match=message):
        parser._inject_s_waitcnt_compat()


def test_rdna4_s_waitcnt_rejects_ambiguous_primary_routes_before_mutation():
    parser, sopp, first_dte = _fake_sopp_parser('rdna4', sub_decode_funcs=[None] * 16)
    first_dte.enc = sopp
    second_dte = SimpleNamespace(
        enc=sopp,
        sub_decode_funcs=[None] * 16,
        decode_func=None,
        inst_name=None,
    )
    parser.isa_spec.primary_decode_table.append(second_dte)
    sopp.primary_dt_ptrs[0] = 0
    sopp.primary_dt_ptrs[1] = 1
    sopp.primary_dt_ptrs[9] = -1
    before = _s_waitcnt_state(sopp, first_dte)

    with pytest.raises(
        ValueError,
        match='requires exactly one unique ENC_SOPP primary decode route',
    ):
        parser._inject_s_waitcnt_compat()

    assert _s_waitcnt_state(sopp, first_dte) == before


@pytest.mark.parametrize(
    ('dt_ptr', 'message'),
    [
        (-1, 'has no ENC_SOPP primary decode route'),
        (4, 'invalid primary decode-table index 4'),
    ],
)
def test_rdna4_s_waitcnt_rejects_invalid_decode_route_before_mutation(dt_ptr, message):
    parser, sopp, dte = _fake_sopp_parser('rdna4', sub_decode_funcs=[None] * 16)
    sopp.primary_dt_ptrs[9] = dt_ptr
    before = _s_waitcnt_state(sopp, dte)

    with pytest.raises(ValueError, match=message):
        parser._inject_s_waitcnt_compat()

    assert _s_waitcnt_state(sopp, dte) == before


@pytest.mark.parametrize(
    ('decode_entry', 'message'),
    [
        (
            SimpleNamespace(
                sub_decode_funcs=[None] * 9,
                decode_func=None,
                inst_name=None,
            ),
            'outside the selected subdecode table',
        ),
        (
            SimpleNamespace(
                sub_decode_funcs=[None] * 9 + ['decodeOther'],
                decode_func=None,
                inst_name=None,
            ),
            'subdecode slot is already occupied by decodeOther',
        ),
        (
            SimpleNamespace(
                sub_decode_funcs=None,
                decode_func='decodeOther',
                inst_name='S_OTHER',
            ),
            'terminal decode entry is already occupied',
        ),
    ],
)
def test_rdna4_s_waitcnt_rejects_terminal_conflicts_before_mutation(
    decode_entry, message
):
    parser, sopp, _ = _fake_sopp_parser('rdna4', sub_decode_funcs=[None] * 16)
    parser.isa_spec.primary_decode_table[0] = decode_entry
    before = _s_waitcnt_state(sopp, decode_entry)

    with pytest.raises(ValueError, match=message):
        parser._inject_s_waitcnt_compat()

    assert _s_waitcnt_state(sopp, decode_entry) == before


def test_s_waitcnt_compat_injection_skips_untargeted_arch():
    parser, sopp, dte = _fake_sopp_parser('rdna3', sub_decode_funcs=[None] * 16)

    parser._inject_compat_insts()

    assert [(inst.name, inst.opcode) for inst in sopp.insts] == [
        ('S_WAIT_ALU', 8),
        ('S_WAIT_IDLE', 10),
    ]
    assert dte.sub_decode_funcs[9] is None


def test_rdna4_s_waitcnt_compat_uses_gfx11_layout():
    codegen = object.__new__(CodeGenerator)
    codegen.isa_spec = SimpleNamespace(
        arch_name='rdna4',
        profile=Rdna4Profile(),
        inst_encodings=[],
        encoding_map={},
    )
    inst = Instruction(
        'S_WAITCNT',
        'ENC_SOPP',
        9,
        [Operand('simm16', 16, 'OPR_WAITCNT', True, False, False, True, 1)],
    )
    sem = InstructionSemantics('S_WAITCNT', 'waitcnt')

    body = codegen._gen_execute_body(inst, sem, 'ENC_SOPP')

    assert 'uint8_t exp = imm & 0x7;' in body
    assert 'uint8_t lgkm = (imm >> 4) & 0x3F;' in body
    assert 'uint8_t vm = (imm >> 10) & 0x3F;' in body


def test_s_trap_executes_as_nop_without_a_trap_handler():
    codegen = object.__new__(CodeGenerator)
    codegen.isa_spec = SimpleNamespace(
        arch_name='rdna4',
        profile=Rdna4Profile(),
        inst_encodings=[],
        encoding_map={},
    )
    inst = Instruction(
        'S_TRAP',
        'ENC_SOPP',
        18,
        [Operand('simm16', 16, 'OPR_SIMM16', True, False, False, True, 1)],
    )
    sem = InstructionSemantics('S_TRAP', 'trap')

    body = codegen._gen_execute_body(inst, sem, 'ENC_SOPP')

    assert body == '  (void)wf;'


def test_rdna4_s_waitcnt_compat_formats_with_gfx11_layout():
    decode = Rdna4Profile().waitcnt_decode

    assert 'uint32_t expcnt = encoding_value_ & 0x7;' in decode
    assert 'uint32_t lgkmcnt = (encoding_value_ >> 4) & 0x3F;' in decode
    assert 'uint32_t vmcnt = (encoding_value_ >> 10) & 0x3F;' in decode


def test_readlane_family_uses_source_vgpr_operand_type():
    codegen = object.__new__(CodeGenerator)
    codegen.isa_spec = SimpleNamespace(operand_types=['OPR_VGPR', 'OPR_SRC_VGPR'])
    sem = InstructionSemantics('V_READFIRSTLANE_B32', 'vector_readfirstlane')
    src0 = Operand('src0', 32, 'OPR_VGPR', True, False, False, False, 0)
    vdst = Operand('vdst', 32, 'OPR_VGPR', False, True, False, False, 0)

    assert codegen._constructor_operand_type(sem, src0) == 'OPR_SRC_VGPR'
    assert codegen._constructor_operand_type(sem, vdst) == 'OPR_VGPR'


def test_pk_mov_b32_keeps_declared_scalar_or_vector_source_types():
    codegen = object.__new__(CodeGenerator)
    codegen.isa_spec = SimpleNamespace(
        operand_types=[
            'OPR_SRC_NOLIT',
            'OPR_SRC_SIMPLE',
            'OPR_SRC_VGPR_OR_ACCVGPR',
        ]
    )
    sem = derive_semantics('V_PK_MOV_B32', 'ENC_VOP3P')
    assert sem is not None
    src0 = Operand('src0', 64, 'OPR_SRC_NOLIT', True, False, False, True, 1)
    src1 = Operand('src1', 64, 'OPR_SRC_SIMPLE', True, False, False, True, 2)

    assert codegen._constructor_operand_type(sem, src0) == 'OPR_SRC_NOLIT'
    assert codegen._constructor_operand_type(sem, src1) == 'OPR_SRC_SIMPLE'


def test_readlane_family_decodes_lane_selector_as_scalar_value():
    codegen = object.__new__(CodeGenerator)
    codegen.isa_spec = SimpleNamespace(
        arch_name='rdna3',
        profile=Rdna3Profile(),
        inst_encodings=[],
        encoding_map={},
    )
    operands = [
        Operand('vdst', 32, 'OPR_SREG', False, True, False, False, 0),
        Operand('src0', 32, 'OPR_SRC_VGPR', True, False, False, False, 1),
        Operand('src1', 32, 'OPR_SSRC_LANESEL', True, False, False, False, 2),
    ]
    inst = Instruction('V_READLANE_B32', 'ENC_VOP3', 0, operands)
    sem = InstructionSemantics('V_READLANE_B32', 'vector_readlane')

    body = codegen._gen_execute_body(inst, sem, 'ENC_VOP3')

    assert 'uint32_t lane = amdgpu::RegisterAccess(wf).read_scalar(src1);' in body
    assert 'lane &= wf.kernel_wave_size() - 1;' not in body
    assert 'read_scalar_selected_lane(src0, lane)' in body
    assert 'src1.encoding_value_' not in body

    operands = [
        Operand('vdst', 32, 'OPR_VGPR', False, True, False, False, 0),
        Operand('src0', 32, 'OPR_SSRC', True, False, False, False, 1),
        Operand('src1', 32, 'OPR_SSRC_LANESEL', True, False, False, False, 2),
    ]
    inst = Instruction('V_WRITELANE_B32', 'ENC_VOP3', 0, operands)
    sem = InstructionSemantics('V_WRITELANE_B32', 'vector_writelane')

    body = codegen._gen_execute_body(inst, sem, 'ENC_VOP3')

    assert 'uint32_t lane = amdgpu::RegisterAccess(wf).read_scalar(src1);' in body
    assert 'lane &= wf.kernel_wave_size() - 1;' not in body
    assert 'write_scalar_selected_lane(vdst, lane, val)' in body
    assert 'src1.encoding_value_' not in body


def test_movrel_uses_two_arg_resolver_without_vgpr_msb_indexing():
    src_body = gen_vector_movrel(
        ['vdst'], ['src0'], 'src', uses_vgpr_msb_indexing=False
    )
    dst_body = gen_vector_movrel(
        ['vdst'], ['src0'], 'dst', uses_vgpr_msb_indexing=False
    )

    assert 'Isa::resolved_vgpr_offset(src0.opr_type_, src0.encoding_value_)' in src_body
    assert 'Isa::resolved_vgpr_offset(wf, src0.opr_type_' not in src_body
    assert 'Isa::resolved_vgpr_offset(vdst.opr_type_, vdst.encoding_value_)' in dst_body
    assert 'Isa::resolved_vgpr_offset(wf, vdst.opr_type_' not in dst_body


def test_movrel_uses_wavefront_resolver_with_vgpr_msb_indexing():
    body = gen_vector_movrel(['vdst'], ['src0'], 'src', uses_vgpr_msb_indexing=True)
    assert 'Isa::resolved_vgpr_offset(wf, src0.opr_type_' in body
    assert 'src0.vgpr_msb_role()' in body


def test_div_scale_writes_explicit_sdst_mask():
    body = gen_vector_div_scale(
        ['vdst', 'sdst'], ['src0', 'src1', 'src2'], 'f32', is_vop3=True
    )

    assert 'amdgpu::write_wave_mask_scalar(sdst, wf, vcc);' in body
    assert 'wf.set_vcc' not in body

    callback_body = gen_vector_div_scale(
        ['vdst', 'sdst'],
        ['src0', 'src1', 'src2'],
        'f32',
        is_vop3=True,
        result_writer='commit_result',
    )
    assert 'commit_result(vcc);' in callback_body
    assert 'write_wave_mask_scalar' not in callback_body


def test_vector_cmp_writes_explicit_sdst_mask():
    body = gen_vector_cmp(['sdst'], ['src0', 'src1'], 't', 'u32', is_vop3=True)

    assert 'amdgpu::write_wave_mask_scalar(sdst, wf, vcc);' in body
    assert 'wf.set_vcc' not in body


def test_vector_cmp_omits_redundant_mask_clears():
    body = gen_vector_cmp(['sdst'], ['src0', 'src1'], 'eq', 'u32', is_vop3=True)

    assert 'uint64_t vcc = 0;' in body
    assert 'vcc &= ~(1ULL << lane)' not in body


def test_vop3_cmp_writes_explicit_mask_width_for_wave_size():
    body = gen_vector_cmp(['vdst'], ['src0', 'src1'], 'eq', 'f32', is_vop3=True)

    assert 'amdgpu::write_wave_mask_scalar(vdst, wf, vcc);' in body
    assert 'wf.set_vcc' not in body


def test_vop3_add_co_writes_explicit_sdst_mask_width_for_wave_size():
    body = gen_vector_add_co(['vdst', 'sdst'], ['src0', 'src1'], 'add', 'u32')

    assert 'amdgpu::write_wave_mask_scalar(sdst, wf, vcc);' in body
    assert 'wf.set_vcc' not in body


def test_vop3_mad_u64_u32_writes_explicit_sdst_carry():
    body = gen_vector_mad_64_32(['vdst', 'sdst'], ['src0', 'src1', 'src2'], 'u64')

    assert 'uint64_t carry = 0;' in body
    assert 'uint64_t product = s0 * s1;' in body
    assert 'if (result < product)' in body
    assert 'carry |= 1ULL << lane;' in body
    assert 'amdgpu::write_wave_mask_scalar(sdst, wf, carry);' in body
    assert 'wf.set_vcc' not in body

    callback_body = gen_vector_mad_64_32(
        ['vdst', 'sdst'],
        ['src0', 'src1', 'src2'],
        'u64',
        result_writer='commit_result',
    )
    assert 'commit_result(carry);' in callback_body
    assert 'write_wave_mask_scalar' not in callback_body


def test_vector_cmp_class_writes_explicit_sdst_mask():
    body = gen_vector_cmp_class(
        ['sdst'], ['src0', 'src1'], 'f32', is_cmpx=False, is_vop3=True
    )

    assert 'amdgpu::write_wave_mask_scalar(sdst, wf, vcc);' in body
    assert 'wf.set_vcc' not in body

    callback_body = gen_vector_cmp_class(
        ['sdst'],
        ['src0', 'src1'],
        'f32',
        is_cmpx=False,
        is_vop3=True,
        result_writer='commit_result',
    )
    assert 'commit_result(vcc);' in callback_body
    assert 'write_wave_mask_scalar' not in callback_body


def test_vector_cmpx_can_delegate_exec_commit():
    body = gen_vector_cmpx(
        ['src0', 'src1'],
        'eq',
        'u32',
        is_vop3=True,
        result_writer='commit_result',
    )

    assert 'commit_result(result);' in body
    assert 'wf.set_exec(result);' not in body

    class_body = gen_vector_cmp_class(
        ['sdst'],
        ['src0', 'src1'],
        'f32',
        is_cmpx=True,
        is_vop3=True,
        result_writer='commit_result',
    )
    assert 'commit_result(result);' in class_body
    assert 'wf.set_exec(result);' not in class_body


def test_vector_cmp_class_omits_redundant_mask_clears():
    body = gen_vector_cmp_class(
        ['sdst'], ['src0', 'src1'], 'f32', is_cmpx=False, is_vop3=True
    )

    assert 'uint64_t vcc = 0;' in body
    assert 'vcc &= ~(1ULL << lane)' not in body


def test_vop3_f16_cmp_class_uses_selected_mask_half():
    for is_cmpx in (False, True):
        body = gen_vector_cmp_class(
            ['sdst'], ['src0', 'src1'], 'f16', is_cmpx=is_cmpx, is_vop3=True
        )

        assert 'uint32_t opsel = amdgpu::vop3_opsel(inst_);' in body
        assert (
            'uint16_t s0_raw = static_cast<uint16_t>('
            '::rocjitsu::amdgpu::read_vop3_true16_src(src0, wf, lane, opsel, 0));'
        ) in body
        assert (
            'uint32_t mask = ::rocjitsu::amdgpu::read_vop3_true16_src(src1, wf, lane, opsel, 1);'
        ) in body
        assert 'uint32_t mask = src1.read_lane(wf, lane);' not in body


def test_true16_vop3_integer_ops_do_not_use_whole_dword_simd_probe():
    assert simd_probe_line('v_mul_lo_u16_vop3') is None
    assert simd_probe_line('v_mad_u16_vop3') is None
    assert simd_probe_line('v_mad_i16_vop3') is None
    assert simd_probe_line('v_cmp_lt_i16_vop3') is None
    assert simd_probe_line('v_cmp_eq_u16_vop3') is None
    assert simd_probe_line('v_or_b16_vop3') is None
    assert simd_probe_line('v_add_nc_u32_vop3') is not None


def test_vop3_compare_simd_probe_can_commit_raw_result():
    default_probe = simd_probe_line('v_cmp_eq_u32_vop3')
    commit_probe = simd_probe_line('v_cmp_eq_u32_vop3', result_writer='commit_result')

    assert default_probe is not None
    assert commit_probe is not None
    assert 'ROCJITSU_TRY_SIMD_VOPC_VOP3_INT(' in default_probe
    assert 'ROCJITSU_TRY_SIMD_VOPC_VOP3_INT_RESULT(commit_result,' in commit_probe


@pytest.mark.parametrize(
    'template_name',
    (
        'v_add_co_ci_u32_vop2',
        'v_sub_co_ci_u32_vop2',
        'v_subrev_co_ci_u32_vop2',
    ),
)
def test_vop2_carry_simd_probe_can_commit_raw_result(template_name: str):
    default_probe = simd_probe_line(template_name)
    commit_probe = simd_probe_line(template_name, result_writer='commit_result')

    assert default_probe is not None
    assert commit_probe is not None
    assert 'ROCJITSU_TRY_SIMD_VOP2_CARRY(' in default_probe
    assert 'ROCJITSU_TRY_SIMD_VOP2_CARRY_RESULT(commit_result,' in commit_probe


def test_true16_vop3_cmp_uses_selected_source_halves():
    body = gen_vector_cmp(['vdst'], ['src0', 'src1'], 'lt', 'i16', is_vop3=True)

    assert 'uint32_t opsel = amdgpu::vop3_opsel(inst_);' in body
    assert body.index('uint32_t opsel = amdgpu::vop3_opsel(inst_);') < body.index(
        'for (uint32_t lane = 0; lane < wf.wf_size(); ++lane)'
    )
    assert 'if (opsel & (1u << 0)) s0_raw >>= 16;' in body
    assert 'if (opsel & (1u << 1)) s1_raw >>= 16;' in body
    assert 'vcc &= ~(1ULL << lane)' not in body


def test_true16_vop3_cmpx_hoists_opsel():
    opsel_line = 'uint32_t opsel = amdgpu::vop3_opsel(inst_);'
    loop_line = 'for (uint32_t lane = 0; lane < wf.wf_size(); ++lane)'

    for dtype in ('i16', 'u16'):
        body = gen_vector_cmpx(['src0', 'src1'], 'lt', dtype, is_vop3=True)

        assert body.count(opsel_line) == 1
        assert body.index(opsel_line) < body.index(loop_line)
        assert 'if (opsel & (1u << 0)) s0_raw >>= 16;' in body
        assert 'if (opsel & (1u << 1)) s1_raw >>= 16;' in body


def test_matrix_resolved_dense_operands_support_vgpr_and_inline_accumulators():
    inst = Instruction('V_WMMA_F32_16X16X32_F16', 'ENC_VOP3P', 0, [])
    profile = _matrix_profile(
        uses_vgpr_msb_indexing=True,
        matrix_layout=MatrixLayout.WMMA_SPLIT_K,
        wave_size=32,
        wave_size_max=64,
    )

    body = _gen_mfma(inst, 'rdna4', profile)

    for operand in ('vdst', 'src0', 'src1'):
        assert (
            f'Isa::resolved_vgpr_offset(wf, {operand}.opr_type_, '
            f'{operand}.encoding_value_, {operand}.vgpr_msb_role())'
        ) in body
    assert (
        'auto src2_off = Isa::resolved_vgpr_offset(wf, src2.opr_type_, '
        'src2.encoding_value_, src2.vgpr_msb_role());'
    ) in body
    assert 'const_acc = amdgpu::ACC_FROM_VGPR;' in body
    assert 's2 = vb + *src2_off;' in body
    assert 'const_acc = amdgpu::RegisterAccess(wf).read_scalar(src2);' in body
    assert (
        'amdgpu::exec_wmma_f32(cu, 16, 16, 32, 16, dst, src0_base, '
        'src1_base, s2,' in body
    )


def test_matrix_resolved_sparse_index_rejects_missing_vgpr_offset():
    inst = Instruction('V_SWMMAC_F32_16X16X32_F16', 'ENC_VOP3P', 0, [])
    profile = _matrix_profile(
        uses_vgpr_msb_indexing=True,
        swmmac_layout=SwmmacLayout.RUNTIME_WAVE,
        matrix_layout=MatrixLayout.WMMA_SPLIT_K,
        wave_size=32,
        wave_size_max=64,
    )

    body = _gen_mfma(inst, 'rdna4', profile)

    assert (
        'auto index_off = Isa::resolved_vgpr_offset(wf, src2.opr_type_, '
        'src2.encoding_value_, src2.vgpr_msb_role());'
    ) in body
    assert 'if (!index_off)\n    throw util::UnimplementedInst(mnemonic());' in body
    assert 'uint32_t index_base = vb + *index_off;' in body
    assert (
        'amdgpu::exec_swmmac_f32(cu, 16, 16, 32, 16, dst, src0_base, '
        'src1_base, s2, index_base, 16, index_key,' in body
    )


def test_matrix_direct_offsets_do_not_use_resolved_operand_setup():
    inst = Instruction('V_WMMA_F32_16X16X16_F16', 'ENC_VOP3P', 0, [])
    profile = _matrix_profile(
        matrix_layout=MatrixLayout.WMMA_SPLIT_K,
        wave_size=32,
        wave_size_max=64,
    )

    body = _gen_mfma(inst, 'rdna4', profile)

    assert 'uint32_t dst = vb + vdst.encoding_value_;' in body
    assert 'Isa::resolved_vgpr_offset' not in body
    assert 'amdgpu::resolve_acc(vb, dst,' in body
    assert (
        'amdgpu::exec_wmma_f32(cu, 16, 16, 16, 16, dst, '
        'src0_base, src1_base, s2,' in body
    )


def test_matrix_direct_sparse_operands_reach_final_call():
    inst = Instruction('V_SWMMAC_F32_16X16X32_F16', 'ENC_VOP3P', 0, [])
    profile = _matrix_profile(
        swmmac_layout=SwmmacLayout.FIXED_WAVE,
        matrix_layout=MatrixLayout.WMMA_SPLIT_K,
        wave_size=32,
        wave_size_max=32,
    )

    body = _gen_mfma(inst, 'cdna5', profile)

    assert 'Isa::resolved_vgpr_offset' not in body
    assert 'uint32_t index_base = amdgpu::src_base(vb, src2.encoding_value_);' in body
    assert (
        'amdgpu::exec_swmmac_f32(cu, 16, 16, 32, 16, dst, '
        'src0_base, src1_base, s2, index_base, '
        '16, index_key,' in body
    )


@pytest.mark.parametrize('uses_vgpr_msb_indexing', [False, True])
def test_unsupported_swmmac_layout_does_not_emit_sparse_setup(
    uses_vgpr_msb_indexing,
):
    body = _gen_mfma(
        Instruction('V_SWMMAC_F32_16X16X32_F16', 'ENC_VOP3P', 0, []),
        'test_arch',
        _matrix_profile(
            uses_vgpr_msb_indexing=uses_vgpr_msb_indexing,
            swmmac_layout=SwmmacLayout.NONE,
            matrix_layout=MatrixLayout.WMMA_SPLIT_K,
            wave_size=32,
            wave_size_max=32,
        ),
    )

    assert 'index_base' not in body
    assert 'index_key' not in body
    assert 'exec_swmmac' not in body
    assert 'amdgpu::exec_f32(' in body


@pytest.mark.parametrize(
    ('mnemonic', 'callee'),
    [
        ('V_SWMMAC_I32_16X16X32_IU8', 'exec_swmmac_i32'),
        ('V_SWMMAC_F32_16X16X32_F16', 'exec_swmmac_f32'),
    ],
)
@pytest.mark.parametrize('uses_vgpr_msb_indexing', [False, True])
def test_fixed_wave_swmmac_layout_selects_sparse_executor_without_arch_name(
    mnemonic,
    callee,
    uses_vgpr_msb_indexing,
):
    body = _gen_mfma(
        Instruction(mnemonic, 'ENC_VOP3P', 0, []),
        'renamed_fixed_wave_arch',
        _matrix_profile(
            uses_vgpr_msb_indexing=uses_vgpr_msb_indexing,
            swmmac_layout=SwmmacLayout.FIXED_WAVE,
            matrix_layout=MatrixLayout.WMMA_SPLIT_K,
            wave_size=32,
            wave_size_max=32,
        ),
    )
    call = _generated_matrix_call(body, callee)

    assert ('dst, src0_base, src1_base, s2, index_base,') in call
    assert 'uint32_t index_key = 0u;' in body
    assert 'uint32_t index_key = inst_.opsel & 0x1u;' not in body
    assert 'wf.wf_size()' not in call


@pytest.mark.parametrize(
    ('mnemonic', 'callee'),
    [
        ('V_WMMA_I32_16X16X64_IU8', 'exec_wmma_i32_16x16x64_iu8'),
        ('V_WMMA_F32_16X16X32_F16', 'exec_wmma_f32_16x16x32_f16'),
    ],
)
def test_fixed_wave32_split_k_dense_dispatch_does_not_depend_on_arch_name(
    mnemonic,
    callee,
):
    body = _gen_mfma(
        Instruction(mnemonic, 'ENC_VOP3P', 0, []),
        'renamed_fixed_wave32_arch',
        _matrix_profile(
            matrix_layout=MatrixLayout.WMMA_SPLIT_K,
            wave_size=32,
            wave_size_max=32,
        ),
    )

    assert f'amdgpu::{callee}(' in body


def test_replicated_halfwave_dense_layout_does_not_depend_on_arch_name():
    body = _gen_mfma(
        Instruction('V_WMMA_F32_16X16X16_F16', 'ENC_VOP3P', 0, []),
        'renamed_replicated_halfwave_arch',
        _matrix_profile(
            matrix_layout=MatrixLayout.WMMA_REPLICATED_HALFWAVE,
            wave_size=32,
            wave_size_max=64,
        ),
    )

    assert 'uint32_t dst = vb + vdst.encoding_value_;' in body
    assert 'amdgpu::exec_gfx11_wmma_f32(' in body


def test_runtime_wave_split_k_dense_layout_does_not_depend_on_arch_name():
    body = _gen_mfma(
        Instruction('V_WMMA_F32_16X16X16_F16', 'ENC_VOP3P', 0, []),
        'renamed_runtime_wave_split_k_arch',
        _matrix_profile(
            matrix_layout=MatrixLayout.WMMA_SPLIT_K,
            wave_size=32,
            wave_size_max=64,
        ),
    )

    assert 'uint32_t dst = vb + vdst.encoding_value_;' in body
    assert 'amdgpu::exec_wmma_f32(' in body
    assert 'wf.wf_size()' in _generated_matrix_call(body, 'exec_wmma_f32')


@pytest.mark.parametrize('uses_vgpr_msb_indexing', [False, True])
def test_runtime_wave_swmmac_layout_does_not_depend_on_arch_name(
    uses_vgpr_msb_indexing,
):
    body = _gen_mfma(
        Instruction('V_SWMMAC_F32_16X16X32_F16', 'ENC_VOP3P', 0, []),
        'renamed_runtime_wave_split_k_arch',
        _matrix_profile(
            uses_vgpr_msb_indexing=uses_vgpr_msb_indexing,
            swmmac_layout=SwmmacLayout.RUNTIME_WAVE,
            matrix_layout=MatrixLayout.WMMA_SPLIT_K,
            wave_size=32,
            wave_size_max=64,
        ),
    )

    if uses_vgpr_msb_indexing:
        assert (
            'uint32_t dst = vb + *Isa::resolved_vgpr_offset('
            'wf, vdst.opr_type_, vdst.encoding_value_, vdst.vgpr_msb_role());'
        ) in body
    else:
        assert 'uint32_t dst = vb + vdst.encoding_value_;' in body
    assert 'uint32_t index_key = inst_.opsel & 0x1u;' in body
    assert 'amdgpu::exec_swmmac_f32(' in body


def test_matrix_i32_final_call_sources_follow_profile_gate():
    inst = Instruction('V_WMMA_I32_16X16X16_IU8', 'ENC_VOP3P', 0, [])

    resolved = _gen_mfma(
        inst,
        'rdna4',
        _matrix_profile(
            uses_vgpr_msb_indexing=True,
            matrix_layout=MatrixLayout.WMMA_SPLIT_K,
            wave_size=32,
            wave_size_max=64,
        ),
    )
    direct = _gen_mfma(
        inst,
        'cdna5',
        _matrix_profile(
            matrix_layout=MatrixLayout.WMMA_SPLIT_K,
            wave_size=32,
            wave_size_max=32,
        ),
    )

    assert (
        'amdgpu::exec_wmma_i32(cu, 16, 16, 16, 8, dst, src0_base, '
        'src1_base, s2,' in resolved
    )
    assert (
        'amdgpu::exec_wmma_i32(cu, 16, 16, 16, 8, dst, src0_base, '
        'src1_base, s2,' in direct
    )


def _generated_matrix_call(body: str, callee: str) -> str:
    start = body.index(f'amdgpu::{callee}')
    end = body.index(');', start) + 2
    return ' '.join(body[start:end].split())


@pytest.mark.parametrize(
    ('mnemonic', 'arch', 'callee', 'is_sparse'),
    [
        # Fixed-wave I32: sparse, specialized dense, and generic dense.
        ('V_SWMMAC_I32_16X16X32_IU8', 'cdna5', 'exec_swmmac_i32', True),
        (
            'V_WMMA_I32_16X16X64_IU8',
            'cdna5',
            'exec_wmma_i32_16x16x64_iu8',
            False,
        ),
        ('V_WMMA_I32_16X16X16_IU4', 'cdna5', 'exec_wmma_i32', False),
        # Runtime-wave and replicated-half-wave I32.
        ('V_SWMMAC_I32_16X16X32_IU8', 'rdna4', 'exec_swmmac_i32', True),
        ('V_WMMA_I32_16X16X16_IU8', 'rdna3', 'exec_gfx11_wmma_i32', False),
        ('V_WMMA_I32_16X16X16_IU8', 'rdna4', 'exec_wmma_i32', False),
        # I32 MFMA-style fallback dispatches.
        ('V_WMMA_I32_16X16X16_IU8', 'test_arch', 'exec_i32_mixed', False),
        ('V_MFMA_I32_16X16X16_IU4', 'cdna3', 'exec_i32_mixed', False),
        ('V_MFMA_I32_16X16X16_I8', 'cdna3', 'exec_i32_i8', False),
        # Dynamic matrix-format WMMA.
        (
            'V_WMMA_F32_16X16X128_F8F6F4',
            'cdna5',
            'exec_wmma_f32_mixed',
            False,
        ),
        # Standalone structured-sparse SMFMAC variants.
        (
            'V_SMFMAC_F32_16X16X32_F16',
            'cdna3',
            'exec_smfmac_f32_16x16x32_f16',
            True,
        ),
        (
            'V_SMFMAC_F32_16X16X64_FP8_BF8',
            'cdna3',
            'exec_smfmac_f32_16x16x64_fp8',
            True,
        ),
        # Fixed-wave sparse float result variants.
        ('V_SWMMAC_F32_16X16X32_F16', 'cdna5', 'exec_swmmac_f32', True),
        ('V_SWMMAC_F16_16X16X32_F16', 'cdna5', 'exec_swmmac_f16', True),
        ('V_SWMMAC_BF16_16X16X32_BF16', 'cdna5', 'exec_swmmac_bf16', True),
        # Runtime-wave sparse float result variants.
        ('V_SWMMAC_F32_16X16X32_F16', 'rdna4', 'exec_swmmac_f32', True),
        ('V_SWMMAC_F16_16X16X32_F16', 'rdna4', 'exec_swmmac_f16', True),
        ('V_SWMMAC_BF16_16X16X32_BF16', 'rdna4', 'exec_swmmac_bf16', True),
        # Fixed-wave specialized dense result variants.
        (
            'V_WMMA_F32_16X16X32_F16',
            'cdna5',
            'exec_wmma_f32_16x16x32_f16',
            False,
        ),
        (
            'V_WMMA_BF16F32_16X16X32_BF16',
            'cdna5',
            'exec_wmma_bf16f32_16x16x32_bf16',
            False,
        ),
        ('V_WMMA_F16_16X16X32_F16', 'cdna5', 'exec_wmma_f16_spec', False),
        ('V_WMMA_BF16_16X16X32_BF16', 'cdna5', 'exec_wmma_bf16_spec', False),
        (
            'V_WMMA_F16_16X16X64_FP8_BF8',
            'cdna5',
            'exec_wmma_f16_f8_spec',
            False,
        ),
        # Fixed-wave generic dense result variants.
        ('V_WMMA_F32_16X16X16_F16', 'cdna5', 'exec_wmma_f32', False),
        ('V_WMMA_F16_16X16X16_F16', 'cdna5', 'exec_wmma_f16', False),
        ('V_WMMA_BF16_16X16X16_BF16', 'cdna5', 'exec_wmma_bf16', False),
        # Replicated-half-wave and runtime-wave float dispatches.
        ('V_WMMA_F32_16X16X16_F16', 'rdna3', 'exec_gfx11_wmma_f32', False),
        ('V_WMMA_F16_16X16X16_F16', 'rdna3', 'exec_gfx11_wmma_f16', False),
        ('V_WMMA_BF16_16X16X16_BF16', 'rdna3', 'exec_gfx11_wmma_bf16', False),
        ('V_WMMA_F32_16X16X16_F16', 'rdna4', 'exec_wmma_f32', False),
        ('V_WMMA_F16_16X16X16_F16', 'rdna4', 'exec_wmma_f16', False),
        ('V_WMMA_BF16_16X16X16_BF16', 'rdna4', 'exec_wmma_bf16', False),
        # MFMA specialized and generic float dispatches.
        (
            'V_MFMA_F32_16X16X32_FP8_FP8',
            'cdna3',
            'exec_f32_mfma_f8_spec',
            False,
        ),
        ('V_MFMA_F32_16X16X16_F16', 'cdna3', 'exec_f32_mfma_f16_spec', False),
        ('V_MFMA_F32_16X16X16_F4', 'cdna3', 'exec_f32', False),
        ('V_MFMA_F32_16X16X16_F16', 'test_arch', 'exec_f32', False),
    ],
)
@pytest.mark.parametrize('uses_vgpr_msb_indexing', [False, True])
def test_every_matrix_executor_uses_profile_selected_operand_bases(
    mnemonic, arch, callee, is_sparse, uses_vgpr_msb_indexing
):
    body = _gen_mfma(
        Instruction(mnemonic, 'ENC_VOP3P', 0, []),
        arch,
        _matrix_profile_for_arch(
            arch,
            uses_vgpr_msb_indexing=uses_vgpr_msb_indexing,
        ),
    )
    call = _generated_matrix_call(body, callee)
    is_standalone_smfmac = mnemonic.startswith('V_SMFMAC_')

    if is_standalone_smfmac:
        expected_operands = 'dst, s0b, s1b, idx'
        if uses_vgpr_msb_indexing:
            assert 'uint32_t s0b = vb + *Isa::resolved_vgpr_offset' in body
            assert 'uint32_t s1b = vb + *Isa::resolved_vgpr_offset' in body
            assert 'uint32_t idx = vb + *index_off;' in body
        else:
            assert 'uint32_t s0b = amdgpu::src_base(vb, src0.encoding_value_);' in body
            assert 'uint32_t s1b = amdgpu::src_base(vb, src1.encoding_value_);' in body
            assert 'uint32_t idx = amdgpu::src_base(vb, src2.encoding_value_);' in body
            assert 'Isa::resolved_vgpr_offset' not in body
    else:
        if uses_vgpr_msb_indexing:
            expected_operands = 'dst, src0_base, src1_base, s2'
            assert 'uint32_t src0_base = vb + *Isa::resolved_vgpr_offset' in body
            assert 'uint32_t src1_base = vb + *Isa::resolved_vgpr_offset' in body
        else:
            expected_operands = 'dst, src0_base, src1_base, s2'
            assert (
                'uint32_t src0_base = amdgpu::src_base(vb, src0.encoding_value_);'
                in body
            )
            assert (
                'uint32_t src1_base = amdgpu::src_base(vb, src1.encoding_value_);'
                in body
            )
            assert 'Isa::resolved_vgpr_offset' not in body

    assert expected_operands in call
    if is_sparse and not is_standalone_smfmac:
        assert f'{expected_operands}, index_base,' in call
        if uses_vgpr_msb_indexing:
            assert 'uint32_t index_base = vb + *index_off;' in body
        else:
            assert (
                'uint32_t index_base = '
                'amdgpu::src_base(vb, src2.encoding_value_);' in body
            )


@pytest.mark.parametrize('uses_vgpr_msb_indexing', [False, True])
def test_dynamic_mfma_aliases_use_profile_selected_operand_bases(
    uses_vgpr_msb_indexing,
):
    body = _gen_mfma(
        Instruction('V_MFMA_F32_16X16X128_F8F6F4', 'ENC_VOP3P_MFMA', 0, []),
        'cdna3',
        _matrix_profile(uses_vgpr_msb_indexing=uses_vgpr_msb_indexing),
    )

    if uses_vgpr_msb_indexing:
        assert 'uint32_t s0b = src0_base;' in body
        assert 'uint32_t s1b = src1_base;' in body
    else:
        assert 'uint32_t s0b = src0_base;' in body
        assert 'uint32_t s1b = src1_base;' in body
    assert (
        'a_bits, b_bits, dst, s0b, s1b, s2, ea, eb, const_acc);'
        in _generated_matrix_call(body, 'exec_f32_mixed')
    )


def test_matrix_f64_resolved_sources_use_normalized_base_expressions():
    inst = Instruction('V_MFMA_F64_16X16X4_F64', 'ENC_VOP3P_MFMA', 0, [])
    profile = _matrix_profile(uses_vgpr_msb_indexing=True)

    body = _gen_mfma(inst, 'rdna4', profile)

    assert (
        'uint32_t src0_base = vb + *Isa::resolved_vgpr_offset(wf, '
        'src0.opr_type_, src0.encoding_value_, src0.vgpr_msb_role());'
    ) in body
    assert (
        'uint32_t src1_base = vb + *Isa::resolved_vgpr_offset(wf, '
        'src1.opr_type_, src1.encoding_value_, src1.vgpr_msb_role());'
    ) in body
    assert '                 src0_base,\n                 src1_base,' in body


def test_matrix_f64_direct_sources_use_direct_base_expressions():
    inst = Instruction('V_MFMA_F64_16X16X4_F64', 'ENC_VOP3P_MFMA', 0, [])
    profile = _matrix_profile()

    body = _gen_mfma(inst, 'cdna5', profile)

    assert 'Isa::resolved_vgpr_offset' not in body
    assert ('                 src0_base,\n' '                 src1_base,') in body


@pytest.mark.parametrize(
    'mnemonic',
    [
        'V_SMFMAC_F32_16X16X32_BF16',
        'V_MFMA_F32_16X16X16_F16',
    ],
)
def test_matrix_acc_cd_destination_uses_encoding_field_presence(mnemonic):
    inst = Instruction(mnemonic, 'ENC_VOP3P_MFMA', 0, [])
    profile = CdnaProfile()

    with_acc_cd = _gen_mfma(inst, 'renamed_matrix_arch', profile, {'acc_cd'})
    without_acc_cd = _gen_mfma(inst, 'renamed_matrix_arch', profile, set())

    assert 'amdgpu::dst_base(vb, vdst.encoding_value_, inst_.acc_cd)' in with_acc_cd
    assert 'amdgpu::dst_base(vb, vdst.encoding_value_, 1)' not in with_acc_cd
    assert 'amdgpu::dst_base(vb, vdst.encoding_value_, 1)' in without_acc_cd
    assert 'inst_.acc_cd' not in without_acc_cd


def test_cdna3_real_spec_mfma_destination_uses_acc_cd(tmp_path):
    isa_xml = _mrisa_dir() / 'amdgpu_isa_cdna3.xml'
    if not isa_xml.is_file():
        pytest.skip('CDNA3 semantics XML not available')

    args = SimpleNamespace(
        isafiles=[f'cdna3:{isa_xml}'],
        gen_isas=True,
        gen_dbt=False,
        isa_output=str(tmp_path),
        dbt_output=None,
    )
    _run(args)

    vop3p = (tmp_path / 'cdna3' / 'vop3p_exec.cpp').read_text()
    body = _generated_method_body(
        vop3p,
        'VMfmaF3216x16x8Xf32Vop3pMfma',
        'VMfmaF3232x32x4Xf32Vop3pMfma',
    )

    assert 'amdgpu::dst_base(vb, vdst.encoding_value_, inst_.acc_cd)' in body


def test_gfx1250_wmma_f32_passes_c_modifier_to_accumulator_helper():
    inst = Instruction('V_WMMA_F32_16X16X32_F16', 'ENC_VOP3P', 0, [])
    body = _gen_mfma(inst, 'cdna5')

    assert 'amdgpu::exec_wmma_f32_16x16x32_f16(' in body
    assert 'amdgpu::wmma_c_modifier(inst_.neg, inst_.neg_hi)' in body


def test_gfx1250_mixed_wmma_uses_public_opsel_hi_2_field():
    inst = Instruction('V_WMMA_F32_16X16X128_F8F6F4', 'ENC_VOP3P', 0, [])
    body = gen_mfma(
        inst,
        ['vdst'],
        ['src0', 'src1', 'src2'],
        'cdna5',
        op_sel_hi_2_expr='inst_.opsel_hi_2',
    )

    assert '(inst_.opsel_hi_2 << 2) | inst_.opsel_hi' in body
    assert 'pad_14' not in body


@pytest.mark.parametrize(
    ('arch_name', 'expected'),
    [('cdna5', True), ('gfx1250', False)],
)
def test_generic_wmma_accumulator_selector_uses_cdna5_logical_key(
    arch_name: str, expected: bool
) -> None:
    generator = CodeGenerator(SimpleNamespace(arch_name=arch_name), '')
    inst_sem = InstructionSemantics('V_WMMA_F32_16X16X128_FP8_FP8', 'vector_wmma')
    operand = Operand(
        'src2', 256, 'OPR_SRC_VGPR_OR_INLINE', True, False, False, False, 3
    )

    assert (
        generator._uses_generic_wmma_accumulator_selector(inst_sem, operand) is expected
    )


@pytest.mark.parametrize('k', [64, 128])
@pytest.mark.parametrize(
    ('input_type', 'a_fp8', 'b_fp8'),
    [
        ('FP8_FP8', 'true', 'true'),
        ('FP8_BF8', 'true', 'false'),
        ('BF8_FP8', 'false', 'true'),
        ('BF8_BF8', 'false', 'false'),
    ],
)
def test_gfx1250_wmma_f16_f8_passes_fp16_overflow_mode(
    k: int, input_type: str, a_fp8: str, b_fp8: str
) -> None:
    inst = Instruction(f'V_WMMA_F16_16X16X{k}_{input_type}', 'ENC_VOP3P', 0, [])
    body = _gen_mfma(inst, 'cdna5')

    assert f'amdgpu::exec_wmma_f16_f8_spec<16, 16, {k}, {a_fp8}, {b_fp8}>(' in body
    assert 'const_acc, wf.fp16_ovfl());' in body


def test_gfx1250_wmma_f16_overflow_mode_is_scoped_to_f8_helper() -> None:
    inst = Instruction('V_WMMA_F16_16X16X32_F16', 'ENC_VOP3P', 0, [])
    body = _gen_mfma(inst, 'cdna5')

    assert 'amdgpu::exec_wmma_f16_spec<16, 16, 32>(' in body
    assert 's2, const_acc);' in body
    assert 'wf.fp16_ovfl()' not in body


def test_rdna_wmma_uses_arch_specific_wave32_operand_layout():
    operands = [
        Operand('vdst', 256, 'OPR_VGPR', False, True, False, False, 0),
        Operand('src0', 256, 'OPR_SRC_VGPR', True, False, False, False, 1),
        Operand('src1', 256, 'OPR_SRC_VGPR', True, False, False, False, 2),
        Operand('src2', 256, 'OPR_SRC_VGPR_OR_INLINE', True, False, False, False, 3),
    ]
    inst = Instruction('V_WMMA_F32_16X16X16_F16', 'ENC_VOP3P', 0, operands)

    for arch in ('rdna3', 'rdna3_5'):
        body = _gen_mfma(inst, arch)

        assert 'uint32_t dst = vb + vdst.encoding_value_;' in body
        assert (
            'amdgpu::exec_gfx11_wmma_f32(cu, wf.wf_size(), 16, 16, 16, 16, dst,' in body
        )
        assert 'amdgpu::exec_f32(cu, 16, 16, 16' not in body

    body = _gen_mfma(inst, 'rdna4')
    assert 'uint32_t dst = vb + vdst.encoding_value_;' in body
    assert 'amdgpu::exec_wmma_f32(cu, 16, 16, 16, 16, dst,' in body
    assert 'amdgpu::exec_f32(cu, 16, 16, 16' not in body


def test_rdna_wmma_i32_iu8_uses_arch_specific_wave32_operand_layout():
    operands = [
        Operand('vdst', 256, 'OPR_VGPR', False, True, False, False, 0),
        Operand('src0', 128, 'OPR_SRC_VGPR', True, False, False, False, 1),
        Operand('src1', 128, 'OPR_SRC_VGPR', True, False, False, False, 2),
        Operand('src2', 256, 'OPR_SRC_VGPR_OR_INLINE', True, False, False, False, 3),
    ]
    inst = Instruction('V_WMMA_I32_16X16X16_IU8', 'ENC_VOP3P', 0, operands)

    for arch in ('rdna3', 'rdna3_5'):
        body = _gen_mfma(inst, arch)

        assert 'uint32_t dst = vb + vdst.encoding_value_;' in body
        assert (
            'auto extract_a = [&](auto &cu, uint32_t base, const amdgpu::InputLoc &loc)'
            in body
        )
        assert '(inst_.neg & 0x1u) ? amdgpu::extract_i8(cu, base, loc)' in body
        assert '(inst_.neg & 0x2u) ? amdgpu::extract_i8(cu, base, loc)' in body
        assert (
            'amdgpu::exec_gfx11_wmma_i32(cu, wf.wf_size(), 16, 16, 16, 8, dst,' in body
        )
        assert 'amdgpu::exec_i32_mixed(cu, 16, 16, 16' not in body

    body = _gen_mfma(inst, 'rdna4')
    assert 'uint32_t dst = vb + vdst.encoding_value_;' in body
    assert (
        'auto extract_a = [&](auto &cu, uint32_t base, const amdgpu::InputLoc &loc)'
        in body
    )
    assert '(inst_.neg & 0x1u) ? amdgpu::extract_i8(cu, base, loc)' in body
    assert '(inst_.neg & 0x2u) ? amdgpu::extract_i8(cu, base, loc)' in body
    assert 'amdgpu::exec_wmma_i32(cu, 16, 16, 16, 8, dst,' in body
    assert 'amdgpu::exec_i32_mixed(cu, 16, 16, 16' not in body


def test_rdna_wmma_f16_bf16_use_arch_specific_wave32_dispatch():
    operands = [
        Operand('vdst', 256, 'OPR_VGPR', False, True, False, False, 0),
        Operand('src0', 256, 'OPR_SRC_VGPR', True, False, False, False, 1),
        Operand('src1', 256, 'OPR_SRC_VGPR', True, False, False, False, 2),
        Operand('src2', 256, 'OPR_SRC_VGPR_OR_INLINE', True, False, False, False, 3),
    ]

    for dtype in ('F16', 'BF16'):
        inst = Instruction(
            f'V_WMMA_{dtype}_16X16X16_{dtype}',
            'ENC_VOP3P',
            0,
            operands,
        )
        lower = dtype.lower()

        for arch in ('rdna3', 'rdna3_5'):
            body = _gen_mfma(inst, arch)

            assert 'uint32_t dst = vb + vdst.encoding_value_;' in body
            assert (
                f'amdgpu::exec_gfx11_wmma_{lower}(cu, wf.wf_size(), 16, 16, 16, 16, dst,'
                in body
            )
            assert '(inst_.op_sel >> 2) & 0x1u' in body
            assert f'amdgpu::exec_wmma_{lower}(cu, 16, 16, 16' not in body

        body = _gen_mfma(inst, 'rdna4')
        assert 'uint32_t dst = vb + vdst.encoding_value_;' in body
        assert f'amdgpu::exec_wmma_{lower}(cu, 16, 16, 16, 16, dst,' in body
        assert 'wf.wf_size());' in body
        assert 'exec_gfx11_wmma' not in body


def test_gfx1250_wmma_i32_iu4_emits_executor():
    inst = Instruction('V_WMMA_I32_16X16X16_IU4', 'ENC_VOP3P', 0, [])
    body = _gen_mfma(inst, 'cdna5')

    assert '(inst_.neg & 0x1u) ? amdgpu::extract_i4(cu, base, loc)' in body
    assert 'amdgpu::exec_wmma_i32(cu, 16, 16, 16, 4, dst, src0_base,' in body


def test_cdna3_fp8_mfma_uses_fnuz_helper_variant():
    inst = Instruction('V_MFMA_F32_16X16X32_FP8_FP8', 'ENC_VOP3P_MFMA', 0, [])
    body = _gen_mfma(inst, 'cdna3')

    assert 'amdgpu::exec_f32_mfma_f8_spec<16, 16, 32, true, true, true>(' in body


def test_cdna3_fp8_smfmac_uses_fnuz_readers():
    inst = Instruction('V_SMFMAC_F32_16X16X64_FP8_BF8', 'ENC_VOP3P_MFMA', 0, [])
    body = _gen_mfma(inst, 'cdna3')

    assert 'amdgpu::smfmac_read_fp8_fnuz' in body
    assert 'amdgpu::smfmac_read_bf8_fnuz' in body


def test_cdna4_fp8_mfma_keeps_ocp_helper_variant():
    inst = Instruction('V_MFMA_F32_16X16X32_FP8_FP8', 'ENC_VOP3P_MFMA', 0, [])
    body = _gen_mfma(inst, 'cdna4')

    assert 'amdgpu::exec_f32_mfma_f8_spec<16, 16, 32, true, true>(' in body
    assert 'amdgpu::exec_f32_mfma_f8_spec<16, 16, 32, true, true, true>(' not in body


def test_cdna4_matrix_bases_apply_gpr_idx_by_operand_role():
    dense = Instruction('V_MFMA_F32_16X16X4_F32', 'ENC_VOP3P_MFMA', 0, [])
    sparse = Instruction('V_SMFMAC_F32_16X16X64_BF16', 'ENC_VOP3P_MFMA', 0, [])

    profile = CdnaProfile()
    dense_body = gen_mfma(
        dense,
        ['vdst'],
        ['src0', 'src1', 'src2'],
        'cdna4',
        supports_gpr_idx=profile.supports_gpr_idx,
    )
    sparse_body = gen_mfma(
        sparse,
        ['vdst'],
        ['src0', 'src1', 'src2'],
        'cdna4',
        supports_gpr_idx=profile.supports_gpr_idx,
    )

    for role in ('Dst', 'Src0', 'Src1', 'Src2'):
        assert f'amdgpu::VgprMsbRole::{role}' in dense_body
        assert f'amdgpu::VgprMsbRole::{role}' in sparse_body
    assert 'if (const_acc == amdgpu::ACC_FROM_VGPR)' in dense_body
    assert 'apply_gpr_idx_to_mma_base' in dense_body
    assert 'apply_gpr_idx_to_mma_base' in sparse_body


def test_cdna5_matrix_bases_follow_profile_gpr_idx_policy():
    inst = Instruction('V_WMMA_I32_16X16X16_IU8', 'ENC_VOP3P_MFMA', 0, [])
    profile = Cdna5Profile()

    body = gen_mfma(
        inst,
        ['vdst'],
        ['src0', 'src1', 'src2'],
        'cdna5',
        supports_gpr_idx=profile.supports_gpr_idx,
    )

    assert not profile.supports_gpr_idx
    assert 'apply_gpr_idx_to_mma_base' not in body


def test_disabled_gpr_idx_capability_reaches_sparse_and_dense_mixed_matrix_branches():
    sparse = Instruction('V_SMFMAC_F32_16X16X64_BF16', 'ENC_VOP3P_MFMA', 0, [])
    dense = Instruction('V_MFMA_F32_16X16X32_F8_F6_F4', 'ENC_VOP3P_MFMA', 0, [])

    sparse_body = gen_mfma(
        sparse,
        ['vdst'],
        ['src0', 'src1', 'src2'],
        'cdna4',
        supports_gpr_idx=False,
    )
    dense_body = gen_mfma(
        dense,
        ['vdst'],
        ['src0', 'src1', 'src2'],
        'cdna4',
        supports_gpr_idx=False,
    )

    assert 'apply_gpr_idx_to_mma_base' not in sparse_body
    assert 'apply_gpr_idx_to_mma_base' not in dense_body
    assert 'exec_f32_mixed' in dense_body
    assert 'exec_f32_scaled_mixed' not in dense_body
    assert 'raw_words_' not in dense_body


def test_cdna4_wide_conversion_indexes_each_base_once():
    body = gen_cvt_scalef32(
        SimpleNamespace(
            op='2xpk16_fp6_f32',
            dst_ops=['vdst'],
            src_ops=['src0', 'src1', 'src2'],
            arch_name='cdna4',
        )
    )

    for operand in ('vdst', 'src0', 'src1'):
        assert body.count(f'auto {operand}_off = Isa::resolved_vgpr_offset') == 1
        assert f'uint32_t {operand}_base = vb + amdgpu::apply_gpr_idx' in body
    assert 'unified_vgpr_index()' not in body
    assert 'rd(src0_base' in body
    assert 'rd(src1_base' in body
    assert 'wr(vdst_base' in body


def test_cdna3_fp8_cvt_uses_fnuz_helper_variant():
    ctx = SimpleNamespace(
        dst_ops=['vdst'],
        src_ops=['src0', 'src1'],
        is_vop3=True,
        arch_name='cdna3',
        enc_field_names={'op_sel'},
        enc_name='ENC_VOP3',
        encoding_map=None,
    )

    widen = gen_cvt_fp8(SimpleNamespace(**ctx.__dict__, op='pk_f32_fp8'))
    assert 'util::fp8_e4m3_fnuz_to_f32' in widen
    assert 'util::fp8_e4m3_to_f32' not in widen

    narrow = gen_cvt_fp8(SimpleNamespace(**ctx.__dict__, op='pk_fp8_f32'))
    assert 'util::f32_to_fp8_e4m3_fnuz_rne_mode' in narrow
    assert 'util::f32_to_fp8_e4m3_rne_mode' not in narrow
    assert 'wf.fp16_ovfl()' in narrow

    packed = gen_vector_cvt_pk(
        ['vdst'],
        ['src0'],
        'vector_cvt_pk',
        'f32_bf8',
        opsel='inst_.op_sel',
        arch_name='cdna3',
    )
    assert 'util::bf8_e5m2_fnuz_to_f32' in packed
    assert 'util::bf8_e5m2_to_f32' not in packed

    unary = gen_vector_unary(
        ['vdst'],
        ['src0'],
        'cvt_f32_fp8',
        None,
        arch_name='cdna3',
    )
    assert 'util::fp8_e4m3_fnuz_to_f32' in unary
    assert 'util::fp8_e4m3_to_f32' not in unary


def test_cdna4_fp8_cvt_keeps_ocp_helper_variant():
    ctx = SimpleNamespace(
        dst_ops=['vdst'],
        src_ops=['src0', 'src1'],
        is_vop3=True,
        arch_name='cdna4',
        enc_field_names={'op_sel'},
        enc_name='ENC_VOP3',
        encoding_map=None,
    )

    widen = gen_cvt_fp8(SimpleNamespace(**ctx.__dict__, op='pk_f32_fp8'))
    assert 'util::fp8_e4m3_to_f32' in widen
    assert 'util::fp8_e4m3_fnuz_to_f32' not in widen

    narrow = gen_cvt_fp8(SimpleNamespace(**ctx.__dict__, op='pk_fp8_f32'))
    assert 'util::f32_to_fp8_e4m3_rne_mode' in narrow
    assert 'util::f32_to_fp8_e4m3_fnuz_rne_mode' not in narrow
    assert 'wf.fp16_ovfl()' in narrow

    unary = gen_vector_unary(
        ['vdst'],
        ['src0'],
        'cvt_f32_fp8',
        None,
        arch_name='cdna4',
    )
    assert 'util::fp8_e4m3_to_f32' in unary
    assert 'util::fp8_e4m3_fnuz_to_f32' not in unary


def test_f32_to_f16_vector_conversion_threads_fp16_ovfl():
    unary = gen_vector_unary(['vdst'], ['src0'], 'cvt', 'f16_f32')

    assert 'util::f32_to_f16_mode(s, wf.fp16_ovfl())' in unary


def test_fp16_ovfl_sensitive_f16_simd_probes_stay_vectorized():
    cvt_probe = simd_probe_line('v_cvt_f16_f32_vop1')
    assert cvt_probe is not None
    assert 'if (wf.fp16_ovfl())' in cvt_probe
    assert 'util::f32_to_f16_ovfl_simd' in cvt_probe
    assert 'util::f32_to_f16_simd' in cvt_probe
    assert 'ROCJITSU_TRY_SIMD_VOP1_UNARY' in cvt_probe

    add_probe = simd_probe_line('v_add_f16_vop3', true16_vop3=True)
    assert add_probe is not None
    assert 'if (wf.fp16_ovfl())' in add_probe
    assert 'util::f32_to_f16_ovfl_simd' in add_probe
    assert 'util::f32_to_f16_simd' in add_probe
    assert 'ROCJITSU_TRY_SIMD_VOP3_BINARY_TRUE16_F16' in add_probe


def test_cdna_f64_mfma_uses_blgp_as_neg_immediate():
    operands = [
        Operand('vdst', 256, 'OPR_VGPR', False, True, False, False, 0),
        Operand('src0', 64, 'OPR_SRC_VGPR', True, False, False, False, 1),
        Operand('src1', 64, 'OPR_SRC_VGPR', True, False, False, False, 2),
        Operand('src2', 256, 'OPR_SRC_VGPR_OR_INLINE', True, False, False, False, 3),
    ]
    inst = Instruction('V_MFMA_F64_16X16X4_F64', 'ENC_VOP3P_MFMA', 0, operands)

    for arch in ('cdna3', 'cdna4'):
        body = _gen_mfma(inst, arch)
        assert 's2, const_acc, inst_.blgp);' in body

    for arch in ('rdna3', 'rdna4', 'cdna5'):
        body = _gen_mfma(inst, arch)
        assert 's2, const_acc, 0u);' in body


def test_div_scale_uses_signed_tiny_exponent_threshold():
    body = gen_vector_div_scale(
        ['vdst', 'sdst'], ['src0', 'src1', 'src2'], 'f32', is_vop3=True
    )

    assert 'exp2 <= -23' in body
    assert 'exp2 <= 23' not in body


def test_gfx1250_profile_enables_generator_backed_quirks():
    profile = Cdna5Profile()

    assert profile.uses_packed_16bit_e32_source_selectors
    assert profile.uses_true16_vop3_opsel
    assert profile.vbuffer_store_data_uses_dst_vgpr_msb_role
    assert profile.buffer_payload_reads_use_effective_exec_mask
    assert profile.generate_scaled_wmma_vop3px2
    assert profile.smem_address_uses_access_size
    assert profile.ds_transpose_ignores_exec
    assert profile.vop3_cmp_sdst_size_bits == 32
    assert profile.vop3_cndmask_selector_size_bits == 32
    assert profile.vop3_carry_mask_size_bits == 32


@pytest.mark.parametrize(
    'enc_name,inst_name,operand_name,expected_role',
    [
        ('ENC_VOP2', 'V_FMAMK_F16', 'vsrc1', 'Src2'),
        ('ENC_VOP2', 'V_FMAMK_F32', 'vsrc1', 'Src2'),
        ('ENC_VOP2', 'V_FMAMK_F64', 'vsrc1', 'Src2'),
        ('ENC_VOP2', 'V_MADMK_F16', 'vsrc1', 'Src2'),
        ('ENC_VOP2', 'V_MADMK_F32', 'vsrc1', 'Src2'),
        ('ENC_VOP2', 'V_ADD_F32', 'vsrc1', 'Src1'),
        ('ENC_VOP1', 'V_SWAP_B32', 'src0', 'Src0'),
        ('ENC_VDS', 'DS_STORE_ADDTID_B32', 'data0', 'Src1'),
        ('ENC_VDS', 'DS_STORE_2ADDR_B32', 'data1', 'Src2'),
        ('ENC_VDS', 'DS_LOAD_TR4_B64', 'vdst', 'Dst'),
        ('ENC_VGLOBAL', 'GLOBAL_STORE_ADDTID_B32', 'vsrc', 'Src1'),
        ('ENC_VGLOBAL', 'GLOBAL_LOAD_ASYNC_TO_LDS_B8', 'vdst', 'Dst'),
        ('ENC_VGLOBAL', 'GLOBAL_LOAD_ASYNC_TO_LDS_B8', 'vaddr', 'Src0'),
    ],
)
def test_gfx1250_vgpr_msb_roles_follow_physical_encoding_slots(
    enc_name: str, inst_name: str, operand_name: str, expected_role: str
):
    codegen = object.__new__(CodeGenerator)
    operand = SimpleNamespace(name=operand_name)

    assert codegen._fixed_vgpr_msb_role(enc_name, inst_name, operand) == expected_role


def test_rdna3_profile_enables_gfx11_vop3_true16_only():
    profile = Rdna3Profile()

    assert profile.uses_packed_16bit_e32_source_selectors
    assert profile.uses_true16_vop3_opsel
    assert not profile.vbuffer_store_data_uses_dst_vgpr_msb_role
    assert not profile.buffer_payload_reads_use_effective_exec_mask


def test_rdna4_profile_enables_gfx12_true16_and_mode_hwregs_only():
    profile = Rdna4Profile()

    assert profile.uses_packed_16bit_e32_source_selectors
    assert profile.uses_true16_vop3_opsel
    assert not profile.vbuffer_store_data_uses_dst_vgpr_msb_role
    assert not profile.buffer_payload_reads_use_effective_exec_mask
    assert not profile.generate_scaled_wmma_vop3px2
    assert not profile.smem_address_uses_access_size
    assert not profile.ds_transpose_ignores_exec
    assert profile.vop3_cmp_sdst_size_bits is None
    assert profile.vop3_cndmask_selector_size_bits is None
    assert profile.vop3_carry_mask_size_bits is None


def test_gfx1250_vop3_compare_sdst_uses_wave32_mask_size():
    codegen = object.__new__(CodeGenerator)
    codegen.isa_spec = SimpleNamespace(profile=Cdna5Profile())
    sem = SimpleNamespace(semantic_class='vector_cmp')

    dst = SimpleNamespace(is_output=True, name='vdst', operand_type='OPR_SREG')
    src = SimpleNamespace(is_output=False, name='src0', operand_type='OPR_SRC')

    assert codegen._operand_size_override('ENC_VOP3', dst, sem) == '32'
    assert codegen._operand_size_override('ENC_VOPC', dst, sem) is None
    assert codegen._operand_size_override('ENC_VOP3', src, sem) is None

    cmpx = SimpleNamespace(semantic_class='vector_cmpx')
    assert codegen._operand_size_override('ENC_VOP3', dst, cmpx) is None


def test_gfx1250_vop3_cndmask_selector_uses_wave32_mask_size():
    codegen = object.__new__(CodeGenerator)
    codegen.isa_spec = SimpleNamespace(profile=Cdna5Profile())
    sem = SimpleNamespace(semantic_class='vector_cndmask')

    selector = SimpleNamespace(is_output=False, name='src2', operand_type='OPR_SREG')
    source = SimpleNamespace(is_output=False, name='src0', operand_type='OPR_SRC')

    assert codegen._operand_size_override('ENC_VOP3', selector, sem) == '32'
    assert codegen._operand_size_override('ENC_VOP3', source, sem) is None


def test_gfx1250_vop3_carry_operands_use_wave32_mask_size():
    codegen = object.__new__(CodeGenerator)
    codegen.isa_spec = SimpleNamespace(profile=Cdna5Profile())
    sem = SimpleNamespace(semantic_class='vector_add_co')

    carry_out = SimpleNamespace(is_output=True, name='sdst', operand_type='OPR_SREG')
    carry_in = SimpleNamespace(is_output=False, name='src2', operand_type='OPR_SREG')
    source = SimpleNamespace(is_output=False, name='src0', operand_type='OPR_SRC')

    assert codegen._operand_size_override('ENC_VOP3', carry_out, sem) == '32'
    assert codegen._operand_size_override('ENC_VOP3', carry_in, sem) == '32'
    assert codegen._operand_size_override('ENC_VOP3', source, sem) is None
    assert codegen._operand_size_override('ENC_VOP2', carry_out, sem) is None


@pytest.mark.parametrize('semantic_class', ['vector_div_scale', 'vector_mad_64_32'])
def test_gfx1250_vop3_auxiliary_mask_outputs_use_wave32_size(semantic_class):
    codegen = object.__new__(CodeGenerator)
    codegen.isa_spec = SimpleNamespace(profile=Cdna5Profile())
    sem = SimpleNamespace(semantic_class=semantic_class)

    mask_out = SimpleNamespace(is_output=True, name='sdst', operand_type='OPR_SREG')
    vector_out = SimpleNamespace(is_output=True, name='vdst', operand_type='OPR_VGPR')

    assert codegen._operand_size_override('ENC_VOP3', mask_out, sem) == '32'
    assert codegen._operand_size_override('ENC_VOP3', vector_out, sem) is None
    assert codegen._operand_size_override('ENC_VOP2', mask_out, sem) is None


def test_ds_swizzle_generator_uses_addr_source_for_ds_and_vds():
    def body_for(enc_name: str) -> str:
        codegen = object.__new__(CodeGenerator)
        codegen.isa_spec = SimpleNamespace(
            arch_name='cdna5',
            profile=Cdna5Profile(),
            inst_encodings=[],
            encoding_map={},
        )
        inst = Instruction(
            'DS_SWIZZLE_B32',
            enc_name,
            0,
            [
                Operand('vdst', 32, 'OPR_VGPR', False, True, False, False, 0),
                Operand('addr', 32, 'OPR_VGPR', True, False, False, False, 1),
            ],
        )
        sem = InstructionSemantics('DS_SWIZZLE_B32', 'ds_swizzle')
        return codegen._gen_execute_body(inst, sem, enc_name)

    vds_body = body_for('ENC_VDS')
    ds_body = body_for('ENC_DS')

    assert 'src_data[i] = regs.read_lane(addr, i);' in vds_body
    assert 'src_data[i] = regs.read_lane(data0, i);' not in vds_body
    assert 'regs.write_lane(vdst, lane, src_data[src_lane]);' in vds_body
    assert 'src_data[i] = regs.read_lane(addr, i);' in ds_body
    assert 'src_data[i] = regs.read_lane(data0, i);' not in ds_body
    assert '2u * (lane & 0x3u)' in vds_body
    assert '2u * (lane & 0x3u)' in ds_body


def test_packed_16bit_source_gate_is_limited_to_e32_16bit_sources():
    codegen = object.__new__(CodeGenerator)
    codegen.isa_spec = SimpleNamespace(profile=Cdna5Profile())

    packed_src = SimpleNamespace(
        is_input=True, is_output=False, size=16, operand_type='OPR_SRC'
    )
    packed_vgpr_src = SimpleNamespace(
        is_input=True, is_output=False, size=16, operand_type='OPR_VGPR'
    )
    wide_src = SimpleNamespace(
        is_input=True, is_output=False, size=32, operand_type='OPR_SRC'
    )
    dst = SimpleNamespace(
        name='vdst', is_input=False, is_output=True, size=16, operand_type='OPR_VGPR'
    )

    assert codegen._operand_uses_packed_16bit_source('ENC_VOP1', packed_src)
    assert codegen._operand_uses_packed_16bit_source('ENC_VOP2', packed_src)
    assert codegen._operand_uses_packed_16bit_source('ENC_VOPC', packed_src)
    assert codegen._operand_uses_packed_16bit_source('ENC_VOPC', packed_vgpr_src)
    assert not codegen._operand_uses_packed_16bit_source('ENC_VOP3', packed_src)
    assert not codegen._operand_uses_packed_16bit_source('ENC_VOP1', wide_src)
    assert not codegen._operand_uses_packed_16bit_source('ENC_VOP1', dst)
    assert codegen._operand_uses_packed_16bit_source('ENC_VOP2', dst, reads_dst=True)
    assert codegen._operand_uses_packed_16bit_dst('ENC_VOP1', dst)
    assert codegen._operand_uses_packed_16bit_dst('ENC_VOP2', dst)
    assert not codegen._operand_uses_packed_16bit_dst('ENC_VOP3', dst)


def test_output_operand_read_facts_cover_liveness_sensitive_families():
    codegen = object.__new__(CodeGenerator)
    codegen.semantics = SimpleNamespace(
        instructions={
            'V_WRITELANE_B32': InstructionSemantics(
                'V_WRITELANE_B32', 'vector_writelane'
            ),
            'V_SWAP_B32': InstructionSemantics('V_SWAP_B32', 'vector_swap'),
            'BUFFER_ATOMIC_ADD': InstructionSemantics(
                'BUFFER_ATOMIC_ADD', 'buffer_atomic'
            ),
            'S_ADDK_I32': InstructionSemantics('S_ADDK_I32', 'scalar_addk'),
            'S_CMOVK_I32': InstructionSemantics('S_CMOVK_I32', 'scalar_cmovk'),
        }
    )

    vdst = Operand('vdst', 32, 'OPR_VGPR', False, True, False, False, 0)
    swap_src0 = Operand('src0', 32, 'OPR_SRC_VGPR', False, True, False, False, 1)
    vdata = Operand('vdata', 32, 'OPR_VGPR', False, True, False, False, 0)
    sdst = Operand('sdst', 32, 'OPR_SDST', False, True, False, False, 0)
    true16_vdst = Operand('vdst', 16, 'OPR_VGPR', False, True, False, False, 0)

    assert codegen._output_operand_is_also_source(
        Instruction('V_WRITELANE_B32', 'ENC_VOP3', 0, [vdst]), vdst
    )
    assert codegen._output_operand_is_also_source(
        Instruction('V_SWAP_B32', 'ENC_VOP3', 0, [vdst, swap_src0]), vdst
    )
    assert codegen._output_operand_is_also_source(
        Instruction('V_SWAP_B32', 'ENC_VOP3', 0, [vdst, swap_src0]), swap_src0
    )
    assert codegen._output_operand_is_also_source(
        Instruction('BUFFER_ATOMIC_ADD', 'ENC_MUBUF', 0, [vdata]), vdata
    )
    assert codegen._output_operand_is_also_source(
        Instruction('S_ADDK_I32', 'ENC_SOPK', 0, [sdst]), sdst
    )
    assert codegen._output_operand_is_also_source(
        Instruction('S_CMOVK_I32', 'ENC_SOPK', 0, [sdst]), sdst
    )
    # A sub-dword (true16) destination is a partial def: the old lane value
    # survives, so it is a read too. That read is surfaced through an
    # implicit_uses() override (see _partial_def_outputs), NOT by appending the
    # destination to src_operands_ — appending it would print the destination a
    # second time in disassembly and misrepresent the architectural sources.
    # _output_operand_is_also_source therefore returns False for this case.
    assert not codegen._output_operand_is_also_source(
        Instruction('V_ADD_F16', 'ENC_VOP3', 0, [true16_vdst]), true16_vdst
    )


def test_mnemonic_fallbacks_cover_unmodeled_read_write_outputs():
    codegen = object.__new__(CodeGenerator)
    codegen.semantics = SimpleNamespace(instructions={})

    vdst = Operand('vdst', 32, 'OPR_VGPR', False, True, False, False, 0)
    vdata = Operand('vdata', 32, 'OPR_VGPR', False, True, False, False, 0)
    sdata = Operand('sdata', 32, 'OPR_SREG', False, True, False, False, 0)

    for name in (
        'V_CVT_PKACCUM_U8_F32',
        'V_PK_FMAC_F16',
        'V_SMFMAC_F32_16X16X32_BF16',
        'V_DOT2C_F32_BF16',
    ):
        assert codegen._output_operand_is_also_source(
            Instruction(name, 'ENC_VOP3', 0, [vdst]), vdst
        )

    for name, opnd in (
        ('BUFFER_ATOMIC_ADD', vdata),
        ('S_ATOMIC_ADD', sdata),
        ('S_BUFFER_ATOMIC_ADD', sdata),
    ):
        assert codegen._output_operand_is_also_source(
            Instruction(name, 'ENC_UNKNOWN', 0, [opnd]), opnd
        )


def test_gfx1250_generated_vop2_uses_packed_16bit_vsrc1(
    gfx1250_generated_root: Path,
):
    vop2_cpp = (gfx1250_generated_root / 'vop2.cpp').read_text()

    assert 'vsrc1(16, OperandType::OPR_VGPR,' in vop2_cpp
    assert (
        'static_cast<unsigned short>(reinterpret_cast<const OpEncoding *>(inst)->vsrc1), true)'
        in vop2_cpp
    )


def test_gfx1250_generated_vop1_selects_packed_destination_overload(
    gfx1250_generated_root: Path,
):
    vop1_cpp = (gfx1250_generated_root / 'vop1.cpp').read_text()

    compact_ctor = ''.join(
        _generated_constructor_body(vop1_cpp, 'VCvtF16F32Vop1').split()
    )

    assert (
        'vdst(16,OperandType::OPR_VGPR,static_cast<unsignedshort>('
        'reinterpret_cast<constOpEncoding*>(inst)->vdst),false,true)' in compact_ctor
    )


def test_gfx1250_generated_vop2_fmac_f16_reads_packed_vdst(
    gfx1250_generated_root: Path,
):
    vop2_cpp = (gfx1250_generated_root / 'vop2.cpp').read_text()

    ctor = _generated_constructor_body(vop2_cpp, 'VFmacF16Vop2')

    assert 'vdst(16, OperandType::OPR_VGPR,' in ctor
    compact_ctor = ''.join(ctor.split())
    assert (
        'vdst(16,OperandType::OPR_VGPR,static_cast<unsignedshort>('
        'reinterpret_cast<constOpEncoding*>(inst)->vdst),true,true)' in compact_ctor
    )


def test_cdna3_generated_pk_fmac_f16_preserves_accumulator_source_order(
    amdgpu_generated_root: Path,
):
    vop2_cpp = (amdgpu_generated_root / 'cdna3' / 'vop2.cpp').read_text()
    ctor = _generated_constructor_body(vop2_cpp, 'VPkFmacF16Vop2')

    assert ctor.index('src_operands_[0] = &vdst;') < ctor.index(
        'src_operands_[1] = &src0;'
    )
    assert ctor.index('src_operands_[1] = &src0;') < ctor.index(
        'src_operands_[2] = &vsrc1;'
    )


def test_gfx1250_generated_high_vgpr_paths_use_logical_operands(
    amdgpu_generated_root: Path,
    gfx1250_generated_root: Path,
    execute_shared_path: Path,
):
    vop1 = (gfx1250_generated_root / 'vop1.cpp').read_text()
    vop1_exec = (gfx1250_generated_root / 'vop1_exec.cpp').read_text()
    vop2 = (gfx1250_generated_root / 'vop2.cpp').read_text()
    vop2_exec = (gfx1250_generated_root / 'vop2_exec.cpp').read_text()
    vds = (gfx1250_generated_root / 'vds.cpp').read_text()
    vds_exec = (gfx1250_generated_root / 'vds_exec.cpp').read_text()
    vglobal = (gfx1250_generated_root / 'vglobal.cpp').read_text()
    vopd = (gfx1250_generated_root / 'vopd.cpp').read_text()
    cdna4_vop2_exec = (amdgpu_generated_root / 'cdna4' / 'vop2_exec.cpp').read_text()
    shared = execute_shared_path.read_text()

    mov_b16 = _generated_method_body(vop1_exec, 'VMovB16Vop1', 'VMovB64Vop1')
    assert 'read_lane(src0, lane)' in mov_b16
    assert 'wf.vgpr_alloc().base + ((inst_.src0 - 256) & 0x7fu)' not in mov_b16

    fmac_f16 = _generated_method_body(vop2_exec, 'VFmacF16Vop2', 'VFmamkF16Vop2')
    assert 'read_lane(vdst, lane)' in fmac_f16
    assert 'base + (inst_.vdst & 0x7fu), lane)' not in fmac_f16

    add_co = _generated_method_body(cdna4_vop2_exec, 'VAddCoU32Vop2', 'VSubCoU32Vop2')
    assert 'execute_v_add_co_u32_vop2' in add_co

    for name, next_name in (
        ('ds_bpermute_b32_vds', 'ds_bpermute_fi_b32_vds'),
        ('ds_permute_b32_vds', 'ds_swizzle_b32_ds'),
        ('ds_swizzle_b32_vds', 'image_bvh_intersect_ray_mimg'),
    ):
        body = _shared_execute_body(shared, name, next_name)
        assert 'regs.read_lane(inst.addr' in body
        assert 'regs.write_lane(inst.vdst' in body
        assert 'vb + inst.inst_' not in body

    assert 'src_data[i] = regs.read_lane(inst.data0, i);' in shared
    assert (
        'd->dst_reg_base =\n'
        '      wf.vgpr_alloc().base +\n'
        '      *Isa::resolved_vgpr_offset(' in vds_exec
    )
    assert 'data0.set_vgpr_msb_role(amdgpu::VgprMsbRole::Src1);' in vds
    assert 'vsrc.set_vgpr_msb_role(amdgpu::VgprMsbRole::Src1);' in vglobal
    assert 'vdst.set_vgpr_msb_role(amdgpu::VgprMsbRole::Dst);' in vglobal
    assert 'vaddr.set_vgpr_msb_role(amdgpu::VgprMsbRole::Src0);' in vglobal
    for class_name in ('VFmamkF16Vop2', 'VFmamkF32Vop2', 'VFmamkF64Vop2'):
        constructor = _generated_constructor_body(vop2, class_name)
        assert 'vsrc1.set_vgpr_msb_role(amdgpu::VgprMsbRole::Src2);' in constructor
    assert 'src0.set_vgpr_msb_role(amdgpu::VgprMsbRole::Src0);' in vop1
    assert 'srcx1_.set_vgpr_msb_role(opx_ == kVopdFmamkF32' in vopd
    assert 'srcy1_.set_vgpr_msb_role(opy_ == kVopdFmamkF32' in vopd


def test_gfx1250_generated_vop3_mad_u16_uses_true16_helpers(
    gfx1250_generated_root: Path,
):
    vop3_ternary = (gfx1250_generated_root / 'vop3_exec_ternary.cpp').read_text()
    body = _generated_method_body(vop3_ternary, 'VMadU16Vop3', 'VXadU32Vop3')

    assert 'ROCJITSU_TRY_SIMD_VOP3_TERNARY_INT' not in body
    assert 'uint32_t opsel = ::rocjitsu::amdgpu::vop3_opsel(inst_);' in body
    assert 'read_vop3_true16_src(src0, wf, lane, opsel, 0)' in body
    assert 'read_vop3_true16_src(src1, wf, lane, opsel, 1)' in body
    assert 'read_vop3_true16_src(src2, wf, lane, opsel, 2)' in body
    assert 'write_vop3_true16_dst(vdst, wf, lane, opsel, result, true)' in body


def test_generated_special_vop3_true16_paths_use_selected_halves(
    execute_shared_path: Path,
):
    execute_shared = execute_shared_path.read_text()

    mad_u32 = _shared_execute_body(
        execute_shared, 'v_mad_u32_u16_vop3', 'v_mad_u32_u24_vop3'
    )
    assert 'ROCJITSU_TRY_SIMD_VOP3_TERNARY_TRUE16_SRC01' in mad_u32
    assert 'read_vop3_true16_src(inst.src0, wf, lane, opsel, 0)' in mad_u32
    assert 'read_vop3_true16_src(inst.src1, wf, lane, opsel, 1)' in mad_u32
    assert 'read_vop3_true16_src(inst.src2' not in mad_u32
    assert (
        'uint32_t s2 = amdgpu::RegisterAccess(wf).read_lane(inst.src2, lane);'
        in mad_u32
    )

    mad_i32 = _shared_execute_body(
        execute_shared, 'v_mad_i32_i16_vop3', 'v_mad_i32_i24_vop3'
    )
    assert 'ROCJITSU_TRY_SIMD_VOP3_TERNARY_TRUE16_SRC01' in mad_i32
    assert 'read_vop3_true16_src(inst.src0, wf, lane, opsel, 0)' in mad_i32
    assert 'read_vop3_true16_src(inst.src1, wf, lane, opsel, 1)' in mad_i32
    assert 'read_vop3_true16_src(inst.src2' not in mad_i32
    assert (
        'int32_t s2 = static_cast<int32_t>(amdgpu::RegisterAccess(wf).read_lane(inst.src2, lane));'
        in mad_i32
    )

    div_fixup = _shared_execute_body(
        execute_shared, 'v_div_fixup_f16_vop3', 'v_div_fixup_f32_vop3'
    )
    assert 'ROCJITSU_TRY_SIMD_VOP3_TERNARY_TRUE16_FP16' in div_fixup
    assert 'read_vop3_true16_src(inst.src0, wf, lane, opsel, 0)' in div_fixup
    assert 'read_vop3_true16_src(inst.src1, wf, lane, opsel, 1)' in div_fixup
    assert 'read_vop3_true16_src(inst.src2, wf, lane, opsel, 2)' in div_fixup
    assert (
        'write_vop3_true16_dst(inst.vdst, wf, lane, opsel, result_bits, true)'
        in div_fixup
    )

    pack = _shared_execute_body(
        execute_shared, 'v_pack_b32_f16_vop3', 'v_perm_b32_vop3'
    )
    assert 'ROCJITSU_TRY_SIMD_VOP3_BINARY_TRUE16_SRC' in pack
    assert 'read_vop3_true16_src(inst.src0, wf, lane, inst.inst_.op_sel, 0)' in pack
    assert 'read_vop3_true16_src(inst.src1, wf, lane, inst.inst_.op_sel, 1)' in pack


def test_cdna_generated_vop3_b16_i16_u16_paths_use_selected_halves(
    cdna4_generated_root: Path,
):
    vop3 = (cdna4_generated_root / 'vop3_exec.cpp').read_text()
    cases = [
        ('VLshlrevB16Vop3', 'VLshrrevB16Vop3', 2),
        ('VLshrrevB16Vop3', 'VAshrrevI16Vop3', 2),
        ('VAshrrevI16Vop3', 'VMaxF16Vop3', 2),
        ('VMadU16Vop3', 'VMadI16Vop3', 3),
        ('VMadI16Vop3', 'VFmaF16Vop3', 3),
    ]

    for class_name, next_class_name, src_count in cases:
        body = _generated_method_body(vop3, class_name, next_class_name)
        assert 'amdgpu::execute_v_' not in body
        assert 'uint32_t opsel = ::rocjitsu::amdgpu::vop3_opsel(inst_);' in body
        for src_idx in range(src_count):
            assert (
                f'read_vop3_true16_src(src{src_idx}, wf, lane, opsel, {src_idx})'
                in body
            )
        assert 'write_vop3_true16_dst(vdst, wf, lane, opsel,' in body


def test_generated_classified_alu_latches_trapsts_before_simd_return(
    execute_shared_path: Path,
):
    execute_shared = execute_shared_path.read_text()
    body = _shared_execute_body(execute_shared, 'v_mul_f32_vop3', 'v_mul_f64_vop2')

    classify_pos = body.index('uint32_t alu_causes = classify_mul_f32_vop3')
    trapsts_pos = body.index('wf.set_trapsts(wf.trapsts() | alu_causes);')
    simd_pos = body.index('ROCJITSU_TRY_SIMD_VOP3_BINARY_FP')
    assert classify_pos < trapsts_pos < simd_pos
    assert body.count('wf.set_trapsts(wf.trapsts() | alu_causes);') == 1


def test_generated_vop3_f16_alu_paths_split_shared_generic_from_true16(
    execute_shared_path: Path,
    gfx1250_generated_root: Path,
):
    execute_shared = execute_shared_path.read_text()
    gfx1250_vop3_alu = (gfx1250_generated_root / 'vop3_exec_alu.cpp').read_text()
    gfx1250_vop3_ternary = (
        gfx1250_generated_root / 'vop3_exec_ternary.cpp'
    ).read_text()

    unary = _shared_execute_body(execute_shared, 'v_ceil_f16_vop3', 'v_ceil_f32_vop1')
    assert 'ROCJITSU_TRY_SIMD_VOP3_UNARY_FP16' in unary
    assert 'ROCJITSU_TRY_SIMD_VOP3_UNARY_TRUE16_FP16' not in unary
    assert 'read_vop3_true16_src' not in unary
    assert 'write_vop3_true16_dst' not in unary

    binary = _shared_execute_body(execute_shared, 'v_add_f16_vop3', 'v_add_f32_vop2')
    assert 'ROCJITSU_TRY_SIMD_VOP3_BINARY_F16' in binary
    assert 'ROCJITSU_TRY_SIMD_VOP3_BINARY_TRUE16_F16' not in binary
    assert 'read_vop3_true16_src' not in binary
    assert 'write_vop3_true16_dst' not in binary

    ternary = _shared_execute_body(execute_shared, 'v_fma_f16_vop3', 'v_fma_f32_vop3')
    assert 'ROCJITSU_TRY_SIMD_VOP3_TERNARY_FP16' not in ternary
    assert 'fp_mode::fma_f16' in ternary
    assert 'read_vop3_true16_src' not in ternary
    assert 'write_vop3_true16_dst' not in ternary

    true16_unary = _generated_method_body(
        gfx1250_vop3_alu, 'VCeilF16Vop3', 'VTruncF16Vop3'
    )
    assert 'ROCJITSU_TRY_SIMD_VOP3_UNARY_TRUE16_FP16' in true16_unary
    assert (
        '[[maybe_unused]] uint32_t opsel = amdgpu::vop3_opsel(inst_);' in true16_unary
    )
    assert 'read_vop3_true16_src(src0, wf, lane, opsel, 0)' in true16_unary
    assert 'write_vop3_true16_dst(vdst, wf, lane, opsel,' in true16_unary
    assert 'inst.vdst.write_lane' not in true16_unary

    true16_binary = _generated_method_body(
        gfx1250_vop3_alu, 'VAddF16Vop3', 'VSubF16Vop3'
    )
    assert 'ROCJITSU_TRY_SIMD_VOP3_BINARY_TRUE16_F16' in true16_binary
    assert (
        '[[maybe_unused]] uint32_t opsel = amdgpu::vop3_opsel(inst_);' in true16_binary
    )
    assert 'read_vop3_true16_src(src0, wf, lane, opsel, 0)' in true16_binary
    assert re.search(
        r'read_vop3_true16_src\(src1,\s*wf,\s*lane,\s*opsel,\s*1\)',
        true16_binary,
    )
    assert 'write_vop3_true16_dst(vdst, wf, lane, opsel,' in true16_binary
    assert 'inst.vdst.write_lane' not in true16_binary

    true16_ternary = _generated_method_body(
        gfx1250_vop3_ternary, 'VFmaF16Vop3', 'VMin3I16Vop3'
    )
    assert 'ROCJITSU_TRY_SIMD_VOP3_TERNARY_TRUE16_FP16' not in true16_ternary
    assert 'fp_mode::fma_f16' in true16_ternary
    assert 'uint32_t opsel = ::rocjitsu::amdgpu::vop3_opsel(inst_);' in true16_ternary
    assert 'read_vop3_true16_src(src0, wf, lane, opsel, 0)' in true16_ternary
    assert 'read_vop3_true16_src(src1, wf, lane, opsel, 1)' in true16_ternary
    assert 'read_vop3_true16_src(src2, wf, lane, opsel, 2)' in true16_ternary
    assert 'write_vop3_true16_dst(vdst, wf, lane, opsel,' in true16_ternary
    assert 'inst.vdst.write_lane' not in true16_ternary


def test_generated_pseudo_scalar_vop3_paths_ignore_exec_and_f16_opsel(
    execute_shared_path: Path,
    gfx1250_generated_root: Path,
    rdna4_generated_root: Path,
):
    execute_shared = execute_shared_path.read_text()
    shared_cases = [
        ('v_s_exp_f32_vop3', 'v_s_log_f32_vop3'),
        ('v_s_log_f32_vop3', 'v_s_rcp_f32_vop3'),
        ('v_s_rcp_f32_vop3', 'v_s_rsq_f32_vop3'),
        ('v_s_rsq_f32_vop3', 'v_s_sqrt_f32_vop3'),
        ('v_s_sqrt_f32_vop3', 'v_sad_hi_u8_vop3'),
    ]
    for name, next_name in shared_cases:
        body = _shared_execute_body(execute_shared, name, next_name)
        assert 'wf.exec()' not in body
        assert 'if (exec != 0)' not in body
        assert 'amdgpu::RegisterAccess(wf).write_scalar(' in body
        assert 'amdgpu::pseudo_scalar::execute_f32(' in body
        assert 'wf.fp_round_mode_f32()' in body
        assert 'wf.fp_denorm_mode_f32()' in body

    f16_generated_cases = [
        ('VSExpF16Vop3', 'VSLogF32Vop3'),
        ('VSLogF16Vop3', 'VSRcpF32Vop3'),
        ('VSRcpF16Vop3', 'VSRsqF32Vop3'),
        ('VSRsqF16Vop3', 'VSSqrtF32Vop3'),
        ('VSSqrtF16Vop3', 'VAddNcU16Vop3'),
    ]
    for generated_root, source_name in (
        (gfx1250_generated_root, 'vop3_exec_alu.cpp'),
        (rdna4_generated_root, 'vop3_exec.cpp'),
    ):
        source = (generated_root / source_name).read_text()
        constructor_source = (
            generated_root
            / (
                'vop3_alu.cpp'
                if generated_root == gfx1250_generated_root
                else 'vop3.cpp'
            )
        ).read_text()
        for class_name, next_class_name in f16_generated_cases:
            body = _generated_method_body(source, class_name, next_class_name)
            assert 'if (exec != 0)' not in body
            assert 'vop3_opsel' not in body
            assert 'read_vop3_true16_src' not in body
            assert '>> 16' not in body
            assert (
                'static_cast<uint16_t>('
                'amdgpu::RegisterAccess(wf).read_scalar(src0))' in body
            )
            assert 'amdgpu::RegisterAccess(wf).write_scalar(' in body
            assert 'amdgpu::pseudo_scalar::execute_f16(' in body
            assert 'wf.fp_round_mode_f16_f64()' in body
            assert 'wf.fp_denorm_mode_f16_f64()' in body

            constructor = _generated_constructor_body(constructor_source, class_name)
            assert 'vop3_opsel' not in constructor
            assert 'simm32 & 0xFFFFu' in constructor

        pseudo_scalar_constructors = [
            'VSExpF32Vop3',
            'VSExpF16Vop3',
            'VSLogF32Vop3',
            'VSLogF16Vop3',
            'VSRcpF32Vop3',
            'VSRcpF16Vop3',
            'VSRsqF32Vop3',
            'VSRsqF16Vop3',
            'VSSqrtF32Vop3',
            'VSSqrtF16Vop3',
        ]
        for class_name in pseudo_scalar_constructors:
            decode_body = _generated_decode_body(constructor_source, class_name)
            assert 'OpSelSdstExec::OPR_SDST_EXEC_EXEC_LO' in decode_body
            assert 'has an invalid SReg_32_XEXEC destination' in decode_body
            assert 'may not use VCC as a destination' not in decode_body
            assert decode_body.index(
                'has an invalid SReg_32_XEXEC destination'
            ) < decode_body.index('does not support DPP')
            assert '->opsel' not in decode_body
            assert 'vop3_opsel' not in decode_body
        assert constructor_source.count(
            'has an invalid SReg_32_XEXEC destination'
        ) == len(pseudo_scalar_constructors)

        generic_true16_constructor = _generated_constructor_body(
            constructor_source, 'VAddF16Vop3'
        )
        assert 'amdgpu::vop3_opsel(inst_)' in generic_true16_constructor
        assert 'may not use VCC as a destination' not in generic_true16_constructor

    ordinary_exp = _shared_execute_body(
        execute_shared, 'v_exp_f32_vop1', 'v_exp_f32_vop3'
    )
    assert 'amdgpu::transcendental::exp_f32(' in ordinary_exp
    assert 'amdgpu::pseudo_scalar::' not in ordinary_exp


def test_generated_scalar_f16_arithmetic_does_not_consume_fp16_ovfl(
    execute_shared_path: Path,
):
    execute_shared = execute_shared_path.read_text()
    body = _shared_execute_body(execute_shared, 's_add_f16_sop2', 's_add_f32_sop2')

    assert 'util::f32_to_f16(result)' in body
    assert 'wf.fp16_ovfl()' not in body


def test_generated_vector_f16_arithmetic_consumes_fp16_ovfl(
    execute_shared_path: Path,
):
    execute_shared = execute_shared_path.read_text()
    vop2 = _shared_execute_body(execute_shared, 'v_add_f16_vop2', 'v_add_f16_vop3')
    vop3 = _shared_execute_body(execute_shared, 'v_add_f16_vop3', 'v_add_f32_vop2')

    assert 'if (wf.fp16_ovfl())' in vop2
    assert 'f32_to_f16_ovfl_simd' in vop2
    assert 'util::f32_to_f16_mode' in vop2
    assert 'wf.fp16_ovfl()' in vop2
    assert 'if (wf.fp16_ovfl())' in vop3
    assert 'f32_to_f16_ovfl_simd' in vop3
    assert 'util::f32_to_f16_mode' in vop3
    assert 'wf.fp16_ovfl()' in vop3


def test_local_true16_vop3_probe_uses_scoped_dpp_binding(tmp_path):
    args = SimpleNamespace(
        isafiles=[f'rdna4:{_mrisa_dir() / "amdgpu_isa_rdna4.xml"}'],
        gen_isas=True,
        gen_dbt=False,
        isa_output=str(tmp_path),
        dbt_output=None,
    )

    _run(args)

    rdna4_vop3 = (tmp_path / 'rdna4' / 'vop3_exec.cpp').read_text()
    ceil_body = _generated_function_body(rdna4_vop3, 'void VCeilF16Vop3::execute_impl')
    ceil_modifier_body = _generated_function_body(
        rdna4_vop3, 'RJ_NOINLINE void VCeilF16Vop3::execute_modifier_impl'
    )

    assert 'ROCJITSU_TRY_SIMD_VOP3_UNARY_TRUE16_FP16' in ceil_body
    assert 'read_vop3_true16_src(src0, wf, lane, opsel, 0)' in ceil_body
    assert 'write_vop3_true16_dst(vdst, wf, lane, opsel,' in ceil_body
    assert 'inst_.src0 != amdgpu::SRC_DPP' not in ceil_body
    assert 'src0.clear_delegate();' not in ceil_body
    assert ceil_body.index(
        'ROCJITSU_TRY_SIMD_VOP3_UNARY_TRUE16_FP16'
    ) < ceil_body.index('read_vop3_true16_src(src0, wf, lane, opsel, 0)')
    assert 'ScopedVgprWriteMask dpp_write_mask_scope_' in ceil_modifier_body
    assert 'ScopedOperandDelegate dpp_src0_binding_' in ceil_modifier_body
    assert 'ROCJITSU_TRY_SIMD_VOP3_UNARY_TRUE16_FP16' in ceil_modifier_body
    assert ceil_modifier_body.index(
        'ScopedOperandDelegate dpp_src0_binding_'
    ) < ceil_modifier_body.index('ROCJITSU_TRY_SIMD_VOP3_UNARY_TRUE16_FP16')


def test_single_isa_cdna1_sources_preprocess_with_source_includes(
    tmp_path, monkeypatch, rocjitsu_source_root: Path
):
    compiler = shutil.which('c++')
    if compiler is None:
        pytest.skip('C++ preprocessor is unavailable')

    monkeypatch.chdir(tmp_path)
    isa_output = Path('generated')
    generated_root = tmp_path / isa_output
    args = SimpleNamespace(
        isafiles=[f'cdna1:{_mrisa_dir() / "amdgpu_isa_cdna1.xml"}'],
        gen_isas=True,
        gen_dbt=False,
        isa_output=str(isa_output),
        dbt_output=None,
    )

    _run(args)

    shared_root = generated_root / 'shared'
    assert (shared_root / 'isa_properties.h').is_file()
    assert not (shared_root / 'execute_shared.h').exists()

    simd_glue_include = '#include "rocjitsu/isa/arch/amdgpu/shared/simd_glue.h"'
    for source_name in ('vop3_exec.cpp', 'vop3p_exec.cpp'):
        source = (generated_root / 'cdna1' / source_name).read_text()
        assert source.count(simd_glue_include) == 1
        assert 'shared/execute_shared.h' not in source

    include_roots = (
        rocjitsu_source_root / 'lib' / 'rocjitsu' / 'src',
        rocjitsu_source_root / 'lib' / 'rocjitsu' / 'include',
        rocjitsu_source_root / 'lib' / 'rocjitsu' / 'external_headers' / 'hsa_headers',
        rocjitsu_source_root / 'lib' / 'util' / 'include',
        rocjitsu_source_root / 'lib' / 'simdojo' / 'include',
    )
    subprocess.run(
        [
            compiler,
            '-E',
            *(flag for root in include_roots for flag in ('-I', str(root))),
            str(generated_root / 'cdna1' / 'vop3p_exec.cpp'),
            '-o',
            os.devnull,
        ],
        check=True,
        capture_output=True,
        text=True,
    )


def test_custom_identity_preprocesses_with_profile_handwritten_includes(
    tmp_path, rocjitsu_source_root: Path
):
    compiler = shutil.which('c++')
    if compiler is None:
        pytest.skip('C++ preprocessor is unavailable')

    args = SimpleNamespace(
        isafiles=[f'gfx1250:{_mrisa_dir() / "amdgpu_isa_cdna5.xml"}'],
        gen_isas=True,
        gen_dbt=False,
        isa_output=str(tmp_path),
        dbt_output=None,
    )

    _run(args)

    generated_root = tmp_path / 'gfx1250'
    vop3p_header = (generated_root / 'vop3p.h').read_text()
    vop3p_source = (generated_root / 'vop3p_exec.cpp').read_text()
    vglobal_source = (generated_root / 'vglobal_exec.cpp').read_text()
    assert '#include "rocjitsu/isa/arch/amdgpu/cdna5/isa.h"' in vop3p_header
    assert '#include "rocjitsu/isa/arch/amdgpu/cdna5/mma_exec.h"' in vop3p_source
    assert '#include "rocjitsu/isa/arch/amdgpu/cdna5/addr_calc.h"' in vglobal_source

    include_roots = (
        rocjitsu_source_root / 'lib' / 'rocjitsu' / 'src',
        rocjitsu_source_root / 'lib' / 'rocjitsu' / 'include',
        rocjitsu_source_root / 'lib' / 'rocjitsu' / 'external_headers' / 'hsa_headers',
        rocjitsu_source_root / 'lib' / 'util' / 'include',
        rocjitsu_source_root / 'lib' / 'simdojo' / 'include',
    )
    for source_path in (
        generated_root / 'vop3p_exec.cpp',
        generated_root / 'vglobal_exec.cpp',
    ):
        subprocess.run(
            [
                compiler,
                '-E',
                *(flag for root in include_roots for flag in ('-I', str(root))),
                str(source_path),
                '-o',
                os.devnull,
            ],
            check=True,
            capture_output=True,
            text=True,
        )


def test_single_isa_skips_dbt_generation(tmp_path, capsys):
    args = SimpleNamespace(
        isafiles=[f'cdna1:{_mrisa_dir() / "amdgpu_isa_cdna1.xml"}'],
        gen_isas=False,
        gen_dbt=True,
        isa_output=str(tmp_path),
        dbt_output=None,
    )

    _run(args)

    assert (
        'Skipping DBT generation: at least two ISAs are required.'
        in capsys.readouterr().err
    )
    assert not list(tmp_path.iterdir())


def test_multi_isa_dbt_output_defaults_to_isa_output(tmp_path):
    args = SimpleNamespace(
        isafiles=[
            f'rdna3_5:{_mrisa_dir() / "amdgpu_isa_rdna3_5.xml"}',
            f'rdna4:{_mrisa_dir() / "amdgpu_isa_rdna4.xml"}',
        ],
        gen_isas=False,
        gen_dbt=True,
        isa_output=str(tmp_path),
        dbt_output=None,
    )

    _run(args)

    assert (tmp_path / 'legalization_rdna3_5_to_rdna4.h').is_file()
    assert (tmp_path / 'encoding_fields.h').is_file()


@pytest.mark.parametrize('gen_isas', [False, True])
def test_encoding_translator_uses_selected_generated_include_tree(tmp_path, gen_isas):
    include_root = tmp_path / 'include'
    isa_output = include_root / 'custom' / 'generated'
    dbt_output = tmp_path / 'dbt'
    args = SimpleNamespace(
        isafiles=[
            f'cdna4:{_mrisa_dir() / "amdgpu_isa_cdna4.xml"}',
            f'rdna4:{_mrisa_dir() / "amdgpu_isa_rdna4.xml"}',
        ],
        gen_isas=gen_isas,
        gen_dbt=True,
        isa_output=str(isa_output),
        include_root=str(include_root),
        dbt_output=str(dbt_output),
    )

    _run(args)

    header = (dbt_output / 'encoding_cdna4_to_rdna4.h').read_text()
    assert '#include "custom/generated/cdna4/machine_insts.h"' in header
    assert '#include "custom/generated/rdna4/builders.h"' in header
    assert '#include "custom/generated/rdna4/machine_insts.h"' in header


def test_dbt_generation_canonicalizes_rdna35_cli_alias(tmp_path):
    generated_tables = []
    for output_name, alias in (
        ('underscore', 'rdna3_5'),
        ('documented', 'rdna3.5'),
    ):
        dbt_output = tmp_path / output_name
        args = SimpleNamespace(
            isafiles=[
                f'{alias}:{_mrisa_dir() / "amdgpu_isa_rdna3_5.xml"}',
                f'rdna4:{_mrisa_dir() / "amdgpu_isa_rdna4.xml"}',
            ],
            gen_isas=False,
            gen_dbt=True,
            isa_output=None,
            dbt_output=str(dbt_output),
        )

        _run(args)

        generated_tables.append(
            (dbt_output / 'legalization_rdna3_5_to_rdna4.h').read_text()
        )

    assert generated_tables[0] == generated_tables[1]


def test_single_isa_cndmask_qualifies_amdgpu_src_modifier():
    spec = Parser(str(_mrisa_dir() / 'amdgpu_isa_rdna4.xml'), Rdna4Profile()).parse()
    semantics = derive_all_semantics(spec)
    generator = CodeGenerator(spec, '', semantics)
    vop3 = next(
        encoding for encoding in spec.inst_encodings if encoding.enc_name == 'ENC_VOP3'
    )
    cndmask = next(inst for inst in vop3.insts if inst.name == 'V_CNDMASK_B32')

    body = generator._gen_execute_body(
        cndmask, semantics.instructions[cndmask.name], vop3.enc_name
    )

    assert body.count('amdgpu::apply_vop3_b32_src_mod(') == 2
    assert re.search(r'(?<!amdgpu::)apply_vop3_b32_src_mod\(', body) is None


def test_rdna4_64bit_literal_widening_is_format_specific(tmp_path):
    args = SimpleNamespace(
        isafiles=[f'rdna4:{_mrisa_dir() / "amdgpu_isa_rdna4.xml"}'],
        gen_isas=True,
        gen_dbt=False,
        isa_output=str(tmp_path),
        dbt_output=None,
    )

    _run(args)

    vop3 = (tmp_path / 'rdna4' / 'vop3.cpp').read_text()
    signed_ctor = _generated_constructor_body(vop3, 'VCmpLtI64Vop3')
    unsigned_ctor = _generated_constructor_body(vop3, 'VCmpLtU64Vop3')
    operand_cpp = (tmp_path / 'rdna4' / 'operand.cpp').read_text()
    operand_exec_cpp = (tmp_path / 'rdna4' / 'operand_exec.cpp').read_text()

    assert 'Operand::make_literal32' in signed_ctor
    assert 'Operand::Literal32Widening::SignExtend' in signed_ctor
    assert 'static_cast<uint32_t>' in signed_ctor
    assert 'Operand::Literal32Widening::ZeroExtend' in unsigned_ctor
    assert 'if (literal32_widening_)' in operand_exec_cpp
    assert 'return widened_literal32_value();' in operand_exec_cpp
    assert 'case Literal32Widening::Replicate32:' in operand_cpp
    assert 'if (has_literal64_)' in operand_exec_cpp
    assert 'return literal64_value_;' in operand_exec_cpp
    assert 'uint64_t signed_literal32_value_' not in operand_cpp
    assert 'static_cast<uint32_t>(encoding_value_)' in operand_cpp
    assert (
        'std::format("0x{:x}", static_cast<uint32_t>(encoding_value_))' in operand_cpp
    )


def test_generated_literal32_widening_shapes_cover_gfx1250_and_cdna_vopc(
    gfx1250_generated_root: Path,
    tmp_path: Path,
):
    args = SimpleNamespace(
        isafiles=[f'cdna4:{_mrisa_dir() / "amdgpu_isa_cdna4.xml"}'],
        gen_isas=True,
        gen_dbt=False,
        isa_output=str(tmp_path),
        dbt_output=None,
    )
    _run(args)

    gfx_operand_exec = (gfx1250_generated_root / 'operand_exec.cpp').read_text()
    gfx_vop3_alu = _generated_split_model_source(gfx1250_generated_root, 'vop3_alu')
    gfx_vop3_data = (gfx1250_generated_root / 'vop3_data.cpp').read_text()
    gfx_vop3_ternary = (gfx1250_generated_root / 'vop3_ternary.cpp').read_text()
    gfx_vop3p = (gfx1250_generated_root / 'vop3p.cpp').read_text()
    cdna_vopc = (tmp_path / 'cdna4' / 'vopc.cpp').read_text()

    assert gfx_operand_exec.count('if (literal32_widening_)') == 2
    assert gfx_operand_exec.count('return widened_literal32_value();') == 2

    mixed_ctor = _generated_constructor_body(gfx_vop3_ternary, 'VMadNcI64I32Vop3')
    assert mixed_ctor.count('Operand::make_literal32') == 1
    assert 'src2 = Operand::make_literal32' in mixed_ctor
    assert 'Operand::Literal32Widening::SignExtend' in mixed_ctor

    packed_ctor = _generated_constructor_body(gfx_vop3p, 'VPkAddF32Vop3p')
    assert packed_ctor.count('Operand::make_literal32') == 2
    assert 'Operand::Literal32Widening::Replicate32' in packed_ctor

    cdna_vopc_ctor = _generated_constructor_body(cdna_vopc, 'VCmpLtI64Vopc')
    assert 'src0 = Operand::make_literal32' in cdna_vopc_ctor
    assert 'Operand::Literal32Widening::SignExtend' in cdna_vopc_ctor

    scalar_mask_ctors = (
        _generated_constructor_body(gfx_vop3_data, 'VCndmaskB32Vop3'),
        _generated_constructor_body(gfx_vop3_data, 'VCndmaskB16Vop3'),
        _generated_constructor_body(gfx_vop3_alu, 'VAddCoCiU32Vop3SdstEnc'),
        _generated_constructor_body(gfx_vop3_alu, 'VSubCoCiU32Vop3SdstEnc'),
        _generated_constructor_body(gfx_vop3_alu, 'VSubrevCoCiU32Vop3SdstEnc'),
    )
    for constructor in scalar_mask_ctors:
        assert 'src2 = Operand::make_literal32' not in constructor
        assert 'src2 = Operand(32, OperandType::OPR_SIMM64' not in constructor


def test_generated_rdna4_local_vop3_pack_paths_use_selected_halves(
    rdna4_generated_root: Path,
):
    rdna4_vop3 = (rdna4_generated_root / 'vop3_exec.cpp').read_text()

    def local_body(class_name: str, next_class_name: str) -> str:
        return _generated_method_body(rdna4_vop3, class_name, next_class_name)

    pack = local_body('VPackB32F16Vop3', 'VCvtPkNormI16F16Vop3')
    assert 'read_vop3_true16_src(src0, wf, lane, inst_.opsel, 0)' in pack
    assert 'read_vop3_true16_src(src1, wf, lane, inst_.opsel, 1)' in pack

    pknorm = local_body('VCvtPkNormI16F16Vop3', 'VCvtPkNormU16F16Vop3')
    assert 'read_vop3_true16_src(src0, wf, lane, inst_.opsel, 0)' in pknorm
    assert 'read_vop3_true16_src(src1, wf, lane, inst_.opsel, 1)' in pknorm
    assert 'float s0 = std::bit_cast<float>' not in pknorm
    assert 'auto cvt_i16 = [](float f) -> int16_t {' in pknorm


def test_gfx1250_generated_vop3_lshrrev_b16_uses_true16_helpers(
    gfx1250_generated_root: Path,
):
    vop3_alu = (gfx1250_generated_root / 'vop3_exec_alu.cpp').read_text()
    body = _generated_method_body(vop3_alu, 'VLshrrevB16Vop3', 'VAshrrevI16Vop3')

    assert 'uint32_t opsel = ::rocjitsu::amdgpu::vop3_opsel(inst_);' in body
    assert 'read_vop3_true16_src(src0, wf, lane, opsel, 0)' in body
    assert 'read_vop3_true16_src(src1, wf, lane, opsel, 1)' in body
    assert 'write_vop3_true16_dst(vdst, wf, lane, opsel, result, true)' in body
    assert 'inst.vdst.write_lane' not in body


def test_gfx1250_generated_vop3_auxiliary_masks_are_wave32(
    gfx1250_generated_root: Path,
):
    source = _generated_split_model_source(gfx1250_generated_root, 'vop3_alu')

    for class_name in (
        'VDivScaleF32Vop3SdstEnc',
        'VDivScaleF64Vop3SdstEnc',
        'VMadCoU64U32Vop3SdstEnc',
        'VMadCoI64I32Vop3SdstEnc',
    ):
        ctor = _generated_constructor_body(source, class_name)
        assert 'sdst(32, OperandType::OPR_SREG' in ctor


def test_generated_operand_validates_scalar_register_selector_intervals(
    amdgpu_generated_root: Path,
):
    for arch in ('rdna1', 'rdna2'):
        operand = ''.join(
            (amdgpu_generated_root / arch / 'operand.cpp').read_text().split()
        )
        assert (
            'caseOperandType::OPR_SREG:'
            'if(!((encoding_value>=0&&encoding_value<=123)||'
            '(encoding_value>=125&&encoding_value<=125))' in operand
        )
        assert (
            'defer_encoding_error(EncodingError::InvalidScalarRegisterSelector);'
            in operand
        )


def test_generated_operand_validates_scalar_register_selector_variants(
    gfx1250_generated_root: Path, rdna4_generated_root: Path
):
    gfx1250_operand = ''.join(
        (gfx1250_generated_root / 'operand.cpp').read_text().split()
    )
    assert 'caseOperandType::OPR_SREG_M0:' in gfx1250_operand

    rdna4_operand = ''.join((rdna4_generated_root / 'operand.cpp').read_text().split())
    assert (
        'caseOperandType::OPR_SREG_LITERAL:'
        'if(!((encoding_value>=0&&encoding_value<=124)||'
        '(encoding_value>=255&&encoding_value<=255))' in rdna4_operand
    )


def test_generated_operand_validates_scalar_selector_families(
    amdgpu_generated_root: Path,
):
    operand = ''.join(
        (amdgpu_generated_root / 'cdna1' / 'operand.cpp').read_text().split()
    )

    assert (
        'caseOperandType::OPR_SDST:'
        'if(!((encoding_value>=0&&encoding_value<=124)||'
        '(encoding_value>=126&&encoding_value<=127))' in operand
    )
    assert (
        'caseOperandType::OPR_SSRC_NOLIT:'
        'if(!((encoding_value>=0&&encoding_value<=124)||'
        '(encoding_value>=126&&encoding_value<=208)||'
        '(encoding_value>=235&&encoding_value<=248)||'
        '(encoding_value>=251&&encoding_value<=253))' in operand
    )
    assert 'defer_encoding_error(EncodingError::InvalidSelector);' in operand
    assert (
        'defer_encoding_error(EncodingError::InvalidScalarSourceSelector);' in operand
    )


def test_generated_operand_validates_barrier_id_selectors(
    gfx1250_generated_root: Path, rdna4_generated_root: Path
):
    for root, negative_max in (
        (gfx1250_generated_root, 196),
        (rdna4_generated_root, 194),
    ):
        operand = ''.join((root / 'operand.cpp').read_text().split())
        assert 'caseOperandType::OPR_SSRC_BARRIER_ID:' in operand
        assert '(encoding_value>=125&&encoding_value<=125)' in operand
        assert '(encoding_value>=128&&encoding_value<=159)' in operand
        assert f'(encoding_value>=193&&encoding_value<={negative_max})' in operand

        test_encodings = (root / 'test_encodings.h').read_text()
        assert '"s_barrier_signal_isfirst", {0xBE804F80U' in test_encodings

    gfx1250_test_encodings = (gfx1250_generated_root / 'test_encodings.h').read_text()
    assert '"s_get_barrier_state", {0xBE805080U' in gfx1250_test_encodings


def test_gfx1250_barrier_init_reads_member_count_from_implicit_m0(
    gfx1250_generated_root: Path,
):
    sop1_exec = (gfx1250_generated_root / 'sop1_exec.cpp').read_text()
    body = _generated_method_body(sop1_exec, 'SBarrierInitSop1', 'SBarrierJoinSop1')

    assert 'read_scalar(ssrc0)' in body
    assert 'uint32_t member_source = wf.m0();' in body
    assert 'member_count = (member_source >> 16) & 0x7fu' in body


def test_generated_operand_validates_direct_selector_namespaces(
    amdgpu_generated_root: Path, gfx1250_generated_root: Path
):
    gfx1250_operand = ''.join(
        (gfx1250_generated_root / 'operand.cpp').read_text().split()
    )
    assert 'caseOperandType::OPR_SRC:' in gfx1250_operand
    assert '(encoding_value>=209&&encoding_value<=229)' not in gfx1250_operand
    assert 'caseOperandType::OPR_SRC_VGPR_OR_INLINE:' in gfx1250_operand
    assert '(encoding_value>=128&&encoding_value<=208)' in gfx1250_operand
    assert '(encoding_value>=240&&encoding_value<=248)' in gfx1250_operand
    assert '(encoding_value>=256&&encoding_value<=511)' in gfx1250_operand

    cdna1_operand = ''.join(
        (amdgpu_generated_root / 'cdna1' / 'operand.cpp').read_text().split()
    )
    assert (
        'caseOperandType::OPR_SMEM_OFFSET:'
        'if(!((encoding_value>=0&&encoding_value<=124))' in cdna1_operand
    )


def test_generated_restricted_operands_do_not_apply_literal_fixups(
    gfx1250_generated_root: Path,
):
    vop3_data = (gfx1250_generated_root / 'vop3_data.cpp').read_text()
    vop3p = (gfx1250_generated_root / 'vop3p.cpp').read_text()
    sop1 = (gfx1250_generated_root / 'sop1.cpp').read_text()

    readfirstlane = _generated_constructor_body(vop3_data, 'VReadfirstlaneB32Vop3')
    readlane = _generated_constructor_body(vop3_data, 'VReadlaneB32Vop3')
    wmma = _generated_constructor_body(vop3p, 'VWmmaF3216x16x128F8f6f4Vop3p')
    generic_accumulator_wmma = _generated_constructor_body(
        vop3p, 'VWmmaF3216x16x128Fp8Fp8Vop3p'
    )
    barrier = _generated_constructor_body(sop1, 'SBarrierSignalIsfirstSop1')

    for body in (readfirstlane, readlane, wmma, barrier):
        assert 'OperandType::OPR_SIMM32' not in body
        assert 'OperandType::OPR_SIMM64' not in body
    assert 'src0 == 255' not in readfirstlane
    assert 'src0 == 254' not in readfirstlane
    assert 'src0 == 255' not in readlane
    assert 'src0 == 254' not in readlane
    assert 'src1 == 255' not in readlane
    assert 'src1 == 254' not in readlane
    assert 'src2 == 255' not in wmma
    assert 'src2 == 254' not in wmma
    assert 'OperandType::OPR_SRC' in generic_accumulator_wmma
    generic_accumulator_decode = _generated_decode_body(
        vop3p, 'VWmmaF3216x16x128Fp8Fp8Vop3p'
    )
    assert 'has an invalid accumulator selector' in generic_accumulator_decode
    assert 'src2 == 255' not in generic_accumulator_wmma
    assert 'src2 == 254' not in generic_accumulator_wmma
    assert 'OperandType::OPR_SIMM32' not in generic_accumulator_wmma
    assert 'OperandType::OPR_SIMM64' not in generic_accumulator_wmma
    assert 'ssrc0 == 255' not in barrier
    assert 'ssrc0 == 254' not in barrier


def test_gfx1250_generated_operand_validates_lane_selectors(
    gfx1250_generated_root: Path,
):
    operand_types = (gfx1250_generated_root / 'operand_types.h').read_text()
    operand = (gfx1250_generated_root / 'operand.cpp').read_text()

    assert 'OPR_SSRC_LANESEL_POS_INT_MAX = 191' in operand_types
    assert '#include "util/except.h"' not in operand
    assert 'case OperandType::OPR_SSRC_LANESEL:' in operand
    assert 'EncodingError::InvalidLaneSelector' in operand


def test_gfx1250_generated_operand_rejects_invalid_exec_selector(
    gfx1250_generated_root: Path,
):
    operand = (gfx1250_generated_root / 'operand.cpp').read_text()
    assert '#include "util/except.h"' not in operand
    assert 'case OperandType::OPR_EXEC:' in operand
    assert '(encoding_value >= 126 && encoding_value <= 126)' in operand
    assert 'EncodingError::InvalidExecSelector' in operand


def test_gfx1250_generated_vop3_add_f16_applies_dpp(
    gfx1250_generated_root: Path,
):
    encodings_h = (gfx1250_generated_root / 'encodings.h').read_text()
    encodings_cpp = (gfx1250_generated_root / 'encodings.cpp').read_text()
    vop3_alu = '\n'.join(
        path.read_text()
        for path in sorted(gfx1250_generated_root.glob('vop3_alu*.cpp'))
    )
    vop3_exec_alu = (gfx1250_generated_root / 'vop3_exec_alu.cpp').read_text()

    vop3_base = encodings_h[
        encodings_h.index('class Vop3') : encodings_h.index('class Vop3p')
    ]
    assert 'uint32_t dpp_ctrl_ = 0;' in vop3_base
    assert 'uint32_t dpp_fi_ = 1;' in vop3_base
    assert 'std::array<uint32_t, 3> raw_words_{};' in vop3_base

    vop3_encoding_ctor = _generated_constructor_body(encodings_cpp, 'Vop3')
    vop3_validation = vop3_base
    assert 'has_encoded_dpp()' in vop3_encoding_ctor
    assert 'has_encoded_dpp8()' in vop3_encoding_ctor
    assert 'DPP and literal operands cannot be combined' in vop3_validation
    assert 'inst_.src0 == amdgpu::SRC_DPP' in vop3_validation
    assert 'size_ += sizeof(MachineInst);' in vop3_encoding_ctor
    assert 'std::memcpy(raw_words_.data(), inst, size_);' in vop3_encoding_ctor
    assert 'raw_encoding_ = raw_words_.data();' in vop3_encoding_ctor

    decode = _generated_decode_body(vop3_alu, 'VAddF16Vop3')
    assert 'Result validation =' in decode
    assert 'Vop3::validate_encoding(' in decode
    assert decode.index('validate_encoding(') < decode.index('validation.failed()')
    assert decode.index('validation.failed()') < decode.index('std::make_unique')

    ctor = _generated_constructor_body(vop3_alu, 'VAddF16Vop3')
    assert 'Vop3VopDpp16MachineInst' in ctor
    assert 'dpp_ctrl_ = dp->dpp_ctrl;' in ctor
    assert 'dpp_fi_ = dp->fi;' in ctor
    assert 'src_dpp8_fi' in ctor

    body = _generated_method_body(vop3_exec_alu, 'VAddF16Vop3', 'VAddNcU16Vop3')
    assert 'dpp_bound_ctrl_, dpp_fi_' in body
    assert 'apply_dpp8(src0, dpp8_lane_sel_, dpp_fi_' in body
    assert 'ScopedOperandDelegate dpp_src0_binding_(src0,' in body
    assert 'dpp_src0_ ? &*dpp_src0_ : nullptr);' in body
    assert 'src0.set_delegate(' not in body
    assert 'src0.clear_delegate();' not in body


def test_gfx1250_generator_wires_integer_clamp_only_from_encoded_vop3_field():
    isa_xml = _mrisa_dir() / 'amdgpu_isa_cdna5.xml'
    parser = Parser(str(isa_xml), Cdna5Profile())
    spec = parser.parse()
    semantics = derive_all_semantics(spec)
    generator = CodeGenerator(spec, '', semantics)

    def generated_body(name: str, enc_name: str, *, result_writer=None) -> str:
        enc = next(enc for enc in spec.inst_encodings if enc.enc_name == enc_name)
        inst = next(inst for inst in enc.insts if inst.name == name)
        generator._current_inst_fields = {field.name for field in enc.ucode_fields}
        generator._current_operand_names = {operand.name for operand in inst.operands}
        generator._current_enc = enc
        return generator._gen_execute_body(
            inst,
            semantics.instructions[name],
            enc.enc_name,
            result_writer=result_writer,
        )

    vop3 = generated_body('V_ADD_NC_U32', 'ENC_VOP3')
    assert 'vop3_integer_add<uint32_t>' in vop3
    assert 'inst_.clamp' in vop3

    subrev = generated_body('V_SUBREV_NC_U32', 'ENC_VOP3')
    assert 'vop3_integer_sub<uint32_t>' in subrev
    assert subrev.index('read_lane(src1, lane)') < subrev.index('read_lane(src0, lane)')
    assert 'inst_.clamp' in subrev

    vop2 = generated_body('V_ADD_NC_U32', 'ENC_VOP2')
    assert 'vop3_integer_add' not in vop2
    assert 'inst_.clamp' not in vop2

    carry = generated_body(
        'V_ADD_CO_U32', 'VOP3_SDST_ENC', result_writer='commit_result'
    )
    assert 'inst_.clamp && w > 0xFFFFFFFFULL ? UINT32_MAX' in carry
    assert 'inst_.clamp' in carry
    assert 'if (w > 0xFFFFFFFFULL) vcc |=' in carry


def test_noop_format_validation_is_inherited(amdgpu_generated_root: Path):
    encodings_h = (amdgpu_generated_root / 'rdna4' / 'encodings.h').read_text()

    sopp_start = encodings_h.index('class Sopp ')
    sopp = encodings_h[sopp_start : encodings_h.index('\n};', sopp_start)]
    assert 'validate_encoding' not in sopp

    vop3_start = encodings_h.index('class Vop3 ')
    vop3 = encodings_h[vop3_start : encodings_h.index('\n};', vop3_start)]
    assert 'validate_encoding' in vop3


def test_generated_dpp8_disassembly_uses_encoding_state(
    amdgpu_generated_root: Path,
) -> None:
    for arch in ('rdna1', 'rdna2', 'rdna3', 'rdna3_5', 'rdna4', 'cdna5'):
        arch_root = amdgpu_generated_root / _generated_dir_name(arch)
        encodings_cpp = (arch_root / 'encodings.cpp').read_text()
        encodings_h = (arch_root / 'encodings.h').read_text()
        generated_cpp = '\n'.join(path.read_text() for path in arch_root.glob('*.cpp'))

        vop1_modifiers = _generated_inline_method_body(
            encodings_h, 'Vop1', 'void build_modifiers'
        )

        assert 'append_dpp8_disassembly' in vop1_modifiers
        assert 'auto *inst = &inst_;' not in vop1_modifiers
        for class_name in ('Vop1', 'Vop2', 'Vopc', 'Vop3', 'Vop3p', 'Vop3SdstEnc'):
            class_match = re.search(
                rf'class {class_name}\b.*?\n}};', encodings_h, flags=re.DOTALL
            )
            assert class_match is not None
            assert 'owned_mnemonic_' not in class_match.group()
        assert 'dpp8_mnemonic' not in generated_cpp
        assert 'void Vop1::append_mnemonic(std::string &out) const' in encodings_cpp
        assert 'has_encoded_dpp() || has_encoded_dpp8()' in encodings_cpp
        assert 'out += "_dpp";' in encodings_cpp
        if arch not in ('rdna1', 'rdna2'):
            assert 'void Vop3::append_mnemonic(std::string &out) const' in encodings_cpp
            assert 'out += "_e64_dpp";' in encodings_cpp


def test_gfx1250_generated_vop3_rejects_literal64_selectors(
    gfx1250_generated_root: Path,
):
    vop1 = (gfx1250_generated_root / 'vop1.cpp').read_text()
    vop3_alu = (gfx1250_generated_root / 'vop3_alu.cpp').read_text()
    vop3p = (gfx1250_generated_root / 'vop3p.cpp').read_text()

    vop1_ctor = _generated_constructor_body(vop1, 'VSqrtF16Vop1')
    assert 'src0 == 254' in vop1_ctor
    assert 'OperandType::OPR_SIMM64' in vop1_ctor

    vop3_decode = _generated_decode_body(vop3_alu, 'VSqrtF16Vop3')
    assert 'LiteralSupport::Literal32, 1' in vop3_decode
    vop3_ctor = _generated_constructor_body(vop3_alu, 'VSqrtF16Vop3')
    assert 'OperandType::OPR_SIMM64' not in vop3_ctor

    vop3p_decode = _generated_decode_body(vop3p, 'VFmaMixF32Vop3p')
    assert 'LiteralSupport::Literal32' in vop3p_decode
    vop3p_ctor = _generated_constructor_body(vop3p, 'VFmaMixF32Vop3p')
    assert 'OperandType::OPR_SIMM64' not in vop3p_ctor


def test_gfx1250_generated_literal_validation(gfx1250_generated_root: Path):
    encodings = (gfx1250_generated_root / 'encodings.cpp').read_text()
    sop1 = (gfx1250_generated_root / 'sop1.cpp').read_text()
    sop2 = (gfx1250_generated_root / 'sop2.cpp').read_text()
    vopd = (gfx1250_generated_root / 'vopd.cpp').read_text()

    sop1_encoding_ctor = _generated_constructor_body(encodings, 'Sop1')
    assert 'has_encoded_literal32()' in sop1_encoding_ctor
    sop1_literal_helper = _generated_bool_method_body(
        encodings, 'Sop1', 'has_encoded_literal32'
    )
    assert 'case 76:' not in sop1_literal_helper
    assert 'case 77:' not in sop1_literal_helper

    barrier_decode = _generated_decode_body(sop1, 'SBarrierSignalSop1')
    assert 'LiteralSupport::None' in barrier_decode
    barrier_ctor = _generated_constructor_body(sop1, 'SBarrierSignalSop1')
    assert 'OperandType::OPR_SIMM32' not in barrier_ctor
    assert 'OperandType::OPR_SIMM64' not in barrier_ctor

    sop2_decode = _generated_decode_body(sop2, 'SAddCoU32Sop2')
    assert 'may not mix 32-bit and 64-bit literals' in sop2_decode
    assert vopd.count('VOPD does not support 64-bit literals') == 1
    assert 'srcx0 == 254 || srcx0 == 255 || srcy0 == 254 || srcy0 == 255' in vopd
    assert 'VOPD3 does not support literal selectors' in vopd


@pytest.mark.parametrize('arch', ['rdna3', 'cdna5'])
def test_generated_sendmsg_return_selectors_are_not_literals(
    amdgpu_generated_root: Path,
    arch: str,
):
    generated_root = amdgpu_generated_root / _generated_dir_name(arch)
    encodings = (generated_root / 'encodings.cpp').read_text()
    sop1 = (generated_root / 'sop1.cpp').read_text()

    sop1_encoding_ctor = _generated_constructor_body(encodings, 'Sop1')
    assert 'has_encoded_literal32()' in sop1_encoding_ctor
    sop1_literal_helper = _generated_bool_method_body(
        encodings, 'Sop1', 'has_encoded_literal32'
    )
    assert 'case 76:' not in sop1_literal_helper
    assert 'case 77:' not in sop1_literal_helper

    for class_name in ('SSendmsgRtnB32Sop1', 'SSendmsgRtnB64Sop1'):
        sendmsg_ctor = _generated_constructor_body(sop1, class_name)
        assert 'OperandType::OPR_SIMM32' not in sendmsg_ctor
        assert 'OperandType::OPR_SIMM64' not in sendmsg_ctor
        if arch == 'cdna5':
            sendmsg_decode = _generated_decode_body(sop1, class_name)
            assert 'LiteralSupport::None, 0' in sendmsg_decode


def test_generated_rdna3_5_sendmsg_return_uses_symbolic_disassembly(
    amdgpu_generated_root: Path,
):
    operand = (
        amdgpu_generated_root / _generated_dir_name('rdna3_5') / 'operand.cpp'
    ).read_text()
    start = operand.index('case OperandType::OPR_SENDMSG_RTN:')
    end = operand.index('case OperandType::OPR_SIMM16:', start)
    sendmsg_return = operand[start:end]

    assert 'return "sendmsg(MSG_RTN_GET_DOORBELL)";' in sendmsg_return
    assert 'return std::format("sendmsg({}, 0, 0)", value);' in sendmsg_return


def test_rdna3_5_disassembly_overrides_do_not_change_rdna3():
    rdna3 = Rdna3Profile()
    rdna3_5 = Rdna3_5Profile()

    assert not rdna3.renders_gfx11_image_syntax
    assert not rdna3.split_ds_2addr_offsets
    assert not rdna3.vop3p_absolute_source_instructions
    assert not rdna3.sendmsg_return_symbolic
    assert rdna3_5.renders_gfx11_image_syntax
    assert rdna3_5.split_ds_2addr_offsets
    assert rdna3_5.vop3p_absolute_source_instructions == frozenset(
        {'V_FMA_MIX_F32', 'V_FMA_MIXLO_F16', 'V_FMA_MIXHI_F16'}
    )
    assert rdna3_5.gfx11_mimg_gather_style_instructions == frozenset(
        {'IMAGE_MSAA_LOAD'}
    )
    assert rdna3_5.gfx11_mimg_fixed_vdata_words == {
        'IMAGE_BVH_INTERSECT_RAY': 4,
        'IMAGE_BVH64_INTERSECT_RAY': 4,
    }
    assert rdna3_5.gfx11_mimg_fixed_vaddr_words == {
        'IMAGE_BVH_INTERSECT_RAY': (11, 8),
        'IMAGE_BVH64_INTERSECT_RAY': (12, 9),
    }
    assert rdna3_5.gfx11_mimg_nsa_group_words == {
        'IMAGE_BVH_INTERSECT_RAY': ((1, 1, 3, 3, 3), (1, 1, 3, 3)),
        'IMAGE_BVH64_INTERSECT_RAY': ((2, 1, 3, 3, 3), (2, 1, 3, 3)),
    }
    assert rdna3_5.sendmsg_return_symbolic


def test_generated_rdna3_5_fuzz_disassembly_rules(amdgpu_generated_root: Path):
    generated_root = amdgpu_generated_root / _generated_dir_name('rdna3_5')
    encodings = (generated_root / 'encodings.cpp').read_text()
    encodings_h = (generated_root / 'encodings.h').read_text()
    encoding_model = encodings_h + '\n' + encodings
    mimg = (generated_root / 'mimg.cpp').read_text()
    vop1 = (generated_root / 'vop1.cpp').read_text()

    assert 'if (!omits_gfx11_mimg_dim_dmask())' in encoding_model
    assert 'out += "0123456789abcdef"[inst->dmask & 0xfu];' in encoding_model
    assert 'nsa_vaddr_operand_ = vaddr;' in encoding_model
    assert 'operand != nsa_vaddr_operand_' in encoding_model
    assert 'gfx11_mimg_nsa_group_width' in encoding_model
    assert 'const Operand *nsa_vaddr_operand_ = nullptr;' in encodings_h
    assert '(reinterpret_cast<const OpEncoding *>(inst)->srsrc * 4)' in mimg
    assert '(reinterpret_cast<const OpEncoding *>(inst)->ssamp * 4)' in mimg
    assert 'mimg_vdata_bits(reinterpret_cast<const OpEncoding *>(inst)' in mimg
    assert 'mimg_vaddr_bits(reinterpret_cast<const OpEncoding *>(inst)' in mimg
    assert 'mimg_vdata_bits(reinterpret_cast<const OpEncoding *>(inst), true)' in mimg
    assert 'vdata(128, OperandType::OPR_VGPR' in _generated_constructor_body(
        mimg, 'ImageBvhIntersectRayMimg'
    )
    assert '(reinterpret_cast<const OpEncoding *>(inst)->a16 ? 256 : 352)' in (
        _generated_constructor_body(mimg, 'ImageBvhIntersectRayMimg')
    )
    assert '(reinterpret_cast<const OpEncoding *>(inst)->a16 ? 288 : 384)' in (
        _generated_constructor_body(mimg, 'ImageBvh64IntersectRayMimg')
    )
    assert 'capture_nsa_words(inst, &vaddr);' in mimg
    assert 'omit_repeated_destination_sources_ = true;' in _generated_constructor_body(
        mimg, 'ImageAtomicSubMimg'
    )
    assert 'inst_.op >= 46 && inst_.op <= 47' in _generated_bool_method_body(
        encodings, 'Ds', 'uses_split_ds_offsets'
    )
    assert 'out += \'-\';' in _generated_inline_method_body(
        encodings_h, 'Vop3p', 'void append_src_operand'
    )
    assert 'out += "lit(";' in _generated_inline_method_body(
        encodings_h, 'Vop3p', 'void append_src_operand'
    )
    assert 'inst_.op == 26' in _generated_inline_method_body(
        encodings_h, 'Vop3p', 'void build_modifiers'
    )
    v_swap_ctor = _generated_constructor_body(vop1, 'VSwapB16Vop1')
    assert (
        '"v_swap_b16_dpp"' in v_swap_ctor
        and ': "v_swap_b16"' in v_swap_ctor
        and 'omit_repeated_destination_sources_ = true;' in v_swap_ctor
        and 'OperandType::OPR_VGPR' in v_swap_ctor
        and 'reinterpret_cast<const OpEncoding *>(inst)->src0 & 0xffu)' in v_swap_ctor
        and 'true, true);' in v_swap_ctor
        and '& 0x7fu' not in v_swap_ctor
    )


def test_gfx1250_compact_literal_policy_precedes_extension_sizing(
    gfx1250_generated_root: Path,
):
    encodings_h = (gfx1250_generated_root / 'encodings.h').read_text()
    vop1 = (gfx1250_generated_root / 'vop1.cpp').read_text()
    vop2 = (gfx1250_generated_root / 'vop2.cpp').read_text()

    assert 'enum class LiteralSupport : uint8_t' in encodings_h
    assert 'LiteralSupport::Both' in encodings_h

    for class_name in ('Vop1', 'Vop2', 'Vopc'):
        start = encodings_h.index(f'class {class_name} ')
        body = encodings_h[start : encodings_h.index('\n};', start)]
        assert 'supports_literal(literal_support, LiteralSupport::Literal32)' in body
        assert 'supports_literal(literal_support, LiteralSupport::Literal64)' in body
        assert 'static Result validate_encoding' in body

    readfirstlane = _generated_decode_body(vop1, 'VReadfirstlaneB32Vop1')
    assert 'LiteralSupport::None' in readfirstlane
    assert readfirstlane.index('validate_encoding(') < readfirstlane.index(
        'validation.failed()'
    )
    assert readfirstlane.index('validation.failed()') < readfirstlane.index(
        'std::make_unique'
    )

    fmamk_f64 = _generated_decode_body(vop2, 'VFmamkF64Vop2')
    assert 'LiteralSupport::Literal64' in fmamk_f64
    assert fmamk_f64.index('validate_encoding(') < fmamk_f64.index(
        'validation.failed()'
    )
    assert fmamk_f64.index('validation.failed()') < fmamk_f64.index('std::make_unique')
    fmamk_f32 = _generated_decode_body(vop2, 'VFmamkF32Vop2')
    assert 'LiteralSupport::Literal32' in fmamk_f32


def test_generated_dpp_disassembly_uses_encoding_state(
    amdgpu_generated_root: Path,
) -> None:
    for arch in ('cdna1', 'cdna2', 'cdna3', 'cdna4'):
        generated_root = amdgpu_generated_root / arch
        encodings_cpp = (generated_root / 'encodings.cpp').read_text()
        encodings_h = (generated_root / 'encodings.h').read_text()
        modifiers = _generated_inline_method_body(
            encodings_h, 'Vop1', 'void build_modifiers'
        )
        assert 'void Vop1::append_mnemonic(std::string &out) const' in encodings_cpp
        assert 'mnemonic_.ends_with("_e32")' in encodings_cpp
        assert 'out += "_dpp";' in encodings_cpp
        assert 'append_dpp16_disassembly' in modifiers
        assert 'dpp_bound_ctrl_, dpp_fi_, false,' in modifiers
        assert 'amdgpu::dpp::DppCtrlDialect::Gfx9' in modifiers

    for arch in ('rdna1', 'rdna2', 'rdna3', 'rdna3_5', 'rdna4', 'gfx1250'):
        generated_root = amdgpu_generated_root / _generated_dir_name(arch)
        encodings_cpp = (generated_root / 'encodings.cpp').read_text()
        encodings_h = (generated_root / 'encodings.h').read_text()
        modifiers = _generated_inline_method_body(
            encodings_h, 'Vop1', 'void build_modifiers'
        )
        assert 'void Vop1::append_mnemonic(std::string &out) const' in encodings_cpp
        assert 'mnemonic_.ends_with("_e32")' in encodings_cpp
        assert 'out += "_dpp";' in encodings_cpp
        assert 'append_dpp16_disassembly' in modifiers
        assert 'dpp_bound_ctrl_, dpp_fi_, true,' in modifiers
        assert 'amdgpu::dpp::DppCtrlDialect::Gfx10Plus' in modifiers
        assert 'append_dpp8_disassembly' in modifiers


@pytest.mark.parametrize(
    'arch',
    [
        'cdna1',
        'cdna2',
        'cdna3',
        'cdna4',
        'rdna1',
        'rdna2',
        'rdna3',
        'rdna3_5',
        'rdna4',
        'gfx1250',
    ],
)
def test_generated_encoding_rendering_overrides_are_inline(
    amdgpu_generated_root: Path,
    arch: str,
) -> None:
    generated_root = amdgpu_generated_root / _generated_dir_name(arch)
    encodings_h = (generated_root / 'encodings.h').read_text()
    encodings_cpp = (generated_root / 'encodings.cpp').read_text()

    scoped_override = re.compile(
        r'^void \w+::(?:build_modifiers|append_src_operand|append_dst_operand)\(',
        flags=re.MULTILINE,
    )
    assert scoped_override.search(encodings_cpp) is None
    assert 'void build_modifiers(std::string &out) const override {' in encodings_h

    if 'amdgpu::sdwa::append_' in encodings_h:
        assert (
            '#include "rocjitsu/isa/arch/amdgpu/shared/dpp_sdwa_ops.h"' in encodings_h
        )
    if 'amdgpu::append_gfx12_cache_policy' in encodings_h:
        assert (
            '#include "rocjitsu/isa/arch/amdgpu/shared/gfx12_cache_flags.h"'
            in encodings_h
        )


@pytest.mark.parametrize('arch', ['cdna4', 'rdna4', 'gfx1250'])
def test_generated_vop3p_disassembly_uses_encoding_state(
    amdgpu_generated_root: Path,
    arch: str,
) -> None:
    generated_root = amdgpu_generated_root / _generated_dir_name(arch)
    encodings_h = (generated_root / 'encodings.h').read_text()
    encodings_cpp = (generated_root / 'encodings.cpp').read_text()
    vop3p_modifiers = _generated_inline_method_body(
        encodings_h, 'Vop3p', 'void build_modifiers'
    )
    assert 'append_vop3p_disassembly' in vop3p_modifiers
    assert 'vop3p_encoded_source_count()' in vop3p_modifiers
    assert 'inst_.op' in vop3p_modifiers
    assert 'void Vop3p::build_modifiers' not in encodings_cpp
    if arch == 'gfx1250':
        assert 'inst->opsel_hi | (inst->opsel_hi_2 << 2)' in vop3p_modifiers


def test_cdna3_accumulator_moves_omit_packed_source_modifiers(
    amdgpu_generated_root: Path,
) -> None:
    generated_root = amdgpu_generated_root / 'cdna3'
    encodings_h = (generated_root / 'encodings.h').read_text()
    encodings_cpp = (generated_root / 'encodings.cpp').read_text()
    vop3p_modifiers = _generated_inline_method_body(
        encodings_h, 'Vop3p', 'void build_modifiers'
    )
    assert 'omits_vop3p_source_modifiers() ? 0 :' in vop3p_modifiers
    assert 'void Vop3p::build_modifiers' not in encodings_cpp

    profile = CdnaProfile()
    assert profile.vop3p_source_modifier_omissions == frozenset(
        {'V_ACCVGPR_READ', 'V_ACCVGPR_WRITE'}
    )


@pytest.mark.parametrize('arch', ['cdna4', 'rdna4'])
def test_generated_vop3_disassembly_uses_encoding_state(
    amdgpu_generated_root: Path,
    arch: str,
) -> None:
    generated_root = amdgpu_generated_root / arch
    encodings_h = (generated_root / 'encodings.h').read_text()
    encodings_cpp = (generated_root / 'encodings.cpp').read_text()
    vop3_modifiers = _generated_inline_method_body(
        encodings_h, 'Vop3', 'void build_modifiers'
    )
    vop3_src = _generated_inline_method_body(
        encodings_h, 'Vop3', 'void append_src_operand'
    )
    assert 'append_vop3_disassembly' in vop3_modifiers
    assert 'vop3_encoded_source_count()' in vop3_modifiers
    assert 'displays_vop3_op_sel()' in vop3_modifiers
    assert 'mnemonic_' not in vop3_modifiers
    assert 'append_vop3_operand' in vop3_src
    assert 'void Vop3::build_modifiers' not in encodings_cpp
    assert 'void Vop3::append_src_operand' not in encodings_cpp
    assert 'void Vop3::append_dst_operand' not in encodings_cpp
    if arch == 'rdna4':
        assert 'const bool half_width = modifier_index >= 0 && true &&' in vop3_src


@pytest.mark.parametrize('arch', ['cdna4', 'rdna4', 'gfx1250'])
def test_generated_vop3_sdst_disassembly_uses_encoding_state(
    amdgpu_generated_root: Path,
    arch: str,
) -> None:
    generated_root = amdgpu_generated_root / _generated_dir_name(arch)
    encodings_h = (generated_root / 'encodings.h').read_text()
    encodings_cpp = (generated_root / 'encodings.cpp').read_text()
    modifiers = _generated_inline_method_body(
        encodings_h, 'Vop3SdstEnc', 'void build_modifiers'
    )
    src = _generated_inline_method_body(
        encodings_h, 'Vop3SdstEnc', 'void append_src_operand'
    )
    assert 'append_vop3_disassembly' in modifiers
    assert 'vop3_encoded_source_count()' in modifiers
    assert 'append_vop3_operand' in src
    assert 'void Vop3SdstEnc::build_modifiers' not in encodings_cpp
    assert 'void Vop3SdstEnc::append_src_operand' not in encodings_cpp
    assert 'void Vop3SdstEnc::append_dst_operand' not in encodings_cpp


@pytest.mark.parametrize('arch', ['rdna4', 'gfx1250'])
def test_gfx12_generated_cache_policy_disassembly(
    amdgpu_generated_root: Path,
    arch: str,
) -> None:
    generated_root = amdgpu_generated_root / _generated_dir_name(arch)
    encodings_h = (generated_root / 'encodings.h').read_text()
    encodings_cpp = (generated_root / 'encodings.cpp').read_text()
    assert (
        '#include "rocjitsu/isa/arch/amdgpu/shared/gfx12_cache_flags.h"'
        in encodings_cpp
    )
    assert 'amdgpu::Gfx12TemporalHintKind::Atomic' in encodings_h
    assert 'amdgpu::Gfx12TemporalHintKind::Store' in encodings_h
    assert (
        'amdgpu::append_gfx12_cache_policy(out, inst->th, inst->scope, hint_kind);'
        in encodings_h
    )
    vbuffer_modifiers = _generated_inline_method_body(
        encodings_h, 'Vbuffer', 'void build_modifiers'
    )
    assert vbuffer_modifiers.index('if (inst->offen)') < vbuffer_modifiers.index(
        'if (inst->idxen)'
    )
    assert vbuffer_modifiers.index('if (inst->idxen)') < vbuffer_modifiers.index(
        'if (inst->ioffset)'
    )
    assert vbuffer_modifiers.index('if (inst->ioffset)') < vbuffer_modifiers.index(
        'amdgpu::append_gfx12_cache_policy'
    )
    assert vbuffer_modifiers.index(
        'amdgpu::append_gfx12_cache_policy'
    ) < vbuffer_modifiers.index('if (inst->nv)')
    if arch == 'rdna4':
        for class_name in ('Vimage', 'Vsample'):
            body = _generated_inline_method_body(
                encodings_h, class_name, 'void build_modifiers'
            )
            assert 'append_gfx12_cache_policy' in body


def test_generated_sdwa_uses_shared_source_staging(
    amdgpu_generated_root: Path,
) -> None:
    checked_sdwa_files = 0
    for arch in ('cdna1', 'cdna2', 'cdna3', 'cdna4', 'rdna1', 'rdna2'):
        for filename in ('vop1.cpp', 'vop2.cpp', 'vopc.cpp'):
            path = amdgpu_generated_root / arch / filename
            assert path.exists(), f'missing generated file: {path}'
            generated = _execution_source_path(
                path, _profile_for_arch(arch)
            ).read_text()
            if 'amdgpu::SRC_SDWA' not in generated:
                continue
            checked_sdwa_files += 1
            assert 'amdgpu::sdwa::stage_source(' in generated
            assert 'sdwa_src_select(' not in generated
            assert 'std::make_unique<DppOperand>' not in generated
        encodings = (amdgpu_generated_root / arch / 'encodings.cpp').read_text()
        assert '#include "rocjitsu/isa/arch/amdgpu/shared/dpp_sdwa_ops.h"' in encodings
    assert checked_sdwa_files > 0


def test_cdna3_generated_disassembly_preserves_ds_and_flat_offsets(
    amdgpu_generated_root: Path,
) -> None:
    generated_root = amdgpu_generated_root / 'cdna3'
    encodings_h = (generated_root / 'encodings.h').read_text()
    encodings_cpp = (generated_root / 'encodings.cpp').read_text()

    ds_modifiers = _generated_inline_method_body(
        encodings_h, 'Ds', 'void build_modifiers'
    )
    assert 'uses_split_ds_offsets()' in ds_modifiers
    assert 'out += " offset0:"' in ds_modifiers
    assert 'out += " offset1:"' in ds_modifiers
    assert 'inst->offset0 | (inst->offset1 << 8)' in ds_modifiers
    assert 'out += " gds"' in ds_modifiers

    flat_modifiers = _generated_inline_method_body(
        encodings_h, 'Flat', 'void build_modifiers'
    )
    assert 'if (inst->seg == 0)' in flat_modifiers
    assert 'flat_offset & 0x1000' in flat_modifiers
    assert 'flat_offset -= 0x2000' in flat_modifiers
    assert 'void Ds::build_modifiers' not in encodings_cpp
    assert 'void Flat::build_modifiers' not in encodings_cpp


def test_generated_sdwa_scalar_destination_uses_explicit_lane_mask_write(
    cdna4_generated_root: Path,
) -> None:
    vopc = (cdna4_generated_root / 'vopc_exec.cpp').read_text()

    lane_mask_write = (
        'amdgpu::write_explicit_lane_mask(sb + sdwa_sdst_, wf, cmp_result);'
    )
    assert lane_mask_write in vopc
    assert 'write_sgpr(sb + sdwa_sdst_' not in vopc
    assert 'write_sgpr(sb + sdwa_sdst_ + 1' not in vopc


def test_generated_sdwa_uses_source_specific_modifier_formats(
    cdna4_generated_root: Path,
) -> None:
    vop1 = (cdna4_generated_root / 'vop1_exec.cpp').read_text()
    vop2 = (cdna4_generated_root / 'vop2_exec.cpp').read_text()
    vopc = (cdna4_generated_root / 'vopc_exec.cpp').read_text()

    cvt_f32_f16 = _generated_method_body(vop1, 'VCvtF32F16Vop1', 'VCvtRpiI32F32Vop1')
    assert 'SourceModifierFormat::F16' in cvt_f32_f16

    cvt_i32_f32 = _generated_method_body(vop1, 'VCvtI32F32Vop1', 'VCvtF16F32Vop1')
    assert 'SourceModifierFormat::F32' in cvt_i32_f32

    cvt_f32_bf16 = vop1[vop1.index('void VCvtF32Bf16Vop1::execute_impl') :]
    assert 'SourceModifierFormat::BF16' in cvt_f32_bf16

    add_f32 = _generated_method_body(vop2, 'VAddF32Vop2', 'VSubF32Vop2')
    assert add_f32.count('SourceModifierFormat::F32') == 2

    add_f16 = _generated_method_body(vop2, 'VAddF16Vop2', 'VSubF16Vop2')
    assert add_f16.count('SourceModifierFormat::F16') == 2

    pk_fmac_f16 = _generated_method_body(vop2, 'VPkFmacF16Vop2', 'VXnorB32Vop2')
    assert pk_fmac_f16.count('SourceModifierFormat::NONE') == 2
    assert 'stage_source(src0,' in pk_fmac_f16
    assert 'stage_source(vsrc1,' in pk_fmac_f16
    assert 'stage_source(vdst,' not in pk_fmac_f16

    ldexp_f16 = _generated_method_body(vop2, 'VLdexpF16Vop2', 'VAddU32Vop2')
    assert 'SourceModifierFormat::F16' in ldexp_f16
    assert 'SourceModifierFormat::NONE' in ldexp_f16

    cmp_class_f16 = _generated_method_body(
        vopc, 'VCmpClassF16Vopc', 'VCmpxClassF16Vopc'
    )
    assert 'SourceModifierFormat::F16' in cmp_class_f16
    assert 'SourceModifierFormat::NONE' in cmp_class_f16


def test_generated_operandless_vop_does_not_stage_missing_source(
    cdna4_generated_root: Path,
) -> None:
    vop1 = (cdna4_generated_root / 'vop1_exec.cpp').read_text()
    nop = _generated_method_body(vop1, 'VNopVop1', 'VMovB32Vop1')
    assert 'apply_dpp(' not in nop
    assert 'apply_dpp8(' not in nop
    assert 'stage_source(' not in nop
    assert 'src_operands_[0]' not in nop


@pytest.mark.parametrize(
    'arch,class_names',
    [
        ('cdna1', ('Vop1', 'Vopc', 'Vop2')),
        ('cdna2', ('Vop1', 'Vopc', 'Vop2')),
        ('cdna3', ('Vop1', 'Vopc', 'Vop2')),
        ('cdna4', ('Vop1', 'Vopc', 'Vop2')),
        ('rdna1', ('Vop1', 'Vop2')),
        ('rdna2', ('Vop1', 'Vop2')),
        ('rdna3', ('Vop1', 'Vopc', 'Vop2', 'Vop3', 'Vop3p', 'Vop3SdstEnc')),
        ('rdna3_5', ('Vop1', 'Vopc', 'Vop2', 'Vop3', 'Vop3p', 'Vop3SdstEnc')),
        ('rdna4', ('Vop1', 'Vopc', 'Vop2', 'Vop3', 'Vop3p', 'Vop3SdstEnc')),
        ('cdna5', ('Vop1', 'Vopc', 'Vop2', 'Vop3', 'Vop3p', 'Vop3SdstEnc')),
    ],
)
def test_generated_dpp_encodings_own_extension_words(
    amdgpu_generated_root: Path,
    arch: str,
    class_names: tuple[str, ...],
):
    arch_root = amdgpu_generated_root / _generated_dir_name(arch)
    encodings_h = (arch_root / 'encodings.h').read_text()
    encodings_cpp = (arch_root / 'encodings.cpp').read_text()

    for class_name in class_names:
        class_start = encodings_h.index(f'class {class_name} ')
        class_end = encodings_h.index('\n};', class_start)
        class_body = encodings_h[class_start:class_end]
        assert 'raw_words_' in class_body, f'{arch} {class_name}'

        constructor = _generated_constructor_body(encodings_cpp, class_name)
        assert 'std::memcpy(raw_words_.data(), inst, size_);' in constructor, (
            arch,
            class_name,
        )
        assert 'raw_encoding_ = raw_words_.data();' in constructor, (
            arch,
            class_name,
        )
        if class_name in ('Vop3', 'Vop3p', 'Vop3SdstEnc'):
            assert 'has_encoded_dpp()' in constructor, (
                arch,
                class_name,
            )
            assert 'has_encoded_dpp8()' in constructor, (
                arch,
                class_name,
            )
            assert 'size_ += sizeof(MachineInst);' in constructor, (
                arch,
                class_name,
            )


@pytest.mark.parametrize(
    'arch',
    [
        'cdna1',
        'cdna2',
        'cdna3',
        'cdna4',
        'rdna1',
        'rdna2',
        'rdna3',
        'rdna3_5',
        'rdna4',
        'cdna5',
    ],
)
def test_generated_scalar_literals_own_extension_words(
    amdgpu_generated_root: Path,
    arch: str,
):
    arch_root = amdgpu_generated_root / _generated_dir_name(arch)
    encodings_h = (arch_root / 'encodings.h').read_text()
    encodings_cpp = (arch_root / 'encodings.cpp').read_text()
    raw_word_counts = {
        'Sop1': 3 if arch == 'cdna5' else 2,
        'Sopc': 3 if arch == 'cdna5' else 2,
        'Sopk': 2,
        'Sop2': 3 if arch == 'cdna5' else 2,
    }

    for class_name, raw_word_count in raw_word_counts.items():
        class_start = encodings_h.index(f'class {class_name} ')
        class_end = encodings_h.index('\n};', class_start)
        class_body = encodings_h[class_start:class_end]
        assert (
            f'std::array<uint32_t, {raw_word_count}> raw_words_{{}};' in class_body
        ), (arch, class_name)

        constructor = _generated_constructor_body(encodings_cpp, class_name)
        assert 'std::memcpy(raw_words_.data(), inst, size_);' in constructor, (
            arch,
            class_name,
        )
        assert 'raw_encoding_ = raw_words_.data();' in constructor, (
            arch,
            class_name,
        )


@pytest.mark.parametrize('arch', ['rdna1', 'rdna2'])
def test_generated_early_rdna_vector_literals_own_extension_words(
    amdgpu_generated_root: Path,
    arch: str,
):
    encodings_h = (amdgpu_generated_root / arch / 'encodings.h').read_text()
    encodings_cpp = (amdgpu_generated_root / arch / 'encodings.cpp').read_text()

    for class_name, raw_word_count in {
        'Vopc': 2,
        'Vop3': 3,
        'Vop3p': 3,
        'Vop3SdstEnc': 3,
    }.items():
        class_start = encodings_h.index(f'class {class_name} ')
        class_end = encodings_h.index('\n};', class_start)
        class_body = encodings_h[class_start:class_end]
        assert (
            f'std::array<uint32_t, {raw_word_count}> raw_words_{{}};' in class_body
        ), (arch, class_name)

        constructor = _generated_constructor_body(encodings_cpp, class_name)
        assert 'std::memcpy(raw_words_.data(), inst, size_);' in constructor, (
            arch,
            class_name,
        )
        assert 'raw_encoding_ = raw_words_.data();' in constructor, (
            arch,
            class_name,
        )


def test_gfx1250_generated_vop3_owns_only_supported_extension_word(
    gfx1250_generated_root: Path,
):
    encodings_h = (gfx1250_generated_root / 'encodings.h').read_text()
    encodings_cpp = (gfx1250_generated_root / 'encodings.cpp').read_text()

    for class_name in ('Vop3', 'Vop3p'):
        class_start = encodings_h.index(f'class {class_name} ')
        class_end = encodings_h.index('\n};', class_start)
        class_body = encodings_h[class_start:class_end]
        assert 'std::array<uint32_t, 3> raw_words_{};' in class_body
        assert 'has_lit64' not in class_body

        constructor = _generated_constructor_body(encodings_cpp, class_name)
        assert 'size_ += sizeof(MachineInst);' in constructor
        assert '2 * sizeof(MachineInst)' not in constructor


def test_generated_dpp_staging_does_not_replace_architectural_operands(
    amdgpu_generated_root: Path,
):
    generated_cpp = '\n'.join(
        path.read_text() for path in sorted(amdgpu_generated_root.rglob('*.cpp'))
    )
    generated_headers = '\n'.join(
        path.read_text()
        for path in sorted(amdgpu_generated_root.rglob('*.h'))
        if path.parent.name != 'shared'
    )

    assert 'apply_dpp(src_operands_' not in generated_cpp
    assert 'apply_dpp8(src_operands_' not in generated_cpp
    assert 'src_operands_[0] = dpp_src0_.get();' not in generated_cpp
    assert 'apply_dpp(None' not in generated_cpp
    assert '.set_delegate(dpp_src' not in generated_cpp
    assert 'std::optional<StagedOperand> dpp_src0_;' not in generated_headers
    assert 'std::optional<StagedOperand> dpp_src1_;' not in generated_headers
    assert 'std::optional<StagedOperand> dpp_src0_;' in generated_cpp
    assert 'std::make_unique<StagedOperand>' not in generated_cpp
    assert 'ScopedOperandDelegate dpp_src0_binding_' in generated_cpp


def test_generated_modifier_storage_is_out_of_ordinary_vop_frames(
    amdgpu_generated_root: Path,
):
    cases = (
        ('rdna4', 'vop3_exec.cpp', 'VMovB32Vop3'),
        ('cdna1', 'vop2_exec.cpp', 'VAddF32Vop2'),
    )
    for arch, filename, class_name in cases:
        source = (amdgpu_generated_root / arch / filename).read_text()
        ordinary = _generated_function_body(source, f'void {class_name}::execute_impl')
        modified = _generated_function_body(
            source, f'RJ_NOINLINE void {class_name}::execute_modifier_impl'
        )

        assert 'execute_modifier_impl(wf);' in ordinary
        assert 'std::optional<StagedOperand>' not in ordinary
        assert 'DppPlan' not in ordinary
        assert 'ScopedOperandDelegate' not in ordinary
        assert 'ScopedVgprWriteMask' not in ordinary
        assert 'std::optional<StagedOperand>' in modified
        assert 'DppPlan' in modified
        assert 'ScopedOperandDelegate' in modified
        assert 'ScopedVgprWriteMask' in modified
        assert 'amdgpu::execute_' in ordinary
        assert 'amdgpu::execute_' in modified

        header = (
            amdgpu_generated_root / arch / filename.replace('_exec.cpp', '.h')
        ).read_text()
        class_start = header.index(f'class {class_name} ')
        class_end = header.index('\n};', class_start)
        class_body = header[class_start:class_end]
        assert 'private:' in class_body
        assert 'void execute_modifier_impl(amdgpu::Wavefront &wf);' in class_body


def test_generated_dpp_legality_checks_are_coalesced(
    amdgpu_generated_root: Path,
):
    generated_cpp = '\n'.join(
        path.read_text() for path in sorted(amdgpu_generated_root.rglob('*.cpp'))
    )
    assert 'DPP is not supported' not in generated_cpp

    cases = (
        ('rdna4', 'vop3.cpp', 'VMulLoU32Vop3'),
        ('rdna3', 'vopc.cpp', 'VCmpEqF64Vopc'),
        ('cdna1', 'vop1.cpp', 'VCvtI32F64Vop1'),
    )
    for arch, filename, class_name in cases:
        source = (amdgpu_generated_root / arch / filename).read_text()
        constructor = _generated_constructor_body(source, class_name)
        decode_body = _generated_decode_body(source, class_name)
        assert decode_body.count('does not support DPP') == 1, class_name
        assert 'does not support DPP' not in constructor, class_name

    vop1 = (amdgpu_generated_root / 'rdna4' / 'vop1.cpp').read_text()
    assert 'does not support DPP' not in _generated_decode_body(vop1, 'VNopVop1')


def test_generated_dpp_constructor_binding_is_emitted_once(
    amdgpu_generated_root: Path,
):
    marker = 'reinterpret_cast<const OpEncoding *>(inst)->src0 == amdgpu::SRC_DPP'
    cases = (
        ('cdna5', 'vop1.cpp', 'VMovB64Vop1', True),
        ('cdna5', 'vop2.cpp', 'VAddF64Vop2', True),
        ('cdna5', 'vop3_ternary.cpp', 'VFmaF64Vop3', True),
        ('rdna4', 'vop3.cpp', 'VMovB32Vop3', False),
    )
    for arch, filename, class_name, requires_feature in cases:
        source = (amdgpu_generated_root / arch / filename).read_text()
        constructor = _generated_constructor_body(source, class_name)
        assert constructor.count(marker) == 1, class_name
        assert ('required_isa_features_' in constructor) is requires_feature


def test_generated_optional_includes_have_direct_uses(amdgpu_generated_root: Path):
    unused = []
    for path in sorted(amdgpu_generated_root.rglob('*.cpp')):
        generated = path.read_text()
        if '#include <optional>' in generated and 'std::optional' not in generated:
            unused.append(str(path.relative_to(amdgpu_generated_root)))

    assert unused == []


def test_gfx1250_generated_vop1_dpp8_uses_src0_marker_for_fi(
    gfx1250_generated_root: Path,
):
    vop1 = (gfx1250_generated_root / 'vop1.cpp').read_text()
    vop1_exec = (gfx1250_generated_root / 'vop1_exec.cpp').read_text()

    ctor = _generated_constructor_body(vop1, 'VMovB32Vop1')
    assert 'src_dpp8_fi(reinterpret_cast<const OpEncoding *>(inst)->src0)' in ctor
    assert 'dp8->fi' not in ctor
    assert 'dpp_fi_ = dp->fi;' in ctor

    body = _generated_method_body(vop1_exec, 'VMovB32Vop1', 'VReadfirstlaneB32Vop1')
    assert 'dpp_bound_ctrl_, dpp_fi_' in body
    assert 'apply_dpp8(src0, dpp8_lane_sel_, dpp_fi_' in body


def test_cdna4_generated_vop1_sdwa_availability_is_instruction_specific(
    amdgpu_generated_root: Path,
):
    vop1 = (amdgpu_generated_root / 'cdna4' / 'vop1.cpp').read_text()
    encodings_h = (amdgpu_generated_root / 'cdna4' / 'encodings.h').read_text()

    supported = _generated_constructor_body(vop1, 'VMovB32Vop1')
    unsupported = _generated_decode_body(vop1, 'VCvtF64I32Vop1')

    assert 'reinterpret_cast<const Vop1VopSdwaMachineInst *>' in supported
    assert 'sdwa_mnemonic' not in vop1
    assert re.search(r'\? "v_floor_f32_sdwa"\s*: "v_floor_f32_e32"', vop1)
    vop1_class = re.search(r'class Vop1\b.*?\n};', encodings_h, flags=re.DOTALL)
    assert vop1_class is not None
    assert 'owned_mnemonic_' not in vop1_class.group()
    assert 'V_MOV_B32 does not support SDWA' not in supported
    assert 'emit_error.emit() << "V_CVT_F64_I32 does not support SDWA"' in unsupported


def test_gfx1250_generated_vop1_dpp8_availability_is_instruction_specific(
    gfx1250_generated_root: Path,
):
    vop1 = (gfx1250_generated_root / 'vop1.cpp').read_text()
    constructor = _generated_constructor_body(vop1, 'VCvtF64I32Vop1')
    decode_body = _generated_decode_body(vop1, 'VCvtF64I32Vop1')

    assert 'reinterpret_cast<const Vop1VopDpp16MachineInst *>' in constructor
    assert 'reinterpret_cast<const Vop1VopDpp8MachineInst *>' not in constructor
    assert 'emit_error.emit() << "V_CVT_F64_I32 does not support DPP8"' in decode_body


def test_rdna4_generated_vop3_dpp_availability_is_instruction_specific(
    rdna4_generated_root: Path,
):
    vop3 = (rdna4_generated_root / 'vop3.cpp').read_text()
    vop3p = (rdna4_generated_root / 'vop3p.cpp').read_text()

    supported = _generated_constructor_body(vop3, 'VAddF32Vop3')
    unsupported_vop3 = _generated_decode_body(vop3, 'VAddF64Vop3')
    unsupported_vop3p = _generated_decode_body(vop3p, 'VPkAddF16Vop3p')

    assert 'reinterpret_cast<const Vop3VopDpp16MachineInst *>' in supported
    assert 'reinterpret_cast<const Vop3VopDpp8MachineInst *>' in supported
    assert 'V_ADD_F32 does not support DPP' not in supported
    assert 'emit_error.emit() << "V_ADD_F64 does not support DPP"' in unsupported_vop3
    assert (
        'emit_error.emit() << "V_PK_ADD_F16 does not support DPP"' in unsupported_vop3p
    )


def test_gfx1250_packed_f32_execute_uses_local_simd_probe(
    gfx1250_generated_root: Path,
):
    source = (gfx1250_generated_root / 'vop3p_exec.cpp').read_text()
    assert 'literal64_value' not in source
    add_body = _generated_method_body(source, 'VPkAddF32Vop3p', 'VPkMulBf16Vop3p')
    mul_body = _generated_method_body(source, 'VPkMulF32Vop3p', 'VPkAddF32Vop3p')
    fma_body = _generated_method_body(source, 'VPkFmaF32Vop3p', 'VFmaMixF32Vop3p')

    for body in (add_body, mul_body):
        assert 'ROCJITSU_TRY_SIMD_VOP3P_PK_BINARY_F32_SELECTORS' in body
        assert 'inst_.opsel, inst_.opsel_hi' in body
        assert body.index('ROCJITSU_TRY_SIMD') < body.index('for (uint32_t lane')
        assert 'amdgpu::execute_v_pk_' not in body

    assert 'ROCJITSU_TRY_SIMD_VOP3P_PK_TERNARY_F32_SELECTORS' in fma_body
    assert 'inst_.opsel, inst_.opsel_hi, inst_.opsel_hi_2' in fma_body
    assert fma_body.index('ROCJITSU_TRY_SIMD') < fma_body.index('for (uint32_t lane')


def test_gfx1250_matrix_codegen_uses_public_opsel_hi_2_field(
    gfx1250_generated_root: Path,
):
    source = (gfx1250_generated_root / 'vop3p.cpp').read_text()

    assert 'pad_14' not in source
    assert 'inst_.opsel_hi_2' in source
    assert 'scale_inst_.opsel_hi_2' in source


def test_gfx1250_vop3p_rejects_unencoded_literal64_selectors(
    gfx1250_generated_root: Path,
):
    encodings_h = (gfx1250_generated_root / 'encodings.h').read_text()
    vop3p = (gfx1250_generated_root / 'vop3p.cpp').read_text()
    class_start = encodings_h.index('class Vop3p ')
    validation = encodings_h[class_start : encodings_h.index('\n};', class_start)]

    assert 'num_encoded_sources > 0 && inst_.src0 == 254' in validation
    assert 'num_encoded_sources > 1 && inst_.src1 == 254' in validation
    assert 'num_encoded_sources > 2 && inst_.src2 == 254' in validation
    assert 'emit_error.emit() << "Vop3p does not support Literal64"' in validation

    for class_name in ('VPkAddF32Vop3p', 'VPkMulF32Vop3p', 'VPkFmaF32Vop3p'):
        body = _generated_constructor_body(vop3p, class_name)
        assert 'OperandType::OPR_SIMM64' not in body
        assert 'literal_word + 1' not in body

    for class_name in ('VPkAddF32Vop3p', 'VPkMulF32Vop3p'):
        body = _generated_decode_body(vop3p, class_name)
        assert 'LiteralSupport::Literal32,' in body
        assert '2);' in body
    fma_body = _generated_decode_body(vop3p, 'VPkFmaF32Vop3p')
    assert 'LiteralSupport::Literal32);' in fma_body


def test_split_execution_ids_name_and_match_callbacks(
    amdgpu_generated_root: Path,
) -> None:
    for arch_root in sorted(amdgpu_generated_root.iterdir()):
        backend_header = arch_root / 'execution_backend.h'
        backend_source = arch_root / 'execution_backend_exec.cpp'
        if not backend_header.exists():
            continue

        header = backend_header.read_text()
        source = backend_source.read_text()
        enum_body = header.split('enum class InstructionExecutionId : size_t {', 1)[
            1
        ].split('\n};', 1)[0]
        execution_ids = [
            line.strip().removesuffix(',')
            for line in enum_body.splitlines()
            if line.strip()
        ]
        assert execution_ids[-1] == 'Count'
        execution_ids = execution_ids[:-1]

        callbacks = re.findall(r'&execute_with_backend<([A-Za-z0-9_]+)>', source)
        assert execution_ids == callbacks
        assert 'static_cast<size_t>(InstructionExecutionId::Count)' in source

        selected_ids = []
        for model_source in sorted(arch_root.glob('*.cpp')):
            if model_source.name.endswith('_exec.cpp'):
                continue
            model = model_source.read_text()
            ids = re.findall(
                r'selected_exec_fn\(InstructionExecutionId::([A-Za-z0-9_]+)\)',
                model,
            )
            if not ids:
                continue
            assert '#include "' in model
            assert f'/generated/{arch_root.name}/execution_backend.h"' in model
            assert not re.search(r'selected_exec_fn\(\d+\)', model)
            selected_ids.extend(ids)

        assert sorted(selected_ids) == sorted(callbacks)


def test_cdna5_model_only_variant_instructions_have_no_execution_callbacks(
    gfx1250_generated_root: Path,
) -> None:
    model_only_classes = (
        'VPkFmaF64Vop3p',
        'VPkMulF64Vop3p',
        'VPkAddF64Vop3p',
        'VPkAddNcU64Vop3p',
        'VPkSubNcU64Vop3p',
        'VPkMaxNumF64Vop3p',
        'VPkMinNumF64Vop3p',
        'VWmmaF6416x16x4F64Vop3p',
    )
    header = (gfx1250_generated_root / 'vop3p.h').read_text()
    model = (gfx1250_generated_root / 'vop3p.cpp').read_text()
    backend_header = (gfx1250_generated_root / 'execution_backend.h').read_text()
    backend_source = (gfx1250_generated_root / 'execution_backend_exec.cpp').read_text()
    execution_source = (gfx1250_generated_root / 'vop3p_exec.cpp').read_text()

    for class_name in model_only_classes:
        class_body = header.split(f'class {class_name} ', 1)[1].split('\n};', 1)[0]
        assert 'execute_impl' not in class_body
        constructor = model.split(f'{class_name}::{class_name}(', 1)[1].split('\n}', 1)[
            0
        ]
        assert 'nullptr' in constructor
        assert class_name not in backend_header
        assert class_name not in backend_source
        assert class_name not in execution_source

    executable_class = 'VPkLshlAddU64Vop3p'
    class_body = header.split(f'class {executable_class} ', 1)[1].split('\n};', 1)[0]
    assert 'execute_impl' in class_body
    constructor = model.split(f'{executable_class}::{executable_class}(', 1)[1].split(
        '\n}', 1
    )[0]
    assert 'selected_exec_fn(InstructionExecutionId::VPkLshlAddU64Vop3p)' in constructor
    assert executable_class in backend_header
    assert executable_class in backend_source
    assert executable_class in execution_source
    assert 'PkU64Pair read_pk_u64_pair' in execution_source
    assert 'PkU32Pair read_pk_u32_pair' in execution_source
    assert 'void write_pk_u64_pair' in execution_source
    u64_read_helper = execution_source.split('PkU64Pair read_pk_u64_pair', 1)[1].split(
        '\n}', 1
    )[0]
    assert 'const auto reg = operand.to_register_ref();' in u64_read_helper
    assert 'if (!reg || reg->cls != RegClass::VGPR)' in u64_read_helper
    u32_read_helper = execution_source.split('PkU32Pair read_pk_u32_pair', 1)[1].split(
        '\n}', 1
    )[0]
    assert 'const auto reg = operand.to_register_ref();' in u32_read_helper
    assert 'if (reg && reg->cls == RegClass::SGPR)' in u32_read_helper
    assert 'read_lane(operand, lane)' in u32_read_helper
    assert 'read_lane_pair32(operand, lane)' in u32_read_helper


def test_generated_vop_execution_has_no_instruction_storage_bypass(
    amdgpu_generated_root: Path,
):
    forbidden = (
        'read_operand_storage',
        'write_operand_storage',
        'read_vgpr_storage',
        'write_vgpr_storage',
        'sdwa_old_dst_',
    )
    model_sources = sorted(
        (arch, path)
        for arch in (
            'cdna1',
            'cdna2',
            'cdna3',
            'cdna4',
            'rdna1',
            'rdna2',
            'rdna3',
            'rdna3_5',
            'rdna4',
            'cdna5',
        )
        for path in (amdgpu_generated_root / _generated_dir_name(arch)).glob('vop*.cpp')
        if '_exec' not in path.stem
    )
    assert model_sources
    for arch, path in model_sources:
        execution_path = _execution_source_path(path, _profile_for_arch(arch))
        source = execution_path.read_text()
        for token in forbidden:
            assert token not in source, (execution_path, token)


def test_generated_legal_64bit_dpp_uses_masked_commit_for_both_dwords(
    amdgpu_generated_root: Path,
):
    for arch in ('cdna3', 'cdna4'):
        vop1 = _execution_source_path(
            amdgpu_generated_root / arch / 'vop1.cpp',
            _profile_for_arch(arch),
        ).read_text()
        mov_b64 = _generated_method_body(vop1, 'VMovB64Vop1', 'VCvtF16U16Vop1')

        assert 'ScopedVgprWriteMask dpp_write_mask_scope_' in mov_b64, arch
        assert 'dpp_write_mask_scope_.bind(' in mov_b64, arch
        assert 'read_vgpr_storage64' not in mov_b64, arch
        assert 'write_vgpr_storage64' not in mov_b64, arch

    for arch in (
        'cdna1',
        'cdna2',
        'cdna3',
        'cdna4',
        'rdna1',
        'rdna2',
        'rdna3',
        'rdna3_5',
        'rdna4',
        'cdna5',
    ):
        vop1 = (
            amdgpu_generated_root / _generated_dir_name(arch) / 'vop1.cpp'
        ).read_text()
        decode_body = _generated_decode_body(vop1, 'VCvtF64I32Vop1')
        assert 'does not support DPP' in decode_body, arch

    cdna5_vop3 = (
        amdgpu_generated_root / _generated_dir_name('cdna5') / 'vop3_exec_alu.cpp'
    ).read_text()
    add_f64 = _generated_method_body(cdna5_vop3, 'VAddF64Vop3', 'VMulF64Vop3')
    assert 'ScopedVgprWriteMask dpp_write_mask_scope_' in add_f64
    assert 'read_operand_storage64' not in add_f64
    assert 'write_operand_storage64' not in add_f64


def test_generated_dpp_commit_separates_modern_source_and_destination_masks(
    amdgpu_generated_root: Path,
):
    legacy_vop1_arches = (
        'cdna1',
        'cdna2',
        'cdna3',
        'cdna4',
        'rdna1',
        'rdna2',
    )
    modern_vop1_arches = (
        'rdna3',
        'rdna3_5',
        'rdna4',
        'cdna5',
    )
    vopc_names = {
        'rdna3': 'vopc.cpp',
        'rdna3_5': 'vopc.cpp',
        'rdna4': 'vopc.cpp',
        'cdna5': 'vopc_cmp.cpp',
    }

    for arch in legacy_vop1_arches:
        arch_root = amdgpu_generated_root / arch
        vop1 = _execution_source_path(
            arch_root / 'vop1.cpp', _profile_for_arch(arch)
        ).read_text()

        body = _generated_method_body(vop1, 'VMovB32Vop1', 'VReadfirstlaneB32Vop1')
        assert body.count('amdgpu::dpp::make_dpp_plan(') == 1
        assert 'dpp_plan_.row_bank_mask & dpp_plan_.source_write_mask' in body
        assert 'dpp_bound_ctrl_' in body
        assert 'apply_dpp(src0, dpp_plan_,' in body
        assert 'dpp_plan_.row_bank_mask' in body
        assert 'ScopedVgprWriteMask' in body
        assert 'dpp_write_mask_scope_.restore();' in body

    for arch in modern_vop1_arches:
        arch_root = amdgpu_generated_root / _generated_dir_name(arch)
        vop1 = _execution_source_path(
            arch_root / 'vop1.cpp', _profile_for_arch(arch)
        ).read_text()

        body = _generated_method_body(vop1, 'VMovB32Vop1', 'VReadfirstlaneB32Vop1')
        assert body.count('amdgpu::dpp::make_dpp_plan(') == 1
        assert 'apply_dpp(src0, dpp_plan_, dpp_old_exec_,' in body
        assert 'dpp_old_exec_ & dpp_plan_.row_bank_mask' not in body
        assert 'ScopedVgprWriteMask' in body
        assert 'wf.exec() & dpp_plan_.row_bank_mask &' in body
        assert 'dpp_plan_.source_write_mask' in body
        assert 'dpp_write_mask_scope_.restore();' in body

    for arch, vopc_name in vopc_names.items():
        arch_root = amdgpu_generated_root / _generated_dir_name(arch)
        vopc = _execution_source_path(
            arch_root / vopc_name, _profile_for_arch(arch)
        ).read_text()

        body = _generated_method_body(vopc, 'VCmpEqU32Vopc', 'VCmpLeU32Vopc')
        assert 'ScopedVgprWriteMask' not in body
        assert 'dpp_row_bank_mask_ = dpp_plan_.row_bank_mask' in body
        assert 'dpp_source_write_mask_ = dpp_plan_.source_write_mask' in body
        assert 'amdgpu::dpp::dpp_compare_result(' in body


def test_rdna1_2_generated_vopc_dpp_is_explicitly_unsupported(
    amdgpu_generated_root: Path,
):
    for arch in ('rdna1', 'rdna2'):
        vopc_model = (amdgpu_generated_root / arch / 'vopc.cpp').read_text()
        vopc_exec = (amdgpu_generated_root / arch / 'vopc_exec.cpp').read_text()
        assert 'amdgpu::dpp::apply_dpp' not in vopc_exec, arch
        assert 'dpp_write_mask' not in vopc_exec, arch

        ctor = _generated_decode_body(vopc_model, 'VCmpEqU32Vopc')
        assert 'emit_error.emit() << "V_CMP_EQ_U32 does not support DPP"' in ctor, arch
        assert 'amdgpu::dpp::is_src_dpp8(' in ctor, arch
        assert 'reinterpret_cast<const Vop1VopDpp16MachineInst *>' not in ctor, arch

        body = _generated_method_body(vopc_exec, 'VCmpEqU32Vopc', 'VCmpLeU32Vopc')
        assert 'throw util::UnimplementedInst(mnemonic());' in body, arch
        assert 'amdgpu::dpp::is_src_dpp8(inst_.src0)' in body, arch


def test_generated_cmpx_dpp_cleanup_applies_arch_compare_rules(
    amdgpu_generated_root: Path,
):
    vopc_paths = {
        'rdna3': amdgpu_generated_root / 'rdna3' / 'vopc.cpp',
        'rdna3_5': amdgpu_generated_root / 'rdna3_5' / 'vopc.cpp',
        'rdna4': amdgpu_generated_root / 'rdna4' / 'vopc.cpp',
        'cdna5': (
            amdgpu_generated_root / _generated_dir_name('cdna5') / 'vopc_cmpx.cpp'
        ),
    }

    for arch, path in vopc_paths.items():
        vopc = _execution_source_path(path, _profile_for_arch(arch)).read_text()

        body = _generated_method_body(vopc, 'VCmpxEqU32Vopc', 'VCmpxLeU32Vopc')
        assert 'uint64_t dpp_old_exec_ = wf.exec();' in body, arch
        if arch.startswith('cdna') and arch != 'cdna5':
            assert 'uint64_t new_exec = wf.exec();' in body, arch
            assert 'dpp_old_exec_ & ~dpp_write_mask_' in body, arch
            assert 'wf.set_exec(merged_exec);' in body, arch
        else:
            assert 'amdgpu::dpp::dpp_compare_result(' in body, arch
            assert 'dpp_new_result, dpp_old_exec_,' in body, arch
            assert 'dpp_old_exec_, dpp_old_exec_' not in body, arch
            assert 'wf.set_exec(dpp_cmp_result);' in body, arch


def test_generated_vop3_compare_dpp_masks_scalar_or_exec_destination(
    amdgpu_generated_root: Path, execute_shared_path: Path
):
    paths = {
        'rdna3': amdgpu_generated_root / 'rdna3' / 'vop3.cpp',
        'rdna3_5': amdgpu_generated_root / 'rdna3_5' / 'vop3.cpp',
        'rdna4': amdgpu_generated_root / 'rdna4' / 'vop3.cpp',
        'gfx1250': amdgpu_generated_root
        / _generated_dir_name('gfx1250')
        / 'vop3_cmp.cpp',
    }

    for arch, path in paths.items():
        vop3 = _execution_source_path(path, _profile_for_arch(arch)).read_text()
        cmp_body = _generated_method_body(vop3, 'VCmpEqU32Vop3', 'VCmpLeU32Vop3')
        assert 'uint64_t dpp_preserve_mask = ' not in cmp_body
        assert 'read_wave_mask_scalar(vdst, wf)' not in cmp_body
        assert 'dpp_source_write_mask_' in cmp_body
        assert cmp_body.count('commit_result') >= 2
        assert (
            cmp_body.count('amdgpu::write_wave_mask_scalar(vdst, wf, final_result)')
            == 1
        )
        assert 'amdgpu::dpp::write_vop3_compare_result(' not in cmp_body
        assert 'execute_v_cmp_eq_u32_vop3(*this, wf, commit_result)' in cmp_body
        assert 'write_vgpr_storage' not in cmp_body

        cmpx_body = _generated_method_body(vop3, 'VCmpxEqU32Vop3', 'VCmpxLeU32Vop3')
        assert 'uint64_t dpp_row_bank_mask_ = ~0ULL;' in cmpx_body
        assert 'uint64_t dpp_source_write_mask_ = ~0ULL;' in cmpx_body
        assert 'dpp_row_bank_mask_ = dpp_plan_.row_bank_mask' in cmpx_body
        assert 'dpp_source_write_mask_ = dpp_plan_.source_write_mask' in cmpx_body
        assert cmpx_body.count('commit_result') >= 2
        assert cmpx_body.count('wf.set_exec(final_result);') == 1
        assert 'wf.set_exec(result);' not in cmpx_body
        assert 'write_vgpr_storage' not in cmpx_body

        if 'void VCmpTF32Vop3::execute_impl' in vop3:
            cmp_true_body = _generated_method_body(vop3, 'VCmpTF32Vop3', 'VCmpFF64Vop3')
            assert cmp_true_body.count('commit_result') >= 2
            assert 'execute_v_cmp_t_f32_vop3' in cmp_true_body

        cmp_class_body = _generated_method_body(
            vop3, 'VCmpClassF32Vop3', 'VCmpClassF64Vop3'
        )
        assert cmp_class_body.count('commit_result') >= 2
        cmpx_class_body = _generated_method_body(
            vop3, 'VCmpxClassF32Vop3', 'VCmpxClassF64Vop3'
        )
        assert cmpx_class_body.count('commit_result') >= 2
        assert cmpx_class_body.count('wf.set_exec(final_result);') == 1

    shared = execute_shared_path.read_text()
    shared_cmp = _shared_execute_body(shared, 'v_cmp_eq_u32_vop3', 'v_cmp_eq_u32_vopc')
    assert 'CommitResult commit_result' in shared_cmp
    assert 'ROCJITSU_TRY_SIMD_VOPC_VOP3_INT_RESULT(commit_result,' in shared_cmp
    assert 'commit_result(vcc);' in shared_cmp
    assert 'write_wave_mask_scalar' not in shared_cmp


def test_generated_modern_dpp_source_policy_covers_non_vop1_families(
    amdgpu_generated_root: Path,
):
    rdna4 = amdgpu_generated_root / 'rdna4'
    cases = (
        ('vop2.cpp', 'VAddF32Vop2', 'VSubF32Vop2'),
        ('vop3.cpp', 'VAddF32Vop3', 'VSubF32Vop3'),
        ('vop3p.cpp', 'VFmaMixF32Vop3p', 'VFmaMixloF16Vop3p'),
        (
            'vop3.cpp',
            'VAddCoCiU32Vop3SdstEnc',
            'VSubCoCiU32Vop3SdstEnc',
        ),
    )
    for filename, class_name, next_class_name in cases:
        source = _execution_source_path(
            rdna4 / filename, _profile_for_arch('rdna4')
        ).read_text()
        body = _generated_method_body(source, class_name, next_class_name)
        assert body.count('amdgpu::dpp::make_dpp_plan(') == 1, class_name
        assert 'dpp_plan_.row_bank_mask' in body, class_name
        assert 'dpp_plan_.source_write_mask' in body, class_name
        assert 'dpp_bound_ctrl_, dpp_fi_,' in body, class_name
        assert 'wf.exec(), true' in body, class_name

    vop3_model = (rdna4 / 'vop3.cpp').read_text()
    vop3_exec = _execution_source_path(
        rdna4 / 'vop3.cpp', _profile_for_arch('rdna4')
    ).read_text()
    sdst_constructor = _generated_constructor_body(vop3_model, 'VAddCoCiU32Vop3SdstEnc')
    assert 'Vop3SdstEncVopDpp16MachineInst' in sdst_constructor
    assert 'Vop3VopDpp16MachineInst' not in sdst_constructor
    sdst_body = _generated_method_body(
        vop3_exec, 'VAddCoCiU32Vop3SdstEnc', 'VSubCoCiU32Vop3SdstEnc'
    )
    assert sdst_body.count('read_wave_mask_scalar(sdst, wf)') == 1
    assert sdst_body.count('write_wave_mask_scalar(sdst, wf, final_result)') == 1
    assert sdst_body.count('amdgpu::dpp::make_dpp_plan(') == 1
    assert 'dpp_secondary_source_write_mask_' not in sdst_body
    assert 'commit_result' in sdst_body
    assert 'amdgpu::dpp::dpp_source_suppressed_result(' in sdst_body
    assert 'dpp_secondary_preserve_mask' not in sdst_body
    assert 'execute_v_add_co_ci_u32_vop3(*this, wf, commit_result)' in sdst_body
    assert 'apply_dpp(src0, dpp_plan_, dpp_old_exec_, dpp_src0_, wf)' in sdst_body
    assert 'ScopedVgprWriteMask' in sdst_body
    assert 'dpp_write_mask_scope_.restore();' in sdst_body


def test_shared_vop3_secondary_result_writer_keeps_simd_enabled(
    execute_shared_path: Path,
):
    shared = execute_shared_path.read_text()
    body = _shared_execute_body(shared, 'v_add_co_ci_u32_vop3', 'v_add_co_u32_vop2')
    assert 'CommitResult commit_result' in body
    assert 'ROCJITSU_TRY_SIMD_VOP3_CIN_RESULT(commit_result,' in body
    assert 'commit_result(vcc);' in body
    assert 'write_wave_mask_scalar' not in body


def test_generated_rdna12_sdwa_explicit_compare_uses_wave_width_destination(
    amdgpu_generated_root: Path,
):
    for arch in ('rdna1', 'rdna2'):
        source = _execution_source_path(
            amdgpu_generated_root / arch / 'vopc.cpp', _profile_for_arch(arch)
        ).read_text()
        body = _generated_function_body(
            source, 'RJ_NOINLINE void VCmpEqU32Vopc::execute_modifier_impl'
        )
        assert (
            'amdgpu::write_explicit_lane_mask(sb + sdwa_sdst_, wf, cmp_result);' in body
        )
        assert 'sb + sdwa_sdst_ + 1' not in body
        assert 'wf.set_vcc_raw(dpp_old_vcc_);' in body


@pytest.mark.parametrize(
    'template_name',
    (
        'v_add_co_ci_u32_vop2',
        'v_sub_co_ci_u32_vop2',
        'v_subrev_co_ci_u32_vop2',
    ),
)
def test_shared_vop2_carry_result_writer_keeps_simd_enabled(
    execute_shared_path: Path,
    template_name: str,
):
    shared = execute_shared_path.read_text()
    body = _generated_function_body(shared, f'inline void execute_{template_name}')
    assert 'CommitResult commit_result' in body
    assert 'ROCJITSU_TRY_SIMD_VOP2_CARRY_RESULT(commit_result,' in body
    assert 'commit_result(vcc);' in body
    assert 'wf.set_vcc' not in body


@pytest.mark.parametrize(
    ('arch', 'class_names'),
    [
        *(
            (
                arch,
                (
                    'VAddCoU32Vop2',
                    'VSubCoU32Vop2',
                    'VSubrevCoU32Vop2',
                    'VAddcCoU32Vop2',
                    'VSubbCoU32Vop2',
                    'VSubbrevCoU32Vop2',
                ),
            )
            for arch in ('cdna1', 'cdna2', 'cdna3', 'cdna4')
        ),
        *(
            (
                arch,
                (
                    'VAddCoCiU32Vop2',
                    'VSubCoCiU32Vop2',
                    'VSubrevCoCiU32Vop2',
                ),
            )
            for arch in (
                'gfx1250',
                'rdna1',
                'rdna2',
                'rdna3',
                'rdna3_5',
                'rdna4',
            )
        ),
    ],
)
def test_generated_vop2_carry_dpp_preserves_only_source_suppressed_vcc(
    amdgpu_generated_root: Path,
    arch: str,
    class_names: tuple[str, ...],
):
    arch_root = amdgpu_generated_root / _generated_dir_name(arch)
    source = _execution_source_path(
        arch_root / 'vop2.cpp', _profile_for_arch(arch)
    ).read_text()
    classified = set()
    for class_name in re.findall(
        r'^RJ_NOINLINE void (\w+)::execute_modifier_impl', source, re.MULTILINE
    ):
        body = _generated_function_body(
            source, f'RJ_NOINLINE void {class_name}::execute_modifier_impl'
        )
        if 'amdgpu::dpp::dpp_source_suppressed_result(' in body:
            classified.add(class_name)
    assert classified == set(class_names)

    for class_name in class_names:
        ordinary_body = _generated_function_body(
            source, f'void {class_name}::execute_impl'
        )
        body = _generated_function_body(
            source, f'RJ_NOINLINE void {class_name}::execute_modifier_impl'
        )

        assert 'auto commit_result = [&](uint64_t raw_result)' in ordinary_body
        assert 'uint64_t dpp_old_exec_ = wf.exec();' in body
        assert 'uint64_t dpp_old_vcc_ = wf.vcc();' in body
        assert 'dpp_source_write_mask_ = dpp_plan_.source_write_mask;' in body
        merge_start = body.index('amdgpu::dpp::dpp_source_suppressed_result(')
        merge_end = body.index('wf.set_vcc_mask(final_result);', merge_start)
        merge = body[merge_start:merge_end]
        assert 'dpp_old_vcc_, dpp_old_exec_, dpp_source_write_mask_' in merge
        assert 'row_bank' not in merge
        assert 'ScopedVgprWriteMask' in body
        assert 'commit_result' in body


def test_generated_rdna4_rejects_opcode_illegal_dpp(
    amdgpu_generated_root: Path,
):
    rdna4 = amdgpu_generated_root / 'rdna4'
    cases = (
        ('vop1.cpp', 'VCvtF64I32Vop1', 'does not support DPP'),
        ('vop3.cpp', 'VMulLoU32Vop3', 'does not support DPP'),
        ('vopc.cpp', 'VCmpEqF64Vopc', 'does not support DPP'),
        ('vop3p.cpp', 'VPkAddU16Vop3p', 'does not support DPP'),
    )
    for filename, class_name, reason in cases:
        source = (rdna4 / filename).read_text()
        constructor = _generated_constructor_body(source, class_name)
        decode_body = _generated_decode_body(source, class_name)
        assert reason in decode_body, class_name
        assert 'emit_error.emit()' in decode_body, class_name
        assert 'util::InvalidInst' not in constructor, class_name

    vop1 = (rdna4 / 'vop1.cpp').read_text()
    v_nop_decode = _generated_decode_body(vop1, 'VNopVop1')
    assert 'does not support DPP' not in v_nop_decode

    fma_mix = (rdna4 / 'vop3p.cpp').read_text()
    legal_constructor = _generated_constructor_body(fma_mix, 'VFmaMixF32Vop3p')
    legal_decode = _generated_decode_body(fma_mix, 'VFmaMixF32Vop3p')
    assert 'Vop3pVopDpp16MachineInst' in legal_constructor
    assert 'does not support DPP' not in legal_constructor
    assert 'dpp_ctrl_is_valid(dp->dpp_ctrl, false, false, true)' in legal_decode
    assert 'reserved DPP control' in legal_decode


@pytest.mark.parametrize('arch', ['rdna4', 'gfx1250'])
def test_generated_modern_rdna_allows_independent_dpp_opsel(
    amdgpu_generated_root: Path,
    arch: str,
):
    arch_root = amdgpu_generated_root / _generated_dir_name(arch)
    vop3_filename = 'vop3_alu.cpp' if arch == 'gfx1250' else 'vop3.cpp'
    vop3 = (
        _generated_split_model_source(arch_root, 'vop3_alu')
        if arch == 'gfx1250'
        else (arch_root / vop3_filename).read_text()
    )
    vop3_decode = _generated_decode_body(vop3, 'VAddF16Vop3')
    assert 'DPP requires matching OPSEL halves' not in vop3_decode

    vop3p = (arch_root / 'vop3p.cpp').read_text()
    vop3p_decode = _generated_decode_body(vop3p, 'VFmaMixF32Vop3p')
    assert 'DPP requires low/low and high/high OPSEL' not in vop3p_decode


@pytest.mark.parametrize(
    ('arch', 'profile_type'),
    [
        ('cdna1', Cdna1Profile),
        ('cdna2', Cdna2Profile),
        ('cdna3', CdnaProfile),
        ('cdna4', CdnaProfile),
        ('rdna1', Rdna1Profile),
        ('rdna2', Rdna2Profile),
        ('rdna3', Rdna3Profile),
        ('rdna3_5', Rdna3_5Profile),
        ('rdna4', Rdna4Profile),
        ('gfx1250', Cdna5Profile),
    ],
)
def test_all_generated_valu_models_match_dpp_profile_rules(
    amdgpu_generated_root: Path, arch: str, profile_type
):
    isa_xml = _mrisa_dir() / f'amdgpu_isa_{arch}.xml'
    if not isa_xml.is_file():
        pytest.skip('Semantics XML not available')

    spec = Parser(str(isa_xml), profile_type()).parse()
    generator = CodeGenerator(spec, '', derive_all_semantics(spec))
    arch_root = amdgpu_generated_root / _generated_dir_name(arch)
    model_source = '\n'.join(
        path.read_text()
        for path in sorted(arch_root.glob('*.cpp'))
        if '_exec' not in path.stem
    )
    constructors = {}
    for match in re.finditer(
        r'^(\w+)::\1\(const MachineInst \*inst', model_source, re.MULTILINE
    ):
        end = model_source.find('\n\n', match.start())
        constructors[match.group(1)] = model_source[match.start() : end]

    checked = 0
    for encoding in spec.inst_encodings:
        for inst in encoding.insts:
            dpp_encoding = (
                spec.profile.derive_parent_enc_name(inst.enc_name)
                if inst.is_implied_literal_enc
                else inst.enc_name
            )
            dpp16_struct, dpp8_struct = generator._vop_dpp_struct_names(dpp_encoding)
            if dpp16_struct is None and dpp8_struct is None:
                continue
            if not generator._supports_dpp_for_encoding(dpp_encoding):
                continue
            rule = generator._dpp_opcode_rule(inst, dpp_encoding)
            supports_dpp16 = bool(
                dpp16_struct is not None
                and generator._instruction_supports_dpp(inst, dpp_encoding)
            )
            supports_dpp8 = bool(
                dpp8_struct is not None
                and generator._instruction_supports_dpp8(inst, dpp_encoding)
            )
            constructor = constructors.get(inst.fmt_name)
            assert constructor is not None, inst.fmt_name
            decode_body = _generated_decode_body(model_source, inst.fmt_name)

            if inst.is_implied_literal_enc:
                # Implied-literal/source-extension decoding is owned by the
                # general decoder-validation work, not DPP execution semantics.
                continue

            src0 = next(
                (
                    op
                    for op in inst.operands
                    if op.is_input and not op.fieldless and op.name == 'src0'
                ),
                None,
            )
            has_src0 = src0 is not None
            if not has_src0:
                assert 'does not support DPP' not in decode_body, inst.fmt_name
                continue

            if rule is DppOpcodeRule.FORBID:
                assert 'does not support DPP' in decode_body, inst.fmt_name
                assert 'does not support DPP' not in constructor, inst.fmt_name
                checked += 1
                continue

            if not supports_dpp16 and not supports_dpp8:
                if dpp16_struct is not None:
                    assert (
                        f'reinterpret_cast<const {dpp16_struct}' not in constructor
                    ), inst.fmt_name
                if dpp8_struct is not None:
                    assert (
                        f'reinterpret_cast<const {dpp8_struct}' not in constructor
                    ), inst.fmt_name
                checked += 1
                continue

            packed_dpp_source = generator._operand_uses_packed_16bit_source(
                dpp_encoding, src0
            )
            if packed_dpp_source and supports_dpp16:
                assert 'static_cast<unsigned short>(dp->vsrc0)' in constructor
            if packed_dpp_source and supports_dpp8:
                assert 'static_cast<unsigned short>(dp8->vsrc0)' in constructor

            if rule is DppOpcodeRule.ROW_SELECT_ONLY:
                assert 'only DPP row-select controls' in decode_body, inst.fmt_name
                assert 'ROW_SELECT_BASE' in decode_body, inst.fmt_name
                assert 'ROW_SELECT_MAX' in decode_body, inst.fmt_name
                if dpp8_struct is not None:
                    assert (
                        f'reinterpret_cast<const {dpp8_struct}' not in constructor
                    ), inst.fmt_name
            else:
                if supports_dpp16:
                    assert (
                        f'reinterpret_cast<const {dpp16_struct}' in constructor
                    ), inst.fmt_name
                    capabilities = ', '.join(
                        'true' if supported else 'false'
                        for supported in (
                            generator.isa_spec.profile.dpp_supports_wave_controls,
                            generator.isa_spec.profile.dpp_supports_row_broadcast_controls,
                            generator.isa_spec.profile.dpp_supports_row_xmask,
                        )
                    )
                    assert (
                        f'dpp_ctrl_is_valid(dp->dpp_ctrl, {capabilities})'
                        in decode_body
                    ), inst.fmt_name
                if supports_dpp8:
                    assert (
                        f'reinterpret_cast<const {dpp8_struct}' in constructor
                    ), inst.fmt_name
            checked += 1

    assert checked > 100


def test_generated_gfx1250_limits_64bit_dpp_to_row_select(
    amdgpu_generated_root: Path,
):
    model_source = _generated_split_model_source(
        amdgpu_generated_root / _generated_dir_name('gfx1250'), 'vop3_alu'
    )
    decode_body = _generated_decode_body(model_source, 'VAddF64Vop3')
    assert 'only DPP row-select controls' in decode_body
    assert 'amdgpu::dpp::ROW_SELECT_BASE' in decode_body
    assert 'amdgpu::dpp::ROW_SELECT_MAX' in decode_body


@pytest.mark.parametrize('arch', ['cdna3', 'cdna4'])
def test_generated_cdna_64bit_input_dpp_is_row_only(
    amdgpu_generated_root: Path, arch: str
):
    model_source = (amdgpu_generated_root / arch / 'vop1.cpp').read_text()
    decode_body = _generated_decode_body(model_source, 'VMovB64Vop1')
    assert 'only DPP row-select controls' in decode_body
    assert 'amdgpu::dpp::ROW_SELECT_BASE' in decode_body
    assert 'amdgpu::dpp::ROW_SELECT_MAX' in decode_body


def test_shared_execute_preflight_detects_cdna3_fp8_cvt_divergence():
    specs = _parse_cdna_specs('cdna3', 'cdna4')
    plan = CrossIsaAnalyzer().analyze(specs)

    variants = _collect_shared_execute_body_variants(specs, plan)
    unshared = _unshared_execute_keys_from_variants(variants)

    fp8_cvt_keys = {
        ('v_cvt_f32_fp8', 'ENC_VOP1'),
        ('v_cvt_f32_fp8', 'ENC_VOP3'),
        ('v_cvt_f32_bf8', 'ENC_VOP1'),
        ('v_cvt_f32_bf8', 'ENC_VOP3'),
    }
    assert fp8_cvt_keys <= unshared

    vop1_fp8_variants = variants[('v_cvt_f32_fp8', 'ENC_VOP1')]
    assert 'util::fp8_e4m3_fnuz_to_f32' in vop1_fp8_variants['cdna3'][2]
    assert 'util::fp8_e4m3_to_f32' in vop1_fp8_variants['cdna4'][2]
    assert 'util::fp8_e4m3_fnuz_to_f32' not in vop1_fp8_variants['cdna4'][2]


def test_multi_isa_regen_keeps_divergent_fp8_cvt_bodies_isa_local(tmp_path):
    args = SimpleNamespace(
        isafiles=[
            f'cdna3:{_mrisa_dir() / "amdgpu_isa_cdna3.xml"}',
            f'cdna4:{_mrisa_dir() / "amdgpu_isa_cdna4.xml"}',
        ],
        gen_isas=True,
        gen_dbt=False,
        isa_output=str(tmp_path),
        dbt_output=None,
    )

    _run(args)

    shared = (tmp_path / 'shared' / 'execute_shared.h').read_text()
    cdna3_vop1 = (tmp_path / 'cdna3' / 'vop1_exec.cpp').read_text()
    cdna4_vop1 = (tmp_path / 'cdna4' / 'vop1_exec.cpp').read_text()

    assert 'inline void execute_v_cvt_f32_fp8_vop1' not in shared
    assert 'inline void execute_v_cvt_f32_bf8_vop1' not in shared
    assert 'util::fp8_e4m3_fnuz_to_f32' not in shared

    cdna3_fp8_body = _generated_method_body(
        cdna3_vop1, 'VCvtF32Fp8Vop1', 'VCvtF32Bf8Vop1'
    )
    cdna4_fp8_body = _generated_method_body(
        cdna4_vop1, 'VCvtF32Fp8Vop1', 'VCvtF32Bf8Vop1'
    )

    assert 'util::fp8_e4m3_fnuz_to_f32' in cdna3_fp8_body
    assert 'util::fp8_e4m3_to_f32' in cdna4_fp8_body
    assert 'util::fp8_e4m3_fnuz_to_f32' not in cdna4_fp8_body
    assert 'amdgpu::execute_v_cvt_f32_fp8_vop1' not in cdna3_fp8_body
    assert 'amdgpu::execute_v_cvt_f32_fp8_vop1' not in cdna4_fp8_body


def test_cdna3_generated_cvt_and_mfma_use_same_fnuz_format(
    amdgpu_generated_root: Path,
):
    cdna3_vop1 = (amdgpu_generated_root / 'cdna3' / 'vop1_exec.cpp').read_text()
    cdna3_vop3 = (amdgpu_generated_root / 'cdna3' / 'vop3_exec.cpp').read_text()
    cdna3_vop3p = (amdgpu_generated_root / 'cdna3' / 'vop3p_exec.cpp').read_text()

    assert 'util::fp8_e4m3_fnuz_to_f32' in cdna3_vop1
    assert 'util::bf8_e5m2_fnuz_to_f32' in cdna3_vop1
    assert 'util::fp8_e4m3_fnuz_to_f32' in cdna3_vop3
    assert 'util::bf8_e5m2_fnuz_to_f32' in cdna3_vop3
    assert 'util::f32_to_fp8_e4m3_fnuz_rne_mode' in cdna3_vop3
    assert 'util::f32_to_bf8_e5m2_fnuz_rne_mode' in cdna3_vop3
    assert 'util::f32_to_fp8_e4m3_fnuz_sr_mode' in cdna3_vop3
    assert 'util::f32_to_bf8_e5m2_fnuz_sr_mode' in cdna3_vop3
    assert 'amdgpu::smfmac_read_fp8_fnuz' in cdna3_vop3p
    assert 'amdgpu::smfmac_read_bf8_fnuz' in cdna3_vop3p


def test_cdna4_generated_cvt_keeps_ocp_format(
    amdgpu_generated_root: Path,
    execute_shared_path: Path,
):
    shared = execute_shared_path.read_text()
    cdna4_vop1 = (amdgpu_generated_root / 'cdna4' / 'vop1_exec.cpp').read_text()
    cdna4_vop3 = (amdgpu_generated_root / 'cdna4' / 'vop3_exec.cpp').read_text()

    assert 'inline void execute_v_cvt_f32_fp8_vop1' not in shared
    assert 'inline void execute_v_cvt_f32_bf8_vop1' not in shared
    assert 'inline void execute_v_cvt_f32_fp8_vop3' not in shared
    assert 'inline void execute_v_cvt_f32_bf8_vop3' not in shared
    assert 'util::fp8_e4m3_to_f32' in cdna4_vop1
    assert 'util::bf8_e5m2_to_f32' in cdna4_vop1
    assert 'util::fp8_e4m3_to_f32' in cdna4_vop3
    assert 'util::bf8_e5m2_to_f32' in cdna4_vop3
    assert 'util::f32_to_fp8_e4m3_rne_mode' in cdna4_vop3
    assert 'util::f32_to_bf8_e5m2_rne_mode' in cdna4_vop3
    assert 'util::f32_to_fp8_e4m3_sr_mode' in cdna4_vop3
    assert 'util::f32_to_bf8_e5m2_sr_mode' in cdna4_vop3
    assert 'util::fp8_e4m3_fnuz_to_f32' not in shared
    assert 'util::fp8_e4m3_fnuz_to_f32' not in cdna4_vop1
    assert 'util::fp8_e4m3_fnuz_to_f32' not in cdna4_vop3
    assert 'util::f32_to_fp8_e4m3_fnuz_rne_mode' not in cdna4_vop3
    assert 'util::f32_to_bf8_e5m2_fnuz_rne_mode' not in cdna4_vop3


def test_generated_pseudo_scalar_users_include_dependency(
    amdgpu_generated_root: Path,
):
    for source in amdgpu_generated_root.glob('*/*_exec*.cpp'):
        text = source.read_text()
        include = '#include "rocjitsu/isa/arch/amdgpu/shared/pseudo_scalar.h"'
        assert (include in text) == ('pseudo_scalar::' in text), source


def test_generated_vop3_dot2_true16_uses_true16_helpers(
    rdna4_generated_root: Path,
):
    vop3 = (rdna4_generated_root / 'vop3_exec.cpp').read_text()

    body = _generated_method_body(vop3, 'VDot2F16F16Vop3', 'VDot2Bf16Bf16Vop3')

    assert 'uint32_t opsel = ::rocjitsu::amdgpu::vop3_opsel(inst_);' in body
    assert 'uint32_t raw0 = amdgpu::RegisterAccess(wf).read_lane(src0, lane);' in body
    assert 'uint32_t raw1 = amdgpu::RegisterAccess(wf).read_lane(src1, lane);' in body
    assert 'read_vop3_true16_src(src2, wf, lane, opsel, 2)' in body
    assert 'util::f16_to_f32' in body
    assert 'util::f32_to_f16_mode(result, wf.fp16_ovfl())' in body
    assert 'write_vop3_true16_dst(vdst, wf, lane, opsel, result_bits, true)' in body
    assert 'throw util::UnimplementedInst' not in body


def test_generated_rdna4_vop3_cvt_f32_f16_applies_true16_source_modifiers(
    rdna4_generated_root: Path,
):
    vop3 = (rdna4_generated_root / 'vop3_exec.cpp').read_text()

    body = _generated_method_body(vop3, 'VCvtF32F16Vop3', 'VCvtU16F16Vop3')

    assert 'read_vop3_true16_src(src0, wf, lane, opsel, 0)' in body
    assert 'float src = util::f16_to_f32(static_cast<uint16_t>(raw));' in body
    assert 'if (inst_.abs & (1u << 0))' in body
    assert 'src = std::fabs(src);' in body
    assert 'if (inst_.neg & (1u << 0))' in body
    assert 'std::bit_cast<uint32_t>(src)' in body


def test_generated_rdna3_dot2acc_uses_dot2c_simd_probe(
    execute_shared_path: Path,
):
    execute_shared = execute_shared_path.read_text()

    start = execute_shared.index('inline void execute_v_dot2acc_f32_f16_vop2')
    end = execute_shared.index('inline void execute_v_dot2c_f32_f16_vop2', start)
    body = execute_shared[start:end]

    assert 'ROCJITSU_TRY_SIMD_DOTC_F16(false);' in body
    assert (
        'uint32_t acc = amdgpu::RegisterAccess(wf).read_lane(inst.vdst, lane);' in body
    )
    assert 'float facc = std::bit_cast<float>(acc);' in body
    assert 'facc += a0 * b0 + a1 * b1;' in body
    assert 'throw util::UnimplementedInst' not in body


def test_gfx1250_swmmac_reuse_hint_does_not_select_sparse_index_set():
    inst = Instruction(
        'V_SWMMAC_I32_16X16X128_IU8',
        'ENC_VOP3P',
        0,
        [
            Operand('vdst', 256, 'OPR_VGPR', True, True, False, False, 0),
            Operand('src0', 256, 'OPR_SRC_VGPR', True, False, False, False, 1),
            Operand('src1', 256, 'OPR_SRC_VGPR', True, False, False, False, 2),
            Operand('src2', 256, 'OPR_SRC_VGPR', True, False, False, False, 3),
        ],
    )

    body = gen_mfma(inst, ['vdst'], ['src0', 'src1', 'src2'], 'cdna5')

    assert 'uint32_t index_key = 0u;' in body
    assert 'index_key = inst_.opsel' not in body
    assert (
        'index_base, 32, index_key, extract_a, extract_b, inst_.clamp, const_acc);'
        in body
    )


def test_rdna4_swmmac_uses_src2_as_sparse_index_vgpr():
    inst = Instruction(
        'V_SWMMAC_F32_16X16X32_FP8_FP8',
        'ENC_VOP3P',
        0,
        [
            Operand('vdst', 256, 'OPR_VGPR', True, True, False, False, 0),
            Operand('src0', 64, 'OPR_SRC_VGPR', True, False, False, False, 1),
            Operand('src1', 128, 'OPR_SRC_VGPR', True, False, False, False, 2),
            Operand('src2', 32, 'OPR_SRC_VGPR', True, False, False, False, 3),
        ],
    )

    body = _gen_mfma(inst, 'rdna4')

    assert 'uint32_t dst = vb + vdst.encoding_value_;' in body
    assert 'uint32_t const_acc = amdgpu::ACC_FROM_VGPR;' in body
    assert 'uint32_t s2 = dst;' in body
    assert 'uint32_t index_base = amdgpu::src_base(vb, src2.encoding_value_);' in body
    assert 'uint32_t index_key = inst_.opsel & 0x1u;' in body
    assert (
        'amdgpu::exec_swmmac_f32(cu, 16, 16, 32, 8, dst, src0_base, '
        'src1_base, s2, index_base, 16, '
        'index_key, amdgpu::extract_fp8, amdgpu::extract_fp8, const_acc, wf.wf_size());'
    ) in body
    assert 'resolve_acc' not in body


def test_rdna4_f16_bf16_swmmac_dispatch_wiring_is_generated():
    operands = [
        Operand('vdst', 256, 'OPR_VGPR', True, True, False, False, 0),
        Operand('src0', 64, 'OPR_SRC_VGPR', True, False, False, False, 1),
        Operand('src1', 128, 'OPR_SRC_VGPR', True, False, False, False, 2),
        Operand('src2', 32, 'OPR_SRC_VGPR', True, False, False, False, 3),
    ]

    cases = [
        ('F16', 'FP8_FP8', 'exec_swmmac_f16', 'extract_fp8', 'extract_fp8'),
        ('BF16', 'BF8_BF8', 'exec_swmmac_bf16', 'extract_bf8', 'extract_bf8'),
    ]
    for result_type, input_type, exec_fn, extract_a, extract_b in cases:
        inst = Instruction(
            f'V_SWMMAC_{result_type}_16X16X32_{input_type}',
            'ENC_VOP3P',
            0,
            operands,
        )

        body = _gen_mfma(inst, 'rdna4')

        assert 'uint32_t dst = vb + vdst.encoding_value_;' in body
        assert 'uint32_t const_acc = amdgpu::ACC_FROM_VGPR;' in body
        assert 'uint32_t s2 = dst;' in body
        assert (
            'uint32_t index_base = amdgpu::src_base(vb, src2.encoding_value_);' in body
        )
        assert 'uint32_t index_key = inst_.opsel & 0x1u;' in body
        assert (
            f'amdgpu::{exec_fn}(cu, 16, 16, 32, 8, dst, src0_base, '
            'src1_base, s2, index_base, 16, '
            f'index_key, amdgpu::{extract_a}, amdgpu::{extract_b}, const_acc, wf.wf_size());'
        ) in body


def test_rdna4_swmmac_uses_32_index_entries_for_wide_8bit_k():
    inst = Instruction(
        'V_SWMMAC_F32_16X16X128_FP8_FP8',
        'ENC_VOP3P',
        0,
        [
            Operand('vdst', 256, 'OPR_VGPR', True, True, False, False, 0),
            Operand('src0', 256, 'OPR_SRC_VGPR', True, False, False, False, 1),
            Operand('src1', 256, 'OPR_SRC_VGPR', True, False, False, False, 2),
            Operand('src2', 32, 'OPR_SRC_VGPR', True, False, False, False, 3),
        ],
    )

    body = _gen_mfma(inst, 'rdna4')

    assert 'index_base, 32, index_key, amdgpu::extract_fp8, amdgpu::extract_fp8' in body


def test_gfx1250_generated_fp8_vop3_byte_select_uses_local_inst_member(
    execute_shared_path: Path,
    gfx1250_generated_root: Path,
):
    execute_shared = execute_shared_path.read_text()

    assert 'inline void execute_v_cvt_f32_fp8_vop3' not in execute_shared

    gfx1250_vop3_cvt = (gfx1250_generated_root / 'vop3_exec_cvt.cpp').read_text()
    assert 'RegisterAccess(wf.cu())' not in gfx1250_vop3_cvt

    body = _generated_method_body(gfx1250_vop3_cvt, 'VCvtF32Fp8Vop3', 'VCvtF32Bf8Vop3')

    assert 'amdgpu::vop3_opsel(inst_)' in body
    assert 'amdgpu::vop3_fp8_decode_e5m3(*this)' in body
    assert 'util::fp8_e5m3_to_f32' in body
    assert 'util::fp8_e4m3_to_f32' in body
    assert 'amdgpu::vop3_opsel(inst.inst_)' not in body
    assert 'amdgpu::vop3_fp8_decode_e5m3(inst_)' not in body

    body = _generated_method_body(gfx1250_vop3_cvt, 'VCvtF16Fp8Vop3', 'VCvtF16Bf8Vop3')
    body_words = ' '.join(body.split())

    assert '[[maybe_unused]] uint32_t opsel = amdgpu::vop3_opsel(inst_);' in body
    assert 'read_vop3_true16_src(src0, wf, lane, opsel, 0)' in body
    assert '>> (((opsel & 0x2u) >> 1) * 8u)' in body_words
    assert '((amdgpu::vop3_opsel(inst_) & 0x1u) << 1)' not in body


def test_gfx1250_generated_operand_rejects_reserved_scalar_source_selectors(
    gfx1250_generated_root: Path,
):
    operand = (gfx1250_generated_root / 'operand.cpp').read_text()

    assert 'case OperandType::OPR_SSRC:' in operand
    assert '(encoding_value >= 209 && encoding_value <= 229)' not in operand
    assert 'EncodingError::InvalidScalarSourceSelector' in operand


def test_generated_execute_shared_calls_have_definitions(
    amdgpu_root: Path, amdgpu_generated_root: Path
):
    import re

    definitions = set()
    for shared_root in (
        amdgpu_root / 'shared',
        amdgpu_generated_root / 'shared',
    ):
        for path in shared_root.glob('*.h'):
            definitions.update(
                re.findall(
                    r'(?:inline\s+)?void\s+(execute_[A-Za-z0-9_]+)\s*\(',
                    path.read_text(),
                )
            )

    missing = []
    for path in sorted(amdgpu_generated_root.rglob('*.cpp')):
        if 'shared' in path.parts:
            continue
        for call in re.findall(
            r'amdgpu::(execute_[A-Za-z0-9_]+)\s*\(',
            path.read_text(),
        ):
            if call not in definitions:
                missing.append(
                    (path.relative_to(amdgpu_generated_root).as_posix(), call)
                )

    assert not missing


def test_gfx1250_helper_blocks_emit_scaled_wmma_table_decoder(
    gfx1250_generated_root: Path,
):
    codegen = object.__new__(CodeGenerator)
    codegen.isa_spec = SimpleNamespace(
        arch_name='cdna5',
        profile=Cdna5Profile(),
        inst_encodings=[
            SimpleNamespace(
                enc_name='ENC_VOP3P',
                ucode_fields=[SimpleNamespace(bit_offset=14, name='opsel_hi_2')],
            )
        ],
    )

    assert codegen._supports_cdna5_scaled_wmma_vop3px2()
    assert 'VWmmaScaleF32Vop3px2' in (codegen._emit_cdna5_scaled_wmma_vop3px2_class())
    impls = codegen._emit_cdna5_scaled_wmma_vop3px2_impls()
    model_impl = ' '.join(impls.model[0].split())
    assert (
        'reinterpret_cast<const OpEncoding *>(inst + 2), '
        'selected_exec_fn(InstructionExecutionId::VWmmaScaleF32Vop3px2), '
        'Vop3p::ExtensionDecodePolicy::Skip),'
    ) in model_impl
    assert model_impl.count('cdna5_scale_operand_size_bits( inst,') == 2

    helper_model = ' '.join(codegen._emit_cdna5_matrix_fmt_helpers().model[0].split())
    assert (
        'int cdna5_scale_operand_size_bits(const MachineInst *inst, uint32_t selector)'
        in helper_model
    )
    assert (
        'return cdna5_scaled_wmma_is_scale16(inst) && is_vgpr ? 64 : 32;'
        in helper_model
    )

    helpers = codegen._emit_cdna5_scaled_wmma_vop3px2_decoder_helpers()
    assert 'isVop3pOp' in helpers
    assert 'isGfx1250WmmaScaleSource' in helpers
    assert 'isGfx1250WmmaScaleFormatPairLegal' in helpers
    assert 'isGfx1250WmmaScalePairValid' in helpers
    assert 'matrix->opsel_hi_2' in helpers
    assert 'matrix->pad_14' not in helpers
    assert 'isWmmaScaleF32Vop3px2' not in helpers

    execution_impl = impls.execution[0]
    assert '0x7f7f7f7f7f7f7f7full' in execution_impl
    assert 'byte * 0x0101010101010101ull' in execution_impl
    assert 'scale0_inline_zero ? 0u : matrix_a_scale_fmt' in execution_impl

    decoder = (gfx1250_generated_root / 'decoder.cpp').read_text()
    decode_body = decoder.split(
        'DecodeResult DecoderImpl::decode(const MachineInst *opcode, '
        'const DecodeErrorEmitter &emit_error) {'
    )[1].split('DecodeResult DecoderImpl::decodeInvalid', 1)[0]
    assert 'isWmmaScaleF32Vop3px2' not in decode_body
    assert decoder.count('&DecoderImpl::decodeVWmmaScaleF32Vop3px2,') == 2
    assert 'if (!isVop3pOp(opcode[2], 0x33)' in decoder
    assert 'if (!isGfx1250WmmaScalePairValid(opcode))' in decoder


def test_generated_decoders_publish_instruction_lookahead_bounds(
    amdgpu_generated_root: Path,
) -> None:
    expected_bounds = {
        'cdna1': 2,
        'cdna2': 2,
        'cdna3': 2,
        'cdna4': 4,
        'cdna5': 4,
        'rdna1': 3,
        'rdna2': 3,
        'rdna3': 3,
        'rdna3_5': 3,
        'rdna4': 3,
    }
    emitted_bounds = {}
    for decoder_path in amdgpu_generated_root.glob('*/decoder.h'):
        match = re.search(r'kMaxInstructionWords\s*=\s*(\d+)', decoder_path.read_text())
        assert match is not None, decoder_path
        emitted_bounds[decoder_path.parent.name] = int(match.group(1))

    assert emitted_bounds == expected_bounds


@pytest.mark.parametrize(
    ('arch_name', 'profile_type', 'expected_bound'),
    [
        ('cdna1', Cdna1Profile, 2),
        ('cdna2', Cdna2Profile, 2),
        ('cdna3', CdnaProfile, 2),
        ('cdna4', Cdna4Profile, 4),
        ('cdna5', Cdna5Profile, 4),
        ('rdna1', Rdna1Profile, 3),
        ('rdna2', Rdna2Profile, 3),
        ('rdna3', Rdna3Profile, 3),
        ('rdna3_5', Rdna3_5Profile, 3),
        ('rdna4', Rdna4Profile, 3),
    ],
)
def test_instruction_lookahead_bound_is_derived_from_isa(
    arch_name: str, profile_type, expected_bound: int
) -> None:
    isa_xml = _mrisa_dir() / f'amdgpu_isa_{arch_name}.xml'
    if not isa_xml.is_file():
        pytest.skip(f'{arch_name} MR ISA XML not available')
    spec = Parser(str(isa_xml), profile_type()).parse()
    generator = CodeGenerator(spec, '', derive_all_semantics(spec))
    assert generator._max_instruction_word_count() == expected_bound


def test_instruction_lookahead_derivation_covers_each_width_source() -> None:
    def generator_for(
        *,
        bit_count: int = 32,
        conditions=(),
        implied_words=(),
        size_overrides=None,
        has_vopd: bool = False,
        arch_name: str = 'synthetic',
        scaled_wmma: bool = False,
        compound_mfma: bool = False,
        owns_dpp_extension: bool = False,
    ) -> CodeGenerator:
        instruction = SimpleNamespace(
            name='SYNTHETIC',
            enc_name='SYNTHETIC_INST',
            is_implied_literal_enc=False,
        )
        implied = dict(implied_words)
        encoding = SimpleNamespace(
            enc_name='SYNTHETIC_ENC',
            bit_cnt=bit_count,
            insts=[instruction],
            enc_conds=list(conditions),
            has_implied_literal_ops=bool(implied),
            implied_literal_ops=implied,
        )
        profile = SimpleNamespace(
            inst_size_overrides=size_overrides or {},
            has_vopd=has_vopd,
            generate_scaled_wmma_vop3px2=scaled_wmma,
            mfma_scale_vop3px2_specs=(
                (SimpleNamespace(dense_name='SYNTHETIC'),) if compound_mfma else ()
            ),
        )
        generator = object.__new__(CodeGenerator)
        generator.isa_spec = SimpleNamespace(
            arch_name=arch_name,
            profile=profile,
            inst_encodings=[encoding],
        )
        generator._vop_dpp_struct_names = lambda _name: (
            ('SyntheticDppMachineInst', None) if owns_dpp_extension else (None, None)
        )
        return generator

    assert generator_for(bit_count=96)._max_instruction_word_count() == 3
    assert generator_for(owns_dpp_extension=True)._max_instruction_word_count() == 2
    assert (
        generator_for(
            conditions=(('default_encoding', 'inst.op == 0'),)
        )._max_instruction_word_count()
        == 2
    )
    assert (
        generator_for(
            conditions=(('has_literal', 'inst.src0 == 255'),)
        )._max_instruction_word_count()
        == 2
    )
    assert (
        generator_for(conditions=(('has_lit64', 'true'),))._max_instruction_word_count()
        == 3
    )
    assert generator_for(implied_words=((1, 3),))._max_instruction_word_count() == 4
    assert (
        generator_for(size_overrides={'SYNTHETIC': 20})._max_instruction_word_count()
        == 5
    )
    assert generator_for(has_vopd=True)._max_instruction_word_count() == 3
    assert generator_for(arch_name='cdna4')._max_instruction_word_count() == 1
    assert generator_for(compound_mfma=True)._max_instruction_word_count() == 4
    assert (
        generator_for(arch_name='cdna5', scaled_wmma=True)._max_instruction_word_count()
        == 4
    )


@pytest.mark.parametrize(
    ('arch_name', 'expected_vopd_indices'),
    [
        ('rdna3', set(range(0x190, 0x198))),
        ('rdna3_5', set(range(0x190, 0x198))),
        ('rdna4', set(range(0x190, 0x198))),
        ('cdna5', set(range(0x190, 0x198)) | set(range(0x19E, 0x1A0))),
    ],
)
def test_vopd_dispatch_uses_primary_decode_table(
    amdgpu_generated_root: Path,
    arch_name: str,
    expected_vopd_indices: set[int],
):
    arch_root = amdgpu_generated_root / _generated_dir_name(arch_name)
    decoder = (arch_root / 'decoder.cpp').read_text()
    decode_body = decoder.split(
        'DecodeResult DecoderImpl::decode(const MachineInst *opcode, '
        'const DecodeErrorEmitter &emit_error) {'
    )[1].split('DecodeResult DecoderImpl::decodeInvalid', 1)[0]

    assert 'Vopd::is_vopd' not in decode_body
    primary_table = decoder.split('DecoderImpl::primary_decode_table = {', 1)[1].split(
        '\n};', 1
    )[0]
    primary_entries = re.findall(
        r'&(?:DecoderImpl|detail)::[A-Za-z0-9_]+', primary_table
    )
    assert len(primary_entries) == 512
    actual_vopd_indices = {
        index
        for index, entry in enumerate(primary_entries)
        if entry == '&DecoderImpl::decodeVopd'
    }
    assert actual_vopd_indices == expected_vopd_indices
    assert re.search(
        r'DecoderImpl::decodeVopd\(const MachineInst \*opcode,\s+'
        r'const DecodeErrorEmitter &emit_error\)',
        decoder,
    )
    assert 'Result validation = Vopd::validate_encoding(opcode, emit_error);' in decoder
    assert 'is_vopd' not in (arch_root / 'vopd.h').read_text()
    assert 'is_vopd' not in (arch_root / 'vopd.cpp').read_text()


def test_decoder_header_keeps_dispatch_details_private(
    amdgpu_generated_root: Path,
):
    for arch_name in (
        'cdna1',
        'cdna2',
        'cdna3',
        'cdna4',
        'cdna5',
        'rdna1',
        'rdna2',
        'rdna3',
        'rdna3_5',
        'rdna4',
    ):
        arch_root = amdgpu_generated_root / _generated_dir_name(arch_name)
        decoder_header = (arch_root / 'decoder.h').read_text()
        assert 'DecodeFunc' not in decoder_header
        assert 'decodeInvalid' not in decoder_header
        assert 'primary_decode_table' not in decoder_header
        assert decoder_header.count('static DecodeResult decode(') == 1


def test_gfx1250_scaled_wmma_skips_vop3p_extension_decode(
    gfx1250_generated_root: Path,
):
    encodings_h = (gfx1250_generated_root / 'encodings.h').read_text()
    encodings_cpp = (gfx1250_generated_root / 'encodings.cpp').read_text()
    decoder_cpp = (gfx1250_generated_root / 'decoder.cpp').read_text()
    vop3p_cpp = ' '.join((gfx1250_generated_root / 'vop3p.cpp').read_text().split())

    assert 'enum class ExtensionDecodePolicy { Decode, Skip };' in encodings_h
    assert 'int num_encoded_sources = 3' in encodings_h
    assert (
        'ExtensionDecodePolicy extension_policy = ExtensionDecodePolicy::Decode'
        in encodings_h
    )

    constructor = _generated_constructor_body(encodings_cpp, 'Vop3p')
    guard = 'if (extension_policy == ExtensionDecodePolicy::Decode) {'
    guarded_suffix = constructor.split(guard, 1)[1]
    guarded_body, constructor_suffix = guarded_suffix.rsplit('\n  }\n}', 1)
    assert not constructor_suffix.strip()
    for extension_step in (
        'has_encoded_literal32()',
        'has_encoded_dpp()',
        'has_encoded_dpp8()',
        'std::memcpy(raw_words_.data(), inst, size_)',
        'raw_encoding_ = raw_words_.data()',
    ):
        assert extension_step in guarded_body

    validation_start = encodings_h.index('class Vop3p ')
    validation = encodings_h[
        validation_start : encodings_h.index('\n};', validation_start)
    ]
    assert 'emit_error.emit() << "Vop3p does not support Literal64"' in validation

    assert (
        'selected_exec_fn(InstructionExecutionId::VWmmaScaleF32Vop3px2), '
        'Vop3p::ExtensionDecodePolicy::Skip)'
    ) in vop3p_cpp
    assert 'scale->src2 == 0x080u || scale->src2 == 0x100u' in decoder_cpp
    assert '!fixed_src2_valid' in decoder_cpp
    assert '(scale->opsel & 0x2u) != 0u' in decoder_cpp
    assert '(scale->opsel_hi & 0x2u) != 0u' in decoder_cpp
    assert '(matrix->neg_hi & 0x3u) != 0u' in decoder_cpp
    assert '(matrix->neg & 0x3u) != 0u' in decoder_cpp


@pytest.mark.parametrize(
    ('arch_name', 'profile'),
    [
        ('cdna1', Cdna1Profile()),
        ('cdna2', Cdna2Profile()),
        ('cdna3', CdnaProfile()),
        ('cdna4', Cdna4Profile()),
        ('rdna1', Rdna1Profile()),
        ('rdna2', Rdna2Profile()),
        ('rdna3', Rdna3Profile()),
        ('rdna3_5', Rdna3_5Profile()),
        ('rdna4', Rdna4Profile()),
    ],
)
def test_mode_status_hwreg_semantics_call_central_vm_helpers(arch_name, profile):
    codegen = object.__new__(CodeGenerator)
    codegen.isa_spec = SimpleNamespace(
        arch_name=arch_name,
        profile=profile,
        inst_encodings=[],
        encoding_map={},
    )
    getreg_inst = Instruction(
        'S_GETREG_B32',
        'ENC_SOPK',
        17,
        [
            Operand('sdst', 32, 'OPR_SDST', False, True, False, True, 1),
            Operand('simm16', 16, 'OPR_HWREG', True, False, False, True, 2),
        ],
    )
    setreg_inst = Instruction(
        'S_SETREG_B32',
        'ENC_SOPK',
        18,
        [
            Operand('simm16', 16, 'OPR_HWREG', False, True, False, True, 1),
            Operand('sdst', 32, 'OPR_SDST', True, False, False, True, 2),
        ],
    )
    setreg_imm_inst = Instruction(
        'S_SETREG_IMM32_B32',
        'SOPK_INST_LITERAL',
        20,
        [Operand('simm16', 16, 'OPR_HWREG', False, True, False, True, 1)],
        is_implied_literal_enc=True,
    )

    getreg = codegen._gen_execute_body(
        getreg_inst, InstructionSemantics('S_GETREG_B32', 'scalar_getreg')
    )
    setreg = codegen._gen_execute_body(
        setreg_inst, InstructionSemantics('S_SETREG_B32', 'scalar_setreg')
    )
    setreg_imm = codegen._gen_execute_body(
        setreg_imm_inst,
        InstructionSemantics('S_SETREG_IMM32_B32', 'scalar_setreg_imm'),
    )

    assert 'amdgpu::read_hwreg_field(wf, hwreg, reg_val)' in getreg
    assert 'amdgpu::write_hwreg_field(wf, hwreg, src)' in setreg
    assert 'amdgpu::write_hwreg_field(wf, hwreg, src)' in setreg_imm
    assert 'wf.status_raw()' not in getreg
    assert 'wf.set_status_raw' not in setreg
    assert 'wf.set_status_raw' not in setreg_imm


def _hwreg_predefined_values(isa_name: str) -> dict[str, int]:
    root = elem_tree.parse(_mrisa_dir() / f'amdgpu_isa_{isa_name}.xml').getroot()
    for operand_type in root.iter('OperandType'):
        if operand_type.findtext('OperandTypeName') != 'OPR_HWREG':
            continue
        values: dict[str, int] = {}
        for predefined in operand_type.iter('PredefinedValue'):
            name = predefined.findtext('Name')
            value = predefined.findtext('Value')
            if name is not None and value is not None:
                values[name.lower()] = int(value)
        return values
    raise AssertionError(f'OPR_HWREG not found in {isa_name} XML')


def _normalized_hwreg_xml_values(isa_name: str) -> dict[int, str]:
    return {
        value: name.removeprefix('hw_reg_').upper()
        for name, value in _hwreg_predefined_values(isa_name).items()
    }


def _cpp_hwreg_table_values(source: str, table_name: str) -> dict[int, str]:
    start = source.index(f'constexpr HwregDescriptor {table_name}[] = {{')
    body = source[start : source.index('};', start)]
    return {
        int(match.group(1)): match.group(2)
        for match in re.finditer(r'\{\s*(\d+),\s*"([^"]+)"', body)
    }


@pytest.mark.parametrize(
    ('isa_name', 'table_name'),
    [
        ('cdna1', 'CDNA1_2_HWREGS'),
        ('cdna2', 'CDNA1_2_HWREGS'),
        ('cdna3', 'CDNA3_4_HWREGS'),
        ('cdna4', 'CDNA3_4_HWREGS'),
        ('rdna1', 'RDNA1_HWREGS'),
        ('rdna2', 'RDNA2_HWREGS'),
        ('rdna3', 'RDNA3_HWREGS'),
        ('rdna3_5', 'RDNA3_HWREGS'),
        ('rdna4', 'RDNA4_HWREGS'),
        ('cdna5', 'GFX1250_HWREGS'),
    ],
)
def test_hwreg_descriptor_tables_cover_checked_in_xml(
    rocjitsu_source_root: Path,
    isa_name,
    table_name,
):
    source = (
        rocjitsu_source_root
        / 'lib'
        / 'rocjitsu'
        / 'src'
        / 'rocjitsu'
        / 'vm'
        / 'amdgpu'
        / 'hwreg.cpp'
    ).read_text()
    xml_values = _normalized_hwreg_xml_values(isa_name)
    cpp_values = _cpp_hwreg_table_values(source, table_name)

    for value, name in xml_values.items():
        assert cpp_values.get(value) == name


def test_gfx1250_vopd_template_uses_dx9_zero_and_fma(tmp_path):
    codegen = object.__new__(CodeGenerator)
    codegen.isa_spec = SimpleNamespace(
        arch_name='cdna5',
        generated_dir_name='cdna5',
        cpp_namespace='cdna5',
        profile=Cdna5Profile(),
        operand_types={'OPR_SRC_SIMPLE'},
    )
    codegen.out_path = str(tmp_path)
    codegen.config = CodegenConfig()

    codegen.gen_vopd()
    cpp = (tmp_path / 'cdna5' / 'vopd.cpp').read_text()
    exec_cpp = (tmp_path / 'cdna5' / 'vopd_exec.cpp').read_text()

    assert '(word0_ >> 24) == 0xCF' in cpp
    assert '[[maybe_unused]] bool vopd3' not in cpp
    assert 'vopd3 ? OperandType::OPR_SRC_SIMPLE : OperandType::OPR_SRC' in cpp
    assert 'literal_uses_f64_high_bits' not in cpp
    assert 'Operand::make_literal32(literal,' not in cpp
    assert 'Operand::Literal32Widening::F64HighBits' not in cpp
    assert 'make_src0(x_bits, true, false, 0, srcx0)' in cpp
    assert 'make_src0(y_bits, true, false, 0, srcy0)' in cpp
    assert 'make_src0(x_bits, false, has_literal_, literal_, srcx0)' in cpp
    assert 'make_src0(y_bits, false, has_literal_, literal_, srcy0)' in cpp
    assert 'if (!is_valid_opcode(opx, kVopdXOpcodeMask))' in cpp
    assert 'if (!is_valid_opcode(opy, kVopdYOpcodeMask))' in cpp
    assert 'if (!is_valid_opcode(opx, kVopd3XOpcodeMask))' in cpp
    assert 'if (!is_valid_opcode(opy, kVopd3YOpcodeMask))' in cpp
    assert 'return emit_error.emit() << "invalid VOPD X opcode";' in cpp
    assert 'throw util::InvalidInst' not in cpp
    assert 'if (vdstx < y_end && vdsty < x_end)' in cpp
    assert 'case 3:\n              case 7:' not in cpp
    assert 'if (lhs == 0.0f || rhs == 0.0f)' in exec_cpp
    src_neg_start = exec_cpp.index('bool Vopd::uses_src_neg_modifier')
    src_neg_body = exec_cpp[
        src_neg_start : exec_cpp.index('uint32_t Vopd::apply_neg', src_neg_start)
    ]
    assert 'case kVopdCndmaskB32:' in src_neg_body
    assert 'if (uses_src_neg_modifier(slot.op))' in exec_cpp
    assert 'Vopd::execute_slot' not in cpp
    assert 'Vopd::execute_impl' not in cpp
    assert 'ROCJITSU_ISA_MODEL_ONLY' not in cpp
    execute_start = exec_cpp.index('uint32_t Vopd::execute_slot')
    fma_start = exec_cpp.index('case kVopdFmaF32', execute_start)
    fma_case = exec_cpp[fma_start : exec_cpp.index('case kVopdSubNcU32:', fma_start)]
    assert 'std::fma(std::bit_cast<float>(src0),' in fma_case
    assert 'std::bit_cast<float>(src1),' in fma_case
    assert 'std::bit_cast<float>(src2))' in fma_case
    assert 'constexpr uint16_t kVopdFmaF64 = 32;' in exec_cpp
    assert 'constexpr uint16_t kVopdAddF64 = 33;' in exec_cpp
    assert 'bool Vopd::is_float64_op' in cpp
    assert 'execute_registered_' not in exec_cpp
    assert 'constexpr uint16_t kVopdMulF64 = 34;' in exec_cpp
    assert 'constexpr uint16_t kVopdMaxNumF64 = 35;' in exec_cpp
    assert 'constexpr uint16_t kVopdMinNumF64 = 36;' in exec_cpp
    assert 'kVopdFmacF64' not in exec_cpp


def test_rdna4_profile_enables_generated_vopd():
    codegen = object.__new__(CodeGenerator)
    codegen.isa_spec = SimpleNamespace(
        arch_name='rdna4',
        profile=Rdna4Profile(),
        operand_types={'OPR_SRC'},
    )

    assert codegen._supports_generated_vopd()


def test_rdna4_vopd_template_uses_available_src_operand_type(tmp_path):
    codegen = object.__new__(CodeGenerator)
    codegen.isa_spec = SimpleNamespace(
        arch_name='rdna4',
        generated_dir_name='rdna4',
        cpp_namespace='rdna4',
        profile=Rdna4Profile(),
        operand_types={'OPR_SRC'},
    )
    codegen.out_path = str(tmp_path)
    codegen.config = CodegenConfig()

    codegen.gen_vopd()
    cpp = (tmp_path / 'rdna4' / 'vopd.cpp').read_text()
    exec_cpp = (tmp_path / 'rdna4' / 'vopd_exec.cpp').read_text()

    assert 'OperandType::OPR_SRC_SIMPLE' not in cpp
    assert 'vopd3 ? OperandType::OPR_SRC : OperandType::OPR_SRC' not in cpp
    assert 'return Operand(bits, OperandType::OPR_SRC, encoded);' in cpp
    assert '(word0_ >> 24) == 0xCF' not in cpp
    assert 'Format::Vopd3' not in cpp
    assert 'literal_uses_f64_high_bits' not in cpp
    assert 'is_float64_op' not in cpp
    assert 'make_src0(x_bits, false, has_literal_, literal_, srcx0)' in cpp
    assert 'make_src0(y_bits, false, has_literal_, literal_, srcy0)' in cpp
    assert 'kVopdAddF64' not in cpp
    assert 'execute_slot64' not in cpp
    assert 'constexpr uint16_t kVopdDot2AccF32F16 = 12;' in cpp
    assert 'constexpr uint16_t kVopdDot2AccF32Bf16 = 13;' in cpp
    assert '[[maybe_unused]] constexpr uint16_t kVopdDot2AccF32Bf16 = 13;' in exec_cpp
    assert 'constexpr uint16_t kVopdAndB32 = 18;' in cpp
    assert 'constexpr uint16_t kVopdBitop2B32' not in cpp
    assert 'constexpr uint16_t kVopdFmaF32' not in cpp
    assert 'v_dual_dot2acc_f32_f16' in cpp
    assert 'v_dual_max_num_f32' in cpp
    assert 'v_dual_max_f32' not in cpp


def test_rdna3_and_rdna35_vopd_generation_matches_common_profile(tmp_path):
    generated = {}
    for arch_name, profile in (
        ('rdna3', Rdna3Profile()),
        ('rdna3_5', Rdna3_5Profile()),
    ):
        codegen = object.__new__(CodeGenerator)
        codegen.isa_spec = SimpleNamespace(
            arch_name=arch_name,
            generated_dir_name=arch_name,
            cpp_namespace=arch_name,
            profile=profile,
            operand_types={'OPR_SRC'},
        )
        codegen.out_path = str(tmp_path)
        codegen.config = CodegenConfig()

        codegen.gen_vopd()
        generated[arch_name] = (tmp_path / arch_name / 'vopd.cpp').read_text()

    rdna3_cpp = generated['rdna3']
    assert 'constexpr uint16_t kVopdDot2AccF32F16 = 12;' in rdna3_cpp
    assert 'constexpr uint16_t kVopdDot2AccF32Bf16 = 13;' in rdna3_cpp
    assert 'constexpr uint16_t kVopdAndB32 = 18;' in rdna3_cpp
    assert 'constexpr uint16_t kVopdBitop2B32' not in rdna3_cpp
    assert 'constexpr uint16_t kVopdFmaF32' not in rdna3_cpp
    assert 'v_dual_max_f32' in rdna3_cpp
    assert 'v_dual_max_num_f32' not in rdna3_cpp

    normalized = {
        arch: cpp.replace('rdna3_5', 'rdnaX').replace('rdna3', 'rdnaX')
        for arch, cpp in generated.items()
    }
    assert normalized['rdna3'] == normalized['rdna3_5']


def test_gfx1250_vopd_uses_plain_src_when_simple_src_operand_is_absent(tmp_path):
    codegen = object.__new__(CodeGenerator)
    codegen.isa_spec = SimpleNamespace(
        arch_name='cdna5',
        generated_dir_name='cdna5',
        cpp_namespace='cdna5',
        profile=Cdna5Profile(),
        operand_types={'OPR_SRC'},
    )
    codegen.out_path = str(tmp_path)
    codegen.config = CodegenConfig()

    codegen.gen_vopd()
    cpp = (tmp_path / 'cdna5' / 'vopd.cpp').read_text()

    assert '[[maybe_unused]] bool vopd3' in cpp
    assert 'OperandType::OPR_SRC_SIMPLE' not in cpp
    assert 'return Operand(bits, OperandType::OPR_SRC, encoded);' in cpp


def test_bf16_mad_mix_half_updates_read_destination_operand():
    codegen = object.__new__(CodeGenerator)
    codegen.semantics = SimpleNamespace(
        instructions={
            'V_FMA_MIX_F32_BF16': SimpleNamespace(
                operation=None, semantic_class='mad_mix_f32_bf16'
            ),
            'V_FMA_MIXLO_BF16': SimpleNamespace(
                operation=None, semantic_class='mad_mixlo_bf16'
            ),
            'V_FMA_MIXHI_BF16': SimpleNamespace(
                operation=None, semantic_class='mad_mixhi_bf16'
            ),
        }
    )

    assert not codegen._dst_is_also_source(SimpleNamespace(name='V_FMA_MIX_F32_BF16'))
    assert codegen._dst_is_also_source(SimpleNamespace(name='V_FMA_MIXLO_BF16'))
    assert codegen._dst_is_also_source(SimpleNamespace(name='V_FMA_MIXHI_BF16'))


def test_addk_and_mulk_register_their_read_write_destination_as_a_source():
    codegen = object.__new__(CodeGenerator)
    codegen.semantics = SimpleNamespace(
        instructions={
            'S_ADDK_I32': SimpleNamespace(operation=None, semantic_class='scalar_addk'),
            'S_MULK_I32': SimpleNamespace(operation=None, semantic_class='scalar_mulk'),
        }
    )

    assert codegen._dst_is_also_source(SimpleNamespace(name='S_ADDK_I32'))
    assert codegen._dst_is_also_source(SimpleNamespace(name='S_MULK_I32'))


def test_gfx1250_ds_atomic_routes_data_through_vgpr_resolver():
    codegen = object.__new__(CodeGenerator)
    codegen.isa_spec = SimpleNamespace(
        arch_name='cdna5',
        profile=Cdna5Profile(),
    )

    expr = codegen._vgpr_base_expr('data0', role='Src1')
    assert 'resolved_vgpr_offset' in expr
    assert 'VgprMsbRole::Src1' in expr

    expr_cmp = codegen._vgpr_base_expr('data1', role='Src2')
    assert 'resolved_vgpr_offset' in expr_cmp
    assert 'VgprMsbRole::Src2' in expr_cmp


@pytest.mark.parametrize(
    ('dst_operands', 'returns_data'),
    [
        (['dsmem'], False),
        (['vdst', 'dsmem'], True),
    ],
)
def test_ds_atomic_codegen_returns_only_with_explicit_vdst(
    dst_operands: list[str], returns_data: bool
):
    codegen = object.__new__(CodeGenerator)
    codegen._vgpr_base_expr = lambda operand, **_kwargs: operand
    codegen._append_wait_counter_type = lambda lines, _semantic_class: lines.append(
        '  d->wait_counter_type = amdgpu::WaitCounterType::DSCNT;'
    )
    sem = InstructionSemantics(
        'DS_ADD_RTN_U64' if returns_data else 'DS_ADD_U64',
        'ds_atomic',
        operation='add',
        elem_size=8,
        num_elems=2,
    )

    body = codegen._gen_ds_atomic(dst_operands, [], sem)

    assert f'd->is_load = {str(returns_data).lower()};' in body
    assert ('d->dst_reg_base = vdst;' in body) is returns_data


def test_rdna4_ds_atomic_uses_raw_encoding():
    codegen = object.__new__(CodeGenerator)
    codegen.isa_spec = SimpleNamespace(
        arch_name='rdna4',
        profile=Rdna4Profile(),
    )

    expr = codegen._vgpr_base_expr('data0', role='Src1')
    assert 'resolved_vgpr_offset' not in expr
    assert 'inst_.data0' in expr


def test_gfx1250_flat_cmpswap_payload_width_is_independent_of_element_width():
    codegen = object.__new__(CodeGenerator)
    codegen.isa_spec = SimpleNamespace(
        arch_name='cdna5',
        profile=Cdna5Profile(),
    )

    b32 = SimpleNamespace(
        name='GLOBAL_ATOMIC_CMPSWAP_B32',
        operation='cmpswap',
        elem_size=4,
        num_elems=2,
    )
    b32_body = codegen._gen_flat_atomic([], [], b32)
    assert 'd->is_load = amdgpu::gfx12_atomic_returns(inst_.th);' in b32_body
    assert 'd->elem_size = 4;' in b32_body
    assert 'd->store_data.resize(wf.wf_size() * 8);' in b32_body
    assert 'data_base + 1' in b32_body

    b64 = SimpleNamespace(
        name='GLOBAL_ATOMIC_CMPSWAP_B64',
        operation='cmpswap',
        elem_size=8,
        num_elems=4,
    )
    b64_body = codegen._gen_flat_atomic([], [], b64)
    assert 'd->elem_size = 8;' in b64_body
    assert 'd->store_data.resize(wf.wf_size() * 16);' in b64_body
    assert 'data_base + 3' in b64_body


def test_gfx1250_flat_u64_atomic_payload_width_uses_two_dwords():
    codegen = object.__new__(CodeGenerator)
    codegen.isa_spec = SimpleNamespace(
        arch_name='cdna5',
        profile=Cdna5Profile(),
    )

    u64 = SimpleNamespace(
        name='GLOBAL_ATOMIC_ADD_U64',
        operation='add',
        elem_size=8,
        num_elems=2,
    )
    body = codegen._gen_flat_atomic([], [], u64)
    assert 'd->elem_size = 8;' in body
    assert 'd->store_data.resize(wf.wf_size() * 8);' in body
    assert 'data_base + 1' in body


def test_gfx1250_cluster_load_generators_force_request_l1_bypass():
    codegen = object.__new__(CodeGenerator)
    codegen.isa_spec = SimpleNamespace(
        arch_name='cdna5',
        profile=Cdna5Profile(),
    )

    cluster = SimpleNamespace(
        name='CLUSTER_LOAD_B32',
        elem_size=4,
        num_elems=1,
        sign_extend=False,
        d16_hi=False,
        d16_lo=False,
    )
    ordinary = SimpleNamespace(
        name='GLOBAL_LOAD_B32',
        elem_size=4,
        num_elems=1,
        sign_extend=False,
        d16_hi=False,
        d16_lo=False,
    )
    cluster_body = codegen._gen_flat_load([], [], cluster)
    ordinary_body = codegen._gen_flat_load([], [], ordinary)

    assert 'd->request_force_l1_bypass = true;' in cluster_body
    assert 'd->request_force_l1_bypass = true;' not in ordinary_body

    cluster_async = SimpleNamespace(
        name='CLUSTER_LOAD_ASYNC_TO_LDS_B32',
        elem_size=4,
        num_elems=1,
    )
    global_async = SimpleNamespace(
        name='GLOBAL_LOAD_ASYNC_TO_LDS_B32',
        elem_size=4,
        num_elems=1,
    )
    cluster_async_body = codegen._gen_global_load_async_to_lds([], [], cluster_async)
    global_async_body = codegen._gen_global_load_async_to_lds([], [], global_async)

    assert 'd->request_force_l1_bypass = true;' in cluster_async_body
    assert 'd->request_force_l1_bypass = true;' not in global_async_body


def test_gfx1250_async_lds_generators_apply_signed_ioffset():
    codegen = object.__new__(CodeGenerator)
    codegen.isa_spec = SimpleNamespace(
        arch_name='cdna5',
        profile=Cdna5Profile(),
    )

    expected_load_address = (
        'd->per_lane_lds_addr[lane] = '
        'async_lds_lane_address(inst_, wf, lane_lds_addr, 4);'
    )

    global_async = SimpleNamespace(
        name='GLOBAL_LOAD_ASYNC_TO_LDS_B32',
        elem_size=4,
        num_elems=1,
    )
    global_async_body = codegen._gen_global_load_async_to_lds([], [], global_async)
    assert expected_load_address in global_async_body

    cluster_async_cases = [
        ('CLUSTER_LOAD_ASYNC_TO_LDS_B8', 1, 1),
        ('CLUSTER_LOAD_ASYNC_TO_LDS_B32', 4, 1),
        ('CLUSTER_LOAD_ASYNC_TO_LDS_B64', 4, 2),
        ('CLUSTER_LOAD_ASYNC_TO_LDS_B128', 4, 4),
    ]
    for name, elem_size, num_elems in cluster_async_cases:
        cluster_async = SimpleNamespace(
            name=name,
            elem_size=elem_size,
            num_elems=num_elems,
        )
        cluster_async_body = codegen._gen_global_load_async_to_lds(
            [], [], cluster_async
        )
        assert (
            'd->per_lane_lds_addr[lane] = '
            f'async_lds_lane_address(inst_, wf, lane_lds_addr, {elem_size * num_elems});'
            in cluster_async_body
        )

    global_store_cases = [
        ('GLOBAL_STORE_ASYNC_FROM_LDS_B8', 1, 1),
        ('GLOBAL_STORE_ASYNC_FROM_LDS_B32', 4, 1),
        ('GLOBAL_STORE_ASYNC_FROM_LDS_B64', 4, 2),
        ('GLOBAL_STORE_ASYNC_FROM_LDS_B128', 4, 4),
    ]
    for name, elem_size, num_elems in global_store_cases:
        global_store = SimpleNamespace(
            name=name,
            elem_size=elem_size,
            num_elems=num_elems,
        )
        global_store_body = codegen._gen_global_store_async_from_lds(
            [], [], global_store
        )
        assert (
            'uint32_t lds_addr = '
            f'async_lds_lane_address(inst_, wf, lane_lds_addr, {elem_size * num_elems});'
            in global_store_body
        )

    addtid_lines = []
    codegen._append_global_addtid_addresses(addtid_lines)
    assert (
        '    int64_t offset = static_cast<int64_t>(signed_ioffset(inst_.ioffset));'
        in addtid_lines
    )

    rdna4_codegen = object.__new__(CodeGenerator)
    rdna4_codegen.isa_spec = SimpleNamespace(
        arch_name='gfx1201',
        profile=Rdna4Profile(),
    )
    rdna4_addtid_lines = []
    rdna4_codegen._append_global_addtid_addresses(rdna4_addtid_lines)
    assert (
        '    int64_t offset = static_cast<int64_t>('
        'static_cast<int32_t>(inst_.ioffset << 8) >> 8);' in rdna4_addtid_lines
    )


def test_gfx1250_buffer_cmpswap_payload_width_is_independent_of_element_width():
    codegen = object.__new__(CodeGenerator)
    codegen.isa_spec = SimpleNamespace(
        arch_name='cdna5',
        profile=Cdna5Profile(),
    )

    b64 = SimpleNamespace(
        name='BUFFER_ATOMIC_CMPSWAP_B64',
        operation='cmpswap',
        elem_size=8,
        num_elems=4,
    )
    body = codegen._gen_buffer_atomic([], [], b64)
    assert 'd->is_load = amdgpu::gfx12_atomic_returns(inst_.th);' in body
    assert 'd->elem_size = 8;' in body
    assert 'd->store_data.resize(wf.wf_size() * 16);' in body
    assert 'data_base + 3' in body


@pytest.mark.parametrize(
    'arch_name,profile,expected_exec,unexpected_exec',
    [
        ('cdna5', Cdna5Profile(), 'd->exec_mask', 'wf.exec()'),
        ('rdna4', Rdna4Profile(), 'wf.exec()', 'd->exec_mask'),
    ],
)
def test_buffer_payload_read_exec_policy_is_profile_owned(
    arch_name, profile, expected_exec, unexpected_exec
):
    codegen = object.__new__(CodeGenerator)
    codegen.isa_spec = SimpleNamespace(
        arch_name=arch_name,
        profile=profile,
    )

    store = SimpleNamespace(
        name='BUFFER_STORE_B32',
        elem_size=4,
        num_elems=1,
    )
    atomic = SimpleNamespace(
        name='BUFFER_ATOMIC_ADD_U32',
        operation='add',
        elem_size=4,
        num_elems=1,
    )

    for body in (
        codegen._gen_buffer_store([], [], store),
        codegen._gen_buffer_atomic([], [], atomic),
    ):
        assert f'uint64_t exec = {expected_exec};' in body
        assert f'uint64_t exec = {unexpected_exec};' not in body


def test_gfx1250_buffer_u64_atomic_payload_width_uses_two_dwords():
    codegen = object.__new__(CodeGenerator)
    codegen.isa_spec = SimpleNamespace(
        arch_name='cdna5',
        profile=Cdna5Profile(),
    )

    u64 = SimpleNamespace(
        name='BUFFER_ATOMIC_ADD_U64',
        operation='add',
        elem_size=8,
        num_elems=2,
    )
    body = codegen._gen_buffer_atomic([], [], u64)
    assert 'd->elem_size = 8;' in body
    assert 'd->store_data.resize(wf.wf_size() * 8);' in body
    assert 'data_base + 1' in body


def test_ev124_125_arch_gating_in_generated_operand(
    amdgpu_root: Path, amdgpu_generated_root: Path
):
    # M0 is encoded as 125 on RDNA3+ (and gfx1250) and as 124 on the
    # older RDNA1/2 and all CDNA arches. Verify the generated operand.cpp
    # carries the correct kM0EncodingValue constant for each ISA.
    expected_m0_encoding = {
        'cdna1': 124,
        'cdna2': 124,
        'cdna3': 124,
        'cdna4': 124,
        'rdna1': 124,
        'rdna2': 124,
        'rdna3': 125,
        'rdna3_5': 125,
        'rdna4': 125,
        'cdna5': 125,
    }
    for arch, encoding in expected_m0_encoding.items():
        arch_root = amdgpu_generated_root / _generated_dir_name(arch)
        # kM0EncodingValue is emitted with execution bodies; its source is
        # profile-selected.
        op = _execution_source_path(
            arch_root / 'operand.cpp', _profile_for_arch(arch)
        ).read_text()
        assert (
            f'constexpr int kM0EncodingValue = {encoding};' in op
        ), f'{arch}: expected kM0EncodingValue = {encoding}'

    # The M0 resolution logic is shared (parameterized by m0_ev) in
    # scalar_operand_resolve.h rather than emitted per-arch, so verify the gating
    # there: encoding value 124 is the NULL slot when M0 is 125, and the operand
    # matching the arch's M0 encoding reads M0.
    shared_resolve = (amdgpu_root / 'shared' / 'scalar_operand_resolve.h').read_text()
    assert re.search(
        r'if \(m0_ev == static_cast<int>\(kModernM0Selector\) &&\s*'
        r'ev == static_cast<int>\(kModernNullSelector\)\)\s*'
        r'return 0u; // NULL',
        shared_resolve,
    )
    assert 'if (ev == m0_ev)\n    return wf.m0();' in shared_resolve


def test_generated_operands_validate_vgpr_source_selectors(
    amdgpu_generated_root: Path,
):
    validation = (
        'case OperandType::OPR_SRC_VGPR:\n'
        '    if (!((encoding_value >= 256 && encoding_value <= 511)))\n'
        '      defer_encoding_error(EncodingError::InvalidVgprSourceSelector);\n'
        '    break;'
    )
    for arch in (
        'cdna1',
        'cdna2',
        'cdna3',
        'cdna4',
        'cdna5',
        'rdna1',
        'rdna2',
        'rdna3',
        'rdna3_5',
        'rdna4',
    ):
        operand = (amdgpu_generated_root / arch / 'operand.cpp').read_text()
        assert validation in operand


def test_generated_operand_validation_switch_is_shared_by_constructors(
    amdgpu_generated_root: Path,
):
    literal16_constructor = (
        'Operand::Operand(int size_bits, OperandType opr_type, int encoding_value,\n'
        '                 uint16_t literal16_display_value, bool has_literal16_display)\n'
        '    : Operand(size_bits, opr_type, encoding_value) {'
    )
    for arch in (
        'cdna1',
        'cdna2',
        'cdna3',
        'cdna4',
        'cdna5',
        'rdna1',
        'rdna2',
        'rdna3',
        'rdna3_5',
        'rdna4',
    ):
        operand = (amdgpu_generated_root / arch / 'operand.cpp').read_text()
        assert operand.count('switch (opr_type) {') == 1
        assert literal16_constructor in operand


def test_cdna4_mfma_f8f6f4_decodes_dense_and_exact_abid1_scaled_encodings(
    amdgpu_generated_root: Path,
):
    decoder = (amdgpu_generated_root / 'cdna4' / 'decoder.cpp').read_text()
    header = (amdgpu_generated_root / 'cdna4' / 'vop3p.h').read_text()
    source = (amdgpu_generated_root / 'cdna4' / 'vop3p.cpp').read_text()
    exec_source = (amdgpu_generated_root / 'cdna4' / 'vop3p_exec.cpp').read_text()

    assert 'VMfmaScaleF3216x16x128F8f6f4Vop3px2>(opcode)' in decoder
    assert 'VMfmaScaleF3232x32x64F8f6f4Vop3px2>(opcode)' in decoder
    assert 'isLegalMfmaScaleF8f6f4Selector(uint32_t selector)' in decoder
    assert 'selector >= 240 && selector <= 248' in decoder
    assert 'selector >= 256 && selector <= 511' in decoder
    assert 'auto abid2 = (opcode[2] >> 11) & 0xFu;' in decoder
    assert (
        'return enc2 == VOP3P_MFMA_ENC && (op2 == 45 || op2 == 46) && abid2 == 1u;'
        in decoder
    )
    assert 'decodeCdna4MfmaF8f6f4Suffix' in decoder
    assert (
        'DecodeResult DecoderImpl::decodeVop3pX2Prefix(const MachineInst *opcode,'
        in decoder
    )
    assert 'return decodeInvalid(opcode, emit_error);' in decoder
    assert 'reinterpret_cast<const Vop3pMfma::OpEncoding *>(opcode)' in decoder
    assert 'isLegalMfmaF8f6f4Format(uint32_t format)' in decoder
    assert (
        'op.abid != 0u || !isLegalMfmaF8f6f4Format(op.cbsz) || '
        '!isLegalMfmaF8f6f4Format(op.blgp)' in decoder
    )
    assert (
        'return std::make_unique<VMfmaF3216x16x128F8f6f4Vop3pMfma>(opcode);' in decoder
    )
    assert (
        'return std::make_unique<VMfmaF3232x32x64F8f6f4Vop3pMfma>(opcode);' in decoder
    )
    assert 'if (!isValidMfmaScaleF8f6f4(opcode))' in decoder
    assert decoder.count('return false;') >= 2
    assert 'return (neg & 0x3u) == 0 && (neg_hi & 0x3u) == 0;' in decoder
    assert 'class VMfmaScaleF3216x16x128F8f6f4Vop3px2 : public Vop3pMfma' in header
    assert 'class VMfmaScaleF3232x32x64F8f6f4Vop3px2 : public Vop3pMfma' in header
    for class_name, mnemonic in (
        (
            'VMfmaScaleF3216x16x128F8f6f4Vop3px2',
            'v_mfma_scale_f32_16x16x128_f8f6f4',
        ),
        (
            'VMfmaScaleF3232x32x64F8f6f4Vop3px2',
            'v_mfma_scale_f32_32x32x64_f8f6f4',
        ),
    ):
        wrapper = _generated_constructor_body(source, class_name)
        assert f'Vop3pMfma("{mnemonic}"' in wrapper
        assert f'InstructionExecutionId::{class_name}' in wrapper
        assert wrapper.count('OperandType::OPR_SRC_SIMPLE') == 2
        assert 'reinterpret_cast<const OpEncoding *>(inst)->src0' in wrapper
        assert 'reinterpret_cast<const OpEncoding *>(inst)->src1' in wrapper
        assert 'raw_words_{inst[0], inst[1], inst[2], inst[3]}' in wrapper
        assert 'src_operands_[3] = &scale_src0;' in wrapper
        assert 'src_operands_[4] = &scale_src1;' in wrapper
        assert 'num_src_ = 5;' in wrapper
        assert f'void {class_name}::execute_impl' in exec_source
    assert 'has_vop3px2_prefix' not in header
    assert 'has_vop3px2_prefix' not in source
    assert 'cdna4_matrix_fmt_element_bits' in source
    assert 'cdna4_matrix_fmt_element_bits(fmt) / 64' in source

    for arch in (
        'cdna1',
        'cdna2',
        'cdna3',
        'cdna5',
        'rdna1',
        'rdna2',
        'rdna3',
        'rdna3_5',
        'rdna4',
    ):
        arch_root = amdgpu_generated_root / arch
        combined = '\n'.join(
            (arch_root / rel).read_text()
            for rel in ('decoder.cpp', 'vop3p.h', 'vop3p.cpp', 'vop3p_exec.cpp')
            if (arch_root / rel).exists()
        )
        assert 'VMfmaScaleF32' not in combined
        assert 'decodeVop3pX2Prefix' not in combined
        assert 'decodeCdna4MfmaF8f6f4Suffix' not in combined
        assert 'has_vop3px2_prefix' not in combined
        assert 'exec_f32_scaled_mixed' not in combined
        assert 'v_mfma_scale_f32_16x16x128_f8f6f4' not in combined
        assert 'v_mfma_scale_f32_32x32x64_f8f6f4' not in combined

    mfma16 = source.split(
        'VMfmaF3216x16x128F8f6f4Vop3pMfma::VMfmaF3216x16x128F8f6f4Vop3pMfma'
    )[1].split('void VMfmaF3216x16x128F8f6f4Vop3pMfma::execute_impl')[0]
    mfma16 = ' '.join(mfma16.split())
    assert (
        'cdna4_matrix_fmt_operand_size_bits('
        'reinterpret_cast<const OpEncoding *>(inst)->cbsz, 16, 128)' in mfma16
    )
    assert (
        'cdna4_matrix_fmt_operand_size_bits('
        'reinterpret_cast<const OpEncoding *>(inst)->blgp, 16, 128)' in mfma16
    )

    mfma32 = source.split(
        'VMfmaF3232x32x64F8f6f4Vop3pMfma::VMfmaF3232x32x64F8f6f4Vop3pMfma'
    )[1].split('void VMfmaF3232x32x64F8f6f4Vop3pMfma::execute_impl')[0]
    mfma32 = ' '.join(mfma32.split())
    assert (
        'cdna4_matrix_fmt_operand_size_bits('
        'reinterpret_cast<const OpEncoding *>(inst)->cbsz, 32, 64)' in mfma32
    )
    assert (
        'cdna4_matrix_fmt_operand_size_bits('
        'reinterpret_cast<const OpEncoding *>(inst)->blgp, 32, 64)' in mfma32
    )


def test_generated_atomic_def_use_follows_return_control(
    amdgpu_generated_root: Path,
):
    cdna4_flat = (amdgpu_generated_root / 'cdna4' / 'flat.cpp').read_text()
    cdna4_add = cdna4_flat.split('FlatAtomicAddX2Flat::FlatAtomicAddX2Flat')[1]
    cdna4_add = cdna4_add.split('void FlatAtomicAddX2Flat::execute_impl')[0]
    # The atomic writes memory with OPR_GPUMEM operand as the baseline
    # destination (num_dst_ = 1)
    assert 'dst_operands_[0] = &gpumem;' in cdna4_add
    assert 'num_dst_ = 1;' in cdna4_add
    assert 'if ((inst_.sc0 != 0))' in cdna4_add
    assert 'dst_operands_[num_dst_++] = &vdst;' in cdna4_add

    gfx1250_buffer = (
        amdgpu_generated_root / _generated_dir_name('cdna5') / 'vbuffer.cpp'
    ).read_text()
    gfx1250_add = gfx1250_buffer.split(
        'BufferAtomicAddU32Vbuffer::BufferAtomicAddU32Vbuffer'
    )[1]
    gfx1250_add = gfx1250_add.split('void BufferAtomicAddU32Vbuffer::execute_impl')[0]
    assert 'src_operands_[0] = &vdata;' in gfx1250_add
    # The atomic writes memory with OPR_GPUMEM operand as the destination
    assert 'dst_operands_[0] = &gpumem;' in gfx1250_add
    assert 'num_dst_ = 1;' in gfx1250_add
    assert 'if (amdgpu::gfx12_atomic_returns(inst_.th))' in gfx1250_add
    assert 'dst_operands_[num_dst_++] = &vdata;' in gfx1250_add

    for mnemonic, payload_bits, return_bits in (
        ('BufferAtomicCmpswapB32Vbuffer', 64, 32),
        ('BufferAtomicCmpswapB64Vbuffer', 128, 64),
    ):
        cmpswap = gfx1250_buffer.split(f'{mnemonic}::{mnemonic}')[1]
        cmpswap = cmpswap.split(f'void {mnemonic}::execute_impl')[0]
        assert f'vdata({payload_bits}, OperandType::OPR_VGPR' in cmpswap
        assert f'vdata_return({return_bits}, OperandType::OPR_VGPR' in cmpswap
        assert 'src_operands_[0] = &vdata;' in cmpswap
        assert 'dst_operands_[num_dst_++] = &vdata_return;' in cmpswap


def test_generated_flat_saddr_null_selector_follows_encoding(
    amdgpu_generated_root: Path,
):
    rdna3_flat = (amdgpu_generated_root / 'rdna3' / 'flat.cpp').read_text()
    assert 'inst_.saddr != 0x7F' in rdna3_flat
    assert 'inst_.saddr != OPR_SREG_NULL' not in rdna3_flat
    assert (
        'if (inst_.seg != 2)\n'
        '    src_operands_[num_src_++] = &flat_scratch;' in rdna3_flat
    )

    gfx1250_root = amdgpu_generated_root / _generated_dir_name('cdna5')
    gfx1250_vflat = (gfx1250_root / 'vflat.cpp').read_text()
    gfx1250_vglobal = (gfx1250_root / 'vglobal.cpp').read_text()
    assert 'inst_.saddr != OPR_SREG_NULL' in gfx1250_vflat
    assert 'inst_.saddr != OPR_SREG_NULL' in gfx1250_vglobal


def test_shared_wave_mask_reads_qualify_instruction_operands(
    amdgpu_generated_root: Path,
):
    shared = (amdgpu_generated_root / 'shared' / 'execute_shared.h').read_text()
    assert 'read_wave_mask_scalar(inst.src2, wf)' in shared
    assert 'read_wave_mask_scalar(src2, wf)' not in shared


def _preserved_dst_codegen(*, uses_vgpr_msb_indexing: bool, supports_dpp: bool):
    """Bare CodeGenerator wired just enough to exercise the two implicit hooks."""
    codegen = object.__new__(CodeGenerator)
    codegen.isa_spec = SimpleNamespace(
        profile=SimpleNamespace(uses_vgpr_msb_indexing=uses_vgpr_msb_indexing)
    )
    codegen._supports_vop_dpp_encoding = lambda enc_name: supports_dpp
    return codegen


@pytest.mark.parametrize('supports_dpp', [False, True])
@pytest.mark.parametrize(
    'enc_name',
    [
        'ENC_VOP1',
        'ENC_VOP2',
        'ENC_VOP3',
        'ENC_VOP3P',
        'VOP3_SDST_ENC',
        'ENC_VOPC',
        'ENC_FLAT',
        'ENC_SOP1',
    ],
)
@pytest.mark.parametrize('enc_field_names', [frozenset({'vdst'}), frozenset()])
def test_implicit_use_hooks_agree_on_preserved_dst_applicability(
    enc_name, enc_field_names, supports_dpp
):
    """implicit_uses() and implicit_use_operands() must cover the same encodings.

    On gfx1250 InstDefUse clears the VGPR class from the flat implicit_uses()
    result and sources banked VGPR reads solely from implicit_use_operands(), so
    an encoding covered by one hook but not the other silently disappears from
    liveness. Both bodies come from one shared predicate; this pins that so a
    future edit to only one side fails here instead of deleting a preserve-read.
    """
    codegen = _preserved_dst_codegen(
        uses_vgpr_msb_indexing=True, supports_dpp=supports_dpp
    )
    inst_enc = SimpleNamespace(enc_name=enc_name)
    fields = set(enc_field_names)

    uses_body = codegen._encoding_implicit_uses_impl(inst_enc, fields)
    operands_body = codegen._encoding_implicit_use_operands_impl(inst_enc, fields)

    # The FLAT saddr read has no backing Operand, so it is intentionally
    # exclusive to implicit_uses(); compare only the preserved-dst predicate.
    preserved_dst_uses = 'num_dst_operands()' in uses_body
    preserved_dst_operands = 'num_dst_operands()' in operands_body
    assert preserved_dst_uses == preserved_dst_operands, (
        f'{enc_name}: implicit_uses={preserved_dst_uses} but '
        f'implicit_use_operands={preserved_dst_operands}'
    )
    if preserved_dst_uses:
        # Identical guard, differing only in how the destination is reported.
        assert uses_body.replace('uses.expand(*ref);', 'X') == operands_body.replace(
            'operands.push_back(dst);', 'X'
        )


def test_implicit_use_operands_is_gated_to_vgpr_msb_profiles():
    """Only profiles with MODE-controlled VGPR banking get the operand hook."""
    inst_enc = SimpleNamespace(enc_name='ENC_VOP1')
    fields = {'vdst'}

    banked = _preserved_dst_codegen(uses_vgpr_msb_indexing=True, supports_dpp=True)
    assert banked._encoding_implicit_use_operands_impl(inst_enc, fields)

    # The predicate itself is profile-independent; the emission site is what is
    # gated, so assert the generated text agrees (see the generated-file test).
    assert banked._encoding_implicit_uses_impl(inst_enc, fields)


def test_only_gfx1250_generates_implicit_use_operands_overrides(
    amdgpu_generated_root: Path,
):
    """The operand hook is emitted for gfx1250 and nowhere else.

    InstDefUse consults implicit_use_operands() only on the vgpr_msb != nullptr
    path, so overrides on other architectures are dead generated/compiled/vtable
    weight.
    """
    gfx1250_hits = sum(
        path.read_text().count('implicit_use_operands')
        for path in (amdgpu_generated_root / _generated_dir_name('cdna5')).rglob('*')
        if path.is_file() and path.suffix in ('.h', '.cpp')
    )
    assert gfx1250_hits > 0, 'gfx1250 must still emit the operand-backed hook'

    for arch in (
        'cdna1',
        'cdna2',
        'cdna3',
        'cdna4',
        'rdna1',
        'rdna2',
        'rdna3',
        'rdna3_5',
        'rdna4',
    ):
        arch_root = amdgpu_generated_root / arch
        if not arch_root.is_dir():
            continue
        offenders = [
            path.relative_to(amdgpu_generated_root).as_posix()
            for path in arch_root.rglob('*')
            if path.is_file()
            and path.suffix in ('.h', '.cpp')
            and 'implicit_use_operands' in path.read_text()
        ]
        assert not offenders, (
            f'{arch} has no VGPR-MSB banking, so implicit_use_operands() overrides '
            f'are dead weight: {offenders}'
        )


def test_generated_smem_rejects_invalid_complete_scalar_selector_ranges(
    amdgpu_generated_root: Path,
):
    """SMEM must issue only after complete address and data ranges validate."""
    for arch in (
        'cdna1',
        'cdna2',
        'cdna3',
        'cdna4',
        'gfx1250',
        'rdna1',
        'rdna2',
        'rdna3',
        'rdna3_5',
        'rdna4',
    ):
        smem_exec = (
            _execution_source_path(
                amdgpu_generated_root / _generated_dir_name(arch) / 'smem.cpp',
                _profile_for_arch(arch),
            )
        ).read_text()
        assert (
            'auto dst_register = amdgpu::resolve_scalar_register_range(wf,' in smem_exec
        )
        assert 'd->dst_register = *dst_register;' in smem_exec
        assert 'auto address = smem_calculate_address(' in smem_exec
        assert 'if (!address)' in smem_exec
        if 'd->is_load = false;' in smem_exec:
            assert 'const uint32_t sdata_sel = inst_.sdata;' in smem_exec
            assert (
                'auto src_register = amdgpu::resolve_scalar_register_range(wf,'
                in smem_exec
            )
            assert 'read_scalar_register(wf, *src_register, i)' in smem_exec


@pytest.mark.parametrize(
    'src_file,class_name,dst_name',
    [
        ('vbuffer.cpp', 'BufferLoadD16U8Vbuffer', 'vdata'),
        ('vds.cpp', 'DsLoadU8D16Vds', 'vdst'),
        ('vflat.cpp', 'FlatLoadD16U8Vflat', 'vdst'),
        ('vglobal.cpp', 'GlobalLoadD16U8Vglobal', 'vdst'),
        ('vscratch.cpp', 'ScratchLoadD16U8Vscratch', 'vdst'),
    ],
)
def test_gfx1250_d16_load_does_not_preserve_destination(
    gfx1250_generated_root: Path, src_file, class_name, dst_name
):
    """A gfx1250 D16 load zero-fills rather than preserve-reading its destination.

    Cover every generated memory family because each independently emits its
    instruction class and hidden-use overrides.
    """
    cpp = (gfx1250_generated_root / src_file).read_text()
    ctor = _generated_constructor_body(cpp, class_name)

    assert f'dst_operands_[0] = &{dst_name};' in ctor
    assert not re.search(rf'src_operands_\[[^\]]*\]\s*=\s*&{dst_name};', ctor)

    uses_override = f'void {class_name}::implicit_uses(RegisterSet &uses) const'
    operands_override = f'void {class_name}::implicit_use_operands('
    assert uses_override not in cpp
    assert operands_override not in cpp


def test_cdna4_d16_load_does_not_preserve_destination(
    amdgpu_generated_root: Path,
):
    """CDNA4 D16 loads are full VGPR writes because gfx950 enables SRAM ECC."""
    cpp = (amdgpu_generated_root / 'cdna4' / 'mubuf.cpp').read_text()
    class_name = 'BufferLoadUbyteD16Mubuf'
    ctor = _generated_constructor_body(cpp, class_name)

    assert 'dst_operands_[0] = &vdata;' in ctor
    assert not re.search(r'src_operands_\[[^\]]*\]\s*=\s*&vdata;', ctor)
    assert f'void {class_name}::implicit_uses(RegisterSet &uses) const' not in cpp
