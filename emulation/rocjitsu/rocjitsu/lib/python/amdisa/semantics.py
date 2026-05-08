# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Derive instruction execution semantics from mnemonic and encoding format.

Instead of maintaining a separate XML file, semantic metadata (class,
operation, data type, SCC behavior, branch condition) is derived directly
from each instruction's mnemonic name and its encoding format name.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import re

if TYPE_CHECKING:
    from amdisa.gpuisa import IsaSpec

@dataclass
class InstructionSemantics:
    """Semantic metadata for a single instruction.

    Attributes:
        name: Instruction mnemonic (e.g. ``S_ADD_U32``).
        semantic_class: High-level category that selects the code-generation
            template in :class:`~amdisa.codegen.CodeGenerator` (e.g.
            ``scalar_binop``, ``vector_cmp``, ``smem_load``).
        operation: Sub-operation within the class (e.g. ``add``, ``eq``).
        data_type: Canonical data-type string (e.g. ``f32``, ``i64``).
        sets_scc: How the instruction updates SCC. ``None`` means no update;
            other values are ``nonzero``, ``carry``, ``borrow``, ``overflow``,
            ``compare``, and ``none`` (explicitly does not set SCC).
        branch_condition: For conditional branches, the predicate to test
            (e.g. ``scc0``, ``vccnz``, ``execz``).
        elem_size: Element size in bytes for memory instructions.
        num_elems: Number of elements for memory instructions.
        sign_extend: True if the load result should be sign-extended.
    """

    name: str
    semantic_class: str
    operation: str | None = None
    data_type: str | None = None
    sets_scc: str | None = None
    branch_condition: str | None = None
    elem_size: int | None = None
    num_elems: int | None = None
    sign_extend: bool = False
    d16_hi: bool = False
    d16_lo: bool = False
    accvgpr_srcs: bool = False

class SemanticsSpec:
    """Collection of instruction semantics keyed by instruction name.

    Wraps a ``dict[str, InstructionSemantics]`` with container protocol
    methods so callers can use ``name in spec`` and ``spec[name]`` directly.
    """

    def __init__(self, instructions: dict[str, InstructionSemantics]) -> None:
        self._instructions = instructions

    def __getitem__(self, name: str) -> InstructionSemantics:
        return self._instructions[name]

    def __contains__(self, name: object) -> bool:
        return name in self._instructions

    def __len__(self) -> int:
        return len(self._instructions)

    def __iter__(self):
        return iter(self._instructions)

    @property
    def instructions(self) -> dict[str, InstructionSemantics]:
        """Read-only view of the underlying instruction map."""
        return self._instructions

# Standard data-type suffixes found at end of AMD ISA mnemonics.
_DTYPE_SUFFIXES = [
    # Compound conversion types (must come before simple ones)
    'F32_UBYTE0', 'F32_UBYTE1', 'F32_UBYTE2', 'F32_UBYTE3',
    'FLR_I32_F32', 'RPI_I32_F32',
    'F64_I32', 'I32_F64', 'F64_U32', 'U32_F64',
    'F64_F32', 'F32_F64',
    'F16_F32', 'F32_F16',
    'F16_U16', 'F16_I16', 'U16_F16', 'I16_F16',
    'F32_I32', 'F32_U32', 'I32_F32', 'U32_F32',
    'I32_I24', 'U32_U24', 'I32_I16', 'I32_I8',
    'U32_U16', 'U32_U8',
    'I32_B32', 'I32_B64', 'I32_I64',
    'B64_B32',
    # Simple types
    'F64', 'F32', 'F16',
    'I64', 'U64', 'I32', 'U32', 'I24', 'U24', 'I16', 'U16',
    'B64', 'B32', 'B16',
]

# Map suffix text to our canonical dtype string.
_DTYPE_MAP = {
    'F64': 'f64', 'F32': 'f32', 'F16': 'f16',
    'I64': 'i64', 'U64': 'u64', 'I32': 'i32', 'U32': 'u32',
    'I24': 'i24', 'U24': 'u24', 'I16': 'i16', 'U16': 'u16',
    'B64': 'b64', 'B32': 'b32', 'B16': 'u16',
    'F32_I32': 'f32_i32', 'F32_U32': 'f32_u32',
    'I32_F32': 'i32_f32', 'U32_F32': 'u32_f32',
    'F64_I32': 'f64_i32', 'I32_F64': 'i32_f64',
    'F64_U32': 'f64_u32', 'U32_F64': 'u32_f64',
    'F64_F32': 'f64_f32', 'F32_F64': 'f32_f64',
    'F16_F32': 'f16_f32', 'F32_F16': 'f32_f16',
    'F16_U16': 'f16_u16', 'F16_I16': 'f16_i16',
    'U16_F16': 'u16_f16', 'I16_F16': 'i16_f16',
    'FLR_I32_F32': 'flr_i32_f32', 'RPI_I32_F32': 'rpi_i32_f32',
    'F32_UBYTE0': 'f32_ubyte0', 'F32_UBYTE1': 'f32_ubyte1',
    'F32_UBYTE2': 'f32_ubyte2', 'F32_UBYTE3': 'f32_ubyte3',
    'I32_I24': 'i24', 'U32_U24': 'u24',
    'I32_I16': 'i32_i16', 'I32_I8': 'i32_i8',
    'U32_U16': 'u32_u16', 'U32_U8': 'u32_u8',
    'I32_B32': 'b32', 'I32_B64': 'b64', 'I32_I64': 'i64',
    'B64_B32': 'b32',
}

def _split_dtype(name: str):
    """Split instruction name into (stem, dtype_str) or (name, None)."""
    for suffix in _DTYPE_SUFFIXES:
        tag = '_' + suffix
        if name.endswith(tag):
            return name[: -len(tag)], _DTYPE_MAP[suffix]
    return name, None


# Comparison operation mapping (LG -> ne, negated FP comparisons, etc.)
_CMP_OP_MAP = {
    'EQ': 'eq', 'NE': 'ne', 'LG': 'lg', 'LT': 'lt', 'LE': 'le',
    'GT': 'gt', 'GE': 'ge', 'F': 'f', 'T': 't', 'TRU': 't',
    # Unordered FP comparisons (true when NaN):
    'NEQ': 'neq',
    'NGE': 'nge', 'NGT': 'ngt', 'NLE': 'nle', 'NLG': 'nlg', 'NLT': 'nlt',
    # Ordered / Unordered predicates:
    'O': 'o', 'U': 'u',
}

# Operations that produce no SCC update.
_SCC_NONE_OPS = frozenset({
    'mul', 'mulhi', 'bfm', 'pack_ll', 'pack_lh', 'pack_hh',
})

# Operations where SCC = (result == src0) i.e. "compare" semantics (min/max).
_SCC_COMPARE_OPS = frozenset({'min', 'max'})

def _scalar_binop_scc(op: str, dtype: str) -> str:
    """Determine SCC semantics for a scalar binary operation."""
    if op in _SCC_NONE_OPS:
        return 'none'
    if op in _SCC_COMPARE_OPS:
        return 'compare'
    # add/addc/lshl*_add on unsigned → carry
    if dtype in ('u32', 'u64') and op in ('add', 'addc', 'lshl1_add',
                                           'lshl2_add', 'lshl3_add',
                                           'lshl4_add'):
        return 'carry'
    # sub/subb on unsigned → borrow
    if dtype in ('u32', 'u64') and op in ('sub', 'subb'):
        return 'borrow'
    # add/sub on signed → overflow
    if dtype in ('i32', 'i64') and op in ('add', 'sub', 'addc', 'subb'):
        return 'overflow'
    # Everything else (bitwise, shifts, bfe, absdiff, …) → nonzero
    return 'nonzero'

# Branch condition mnemonics.
_CBRANCH_COND = {
    'S_CBRANCH_SCC0': 'scc0', 'S_CBRANCH_SCC1': 'scc1',
    'S_CBRANCH_VCCZ': 'vccz', 'S_CBRANCH_VCCNZ': 'vccnz',
    'S_CBRANCH_EXECZ': 'execz', 'S_CBRANCH_EXECNZ': 'execnz',
}

def _derive_sopp(name: str) -> InstructionSemantics | None:
    """Derive semantics for an SOPP (Scalar One-operand Program) instruction."""
    if name in _CBRANCH_COND:
        return InstructionSemantics(name, 'cbranch',
                                   branch_condition=_CBRANCH_COND[name])
    if name == 'S_BRANCH':
        return InstructionSemantics(name, 'branch', branch_condition='always')
    if name.startswith('S_ENDPGM'):
        return InstructionSemantics(name, 'endpgm')
    if name == 'S_WAITCNT':
        return InstructionSemantics(name, 'waitcnt')
    # RDNA3/3.5 named per-counter wait instructions (GFX11 — these coexist with
    # S_WAITCNT; each waits on a single counter via its immediate operand).
    _NAMED_WAIT = {
        'S_WAITCNT_VSCNT': 'waitcnt_vscnt',
        'S_WAITCNT_VMCNT': 'waitcnt_vmcnt',
        'S_WAITCNT_LGKMCNT': 'waitcnt_lgkmcnt',
        'S_WAITCNT_EXPCNT': 'waitcnt_expcnt',
    }
    if name in _NAMED_WAIT:
        return InstructionSemantics(name, 'wait_counter',
                                   operation=_NAMED_WAIT[name])
    # RDNA4 split-wait instructions (GFX12 — no S_WAITCNT; each waits on a
    # single counter whose threshold is the immediate operand directly).
    _SPLIT_WAIT = {
        'S_WAIT_LOADCNT', 'S_WAIT_STORECNT', 'S_WAIT_KMCNT',
        'S_WAIT_DSCNT', 'S_WAIT_EXPCNT', 'S_WAIT_SAMPLECNT',
        'S_WAIT_BVHCNT', 'S_WAIT_LOADCNT_DSCNT',
        'S_WAIT_STORECNT_DSCNT',
    }
    if name in _SPLIT_WAIT:
        return InstructionSemantics(name, 'wait_counter',
                                   operation=name[2:].lower())
    # S_BARRIER: workgroup synchronization. Set WfState::BARRIER.
    if name == 'S_BARRIER':
        return InstructionSemantics(name, 'barrier')

    # S_NOP, S_SLEEP, S_SETHALT, S_SETPRIO, S_SENDMSG, S_ICACHE_INV,
    # S_INCPERFLEVEL, S_DECPERFLEVEL — all are either no-ops or system/debug
    # instructions that don't affect compute simulation correctness.
    return InstructionSemantics(name, 'true_nop')

# Mnemonic stems (after stripping dtype) → (semantic_class, operation).
# Operations that need special handling are listed separately.
_SOP1_SPECIAL = {
    'S_MOV': ('scalar_mov', None),
    'S_CMOV': ('scalar_cmov', None),
    'S_NOT': ('scalar_unary', 'not'),
    'S_WQM': ('scalar_unary', 'wqm'),
    'S_BREV': ('scalar_unary', 'brev'),
    'S_BCNT0_I32': ('scalar_unary', 'bcnt0'),
    'S_BCNT1_I32': ('scalar_unary', 'bcnt1'),
    'S_FF0_I32': ('scalar_unary', 'ff0'),
    'S_FF1_I32': ('scalar_unary', 'ff1'),
    'S_FLBIT_I32': ('scalar_unary', 'flbit'),
    'S_FLBIT_I32_I64': None,  # handled specially below
    'S_BITSET0': ('scalar_unary', 'bitset0'),
    'S_BITSET1': ('scalar_unary', 'bitset1'),
    'S_GETPC': ('scalar_getpc', None),
    'S_SETPC': ('scalar_setpc', None),
    'S_SWAPPC': ('scalar_swappc', None),
    'S_RFE': ('true_nop', None),
    'S_RFE_RESTORE': ('true_nop', None),
    'S_MOVRELD': ('nop', None),
    'S_MOVRELS': ('nop', None),
    'S_ABS': ('scalar_unary', 'abs'),
    'S_SEXT_I32_I8': None,  # handled specially below
    'S_SEXT_I32_I16': None,
    'S_QUADMASK': ('scalar_unary', 'quadmask'),
    'S_SET_GPR_IDX_ON': ('true_nop', None),
    'S_SET_GPR_IDX_IDX': ('true_nop', None),
    'S_BITREPLICATE': ('nop', None),
    'S_CBRANCH_JOIN': ('true_nop', None),
    'S_BITREPL_B64_B32': ('nop', None),
    # RDNA4-exclusive SOP1 instructions:
    'S_CTZ_I32': ('scalar_unary', 'ctz'),
    'S_CLZ_I32_U32': ('scalar_unary', 'clz'),
    'S_CLZ_I32_U64': ('scalar_unary', 'clz64'),
    'S_CLS_I32': ('scalar_unary', 'cls'),
    'S_CLS_I32_I64': ('scalar_unary', 'cls64'),
    'S_MOVRELSD2': ('nop', None),
    'S_MOVRELSD2_B32': ('nop', None),
    'S_MOVRELSD_2': ('nop', None),
    'S_MOVRELSD_2_B32': ('nop', None),
    'S_SENDMSG_RTN': ('nop', None),
    'S_BARRIER_SIGNAL': ('true_nop', None),
    'S_BARRIER_SIGNAL_ISFIRST': ('true_nop', None),
    'S_ALLOC_VGPR': ('true_nop', None),
    'S_SLEEP_VAR': ('true_nop', None),
    'S_CEIL': ('scalar_unary', 'ceil'),
    'S_FLOOR': ('scalar_unary', 'floor'),
    'S_TRUNC': ('scalar_unary', 'trunc'),
    'S_RNDNE': ('scalar_unary', 'rndne'),
    'S_CVT_F32_I32': ('scalar_unary', 'cvt_f32_i32'),
    'S_CVT_F32_U32': ('scalar_unary', 'cvt_f32_u32'),
    'S_CVT_I32_F32': ('scalar_unary', 'cvt_i32_f32'),
    'S_CVT_U32_F32': ('scalar_unary', 'cvt_u32_f32'),
    'S_CVT_F16_F32': ('scalar_unary', 'cvt_f16_f32'),
    'S_CVT_F32_F16': ('scalar_unary', 'cvt_f32_f16'),
    'S_CVT_HI_F32_F16': ('scalar_unary', 'cvt_hi_f32_f16'),
}

def _derive_sop1(name: str) -> InstructionSemantics | None:
    """Derive semantics for an SOP1 (Scalar ALU One-operand) instruction."""
    # SAVEEXEC / WREXEC patterns (B64 on CDNA/Wave64, B32 on RDNA/Wave32)
    m = re.match(r'S_(\w+)_SAVEEXEC_(B32|B64)', name)
    if m:
        op = m.group(1).lower()
        return InstructionSemantics(name, 'scalar_saveexec',
                                   operation=op, data_type=m.group(2).lower())
    m = re.match(r'S_(\w+)_WREXEC_(B32|B64)', name)
    if m:
        op = m.group(1).lower()
        return InstructionSemantics(name, 'scalar_wrexec',
                                   operation=op, data_type=m.group(2).lower())

    # S_FLBIT_I32 (the one without further suffix) is a special case
    if name == 'S_FLBIT_I32':
        return InstructionSemantics(name, 'scalar_unary',
                                   operation='flbit_i32', data_type='i32',
                                   sets_scc='nonzero')

    # Sign-extend instructions
    if name == 'S_SEXT_I32_I8':
        return InstructionSemantics(name, 'scalar_unary',
                                   operation='sext8', data_type='i32',
                                   sets_scc='none')
    if name == 'S_SEXT_I32_I16':
        return InstructionSemantics(name, 'scalar_unary',
                                   operation='sext16', data_type='i32',
                                   sets_scc='none')

    # S_FLBIT_I32_I64 - special: reads 64-bit, finds leading bit of signed
    if name == 'S_FLBIT_I32_I64':
        return InstructionSemantics(name, 'scalar_unary',
                                   operation='flbit_i32_i64', data_type='i64',
                                   sets_scc='nonzero')

    # Try matching each known stem against the name. We prefer a
    # stem-first lookup over _split_dtype because compound names like
    # S_BCNT0_I32_B32 must split as (S_BCNT0_I32, B32) not (S_BCNT0, I32_B32).
    stem, dtype = None, None
    for key in _SOP1_SPECIAL:
        if name == key:
            stem, dtype = key, None
            break
        if name.startswith(key + '_'):
            rest = name[len(key) + 1:]
            stem = key
            dtype = _DTYPE_MAP.get(rest)
            break
    if stem is None:
        # Fallback to generic dtype stripping
        stem, dtype = _split_dtype(name)
    entry = _SOP1_SPECIAL.get(stem)
    if entry is not None:
        cls, op = entry
        # SCC for unary ops: bitset0/1 produce no SCC, most others → nonzero
        scc = None
        if cls == 'scalar_unary':
            if op in ('bitset0', 'bitset1'):
                scc = 'none'
            else:
                scc = 'nonzero'
        return InstructionSemantics(name, cls, operation=op,
                                   data_type=dtype, sets_scc=scc)
    return None

# Map mnemonic stem (after stripping dtype) to operation name.
_SOP2_OP_MAP = {
    'S_ADD': 'add', 'S_SUB': 'sub', 'S_ADDC': 'addc', 'S_SUBB': 'subb',
    'S_MIN': 'min', 'S_MAX': 'max', 'S_MUL': 'mul',
    'S_MUL_HI': 'mulhi', 'S_ABSDIFF': 'absdiff',
    'S_AND': 'and', 'S_OR': 'or', 'S_XOR': 'xor',
    'S_NAND': 'nand', 'S_NOR': 'nor', 'S_XNOR': 'xnor',
    'S_ANDN2': 'andn2', 'S_ORN2': 'orn2',
    'S_LSHL': 'shl', 'S_LSHR': 'shr', 'S_ASHR': 'ashr',
    'S_BFM': 'bfm', 'S_BFE': 'bfe',
    'S_CSELECT': None,  # special class
    'S_LSHL1_ADD': 'lshl1_add', 'S_LSHL2_ADD': 'lshl2_add',
    'S_LSHL3_ADD': 'lshl3_add', 'S_LSHL4_ADD': 'lshl4_add',
    'S_PACK_LL_B32_B16': None, 'S_PACK_LH_B32_B16': None,
    'S_PACK_HH_B32_B16': None,
}

def _derive_sop2(name: str) -> InstructionSemantics | None:
    """Derive semantics for an SOP2 (Scalar ALU Two-operand) instruction."""
    # CSELECT
    if name.startswith('S_CSELECT_'):
        _, dtype = _split_dtype(name)
        return InstructionSemantics(name, 'scalar_cselect', data_type=dtype)

    # PACK instructions (no dtype suffix to strip - full name is the key)
    if name == 'S_PACK_LL_B32_B16':
        return InstructionSemantics(name, 'scalar_binop',
                                   operation='pack_ll', data_type='b32',
                                   sets_scc='none')
    if name == 'S_PACK_LH_B32_B16':
        return InstructionSemantics(name, 'scalar_binop',
                                   operation='pack_lh', data_type='b32',
                                   sets_scc='none')
    if name == 'S_PACK_HH_B32_B16':
        return InstructionSemantics(name, 'scalar_binop',
                                   operation='pack_hh', data_type='b32',
                                   sets_scc='none')

    stem, dtype = _split_dtype(name)
    op = _SOP2_OP_MAP.get(stem)
    if op is not None and dtype is not None:
        scc = _scalar_binop_scc(op, dtype)
        return InstructionSemantics(name, 'scalar_binop',
                                   operation=op, data_type=dtype,
                                   sets_scc=scc)
    # Unrecognized SOP2 instructions (S_CBRANCH_G_FORK, S_RFE_RESTORE, …)
    return InstructionSemantics(name, 'nop')

def _derive_sopc(name: str) -> InstructionSemantics | None:
    """Derive semantics for an SOPC (Scalar ALU Compare) instruction."""
    # S_BITCMP0_B32, S_BITCMP1_B32, etc.
    m = re.match(r'S_BITCMP([01])_(\w+)', name)
    if m:
        _, dtype = _split_dtype('X_' + m.group(2))  # reuse dtype parser
        return InstructionSemantics(name, 'scalar_bitcmp',
                                   operation='bitcmp' + m.group(1),
                                   data_type=dtype)

    # S_CMP_<op>_<dtype>
    m = re.match(r'S_CMP_(\w+?)_(I32|U32|I64|U64)$', name)
    if m:
        cmp_name, dt_raw = m.group(1), m.group(2)
        op = _CMP_OP_MAP.get(cmp_name)
        if op is not None:
            return InstructionSemantics(name, 'scalar_cmp',
                                       operation=op,
                                       data_type=_DTYPE_MAP[dt_raw])
    # Unrecognized SOPC instructions (S_SETVSKIP, S_SET_GPR_IDX_ON, …) → nop
    return InstructionSemantics(name, 'nop')

def _derive_sopk(name: str) -> InstructionSemantics | None:
    """Derive semantics for an SOPK (Scalar with 16-bit Immediate) instruction."""
    if name == 'S_MOVK_I32':
        return InstructionSemantics(name, 'scalar_movk')
    if name == 'S_CMOVK_I32':
        return InstructionSemantics(name, 'scalar_cmovk')
    if name == 'S_ADDK_I32':
        return InstructionSemantics(name, 'scalar_addk')
    if name == 'S_MULK_I32':
        return InstructionSemantics(name, 'scalar_mulk')
    if name == 'S_CALL_B64':
        return InstructionSemantics(name, 'scalar_call')

    # S_CMPK_<op>_<dtype>
    m = re.match(r'S_CMPK_(\w+?)_(I32|U32)$', name)
    if m:
        cmp_name, dt_raw = m.group(1), m.group(2)
        op = _CMP_OP_MAP.get(cmp_name)
        if op is not None:
            return InstructionSemantics(name, 'scalar_cmpk',
                                       operation=op,
                                       data_type=_DTYPE_MAP[dt_raw])

    # Everything else (GETREG, SETREG, CBRANCH_I_FORK, …) → nop
    return InstructionSemantics(name, 'nop')

# Map mnemonic stem → operation.
_VOP1_OP_MAP = {
    'V_MOV': ('vector_mov', None),
    'V_READFIRSTLANE': ('vector_readfirstlane', None),
    'V_NOP': ('true_nop', None),
    'V_CLREXCP': ('true_nop', None),
    'V_SAT_PK_U8_I16': ('nop', None),
    'V_SCREEN_PARTITION_4SE': ('nop', None),
    'V_ACCVGPR_MOV': ('vector_mov', None),
    'V_CVT_F32_FP8': ('vector_unary', 'cvt_f32_fp8'),
    'V_CVT_F32_BF8': ('vector_unary', 'cvt_f32_bf8'),
    'V_CVT_F32_BF16': ('vector_unary', 'cvt_f32_bf16'),
    'V_CVT_PK_F32_FP8': ('nop', None),  # TODO: needs dual-VGPR write
    'V_CVT_PK_F32_BF8': ('nop', None),  # TODO: needs dual-VGPR write
    'V_CVT_OFF_F32_I4': ('nop', None),
    'V_CVT_NORM_I16': ('nop', None),
    'V_CVT_NORM_U16': ('nop', None),
    'V_SWAP': ('vector_swap', None),
    'V_NOT': ('vector_unary', 'not'),
    'V_BFREV': ('vector_unary', 'bfrev'),
    'V_BCNT_U32': ('vector_unary', 'bcnt'),
    'V_FFBL': ('vector_unary', 'ffbl'),
    'V_FFBH_U32': ('vector_unary', 'ffbh_u32'),
    'V_FFBH_I32': ('vector_unary', 'ffbh_i32'),
    'V_CEIL': ('vector_unary', 'ceil'),
    'V_FLOOR': ('vector_unary', 'floor'),
    'V_TRUNC': ('vector_unary', 'trunc'),
    'V_FRACT': ('vector_unary', 'fract'),
    'V_RNDNE': ('vector_unary', 'rndne'),
    'V_RCP': ('vector_unary', 'rcp'),
    'V_RCP_IFLAG': ('vector_unary', 'rcp_iflag'),
    'V_RSQ': ('vector_unary', 'rsq'),
    'V_SQRT': ('vector_unary', 'sqrt'),
    'V_SIN': ('vector_unary', 'sin'),
    'V_COS': ('vector_unary', 'cos'),
    'V_LOG': ('vector_unary', 'log2'),
    'V_LOG_LEGACY': ('vector_unary', 'log2'),
    'V_EXP': ('vector_unary', 'exp2'),
    'V_EXP_LEGACY': ('vector_unary', 'exp2'),
    'V_PRNG': ('true_nop', None),  # PRNG not simulated
    'V_PERMLANE16_SWAP': ('vector_permlane16_swap', None),
    'V_PERMLANE32_SWAP': ('vector_permlane32_swap', None),
    'V_SWAPREL': ('nop', None),
    'V_S_EXP': ('vector_unary', 'exp2'),
    'V_S_LOG': ('vector_unary', 'log2'),
    'V_S_RCP': ('vector_unary', 'rcp'),
    'V_S_RSQ': ('vector_unary', 'rsq'),
    'V_S_SQRT': ('vector_unary', 'sqrt'),
    'V_FREXP_EXP_I32': ('vector_unary', 'frexp_exp_f32'),
    'V_FREXP_EXP_I16': ('vector_unary', 'frexp_exp_f16'),
    'V_FREXP_MANT': ('vector_unary', 'frexp_mant_f32'),
    'V_CVT': ('vector_unary', 'cvt'),
    'V_CVT_FLR': ('vector_unary', 'cvt'),
    'V_CVT_RPI': ('vector_unary', 'cvt'),
    'V_CVT_F32_UBYTE0': ('vector_unary', 'cvt'),
    'V_CVT_F32_UBYTE1': ('vector_unary', 'cvt'),
    'V_CVT_F32_UBYTE2': ('vector_unary', 'cvt'),
    'V_CVT_F32_UBYTE3': ('vector_unary', 'cvt'),
    # RDNA4 renamed conversions (same semantics as FLR/RPI):
    'V_CVT_NEAREST': ('vector_unary', 'cvt'),
    'V_CVT_FLOOR': ('vector_unary', 'cvt'),
    # RDNA4 renamed bit-counting ops:
    'V_CLZ_I32_U32': ('vector_unary', 'ffbh_u32'),
    'V_CTZ_I32_B32': ('vector_unary', 'ffbl'),
    'V_CLS_I32': ('vector_unary', 'ffbh_i32'),
    # Pipeline / system (nop in simulation):
    'V_PIPEFLUSH': ('true_nop', None),
    # Relative addressing (nop — wave-level register indexing):
    'V_MOVRELD': ('nop', None),
    'V_MOVRELS': ('nop', None),
    'V_MOVRELSD': ('nop', None),
    'V_MOVRELSD2': ('nop', None),
    'V_MOVRELSD2_B32': ('nop', None),
    'V_MOVRELSD_2': ('nop', None),
    'V_MOVRELSD_2_B32': ('nop', None),
    'V_SWAP_REL': ('nop', None),
    'V_PERMLANE64': ('vector_permlane64', None),
}

def _derive_vop1(name: str) -> InstructionSemantics | None:
    """Derive semantics for a VOP1 (Vector ALU One-operand) instruction."""
    # Stem-first: check each table key against the name to handle compound
    # suffixes like V_FREXP_EXP_I32_F32 → (V_FREXP_EXP_I32, F32).
    stem, dtype, entry = None, None, None
    for key, val in _VOP1_OP_MAP.items():
        if name == key:
            stem, dtype, entry = key, None, val
            break
        if name.startswith(key + '_'):
            rest = name[len(key) + 1:]
            dt = _DTYPE_MAP.get(rest)
            if dt is not None:
                stem, dtype, entry = key, dt, val
                break
    if entry is None:
        # Fallback to generic dtype stripping
        stem, dtype = _split_dtype(name)
        entry = _VOP1_OP_MAP.get(stem)
    if entry is not None:
        cls, op = entry
        return InstructionSemantics(name, cls, operation=op, data_type=dtype)
    return None

_VOP2_OP_MAP = {
    'V_ADD': 'add', 'V_SUB': 'sub', 'V_SUBREV': 'rsub',
    'V_MUL': 'mul', 'V_MUL_LEGACY': 'mul_legacy',
    'V_MUL_I32_I24': None, 'V_MUL_U32_U24': None,
    'V_MUL_HI_I32_I24': None, 'V_MUL_HI_U32_U24': None,
    'V_AND': 'and', 'V_OR': 'or', 'V_XOR': 'xor', 'V_XNOR': 'xnor',
    'V_LSHLREV': 'shl', 'V_LSHRREV': 'shr', 'V_ASHRREV': 'ashr',
    'V_MIN': 'min', 'V_MAX': 'max',
    # RDNA4 renamed min/max with IEEE 754 NaN semantics:
    'V_MIN_NUM': 'min', 'V_MAX_NUM': 'max',
    # RDNA4 renamed no-carry add/sub:
    'V_ADD_NC': 'add', 'V_SUB_NC': 'sub', 'V_SUBREV_NC': 'rsub',
    'V_FMAC': 'fmac', 'V_LDEXP': 'ldexp',
    'V_BFM': 'bfm',
    'V_CNDMASK': None,  # special class
    # RDNA4 V_MUL_DX9_ZERO variant (same as mul_legacy for zero*anything=0):
    'V_MUL_DX9_ZERO': 'mul_legacy',
    'V_MAC_LEGACY': 'fmac', 'V_FMAC_LEGACY': 'fmac', 'V_FMAC_DX9_ZERO': 'fmac',
}

def _derive_vop2(name: str) -> InstructionSemantics | None:
    """Derive semantics for a VOP2 (Vector ALU Two-operand) instruction."""
    if name.startswith('V_CNDMASK_'):
        _, dtype = _split_dtype(name)
        return InstructionSemantics(name, 'vector_cndmask', data_type=dtype)

    # RDNA carry-in variants: V_ADD_CO_CI_U32 = V_ADDC_CO_U32 equivalent.
    # Check _CO_CI_ BEFORE the generic _CO_ pattern to avoid greedy match.
    m = re.match(r'V_(ADD|SUB|SUBREV)_CO_CI_(\w+)', name)
    if m:
        ci_map = {'ADD': 'addc', 'SUB': 'subbc', 'SUBREV': 'subbrevco'}
        return InstructionSemantics(name, 'vector_add_co',
                                   operation=ci_map[m.group(1)],
                                   data_type=_DTYPE_MAP.get(m.group(2)))
    # ADD_CO / SUB_CO / ADDC_CO etc. → vector_add_co
    m = re.match(r'V_(ADD|SUB|SUBREV|ADDC|SUBB|SUBBREV)_CO_(\w+)', name)
    if m:
        op_raw, dt_raw = m.group(1), m.group(2)
        op_map = {
            'ADD': 'add', 'SUB': 'sub', 'SUBREV': 'rsub',
            'ADDC': 'addc', 'SUBB': 'subbc', 'SUBBREV': 'subbrevco',
        }
        return InstructionSemantics(name, 'vector_add_co',
                                   operation=op_map[op_raw],
                                   data_type=_DTYPE_MAP.get(dt_raw))

    # FMAMK / FMAAK / MADMK / MADAK - FMA/MAD with inline constant
    if name == 'V_FMAMK_F32':
        return InstructionSemantics(name, 'vector_fmamk', data_type='f32')
    if name == 'V_FMAAK_F32':
        return InstructionSemantics(name, 'vector_fmaak', data_type='f32')
    if name in ('V_MADMK_F16', 'V_FMAMK_F16'):
        return InstructionSemantics(name, 'vector_fmamk', data_type='f16')
    if name in ('V_MADAK_F16', 'V_FMAAK_F16'):
        return InstructionSemantics(name, 'vector_fmaak', data_type='f16')

    if name == 'V_MAC_F32':
        return InstructionSemantics(name, 'vector_binop',
                                   operation='fmac', data_type='f32')
    if name == 'V_MAC_F16':
        return InstructionSemantics(name, 'vector_binop',
                                   operation='fmac', data_type='f16')
    if name == 'V_MADMK_F32':
        return InstructionSemantics(name, 'vector_fmamk', data_type='f32')
    if name == 'V_MADAK_F32':
        return InstructionSemantics(name, 'vector_fmaak', data_type='f32')

    if name == 'V_MUL_LO_U16':
        return InstructionSemantics(name, 'vector_binop',
                                   operation='mul', data_type='u16')

    # Packed FP16 FMA and DOT2ACC VOP2 forms (nop; VOP3P forms are functional)
    if name in ('V_PK_FMAC_F16', 'V_DOT2ACC_F32_F16'):
        return InstructionSemantics(name, 'nop')

    # DOT product instructions
    if name == 'V_DOT2C_F32_F16':
        return InstructionSemantics(name, 'vector_dot', operation='dot2c',
                                   data_type='f32')
    if name == 'V_DOT2C_I32_I16':
        return InstructionSemantics(name, 'vector_dot', operation='dot2c',
                                   data_type='i32')
    if name == 'V_DOT4C_I32_I8':
        return InstructionSemantics(name, 'vector_dot', operation='dot4c',
                                   data_type='i32')
    if name == 'V_DOT8C_I32_I4':
        return InstructionSemantics(name, 'vector_dot', operation='dot8c',
                                   data_type='i32')

    # 24-bit multiply variants (full name is the key)
    _24bit = {
        'V_MUL_I32_I24': ('mul', 'i24'),
        'V_MUL_U32_U24': ('mul', 'u24'),
        'V_MUL_HI_I32_I24': ('mulhi', 'i24'),
        'V_MUL_HI_U32_U24': ('mulhi', 'u24'),
    }
    if name in _24bit:
        op, dt = _24bit[name]
        return InstructionSemantics(name, 'vector_binop',
                                   operation=op, data_type=dt)

    # V_MUL_LO_U32 → mul, u32
    if name == 'V_MUL_LO_U32':
        return InstructionSemantics(name, 'vector_binop',
                                   operation='mul', data_type='u32')
    # V_MUL_HI_I32 / V_MUL_HI_U32
    if name == 'V_MUL_HI_I32':
        return InstructionSemantics(name, 'vector_binop',
                                   operation='mulhi', data_type='i32')
    if name == 'V_MUL_HI_U32':
        return InstructionSemantics(name, 'vector_binop',
                                   operation='mulhi', data_type='u32')

    stem, dtype = _split_dtype(name)
    op = _VOP2_OP_MAP.get(stem)
    if op is not None and dtype is not None:
        return InstructionSemantics(name, 'vector_binop',
                                   operation=op, data_type=dtype)
    return None

def _derive_vopc(name: str) -> InstructionSemantics | None:
    """Derive semantics for a VOPC (Vector ALU Compare) instruction."""
    # V_CMP[X]_CLASS_<dtype>
    m = re.match(r'V_(CMPX?)_CLASS_(F32|F64|F16)$', name)
    if m:
        prefix, dt_raw = m.group(1), m.group(2)
        cls = 'vector_cmpx_class' if prefix == 'CMPX' else 'vector_cmp_class'
        return InstructionSemantics(name, cls,
                                   data_type=_DTYPE_MAP[dt_raw])

    # V_CMP[X]_<op>_<dtype>
    m = re.match(r'V_(CMPX?)_(\w+?)_(F32|F64|F16|I32|U32|I64|U64|I16|U16)$',
                 name)
    if m:
        prefix, cmp_name, dt_raw = m.group(1), m.group(2), m.group(3)
        op = _CMP_OP_MAP.get(cmp_name)
        if op is not None:
            cls = 'vector_cmpx' if prefix == 'CMPX' else 'vector_cmp'
            return InstructionSemantics(name, cls,
                                       operation=op,
                                       data_type=_DTYPE_MAP[dt_raw])
    return None

_VOP3_TERNARY_MAP = {
    'V_MAD': 'mad', 'V_MAD_LEGACY': 'mad',
    'V_FMA': 'fma', 'V_FMA_LEGACY': 'fma',
    'V_MIN3': 'min3', 'V_MAX3': 'max3', 'V_MED3': 'med3',
    'V_BFE': None, 'V_BFI': 'bfi',
    'V_ALIGNBIT': 'alignbit', 'V_ALIGNBYTE': 'alignbyte',
    'V_ADD3': 'add3', 'V_LSHL_OR': 'lshl_or', 'V_AND_OR': 'and_or',
    'V_OR3': 'or3', 'V_LSHL_ADD': 'lshl_add', 'V_ADD_LSHL': 'add_lshl',
    'V_XAD': 'xad', 'V_PERM': 'perm',
}

def _derive_vop3(name: str) -> InstructionSemantics | None:
    """Derive semantics for a VOP3 (Vector ALU Three-operand) instruction."""
    if name == 'V_READLANE_B32':
        return InstructionSemantics(name, 'vector_readlane')
    if name == 'V_WRITELANE_B32':
        return InstructionSemantics(name, 'vector_writelane')

    # BFE: signed vs unsigned
    if name.startswith('V_BFE_'):
        _, dtype = _split_dtype(name)
        op = 'bfe_i' if dtype and dtype.startswith('i') else 'bfe_u'
        return InstructionSemantics(name, 'vector_ternary',
                                   operation=op, data_type=dtype)

    # Cube map instructions (VOP3-only)
    _CUBE_MAP = {
        'V_CUBEID_F32': 'cubeid', 'V_CUBESC_F32': 'cubesc',
        'V_CUBETC_F32': 'cubetc', 'V_CUBEMA_F32': 'cubema',
    }
    if name in _CUBE_MAP:
        return InstructionSemantics(name, 'vector_ternary',
                                   operation=_CUBE_MAP[name], data_type='f32')

    # Division helper instructions (VOP3-only)
    for prefix in ('V_DIV_FIXUP_', 'V_DIV_FIXUP_LEGACY_'):
        if name.startswith(prefix):
            _, dtype = _split_dtype(name)
            return InstructionSemantics(name, 'vector_div_fixup',
                                       data_type=dtype)
    if name.startswith('V_DIV_SCALE_'):
        _, dtype = _split_dtype(name)
        return InstructionSemantics(name, 'vector_div_scale', data_type=dtype)
    if name.startswith('V_DIV_FMAS_'):
        _, dtype = _split_dtype(name)
        return InstructionSemantics(name, 'vector_div_fmas', data_type=dtype)

    # LERP
    if name == 'V_LERP_U8':
        return InstructionSemantics(name, 'vector_ternary',
                                   operation='lerp_u8', data_type='u32')

    # SAD (sum of absolute differences) - ternary: D = |S0 - S1| + S2
    _SAD_MAP = {
        'V_SAD_U8': 'sad_u8', 'V_SAD_HI_U8': 'sad_hi_u8',
        'V_SAD_U16': 'sad_u16', 'V_SAD_U32': 'sad_u32',
        'V_MSAD_U8': 'msad_u8',
    }
    if name in _SAD_MAP:
        return InstructionSemantics(name, 'vector_ternary',
                                   operation=_SAD_MAP[name], data_type='u32')

    # MQSAD (complex, rarely used) → nop
    if name in ('V_MQSAD_PK_U16_U8', 'V_MQSAD_U32_U8',
                'V_QSAD_PK_U16_U8'):
        return InstructionSemantics(name, 'nop')

    # Masked bit count
    if name == 'V_MBCNT_LO_U32_B32':
        return InstructionSemantics(name, 'vector_mbcnt',
                                   operation='lo', data_type='u32')
    if name == 'V_MBCNT_HI_U32_B32':
        return InstructionSemantics(name, 'vector_mbcnt',
                                   operation='hi', data_type='u32')

    # Mixed-precision multiply-add: D.u32 = S0.u16 * S1.u16 + S2.u32
    if name == 'V_MAD_U32_U16':
        return InstructionSemantics(name, 'vector_mad_32_16',
                                   data_type='u32')
    if name == 'V_MAD_I32_I16':
        return InstructionSemantics(name, 'vector_mad_32_16',
                                   data_type='i32')

    # Widening multiply-add: D.u64 = S0.u32 * S1.u32 + S2.u64
    if name == 'V_MAD_U64_U32':
        return InstructionSemantics(name, 'vector_mad_64_32',
                                   data_type='u64')
    if name == 'V_MAD_I64_I32':
        return InstructionSemantics(name, 'vector_mad_64_32',
                                   data_type='i64')

    # Pack/convert instructions
    if name == 'V_CVT_PK_U8_F32':
        return InstructionSemantics(name, 'vector_cvt_pk_u8_f32')
    if name == 'V_CVT_PKACCUM_U8_F32':
        return InstructionSemantics(name, 'vector_cvt_pk_u8_f32')
    if name == 'V_CVT_PKNORM_I16_F32':
        return InstructionSemantics(name, 'vector_cvt_pknorm',
                                   operation='i16', data_type='f32')
    if name == 'V_CVT_PKNORM_U16_F32':
        return InstructionSemantics(name, 'vector_cvt_pknorm',
                                   operation='u16', data_type='f32')
    if name == 'V_CVT_PKRTZ_F16_F32':
        return InstructionSemantics(name, 'vector_cvt_pkrtz_f16_f32')
    if name == 'V_CVT_PK_U16_U32':
        return InstructionSemantics(name, 'vector_cvt_pk',
                                   operation='u16_u32')
    if name == 'V_CVT_PK_I16_I32':
        return InstructionSemantics(name, 'vector_cvt_pk',
                                   operation='i16_i32')

    # Trig preop → nop (internal microcode helper)
    if name == 'V_TRIG_PREOP_F64':
        return InstructionSemantics(name, 'nop')

    # ── CDNA4 / RDNA4 new conversions ─────────────────────────────────
    if name == 'V_CVT_F32_BF16':
        return InstructionSemantics(name, 'vector_unary',
                                   operation='cvt_f32_bf16', data_type='f32')
    if name == 'V_CVT_PK_F16_F32':
        return InstructionSemantics(name, 'vector_cvt_pk_f16_f32')
    if name == 'V_CVT_PK_BF16_F32':
        return InstructionSemantics(name, 'vector_cvt_pk_bf16_f32')
    if name == 'V_CVT_SR_F16_F32':
        return InstructionSemantics(name, 'vector_cvt_sr_f16_f32')
    if name == 'V_CVT_SR_BF16_F32':
        return InstructionSemantics(name, 'vector_cvt_sr_bf16_f32')
    if name == 'V_DOT2C_F32_BF16':
        return InstructionSemantics(name, 'vector_dot2c_bf16')
    if name == 'V_MINIMUM3_F32':
        return InstructionSemantics(name, 'vector_ternary',
                                   operation='minimum3', data_type='f32')
    if name == 'V_MAXIMUM3_F32':
        return InstructionSemantics(name, 'vector_ternary',
                                   operation='maximum3', data_type='f32')
    if name in ('V_BITOP3_B32', 'V_BITOP3_B16'):
        dt = 'b16' if name.endswith('B16') else 'b32'
        return InstructionSemantics(name, 'vector_bitop3', data_type=dt)
    if name in ('V_ASHR_PK_I8_I32', 'V_ASHR_PK_U8_I32'):
        return InstructionSemantics(name, 'nop')  # TODO: packed shift-right
    # V_PK_MINIMUM3/MAXIMUM3 are VOP3P, handled in _derive_vop3p.

    # ── Additional VOP3 ternary instructions ───────────────────────────
    if name == 'V_XOR3_B32':
        return InstructionSemantics(name, 'vector_ternary',
                                   operation='xor3', data_type='b32')
    if name == 'V_FMA_DX9_ZERO_F32':
        return InstructionSemantics(name, 'vector_ternary',
                                   operation='fma', data_type='f32')
    if name == 'V_MULLIT_F32':
        return InstructionSemantics(name, 'nop')

    # MAXMIN/MINMAX/MINIMUMMAXIMUM/MAXIMUMMINIMUM variants (all types)
    _CLAMP_TERNARY = {
        'V_MAXMIN': 'maxmin', 'V_MINMAX': 'minmax',
        'V_MINIMUMMAXIMUM': 'minimummaximum',
        'V_MAXIMUMMINIMUM': 'maximumminimum',
        'V_MAXMIN_NUM': 'maxmin_num', 'V_MINMAX_NUM': 'minmax_num',
    }
    for prefix, op in _CLAMP_TERNARY.items():
        for suffix in ('_B32', '_U32', '_I32', '_F32', '_F16', '_F64'):
            if name == f'{prefix}{suffix}':
                _, dt = _split_dtype(name)
                return InstructionSemantics(name, 'vector_ternary',
                                           operation=op, data_type=dt)

    # MIN3/MAX3/MED3 F16 and NUM variants
    if name == 'V_MINIMUM3_F16':
        return InstructionSemantics(name, 'vector_ternary',
                                   operation='minimum3', data_type='f16')
    if name == 'V_MAXIMUM3_F16':
        return InstructionSemantics(name, 'vector_ternary',
                                   operation='maximum3', data_type='f16')
    for op in ('MIN3', 'MAX3', 'MED3'):
        for suffix in ('_F32', '_F16'):
            if name == f'V_{op}_NUM{suffix}':
                _, dt = _split_dtype(name)
                return InstructionSemantics(name, 'vector_ternary',
                                           operation=op.lower(), data_type=dt)

    # IEEE 754-2019 binary min/max (VOP3 forms)
    if name.startswith('V_MINIMUM_') or name.startswith('V_MAXIMUM_'):
        _, dt = _split_dtype(name)
        op = 'fmin' if 'MINIMUM' in name else 'fmax'
        return InstructionSemantics(name, 'vector_binop',
                                   operation=op, data_type=dt)

    # Dot product with F16/BF16 output
    if name in ('V_DOT2_F16_F16', 'V_DOT2_BF16_BF16'):
        return InstructionSemantics(name, 'nop')

    # Pack/convert variants
    if name == 'V_CVT_PK_I16_F32':
        return InstructionSemantics(name, 'vector_cvt_pk',
                                   operation='i16_f32')
    if name == 'V_CVT_PK_U16_F32':
        return InstructionSemantics(name, 'vector_cvt_pk',
                                   operation='u16_f32')
    if name == 'V_CVT_PK_RTZ_F16_F32':
        return InstructionSemantics(name, 'vector_cvt_pkrtz_f16_f32')
    for norm_suffix in ('_I16_F16', '_U16_F16', '_I16_F32', '_U16_F32'):
        if name == f'V_CVT_PK_NORM{norm_suffix}':
            op = norm_suffix[1:3].lower() + '16'  # i16 or u16
            return InstructionSemantics(name, 'vector_cvt_pknorm',
                                       operation=op[:3],
                                       data_type=norm_suffix[-3:].lower())

    # 64-bit multiply-add with carry
    if name == 'V_MAD_CO_U64_U32':
        return InstructionSemantics(name, 'vector_mad_64_32',
                                   data_type='u64')
    if name == 'V_MAD_CO_I64_I32':
        return InstructionSemantics(name, 'vector_mad_64_32',
                                   data_type='i64')

    if name == 'V_PERMLANE16_B32':
        return InstructionSemantics(name, 'vector_permlane16', operation='imm')
    if name == 'V_PERMLANEX16_B32':
        return InstructionSemantics(name, 'vector_permlanex16', operation='imm')
    if name == 'V_PERMLANE16_VAR_B32':
        return InstructionSemantics(name, 'vector_permlane16', operation='var')
    if name == 'V_PERMLANEX16_VAR_B32':
        return InstructionSemantics(name, 'vector_permlanex16', operation='var')

    if name == 'V_PACK_B32_F16':
        sem = InstructionSemantics(name, 'vector_pack_b32_f16')
        sem.semantic_class = 'vector_pack_b32_f16'
        return sem

    # FP8/BF8 pack/convert (non-scaled, CDNA3/4)
    _FP8_PATTERNS = (
        'V_CVT_PK_FP8_F32', 'V_CVT_PK_BF8_F32',
        'V_CVT_SR_FP8_F32', 'V_CVT_SR_BF8_F32',
        'V_CVT_PKNORM_I16_F16', 'V_CVT_PKNORM_U16_F16',
    )
    if name in _FP8_PATTERNS:
        return InstructionSemantics(name, 'nop')

    # CDNA4 scaled FP8/BF8/FP6/FP4 conversions
    if 'CVT_SCALEF32' in name.upper():
        return InstructionSemantics(name, 'nop')  # TODO: Phase 3 scaled conversions

    # V_PK_FMAC_F16 VOP2 form — the VOP3P form is handled via _VOP3P_PK16_MAP.
    if name == 'V_PK_FMAC_F16':
        return InstructionSemantics(name, 'nop')

    # Legacy interpolation (CDNA1, RDNA1/2) — niche, not simulated
    if name.startswith('V_INTERP_P1') or name.startswith('V_INTERP_P2'):
        return InstructionSemantics(name, 'nop')

    # V_DOT2ACC VOP2 form
    if name == 'V_DOT2ACC_F32_F16':
        return InstructionSemantics(name, 'nop')

    # 64-bit shift/add (VOP3-only)
    if name == 'V_LSHL_ADD_U64':
        return InstructionSemantics(name, 'vector_ternary',
                                   operation='lshl_add', data_type='u64')
    if name == 'V_LSHLREV_B64':
        return InstructionSemantics(name, 'vector_binop',
                                   operation='shl', data_type='b64')
    if name == 'V_LSHRREV_B64':
        return InstructionSemantics(name, 'vector_binop',
                                   operation='shr', data_type='b64')
    if name == 'V_ASHRREV_I64':
        return InstructionSemantics(name, 'vector_binop',
                                   operation='ashr', data_type='i64')

    stem, dtype = _split_dtype(name)
    op = _VOP3_TERNARY_MAP.get(stem)
    if op is not None and dtype is not None:
        # Integer-logical ternary ops use u32 codegen path even when
        # the mnemonic suffix is B32.
        _TERNARY_U32_OPS = frozenset({
            'add3', 'lshl_or', 'and_or', 'or3', 'lshl_add', 'add_lshl',
            'xad', 'perm',
        })
        if dtype == 'b32' and op in _TERNARY_U32_OPS:
            dtype = 'u32'
        return InstructionSemantics(name, 'vector_ternary',
                                   operation=op, data_type=dtype)

    # MAD with 24-bit types: V_MAD_I32_I24, V_MAD_U32_U24
    if name == 'V_MAD_I32_I24':
        return InstructionSemantics(name, 'vector_ternary',
                                   operation='mad', data_type='i24')
    if name == 'V_MAD_U32_U24':
        return InstructionSemantics(name, 'vector_ternary',
                                   operation='mad', data_type='u24')

    return None

# Packed 16-bit operation map: stem → (semantic_class, operation, data_type)
_VOP3P_PK16_MAP = {
    'V_PK_ADD_I16': ('pk_binop', 'add', 'i16'),
    'V_PK_SUB_I16': ('pk_binop', 'sub', 'i16'),
    'V_PK_MUL_LO_U16': ('pk_binop', 'mul', 'u16'),
    'V_PK_ADD_U16': ('pk_binop', 'add', 'u16'),
    'V_PK_SUB_U16': ('pk_binop', 'sub', 'u16'),
    'V_PK_MAX_I16': ('pk_binop', 'max', 'i16'),
    'V_PK_MIN_I16': ('pk_binop', 'min', 'i16'),
    'V_PK_MAX_U16': ('pk_binop', 'max', 'u16'),
    'V_PK_MIN_U16': ('pk_binop', 'min', 'u16'),
    'V_PK_LSHLREV_B16': ('pk_binop', 'shl', 'u16'),
    'V_PK_LSHRREV_B16': ('pk_binop', 'shr', 'u16'),
    'V_PK_ASHRREV_I16': ('pk_binop', 'ashr', 'i16'),
    'V_PK_MAD_I16': ('pk_ternary', 'mad', 'i16'),
    'V_PK_MAD_U16': ('pk_ternary', 'mad', 'u16'),
    'V_PK_FMA_F16': ('pk_ternary', 'fma', 'f16'),
    'V_PK_ADD_F16': ('pk_binop', 'add', 'f16'),
    'V_PK_MUL_F16': ('pk_binop', 'mul', 'f16'),
    'V_PK_MIN_F16': ('pk_binop', 'min', 'f16'),
    'V_PK_MAX_F16': ('pk_binop', 'max', 'f16'),
    'V_PK_MIN_NUM_F16': ('pk_binop', 'min', 'f16'),
    'V_PK_MAX_NUM_F16': ('pk_binop', 'max', 'f16'),
    'V_PK_MINIMUM_F16': ('pk_binop', 'min', 'f16'),
    'V_PK_MAXIMUM_F16': ('pk_binop', 'max', 'f16'),
    'V_PK_FMAC_F16': ('pk_ternary', 'fmac', 'f16'),
}

def _derive_vop3p(name: str) -> InstructionSemantics | None:
    """Derive semantics for a VOP3P (Vector Packed/Matrix) instruction."""
    # Packed 16-bit ops
    pk16 = _VOP3P_PK16_MAP.get(name)
    if pk16 is not None:
        cls, op, dt = pk16
        return InstructionSemantics(name, cls, operation=op, data_type=dt)

    # DOT2ACC with F32 output and F16 input
    if name == 'V_DOT2ACC_F32_F16':
        return InstructionSemantics(name, 'dot2_f32_f16')

    # Packed F32 ops (VGPR pairs)
    if name == 'V_PK_FMA_F32':
        return InstructionSemantics(name, 'pk_ternary_f32',
                                   operation='fma', data_type='f32')
    if name == 'V_PK_MUL_F32':
        return InstructionSemantics(name, 'pk_binop_f32',
                                   operation='mul', data_type='f32')
    if name == 'V_PK_ADD_F32':
        return InstructionSemantics(name, 'pk_binop_f32',
                                   operation='add', data_type='f32')
    if name == 'V_PK_MOV_B32':
        return InstructionSemantics(name, 'pk_mov_b32', accvgpr_srcs=True)

    # Packed min3/max3 (CDNA4 / RDNA4)
    if name in ('V_PK_MINIMUM3_F16', 'V_PK_MAXIMUM3_F16'):
        op = 'minimum3' if 'MINIMUM' in name else 'maximum3'
        return InstructionSemantics(name, 'pk_ternary',
                                   operation=op, data_type='f16')

    # Mixed-precision MAD_MIX
    if name == 'V_MAD_MIX_F32':
        return InstructionSemantics(name, 'mad_mix_f32')
    if name == 'V_MAD_MIXLO_F16':
        return InstructionSemantics(name, 'mad_mixlo_f16')
    if name == 'V_MAD_MIXHI_F16':
        return InstructionSemantics(name, 'mad_mixhi_f16')

    # Dot products
    if name == 'V_DOT2_F32_F16':
        return InstructionSemantics(name, 'dot2_f32_f16')
    if name == 'V_DOT2_I32_I16':
        return InstructionSemantics(name, 'dot2_i32_i16')
    if name == 'V_DOT2_U32_U16':
        return InstructionSemantics(name, 'dot2_u32_u16')
    if name == 'V_DOT4_I32_I8':
        return InstructionSemantics(name, 'dot4_i32_i8')
    if name == 'V_DOT4_U32_U8':
        return InstructionSemantics(name, 'dot4_u32_u8')
    # FP8 dot products (RDNA4)
    if name in ('V_DOT4_F32_FP8_FP8', 'V_DOT4_F32_FP8_BF8',
                'V_DOT4_F32_BF8_FP8', 'V_DOT4_F32_BF8_BF8'):
        return InstructionSemantics(name, 'dot4_f32_fp8')
    if name == 'V_DOT8_I32_I4':
        return InstructionSemantics(name, 'dot8_i32_i4')
    if name == 'V_DOT8_U32_U4':
        return InstructionSemantics(name, 'dot8_u32_u4')

    # ACCVGPR read/write
    if name == 'V_ACCVGPR_READ':
        return InstructionSemantics(name, 'accvgpr_read')
    if name == 'V_ACCVGPR_WRITE':
        return InstructionSemantics(name, 'accvgpr_write')

    # Additional dot product variants (RDNA3/4 naming)
    if name == 'V_DOT2_F32_BF16':
        return InstructionSemantics(name, 'dot2_f32_f16')  # BF16 uses same dot2 pattern
    if name == 'V_DOT4_I32_IU8':
        return InstructionSemantics(name, 'dot4_i32_i8')   # IU8 = signed/unsigned mixed
    if name == 'V_DOT8_I32_IU4':
        return InstructionSemantics(name, 'dot8_i32_i4')   # IU4 = signed/unsigned mixed

    # FMA_MIX variants (RDNA3/4 renamed from MAD_MIX)
    if name == 'V_FMA_MIX_F32':
        return InstructionSemantics(name, 'mad_mix_f32')
    if name == 'V_FMA_MIXLO_F16':
        return InstructionSemantics(name, 'mad_mixlo_f16')
    if name == 'V_FMA_MIXHI_F16':
        return InstructionSemantics(name, 'mad_mixhi_f16')

    # MFMA / SMFMAC - all map to 'mfma' semantic class
    if name.startswith('V_MFMA_') or name.startswith('V_SMFMAC_'):
        return InstructionSemantics(name, 'mfma')

    # WMMA (Wave Matrix Multiply-Accumulate) — RDNA3/3.5/4
    import re
    m = re.match(r'V_(?:S?WMMA[C]?)_(F32|F16|BF16|I32|FP8|BF8)_'
                 r'(\d+)X(\d+)X(\d+)_?(F16|BF16|IU8|IU4|FP8|BF8'
                 r'|FP8_FP8|FP8_BF8|BF8_FP8|BF8_BF8'
                 r'|F16_FP8|F16_BF8|BF16_FP8|BF16_BF8)?$', name)
    if m:
        return InstructionSemantics(name, 'mfma')  # Reuse MFMA semantic class — same matrix pattern

    # SWMMAC variants
    if name.startswith('V_SWMMAC_'):
        return InstructionSemantics(name, 'mfma')

    return None

_SMEM_DWORD_MAP = {
    'DWORD': 1, 'DWORDX2': 2, 'DWORDX4': 4, 'DWORDX8': 8, 'DWORDX16': 16,
    # RDNA4 (GFX12) byte-width naming:
    'B32': 1, 'B64': 2, 'B128': 4, 'B256': 8, 'B512': 16,
}

def _derive_smem(name: str) -> InstructionSemantics | None:
    """Derive semantics for an SMEM (Scalar Memory) instruction."""
    upper = name.upper()
    # Cache invalidation / writeback instructions.
    if upper in ('S_DCACHE_INV', 'S_DCACHE_INV_VOL'):
        return InstructionSemantics(name, 'dcache_inv')
    if upper in ('S_DCACHE_WB', 'S_DCACHE_WB_VOL'):
        return InstructionSemantics(name, 'dcache_wb')
    if upper == 'S_GL1_INV':
        return InstructionSemantics(name, 'gl1_inv')
    # Remaining scalar cache / special instructions.
    if 'DCACHE' in upper or upper in ('S_MEMTIME', 'S_MEMREALTIME',
                                       'S_ATC_PROBE',
                                       'S_ATC_PROBE_BUFFER'):
        return InstructionSemantics(name, 'nop')

    # S_ATOMIC_* are scalar atomics — not currently simulated.
    if '_ATOMIC_' in upper:
        return InstructionSemantics(name, 'nop')

    is_store = '_STORE_' in upper or '_SCRATCH_STORE_' in upper
    for suffix, ndw in _SMEM_DWORD_MAP.items():
        if upper.endswith(suffix):
            cls = 'smem_store' if is_store else 'smem_load'
            return InstructionSemantics(name, cls, num_elems=ndw)
    # BUFFER_WBL2, BUFFER_INV, etc. — cache control for SMEM buffer paths.
    if upper in ('BUFFER_WBL2', 'BUFFER_INV', 'BUFFER_GL0_INV', 'BUFFER_GL1_INV',
                 'S_BUFFER_GL0_INV', 'S_BUFFER_GL1_INV'):
        return InstructionSemantics(name, 'dcache_inv')
    return InstructionSemantics(name, 'nop')

_FLAT_DATA_MAP: dict[str, tuple[int, int, bool]] = {
    # CDNA / RDNA1-3 suffixes (DWORD-based naming).
    'UBYTE': (1, 1, False), 'SBYTE': (1, 1, True),
    'USHORT': (2, 1, False), 'SSHORT': (2, 1, True),
    'SHORT': (2, 1, False),
    'BYTE': (1, 1, False),
    'BYTE_D16_HI': (1, 1, False),
    'SHORT_D16_HI': (2, 1, False),
    'DWORD': (4, 1, False),
    'DWORDX2': (4, 2, False), 'DWORDX3': (4, 3, False),
    'DWORDX4': (4, 4, False),
    'UBYTE_D16': (1, 1, False), 'UBYTE_D16_HI': (1, 1, False),
    'SBYTE_D16': (1, 1, True), 'SBYTE_D16_HI': (1, 1, True),
    'SHORT_D16': (2, 1, False),
    # RDNA4 suffixes (byte-width naming: B32/B64/B96/B128 instead of
    # DWORD/DWORDX2/DWORDX3/DWORDX4, and U8/I8/U16/I16 instead of
    # UBYTE/SBYTE/USHORT/SSHORT).
    'B32': (4, 1, False), 'B64': (4, 2, False),
    'B96': (4, 3, False), 'B128': (4, 4, False),
    'U8': (1, 1, False), 'I8': (1, 1, True),
    'U16': (2, 1, False), 'I16': (2, 1, True),
    'B8': (1, 1, False), 'B16': (2, 1, False),
    'D16_U8': (1, 1, False), 'D16_I8': (1, 1, True),
    'D16_B16': (2, 1, False),
    'D16_HI_U8': (1, 1, False), 'D16_HI_I8': (1, 1, True),
    'D16_HI_B16': (2, 1, False), 'D16_HI_B8': (1, 1, False),
}

_FLAT_ATOMIC_OPS: dict[str, tuple[str, int]] = {
    # op_suffix -> (operation, data_dwords for store_data)
    # Integer atomics.
    'SWAP': ('swap', 1),
    'CMPSWAP': ('cmpswap', 2),  # src + cmp
    'FCMPSWAP': ('cmpswap', 2),  # FP compare-and-swap
    'ADD': ('add', 1),
    'SUB': ('sub', 1),
    'SMIN': ('smin', 1),
    'UMIN': ('umin', 1),
    'SMAX': ('smax', 1),
    'UMAX': ('umax', 1),
    'AND': ('and', 1),
    'OR': ('or', 1),
    'XOR': ('xor', 1),
    'INC': ('inc', 1),
    'DEC': ('dec', 1),
    'CSUB': ('sub', 1),
    'SUB_CLAMP': ('sub', 1),
    # Floating-point atomics.
    'ADD_F32': ('fadd', 1),
    'ADD_F64': ('fadd', 2),
    'MIN_F64': ('fmin', 2),
    'MAX_F64': ('fmax', 2),
    'FMIN': ('fmin', 1),
    'FMAX': ('fmax', 1),
    'MIN_F32': ('fmin', 1), 'MAX_F32': ('fmax', 1),
    'MIN_F64': ('fmin', 2), 'MAX_F64': ('fmax', 2),
    'MIN_NUM_F32': ('fmin', 1), 'MAX_NUM_F32': ('fmax', 1),
    'MIN_NUM_F64': ('fmin', 2), 'MAX_NUM_F64': ('fmax', 2),
    'COND_SUB': ('sub', 1),
    'ORDERED_ADD': ('add', 2),
    # RDNA3+ typed MIN/MAX (suffix stripped from the full instruction name).
    'MIN_I32': ('smin', 1), 'MIN_U32': ('umin', 1),
    'MAX_I32': ('smax', 1), 'MAX_U32': ('umax', 1),
    'MIN_I64': ('smin', 2), 'MIN_U64': ('umin', 2),
    'MAX_I64': ('smax', 2), 'MAX_U64': ('umax', 2),
    # Packed FP atomics (treated as 32-bit fadd for now).
    'PK_ADD_F16': ('fadd', 1),
    'PK_ADD_BF16': ('fadd', 1),
}

def _derive_flat(name: str) -> InstructionSemantics | None:
    """Derive semantics for a FLAT (Flat/Global/Scratch memory) instruction."""
    upper = name.upper()
    if '_ATOMIC_' in upper:
        # Extract the operation suffix after _ATOMIC_.
        # Strip _X2 (64-bit variant) and type suffixes (_B32, _U32, _I32, etc.).
        for prefix in ('FLAT_ATOMIC_', 'GLOBAL_ATOMIC_', 'SCRATCH_ATOMIC_'):
            if upper.startswith(prefix):
                suffix = upper[len(prefix):]
                is_x2 = suffix.endswith('_X2')
                if is_x2:
                    suffix = suffix[:-3]
                # Try exact match first (handles ADD_F32, PK_ADD_F16, etc.).
                info = _FLAT_ATOMIC_OPS.get(suffix)
                if not info:
                    # Strip type suffix (_B32, _U32, _I32, _F32, _B64, etc.).
                    for tsuf in ('_B32', '_U32', '_I32', '_F32', '_B64',
                                 '_U64', '_I64', '_F64'):
                        if suffix.endswith(tsuf):
                            info = _FLAT_ATOMIC_OPS.get(suffix[:len(suffix) - len(tsuf)])
                            break
                if info:
                    op, data_dw = info
                    elem_size = 8 if is_x2 else 4
                    data_dw_actual = data_dw * (2 if is_x2 else 1)
                    return InstructionSemantics(name, 'flat_atomic',
                                                operation=op,
                                                elem_size=elem_size,
                                                num_elems=data_dw_actual)
        # FP atomics and other unrecognized patterns.
        return InstructionSemantics(name, 'flat_atomic')

    is_store = '_STORE_' in upper
    # Strip prefix: FLAT_LOAD_, FLAT_STORE_, GLOBAL_LOAD_, SCRATCH_LOAD_, etc.
    for prefix in ('FLAT_LOAD_', 'FLAT_STORE_', 'GLOBAL_LOAD_', 'GLOBAL_STORE_',
                   'SCRATCH_LOAD_', 'SCRATCH_STORE_'):
        if upper.startswith(prefix):
            suffix = upper[len(prefix):]
            info = _FLAT_DATA_MAP.get(suffix)
            if info:
                esz, ne, se = info
                cls = 'flat_store' if is_store else 'flat_load'
                return InstructionSemantics(name, cls, elem_size=esz,
                                            num_elems=ne, sign_extend=se,
                                            d16_hi='D16_HI' in suffix,
                                            d16_lo='D16' in suffix and 'D16_HI' not in suffix)
    return InstructionSemantics(name, 'nop')

_BUFFER_FORMAT_MAP: dict[str, tuple[int, int]] = {
    'FORMAT_X': (4, 1), 'FORMAT_XY': (4, 2),
    'FORMAT_XYZ': (4, 3), 'FORMAT_XYZW': (4, 4),
    'FORMAT_D16_X': (2, 1), 'FORMAT_D16_XY': (2, 2),
    'FORMAT_D16_XYZ': (2, 3), 'FORMAT_D16_XYZW': (2, 4),
}

def _derive_mubuf(name: str) -> InstructionSemantics | None:
    """Derive semantics for a MUBUF (Untyped Buffer memory) instruction."""
    upper = name.upper()
    # Buffer cache control instructions.
    if upper in ('BUFFER_WBINVL1', 'BUFFER_WBINVL1_SC', 'BUFFER_WBINVL1_VOL',
                 'BUFFER_GL0_INV', 'BUFFER_GL1_INV',
                 'BUFFER_WBL2', 'BUFFER_INV'):
        return InstructionSemantics(name, 'dcache_inv')
    if '_ATOMIC_' in upper:
        for prefix in ('BUFFER_ATOMIC_',):
            if upper.startswith(prefix):
                suffix = upper[len(prefix):]
                is_x2 = suffix.endswith('_X2')
                if is_x2:
                    suffix = suffix[:-3]
                info = _FLAT_ATOMIC_OPS.get(suffix)
                if not info:
                    for tsuf in ('_B32', '_U32', '_I32', '_F32', '_B64',
                                 '_U64', '_I64', '_F64'):
                        if suffix.endswith(tsuf):
                            info = _FLAT_ATOMIC_OPS.get(suffix[:len(suffix) - len(tsuf)])
                            break
                if info:
                    op, data_dw = info
                    elem_size = 8 if is_x2 else 4
                    data_dw_actual = data_dw * (2 if is_x2 else 1)
                    return InstructionSemantics(name, 'buffer_atomic',
                                                operation=op,
                                                elem_size=elem_size,
                                                num_elems=data_dw_actual)
        return InstructionSemantics(name, 'buffer_atomic')

    is_store = '_STORE_' in upper
    for prefix in ('BUFFER_LOAD_', 'BUFFER_STORE_'):
        if upper.startswith(prefix):
            suffix = upper[len(prefix):]
            info = _FLAT_DATA_MAP.get(suffix)
            if info:
                esz, ne, se = info
                cls = 'buffer_store' if is_store else 'buffer_load'
                return InstructionSemantics(name, cls, elem_size=esz,
                                            num_elems=ne, sign_extend=se,
                                            d16_hi='D16_HI' in suffix,
                                            d16_lo='D16' in suffix and 'D16_HI' not in suffix)
    return InstructionSemantics(name, 'nop')

_MTBUF_FORMAT_MAP: dict[str, tuple[int, int, bool]] = {
    'FORMAT_X':    (4, 1, False),
    'FORMAT_XY':   (4, 2, False),
    'FORMAT_XYZ':  (4, 3, False),
    'FORMAT_XYZW': (4, 4, False),
    'FORMAT_D16_X':    (2, 1, False),
    'FORMAT_D16_XY':   (2, 2, False),
    'FORMAT_D16_XYZ':  (2, 3, False),
    'FORMAT_D16_XYZW': (2, 4, False),
}

def _derive_mtbuf(name: str) -> InstructionSemantics | None:
    """Derive semantics for an MTBUF (Typed Buffer memory) instruction."""
    upper = name.upper()
    is_store = '_STORE_' in upper
    for prefix in ('TBUFFER_LOAD_', 'TBUFFER_STORE_'):
        if upper.startswith(prefix):
            suffix = upper[len(prefix):]
            info = _FLAT_DATA_MAP.get(suffix) or _MTBUF_FORMAT_MAP.get(suffix)
            if info:
                esz, ne, se = info
                cls = 'tbuffer_store' if is_store else 'tbuffer_load'
                return InstructionSemantics(name, cls, elem_size=esz,
                                            num_elems=ne, sign_extend=se,
                                            d16_hi='D16_HI' in suffix,
                                            d16_lo='D16' in suffix and 'D16_HI' not in suffix)
    return InstructionSemantics(name, 'nop')

_DS_DATA_MAP: dict[str, tuple[int, int]] = {
    'B32': (4, 1), 'B64': (8, 1), 'B128': (4, 4),
    'U8': (1, 1), 'I8': (1, 1), 'U16': (2, 1), 'I16': (2, 1),
    'D16_HI_B32': (2, 1),
    'B96': (4, 3),
    '2ADDR_STRIDE64_B32': (4, 1),
    '2ADDR_STRIDE64_B64': (8, 1),
    # RDNA4 adds B8/B16 stores and D16 packing variants.
    'B8': (1, 1), 'B16': (2, 1),
    'U8_D16': (1, 1), 'U8_D16_HI': (1, 1),
    'I8_D16': (1, 1), 'I8_D16_HI': (1, 1),
    'U16_D16': (2, 1), 'U16_D16_HI': (2, 1),
    'B8_D16_HI': (1, 1), 'B16_D16_HI': (2, 1),
}

def _derive_ds(name: str) -> InstructionSemantics | None:
    """Derive semantics for a DS (Data Share / LDS) instruction.

    Handles both CDNA/RDNA1-3 naming (``DS_READ_*``/``DS_WRITE_*``) and
    RDNA4 naming (``DS_LOAD_*``/``DS_STORE_*``). RDNA4 also renames
    ``DS_STOREXCHG`` (was ``DS_WRXCHG``) and ``DS_CMPSTORE``
    (was ``DS_CMPST``), which are handled by the atomic fallthrough.
    """
    upper = name.upper()
    is_write2 = '_WRITE2' in upper or 'DS_STORE_2ADDR' in upper
    is_read2 = '_READ2' in upper or 'DS_LOAD_2ADDR' in upper
    is_write = ('_WRITE_' in upper or 'DS_STORE_' in upper) and not is_write2
    is_read = ('_READ_' in upper or 'DS_LOAD_' in upper) and not is_read2
    is_st64 = 'ST64' in upper or 'STRIDE64' in upper
    if is_write2:
        esz = 8 if 'B64' in upper else 4
        return InstructionSemantics(name, 'ds_write2', elem_size=esz,
                                    num_elems=1,
                                    operation='st64' if is_st64 else None)
    if is_read2:
        esz = 8 if 'B64' in upper else 4
        return InstructionSemantics(name, 'ds_read2', elem_size=esz,
                                    num_elems=1,
                                    operation='st64' if is_st64 else None)
    if is_write:
        for suffix, (esz, ne) in _DS_DATA_MAP.items():
            if upper.endswith(suffix):
                return InstructionSemantics(name, 'ds_write', elem_size=esz,
                                            num_elems=ne,
                                            d16_hi='D16_HI' in suffix)
        return InstructionSemantics(name, 'ds_write', elem_size=4, num_elems=1)
    # Transpose read instructions (CDNA4): cross-lane data shuffling.
    if '_TR_' in upper and is_read:
        # ds_read_b64_tr_b16 → 64-bit result, 16-bit elements (4x FP16)
        # ds_read_b64_tr_b8  → 64-bit result, 8-bit elements (8x INT8)
        # ds_read_b64_tr_b4  → 64-bit result, 4-bit elements (16x INT4)
        # ds_read_b96_tr_b6  → 96-bit result, 6-bit elements
        if 'B64_TR_B16' in upper:
            return InstructionSemantics(name, 'ds_read_tr_b16', elem_size=2, num_elems=4)
        if 'B64_TR_B8' in upper:
            return InstructionSemantics(name, 'ds_read_tr_b8', elem_size=1, num_elems=8)
        if 'B64_TR_B4' in upper:
            return InstructionSemantics(name, 'ds_read_tr_b4', elem_size=1, num_elems=8)
        if 'B96_TR_B6' in upper:
            return InstructionSemantics(name, 'ds_read_tr_b6', elem_size=1, num_elems=12)
        return InstructionSemantics(name, 'nop')
    # ADDTID instructions: address = thread_id * M0_stride + offset.
    if 'ADDTID' in upper:
        if is_write:
            return InstructionSemantics(name, 'ds_write_addtid', elem_size=4, num_elems=1)
        if is_read:
            return InstructionSemantics(name, 'ds_read_addtid', elem_size=4, num_elems=1)
        return InstructionSemantics(name, 'nop')
    if is_read:
        for suffix, (esz, ne) in _DS_DATA_MAP.items():
            if upper.endswith(suffix):
                se = suffix.startswith('I')
                return InstructionSemantics(name, 'ds_read', elem_size=esz,
                                            num_elems=ne, sign_extend=bool(se),
                                            d16_hi='D16_HI' in suffix,
                                            d16_lo='D16' in suffix and 'D16_HI' not in suffix)
        return InstructionSemantics(name, 'ds_read', elem_size=4, num_elems=1)
    # DS atomic operations — extract the specific op and data width.
    # RTN variants use the same operation; the codegen sets is_load based
    # on whether vdst is an explicit destination (num_dst_ > 0).
    _DS_ATOMIC_MAP: dict[str, tuple[str, int, int]] = {
        # keyword -> (operation, elem_size, data_dwords)
        '_ADD_U32': ('add', 4, 1), '_ADD_U64': ('add', 8, 2),
        '_ADD_RTN_U32': ('add', 4, 1), '_ADD_RTN_U64': ('add', 8, 2),
        '_SUB_U32': ('sub', 4, 1), '_SUB_U64': ('sub', 8, 2),
        '_SUB_RTN_U32': ('sub', 4, 1), '_SUB_RTN_U64': ('sub', 8, 2),
        '_RSUB_U32': ('rsub', 4, 1), '_RSUB_U64': ('rsub', 8, 2),
        '_RSUB_RTN_U32': ('rsub', 4, 1), '_RSUB_RTN_U64': ('rsub', 8, 2),
        '_MIN_I32': ('smin', 4, 1), '_MIN_I64': ('smin', 8, 2),
        '_MIN_RTN_I32': ('smin', 4, 1), '_MIN_RTN_I64': ('smin', 8, 2),
        '_MIN_U32': ('umin', 4, 1), '_MIN_U64': ('umin', 8, 2),
        '_MIN_RTN_U32': ('umin', 4, 1), '_MIN_RTN_U64': ('umin', 8, 2),
        '_MAX_I32': ('smax', 4, 1), '_MAX_I64': ('smax', 8, 2),
        '_MAX_RTN_I32': ('smax', 4, 1), '_MAX_RTN_I64': ('smax', 8, 2),
        '_MAX_U32': ('umax', 4, 1), '_MAX_U64': ('umax', 8, 2),
        '_MAX_RTN_U32': ('umax', 4, 1), '_MAX_RTN_U64': ('umax', 8, 2),
        '_AND_B32': ('and', 4, 1), '_AND_B64': ('and', 8, 2),
        '_AND_RTN_B32': ('and', 4, 1), '_AND_RTN_B64': ('and', 8, 2),
        '_OR_B32': ('or', 4, 1), '_OR_B64': ('or', 8, 2),
        '_OR_RTN_B32': ('or', 4, 1), '_OR_RTN_B64': ('or', 8, 2),
        '_XOR_B32': ('xor', 4, 1), '_XOR_B64': ('xor', 8, 2),
        '_XOR_RTN_B32': ('xor', 4, 1), '_XOR_RTN_B64': ('xor', 8, 2),
        '_INC_U32': ('inc', 4, 1), '_INC_U64': ('inc', 8, 2),
        '_INC_RTN_U32': ('inc', 4, 1), '_INC_RTN_U64': ('inc', 8, 2),
        '_DEC_U32': ('dec', 4, 1), '_DEC_U64': ('dec', 8, 2),
        '_DEC_RTN_U32': ('dec', 4, 1), '_DEC_RTN_U64': ('dec', 8, 2),
        '_SWAP_B32': ('swap', 4, 1), '_SWAP_B64': ('swap', 8, 2),
        '_SWAP_RTN_B32': ('swap', 4, 1), '_SWAP_RTN_B64': ('swap', 8, 2),
        '_WRXCHG_RTN_B32': ('swap', 4, 1), '_WRXCHG_RTN_B64': ('swap', 8, 2),
        '_WRXCHG2_RTN_B32': ('swap', 4, 1), '_WRXCHG2_RTN_B64': ('swap', 8, 2),
        '_STOREXCHG_RTN_B32': ('swap', 4, 1), '_STOREXCHG_RTN_B64': ('swap', 8, 2),
        '_CMPST_B32': ('cmpswap', 4, 2), '_CMPST_B64': ('cmpswap', 8, 4),
        '_CMPST_RTN_B32': ('cmpswap', 4, 2), '_CMPST_RTN_B64': ('cmpswap', 8, 4),
        '_CMPST_F32': ('cmpswap', 4, 2), '_CMPST_F64': ('cmpswap', 8, 4),
        '_CMPST_RTN_F32': ('cmpswap', 4, 2), '_CMPST_RTN_F64': ('cmpswap', 8, 4),
        '_CMPSTORE_B32': ('cmpswap', 4, 2), '_CMPSTORE_B64': ('cmpswap', 8, 4),
        '_CMPSTORE_RTN_B32': ('cmpswap', 4, 2), '_CMPSTORE_RTN_B64': ('cmpswap', 8, 4),
        '_CONDXCHG32_RTN_B64': ('cmpswap', 8, 4),
        '_ADD_F32': ('fadd', 4, 1), '_ADD_RTN_F32': ('fadd', 4, 1),
        '_ADD_F64': ('fadd', 8, 2), '_ADD_RTN_F64': ('fadd', 8, 2),
        '_MIN_F32': ('fmin', 4, 1), '_MIN_F64': ('fmin', 8, 2),
        '_MIN_RTN_F32': ('fmin', 4, 1), '_MIN_RTN_F64': ('fmin', 8, 2),
        '_MAX_F32': ('fmax', 4, 1), '_MAX_F64': ('fmax', 8, 2),
        '_MAX_RTN_F32': ('fmax', 4, 1), '_MAX_RTN_F64': ('fmax', 8, 2),
        '_MIN_NUM_F32': ('fmin', 4, 1), '_MIN_NUM_RTN_F32': ('fmin', 4, 1),
        '_MAX_NUM_F32': ('fmax', 4, 1), '_MAX_NUM_RTN_F32': ('fmax', 4, 1),
        '_MIN_NUM_F64': ('fmin', 8, 2), '_MIN_NUM_RTN_F64': ('fmin', 8, 2),
        '_MAX_NUM_F64': ('fmax', 8, 2), '_MAX_NUM_RTN_F64': ('fmax', 8, 2),
        '_SUB_CLAMP_U32': ('sub', 4, 1), '_SUB_CLAMP_RTN_U32': ('sub', 4, 1),
        '_CMPSTORE_F32': ('cmpswap', 4, 2), '_CMPSTORE_RTN_F32': ('cmpswap', 4, 2),
        '_CMPSTORE_F64': ('cmpswap', 8, 4), '_CMPSTORE_RTN_F64': ('cmpswap', 8, 4),
        '_WRXCHG2ST64_RTN_B32': ('swap', 4, 1), '_WRXCHG2ST64_RTN_B64': ('swap', 8, 2),
        '_STOREXCHG2ADDR_RTN_B32': ('swap', 4, 1),
        '_STOREXCHG2ADDR_STRIDE64_RTN_B32': ('swap', 4, 1),
        '_STOREXCHG2ADDR_RTN_B64': ('swap', 8, 2),
        '_STOREXCHG2ADDR_STRIDE64_RTN_B64': ('swap', 8, 2),
        '_STOREXCHG_2ADDR_RTN_B32': ('swap', 4, 1),
        '_STOREXCHG_2ADDR_STRIDE64_RTN_B32': ('swap', 4, 1),
        '_STOREXCHG_2ADDR_RTN_B64': ('swap', 8, 2),
        '_STOREXCHG_2ADDR_STRIDE64_RTN_B64': ('swap', 8, 2),
        '_PK_ADD_F16': ('fadd', 4, 1), '_PK_ADD_RTN_F16': ('fadd', 4, 1),
        '_PK_ADD_BF16': ('fadd', 4, 1), '_PK_ADD_RTN_BF16': ('fadd', 4, 1),
    }
    for suffix, (op, esz, dw) in _DS_ATOMIC_MAP.items():
        if suffix in upper:
            return InstructionSemantics(name, 'ds_atomic', operation=op,
                                        elem_size=esz, num_elems=dw)
    # ── Lane permutation / swizzle ──────────────────────────────────────
    if upper in ('DS_PERMUTE_B32', 'DS_BPERMUTE_B32'):
        return InstructionSemantics(name, 'ds_permute')
    if upper == 'DS_SWIZZLE_B32':
        return InstructionSemantics(name, 'ds_swizzle')
    # ── Explicitly classified as nop (not simulated) ────────────────────
    # GS register operations — must precede the atomic fallback because
    # DS_ADD_GS_REG_RTN / DS_SUB_GS_REG_RTN contain _ADD / _SUB.
    if upper in ('DS_ADD_GS_REG_RTN', 'DS_SUB_GS_REG_RTN'):
        return InstructionSemantics(name, 'nop')
    # GWS (Global Wave Sync) — hardware scheduling primitive, not needed
    # for compute simulation.
    if upper.startswith('DS_GWS_'):
        return InstructionSemantics(name, 'nop')
    # GDS append/consume counters — GDS not simulated.
    if upper in ('DS_CONSUME', 'DS_APPEND'):
        return InstructionSemantics(name, 'nop')
    # GDS ordered count.
    if upper == 'DS_ORDERED_COUNT':
        return InstructionSemantics(name, 'nop')
    # Explicit DS no-op.
    if upper == 'DS_NOP':
        return InstructionSemantics(name, 'nop')
    # Fallback for unrecognized DS atomics.
    if '_ADD' in upper or '_SUB' in upper or '_RSUB' in upper or \
       '_MIN' in upper or '_MAX' in upper or '_AND' in upper or \
       '_OR' in upper or '_XOR' in upper or '_INC' in upper or \
       '_DEC' in upper or '_CMPST' in upper or '_SWAP' in upper or \
       '_CONDXCHG' in upper or '_WRXCHG' in upper or \
       '_STOREXCHG' in upper or '_CMPSTORE' in upper:
        return InstructionSemantics(name, 'ds_atomic')
    # Unrecognized DS instruction — fallthrough to nop.
    return InstructionSemantics(name, 'nop')

def _derive_mimg(name: str) -> InstructionSemantics | None:
    """Derive semantics for an MIMG (Memory Image) instruction."""
    upper = name.upper()
    if upper.startswith('IMAGE_STORE'):
        return InstructionSemantics(name, 'image_store')
    if upper.startswith('IMAGE_LOAD'):
        return InstructionSemantics(name, 'image_load')
    if upper.startswith('IMAGE_ATOMIC'):
        return InstructionSemantics(name, 'image_atomic')
    if upper.startswith('IMAGE_SAMPLE') or upper.startswith('IMAGE_GATHER'):
        return InstructionSemantics(name, 'image_sample')
    if upper.startswith('IMAGE_GET_RESINFO') or upper.startswith('IMAGE_GET_LOD'):
        return InstructionSemantics(name, 'image_query')
    if upper.startswith('IMAGE_MSAA_LOAD'):
        return InstructionSemantics(name, 'image_load')
    if upper.startswith('IMAGE_BVH_INTERSECT'):
        return InstructionSemantics(name, 'image_bvh')
    return InstructionSemantics(name, 'nop')

def _derive_exp(name: str) -> InstructionSemantics | None:
    """Derive semantics for an EXP/VEXPORT (Export) instruction."""
    return InstructionSemantics(name, 'export')

def _derive_vintrp(name: str) -> InstructionSemantics | None:
    """Derive semantics for a VINTRP (Vertex Interpolation) instruction."""
    return InstructionSemantics(name, 'interp')

def _derive_ldsdir(name: str) -> InstructionSemantics | None:
    """Derive semantics for an LDSDIR/VDSDIR (LDS Direct) instruction."""
    return InstructionSemantics(name, 'lds_direct')

# Map encoding format name (with ENC_ prefix stripped) to derivation function.
_ENC_DERIVE = {
    'SOPP': _derive_sopp,
    'SOP1': _derive_sop1,
    'SOP2': _derive_sop2,
    'SOPC': _derive_sopc,
    'SOPK': _derive_sopk,
    'VOP1': _derive_vop1,
    'VOP2': _derive_vop2,
    'VOPC': _derive_vopc,
    'VOP3': _derive_vop3,
    'VOP3_SDST': _derive_vop3,
    'VOP3_SDST_ENC': _derive_vop3,
    'VOP3P': _derive_vop3p,
    'VOP3P_MFM': _derive_vop3p,  # VOP3P_MFMA after trailing A strip
    'SMEM': _derive_smem,
    'FLAT': _derive_flat,
    'MUBUF': _derive_mubuf,
    'MTBUF': _derive_mtbuf,
    'DS': _derive_ds,
    # RDNA1-3 formats.
    'MIMG': _derive_mimg,
    'EXP': _derive_exp,
    'VINTRP': _derive_vintrp,
    'LDSDIR': _derive_ldsdir,
    'VINTERP': _derive_vintrp,
    # RDNA4 renamed formats (same instruction patterns, different encoding
    # names): VFLAT/VGLOBAL/VSCRATCH reuse the FLAT derivation,
    # VBUFFER reuses MUBUF, VDS reuses DS, etc.
    'VFLAT': _derive_flat,
    'VGLOBAL': _derive_flat,
    'VSCRATCH': _derive_flat,
    'VBUFFER': _derive_mubuf,
    'VDS': _derive_ds,
    'VIMAGE': _derive_mimg,
    'VSAMPLE': _derive_mimg,
    'VEXPORT': _derive_exp,
    'VDSDIR': _derive_ldsdir,
}

def derive_semantics(name: str, enc_name: str) -> InstructionSemantics | None:
    """Derive execution semantics for an instruction from its name and encoding.

    The encoding format name selects a format-specific derivation function
    (e.g. ``_derive_sop2`` for SOP2). For VOP3, the VOP1/VOP2/VOPC
    derivations are tried as fallbacks since VOP3 can re-encode those.

    Args:
        name: Instruction mnemonic (e.g. ``S_ADD_I32``).
        enc_name: Encoding format name (e.g. ``ENC_SOP2``, ``VOP3A``).

    Returns:
        InstructionSemantics if the instruction could be classified, or
        ``None`` for unrecognized encodings.
    """
    # Normalize encoding name: "ENC_SOP1"→"SOP1", "VOP3A"→"VOP3", etc.
    base_enc = re.sub(r'^ENC_', '', enc_name)
    # Strip trailing A/B only when the result maps to a known encoding
    # (e.g. VOP3A→VOP3). A generic regex would mangle future encoding
    # names that legitimately end in A or B.
    if base_enc not in _ENC_DERIVE and base_enc[-1:] in ('A', 'B'):
        stripped = base_enc[:-1]
        if stripped in _ENC_DERIVE:
            base_enc = stripped

    derive_fn = _ENC_DERIVE.get(base_enc)
    if derive_fn is None:
        return None
    sem = derive_fn(name)

    # VOP3 (and VOP3_SDST_ENC) can re-encode VOP1/VOP2/VOPC instructions.
    if sem is None and base_enc.startswith('VOP3'):
        for fallback in (_derive_vop1, _derive_vop2, _derive_vopc):
            sem = fallback(name)
            if sem is not None:
                break

    return sem

def derive_all_semantics(isa_spec: IsaSpec) -> SemanticsSpec:
    """Derive semantics for every instruction in an ISA spec.

    Iterates over all encodings and their instructions, calling
    :func:`derive_semantics` for each. If the ISA profile provides
    :attr:`~amdisa.isa_profile.IsaProfile.semantic_overrides`, those
    take priority over the generic mnemonic-based derivation.

    Duplicate instruction names (same mnemonic under multiple encodings)
    are skipped.

    Args:
        isa_spec: Parsed ISA specification with populated encodings.

    Returns:
        SemanticsSpec containing all derivable instructions.
    """
    overrides = isa_spec.profile.semantic_overrides if isa_spec.profile else {}
    instructions: dict[str, InstructionSemantics] = {}
    for enc in isa_spec.inst_encodings:
        for inst in enc.insts:
            if inst.name in instructions:
                continue
            override = overrides.get(inst.name)
            if override is not None:
                sem_class, operation, data_type = override
                instructions[inst.name] = InstructionSemantics(
                    inst.name,
                    sem_class,
                    operation=operation or None,
                    data_type=data_type or None,
                )
                continue
            sem = derive_semantics(inst.name, enc.enc_name)
            if sem is not None:
                instructions[inst.name] = sem
    return SemanticsSpec(instructions)
