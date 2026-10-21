# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Emit shared/conversion.h calls for VALU conversions.

Each instruction resolves one stage object, such as
``amdgpu::conversion_to_float<From, To>(wf, ...)``, from MODE and its VOP3
fields. The scalar body declares it and the SIMD probe passes the same object
(see share_stages), so both paths apply the same ABS/NEG, MODE input flushing,
rounding, OMOD and CLAMP, all on raw encodings.
"""

import re

from amdisa.codegen.execute.fp8_formats import FNUZ_FP8_ARCHES
from amdisa.semantics import F32_TO_INTEGER_CONVERSIONS, TO_FLOAT_CONVERSIONS

# The per-instruction stage object declared before the scalar lane loop.
STAGES = 'conversion'

_TEMPLATE = re.compile(r'v_cvt_(?P<dtype>[a-z0-9]+_[a-z0-9]+)_(?P<encoding>vop1|vop3)')
_DECLARATION = re.compile(rf'^ *const auto {STAGES} = [^\n]*;\n?', re.MULTILINE)
_NAMES_STAGES = re.compile(rf'(?<![\w:]){STAGES}(?![\w:])')


def _format(name: str) -> str:
    if name.startswith('F'):
        return f'amdgpu::fp_format::{name}'
    return f'amdgpu::conversion::{name}'


def is_integer_source(dtype: str) -> bool:
    return not TO_FLOAT_CONVERSIONS[dtype][0].startswith('F')


def destination(dtype: str) -> str:
    """Return the destination's floating format name, e.g. 'F16'."""
    return TO_FLOAT_CONVERSIONS[dtype][1]


def stages_expr(dtype: str, fields: tuple[str, ...] = ()) -> str:
    """Resolve the stages; ``fields`` are (ABS, NEG, OMOD, CLAMP) expressions.

    Trailing zero fields are omitted, so a VOP1 form passes only ``wf``.
    """
    source, result = TO_FLOAT_CONVERSIONS[dtype]
    fields = list(fields)
    while fields and fields[-1] == '0u':
        fields.pop()
    args = ', '.join(['wf', *fields])
    return f'amdgpu::conversion_to_float<{_format(source)}, {_format(result)}>({args})'


def declaration(dtype: str, fields: tuple[str, ...] | None = None) -> str:
    """Declare the scalar stage object once, before the lane loop.

    ``fields`` are the VOP3 (ABS, NEG, OMOD, CLAMP) expressions; an
    F32-to-integer conversion uses only ABS and NEG. Without them an F16
    result takes the SDWA OMOD, which scales before rounding and is zero for
    encodings without SDWA.
    """
    if dtype in F32_TO_INTEGER_CONVERSIONS:
        operation = f'amdgpu::conversion::{F32_TO_INTEGER_CONVERSIONS[dtype]}'
        modifiers = list((fields or ())[:2])
        while modifiers and modifiers[-1] == '0u':
            modifiers.pop()
        args = ', '.join(['wf', *modifiers])
        stages = f'amdgpu::conversion_f32_to_integer<{operation}>({args})'
    elif fields is None and destination(dtype) == 'F16':
        source = _format(TO_FLOAT_CONVERSIONS[dtype][0])
        stages = f'amdgpu::sdwa_conversion_to_f16<{source}>(*this, wf)'
    else:
        stages = stages_expr(dtype, fields or ())
    return f'  const auto {STAGES} = {stages};'


def share_stages(probe: str | None, body: str) -> tuple[str | None, str]:
    """Move the scalar body's stage declaration above a probe that passes it.

    Returns the probe preceded by the declaration and the body without it.
    A probe that does not name the stage object is returned unchanged.
    """
    if probe is None or not _NAMES_STAGES.search(probe):
        return probe, body
    match = _DECLARATION.search(body)
    if match is None:
        raise ValueError(f'SIMD probe passes {STAGES!r}, but the body declares none')
    hoisted = match.group(0).rstrip('\n')
    return f'{hoisted}\n{probe}', body[: match.start()] + body[match.end() :]


def needs_half_glue(template_name: str) -> bool:
    """Whether a local true16 body needs the half-register word glue."""
    match = _TEMPLATE.fullmatch(template_name)
    return (
        match is not None
        and match['dtype'] in TO_FLOAT_CONVERSIONS
        and destination(match['dtype']) == 'F16'
    )


def simd_probe(
    template_name: str,
    *,
    true16_vop3: bool = False,
    e32: bool = False,
    e32_half_inputs: int = 0,
) -> str | None:
    """Emit the SIMD probe for a conversion to a floating format, or None.

    The probe passes the scalar body's stage object, modifiers included, to
    the raw-word glue. A true16 F16 result goes to the selected destination
    half, keeping the other half.
    """
    match = _TEMPLATE.fullmatch(template_name)
    if match is None or match['dtype'] not in TO_FLOAT_CONVERSIONS:
        return None
    dtype = match['dtype']
    vop3 = match['encoding'] == 'vop3'
    half_dst = destination(dtype) == 'F16'
    half_source = int(dtype.endswith('16'))
    if e32 or (vop3 and true16_vop3 and half_dst):
        half_inputs = e32_half_inputs if e32 else half_source
        args = f'1, {str(e32).lower()}, {str(half_dst).lower()}, {half_inputs}'
        return (
            f'  if (amdgpu::try_execute_words_simd<{args}>(inst, wf, {STAGES})) return;'
        )
    source, result = TO_FLOAT_CONVERSIONS[dtype]
    if source == 'F64':
        return f'  ROCJITSU_TRY_SIMD_CONVERSION_FROM_F64({STAGES});'
    if result == 'F64':
        return f'  ROCJITSU_TRY_SIMD_CONVERSION_TO_F64({STAGES});'
    return f'  ROCJITSU_TRY_SIMD_VOP1_UNARY(uint32_t, uint32_t, {STAGES});'


def _resolve_with_source_modifiers(
    resolver: str, is_vop3: bool, has_abs: bool, inst: str
) -> str:
    """Call a stage resolver; a VOP3 form passes its ABS/NEG fields."""
    if not is_vop3:
        return f'{resolver}(wf)'
    abs_field = f'{inst}.abs' if has_abs else '0u'
    return f'{resolver}(wf, {abs_field}, {inst}.neg)'


def pk_rtz_f16_expr(is_vop3: bool, has_abs: bool, inst: str) -> str:
    """Resolve V_CVT_PK_RTZ_F16_F32's stages."""
    return _resolve_with_source_modifiers(
        'amdgpu::conversion_pk_rtz_f16', is_vop3, has_abs, inst
    )


def pk_u8_expr(is_vop3: bool, has_abs: bool, inst: str) -> str:
    """Resolve V_CVT_PK_U8_F32's stages."""
    return _resolve_with_source_modifiers(
        'amdgpu::conversion_pk_u8', is_vop3, has_abs, inst
    )


def pack_b32_f16_expr(is_vop3: bool, has_abs: bool, inst: str) -> str:
    """Resolve V_PACK_B32_F16's stages."""
    return _resolve_with_source_modifiers(
        'amdgpu::conversion_pack_b32_f16', is_vop3, has_abs, inst
    )


# OCP 8-bit float formats (shared/conversion.h); CDNA3 uses FNUZ helpers instead.
_FP8_FORMATS = {'fp8': 'amdgpu::conversion::Fp8', 'bf8': 'amdgpu::conversion::Bf8'}

# OCP targets that keep util's FP8/BF8 helpers. The CDNA4 manual saturates an
# infinity under FP16_OVFL, which gfx1201 does not, and gfx950 is uncaptured.
_UTIL_OCP_FP8_ARCHES = frozenset({'cdna4'})


def uses_ocp_fp8(arch_name: str) -> bool:
    """Whether a target's FP8/BF8 conversions use the shared OCP encodings."""
    return arch_name not in FNUZ_FP8_ARCHES | _UTIL_OCP_FP8_ARCHES


def fp8_format(name: str) -> str:
    """Return the conversion.h layout for 'fp8' or 'bf8'."""
    return _FP8_FORMATS[name]


def to_fp8_expr(name: str, is_vop3: bool, has_abs: bool, inst: str) -> str:
    """Resolve an F32-to-FP8/BF8 conversion."""
    return _resolve_with_source_modifiers(
        f'amdgpu::conversion_to_fp8<{fp8_format(name)}>', is_vop3, has_abs, inst
    )


def decode_fp8_expr(name: str, byte: str) -> str:
    """Decode the FP8/BF8 byte in the low bits of ``byte`` to F32 bits."""
    return f'amdgpu::conversion::decode_fp8<{fp8_format(name)}>({byte})'
