# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Emit shared/conversion.h calls for VALU conversions to floating formats.

Scalar bodies and SIMD functors share one stage object, resolved before the
lane loop by ``amdgpu::conversion_to_float<From, To>(wf, ...)``. It applies
ABS/NEG and MODE input flushing to a floating source, rounds once in the
destination format's MODE, then applies OMOD and CLAMP, all on raw encodings.
"""

import re

# Conversion data type -> (source format, destination format). Integer
# formats are shared/conversion.h tags; floating formats are fp_format layouts.
TO_FLOAT: dict[str, tuple[str, str]] = {
    'f32_i32': ('I32', 'F32'),
    'f32_u32': ('U32', 'F32'),
    'f16_i16': ('I16', 'F16'),
    'f16_u16': ('U16', 'F16'),
}

# The per-instruction stage object declared before the scalar lane loop.
STAGES = 'conversion'

_TEMPLATE = re.compile(r'v_cvt_(?P<dtype>[a-z0-9]+_[a-z0-9]+)_(?P<encoding>vop1|vop3)')


def _format(name: str) -> str:
    if name.startswith('F'):
        return f'amdgpu::fp_format::{name}'
    return f'amdgpu::conversion::{name}'


def is_integer_source(dtype: str) -> bool:
    return not TO_FLOAT[dtype][0].startswith('F')


def destination(dtype: str) -> str:
    """Return the destination's floating format name, e.g. 'F16'."""
    return TO_FLOAT[dtype][1]


def stages_expr(dtype: str, fields: tuple[str, ...] = ()) -> str:
    """Resolve the stages; ``fields`` are (ABS, NEG, OMOD, CLAMP) expressions.

    Trailing zero fields are omitted, so a VOP1 form passes only ``wf``.
    """
    source, result = TO_FLOAT[dtype]
    fields = list(fields)
    while fields and fields[-1] == '0u':
        fields.pop()
    args = ', '.join(['wf', *fields])
    return f'amdgpu::conversion_to_float<{_format(source)}, {_format(result)}>({args})'


def declaration(dtype: str, fields: tuple[str, ...] | None = None) -> str:
    """Declare the scalar stage object once, before the lane loop.

    ``fields`` are the VOP3 (ABS, NEG, OMOD, CLAMP) expressions. Without them
    an F16 result takes the SDWA OMOD, which scales before rounding and is zero
    for encodings without SDWA.
    """
    if fields is None and destination(dtype) == 'F16':
        source = _format(TO_FLOAT[dtype][0])
        stages = f'amdgpu::sdwa_conversion_to_f16<{source}>(*this, wf)'
    else:
        stages = stages_expr(dtype, fields or ())
    return f'  const auto {STAGES} = {stages};'


def _vop3_fields(dtype: str) -> tuple[str, ...]:
    modifiers = (
        ('0u', '0u')
        if is_integer_source(dtype)
        else ('inst.inst_.abs', 'inst.inst_.neg')
    )
    return (*modifiers, 'inst.inst_.omod', 'inst.inst_.clamp')


def needs_half_glue(template_name: str) -> bool:
    """Whether a local true16 body needs the half-register word glue."""
    match = _TEMPLATE.fullmatch(template_name)
    return (
        match is not None
        and match['dtype'] in TO_FLOAT
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

    Every form passes the stage object to the raw-word glue. A true16 F16
    result goes to the selected destination half, keeping the other half.
    """
    match = _TEMPLATE.fullmatch(template_name)
    if match is None or match['dtype'] not in TO_FLOAT:
        return None
    dtype = match['dtype']
    vop3 = match['encoding'] == 'vop3'
    stages = stages_expr(dtype, _vop3_fields(dtype) if vop3 else ())
    half_dst = destination(dtype) == 'F16'
    half_source = int(dtype.endswith('16'))
    if e32 or (vop3 and true16_vop3 and half_dst):
        half_inputs = e32_half_inputs if e32 else half_source
        args = f'1, {str(e32).lower()}, {str(half_dst).lower()}, {half_inputs}'
        return (
            f'  if (amdgpu::try_execute_words_simd<{args}>(inst, wf, {stages})) return;'
        )
    return f'  ROCJITSU_TRY_SIMD_VOP1_UNARY(uint32_t, uint32_t, {stages});'
