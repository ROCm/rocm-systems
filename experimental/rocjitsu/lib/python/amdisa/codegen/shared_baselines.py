# Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Shared struct baselines for --use-shared mode.

``SCALAR_BASELINE`` and ``CDNA_BASELINE`` map struct name to
``[(field_name, bit_cnt), ...]``. The corresponding shared include paths
are provided separately by ``SCALAR_SHARED_INCLUDE`` and
``CDNA_SHARED_INCLUDE``.

When --use-shared is active, ``gen_machine_inst_encodings()`` checks each
generated struct against these baselines.  If the name matches AND every
field (name + width) matches, it emits a ``using`` alias referencing the
``amdgpu::`` shared struct instead of re-emitting the struct definition.
"""

from __future__ import annotations

SCALAR_SHARED_INCLUDE = 'rocjitsu/isa/arch/amdgpu/shared/machine_insts_scalar.h'
CDNA_SHARED_INCLUDE = 'rocjitsu/isa/arch/amdgpu/shared/machine_insts_cdna.h'

SCALAR_BASELINE: dict[str, list[tuple[str, int]]] = {
    'Sop1MachineInst': [('ssrc0', 8), ('op', 8), ('sdst', 7), ('encoding', 9)],
    'SopcMachineInst': [('ssrc0', 8), ('ssrc1', 8), ('op', 7), ('encoding', 9)],
    'SoppMachineInst': [('simm16', 16), ('op', 7), ('encoding', 9)],
    'SopkMachineInst': [('simm16', 16), ('sdst', 7), ('op', 5), ('encoding', 4)],
    'Sop2MachineInst': [('ssrc0', 8), ('ssrc1', 8), ('sdst', 7), ('op', 7), ('encoding', 2)],
    'Sop1InstLiteralMachineInst': [
        ('ssrc0', 8), ('op', 8), ('sdst', 7), ('encoding', 9), ('simm32', 32)],
    'Sop2InstLiteralMachineInst': [
        ('ssrc0', 8), ('ssrc1', 8), ('sdst', 7), ('op', 7), ('encoding', 2), ('simm32', 32)],
    'SopcInstLiteralMachineInst': [
        ('ssrc0', 8), ('ssrc1', 8), ('op', 7), ('encoding', 9), ('simm32', 32)],
    'SopkInstLiteralMachineInst': [
        ('simm16', 16), ('sdst', 7), ('op', 5), ('encoding', 4), ('simm32', 32)],
}

CDNA_BASELINE: dict[str, list[tuple[str, int]]] = {
    'SmemMachineInst': [
        ('sbase', 6), ('sdata', 7), ('pad_13', 1), ('soffset_en', 1),
        ('nv', 1), ('glc', 1), ('imm', 1), ('op', 8), ('encoding', 6),
        ('offset', 21), ('pad_53_56', 4), ('soffset', 7)],
    'Vop1MachineInst': [('src0', 9), ('op', 8), ('vdst', 8), ('encoding', 7)],
    'VopcMachineInst': [('src0', 9), ('vsrc1', 8), ('op', 8), ('encoding', 7)],
    'Vop2MachineInst': [('src0', 9), ('vsrc1', 8), ('vdst', 8), ('op', 6), ('encoding', 1)],
    'Vop3MachineInst': [
        ('vdst', 8), ('abs', 3), ('op_sel', 4), ('clamp', 1), ('op', 10), ('encoding', 6),
        ('src0', 9), ('src1', 9), ('src2', 9), ('omod', 2), ('neg', 3)],
    'Vop3SdstEncMachineInst': [
        ('vdst', 8), ('sdst', 7), ('clamp', 1), ('op', 10), ('encoding', 6),
        ('src0', 9), ('src1', 9), ('src2', 9), ('omod', 2), ('neg', 3)],
    'Vop1InstLiteralMachineInst': [
        ('src0', 9), ('op', 8), ('vdst', 8), ('encoding', 7), ('simm32', 32)],
    'Vop2InstLiteralMachineInst': [
        ('src0', 9), ('vsrc1', 8), ('vdst', 8), ('op', 6), ('encoding', 1), ('simm32', 32)],
    'VopcInstLiteralMachineInst': [
        ('src0', 9), ('vsrc1', 8), ('op', 8), ('encoding', 7), ('simm32', 32)],
}

CDNA_ARCHES = frozenset({'cdna1', 'cdna2', 'cdna3', 'cdna4'})
