# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""VOP3 modifier configuration enumerator for CTS.

Generates representative modifier configurations (neg, abs, omod, clamp)
without exhaustively testing all 512 combinations.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModifierConfig:
    neg: int = 0  # 3-bit bitmask: bit0=src0, bit1=src1, bit2=src2
    abs: int = 0  # 3-bit bitmask: bit0=src0, bit1=src1, bit2=src2
    omod: int = 0  # 0=none, 1=*2, 2=*4, 3=*0.5
    clamp: int = 0  # 0=off, 1=on

    @property
    def label(self) -> str:
        parts = []
        if self.neg:
            bits = ''.join(str((self.neg >> i) & 1) for i in range(3))
            parts.append(f'neg={bits}')
        if self.abs:
            bits = ''.join(str((self.abs >> i) & 1) for i in range(3))
            parts.append(f'abs={bits}')
        if self.omod:
            parts.append(f'omod={self.omod}')
        if self.clamp:
            parts.append('clamp')
        return '+'.join(parts) if parts else 'none'

    @property
    def is_identity(self) -> bool:
        return self.neg == 0 and self.abs == 0 and self.omod == 0 and self.clamp == 0


def unary_modifier_configs() -> list[ModifierConfig]:
    """Representative modifier configs for unary FP operations (1 source)."""
    return [
        ModifierConfig(),  # baseline (no modifiers)
        ModifierConfig(neg=0b001),  # neg src0
        ModifierConfig(abs=0b001),  # abs src0
        ModifierConfig(neg=0b001, abs=0b001),  # abs then neg src0
        ModifierConfig(clamp=1),  # clamp output
        ModifierConfig(omod=1),  # *2
        ModifierConfig(omod=2),  # *4
        ModifierConfig(omod=3),  # *0.5
        ModifierConfig(neg=0b001, clamp=1),  # neg + clamp
        ModifierConfig(abs=0b001, omod=1, clamp=1),  # abs + omod*2 + clamp
    ]


def binary_modifier_configs() -> list[ModifierConfig]:
    """Representative modifier configs for binary FP operations (2 sources)."""
    return [
        ModifierConfig(),  # baseline
        ModifierConfig(neg=0b001),  # neg src0
        ModifierConfig(neg=0b010),  # neg src1
        ModifierConfig(neg=0b011),  # neg both
        ModifierConfig(abs=0b001),  # abs src0
        ModifierConfig(abs=0b010),  # abs src1
        ModifierConfig(abs=0b011),  # abs both
        ModifierConfig(neg=0b001, abs=0b001),  # abs+neg src0
        ModifierConfig(clamp=1),  # clamp
        ModifierConfig(omod=1),  # *2
        ModifierConfig(omod=2),  # *4
        ModifierConfig(omod=3),  # *0.5
        ModifierConfig(neg=0b010, clamp=1),  # neg src1 + clamp
        ModifierConfig(abs=0b011, neg=0b001, omod=1),  # abs both + neg src0 + omod*2
    ]
