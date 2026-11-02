# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Architecture-specific FP8/BF8 encoding and implementation selection."""

from __future__ import annotations

from enum import Enum, auto


class Fp8Policy(Enum):
    """A target's FP8/BF8 encoding and the helpers that implement it."""

    # FNUZ encodings through util's *_fnuz helpers.
    FNUZ = auto()
    # OCP encodings through util's helpers.
    OCP_UTIL = auto()
    # OCP encodings through shared/conversion.h, matching gfx1201 captures.
    OCP_SHARED = auto()


# CDNA4 and CDNA5 keep util's helpers: their manuals saturate an infinity under
# FP16_OVFL, which gfx1201 does not, and neither target has captures.
_FP8_POLICIES = {
    'cdna3': Fp8Policy.FNUZ,
    'cdna4': Fp8Policy.OCP_UTIL,
    'cdna5': Fp8Policy.OCP_UTIL,
    'rdna4': Fp8Policy.OCP_SHARED,
}

# Targets not listed above have no FP8/BF8 conversions; they keep util's names.
DEFAULT_FP8_POLICY = Fp8Policy.OCP_UTIL

_FNUZ_HELPER_NAMES = {
    'amdgpu::extract_fp8': 'amdgpu::extract_fp8_fnuz',
    'amdgpu::extract_bf8': 'amdgpu::extract_bf8_fnuz',
    'amdgpu::smfmac_read_fp8': 'amdgpu::smfmac_read_fp8_fnuz',
    'amdgpu::smfmac_read_bf8': 'amdgpu::smfmac_read_bf8_fnuz',
    'util::fp8_e4m3_to_f32': 'util::fp8_e4m3_fnuz_to_f32',
    'util::bf8_e5m2_to_f32': 'util::bf8_e5m2_fnuz_to_f32',
    'util::f32_to_fp8_e4m3_rne': 'util::f32_to_fp8_e4m3_fnuz_rne',
    'util::f32_to_bf8_e5m2_rne': 'util::f32_to_bf8_e5m2_fnuz_rne',
    'util::f32_to_fp8_e4m3_sr': 'util::f32_to_fp8_e4m3_fnuz_sr',
    'util::f32_to_bf8_e5m2_sr': 'util::f32_to_bf8_e5m2_fnuz_sr',
    'util::f32_to_fp8_e4m3_rne_mode': 'util::f32_to_fp8_e4m3_fnuz_rne_mode',
    'util::f32_to_bf8_e5m2_rne_mode': 'util::f32_to_bf8_e5m2_fnuz_rne_mode',
    'util::f32_to_fp8_e4m3_sr_mode': 'util::f32_to_fp8_e4m3_fnuz_sr_mode',
    'util::f32_to_bf8_e5m2_sr_mode': 'util::f32_to_bf8_e5m2_fnuz_sr_mode',
}


def fp8_policy(arch_name: str | None) -> Fp8Policy:
    """Return a target's FP8/BF8 policy."""
    return _FP8_POLICIES.get(arch_name or '', DEFAULT_FP8_POLICY)


def uses_fnuz_fp8(arch_name: str | None) -> bool:
    """Whether a target's FP8/BF8 values use the FNUZ encodings."""
    return fp8_policy(arch_name) is Fp8Policy.FNUZ


def uses_shared_fp8(arch_name: str | None) -> bool:
    """Whether a target's F32<->FP8/BF8 conversions use shared/conversion.h."""
    return fp8_policy(arch_name) is Fp8Policy.OCP_SHARED


def fp8_helper_name(arch_name: str | None, name: str) -> str:
    """Select FNUZ f8 helpers while preserving OCP helper names elsewhere."""
    if uses_fnuz_fp8(arch_name):
        return _FNUZ_HELPER_NAMES.get(name, name)
    return name
