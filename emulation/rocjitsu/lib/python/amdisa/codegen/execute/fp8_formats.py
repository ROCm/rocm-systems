# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""FP8/BF8 format helper selection."""

from __future__ import annotations

from amdisa.isa_profile import Fp8EncodingMode

FNUZ_FP8_ARCHES = frozenset({'cdna3'})

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


def fp8_helper_name_for_mode(mode: Fp8EncodingMode, name: str) -> str:
    """Select a helper implementing the requested FP8/BF8 encoding."""
    if mode is Fp8EncodingMode.FNUZ:
        return _FNUZ_HELPER_NAMES.get(name, name)
    return name


def fp8_fnuz_template_suffix(mode: Fp8EncodingMode) -> str:
    """Append the non-default C++ template argument required for FNUZ FP8."""
    return ', true' if mode is Fp8EncodingMode.FNUZ else ''


def fp8_helper_name(arch_name: str, name: str) -> str:
    """Select FNUZ f8 helpers while preserving OCP helper names elsewhere."""
    mode = Fp8EncodingMode.FNUZ if arch_name in FNUZ_FP8_ARCHES else Fp8EncodingMode.OCP
    return fp8_helper_name_for_mode(mode, name)
