# Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Registry mapping ``.call`` target names to C++ functions and treatment.

Each ``.call`` node in the SemaAST invokes a named helper function. This
module classifies every helper into one of three treatments:

- **INLINE_CPP**: Emit a C++ function call to an existing simulator helper.
- **OPAQUE_NOP**: System/microarchitectural function — emit a no-op comment.
- **RECURSIVE**: The callee is another instruction — inline its SemaAST or
  emit a call to its execute function.
"""

from __future__ import annotations

from enum import Enum, auto
from typing import Protocol

from amdisa.sema_ast import SemaBlock, SemaNodeKind


class HelperTreatment(Enum):
    INLINE_CPP = auto()
    OPAQUE_NOP = auto()
    RECURSIVE = auto()


class HelperResolver(Protocol):
    """Protocol for resolving ``.call`` helper functions."""

    def resolve(
        self,
        name: str,
        all_blocks: dict[str, SemaBlock] | None = None,
        visited: set[str] | None = None,
        depth: int = 0,
    ) -> tuple[HelperTreatment, str | None]:
        """Resolve a helper function name to its treatment and C++ target.

        Args:
            name: The ``.call`` target name (e.g., ``CalcBufferAddr``).
            all_blocks: Full instruction SemaBlock dict for recursive lookup.
            visited: Set of already-visited names for cycle detection.
            depth: Current recursion depth (max 3).

        Returns:
            Tuple of (treatment, cpp_function_name_or_none).

        Raises:
            ValueError: If a cycle is detected or depth exceeds 3.
        """
        ...


# --------------------------------------------------------------------------
# Static registry: maps .call target -> (treatment, cpp_function_name)
# --------------------------------------------------------------------------

HELPER_REGISTRY: dict[str, tuple[HelperTreatment, str | None]] = {
    # --- Address calculation (7) ---
    'CalcBufferAddr': (HelperTreatment.INLINE_CPP, 'calc_buffer_addr'),
    'CalcFlatAddr': (HelperTreatment.INLINE_CPP, 'calc_flat_addr'),
    'CalcGlobalAddr': (HelperTreatment.INLINE_CPP, 'calc_global_addr'),
    'CalcDsAddr': (HelperTreatment.INLINE_CPP, 'calc_ds_addr'),
    'CalcScalarGlobalAddr': (HelperTreatment.INLINE_CPP, 'calc_scalar_global_addr'),
    'CalcScalarBufferAddr': (HelperTreatment.INLINE_CPP, 'calc_scalar_buffer_addr'),
    'CalcScratchAddr': (HelperTreatment.INLINE_CPP, 'calc_scratch_addr'),
    # --- Type conversion: standard float/int (14) ---
    'f16_to_f32': (HelperTreatment.INLINE_CPP, 'util::f16_to_f32'),
    'f32_to_f16': (HelperTreatment.INLINE_CPP, 'util::f32_to_f16'),
    'f32_to_bf16': (HelperTreatment.INLINE_CPP, 'util::f32_to_bf16'),
    'bf16_to_f32': (HelperTreatment.INLINE_CPP, 'util::bf16_to_f32'),
    'f32_to_f64': (HelperTreatment.INLINE_CPP, 'static_cast<double>'),
    'f64_to_f32': (HelperTreatment.INLINE_CPP, 'static_cast<float>'),
    'f32_to_i32': (HelperTreatment.INLINE_CPP, 'util::f32_to_i32'),
    'i32_to_f32': (HelperTreatment.INLINE_CPP, 'util::i32_to_f32'),
    'u32_to_f32': (HelperTreatment.INLINE_CPP, 'util::u32_to_f32'),
    'f32_to_u32': (HelperTreatment.INLINE_CPP, 'util::f32_to_u32'),
    'f64_to_i32': (HelperTreatment.INLINE_CPP, 'util::f64_to_i32'),
    'f64_to_u32': (HelperTreatment.INLINE_CPP, 'util::f64_to_u32'),
    'i32_to_f64': (HelperTreatment.INLINE_CPP, 'static_cast<double>'),
    'u32_to_f64': (HelperTreatment.INLINE_CPP, 'static_cast<double>'),
    # --- Type conversion: 16-bit int/float (6) ---
    'i16_to_f16': (HelperTreatment.INLINE_CPP, 'util::i16_to_f16'),
    'u16_to_f16': (HelperTreatment.INLINE_CPP, 'util::u16_to_f16'),
    'f16_to_i16': (HelperTreatment.INLINE_CPP, 'util::f16_to_i16'),
    'f16_to_u16': (HelperTreatment.INLINE_CPP, 'util::f16_to_u16'),
    'i32_to_i16': (HelperTreatment.INLINE_CPP, 'static_cast<int16_t>'),
    'u32_to_u16': (HelperTreatment.INLINE_CPP, 'static_cast<uint16_t>'),
    # --- Type conversion: narrow widening (2) ---
    'u4_to_u32': (HelperTreatment.INLINE_CPP, 'util::u4_to_u32'),
    'u8_to_u32': (HelperTreatment.INLINE_CPP, 'static_cast<uint32_t>'),
    # --- Type conversion: FP8/BF8 (4) ---
    'fp8_to_f32': (HelperTreatment.INLINE_CPP, 'util::fp8_to_f32'),
    'fp8_to_f16': (HelperTreatment.INLINE_CPP, 'util::fp8_to_f16'),
    'bf8_to_f32': (HelperTreatment.INLINE_CPP, 'util::bf8_to_f32'),
    'bf8_to_f16': (HelperTreatment.INLINE_CPP, 'util::bf8_to_f16'),
    # --- Type conversion: F32/F16/BF16 to FP8/BF8 (6) ---
    'f32_to_fp8': (HelperTreatment.INLINE_CPP, 'util::f32_to_fp8'),
    'f32_to_bf8': (HelperTreatment.INLINE_CPP, 'util::f32_to_bf8'),
    'f16_to_fp8': (HelperTreatment.INLINE_CPP, 'util::f16_to_fp8'),
    'f16_to_bf8': (HelperTreatment.INLINE_CPP, 'util::f16_to_bf8'),
    'f32_to_u8': (HelperTreatment.INLINE_CPP, 'util::f32_to_u8'),
    # --- Type conversion: scaled FP8/BF8/FP6/FP4 (24) ---
    'f32_to_fp8_scale': (HelperTreatment.INLINE_CPP, 'util::f32_to_fp8_scale'),
    'f32_to_fp8_sr_scale': (HelperTreatment.INLINE_CPP, 'util::f32_to_fp8_sr_scale'),
    'f32_to_bf8_scale': (HelperTreatment.INLINE_CPP, 'util::f32_to_bf8_scale'),
    'f32_to_bf8_sr_scale': (HelperTreatment.INLINE_CPP, 'util::f32_to_bf8_sr_scale'),
    'f32_to_fp6_scale': (HelperTreatment.INLINE_CPP, 'util::f32_to_fp6_scale'),
    'f32_to_fp6_sr_scale': (HelperTreatment.INLINE_CPP, 'util::f32_to_fp6_sr_scale'),
    'f32_to_bf6_scale': (HelperTreatment.INLINE_CPP, 'util::f32_to_bf6_scale'),
    'f32_to_bf6_sr_scale': (HelperTreatment.INLINE_CPP, 'util::f32_to_bf6_sr_scale'),
    'f32_to_fp4_scale': (HelperTreatment.INLINE_CPP, 'util::f32_to_fp4_scale'),
    'f32_to_fp4_sr_scale': (HelperTreatment.INLINE_CPP, 'util::f32_to_fp4_sr_scale'),
    'f16_to_fp8_scale': (HelperTreatment.INLINE_CPP, 'util::f16_to_fp8_scale'),
    'f16_to_fp8_sr_scale': (HelperTreatment.INLINE_CPP, 'util::f16_to_fp8_sr_scale'),
    'f16_to_bf8_scale': (HelperTreatment.INLINE_CPP, 'util::f16_to_bf8_scale'),
    'f16_to_bf8_sr_scale': (HelperTreatment.INLINE_CPP, 'util::f16_to_bf8_sr_scale'),
    'f16_to_fp6_scale': (HelperTreatment.INLINE_CPP, 'util::f16_to_fp6_scale'),
    'f16_to_fp6_sr_scale': (HelperTreatment.INLINE_CPP, 'util::f16_to_fp6_sr_scale'),
    'f16_to_bf6_scale': (HelperTreatment.INLINE_CPP, 'util::f16_to_bf6_scale'),
    'f16_to_bf6_sr_scale': (HelperTreatment.INLINE_CPP, 'util::f16_to_bf6_sr_scale'),
    'f16_to_fp4_scale': (HelperTreatment.INLINE_CPP, 'util::f16_to_fp4_scale'),
    'f16_to_fp4_sr_scale': (HelperTreatment.INLINE_CPP, 'util::f16_to_fp4_sr_scale'),
    'bf16_to_fp8_scale': (HelperTreatment.INLINE_CPP, 'util::bf16_to_fp8_scale'),
    'bf16_to_fp8_sr_scale': (HelperTreatment.INLINE_CPP, 'util::bf16_to_fp8_sr_scale'),
    'bf16_to_bf8_scale': (HelperTreatment.INLINE_CPP, 'util::bf16_to_bf8_scale'),
    'bf16_to_bf8_sr_scale': (HelperTreatment.INLINE_CPP, 'util::bf16_to_bf8_sr_scale'),
    'bf16_to_bf6_scale': (HelperTreatment.INLINE_CPP, 'util::bf16_to_bf6_scale'),
    'bf16_to_bf6_sr_scale': (HelperTreatment.INLINE_CPP, 'util::bf16_to_bf6_sr_scale'),
    'bf16_to_fp6_scale': (HelperTreatment.INLINE_CPP, 'util::bf16_to_fp6_scale'),
    'bf16_to_fp6_sr_scale': (HelperTreatment.INLINE_CPP, 'util::bf16_to_fp6_sr_scale'),
    'bf16_to_fp4_scale': (HelperTreatment.INLINE_CPP, 'util::bf16_to_fp4_scale'),
    'bf16_to_fp4_sr_scale': (HelperTreatment.INLINE_CPP, 'util::bf16_to_fp4_sr_scale'),
    # --- Type conversion: SNORM/UNORM (4) ---
    'f32_to_snorm': (HelperTreatment.INLINE_CPP, 'util::f32_to_snorm'),
    'f32_to_unorm': (HelperTreatment.INLINE_CPP, 'util::f32_to_unorm'),
    'f16_to_snorm': (HelperTreatment.INLINE_CPP, 'util::f16_to_snorm'),
    'f16_to_unorm': (HelperTreatment.INLINE_CPP, 'util::f16_to_unorm'),
    # --- Numeric constructors (2) ---
    'float16': (HelperTreatment.INLINE_CPP, 'util::make_f16'),
    'float32': (HelperTreatment.INLINE_CPP, 'util::make_f32'),
    # --- FP classification (4) ---
    'isNAN': (HelperTreatment.INLINE_CPP, 'std::isnan'),
    'isQuietNAN': (HelperTreatment.INLINE_CPP, 'util::is_quiet_nan'),
    'isSignalNAN': (HelperTreatment.INLINE_CPP, 'util::is_signal_nan'),
    'cvtToQuietNAN': (HelperTreatment.INLINE_CPP, 'util::cvt_to_quiet_nan'),
    # --- Special ALU helpers (7) ---
    'ABSDIFF': (HelperTreatment.INLINE_CPP, 'util::absdiff'),
    'BYTE_PERMUTE': (HelperTreatment.INLINE_CPP, 'util::byte_permute'),
    'SAT4': (HelperTreatment.INLINE_CPP, 'util::sat4'),
    'SAT8': (HelperTreatment.INLINE_CPP, 'util::sat8'),
    'isEven': (HelperTreatment.INLINE_CPP, 'util::is_even'),
    'IsM0': (HelperTreatment.INLINE_CPP, 'util::is_m0'),
    'HwRegWriteMask': (HelperTreatment.INLINE_CPP, 'util::hwreg_write_mask'),
    'TanhCubicApproximation': (HelperTreatment.INLINE_CPP, 'util::tanh_cubic_approx'),
    # --- Type conversion: V_CVT helpers (2) ---
    'v_cvt_i16_f32': (HelperTreatment.INLINE_CPP, 'util::v_cvt_i16_f32'),
    'v_cvt_u16_f32': (HelperTreatment.INLINE_CPP, 'util::v_cvt_u16_f32'),
    # --- System / microarchitectural (opaque) (7) ---
    'CheckBarrierComplete': (HelperTreatment.OPAQUE_NOP, None),
    'PrefetchScalarData': (HelperTreatment.OPAQUE_NOP, None),
    'PrefetchScalarInst': (HelperTreatment.OPAQUE_NOP, None),
    'ReallocVgprs': (HelperTreatment.OPAQUE_NOP, None),
    'WaitIdleExceptStoreCnt': (HelperTreatment.OPAQUE_NOP, None),
    'InCluster': (HelperTreatment.OPAQUE_NOP, None),
    'InWorkgroup': (HelperTreatment.OPAQUE_NOP, None),
    # --- NOPs (2) ---
    'nop': (HelperTreatment.OPAQUE_NOP, None),
    's_nop': (HelperTreatment.OPAQUE_NOP, None),
    # --- Recursive: instruction cross-references (39) ---
    's_ff1_i32_b32': (HelperTreatment.RECURSIVE, None),
    's_ff1_i32_b64': (HelperTreatment.RECURSIVE, None),
    'v_add_nc_i16': (HelperTreatment.RECURSIVE, None),
    'v_add_nc_i32': (HelperTreatment.RECURSIVE, None),
    'v_add_nc_u16': (HelperTreatment.RECURSIVE, None),
    'v_add_nc_u32': (HelperTreatment.RECURSIVE, None),
    'v_max3_i16': (HelperTreatment.RECURSIVE, None),
    'v_max3_i32': (HelperTreatment.RECURSIVE, None),
    'v_max3_num_f16': (HelperTreatment.RECURSIVE, None),
    'v_max3_num_f32': (HelperTreatment.RECURSIVE, None),
    'v_max3_u16': (HelperTreatment.RECURSIVE, None),
    'v_max3_u32': (HelperTreatment.RECURSIVE, None),
    'v_max_i16': (HelperTreatment.RECURSIVE, None),
    'v_max_i32': (HelperTreatment.RECURSIVE, None),
    'v_max_num_bf16': (HelperTreatment.RECURSIVE, None),
    'v_max_num_f16': (HelperTreatment.RECURSIVE, None),
    'v_max_num_f32': (HelperTreatment.RECURSIVE, None),
    'v_max_u16': (HelperTreatment.RECURSIVE, None),
    'v_max_u32': (HelperTreatment.RECURSIVE, None),
    'v_maximum3_f16': (HelperTreatment.RECURSIVE, None),
    'v_maximum_f16': (HelperTreatment.RECURSIVE, None),
    'v_maximum_f32': (HelperTreatment.RECURSIVE, None),
    'v_min3_num_f16': (HelperTreatment.RECURSIVE, None),
    'v_min3_num_f32': (HelperTreatment.RECURSIVE, None),
    'v_min_i16': (HelperTreatment.RECURSIVE, None),
    'v_min_i32': (HelperTreatment.RECURSIVE, None),
    'v_min_num_bf16': (HelperTreatment.RECURSIVE, None),
    'v_min_num_f16': (HelperTreatment.RECURSIVE, None),
    'v_min_num_f32': (HelperTreatment.RECURSIVE, None),
    'v_min_u16': (HelperTreatment.RECURSIVE, None),
    'v_min_u32': (HelperTreatment.RECURSIVE, None),
    'v_minimum3_f16': (HelperTreatment.RECURSIVE, None),
    'v_minimum_f16': (HelperTreatment.RECURSIVE, None),
    'v_minimum_f32': (HelperTreatment.RECURSIVE, None),
    'v_msad_u8': (HelperTreatment.RECURSIVE, None),
    'v_prng_b32': (HelperTreatment.RECURSIVE, None),
    'v_sad_u8': (HelperTreatment.RECURSIVE, None),
    'v_tanh_f32': (HelperTreatment.RECURSIVE, None),
}

_MAX_RECURSIVE_DEPTH = 3


def resolve_helper(
    name: str,
    all_blocks: dict[str, SemaBlock] | None = None,
    visited: set[str] | None = None,
    depth: int = 0,
) -> tuple[HelperTreatment, str | None]:
    """Resolve a ``.call`` target name to its treatment and C++ target.

    For RECURSIVE helpers, looks up the callee's SemaBlock in ``all_blocks``
    and checks for cycles and depth limits.

    Args:
        name: The ``.call`` target name.
        all_blocks: Full instruction dict for recursive lookup.
        visited: Cycle detection set.
        depth: Current recursion depth.

    Returns:
        ``(treatment, cpp_name_or_none)``.

    Raises:
        ValueError: If a cycle is detected or depth exceeds the limit.
    """
    if depth > _MAX_RECURSIVE_DEPTH:
        raise ValueError(
            f"Recursive helper depth exceeded ({depth} > {_MAX_RECURSIVE_DEPTH}) "
            f"for '{name}'"
        )

    entry = HELPER_REGISTRY.get(name)
    if entry is None:
        return (HelperTreatment.INLINE_CPP, name)

    treatment, cpp_name = entry

    if treatment == HelperTreatment.RECURSIVE and all_blocks is not None:
        if visited is None:
            visited = set()
        canonical = name.upper()
        if canonical in visited:
            raise ValueError(f"Cycle detected: {' -> '.join(visited)} -> {canonical}")
        visited.add(canonical)
        block = all_blocks.get(canonical)
        if block is not None:
            for node in block.body.walk():
                if node.kind == SemaNodeKind.CALL and node.call_name:
                    sub_entry = HELPER_REGISTRY.get(node.call_name)
                    if sub_entry and sub_entry[0] == HelperTreatment.RECURSIVE:
                        resolve_helper(
                            node.call_name,
                            all_blocks,
                            visited.copy(),
                            depth + 1,
                        )

    return entry


def list_unresolved(blocks: dict[str, SemaBlock]) -> list[str]:
    """Return ``.call`` target names not found in the registry."""
    unresolved: set[str] = set()
    for block in blocks.values():
        for node in block.body.walk():
            if node.kind == SemaNodeKind.CALL and node.call_name:
                if node.call_name not in HELPER_REGISTRY:
                    unresolved.add(node.call_name)
    return sorted(unresolved)
