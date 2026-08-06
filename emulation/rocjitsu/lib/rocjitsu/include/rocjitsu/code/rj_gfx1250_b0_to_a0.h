// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file rj_gfx1250_b0_to_a0.h
/// @brief Fixed-profile gfx1250 B0-to-A0 code-object translation API.

#ifndef ROCJITSU_CODE_RJ_GFX1250_B0_TO_A0_H_
#define ROCJITSU_CODE_RJ_GFX1250_B0_TO_A0_H_

#include "rocjitsu/base/rj_compiler.h"
#include "rocjitsu/base/rj_status.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/// Translate one gfx1250 B0 AMDGPU code-object ELF for execution on gfx1250 A0.
///
/// The input must be a standalone gfx1250 AMDGPU code object. On success, the
/// library allocates @p translated_elf with malloc-compatible storage and writes
/// its size to @p translated_size. Release the allocation with
/// rj_gfx1250_b0_to_a0_free(). Output arguments are cleared on failure.
///
/// @param[in] source_elf Source code-object bytes.
/// @param[in] source_size Number of bytes in @p source_elf.
/// @param[out] translated_elf Newly allocated translated code-object bytes.
/// @param[out] translated_size Number of bytes in @p translated_elf.
/// @retval ROCJITSU_STATUS_SUCCESS Translation succeeded.
/// @retval ROCJITSU_STATUS_INVALID_ARGUMENT An input or output argument is invalid.
/// @retval ROCJITSU_STATUS_INVALID_CODE_OBJECT The input is not a gfx1250 code object.
/// @retval ROCJITSU_STATUS_OUT_OF_RESOURCES Output allocation failed.
/// @retval ROCJITSU_STATUS_ERROR Translation failed or produced a non-dispatchable object.
#ifdef __cplusplus
[[nodiscard]]
#endif
RJ_API_EXPORT rj_status_t rj_gfx1250_b0_to_a0_translate(const void *source_elf, size_t source_size,
                                                        uint8_t **translated_elf,
                                                        size_t *translated_size);

/// Release storage returned by rj_gfx1250_b0_to_a0_translate().
/// @param[in] translated_elf Allocation to release; NULL is accepted.
RJ_API_EXPORT void rj_gfx1250_b0_to_a0_free(void *translated_elf);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // ROCJITSU_CODE_RJ_GFX1250_B0_TO_A0_H_
