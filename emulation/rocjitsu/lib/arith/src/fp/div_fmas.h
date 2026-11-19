// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Cross-ISA backend declarations for division fmas.
#pragma once

#include <stdint.h>

namespace rocjitsu::arith {

// Evaluate division FMA with conditional scaling. Preserve destination lanes outside
// exec_mask. quiet_nan controls quieting of propagated signaling NaNs.
// gradual_before_omod selects legacy CDNA core rounding before output scaling.
// Requires the named CPU features and validated instruction controls.
template <int Width>
void div_fmas_x86_64_v3(uint32_t exec_mask, uint32_t mode, uint32_t *const *d,
                        const uint32_t *const *a, const uint32_t *const *b,
                        const uint32_t *const *c, uint32_t condition, uint64_t flags,
                        uint32_t *state, bool quiet_nan, bool gradual_before_omod = false);

// Evaluate division FMA with conditional scaling. Preserve destination lanes outside
// exec_mask. quiet_nan controls quieting of propagated signaling NaNs.
// gradual_before_omod selects legacy CDNA core rounding before output scaling.
// Requires the named CPU features and validated instruction controls.
template <int Width>
void div_fmas_x86_64_v4(uint32_t exec_mask, uint32_t mode, uint32_t *const *d,
                        const uint32_t *const *a, const uint32_t *const *b,
                        const uint32_t *const *c, uint32_t condition, uint64_t flags,
                        uint32_t *state, bool quiet_nan, bool gradual_before_omod = false);

#if defined(RJ_ARITH_HAVE_X86_64_V4)
// Evaluate division FMA with exact guest rounding; decline without writes when unsupported.
bool div_fmas_exact_x86_64_v4(uint64_t flags, uint32_t exec_mask, uint32_t *d, const uint32_t *a,
                              const uint32_t *b, const uint32_t *c, uint32_t *state,
                              uint32_t mode = 0);
#endif

} // namespace rocjitsu::arith
