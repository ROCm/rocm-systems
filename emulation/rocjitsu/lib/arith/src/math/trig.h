// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// FP32 sine/cosine, modifiers, and exact/loose policy selection.
// Internal helpers shared by instruction implementations; not part of the public C API.

#ifndef RJ_ARITH_TRIG_H_
#define RJ_ARITH_TRIG_H_

#include <stdint.h>

namespace rocjitsu::arith {

// Evaluate an unmodified full wave using the empirical FP32 sine model.
void sin_exact_x86_64_v3(uint32_t *d, const uint32_t *a, unsigned denorm, bool quiet_snan);
void sin_exact_x86_64_v4(uint32_t *d, const uint32_t *a, unsigned denorm, bool quiet_snan);

// Evaluate FP32 sine/cosine. Preserve destination lanes outside exec_mask.
// Requires the named CPU features and validated instruction controls.
void trig_x86_64_v3(bool cosine, uint32_t exec_mask, uint32_t mode, uint32_t *d, const uint32_t *a);

// Evaluate FP32 sine/cosine. Preserve destination lanes outside exec_mask.
// Requires the named CPU features and validated instruction controls.
void half_trig_x86_64_v3(bool cosine, uint32_t exec_mask, uint32_t mode, uint32_t *d,
                         const uint32_t *a);

} // namespace rocjitsu::arith

#endif
