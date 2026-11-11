// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// FP32 sine/cosine, modifiers, and exact/loose policy selection.
// Internal helpers shared by instruction implementations; not part of the public C API.

#ifndef RJ_ARITH_TRIG_H_
#define RJ_ARITH_TRIG_H_

#include <stdint.h>

namespace rocjitsu::arith {

// Evaluate FP32 sine/cosine. Preserve destination lanes outside exec_mask.
// Requires the named CPU features and validated instruction controls.
void trig_x86_64_v3(bool cosine, uint32_t exec_mask, uint32_t mode, uint32_t *d, const uint32_t *a);

// Evaluate FP32 sine/cosine. Preserve destination lanes outside exec_mask.
// Requires the named CPU features and validated instruction controls.
void half_trig_x86_64_v3(bool cosine, uint32_t exec_mask, uint32_t mode, uint32_t *d,
                         const uint32_t *a);

} // namespace rocjitsu::arith

#endif
