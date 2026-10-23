// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Reciprocal-with-flags entry points and optional guest FP exception reporting.
// Internal helpers shared by instruction implementations; not part of the public C API.

#ifndef RJ_ARITH_RCP_IFLAG_H_
#define RJ_ARITH_RCP_IFLAG_H_

#include <stdint.h>

namespace rocjitsu::arith {

// Compute reciprocal and optional guest causes. Preserve destination lanes outside exec_mask.
// Requires the named CPU features and validated instruction controls.
uint32_t rcp_iflag_x86_64_v3(uint64_t flags, uint32_t exec_mask, uint32_t mode, uint32_t *d,
                             const uint32_t *a);

// Compute reciprocal and optional guest causes. Preserve destination lanes outside exec_mask.
// Requires the named CPU features and validated instruction controls.
uint32_t rcp_iflag_x86_64_v4(uint64_t flags, uint32_t exec_mask, uint32_t mode, uint32_t *d,
                             const uint32_t *a);

} // namespace rocjitsu::arith

#endif
