// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Loose reciprocal backends; the dispatcher owns exact results and FP status.
// Internal helpers shared by instruction implementations; not part of the public C API.

#ifndef RJ_ARITH_RCP_IFLAG_H_
#define RJ_ARITH_RCP_IFLAG_H_

#include <stdint.h>

namespace rocjitsu::arith {

// Loose reciprocal with fixed input/output denormal flushing and no FP status.
// Preserve inactive lanes and support D/A aliasing.
// Requires the named CPU features and validated instruction controls.
void rcp_iflag_x86_64_v3(uint64_t flags, uint32_t exec_mask, uint32_t mode, uint32_t *d,
                         const uint32_t *a);

// Loose reciprocal with fixed input/output denormal flushing and no FP status.
// Preserve inactive lanes and support D/A aliasing.
// Requires the named CPU features and validated instruction controls.
void rcp_iflag_x86_64_v4(uint64_t flags, uint32_t exec_mask, uint32_t mode, uint32_t *d,
                         const uint32_t *a);

} // namespace rocjitsu::arith

#endif
