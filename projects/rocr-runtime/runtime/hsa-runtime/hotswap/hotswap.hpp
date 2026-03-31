//===- hotswap.hpp - HotSwap B0-to-A0 API for ROCR loader ----------------===//
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ROCR_HOTSWAP_HPP
#define ROCR_HOTSWAP_HPP

#include <cstddef>
#include <cstdint>

namespace rocr {
namespace hotswap {

/// Apply GFX1250 B0-to-A0 silicon stepping patches to a code object.
///
/// Called by the loader when it detects a gfx1250 code object on A0
/// hardware. Delegates to COMGR's amd_comgr_hotswap_rewrite_b0a0
/// via dlopen. The caller must free *out_data with free() if it
/// differs from elf_data.
///
/// Returns 0 on success, non-zero on failure.
int RetargetCodeObjectB0A0Grow(const void *elf_data, size_t elf_size,
                               void **out_data, size_t *out_size);

} // namespace hotswap
} // namespace rocr

#endif // ROCR_HOTSWAP_HPP
