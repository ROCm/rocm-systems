//===- hotswap.hpp - HotSwap ISA rewriting API ----------------------------===//
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

/// Rewrite a code object from source_isa to target_isa via COMGR.
///
/// Called by the hotswap tools lib when the code object's ISA differs from
/// the agent's ISA, or when stepping patches are needed (e.g., B0-to-A0).
/// Delegates to COMGR's amd_comgr_hotswap_rewrite via dlopen.
///
/// The caller must free *out_data with free() if it differs from elf_data.
/// Returns 0 on success, non-zero on failure.
int RetargetCodeObject(const void *elf_data, size_t elf_size,
                       const char *source_isa, const char *target_isa,
                       void **out_data, size_t *out_size);

} // namespace hotswap
} // namespace rocr

#endif // ROCR_HOTSWAP_HPP
