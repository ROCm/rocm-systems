//===- hotswap.hpp - HotSwap ISA rewriting API ----------------------------===//
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ROCR_HOTSWAP_HPP
#define ROCR_HOTSWAP_HPP

#include <cstddef>

namespace rocr::hotswap {

/// Rewrite a code object from source_isa to target_isa via COMGR.
///
/// Called by the hotswap tools lib when the code object's ISA differs from
/// the agent's ISA, or when stepping patches are needed (e.g., B0-to-A0).
/// Delegates to COMGR's amd_comgr_hotswap_rewrite (linked directly).
///
/// On success, *out_data and *out_size describe the rewritten code object.
/// If *out_data differs from elf_data, it was allocated by this function
/// and the caller must free it with free().
///
/// On failure, *out_data is set to elf_data and *out_size is set to
/// elf_size so callers can continue using the original code object.
/// When *out_data == elf_data, the caller must not free it.
///
/// Returns 0 on success, non-zero on failure.
int RetargetCodeObject(const void *elf_data, size_t elf_size,
                       const char *source_isa, const char *target_isa,
                       void **out_data, size_t *out_size);

} // namespace rocr::hotswap

#endif // ROCR_HOTSWAP_HPP
