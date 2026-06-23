//===- hotswap.hpp - HotSwap ISA rewriting API ----------------------------===//
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ROCR_HOTSWAP_HPP
#define ROCR_HOTSWAP_HPP

#include <cstddef>

namespace rocr::hotswap {

/// Retarget a code object from its own ISA to the running GPU's ISA via COMGR.
///
/// The source ISA is read from the code object via COMGR
/// (amd_comgr_get_data_isa_name); the target ISA is the running GPU's ISA,
/// supplied by the caller (e.g. from the HSA agent). COMGR's
/// amd_comgr_hotswap_rewrite (linked directly) applies whatever transformation
/// the source/target pair calls for -- same-ISA stepping patches (e.g. gfx1250
/// B0 to A0) or cross-family transpilation -- and returns the rewritten code
/// object. If no transformation is needed, the output is a copy of the input.
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
                       const char *target_isa, void **out_data,
                       size_t *out_size);

} // namespace rocr::hotswap

#endif // ROCR_HOTSWAP_HPP
