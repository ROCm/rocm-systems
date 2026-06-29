//===- hotswap.hpp - HotSwap ISA rewriting API ----------------------------===//
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ROCR_HOTSWAP_HPP
#define ROCR_HOTSWAP_HPP

#include <cstddef>
#include <string>

namespace rocr::hotswap {

/// Read a code object's own ISA name via COMGR (amd_comgr_get_data_isa_name).
///
/// Uses COMGR's LLVM-canonical parser, so it tracks triple normalization
/// without hand-rolled metadata parsing. Returns an empty string on failure.
std::string GetCodeObjectIsaName(const void *elf_data, size_t elf_size);

/// Retarget a code object from source_isa to target_isa via COMGR.
///
/// Both ISA names are supplied by the caller: source_isa typically comes from
/// the code object (see GetCodeObjectIsaName) and target_isa from the running
/// GPU (e.g. the HSA agent), but either may be overridden. COMGR's
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
                       const char *source_isa, const char *target_isa,
                       void **out_data, size_t *out_size);

} // namespace rocr::hotswap

#endif // ROCR_HOTSWAP_HPP
