//===- hotswap_comgr_client.hpp - dlopen interface to COMGR hotswap -------===//
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ROCR_HOTSWAP_COMGR_CLIENT_HPP
#define ROCR_HOTSWAP_COMGR_CLIENT_HPP

#include <cstddef>

namespace rocr::hotswap {

/// Check if COMGR's hotswap backend is available (found via dlopen).
bool ComgrHotswapAvailable();

/// Call COMGR's amd_comgr_hotswap_rewrite via dlopen/dlsym.
/// @param source_isa  Code object ISA, e.g. "amdgcn-amd-amdhsa--gfx1250"
/// @param target_isa  Agent hardware ISA, e.g. "amdgcn-amd-amdhsa--gfx1250"
/// Returns 0 on success, non-zero on failure. Caller frees *out_elf with
/// free().
int ComgrHotswapRewrite(const void *elf_data, size_t elf_size,
                        const char *source_isa, const char *target_isa,
                        void **out_elf, size_t *out_elf_size);

} // namespace rocr::hotswap

#endif // ROCR_HOTSWAP_COMGR_CLIENT_HPP
