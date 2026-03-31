//===- hotswap_comgr_client.hpp - dlopen interface to COMGR hotswap -------===//
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ROCR_HOTSWAP_COMGR_CLIENT_HPP
#define ROCR_HOTSWAP_COMGR_CLIENT_HPP

#include <cstddef>
#include <cstdint>

namespace rocr {
namespace hotswap {

/// Check if COMGR's hotswap backend is available (found via dlopen).
bool ComgrHotswapAvailable();

/// Call COMGR's amd_comgr_hotswap_rewrite_b0a0 via dlopen/dlsym.
/// Returns 0 on success, non-zero on failure.
int ComgrHotswapRewriteB0A0(const void *elf_data, size_t elf_size,
                            void **out_elf, size_t *out_elf_size);

} // namespace hotswap
} // namespace rocr

#endif // ROCR_HOTSWAP_COMGR_CLIENT_HPP
