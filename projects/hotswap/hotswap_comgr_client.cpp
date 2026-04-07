//===- hotswap_comgr_client.cpp - Direct link to COMGR --------------------===//
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hotswap_comgr_client.hpp"

// TODO(COMGR-upstream): Replace this forward declaration with
// #include <amd_comgr.h> once the hotswap API lands in a released COMGR.
// The return type is amd_comgr_status_t (enum, ABI-compatible with int).
extern "C" int amd_comgr_hotswap_rewrite(const void *elf_data, size_t elf_size,
                                         const char *source_isa_name,
                                         const char *target_isa_name,
                                         void **out_elf,
                                         size_t *out_elf_size);

namespace rocr::hotswap {

bool ComgrHotswapAvailable() { return true; }

int ComgrHotswapRewrite(const void *elf_data, size_t elf_size,
                        const char *source_isa, const char *target_isa,
                        void **out_elf, size_t *out_elf_size) {
  return amd_comgr_hotswap_rewrite(elf_data, elf_size, source_isa, target_isa,
                                   out_elf, out_elf_size);
}

} // namespace rocr::hotswap
