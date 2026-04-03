//===- hotswap_comgr_client.cpp - dlopen binding to COMGR (stub) ----------===//
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hotswap_comgr_client.hpp"

namespace rocr {
namespace hotswap {

bool ComgrHotswapAvailable() {
  // Stub: full dlopen("libamd_comgr.so") + dlsym("amd_comgr_hotswap_rewrite")
  // implementation follows in subsequent commits.
  return false;
}

int ComgrHotswapRewrite(const void *elf_data, size_t elf_size,
                        const char *source_isa, const char *target_isa,
                        void **out_elf, size_t *out_elf_size) {
  (void)elf_data;
  (void)elf_size;
  (void)source_isa;
  (void)target_isa;
  if (out_elf)
    *out_elf = nullptr;
  if (out_elf_size)
    *out_elf_size = 0;
  return -1;
}

} // namespace hotswap
} // namespace rocr
