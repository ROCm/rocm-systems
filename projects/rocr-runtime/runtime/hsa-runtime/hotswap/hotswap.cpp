//===- hotswap.cpp - HotSwap B0-to-A0 ROCR integration (stub) ------------===//
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hotswap.hpp"
#include "hotswap_comgr_client.hpp"
#include <cstdlib>
#include <iostream>

namespace rocr {
namespace hotswap {

int RetargetCodeObjectB0A0Grow(const void *elf_data, size_t elf_size,
                               void **out_data, size_t *out_size) {
  *out_data = const_cast<void *>(elf_data);
  *out_size = elf_size;

  if (!ComgrHotswapAvailable()) {
    std::cerr << "hotswap: COMGR not available for B0->A0\n";
    return -1;
  }

  void *out_elf = nullptr;
  size_t out_elf_size = 0;
  int rc = ComgrHotswapRewriteB0A0(elf_data, elf_size,
                                   &out_elf, &out_elf_size);
  if (rc != 0) {
    std::cerr << "hotswap: COMGR B0->A0 rewrite failed (rc=" << rc << ")\n";
    if (out_elf)
      std::free(out_elf);
    return rc;
  }

  if (out_elf) {
    *out_data = out_elf;
    *out_size = out_elf_size;
  }
  return 0;
}

} // namespace hotswap
} // namespace rocr
