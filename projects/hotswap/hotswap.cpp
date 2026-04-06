//===- hotswap.cpp - HotSwap ISA rewriting (stub) -------------------------===//
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hotswap.hpp"
#include "hotswap_comgr_client.hpp"
#include <cstdio>
#include <cstdlib>

namespace rocr {
namespace hotswap {

int RetargetCodeObject(const void *elf_data, size_t elf_size,
                       const char *source_isa, const char *target_isa,
                       void **out_data, size_t *out_size) {
  if (!out_data || !out_size) {
    fprintf(stderr, "hotswap: invalid null output pointer(s)\n");
    return -1;
  }

  *out_data = const_cast<void *>(elf_data);
  *out_size = elf_size;

  if (!elf_data || elf_size == 0 || !source_isa || !target_isa) {
    fprintf(stderr, "hotswap: invalid null input argument(s)\n");
    return -1;
  }

  if (!ComgrHotswapAvailable()) {
    fprintf(stderr, "hotswap: COMGR not available for %s -> %s\n",
            source_isa ? source_isa : "(null)",
            target_isa ? target_isa : "(null)");
    return -1;
  }

  void *out_elf = nullptr;
  size_t out_elf_size = 0;
  int rc = ComgrHotswapRewrite(elf_data, elf_size,
                               source_isa, target_isa,
                               &out_elf, &out_elf_size);
  if (rc != 0) {
    fprintf(stderr, "hotswap: COMGR rewrite failed for %s -> %s (rc=%d)\n",
            source_isa ? source_isa : "(null)",
            target_isa ? target_isa : "(null)", rc);
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
