//===- hotswap.cpp - HotSwap ISA rewriting --------------------------------===//
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hotswap.hpp"
#include <cstdio>
#include <cstdlib>
#include <memory>

// TODO(COMGR-upstream): Replace this forward declaration with
// #include <amd_comgr.h> once the hotswap API lands in a released COMGR.
// The return type is amd_comgr_status_t (enum, ABI-compatible with int).
extern "C" int amd_comgr_hotswap_rewrite(const void *elf_data, size_t elf_size,
                                         const char *source_isa_name,
                                         const char *target_isa_name,
                                         void **out_elf,
                                         size_t *out_elf_size);

namespace rocr::hotswap {

int RetargetCodeObject(const void *elf_data, size_t elf_size,
                       const char *source_isa, const char *target_isa,
                       void **out_data, size_t *out_size) {
  using OwnedElf = std::unique_ptr<void, decltype(&std::free)>;

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

  void *out_elf = nullptr;
  size_t out_elf_size = 0;
  const int rc = amd_comgr_hotswap_rewrite(elf_data, elf_size, source_isa,
                                           target_isa, &out_elf, &out_elf_size);
  OwnedElf owned_elf(out_elf, &std::free);
  if (rc != 0) {
    fprintf(stderr, "hotswap: COMGR rewrite failed for %s -> %s (rc=%d)\n",
            source_isa, target_isa, rc);
    return rc;
  }

  if (owned_elf) {
    *out_data = owned_elf.release();
    *out_size = out_elf_size;
  }
  return 0;
}

} // namespace rocr::hotswap
