//===- mock_comgr.cpp - Mock amd_comgr_hotswap_rewrite for testing --------===//
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Minimal mock of libamd_comgr.so that exports amd_comgr_hotswap_rewrite.
// Returns a malloc'd copy of the input unchanged (passthrough).
//
//===----------------------------------------------------------------------===//

#include <cstdlib>
#include <cstring>

extern "C" {

__attribute__((visibility("default")))
int amd_comgr_hotswap_rewrite(
    const void *elf_data, size_t elf_size,
    const char *source_isa_name, const char *target_isa_name,
    void **out_elf, size_t *out_elf_size) {
  (void)source_isa_name;
  (void)target_isa_name;

  if (!elf_data || elf_size == 0 || !out_elf || !out_elf_size)
    return 1;

  void *copy = std::malloc(elf_size);
  if (!copy)
    return 2;

  std::memcpy(copy, elf_data, elf_size);
  *out_elf = copy;
  *out_elf_size = elf_size;
  return 0;
}

} // extern "C"
