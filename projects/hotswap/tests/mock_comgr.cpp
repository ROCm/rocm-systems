//===- mock_comgr.cpp - Mock amd_comgr_hotswap_rewrite for testing --------===//
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Minimal mock of libamd_comgr.so that exports amd_comgr_hotswap_rewrite.
// Accepts gfx1250 ISA pairs (returns a malloc'd copy unchanged).
// Rejects unsupported ISA pairs with INVALID_ARGUMENT (0x2).
//
// TODO(COMGR-upstream): Once the real amd_comgr_hotswap_rewrite lands in
// a released COMGR, this mock is only needed for test environments without
// COMGR installed. It can be moved behind a BUILD_TESTING guard.
//
//===----------------------------------------------------------------------===//

#include <cstdlib>
#include <cstring>

extern "C" {

__attribute__((visibility("default"))) int amd_comgr_hotswap_rewrite(
    const void *elf_data, size_t elf_size, const char *source_isa_name,
    const char *target_isa_name, void **out_elf, size_t *out_elf_size) {
  if (!elf_data || elf_size == 0 || !out_elf || !out_elf_size ||
      !source_isa_name || !target_isa_name) {
    return 1;
  }

  // Only accept gfx1250 pairs (matches real COMGR stub behavior)
  const bool supported =
      std::strstr(source_isa_name, "gfx1250") != nullptr &&
      std::strstr(target_isa_name, "gfx1250") != nullptr;
  if (!supported) {
    return 2; // AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT
  }

  void *copy = std::malloc(elf_size);
  if (!copy) {
    return 3; // AMD_COMGR_STATUS_ERROR_OUT_OF_RESOURCES
  }

  std::memcpy(copy, elf_data, elf_size);
  *out_elf = copy;
  *out_elf_size = elf_size;
  return 0;
}

} // extern "C"
