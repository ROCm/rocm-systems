//===- hotswap.cpp - HotSwap ISA rewriting --------------------------------===//
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hotswap.hpp"
#include <amd_comgr.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>

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

  // Wrap input bytes in a COMGR data object.
  amd_comgr_data_t input = {0};
  amd_comgr_status_t status =
      amd_comgr_create_data(AMD_COMGR_DATA_KIND_EXECUTABLE, &input);
  if (status != AMD_COMGR_STATUS_SUCCESS) {
    fprintf(stderr, "hotswap: failed to create COMGR input data object\n");
    return static_cast<int>(status);
  }

  status = amd_comgr_set_data(input, elf_size, static_cast<const char *>(elf_data));
  if (status != AMD_COMGR_STATUS_SUCCESS) {
    fprintf(stderr, "hotswap: failed to set COMGR input data\n");
    amd_comgr_release_data(input);
    return static_cast<int>(status);
  }

  // Call the hotswap rewrite API.
  amd_comgr_data_t output = {0};
  status = amd_comgr_hotswap_rewrite(input, source_isa, target_isa, &output);
  amd_comgr_release_data(input);

  if (status != AMD_COMGR_STATUS_SUCCESS) {
    fprintf(stderr, "hotswap: COMGR rewrite failed for %s -> %s (rc=%d)\n",
            source_isa, target_isa, static_cast<int>(status));
    return static_cast<int>(status);
  }

  // Extract output bytes.
  size_t output_size = 0;
  status = amd_comgr_get_data(output, &output_size, nullptr);
  if (status != AMD_COMGR_STATUS_SUCCESS) {
    amd_comgr_release_data(output);
    fprintf(stderr, "hotswap: failed to query COMGR output size\n");
    return static_cast<int>(status);
  }
  if (output_size == 0) {
    amd_comgr_release_data(output);
    fprintf(stderr, "hotswap: COMGR returned empty output\n");
    return -1;
  }

  void *output_buf = std::malloc(output_size);
  if (!output_buf) {
    amd_comgr_release_data(output);
    fprintf(stderr, "hotswap: failed to allocate output buffer\n");
    return -1;
  }

  status = amd_comgr_get_data(output, &output_size,
                              static_cast<char *>(output_buf));
  amd_comgr_release_data(output);

  if (status != AMD_COMGR_STATUS_SUCCESS) {
    std::free(output_buf);
    fprintf(stderr, "hotswap: failed to extract COMGR output data\n");
    return static_cast<int>(status);
  }

  OwnedElf owned_output(output_buf, &std::free);
  *out_data = owned_output.release();
  *out_size = output_size;
  return 0;
}

} // namespace rocr::hotswap
