// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/* CPU-only loader extension discovery and version-specific table retrieval. */
#include <hsa/hsa.h>
#include <hsa/hsa_ven_amd_loader.h>
#include <stdbool.h>
#include <stdio.h>

#define REQUIRE(condition)                                                      \
  do {                                                                          \
    if (!(condition)) {                                                         \
      fprintf(stderr, "Loader extension check failed at line %d: %s\n",         \
              __LINE__, #condition);                                            \
      return 1;                                                                 \
    }                                                                           \
  } while (0)

int main(void) {
  hsa_status_t initialized = hsa_init();
  if (initialized != HSA_STATUS_SUCCESS) {
    fprintf(stderr, "CPU-only HSA initialization unavailable: %#x\n",
            (unsigned)initialized);
    return 77;
  }

  bool supported = false;
  REQUIRE(hsa_system_extension_supported(HSA_EXTENSION_AMD_LOADER, 1, 4,
                                         &supported) == HSA_STATUS_SUCCESS);
  REQUIRE(supported);
  hsa_ven_amd_loader_1_04_pfn_t table = {0};
  REQUIRE(hsa_system_get_extension_table(HSA_EXTENSION_AMD_LOADER, 1, 4,
                                         &table) == HSA_STATUS_SUCCESS);
  REQUIRE(table.hsa_ven_amd_loader_query_host_address != NULL);
  REQUIRE(table.hsa_ven_amd_loader_query_segment_descriptors != NULL);
  REQUIRE(table.hsa_ven_amd_loader_query_executable != NULL);
  REQUIRE(table.hsa_ven_amd_loader_executable_iterate_loaded_code_objects != NULL);
  REQUIRE(table.hsa_ven_amd_loader_loaded_code_object_get_info != NULL);
  REQUIRE(table.hsa_ven_amd_loader_iterate_executables != NULL);
  REQUIRE(table.hsa_ven_amd_loader_code_object_reader_create_from_file_with_offset_size !=
          NULL);
  REQUIRE(table.hsa_ven_amd_loader_code_object_reader_get_load_size != NULL);
  REQUIRE(table.hsa_ven_amd_loader_executable_load_agent_code_object_at_address !=
          NULL);
  REQUIRE(hsa_system_extension_supported(HSA_EXTENSION_AMD_LOADER, 1, 5,
                                         &supported) == HSA_STATUS_SUCCESS);
  REQUIRE(!supported);
  REQUIRE(hsa_shut_down() == HSA_STATUS_SUCCESS);
  return 0;
}
