// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/* Modern ELF code objects do not declare legacy HSAIL execution properties. */
#include <hsa/hsa.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define REQUIRE(condition)                                                      \
  do {                                                                          \
    if (!(condition)) {                                                         \
      fprintf(stderr, "Code object check failed at line %d: %s\n", __LINE__,     \
              #condition);                                                      \
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

  /* ELF64 little-endian AMDGPU HSA ET_DYN header, without HSAIL notes. */
  uint8_t bytes[64] = {
      [0] = 0x7f, [1] = 'E', [2] = 'L', [3] = 'F', [4] = 2, [5] = 1,
      [6] = 1, [7] = 64, [16] = 3, [18] = 224, [20] = 1, [52] = 64};
  for (uint8_t abi = 1; abi <= 4; ++abi) {
    bytes[8] = abi;
    hsa_code_object_t object;
    REQUIRE(hsa_code_object_deserialize(bytes, sizeof(bytes), NULL, &object) ==
            HSA_STATUS_SUCCESS);
    char version[64] = {0};
    REQUIRE(hsa_code_object_get_info(object, HSA_CODE_OBJECT_INFO_VERSION,
                                     version) == HSA_STATUS_SUCCESS);
    char expected[] = {(char)('2' + abi), '.', '0', '\0'};
    REQUIRE(strcmp(version, expected) == 0);
    hsa_machine_model_t model;
    REQUIRE(hsa_code_object_get_info(object, HSA_CODE_OBJECT_INFO_MACHINE_MODEL,
                                     &model) == HSA_STATUS_SUCCESS);
    REQUIRE(model == HSA_MACHINE_MODEL_LARGE);
    uint32_t property = UINT32_MAX;
    REQUIRE(hsa_code_object_get_info(object, HSA_CODE_OBJECT_INFO_PROFILE,
                                     &property) ==
            HSA_STATUS_ERROR_INVALID_CODE_OBJECT);
    REQUIRE(property == UINT32_MAX);
    REQUIRE(hsa_code_object_get_info(
                object, HSA_CODE_OBJECT_INFO_DEFAULT_FLOAT_ROUNDING_MODE,
                &property) == HSA_STATUS_ERROR_INVALID_CODE_OBJECT);
    REQUIRE(property == UINT32_MAX);
    REQUIRE(hsa_code_object_destroy(object) == HSA_STATUS_SUCCESS);
  }
  REQUIRE(hsa_shut_down() == HSA_STATUS_SUCCESS);
  return 0;
}
