// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/* Modern ELF code objects do not declare legacy HSAIL execution properties. */
#include <hsa/hsa.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define REQUIRE(condition)                                                     \
  do {                                                                         \
    if (!(condition)) {                                                        \
      fprintf(stderr, "Code object check failed at line %d: %s\n", __LINE__,   \
              #condition);                                                     \
      return 1;                                                                \
    }                                                                          \
  } while (0)

static int check_object(void *bytes, size_t size, uint8_t abi) {
  hsa_code_object_t object;
  REQUIRE(hsa_code_object_deserialize(bytes, size, NULL, &object) ==
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
  REQUIRE(hsa_code_object_get_info(object, (hsa_code_object_info_t)UINT32_MAX,
                                   &property) ==
          HSA_STATUS_ERROR_INVALID_ARGUMENT);
  REQUIRE(property == UINT32_MAX);
  REQUIRE(hsa_code_object_get_info(
              object, HSA_CODE_OBJECT_INFO_DEFAULT_FLOAT_ROUNDING_MODE, NULL) ==
          HSA_STATUS_ERROR_INVALID_ARGUMENT);
  REQUIRE(hsa_code_object_destroy(object) == HSA_STATUS_SUCCESS);
  REQUIRE(hsa_code_object_get_info(
              object, HSA_CODE_OBJECT_INFO_DEFAULT_FLOAT_ROUNDING_MODE,
              &property) == HSA_STATUS_ERROR_INVALID_CODE_OBJECT);
  REQUIRE(property == UINT32_MAX);
  return 0;
}

int main(int argc, char **argv) {
  hsa_status_t initialized = hsa_init();
  if (initialized != HSA_STATUS_SUCCESS) {
    fprintf(stderr, "CPU-only HSA initialization unavailable: %#x\n",
            (unsigned)initialized);
    return 77;
  }

  /* ELF64 little-endian AMDGPU HSA ET_DYN header, without HSAIL notes. */
  uint8_t bytes[64] = {[0] = 0x7f, [1] = 'E',  [2] = 'L', [3] = 'F',
                       [4] = 2,    [5] = 1,    [6] = 1,   [7] = 64,
                       [16] = 3,   [18] = 224, [20] = 1,  [52] = 64};
  for (uint8_t abi = 1; abi <= 4; ++abi) {
    bytes[8] = abi;
    REQUIRE(check_object(bytes, sizeof(bytes), abi) == 0);
  }
  /* The same ABI checks accept compiler-produced fixtures as optional inputs.
   */
  for (int index = 1; index < argc; ++index) {
    FILE *file = fopen(argv[index], "rb");
    REQUIRE(file != NULL);
    REQUIRE(fseek(file, 0, SEEK_END) == 0);
    long length = ftell(file);
    REQUIRE(length >= 64);
    REQUIRE(fseek(file, 0, SEEK_SET) == 0);
    uint8_t *fixture = malloc((size_t)length);
    REQUIRE(fixture != NULL);
    REQUIRE(fread(fixture, 1, (size_t)length, file) == (size_t)length);
    REQUIRE(fclose(file) == 0);
    REQUIRE(fixture[8] >= 1 && fixture[8] <= 4);
    REQUIRE(check_object(fixture, (size_t)length, fixture[8]) == 0);
    free(fixture);
  }
  /* Legacy ABI versions and malformed ELF stay outside the modern decoder. */
  for (unsigned invalid = 0; invalid < 2; ++invalid) {
    bytes[8] = invalid == 0 ? 0 : 4;
    bytes[0] = invalid == 0 ? 0x7f : 0;
    hsa_code_object_t object;
    REQUIRE(hsa_code_object_deserialize(bytes, sizeof(bytes), NULL, &object) ==
            HSA_STATUS_SUCCESS);
    uint32_t property = UINT32_MAX;
    REQUIRE(hsa_code_object_get_info(
                object, HSA_CODE_OBJECT_INFO_DEFAULT_FLOAT_ROUNDING_MODE,
                &property) == HSA_STATUS_ERROR_INVALID_CODE_OBJECT);
    REQUIRE(property == UINT32_MAX);
    REQUIRE(hsa_code_object_destroy(object) == HSA_STATUS_SUCCESS);
  }
  const hsa_profile_t profiles[] = {HSA_PROFILE_BASE, HSA_PROFILE_FULL};
  const hsa_default_float_rounding_mode_t roundings[] = {
      HSA_DEFAULT_FLOAT_ROUNDING_MODE_DEFAULT,
      HSA_DEFAULT_FLOAT_ROUNDING_MODE_ZERO,
      HSA_DEFAULT_FLOAT_ROUNDING_MODE_NEAR};
  for (size_t profile = 0; profile < sizeof(profiles) / sizeof(profiles[0]);
       ++profile) {
    for (size_t rounding = 0;
         rounding < sizeof(roundings) / sizeof(roundings[0]); ++rounding) {
      hsa_executable_t executable;
      REQUIRE(hsa_executable_create_alt(profiles[profile], roundings[rounding],
                                        NULL,
                                        &executable) == HSA_STATUS_SUCCESS);
      hsa_profile_t actual_profile;
      hsa_default_float_rounding_mode_t actual_rounding;
      REQUIRE(hsa_executable_get_info(executable, HSA_EXECUTABLE_INFO_PROFILE,
                                      &actual_profile) == HSA_STATUS_SUCCESS);
      REQUIRE(actual_profile == profiles[profile]);
      REQUIRE(hsa_executable_get_info(
                  executable, HSA_EXECUTABLE_INFO_DEFAULT_FLOAT_ROUNDING_MODE,
                  &actual_rounding) == HSA_STATUS_SUCCESS);
      REQUIRE(actual_rounding == roundings[rounding]);
      REQUIRE(hsa_executable_destroy(executable) == HSA_STATUS_SUCCESS);
    }
  }
  hsa_executable_t executable = {UINT64_MAX};
  REQUIRE(hsa_executable_create_alt((hsa_profile_t)UINT32_MAX,
                                    HSA_DEFAULT_FLOAT_ROUNDING_MODE_DEFAULT,
                                    NULL, &executable) ==
          HSA_STATUS_ERROR_INVALID_ARGUMENT);
  REQUIRE(executable.handle == UINT64_MAX);
  REQUIRE(hsa_executable_create_alt(
              HSA_PROFILE_FULL, (hsa_default_float_rounding_mode_t)UINT32_MAX,
              NULL, &executable) == HSA_STATUS_ERROR_INVALID_ARGUMENT);
  REQUIRE(executable.handle == UINT64_MAX);
  REQUIRE(hsa_shut_down() == HSA_STATUS_SUCCESS);
  return 0;
}
