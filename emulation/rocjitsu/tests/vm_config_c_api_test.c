// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/api.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv) {
  if (argc != 2)
    return EXIT_FAILURE;
  FILE *input = fopen(argv[1], "rb");
  if (!input)
    return EXIT_FAILURE;
  if (fseek(input, 0, SEEK_END) != 0) {
    fclose(input);
    return EXIT_FAILURE;
  }
  const long length = ftell(input);
  if (length < 0 || (unsigned long)length >= SIZE_MAX || fseek(input, 0, SEEK_SET) != 0) {
    fclose(input);
    return EXIT_FAILURE;
  }
  char *json = malloc((size_t)length + 1);
  if (!json) {
    fclose(input);
    return EXIT_FAILURE;
  }
  const size_t read_length = fread(json, 1, (size_t)length, input);
  const int close_result = fclose(input);
  if (read_length != (size_t)length || close_result != 0) {
    free(json);
    return EXIT_FAILURE;
  }
  json[length] = '\0';

  int result = EXIT_FAILURE;
  rj_vm_config_t *config = NULL;
  rj_vm_config_t *selected = NULL;
  rj_vm_t *vm = NULL;
  uint32_t count = 0;
  rj_vm_gpu_info_t passive = {0};
  rj_vm_gpu_info_t active = {0};
  if (rj_vm_config_create_from_string(json, &config) != ROCJITSU_STATUS_SUCCESS ||
      rj_vm_config_create_from_string_for_node(json, "node0", &selected) != ROCJITSU_STATUS_SUCCESS)
    goto cleanup;
  free(json);
  json = NULL;
  if (rj_vm_config_gpu_count(config, &count) != ROCJITSU_STATUS_SUCCESS || count != 1 ||
      rj_vm_config_gpu_info_at(selected, 0, &passive) != ROCJITSU_STATUS_SUCCESS ||
      rj_vm_create_from_config(config, RJ_VM_MODE_LOCAL, &vm) != ROCJITSU_STATUS_SUCCESS)
    goto cleanup;
  rj_vm_config_destroy(config);
  config = NULL;
  rj_vm_config_destroy(selected);
  selected = NULL;
  if (rj_vm_gpu_info_at(vm, 0, &active) != ROCJITSU_STATUS_SUCCESS ||
      passive.gpu_id != active.gpu_id || passive.gfx_target_version != active.gfx_target_version ||
      strcmp(passive.marketing_name, active.marketing_name) != 0)
    goto cleanup;
  result = EXIT_SUCCESS;

cleanup:
  if (result != EXIT_SUCCESS)
    fputs("Passive VM configuration C API lifecycle failed\n", stderr);
  rj_vm_destroy(vm);
  rj_vm_config_destroy(selected);
  rj_vm_config_destroy(config);
  free(json);
  return result;
}
