// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/* Pre-initialization ABI check: does not discover or activate GPU devices. */
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>
#include <hsa/hsa_ext_image.h>
#include <stdio.h>

static int check_not_initialized(const char *name, hsa_status_t status) {
  if (status == HSA_STATUS_ERROR_NOT_INITIALIZED) {
    return 0;
  }
  fprintf(stderr, "%s before init returned %u\n", name, (unsigned)status);
  return 1;
}

int main(void) {
  const char *description = NULL;
  hsa_status_t status = hsa_status_string(HSA_STATUS_SUCCESS, &description);
  if (status != HSA_STATUS_SUCCESS || description == NULL || description[0] == '\0') {
    fprintf(stderr, "HSA status-string ABI check failed\n");
    return 1;
  }
  uint16_t major = 0;
  status = hsa_system_get_info(HSA_SYSTEM_INFO_VERSION_MAJOR, &major);
  if (check_not_initialized("system query", status)) {
    return 1;
  }
  const hsa_agent_t no_agent = {0};
  status = hsa_ext_image_get_capability_with_layout(
      no_agent, HSA_EXT_IMAGE_GEOMETRY_1D, NULL,
      HSA_EXT_IMAGE_DATA_LAYOUT_OPAQUE, NULL);
  if (check_not_initialized("image layout query", status)) {
    return 1;
  }
  status = hsa_amd_image_get_info_max_dim(no_agent, (hsa_agent_info_t)0, NULL);
  if (check_not_initialized("image dimension query", status)) {
    return 1;
  }
  status = hsa_amd_vmem_address_reserve(NULL, 0, 0, 0);
  if (check_not_initialized("virtual address reservation", status)) {
    return 1;
  }
  return 0;
}
