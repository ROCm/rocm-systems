// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <hsa/hsa.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

static void report_error(const char *operation, hsa_status_t status) {
  const char *description = NULL;
  if (hsa_status_string(status, &description) != HSA_STATUS_SUCCESS || !description)
    description = "unknown HSA status";
  fprintf(stderr, "%s failed: %s (0x%x)\n", operation, description,
          (unsigned int)status);
}

int main(void) {
  const char *requested = getenv("HSA_XNACK");
  printf("HSA_XNACK requested=%s\n", requested ? requested : "<unset>");
  fflush(stdout);

  hsa_status_t status = hsa_init();
  if (status != HSA_STATUS_SUCCESS) {
    report_error("hsa_init", status);
    return EXIT_FAILURE;
  }

  bool enabled = false;
  status = hsa_system_get_info(HSA_AMD_SYSTEM_INFO_XNACK_ENABLED, &enabled);
  int result = EXIT_SUCCESS;
  if (status != HSA_STATUS_SUCCESS) {
    report_error("hsa_system_get_info(HSA_AMD_SYSTEM_INFO_XNACK_ENABLED)", status);
    result = EXIT_FAILURE;
  } else {
    printf("HSA_XNACK effective=%d\n", enabled ? 1 : 0);
    fflush(stdout);
  }

  status = hsa_shut_down();
  if (status != HSA_STATUS_SUCCESS) {
    report_error("hsa_shut_down", status);
    result = EXIT_FAILURE;
  }
  return result;
}
