// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <amd_cuid.h>
#include <stdio.h>

int main(void) {
  uint32_t count = 0;
  const amdcuid_status_t status = amdcuid_get_all_handles(NULL, &count);
  printf("%s %s\n", amdcuid_library_version_to_string(), amdcuid_status_to_string(status));
  return 0;
}
