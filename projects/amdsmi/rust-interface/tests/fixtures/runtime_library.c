// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <stdint.h>

uint32_t amdsmi_init(uint64_t flags) { return flags == 2 ? 0 : 1; }

uint32_t amdsmi_shut_down(void) { return 0; }

uint32_t amdsmi_get_gpu_memory_total(void* handle, uint32_t memory_type, uint64_t* total) {
  if (handle != 0 || memory_type != 0 || total == 0) return 1;
  *total = 123456;
  return 0;
}
