// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCDDI_FAULT_IOCTL_H
#define ROCDDI_FAULT_IOCTL_H

#include <stdint.h>

enum {
  ROCDDI_FAULT_CREATE_EFAULT = 1,
  ROCDDI_FAULT_CU_MASK_ROLLBACK = 2,
  ROCDDI_FAULT_DOORBELL_ROLLBACK = 3
};

struct fault_stats {
  unsigned creates, create_faults, mask_faults, destroy_faults, allocations,
      frees;
  unsigned signal_seen, signal_freed, scratch_seen, scratch_freed;
  uint64_t signal_handle, scratch_handle;
};

void rocddi_fault_arm(int selected);
void rocddi_fault_stats(struct fault_stats *out);

#endif
