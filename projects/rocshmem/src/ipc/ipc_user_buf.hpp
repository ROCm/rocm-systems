/******************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *****************************************************************************/

#ifndef LIBRARY_SRC_IPC_USER_BUF_HPP_
#define LIBRARY_SRC_IPC_USER_BUF_HPP_

#include <stdint.h>

// Limits for constant-memory user buffer lookup on device
#define IPC_MAX_USER_BUFS 8
#define IPC_MAX_PES       16

namespace rocshmem {

// GPU-side user buffer info stored in __constant__ memory.
// Contains both local and remote base addresses so putmem can
// compute offsets with pure arithmetic (no HBM loads).
struct ipc_user_buf_entry_t {
  uintptr_t local_base;                   // local VA of this buffer
  uintptr_t remote_bases[IPC_MAX_PES];    // per-PE remote VAs
  size_t    length;                        // buffer length
};

// Host-side function to update the constant memory table
void ipc_user_buf_update_table(const ipc_user_buf_entry_t* entries, int count);

}  // namespace rocshmem

#endif  // LIBRARY_SRC_IPC_USER_BUF_HPP_
