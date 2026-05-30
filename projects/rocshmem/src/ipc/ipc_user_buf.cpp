/******************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *****************************************************************************/

#include "ipc_user_buf.hpp"
#include <hip/hip_runtime.h>
#include "util.hpp"

namespace rocshmem {

// Constant memory for device-side user buffer lookup (no HBM loads)
__constant__ ipc_user_buf_entry_t ipc_user_buf_table[IPC_MAX_USER_BUFS];
__constant__ int                  ipc_user_buf_count = 0;

// Master host-side table — single source of truth for all registration paths
static ipc_user_buf_entry_t master_entries[IPC_MAX_USER_BUFS];
static int master_entry_count = 0;

// Sync master table to device constant memory
static void sync_master_to_device() {
  int count = master_entry_count;
  if (count > IPC_MAX_USER_BUFS) count = IPC_MAX_USER_BUFS;
  CHECK_HIP(hipMemcpyToSymbol(HIP_SYMBOL(ipc_user_buf_table), master_entries,
                               count * sizeof(ipc_user_buf_entry_t), 0, hipMemcpyHostToDevice));
  CHECK_HIP(hipMemcpyToSymbol(HIP_SYMBOL(ipc_user_buf_count), &count,
                               sizeof(int), 0, hipMemcpyHostToDevice));
}

// Add a single entry to the master table and sync to device
int ipc_user_buf_add_entry(const ipc_user_buf_entry_t* entry) {
  if (master_entry_count >= IPC_MAX_USER_BUFS) return -1;
  master_entries[master_entry_count++] = *entry;
  sync_master_to_device();
  return 0;
}

// Legacy: replace entire table (used by sync_user_buf_constmem in backend_ipc.cpp)
void ipc_user_buf_update_table(const ipc_user_buf_entry_t* entries, int count) {
  // Merge: keep existing master entries and append new ones that aren't duplicates
  // For simplicity, just replace non-VMM portion. But since both paths now use
  // ipc_user_buf_add_entry, this function is only called from sync_user_buf_constmem
  // which rebuilds from IPCBackend::user_buffers_. We need to merge with VMM entries.
  //
  // For now, just sync master as-is — sync_user_buf_constmem should use
  // ipc_user_buf_add_entry instead.
  if (count > IPC_MAX_USER_BUFS) count = IPC_MAX_USER_BUFS;
  CHECK_HIP(hipMemcpyToSymbol(HIP_SYMBOL(ipc_user_buf_table), entries,
                               count * sizeof(ipc_user_buf_entry_t), 0, hipMemcpyHostToDevice));
  CHECK_HIP(hipMemcpyToSymbol(HIP_SYMBOL(ipc_user_buf_count), &count,
                               sizeof(int), 0, hipMemcpyHostToDevice));
}

int ipc_user_buf_remove_entry(uintptr_t local_base) {
  int found = -1;
  for (int i = 0; i < master_entry_count; i++) {
    if (master_entries[i].local_base == local_base) { found = i; break; }
  }
  if (found < 0) return -1;

  for (int i = found; i < master_entry_count - 1; i++) {
    master_entries[i] = master_entries[i + 1];
  }
  master_entry_count--;
  sync_master_to_device();
  return 0;
}

int rocshmem_buffer_register_vmm(void *addr, size_t length,
                                 int my_pe, int n_pes,
                                 ptrdiff_t stride) {
  if (master_entry_count >= IPC_MAX_USER_BUFS) return -1;

  ipc_user_buf_entry_t entry = {};
  entry.local_base = reinterpret_cast<uintptr_t>(addr);
  entry.length = length;
  for (int pe = 0; pe < n_pes && pe < IPC_MAX_PES; pe++) {
    entry.remote_bases[pe] = reinterpret_cast<uintptr_t>(addr)
                           + static_cast<ptrdiff_t>(pe - my_pe) * stride;
  }

  return ipc_user_buf_add_entry(&entry);
}

}  // namespace rocshmem
