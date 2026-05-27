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

// Host-side function to update the constant memory table
void ipc_user_buf_update_table(const ipc_user_buf_entry_t* entries, int count) {
  if (count > IPC_MAX_USER_BUFS) count = IPC_MAX_USER_BUFS;
  CHECK_HIP(hipMemcpyToSymbol(HIP_SYMBOL(ipc_user_buf_table), entries,
                               count * sizeof(ipc_user_buf_entry_t), 0, hipMemcpyHostToDevice));
  CHECK_HIP(hipMemcpyToSymbol(HIP_SYMBOL(ipc_user_buf_count), &count,
                               sizeof(int), 0, hipMemcpyHostToDevice));
}

// Accumulated entries for the VMM registration path
static ipc_user_buf_entry_t vmm_entries[IPC_MAX_USER_BUFS];
static int vmm_entry_count = 0;

int rocshmem_buffer_register_vmm(void *addr, size_t length,
                                 int my_pe, int n_pes,
                                 ptrdiff_t stride) {
  int dev = -1;
  hipGetDevice(&dev);
  fprintf(stderr, "[rocshmem] buffer_register_vmm: addr=%p len=%zu my_pe=%d n_pes=%d stride=%ld hipDevice=%d\n",
          addr, length, my_pe, n_pes, (long)stride, dev);
  if (vmm_entry_count >= IPC_MAX_USER_BUFS) return -1;

  ipc_user_buf_entry_t &entry = vmm_entries[vmm_entry_count];
  entry.local_base = reinterpret_cast<uintptr_t>(addr);
  entry.length = length;
  for (int pe = 0; pe < n_pes && pe < IPC_MAX_PES; pe++) {
    entry.remote_bases[pe] = reinterpret_cast<uintptr_t>(addr)
                           + static_cast<ptrdiff_t>(pe - my_pe) * stride;
  }
  vmm_entry_count++;

  ipc_user_buf_update_table(vmm_entries, vmm_entry_count);

  // Verify the write: read back ipc_user_buf_count from device
  int readback = -1;
  hipMemcpyFromSymbol(&readback, HIP_SYMBOL(ipc_user_buf_count), sizeof(int), 0, hipMemcpyDeviceToHost);
  fprintf(stderr, "[rocshmem] buffer_register_vmm: wrote count=%d, readback=%d\n",
          vmm_entry_count, readback);

  return 0;
}

}  // namespace rocshmem
