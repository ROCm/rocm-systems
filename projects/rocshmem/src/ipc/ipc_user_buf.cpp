/******************************************************************************
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *****************************************************************************/

#include "ipc_user_buf.hpp"
#include <hip/hip_runtime.h>
#include <vector>
#include "util.hpp"
#include "ipc_policy.hpp"

namespace rocshmem {

static IpcSymmTable *g_symm_table = nullptr;
static int g_num_pes = 0;

void ipc_user_buf_set_symm_table(IpcSymmTable *table, int num_pes) {
  g_symm_table = table;
  g_num_pes = num_pes;
}

int rocshmem_buffer_register_vmm(void *addr, size_t length,
                                 int my_pe, int n_pes,
                                 ptrdiff_t stride) {
  if (g_symm_table == nullptr) {
    LOG_ERROR("rocshmem_buffer_register_vmm: symm_table not initialized");
    return -1;
  }

  IpcSymmTable host_table{};
  CHECK_HIP(hipMemcpy(&host_table, g_symm_table, sizeof(IpcSymmTable),
                       hipMemcpyDeviceToHost));

  if (host_table.count >= host_table.capacity) {
    LOG_ERROR("rocshmem_buffer_register_vmm: symm_table full (%d/%d)",
              host_table.count, host_table.capacity);
    return -1;
  }

  int num_pes = (n_pes > 0) ? n_pes : g_num_pes;

  std::vector<char *> host_peer_bases(num_pes, nullptr);
  uintptr_t base = reinterpret_cast<uintptr_t>(addr);
  for (int pe = 0; pe < num_pes; pe++) {
    host_peer_bases[pe] = reinterpret_cast<char *>(
        base + static_cast<ptrdiff_t>(pe - my_pe) * stride);
  }

  char **peer_bases{nullptr};
  CHECK_HIP(hipMalloc(reinterpret_cast<void **>(&peer_bases),
                      num_pes * sizeof(char *)));
  CHECK_HIP(hipMemcpy(peer_bases, host_peer_bases.data(),
                      num_pes * sizeof(char *), hipMemcpyHostToDevice));

  IpcSymmRegion host_region{};
  host_region.local_base = base;
  host_region.length = length;
  host_region.peer_bases = peer_bases;

  int slot = host_table.count;
  CHECK_HIP(hipMemcpy(&host_table.regions[slot], &host_region,
                      sizeof(IpcSymmRegion), hipMemcpyHostToDevice));

  host_table.count = slot + 1;
  CHECK_HIP(hipMemcpy(g_symm_table, &host_table, sizeof(IpcSymmTable),
                      hipMemcpyHostToDevice));

  return 0;
}

}  // namespace rocshmem
