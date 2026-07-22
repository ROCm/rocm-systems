/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef _NCCL_DEVICE_GIN_ANVIL_IPC_TABLE_DEVICE_H_
#define _NCCL_DEVICE_GIN_ANVIL_IPC_TABLE_DEVICE_H_

/** Device-only IPC peer VA lookup. Include from gin_anvil_sdma.h, not host .cc files. */

#include "gin_anvil_ipc_table.h"
#include "../../hip_compat.h"
#include "../../utility.h"

namespace nccl {
namespace gin {
namespace anvil {
namespace detail {

NCCL_DEVICE_INLINE void* ginAnvilResolvePeerVa(void* localAddr, int peer,
                                               const ncclGinAnvilIpcBufEntry* table, int count) {
  if (table == nullptr || count <= 0) return nullptr;
  uintptr_t addr = reinterpret_cast<uintptr_t>(localAddr);
  for (int b = 0; b < count; b++) {
    uintptr_t base = nccl::utility::loadConst(&table[b].local_base);
    size_t len = nccl::utility::loadConst(&table[b].length);
    if (addr >= base && addr < base + len) {
      uintptr_t off = addr - base;
      if (peer < 0 || peer >= NCCL_GIN_ANVIL_IPC_MAX_RANKS) return nullptr;
      uintptr_t remote = nccl::utility::loadConst(&table[b].remote_bases[peer]);
      return reinterpret_cast<void*>(remote + off);
    }
  }
  return nullptr;
}

}  // namespace detail
}  // namespace anvil
}  // namespace gin
}  // namespace nccl

#endif  // _NCCL_DEVICE_GIN_ANVIL_IPC_TABLE_DEVICE_H_
