/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Compiles src/transport/net_ib_cast/p2p_resiliency_recovery.cc into the including TU and defines the helpers
// that reach its statics. Include it from one TU per binary: the recovery test, or p2p_resiliency_recovery_wrapper.cc.

#ifndef RCCL_TEST_HOST_P2P_RESILIENCY_RECOVERY_UNIT_H_
#define RCCL_TEST_HOST_P2P_RESILIENCY_RECOVERY_UNIT_H_

#include <list>
#include <mutex>

#include "P2pResiliencyScene.h"
#include "fakes/param_redirect.h"

// Only malloc and free from fakes/libc_seam.h: its send/recv renames would hit the context's members.
extern "C" void* micro_malloc(size_t);
extern "C" void micro_free(void*);

#define clockNano P2pResiliencyTestClockNano
#define malloc micro_malloc
#define free micro_free
#include NET_IB_CAST_P2P_RESILIENCY_RECOVERY_CC_PATH
#undef free
#undef malloc
#undef clockNano

// ThreadStop clears the flag without the mutex, so its wakeup can be lost (AICOMRCCL-3103).
// Clearing it under the mutex first makes the stop deterministic.
void P2pResiliencyStopRecoveryThread() {
  if (IbCastPortRecoveryRefCount.load() == 1) {
    std::lock_guard<std::mutex> lock(IbCastPortRecoveryMutex);
    IbCastPortRecoveryThreadActive.store(false);
  }
  EXPECT_EQ(IbCastPortRecoveryThreadStop(), ncclSuccess);
}

int P2pResiliencyDrainRecoveryInbox() {
  std::list<ncclIbPortRecoveryContext*> queued;
  {
    std::lock_guard<std::mutex> lock(IbCastPortRecoveryMutex);
    queued.swap(recoveryInbox);
  }
  for (ncclIbPortRecoveryContext* ctx : queued) IbCastPortRecoveryContextDestroy(ctx);
  return queued.size();
}

#endif  // RCCL_TEST_HOST_P2P_RESILIENCY_RECOVERY_UNIT_H_
