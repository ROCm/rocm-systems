/*************************************************************************
 * Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
 ************************************************************************/

#ifndef RCCL_CPU_PRIMITIVES_H_
#define RCCL_CPU_PRIMITIVES_H_

#include "cpu_device_context.h"
#include "cpu_func_decode.h"
#include "cpu_memory_model.h"

#include "device.h"

struct rcclCpuShmemGroup {
  struct ncclConnInfo* recvConns[NCCL_MAX_ARITY];
  struct ncclConnInfo* sendConns[NCCL_MAX_ARITY];
  void* userInput;
  void* userOutput;
  void* srcs[NCCL_MAX_ARITY + 1];
  void* dsts[NCCL_MAX_ARITY + 1];
  uint64_t barrier;
};

struct rcclCpuPrimitives {
  struct rcclCpuBlockContext* ctx;
  struct rcclCpuBlockBarrier* bar;
  struct rcclCpuFuncDesc desc;
  struct ncclDevWorkColl* work;
  struct rcclCpuShmemGroup group;

  int tid;
  int nthreads;
  int groupId;
  int recvPeer;
  int sendPeer;
  uint8_t connIndexRecv;
  uint8_t connIndexSend;

  int slicePerChunk;
  int stepPerSlice;
  int stepSize;

  int flags;
  int index;
  uint64_t step;
  uint64_t connStepCache;
  uint64_t* connStepPtr;
  int connStepSize;
  void* connEltsFifo;
  uint32_t* nextHdpReg;

  static constexpr int RoleWaitRecv = 1 << 0;
  static constexpr int RoleWaitSend = 1 << 1;
  static constexpr int RolePostRecv = 1 << 2;
  static constexpr int RolePostSend = 1 << 3;

  rcclCpuPrimitives(
      struct rcclCpuBlockContext* ctx, struct rcclCpuBlockBarrier* bar,
      struct rcclCpuFuncDesc const& desc, int tid, int tn,
      int const* recvPeers, int const* sendPeers,
      struct ncclDevWorkColl* work, int groupId = 0);

  void barrier();
  void subBarrier();

  void directSend(intptr_t inpIx, intptr_t outIx, int eltN);
  void directRecv(intptr_t outIx, int eltN, bool postOp = false);
  void directRecvReduceDirectSend(intptr_t inpIx, intptr_t outIx, int eltN, bool postOp = false);
  void directRecvReduceCopyDirectSend(intptr_t inpIx, intptr_t outIx, int eltN, bool postOp = false);
  void directRecvCopyDirectSend(intptr_t outIx, int eltN, bool postOp = false);
  void directCopySend(intptr_t inpIx, intptr_t outIx, int eltN, bool postOp = false);

 private:
  void genericOp(int recv, int send, int srcBuf, int dstBuf, intptr_t inpIx, intptr_t outIx, int eltN, bool postOp);
  void waitPeer(int recv, int send, int nbytes);
  void postPeer(int recv, int send, bool dataStored);
  uint64_t loadStepValue(uint64_t* ptr);
};

#endif
