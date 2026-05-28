/*************************************************************************
 * Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
 ************************************************************************/

#ifndef RCCL_CPU_HOST_COLL_H_
#define RCCL_CPU_HOST_COLL_H_

#include "collectives.h"
#include "device.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>

inline int rcclCpuProtoGrainSize(int proto) {
  return proto == NCCL_PROTO_LL ? 16 :
         proto == NCCL_PROTO_LL128 ? 64 :
         proto == NCCL_PROTO_SIMPLE ? 512 : -1;
}

template<typename Int>
inline void rcclCpuCollCbdPart(
    struct ncclDevWorkColl* work, uint32_t channelId, int proto, int eltSize,
    Int* count, Int* partOffset, Int* partCount, Int* chunkCount) {
  int eltPerGrain = rcclCpuProtoGrainSize(proto) / eltSize;
  int nMidChannels = work->channelHi - work->channelLo - 1;
  if (count != nullptr) {
    *count = work->cbd.countLo + work->cbd.countMid * nMidChannels + work->cbd.countHi;
  }
  if (channelId == work->channelLo) {
    *partOffset = 0;
    *partCount = work->cbd.countLo;
    *chunkCount = static_cast<int>(work->cbd.chunkGrainsLo * eltPerGrain);
  } else if (channelId == work->channelHi) {
    *partOffset = work->cbd.countLo + nMidChannels * work->cbd.countMid;
    *partCount = work->cbd.countHi;
    *chunkCount = static_cast<int>(work->cbd.chunkGrainsHi * eltPerGrain);
  } else {
    int mid = static_cast<int>(channelId) - work->channelLo - 1;
    *partOffset = work->cbd.countLo + mid * work->cbd.countMid;
    *partCount = work->cbd.countMid;
    *chunkCount = static_cast<int>(work->cbd.chunkGrainsMid * eltPerGrain);
  }
}

inline int64_t rcclCpuAlignUp(int64_t x, int a) {
  return (x + a - 1) / a * a;
}

inline int64_t rcclCpuDivUp(int64_t a, int64_t b) {
  return (a + b - 1) / b;
}

#endif
