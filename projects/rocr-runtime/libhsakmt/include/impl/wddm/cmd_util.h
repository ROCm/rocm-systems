/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <string.h>
#include "hsa-runtime/inc/hsa.h"
#include "hsa-runtime/inc/amd_hsa_queue.h"
#include "hsa-runtime/inc/amd_hsa_kernel_code.h"
#include "impl/pm4_cmds.h"
#include "util/utils.h"

namespace wsl {
namespace thunk {

struct DispatchInfo {
  uint8_t                       major;
  hsa_kernel_dispatch_packet_t  *pPacket;
  void                          *pEntry;
  const amd_kernel_code_t       *pKernelObject;
  uint32_t                      ldsBlks;
  amd_queue_v2_t                *pAmdQueue;
  bool                          wave32;
  uint32_t                      srd;
  void                          *pScratchBase;
  uint32_t                      scratchSizePerWave;
  uint32_t                      scratchBaseOffset[2];
  uint32_t                      offsetCnt;
};

class CmdUtil {
public:
  CmdUtil() {};
  ~CmdUtil() {};

  static size_t BuildCopyData(
    uint64_t  *pDstAddr,
    void      *pBuffer,
    uint32_t  dstSel = dst_sel__mec_copy_data__tc_l2,
    uint32_t  dstCachePolicy = dst_cache_policy__mec_copy_data__stream,
    uint32_t  srcSel = src_sel__mec_copy_data__gpu_clock_count,
    uint32_t  srcCachePolicy = src_cache_policy__mec_copy_data__lru,
    uint32_t  countSel = count_sel__mec_copy_data__64_bits_of_data,
    uint32_t  wrConfirm = wr_confirm__mec_copy_data__wait_for_confirmation);

  static size_t BuildBarrier(
    void      *pBuffer,
    uint32_t  eventIndex = event_index__mec_event_write__cs_partial_flush,
    uint32_t  eventType = CS_PARTIAL_FLUSH);

  static size_t BuildWriteData64Command(
    void      *pBuffer,
    uint64_t* write_addr,
    uint64_t write_value);

  static size_t BuildAcquireMem(
    uint8_t major,
    void    *pBuffer);

  static size_t BuildScratch(
    void  *pScratchBase,
    void  *pBuffer);

  static size_t BuildComputeShaderParams(
    void  *pBuffer);

  static size_t BuildDispatch(
    struct DispatchInfo *pInfo,
    void                *pBuffer);

  static size_t BuildAtomicMem(
    uint64_t  *pAddr,
    uint32_t  atomic,
    void      *pBuffer,
    uint32_t  cachePolicy = cache_policy__mec_atomic_mem__stream,
    uint64_t  srcData = 1);
};

} // namespace thunk
} // namespace wsl
