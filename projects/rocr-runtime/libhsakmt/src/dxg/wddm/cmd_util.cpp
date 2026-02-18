/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include "impl/wddm/cmd_util.h"

namespace wsl {
namespace thunk {

/*
 * Builds a COPY_DATA packet that copies data.
 */
size_t CmdUtil::BuildCopyData(
  uint64_t  *pDstAddr,
  void      *pBuffer,
  uint32_t  dstSel,
  uint32_t  dstCachePolicy,
  uint32_t  srcSel,
  uint32_t  srcCachePolicy,
  uint32_t  countSel,
  uint32_t  wrConfirm) {
  PM4MEC_COPY_DATA copy_data = {0};

  GenerateCmdHeader(&copy_data, IT_COPY_DATA);
  copy_data.bitfields2.dst_sel = dstSel;
  copy_data.bitfields2.src_sel = srcSel;
  copy_data.bitfields2.dst_cache_policy = dstCachePolicy;
  copy_data.bitfields2.src_cache_policy = srcCachePolicy;
  copy_data.bitfields2.count_sel = countSel;
  copy_data.bitfields2.wr_confirm = wrConfirm;
  copy_data.bitfields5c.dst_64b_addr_lo = (PtrLow32(pDstAddr) >> 3);
  copy_data.dst_addr_hi = PtrHigh32(pDstAddr);
  memcpy(pBuffer, &copy_data, sizeof(copy_data));

  return sizeof(copy_data);
}

/*
 * Builds a EVENT_WRITE packet.
 * Applications can use Barrier command to ensure their
 * command is executed only after all other commands have
 * completed their execution.
 */
size_t CmdUtil::BuildBarrier(
  void      *pBuffer,
  uint32_t  eventIndex,
  uint32_t  eventType) {
  BarrierTemplate barrier = {0};

  GenerateCmdHeader(&barrier.event_write, IT_EVENT_WRITE);
  barrier.event_write.bitfields2.event_index = eventIndex;
  barrier.event_write.bitfields2.event_type = eventType;
  memcpy(pBuffer, &barrier, sizeof(barrier));

  return sizeof(barrier);
}

/**
 * Builds a WRITE_DATA packet.
 * Writes two DWORDs into the GPU memory address "write_addr"
 */

size_t CmdUtil::BuildWriteData64Command(
  void*     pBuffer,
  uint64_t* write_addr,
  uint64_t  write_value) {
  WriteDataTemplate command = {0};
  GenerateCmdHeader(&command.write_data, IT_WRITE_DATA);

  // Encode the user specified address to write to
  uint64_t addr = uintptr_t(write_addr);
  assert(!(addr & 0x3) && "WriteData address must be 4 byte aligned");

  // Set the bit to confirm the write operation and cache policy
  command.write_data.bitfields2.wr_confirm = wr_confirm__mec_write_data__wait_for_write_confirmation;
  command.write_data.bitfields2.cache_policy = cache_policy__mec_write_data__bypass;

  // Specify the command to increment address if writing more than one DWord
  command.write_data.bitfields2.addr_incr = addr_incr__mec_write_data__increment_address;
  // Specify the class to which the write destination belongs
  command.write_data.bitfields2.dst_sel = dst_sel__mec_write_data__memory;

  command.write_data.bitfields3c.dst_mem_addr_lo = (PtrLow32(write_addr) >> 2);
  command.write_data.dst_mem_addr_hi = PtrHigh32(write_addr);

  // Specify the value to write
  command.write_data.write_data_value = write_value;

  memcpy(pBuffer, &command, sizeof(command));
  return sizeof(command);
}

/*
 * Builds a ACQUIRE_MEM packet.
 * Users can submit this command to
 * invalidate Gpu caches - L1 and or L2.
 */
size_t CmdUtil::BuildAcquireMem(
  uint8_t major,
  void    *pBuffer) {
  size_t ret;
  if (major == 9) {
    gfx9::AcquireMemTemplate acq = {0};
    GenerateCmdHeader(&acq.acquire_mem, IT_ACQUIRE_MEM);
    // Specify the size of memory to invalidate. Size is
    // specified in terms of 256 byte chunks. A coher_size
    // of 0xFFFFFFFF actually specified 0xFFFFFFFF00 (40 bits)
    // of memory. The field coher_size_hi specifies memory from
    // bits 40-64 for a total of 256 TB.
    acq.acquire_mem.coher_size = 0xFFFFFFFF;
    acq.acquire_mem.bitfields4.coher_size_hi = 0xFF;
    // Specify the address of memory to invalidate. The
    // address must be 256 byte aligned.
    acq.acquire_mem.coher_base_lo = 0;
    acq.acquire_mem.bitfields6.coher_base_hi = 0;
    // Specify the poll interval for determing if operation is complete
    acq.acquire_mem.bitfields7.poll_interval = 4;
    acq.acquire_mem.bitfields2.coher_cntl =
      (1 << 29) | // CP_COHER_CNTL__SH_ICACHE_ACTION_ENA_MASK
      (1 << 27) | // CP_COHER_CNTL__SH_KCACHE_ACTION_ENA_MASK
      (1 << 28);  // CP_COHER_CNTL__SH_KCACHE_VOL_ACTION_ENA_MASK
    memcpy(pBuffer, &acq, sizeof(acq));
    ret = sizeof(acq);
  } else if (major >= 10) {
    gfx10::AcquireMemTemplate acq = {0};
    GenerateCmdHeader(&acq.acquire_mem, IT_ACQUIRE_MEM);
    acq.acquire_mem.coher_size = 0xFFFFFFFF;
    acq.acquire_mem.bitfields4.coher_size_hi = 0xFF;
    acq.acquire_mem.coher_base_lo = 0;
    acq.acquire_mem.bitfields6.coher_base_hi = 0;
    acq.acquire_mem.bitfields7.poll_interval = 4;
    acq.acquire_mem.bitfields8.gcr_cntl =
      (1 << 16) | // SEQ = FORWARD
      (1 << 15) | // GL2_WB
      (1 << 14) | // GL2_INV
      (1 << 9) |  // GL1_INV
      (1 << 8) |  // GLV_INV
      (1 << 7) |  // GLK_INV
      (1 << 6) |  // GLK_WB
      (1 << 5) |  // GLM_INV
      (1 << 4) |  // GLM_WB
      (1 << 0);   // GLI_INV = ALL
    memcpy(pBuffer, &acq, sizeof(acq));
    ret = sizeof(acq);
  }

  return ret;
}

/*
 * Builds a scratch packet.
 */
size_t CmdUtil::BuildScratch(
  void  *pScratchBase,
  void  *pBuffer) {
  struct SetScratchTemplate scratch = {0};

  GenerateSetShRegHeader(&scratch, mmCOMPUTE_DISPATCH_SCRATCH_BASE_LO);
  scratch.scratch_lo = Ptr48Low32(pScratchBase);
  scratch.scratch_hi = Ptr48High8(pScratchBase);
  memcpy(pBuffer, &scratch, sizeof(scratch));

  return sizeof(scratch);
}

/**
 * @ Set Compute Shader parameter for gfx11 and above
 */
size_t CmdUtil::BuildComputeShaderParams(void  *pBuffer) {
  struct DispatchProgramResourceRegs compute_shader_params = {0};

  GenerateSetShRegHeader(&compute_shader_params, mmCOMPUTE_PGM_RSRC3);
  // IMAGE_OP: Indicates the compute program contains an image op
  // instruction and should be stalled by its WAIT_SYNC fence.
  compute_shader_params.compute_pgm_rsrc3 = (1 << 31);

  memcpy(pBuffer, &compute_shader_params, sizeof(compute_shader_params));

  return sizeof(compute_shader_params);
}


/*
 * Builds a dispatch packet.
 */
size_t CmdUtil::BuildDispatch(
  struct DispatchInfo *pInfo,
  void                *pBuffer) {
  DispatchTemplate dispatch = {0};

  GenerateSetShRegHeader(&dispatch.dimension_regs, mmCOMPUTE_NUM_THREAD_X);
  dispatch.dimension_regs.compute_num_thread_x = pInfo->pPacket->workgroup_size_x;
  dispatch.dimension_regs.compute_num_thread_y = pInfo->pPacket->workgroup_size_y;
  dispatch.dimension_regs.compute_num_thread_z = pInfo->pPacket->workgroup_size_z;

  // TODO: Add AQL packet index for debugger
  // Debugger requires AQL packet index in COMPUTE_DISPATCH_PKT_ADDR_LO
  GenerateSetShRegHeader(&dispatch.program_regs, mmCOMPUTE_PGM_LO);
  dispatch.program_regs.compute_pgm_lo = Ptr48Low32(pInfo->pEntry);
  dispatch.program_regs.compute_pgm_hi = Ptr48High8(pInfo->pEntry);

  GenerateSetShRegHeader(&dispatch.program_resource_regs, mmCOMPUTE_PGM_RSRC1);
  dispatch.program_resource_regs.compute_pgm_rsrc1 = pInfo->pKernelObject->compute_pgm_rsrc1;
  if (pInfo->major == 11) {
    AMD_HSA_BITS_SET(dispatch.program_resource_regs.compute_pgm_rsrc1,
        AMD_COMPUTE_PGM_RSRC_ONE_PRIV, 1);
  }
  dispatch.program_resource_regs.compute_pgm_rsrc2 =
    (pInfo->ldsBlks << 15) | pInfo->pKernelObject->compute_pgm_rsrc2;

  GenerateSetShRegHeader(&dispatch.resource_regs, mmCOMPUTE_RESOURCE_LIMITS);
  dispatch.resource_regs.compute_resource_limits = 0x3ff;
  dispatch.resource_regs.compute_static_thread_mgmt_se0 = 0xFFFFFFFF;
  dispatch.resource_regs.compute_static_thread_mgmt_se1 = 0xFFFFFFFF;
  dispatch.resource_regs.compute_static_thread_mgmt_se2 = 0xFFFFFFFF;
  dispatch.resource_regs.compute_static_thread_mgmt_se3 = 0xFFFFFFFF;

  dispatch.resource_regs.compute_tmpring_size = pInfo->pAmdQueue->compute_tmpring_size;

  GenerateSetShRegHeader(&dispatch.compute_user_data_regs, mmCOMPUTE_USER_DATA_0);

  uint32_t sgpr_no = 0;
  if (AMD_HSA_BITS_GET(pInfo->pKernelObject->kernel_code_properties,
		       AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_PRIVATE_SEGMENT_BUFFER)) {
    assert(pInfo->major < 11);
    pInfo->scratchBaseOffset[pInfo->offsetCnt++] =
      offsetof(struct DispatchTemplate, compute_user_data_regs.compute_user_data[0]) +
      sgpr_no * sizeof(uint32_t);

    dispatch.compute_user_data_regs.compute_user_data[sgpr_no++] =
      pInfo->pAmdQueue->scratch_resource_descriptor[0];
    dispatch.compute_user_data_regs.compute_user_data[sgpr_no++] =
      pInfo->pAmdQueue->scratch_resource_descriptor[1];
    dispatch.compute_user_data_regs.compute_user_data[sgpr_no++] =
      pInfo->pAmdQueue->scratch_resource_descriptor[2];
    dispatch.compute_user_data_regs.compute_user_data[sgpr_no++] =
      pInfo->srd;
  }
  if (AMD_HSA_BITS_GET(pInfo->pKernelObject->kernel_code_properties,
		       AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_DISPATCH_PTR)) {
    dispatch.compute_user_data_regs.compute_user_data[sgpr_no++] = PtrLow32(pInfo->pPacket);
    dispatch.compute_user_data_regs.compute_user_data[sgpr_no++] = PtrHigh32(pInfo->pPacket);
  }
  if (AMD_HSA_BITS_GET(pInfo->pKernelObject->kernel_code_properties,
		       AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_QUEUE_PTR)) {
    dispatch.compute_user_data_regs.compute_user_data[sgpr_no++] = PtrLow32(pInfo->pAmdQueue);
    dispatch.compute_user_data_regs.compute_user_data[sgpr_no++] = PtrHigh32(pInfo->pAmdQueue);
  }
  if (AMD_HSA_BITS_GET(pInfo->pKernelObject->kernel_code_properties,
		       AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_KERNARG_SEGMENT_PTR)) {
    dispatch.compute_user_data_regs.compute_user_data[sgpr_no++] =
      PtrLow32(pInfo->pPacket->kernarg_address);
    dispatch.compute_user_data_regs.compute_user_data[sgpr_no++] =
      PtrHigh32(pInfo->pPacket->kernarg_address);
  }
  if (AMD_HSA_BITS_GET(pInfo->pKernelObject->kernel_code_properties,
		       AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_DISPATCH_ID)) {
    // This feature may be enabled as a side effect of indirect calls.
    // However, the compiler team confirmed that the dispatch id itself is not used,
    // so safe to send 0 for each dispatch.
    dispatch.compute_user_data_regs.compute_user_data[sgpr_no++] = 0;
    dispatch.compute_user_data_regs.compute_user_data[sgpr_no++] = 0;
  }
  if (AMD_HSA_BITS_GET(pInfo->pKernelObject->kernel_code_properties,
		       AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_FLAT_SCRATCH_INIT)) {
    assert(pInfo->major < 11);
    pInfo->scratchBaseOffset[pInfo->offsetCnt++] =
      offsetof(struct DispatchTemplate, compute_user_data_regs.compute_user_data[0]) +
      sgpr_no * sizeof(uint32_t);

    dispatch.compute_user_data_regs.compute_user_data[sgpr_no++] =
      PtrLow32(pInfo->pScratchBase);
    dispatch.compute_user_data_regs.compute_user_data[sgpr_no++] =
      PtrHigh32(pInfo->pScratchBase);
  }
  if (AMD_HSA_BITS_GET(pInfo->pKernelObject->kernel_code_properties,
		       AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_PRIVATE_SEGMENT_SIZE)) {
    dispatch.compute_user_data_regs.compute_user_data[sgpr_no++] =
      pInfo->scratchSizePerWave / (pInfo->wave32 ? 32 : 64);
  }

  GenerateCmdHeader(&dispatch.dispatch_direct, IT_DISPATCH_DIRECT);
  dispatch.dispatch_direct.dispatch_initiator =
    (1 << 0) | // COMPUTE_SHADER_EN
    (1 << 2) | // FORCE_START_AT_000
    (1 << 5); // USE_THREAD_DIMENSIONS
  if (pInfo->wave32) dispatch.dispatch_direct.dispatch_initiator |= (1 << 15); // CS_W32_EN
  dispatch.dispatch_direct.dim_x = pInfo->pPacket->grid_size_x;
  dispatch.dispatch_direct.dim_y = pInfo->pPacket->grid_size_y;
  dispatch.dispatch_direct.dim_z = pInfo->pPacket->grid_size_z;
  memcpy(pBuffer, &dispatch, sizeof(dispatch));

  return sizeof(dispatch);
}

/*
 * Builds a ATOMIC_MEM packet.
 * Users can submit this command
 * to perform atomic operations.
 */
size_t CmdUtil::BuildAtomicMem(
  uint64_t  *pAddr,
  uint32_t  atomic,
  void      *pBuffer,
  uint32_t  cachePolicy,
  uint64_t  srcData) {
  AtomicTemplate atom = {0};

  GenerateCmdHeader(&atom.atomic, IT_ATOMIC_MEM);
  atom.atomic.addr_lo = PtrLow32(pAddr);
  atom.atomic.addr_hi = PtrHigh32(pAddr);
  atom.atomic.bitfields2.atomic = atomic;
  atom.atomic.bitfields2.cache_policy = cachePolicy;
  atom.atomic.src_data_lo = LowPart(srcData);
  atom.atomic.src_data_hi = HighPart(srcData);
  memcpy(pBuffer, &atom, sizeof(atom));

  return sizeof(atom);
}

} // namespace thunk
} // namespace wsl
