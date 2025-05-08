////////////////////////////////////////////////////////////////////////////////
//
// The University of Illinois/NCSA
// Open Source License (NCSA)
//
// Copyright (c) 2014-2024, Advanced Micro Devices, Inc. All rights reserved.
//
// Developed by:
//
//                 AMD Research and AMD HSA Software Development
//
//                 Advanced Micro Devices, Inc.
//
//                 www.amd.com
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to
// deal with the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
//  - Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimers.
//  - Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimers in
//    the documentation and/or other materials provided with the distribution.
//  - Neither the names of Advanced Micro Devices, Inc,
//    nor the names of its contributors may be used to endorse or promote
//    products derived from this Software without specific prior written
//    permission.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
// THE CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
// OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
// ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS WITH THE SOFTWARE.
//
////////////////////////////////////////////////////////////////////////////////

#include "core/inc/amd_blit_kernel.h"

#include <algorithm>
#include <sstream>
#include <string>

#include "core/inc/amd_gpu_agent.h"
#include "core/inc/hsa_internal.h"
#include "core/util/utils.h"

namespace rocr {
namespace AMD {
static const uint16_t kInvalidPacketHeader = HSA_PACKET_TYPE_INVALID;

static std::string& kBlitKernelSource() {
  static std::string kBlitKernelSource_(R"(
  // Compatibility function for GFXIP 7.

  function s_load_dword_offset(byte_offset)
    if kGFXIPVersion == 7
      return byte_offset / 4
    else
      return byte_offset
    end
  end

  // Memory copy for all cases except:
  //  (src_addr & 0x3) != (dst_addr & 0x3)
  //
  // Kernel argument buffer:
  //   [DW  0, 1]  Phase 1 src start address
  //   [DW  2, 3]  Phase 1 dst start address
  //   [DW  4, 5]  Phase 2 src start address
  //   [DW  6, 7]  Phase 2 dst start address
  //   [DW  8, 9]  Phase 3 src start address
  //   [DW 10,11]  Phase 3 dst start address
  //   [DW 12,13]  Phase 4 src start address
  //   [DW 14,15]  Phase 4 dst start address
  //   [DW 16,17]  Phase 4 src end address
  //   [DW 18,19]  Phase 4 dst end address
  //   [DW 20   ]  Total number of workitems

  var kCopyAlignedVecWidth = 4
  var kCopyAlignedUnroll = 1

  shader CopyAligned
    type(CS)
    user_sgpr_count(2)
    sgpr_count(32)
    vgpr_count(8 + (kCopyAlignedUnroll * kCopyAlignedVecWidth))

    // Retrieve kernel arguments.
    s_load_dwordx4          s[4:7], s[0:1], s_load_dword_offset(0x0)
    s_load_dwordx4          s[8:11], s[0:1], s_load_dword_offset(0x10)
    s_load_dwordx4          s[12:15], s[0:1], s_load_dword_offset(0x20)
    s_load_dwordx4          s[16:19], s[0:1], s_load_dword_offset(0x30)
    s_load_dwordx4          s[20:23], s[0:1], s_load_dword_offset(0x40)
    s_load_dword            s24, s[0:1], s_load_dword_offset(0x50)
    s_waitcnt               lgkmcnt(0)

    // Compute workitem id.
    s_lshl_b32              s2, s2, 0x6
    v_add_u32               v0, vcc, s2, v0

    // =====================================================
    // Phase 1: Byte copy up to 0x100 destination alignment.
    // =====================================================

    // Compute phase source address.
    v_mov_b32               v3, s5
    v_add_u32               v2, vcc, v0, s4
    v_addc_u32              v3, vcc, v3, 0x0, vcc

    // Compute phase destination address.
    v_mov_b32               v5, s7
    v_add_u32               v4, vcc, v0, s6
    v_addc_u32              v5, vcc, v5, 0x0, vcc

  L_COPY_ALIGNED_PHASE_1_LOOP:
    // Mask off lanes (or branch out) after phase end.
    v_cmp_lt_u64            vcc, v[2:3], s[8:9]
    s_cbranch_vccz          L_COPY_ALIGNED_PHASE_1_DONE
    s_and_b64               exec, exec, vcc

    // Load from/advance the source address.
    flat_load_ubyte         v1, v[2:3]
    s_waitcnt               vmcnt(0)
    v_add_u32               v2, vcc, v2, s24
    v_addc_u32              v3, vcc, v3, 0x0, vcc

    // Write to/advance the destination address.
    flat_store_byte         v[4:5], v1
    v_add_u32               v4, vcc, v4, s24
    v_addc_u32              v5, vcc, v5, 0x0, vcc

    // Repeat until branched out.
    s_branch                L_COPY_ALIGNED_PHASE_1_LOOP

  L_COPY_ALIGNED_PHASE_1_DONE:
    // Restore EXEC mask for all lanes.
    s_mov_b64               exec, 0xFFFFFFFFFFFFFFFF

    // ========================================================
    // Phase 2: Unrolled dword[x4] copy up to last whole block.
    // ========================================================

    // Compute unrolled dword[x4] stride across all threads.
    if kCopyAlignedVecWidth == 4
      s_lshl_b32            s25, s24, 0x4
    else
      s_lshl_b32            s25, s24, 0x2
    end

    // Compute phase source address.
    if kCopyAlignedVecWidth == 4
      v_lshlrev_b32         v1, 0x4, v0
    else
      v_lshlrev_b32         v1, 0x2, v0
    end

    v_mov_b32               v3, s9
    v_add_u32               v2, vcc, v1, s8
    v_addc_u32              v3, vcc, v3, 0x0, vcc

    // Compute phase destination address.
    v_mov_b32               v5, s11
    v_add_u32               v4, vcc, v1, s10
    v_addc_u32              v5, vcc, v5, 0x0, vcc

  L_COPY_ALIGNED_PHASE_2_LOOP:
    // Branch out after phase end.
    v_cmp_lt_u64            vcc, v[2:3], s[12:13]
    s_cbranch_vccz          L_COPY_ALIGNED_PHASE_2_DONE

    // Load from/advance the source address.
    for var i = 0; i < kCopyAlignedUnroll; i ++
      if kCopyAlignedVecWidth == 4
        flat_load_dwordx4   v[8 + (i * 4)], v[2:3]
      else
        flat_load_dword     v[8 + i], v[2:3]
      end

      v_add_u32             v2, vcc, v2, s25
      v_addc_u32            v3, vcc, v3, 0x0, vcc
    end

    // Write to/advance the destination address.
    s_waitcnt               vmcnt(0)

    for var i = 0; i < kCopyAlignedUnroll; i ++
      if kCopyAlignedVecWidth == 4
        flat_store_dwordx4  v[4:5], v[8 + (i * 4)]
      else
        flat_store_dword    v[4:5], v[8 + i]
      end

      v_add_u32             v4, vcc, v4, s25
      v_addc_u32            v5, vcc, v5, 0x0, vcc
    end

    // Repeat until branched out.
    s_branch                L_COPY_ALIGNED_PHASE_2_LOOP

  L_COPY_ALIGNED_PHASE_2_DONE:

    // ===========================================
    // Phase 3: Dword copy up to last whole dword.
    // ===========================================

    // Compute dword stride across all threads.
    s_lshl_b32              s25, s24, 0x2

    // Compute phase source address.
    v_lshlrev_b32           v1, 0x2, v0
    v_mov_b32               v3, s13
    v_add_u32               v2, vcc, v1, s12
    v_addc_u32              v3, vcc, v3, 0x0, vcc

    // Compute phase destination address.
    v_mov_b32               v5, s15
    v_add_u32               v4, vcc, v1, s14
    v_addc_u32              v5, vcc, v5, 0x0, vcc

  L_COPY_ALIGNED_PHASE_3_LOOP:
    // Mask off lanes (or branch out) after phase end.
    v_cmp_lt_u64            vcc, v[2:3], s[16:17]
    s_cbranch_vccz          L_COPY_ALIGNED_PHASE_3_DONE
    s_and_b64               exec, exec, vcc

    // Load from/advance the source address.
    flat_load_dword         v1, v[2:3]
    v_add_u32               v2, vcc, v2, s25
    v_addc_u32              v3, vcc, v3, 0x0, vcc
    s_waitcnt               vmcnt(0)

    // Write to/advance the destination address.
    flat_store_dword        v[4:5], v1
    v_add_u32               v4, vcc, v4, s25
    v_addc_u32              v5, vcc, v5, 0x0, vcc

    // Repeat until branched out.
    s_branch                L_COPY_ALIGNED_PHASE_3_LOOP

  L_COPY_ALIGNED_PHASE_3_DONE:
    // Restore EXEC mask for all lanes.
    s_mov_b64               exec, 0xFFFFFFFFFFFFFFFF

    // =============================
    // Phase 4: Byte copy up to end.
    // =============================

    // Compute phase source address.
    v_mov_b32               v3, s17
    v_add_u32               v2, vcc, v0, s16
    v_addc_u32              v3, vcc, v3, 0x0, vcc

    // Compute phase destination address.
    v_mov_b32               v5, s19
    v_add_u32               v4, vcc, v0, s18
    v_addc_u32              v5, vcc, v5, 0x0, vcc

    // Mask off lanes (or branch out) after phase end.
    v_cmp_lt_u64            vcc, v[2:3], s[20:21]
    s_cbranch_vccz          L_COPY_ALIGNED_PHASE_4_DONE
    s_and_b64               exec, exec, vcc

    // Load from the source address.
    flat_load_ubyte         v1, v[2:3]
    s_waitcnt               vmcnt(0)

    // Write to the destination address.
    flat_store_byte         v[4:5], v1

  L_COPY_ALIGNED_PHASE_4_DONE:
    s_endpgm
  end

  // Memory copy for this case:
  //  (src_addr & 0x3) != (dst_addr & 0x3)
  //
  // Kernel argument buffer:
  //   [DW  0, 1]  Phase 1 src start address
  //   [DW  2, 3]  Phase 1 dst start address
  //   [DW  4, 5]  Phase 2 src start address
  //   [DW  6, 7]  Phase 2 dst start address
  //   [DW  8, 9]  Phase 2 src end address
  //   [DW 10,11]  Phase 2 dst end address
  //   [DW 12   ]  Total number of workitems

  var kCopyMisalignedUnroll = 4

  shader CopyMisaligned
    type(CS)
    user_sgpr_count(2)
    sgpr_count(23)
    vgpr_count(6 + kCopyMisalignedUnroll)

    // Retrieve kernel arguments.
    s_load_dwordx4          s[4:7], s[0:1], s_load_dword_offset(0x0)
    s_load_dwordx4          s[8:11], s[0:1], s_load_dword_offset(0x10)
    s_load_dwordx4          s[12:15], s[0:1], s_load_dword_offset(0x20)
    s_load_dword            s16, s[0:1], s_load_dword_offset(0x30)
    s_waitcnt               lgkmcnt(0)

    // Compute workitem id.
    s_lshl_b32              s2, s2, 0x6
    v_add_u32               v0, vcc, s2, v0

    // ===================================================
    // Phase 1: Unrolled byte copy up to last whole block.
    // ===================================================

    // Compute phase source address.
    v_mov_b32               v3, s5
    v_add_u32               v2, vcc, v0, s4
    v_addc_u32              v3, vcc, v3, 0x0, vcc

    // Compute phase destination address.
    v_mov_b32               v5, s7
    v_add_u32               v4, vcc, v0, s6
    v_addc_u32              v5, vcc, v5, 0x0, vcc

  L_COPY_MISALIGNED_PHASE_1_LOOP:
    // Branch out after phase end.
    v_cmp_lt_u64            vcc, v[2:3], s[8:9]
    s_cbranch_vccz          L_COPY_MISALIGNED_PHASE_1_DONE

    // Load from/advance the source address.
    for var i = 0; i < kCopyMisalignedUnroll; i ++
      flat_load_ubyte       v[6 + i], v[2:3]
      v_add_u32             v2, vcc, v2, s16
      v_addc_u32            v3, vcc, v3, 0x0, vcc
    end

    // Write to/advance the destination address.
    s_waitcnt               vmcnt(0)

    for var i = 0; i < kCopyMisalignedUnroll; i ++
      flat_store_byte       v[4:5], v[6 + i]
      v_add_u32             v4, vcc, v4, s16
      v_addc_u32            v5, vcc, v5, 0x0, vcc
    end

    // Repeat until branched out.
    s_branch                L_COPY_MISALIGNED_PHASE_1_LOOP

  L_COPY_MISALIGNED_PHASE_1_DONE:

    // =============================
    // Phase 2: Byte copy up to end.
    // =============================

    // Compute phase source address.
    v_mov_b32               v3, s9
    v_add_u32               v2, vcc, v0, s8
    v_addc_u32              v3, vcc, v3, 0x0, vcc

    // Compute phase destination address.
    v_mov_b32               v5, s11
    v_add_u32               v4, vcc, v0, s10
    v_addc_u32              v5, vcc, v5, 0x0, vcc

  L_COPY_MISALIGNED_PHASE_2_LOOP:
    // Mask off lanes (or branch out) after phase end.
    v_cmp_lt_u64            vcc, v[2:3], s[12:13]
    s_cbranch_vccz          L_COPY_MISALIGNED_PHASE_2_DONE
    s_and_b64               exec, exec, vcc

    // Load from/advance the source address.
    flat_load_ubyte         v1, v[2:3]
    v_add_u32               v2, vcc, v2, s16
    v_addc_u32              v3, vcc, v3, 0x0, vcc
    s_waitcnt               vmcnt(0)

    // Write to/advance the destination address.
    flat_store_byte         v[4:5], v1
    v_add_u32               v4, vcc, v4, s16
    v_addc_u32              v5, vcc, v5, 0x0, vcc

    // Repeat until branched out.
    s_branch                L_COPY_MISALIGNED_PHASE_2_LOOP

  L_COPY_MISALIGNED_PHASE_2_DONE:
    s_endpgm
  end

  // Memory fill for dword-aligned region.
  //
  // Kernel argument buffer:
  //   [DW  0, 1]  Phase 1 dst start address
  //   [DW  2, 3]  Phase 2 dst start address
  //   [DW  4, 5]  Phase 2 dst end address
  //   [DW  6   ]  Value to fill memory with
  //   [DW  7   ]  Total number of workitems

  var kFillVecWidth = 4
  var kFillUnroll = 1

  shader Fill
    type(CS)
    user_sgpr_count(2)
    sgpr_count(19)
    vgpr_count(8)

    // Retrieve kernel arguments.
    s_load_dwordx4          s[4:7], s[0:1], s_load_dword_offset(0x0)
    s_load_dwordx4          s[8:11], s[0:1], s_load_dword_offset(0x10)
    s_waitcnt               lgkmcnt(0)

    // Compute workitem id.
    s_lshl_b32              s2, s2, 0x6
    v_add_u32               v0, vcc, s2, v0

    // Copy fill pattern into VGPRs.
    for var i = 0; i < kFillVecWidth; i ++
      v_mov_b32           v[4 + i], s10
    end

    // ========================================================
    // Phase 1: Unrolled dword[x4] fill up to last whole block.
    // ========================================================

    // Compute unrolled dword[x4] stride across all threads.
    if kFillVecWidth == 4
      s_lshl_b32            s12, s11, 0x4
    else
      s_lshl_b32            s12, s11, 0x2
    end

    // Compute phase destination address.
    if kFillVecWidth == 4
      v_lshlrev_b32         v1, 0x4, v0
    else
      v_lshlrev_b32         v1, 0x2, v0
    end

    v_mov_b32               v3, s5
    v_add_u32               v2, vcc, v1, s4
    v_addc_u32              v3, vcc, v3, 0x0, vcc

  L_FILL_PHASE_1_LOOP:
    // Branch out after phase end.
    v_cmp_lt_u64            vcc, v[2:3], s[6:7]
    s_cbranch_vccz          L_FILL_PHASE_1_DONE

    // Write to/advance the destination address.
    for var i = 0; i < kFillUnroll; i ++
      if kFillVecWidth == 4
        flat_store_dwordx4  v[2:3], v[4:7]
      else
        flat_store_dword    v[2:3], v4
      end

      v_add_u32             v2, vcc, v2, s12
      v_addc_u32            v3, vcc, v3, 0x0, vcc
    end

    // Repeat until branched out.
    s_branch                L_FILL_PHASE_1_LOOP

  L_FILL_PHASE_1_DONE:

    // ==============================
    // Phase 2: Dword fill up to end.
    // ==============================

    // Compute dword stride across all threads.
    s_lshl_b32              s12, s11, 0x2

    // Compute phase destination address.
    v_lshlrev_b32           v1, 0x2, v0
    v_mov_b32               v3, s7
    v_add_u32               v2, vcc, v1, s6
    v_addc_u32              v3, vcc, v3, 0x0, vcc

  L_FILL_PHASE_2_LOOP:
    // Mask off lanes (or branch out) after phase end.
    v_cmp_lt_u64            vcc, v[2:3], s[8:9]
    s_cbranch_vccz          L_FILL_PHASE_2_DONE
    s_and_b64               exec, exec, vcc

    // Write to/advance the destination address.
    flat_store_dword        v[2:3], v4
    v_add_u32               v2, vcc, v2, s12
    v_addc_u32              v3, vcc, v3, 0x0, vcc

    // Repeat until branched out.
    s_branch                L_FILL_PHASE_2_LOOP

  L_FILL_PHASE_2_DONE:
    s_endpgm
  end
)");
  return kBlitKernelSource_;
}

// Search kernel source for variable definition and return value.
int GetKernelSourceParam(const char* paramName) {
  std::stringstream paramDef;
  paramDef << "var " << paramName << " = ";

  std::string::size_type paramDefLoc =
                              kBlitKernelSource().find(paramDef.str());
  assert(paramDefLoc != std::string::npos);
  std::string::size_type paramValLoc = paramDefLoc + paramDef.str().size();
  std::string::size_type paramEndLoc =
      kBlitKernelSource().find('\n', paramDefLoc);
  assert(paramDefLoc != std::string::npos);

  std::string paramVal(&kBlitKernelSource()[paramValLoc],
                       &kBlitKernelSource()[paramEndLoc]);
  return std::stoi(paramVal);
}


#define DEFINE_KERNEL_PARAM_FUNC(name) \
static int& name() { \
    static std::once_flag initFlag; \
    static int val; \
    std::call_once(initFlag, [&]() { \
        val = GetKernelSourceParam(#name); \
    }); \
    return val; \
}

// Use the macro to define the functions
DEFINE_KERNEL_PARAM_FUNC(kCopyAlignedVecWidth)
DEFINE_KERNEL_PARAM_FUNC(kCopyAlignedUnroll)
DEFINE_KERNEL_PARAM_FUNC(kCopyMisalignedUnroll)
DEFINE_KERNEL_PARAM_FUNC(kFillVecWidth)
DEFINE_KERNEL_PARAM_FUNC(kFillUnroll)

static unsigned extractAqlBits(unsigned v, unsigned pos, unsigned width) {
  return (v >> pos) & ((1 << width) - 1);
};

BlitKernel::BlitKernel(core::Queue* queue)
    : core::Blit(),
      queue_(queue),
      kernarg_async_(NULL),
      kernarg_async_mask_(0),
      kernarg_async_counter_(0),
      bytes_queued_(0),
      last_queued_(0),
      pending_search_index_(0),
      num_cus_(0) {
  completion_signal_.handle = 0;
}

BlitKernel::~BlitKernel() {}

hsa_status_t BlitKernel::Initialize(const core::Agent& agent) {
  queue_bitmask_ = queue_->public_handle()->size - 1;

  bytes_written_.resize(queue_->public_handle()->size);
  memset(&bytes_written_[0], -1, bytes_written_.size() * sizeof(BytesWritten));

  hsa_status_t status = HSA::hsa_signal_create(1, 0, NULL, &completion_signal_);
  if (HSA_STATUS_SUCCESS != status) {
    return status;
  }

  const AMD::GpuAgent& gpuAgent = static_cast<const AMD::GpuAgent&>(agent);
  kernarg_async_ = reinterpret_cast<KernelArgs*>(
      gpuAgent.system_allocator()(queue_->public_handle()->size * AlignUp(sizeof(KernelArgs), 16),
                                  16, core::MemoryRegion::AllocateNoFlags));

  kernarg_async_mask_ = queue_->public_handle()->size - 1;

  // Obtain the number of compute units in the underlying agent.
  num_cus_ = gpuAgent.properties().NumFComputeCores / 4;

  // Assemble shaders to AQL code objects.
  std::map<KernelType, const char*> kernel_names = {
      {KernelType::CopyAligned, "CopyAligned"},
      {KernelType::CopyMisaligned, "CopyMisaligned"},
      {KernelType::Fill, "Fill"}};

  for (auto kernel_name : kernel_names) {
    KernelCode& kernel = kernels_[kernel_name.first];
    gpuAgent.AssembleShader(kernel_name.second, AMD::GpuAgent::AssembleTarget::AQL, kernel.code_buf_,
                            kernel.code_buf_size_);
  }

  if (agent.profiling_enabled()) {
    return EnableProfiling(true);
  }

  return HSA_STATUS_SUCCESS;
}

hsa_status_t BlitKernel::Destroy(const core::Agent& agent) {
  std::lock_guard<std::mutex> guard(lock_);

  const AMD::GpuAgent& gpuAgent = static_cast<const AMD::GpuAgent&>(agent);

  for (auto kernel_pair : kernels_) {
    gpuAgent.ReleaseShader(kernel_pair.second.code_buf_,
                           kernel_pair.second.code_buf_size_);
  }

  if (kernarg_async_ != NULL) {
    gpuAgent.system_deallocator()(kernarg_async_);
  }

  if (completion_signal_.handle != 0) {
    HSA::hsa_signal_destroy(completion_signal_);
  }

  return HSA_STATUS_SUCCESS;
}

hsa_status_t BlitKernel::SubmitLinearCopyCommand(void* dst, const void* src,
                                                 size_t size) {
  // Protect completion_signal_.
  std::lock_guard<std::mutex> guard(lock_);

  HSA::hsa_signal_store_relaxed(completion_signal_, 1);

  std::vector<core::Signal*> dep_signals(0);
  std::vector<core::Signal*> gang_signals(0);

  hsa_status_t stat = SubmitLinearCopyCommand(
      dst, src, size, dep_signals, *core::Signal::Convert(completion_signal_), gang_signals);

  if (stat != HSA_STATUS_SUCCESS) {
    return stat;
  }

  // Wait for the packet to finish.
  if (HSA::hsa_signal_wait_scacquire(completion_signal_, HSA_SIGNAL_CONDITION_LT, 1, uint64_t(-1),
                                     HSA_WAIT_STATE_ACTIVE) != 0) {
    // Signal wait returned unexpected value.
    return HSA_STATUS_ERROR;
  }

  return HSA_STATUS_SUCCESS;
}

hsa_status_t BlitKernel::SubmitLinearCopyCommand(
    void* dst, const void* src, size_t size,
    std::vector<core::Signal*>& dep_signals, core::Signal& out_signal,
    std::vector<core::Signal*>& gang_signals) {
  // Reserve write index for barrier(s) + dispatch packet.
  const uint32_t num_barrier_packet = uint32_t((dep_signals.size() + 4) / 5);
  const uint32_t total_num_packet = num_barrier_packet + 1;

  uint64_t write_index;
  {
    std::lock_guard<std::mutex> lock(reservation_lock_);
    write_index = AcquireWriteIndex(total_num_packet);
    RecordBlitHistory(size, write_index + total_num_packet - 1);
  }

  uint64_t write_index_temp = write_index;

  // Insert barrier packets to handle dependent signals.
  // Barrier bit keeps signal checking traffic from competing with a copy.
  const uint16_t kBarrierPacketHeader = (HSA_PACKET_TYPE_BARRIER_AND << HSA_PACKET_HEADER_TYPE) |
      (HSA_FENCE_SCOPE_NONE << HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE) |
      (HSA_FENCE_SCOPE_NONE << HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE);

  hsa_barrier_and_packet_t barrier_packet = {0};
  barrier_packet.header = HSA_PACKET_TYPE_INVALID;

  hsa_barrier_and_packet_t* queue_buffer =
      reinterpret_cast<hsa_barrier_and_packet_t*>(
          queue_->public_handle()->base_address);

  const size_t dep_signal_count = dep_signals.size();
  for (size_t i = 0; i < dep_signal_count; ++i) {
    const size_t idx = i % 5;
    barrier_packet.dep_signal[idx] = core::Signal::Convert(dep_signals[i]);
    if (i == (dep_signal_count - 1) || idx == 4) {
      std::atomic_thread_fence(std::memory_order_acquire);
      queue_buffer[(write_index)&queue_bitmask_] = barrier_packet;
      std::atomic_thread_fence(std::memory_order_release);
      queue_buffer[(write_index)&queue_bitmask_].header = kBarrierPacketHeader;

      LogPrint(HSA_AMD_LOG_FLAG_BLIT_KERNEL_PKTS,
      "HWq=%p, id=%d, Barrier Header = "
      "0x%x (type=%d, barrier=%d, acquire=%d, release=%d), "
      "dep_signal=[0x%zx 0x%zx 0x%zx 0x%zx 0x%zx], completion_signal=0x%zx "
      "rptr=%u, wptr=%u",
      queue_->public_handle()->base_address, queue_->public_handle()->id,
      kBarrierPacketHeader,
      extractAqlBits(kBarrierPacketHeader,
                    HSA_PACKET_HEADER_TYPE, HSA_PACKET_HEADER_WIDTH_TYPE),
      extractAqlBits(kBarrierPacketHeader,
                    HSA_PACKET_HEADER_BARRIER, HSA_PACKET_HEADER_WIDTH_BARRIER),
      extractAqlBits(kBarrierPacketHeader, HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE,
                    HSA_PACKET_HEADER_WIDTH_SCACQUIRE_FENCE_SCOPE),
      extractAqlBits(kBarrierPacketHeader, HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE,
                    HSA_PACKET_HEADER_WIDTH_SCRELEASE_FENCE_SCOPE),
      barrier_packet.dep_signal[0], barrier_packet.dep_signal[1], barrier_packet.dep_signal[2],
      barrier_packet.dep_signal[3], barrier_packet.dep_signal[4],
      barrier_packet.completion_signal, queue_->LoadReadIndexRelaxed(), write_index);

      ++write_index;

      memset(&barrier_packet, 0, sizeof(hsa_barrier_and_packet_t));
      barrier_packet.header = HSA_PACKET_TYPE_INVALID;
    }
  }

  // Insert dispatch packet for copy kernel.
  KernelArgs* args = ObtainAsyncKernelCopyArg();
  KernelCode* kernel_code = nullptr;
  int num_workitems = 0;

  bool aligned = ((uintptr_t(src) & 0x3) == (uintptr_t(dst) & 0x3));

  if (aligned) {
    // Use dword-based aligned kernel.
    kernel_code = &kernels_[KernelType::CopyAligned];

    // Compute the size of each copy phase.
    num_workitems = 64 * 4 * num_cus_;

    // Phase 1 (byte copy) ends when destination is 0x100-aligned.
    uintptr_t src_start = uintptr_t(src);
    uintptr_t dst_start = uintptr_t(dst);
    uint64_t phase1_size =
        std::min(size, uint64_t(0x100 - (dst_start & 0xFF)) & 0xFF);

    // Phase 2 (unrolled dwordx4 copy) ends when last whole block fits.
    uint64_t phase2_block = num_workitems * sizeof(uint32_t) *
                            kCopyAlignedUnroll() * kCopyAlignedVecWidth();
    uint64_t phase2_size = ((size - phase1_size) / phase2_block) * phase2_block;

    // Phase 3 (dword copy) ends when last whole dword fits.
    uint64_t phase3_size =
        ((size - phase1_size - phase2_size) / sizeof(uint32_t)) *
        sizeof(uint32_t);

    args->copy_aligned.phase1_src_start = src_start;
    args->copy_aligned.phase1_dst_start = dst_start;
    args->copy_aligned.phase2_src_start = src_start + phase1_size;
    args->copy_aligned.phase2_dst_start = dst_start + phase1_size;
    args->copy_aligned.phase3_src_start = src_start + phase1_size + phase2_size;
    args->copy_aligned.phase3_dst_start = dst_start + phase1_size + phase2_size;
    args->copy_aligned.phase4_src_start =
        src_start + phase1_size + phase2_size + phase3_size;
    args->copy_aligned.phase4_dst_start =
        dst_start + phase1_size + phase2_size + phase3_size;
    args->copy_aligned.phase4_src_end = src_start + size;
    args->copy_aligned.phase4_dst_end = dst_start + size;
    args->copy_aligned.num_workitems = num_workitems;
  } else {
    // Use byte-based misaligned kernel.
    kernel_code = &kernels_[KernelType::CopyMisaligned];

    // Compute the size of each copy phase.
    num_workitems = 64 * 4 * num_cus_;

    // Phase 1 (unrolled byte copy) ends when last whole block fits.
    uintptr_t src_start = uintptr_t(src);
    uintptr_t dst_start = uintptr_t(dst);
    uint64_t phase1_block =
        num_workitems * sizeof(uint8_t) * kCopyMisalignedUnroll();
    uint64_t phase1_size = (size / phase1_block) * phase1_block;

    args->copy_misaligned.phase1_src_start = src_start;
    args->copy_misaligned.phase1_dst_start = dst_start;
    args->copy_misaligned.phase2_src_start = src_start + phase1_size;
    args->copy_misaligned.phase2_dst_start = dst_start + phase1_size;
    args->copy_misaligned.phase2_src_end = src_start + size;
    args->copy_misaligned.phase2_dst_end = dst_start + size;
    args->copy_misaligned.num_workitems = num_workitems;
  }

  hsa_signal_t signal = {(core::Signal::Convert(&out_signal)).handle};
  PopulateQueue(write_index, uintptr_t(kernel_code->code_buf_), args,
                num_workitems, signal);

  // Submit barrier(s) and dispatch packets.
  ReleaseWriteIndex(write_index_temp, total_num_packet);

  return HSA_STATUS_SUCCESS;
}

hsa_status_t BlitKernel::SubmitLinearFillCommand(void* ptr, uint32_t value,
                                                 size_t count) {
  std::lock_guard<std::mutex> guard(lock_);

  // Reject misaligned base address.
  if ((uintptr_t(ptr) & 0x3) != 0) {
    return HSA_STATUS_ERROR;
  }

  // Compute the size of each fill phase.
  int num_workitems = 64 * num_cus_;

  // Phase 1 (unrolled dwordx4 copy) ends when last whole block fits.
  uintptr_t dst_start = uintptr_t(ptr);
  uint64_t fill_size = count * sizeof(uint32_t);

  uint64_t phase1_block =
      num_workitems * sizeof(uint32_t) * kFillUnroll() * kFillVecWidth();
  uint64_t phase1_size = (fill_size / phase1_block) * phase1_block;

  KernelArgs* args = ObtainAsyncKernelCopyArg();
  args->fill.phase1_dst_start = dst_start;
  args->fill.phase2_dst_start = dst_start + phase1_size;
  args->fill.phase2_dst_end = dst_start + fill_size;
  args->fill.fill_value = value;
  args->fill.num_workitems = num_workitems;

  // Submit dispatch packet.
  HSA::hsa_signal_store_relaxed(completion_signal_, 1);

  uint64_t write_index;
  {
    std::lock_guard<std::mutex> lock(reservation_lock_);
    write_index = AcquireWriteIndex(1);
    RecordBlitHistory(fill_size, write_index);
  }

  PopulateQueue(write_index, uintptr_t(kernels_[KernelType::Fill].code_buf_),
                args, num_workitems, completion_signal_);

  ReleaseWriteIndex(write_index, 1);

  // Wait for the packet to finish.
  if (HSA::hsa_signal_wait_scacquire(completion_signal_, HSA_SIGNAL_CONDITION_LT, 1, uint64_t(-1),
                                     HSA_WAIT_STATE_ACTIVE) != 0) {
    // Signal wait returned unexpected value.
    return HSA_STATUS_ERROR;
  }

  return HSA_STATUS_SUCCESS;
}

hsa_status_t BlitKernel::EnableProfiling(bool enable) {
  queue_->SetProfiling(enable);
  return HSA_STATUS_SUCCESS;
}

uint64_t BlitKernel::AcquireWriteIndex(uint32_t num_packet) {
  assert(queue_->public_handle()->size >= num_packet);

  uint64_t write_index = queue_->AddWriteIndexAcqRel(num_packet);

  while (write_index + num_packet - queue_->LoadReadIndexRelaxed() > queue_->public_handle()->size) {
    os::YieldThread();
  }

  return write_index;
}

void BlitKernel::ReleaseWriteIndex(uint64_t write_index, uint32_t num_packet) {
  // Update doorbel register with last packet id.
  core::Signal* doorbell =
      core::Signal::Convert(queue_->public_handle()->doorbell_signal);
  doorbell->StoreRelease(write_index + num_packet - 1);
}

void BlitKernel::PopulateQueue(uint64_t index, uint64_t code_handle, void* args,
                               uint32_t grid_size_x,
                               hsa_signal_t completion_signal) {
  assert(IsMultipleOf(args, 16));

  hsa_kernel_dispatch_packet_t packet = {0};

  static const uint16_t kDispatchPacketHeader =
      (HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE) |
      (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE) |
      (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE);

  packet.header = kInvalidPacketHeader;
  packet.kernel_object = code_handle;
  packet.kernarg_address = args;

  // Setup working size.
  const int kNumDimension = 1;
  packet.setup = kNumDimension << HSA_KERNEL_DISPATCH_PACKET_SETUP_DIMENSIONS;
  packet.grid_size_x = AlignUp(static_cast<uint32_t>(grid_size_x), 64);
  packet.grid_size_y = packet.grid_size_z = 1;
  packet.workgroup_size_x = 64;
  packet.workgroup_size_y = packet.workgroup_size_z = 1;

  packet.completion_signal = completion_signal;

  // Populate queue buffer with AQL packet.
  hsa_kernel_dispatch_packet_t* queue_buffer =
      reinterpret_cast<hsa_kernel_dispatch_packet_t*>(
          queue_->public_handle()->base_address);
  std::atomic_thread_fence(std::memory_order_acquire);
  queue_buffer[index & queue_bitmask_] = packet;
  std::atomic_thread_fence(std::memory_order_release);
  if (queue_->IsDeviceMemRingBuf() && queue_->needsPcieOrdering()) {
    // Ensure the packet body is written as header may get reordered when writing over PCIE
    _mm_sfence();
  }
  __atomic_store_n(&(queue_buffer[index & queue_bitmask_].full_header),
                    kDispatchPacketHeader | packet.setup << 16, __ATOMIC_RELEASE);

  LogPrint(HSA_AMD_LOG_FLAG_BLIT_KERNEL_PKTS,
    "HWq=%p, id=%d, Dispatch Header = "
    "0x%x (type=%d, barrier=%d, acquire=%d, release=%d), "
    "setup=%d, grid=[%zu, %zu, %zu], workgroup=[%zu, %zu, %zu], private_seg_size=%zu, "
    "group_seg_size=%zu, kernel_obj=0x%zx, kernarg_address=0x%zx, completion_signal=0x%zx "
    "rptr=%u, wptr=%u",
    queue_->public_handle()->base_address, queue_->public_handle()->id,
    kDispatchPacketHeader,
    extractAqlBits(kDispatchPacketHeader,
                   HSA_PACKET_HEADER_TYPE, HSA_PACKET_HEADER_WIDTH_TYPE),
    extractAqlBits(kDispatchPacketHeader,
                   HSA_PACKET_HEADER_BARRIER, HSA_PACKET_HEADER_WIDTH_BARRIER),
    extractAqlBits(kDispatchPacketHeader, HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE,
                   HSA_PACKET_HEADER_WIDTH_SCACQUIRE_FENCE_SCOPE),
    extractAqlBits(kDispatchPacketHeader, HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE,
                   HSA_PACKET_HEADER_WIDTH_SCRELEASE_FENCE_SCOPE),
    packet.setup, packet.grid_size_x, packet.grid_size_y, packet.grid_size_z,
    packet.workgroup_size_x, packet.workgroup_size_y, packet.workgroup_size_z,
    packet.private_segment_size, packet.group_segment_size,
    packet.kernel_object,packet.kernarg_address,
    completion_signal, queue_->LoadReadIndexRelaxed(), index);
}

BlitKernel::KernelArgs* BlitKernel::ObtainAsyncKernelCopyArg() {
  const uint32_t index =
      atomic::Add(&kernarg_async_counter_, 1U, std::memory_order_acquire) & kernarg_async_mask_;

  KernelArgs* arg = &kernarg_async_[index];
  assert(IsMultipleOf(arg, 16));
  return arg;
}

void BlitKernel::RecordBlitHistory(uint64_t size, uint64_t index) {
  uint64_t queued = bytes_queued_;
  bytes_queued_ += size;
  bytes_written_[index & queue_bitmask_].bytes = queued;
  bytes_written_[index & queue_bitmask_].index = index;
  last_queued_ = index;
}

uint64_t BlitKernel::PendingBytes() {
  uint64_t read = queue_->LoadReadIndexRelaxed();
  uint64_t index = pending_search_index_.load();
  uint64_t last = last_queued_;
  // If the last blit command has been run then the blit is empty.
  if (read > last) return 0;

  index = Max(index, read);
  while (index <= last) {
    // Ensure any record we use was not wrapped.
    if (index == bytes_written_[index & queue_bitmask_].index) {
      uint64_t ret = bytes_queued_ - bytes_written_[index & queue_bitmask_].bytes;

      // Store max search index.
      uint64_t old = pending_search_index_.load();
      while (old < index) {
        if (pending_search_index_.compare_exchange_strong(old, index)) break;
      }

      return ret;
    }
    index++;
  }
  debug_warning(false && "Race between PendingBytes and blit submission detected.");
  // Zero is a valid return in this case since the command which was last when the search started is
  // now complete.
  return 0;
}

}  // namespace amd
}  // namespace rocr
