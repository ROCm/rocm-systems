////////////////////////////////////////////////////////////////////////////////
//
// The University of Illinois/NCSA
// Open Source License (NCSA)
//
// Copyright (c) 2014-2025, Advanced Micro Devices, Inc. All rights reserved.
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
#include <cstring>
#include <sstream>
#include <string>

#include "core/inc/amd_gpu_agent.h"
#include "core/inc/hsa_internal.h"
#include "core/util/utils.h"

namespace rocr {
namespace AMD {

static constexpr const char kBlitKernelSource_[] = R"(
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

  // ====================================================================
  // Broadcast Copy Kernel: Single source to multiple destinations
  // Strategy: Each workgroup handles one destination
  // ====================================================================
  //
  // Kernel argument buffer:
  //   [DW  0, 1]  Source buffer address (64-bit)
  //   [DW  2, 3]  Destination list address (GPU array of uint64_t[])
  //   [DW  4   ]  Number of destinations (1-1024)
  //   [DW  5   ]  Copy size (bytes to copy to each destination)
  //   [DW  6   ]  Total number of workitems
  //
  // Note: dst_list must be a GPU-accessible buffer containing destination addresses.
  // Each workgroup copies to one destination using all threads in parallel.

  var kBroadcastCopyVecWidth = 4
  var kBroadcastCopyUnroll = 4

  shader BroadcastCopy
    type(CS)
    user_sgpr_count(2)
    sgpr_count(24)
    vgpr_count(16)

    // Load kernel arguments
    s_load_dwordx4          s[4:7], s[0:1], s_load_dword_offset(0x0)
    s_load_dwordx2          s[8:9], s[0:1], s_load_dword_offset(0x10)
    s_load_dword            s10, s[0:1], s_load_dword_offset(0x18)
    s_waitcnt               lgkmcnt(0)

    // Save local work-item ID (v0 at kernel entry contains local_id)
    v_mov_b32               v14, v0

    // Compute destination index from workgroup ID
    v_mov_b32               v0, s2

    // Load destination address from dst_list[workgroup_id]
    v_lshlrev_b32           v1, 3, v0
    v_mov_b32               v3, s7
    v_add_u32               v2, vcc, v1, s6
    v_addc_u32              v3, vcc, v3, 0, vcc

    flat_load_dwordx2       v[4:5], v[2:3]
    s_waitcnt               vmcnt(0)

    // Compute global thread offset: local_id + workgroup_id * 64
    s_lshl_b32              s11, s2, 6
    v_add_u32               v0, vcc, v14, s11

    // Main Loop: 16-byte vectorized copy
    s_lshl_b32              s12, s10, 4

    v_lshlrev_b32           v1, 4, v0
    v_mov_b32               v7, s5
    v_add_u32               v6, vcc, v1, s4
    v_addc_u32              v7, vcc, v7, 0, vcc

    v_add_u32               v8, vcc, v1, v4
    v_addc_u32              v9, vcc, v5, 0, vcc

    v_and_b32               v10, 0xFFFFFFF0, s9
    v_mov_b32               v11, s5
    v_add_u32               v10, vcc, v10, s4
    v_addc_u32              v11, vcc, v11, 0, vcc

  L_BROADCAST_VEC_LOOP:
    v_cmp_lt_u64            vcc, v[6:7], v[10:11]
    s_cbranch_vccz          L_BROADCAST_VEC_DONE

    for var i = 0; i < kBroadcastCopyUnroll; i ++
      flat_load_dwordx4     v[12:15], v[6:7]
      s_waitcnt             vmcnt(0)
      flat_store_dwordx4    v[8:9], v[12:15]

      v_add_u32             v6, vcc, v6, s12
      v_addc_u32            v7, vcc, v7, 0, vcc
      v_add_u32             v8, vcc, v8, s12
      v_addc_u32            v9, vcc, v9, 0, vcc
    end

    s_branch                L_BROADCAST_VEC_LOOP

  L_BROADCAST_VEC_DONE:

    // Tail Loop: Byte-by-byte copy
    v_mov_b32               v7, s5
    v_add_u32               v6, vcc, v0, s4
    v_addc_u32              v7, vcc, v7, 0, vcc

    v_add_u32               v8, vcc, v0, v4
    v_addc_u32              v9, vcc, v5, 0, vcc

    v_mov_b32               v11, s5
    v_add_u32               v10, vcc, s9, s4
    v_addc_u32              v11, vcc, v11, 0, vcc

  L_BROADCAST_BYTE_LOOP:
    v_cmp_lt_u64            vcc, v[6:7], v[10:11]
    s_cbranch_vccz          L_BROADCAST_BYTE_DONE
    s_and_b64               exec, exec, vcc

    flat_load_ubyte         v1, v[6:7]
    s_waitcnt               vmcnt(0)
    flat_store_byte         v[8:9], v1

    v_add_u32               v6, vcc, v6, s10
    v_addc_u32              v7, vcc, v7, 0, vcc
    v_add_u32               v8, vcc, v8, s10
    v_addc_u32              v9, vcc, v9, 0, vcc

    s_branch                L_BROADCAST_BYTE_LOOP

  L_BROADCAST_BYTE_DONE:
    s_endpgm
  end
)";

// Search kernel source for variable definition and return value.
int GetKernelSourceParam(const char* paramName) {
  std::string paramDef = std::string("var ") + paramName + " = ";

  const char* paramDefPtr = strstr(kBlitKernelSource_, paramDef.c_str());
  assert(paramDefPtr != nullptr);

  const char* paramValPtr = paramDefPtr + paramDef.size();
  const char* paramEndPtr = strchr(paramValPtr, '\n');
  assert(paramEndPtr != nullptr);

  std::string paramVal(paramValPtr, paramEndPtr);
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
DEFINE_KERNEL_PARAM_FUNC(kBroadcastCopyVecWidth)
DEFINE_KERNEL_PARAM_FUNC(kBroadcastCopyUnroll)

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
  agent_ = &agent;
  queue_bitmask_ = queue_->public_handle()->size - 1;

  bytes_written_.resize(queue_->public_handle()->size);
  memset(&bytes_written_[0], -1, bytes_written_.size() * sizeof(BytesWritten));

  hsa_status_t status = HSA::hsa_signal_create(1, 0, NULL, &completion_signal_);
  if (HSA_STATUS_SUCCESS != status) {
    return status;
  }

  const AMD::GpuAgent* gpuAgent = static_cast<const AMD::GpuAgent*>(agent_);
  kernarg_async_ = reinterpret_cast<KernelArgs*>(
      gpuAgent->system_allocator()(queue_->public_handle()->size * AlignUp(sizeof(KernelArgs), 16),
                                  16, core::MemoryRegion::AllocateNoFlags));

  kernarg_async_mask_ = queue_->public_handle()->size - 1;

  // Obtain the number of compute units in the underlying agent.
  num_cus_ = gpuAgent->properties().NumFComputeCores / 4;

  // Assemble shaders to AQL code objects.
  std::map<KernelType, const char*> kernel_names = {{KernelType::CopyAligned, "CopyAligned"},
                                                    {KernelType::CopyMisaligned, "CopyMisaligned"},
                                                    {KernelType::Fill, "Fill"},
                                                    {KernelType::BroadcastCopy, "BroadcastCopy"},
                                                    {KernelType::SwapCopy, "SwapCopy"},
                                                    {KernelType::IndirectCopy, "IndirectCopy"}};

  for (auto kernel_name : kernel_names) {
    KernelCode& kernel = kernels_[kernel_name.first];
    gpuAgent->AssembleShader(kernel_name.second, AMD::GpuAgent::AssembleTarget::AQL, kernel.code_buf_,
                            kernel.code_buf_size_);
  }

  if (agent_->profiling_enabled()) {
    return EnableProfiling(true);
  }

  return HSA_STATUS_SUCCESS;
}

hsa_status_t BlitKernel::Destroy() {
  std::lock_guard<std::mutex> guard(lock_);

  const AMD::GpuAgent* gpuAgent = static_cast<const AMD::GpuAgent*>(agent_);

  for (auto kernel_pair : kernels_) {
    gpuAgent->ReleaseShader(kernel_pair.second.code_buf_,
                           kernel_pair.second.code_buf_size_);
  }

  if (kernarg_async_ != NULL) {
    gpuAgent->system_deallocator()(kernarg_async_);
  }

  if (completion_signal_.handle != 0) {
    core::Signal* signal = core::Signal::Convert(completion_signal_);
    signal->DestroySignal();
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

  if(agent_->profiling_enabled()) {
    LogSignalDuration(HSA_AMD_LOG_FLAG_BLIT_KERNEL_PKTS, completion_signal_,
                      "BlitKernel::SubmitLinearCopyCommand");
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

      LogPrint(HSA_AMD_LOG_FLAG_AQL,
      "HWq=%p, id=%lu, Barrier Header = "
      "0x%x (type=%d, barrier=%d, acquire=%d, release=%d), "
      "dep_signal=[0x%zx 0x%zx 0x%zx 0x%zx 0x%zx], completion_signal=0x%zx "
      "rptr=%lu, wptr=%lu",
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
      barrier_packet.dep_signal[0].handle,
      barrier_packet.dep_signal[1].handle,
      barrier_packet.dep_signal[2].handle,
      barrier_packet.dep_signal[3].handle,
      barrier_packet.dep_signal[4].handle,
      barrier_packet.completion_signal.handle,
      queue_->LoadReadIndexRelaxed(), write_index);

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

hsa_status_t BlitKernel::SubmitBroadcastCopyCommand(const void* src, void* const* dst_list,
                                                    uint32_t num_destinations, size_t size,
                                                    std::vector<core::Signal*>& dep_signals,
                                                    core::Signal& out_signal,
                                                    std::vector<core::Signal*>& gang_signals) {
  if (src == nullptr || dst_list == nullptr || num_destinations == 0 || num_destinations > 1024) {
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }

  for (uint32_t i = 0; i < num_destinations; ++i) {
    if (dst_list[i] == nullptr) {
      return HSA_STATUS_ERROR_INVALID_ARGUMENT;
    }
  }

  if (size == 0) {
    return HSA_STATUS_SUCCESS;
  }

  // copy_size is uint32_t in kernel args, so limit to 4GB per destination
  if (size > UINT32_MAX) {
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }

  // Allocate GPU-accessible memory for destination address list
  void* dst_list_gpu = nullptr;
  const size_t dst_list_size = num_destinations * sizeof(uint64_t);

  const AMD::GpuAgent* gpuAgent = static_cast<const AMD::GpuAgent*>(agent_);
  dst_list_gpu =
      gpuAgent->system_allocator()(dst_list_size, 16, core::MemoryRegion::AllocateNoFlags);

  if (dst_list_gpu == nullptr) {
    return HSA_STATUS_ERROR_OUT_OF_RESOURCES;
  }

  // Copy destination addresses to GPU buffer
  uint64_t* dst_addrs_gpu = reinterpret_cast<uint64_t*>(dst_list_gpu);
  for (uint32_t i = 0; i < num_destinations; ++i) {
    dst_addrs_gpu[i] = reinterpret_cast<uint64_t>(dst_list[i]);
  }

  std::atomic_thread_fence(std::memory_order_release);

  const uint32_t num_barrier_packet = uint32_t((dep_signals.size() + 4) / 5);
  const uint32_t total_num_packet = num_barrier_packet + 1;

  uint64_t write_index;
  {
    std::lock_guard<std::mutex> lock(reservation_lock_);
    write_index = AcquireWriteIndex(total_num_packet);
    RecordBlitHistory(size * num_destinations, write_index + total_num_packet - 1);
  }

  uint64_t write_index_temp = write_index;

  const uint16_t kBarrierPacketHeader = (HSA_PACKET_TYPE_BARRIER_AND << HSA_PACKET_HEADER_TYPE) |
      (HSA_FENCE_SCOPE_NONE << HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE) |
      (HSA_FENCE_SCOPE_NONE << HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE);

  hsa_barrier_and_packet_t barrier_packet = {0};
  barrier_packet.header = HSA_PACKET_TYPE_INVALID;

  hsa_barrier_and_packet_t* queue_buffer =
      reinterpret_cast<hsa_barrier_and_packet_t*>(queue_->public_handle()->base_address);

  const size_t dep_signal_count = dep_signals.size();
  for (size_t i = 0; i < dep_signal_count; ++i) {
    const size_t idx = i % 5;
    barrier_packet.dep_signal[idx] = core::Signal::Convert(dep_signals[i]);
    if (i == (dep_signal_count - 1) || idx == 4) {
      std::atomic_thread_fence(std::memory_order_acquire);
      queue_buffer[(write_index_temp)&queue_bitmask_] = barrier_packet;
      std::atomic_thread_fence(std::memory_order_release);
      queue_buffer[(write_index_temp)&queue_bitmask_].header = kBarrierPacketHeader;
      write_index_temp++;
      barrier_packet = {0};
      barrier_packet.header = HSA_PACKET_TYPE_INVALID;
    }
  }

  // Calculate dispatch parameters: 1 workgroup per destination, 64 threads per workgroup
  const uint32_t threads_per_workgroup = 64;
  const uint32_t num_workgroups = num_destinations;
  const uint32_t total_workitems = threads_per_workgroup * num_workgroups;

  KernelArgs* args = ObtainAsyncKernelCopyArg();
  args->broadcast.src_addr = reinterpret_cast<uint64_t>(src);
  args->broadcast.dst_list_addr = reinterpret_cast<uint64_t>(dst_list_gpu);
  args->broadcast.num_destinations = num_destinations;
  args->broadcast.copy_size = static_cast<uint32_t>(size);
  args->broadcast.num_workitems = total_workitems;

  const uint64_t dispatch_index = write_index + num_barrier_packet;

  // Register async cleanup for dst_list_gpu BEFORE dispatch.
  // Handler fires when signal reaches 0 (after kernel completion).
  // Registering before dispatch allows us to fail fast if registration fails.
  bool handler_registered = core::Runtime::runtime_singleton_->SetAsyncSignalHandler(
      core::Signal::Convert(&out_signal), HSA_SIGNAL_CONDITION_EQ, 0,
      [](hsa_signal_value_t, void* arg) -> bool {
        void* buf = arg;
        hsa_amd_memory_pool_free(buf);
        return false;
      },
      dst_list_gpu);

  if (!handler_registered) {
    // Handler registration failed - free buffer and return error before dispatch
    hsa_amd_memory_pool_free(dst_list_gpu);
    return HSA_STATUS_ERROR_OUT_OF_RESOURCES;
  }

  PopulateQueue(dispatch_index, uintptr_t(kernels_[KernelType::BroadcastCopy].code_buf_), args,
                total_workitems, core::Signal::Convert(&out_signal));

  ReleaseWriteIndex(write_index, total_num_packet);

  return HSA_STATUS_SUCCESS;
}

hsa_status_t BlitKernel::SubmitSwapCopyCommand(void* addr_a, void* addr_b, size_t size,
                                               std::vector<core::Signal*>& dep_signals,
                                               core::Signal& out_signal,
                                               std::vector<core::Signal*>& gang_signals) {
  if (addr_a == nullptr || addr_b == nullptr) {
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }

  if (size == 0) {
    return HSA_STATUS_SUCCESS;
  }

  // swap_size is uint32_t in kernel args, so limit to 4GB
  if (size > UINT32_MAX) {
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }

  const uint32_t num_barrier_packet = uint32_t((dep_signals.size() + 4) / 5);
  const uint32_t total_num_packet = num_barrier_packet + 1;

  uint64_t write_index;
  {
    std::lock_guard<std::mutex> lock(reservation_lock_);
    write_index = AcquireWriteIndex(total_num_packet);
    // Swap moves 2x the size (read both, write both)
    RecordBlitHistory(size * 2, write_index + total_num_packet - 1);
  }

  uint64_t write_index_temp = write_index;

  const uint16_t kBarrierPacketHeader = (HSA_PACKET_TYPE_BARRIER_AND << HSA_PACKET_HEADER_TYPE) |
      (HSA_FENCE_SCOPE_NONE << HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE) |
      (HSA_FENCE_SCOPE_NONE << HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE);

  hsa_barrier_and_packet_t barrier_packet = {0};
  barrier_packet.header = HSA_PACKET_TYPE_INVALID;

  hsa_barrier_and_packet_t* queue_buffer =
      reinterpret_cast<hsa_barrier_and_packet_t*>(queue_->public_handle()->base_address);

  const size_t dep_signal_count = dep_signals.size();
  for (size_t i = 0; i < dep_signal_count; ++i) {
    const size_t idx = i % 5;
    barrier_packet.dep_signal[idx] = core::Signal::Convert(dep_signals[i]);
    if (i == (dep_signal_count - 1) || idx == 4) {
      std::atomic_thread_fence(std::memory_order_acquire);
      queue_buffer[(write_index_temp)&queue_bitmask_] = barrier_packet;
      std::atomic_thread_fence(std::memory_order_release);
      queue_buffer[(write_index_temp)&queue_bitmask_].header = kBarrierPacketHeader;
      write_index_temp++;
      barrier_packet = {0};
      barrier_packet.header = HSA_PACKET_TYPE_INVALID;
    }
  }

  // Calculate dispatch parameters: each thread handles 16 bytes
  const uint32_t threads_per_workgroup = 64;
  const uint32_t bytes_per_thread = 16;
  // Align size up to 16 bytes, then divide to get thread count
  const uint32_t aligned_size =
      (static_cast<uint32_t>(size) + bytes_per_thread - 1) & ~(bytes_per_thread - 1);
  const uint32_t total_workitems = aligned_size / bytes_per_thread;

  KernelArgs* args = ObtainAsyncKernelCopyArg();
  args->swap.addr_a = reinterpret_cast<uint64_t>(addr_a);
  args->swap.addr_b = reinterpret_cast<uint64_t>(addr_b);
  args->swap.swap_size = static_cast<uint32_t>(size);

  const uint64_t dispatch_index = write_index + num_barrier_packet;

  PopulateQueue(dispatch_index, uintptr_t(kernels_[KernelType::SwapCopy].code_buf_), args,
                total_workitems, core::Signal::Convert(&out_signal));

  ReleaseWriteIndex(write_index, total_num_packet);

  return HSA_STATUS_SUCCESS;
}

hsa_status_t BlitKernel::SubmitIndirectCopyCommand(const void* src, void* dst, size_t size,
                                                   bool src_indirect, bool dst_indirect,
                                                   std::vector<core::Signal*>& dep_signals,
                                                   core::Signal& out_signal,
                                                   std::vector<core::Signal*>& gang_signals) {
  if (src == nullptr || dst == nullptr) {
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }

  // Indirect pointers must be 8-byte aligned (they point to 64-bit addresses)
  if (src_indirect && (reinterpret_cast<uintptr_t>(src) & 0x7) != 0) {
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }
  if (dst_indirect && (reinterpret_cast<uintptr_t>(dst) & 0x7) != 0) {
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }

  if (size == 0) {
    return HSA_STATUS_SUCCESS;
  }

  // copy_size is uint32_t in kernel args, so limit to 4GB
  if (size > UINT32_MAX) {
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }

  const uint32_t num_barrier_packet = uint32_t((dep_signals.size() + 4) / 5);
  const uint32_t total_num_packet = num_barrier_packet + 1;

  uint64_t write_index;
  {
    std::lock_guard<std::mutex> lock(reservation_lock_);
    write_index = AcquireWriteIndex(total_num_packet);
    RecordBlitHistory(size, write_index + total_num_packet - 1);
  }

  uint64_t write_index_temp = write_index;

  const uint16_t kBarrierPacketHeader = (HSA_PACKET_TYPE_BARRIER_AND << HSA_PACKET_HEADER_TYPE) |
      (HSA_FENCE_SCOPE_NONE << HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE) |
      (HSA_FENCE_SCOPE_NONE << HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE);

  hsa_barrier_and_packet_t barrier_packet = {0};
  barrier_packet.header = HSA_PACKET_TYPE_INVALID;

  hsa_barrier_and_packet_t* queue_buffer =
      reinterpret_cast<hsa_barrier_and_packet_t*>(queue_->public_handle()->base_address);

  const size_t dep_signal_count = dep_signals.size();
  for (size_t i = 0; i < dep_signal_count; ++i) {
    const size_t idx = i % 5;
    barrier_packet.dep_signal[idx] = core::Signal::Convert(dep_signals[i]);
    if (i == (dep_signal_count - 1) || idx == 4) {
      std::atomic_thread_fence(std::memory_order_acquire);
      queue_buffer[(write_index_temp)&queue_bitmask_] = barrier_packet;
      std::atomic_thread_fence(std::memory_order_release);
      queue_buffer[(write_index_temp)&queue_bitmask_].header = kBarrierPacketHeader;
      write_index_temp++;
      barrier_packet = {0};
      barrier_packet.header = HSA_PACKET_TYPE_INVALID;
    }
  }

  // Calculate dispatch parameters: single workgroup with 64 threads
  const uint32_t threads_per_workgroup = 64;
  const uint32_t total_workitems = threads_per_workgroup;

  // Build indirect flags
  uint32_t indirect_flags = 0;
  if (src_indirect) indirect_flags |= 1;
  if (dst_indirect) indirect_flags |= 2;

  KernelArgs* args = ObtainAsyncKernelCopyArg();
  args->indirect_copy.src_addr = reinterpret_cast<uint64_t>(src);
  args->indirect_copy.dst_addr = reinterpret_cast<uint64_t>(dst);
  args->indirect_copy.copy_size = static_cast<uint32_t>(size);
  args->indirect_copy.num_workitems = total_workitems;
  args->indirect_copy.indirect_flags = indirect_flags;

  const uint64_t dispatch_index = write_index + num_barrier_packet;

  PopulateQueue(dispatch_index, uintptr_t(kernels_[KernelType::IndirectCopy].code_buf_), args,
                total_workitems, core::Signal::Convert(&out_signal));

  ReleaseWriteIndex(write_index, total_num_packet);

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

  hsa_kernel_dispatch_packet_t packet = { };

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
#if defined(__linux__)
  __atomic_store_n(&(queue_buffer[index & queue_bitmask_].full_header),
                    kDispatchPacketHeader | packet.setup << 16, __ATOMIC_RELEASE);
#else
  std::atomic_ref<uint32_t> atomic_header(queue_buffer[index & queue_bitmask_].full_header);
  atomic_header.store(kDispatchPacketHeader | packet.setup << 16, std::memory_order_release);
#endif
  LogPrint(HSA_AMD_LOG_FLAG_AQL,
    "HWq=%p, id=%lu, Dispatch Header = "
    "0x%x (type=%d, barrier=%d, acquire=%d, release=%d), "
    "setup=%d, grid=[%zu, %zu, %zu], workgroup=[%zu, %zu, %zu], private_seg_size=%zu, "
    "group_seg_size=%zu, kernel_obj=0x%zx, kernarg_address=0x%zx, completion_signal=0x%zx "
    "rptr=%lu, wptr=%lu",
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
    packet.setup, static_cast<size_t>(packet.grid_size_x), static_cast<size_t>(packet.grid_size_y), static_cast<size_t>(packet.grid_size_z),
    static_cast<size_t>(packet.workgroup_size_x), static_cast<size_t>(packet.workgroup_size_y), static_cast<size_t>(packet.workgroup_size_z),
    static_cast<size_t>(packet.private_segment_size), static_cast<size_t>(packet.group_segment_size),
    packet.kernel_object,reinterpret_cast<uintptr_t>(packet.kernarg_address),
    completion_signal.handle, queue_->LoadReadIndexRelaxed(), index);
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
