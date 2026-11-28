////////////////////////////////////////////////////////////////////////////////
//
// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT
//
////////////////////////////////////////////////////////////////////////////////

// BroadcastCopy Shader
// Copies a single source buffer to multiple destinations using compute shader.
// Each workgroup handles one destination, all threads in workgroup copy in parallel.
//
// Kernel Arguments (24 bytes):
//   [DW 0-1]  Source address (64-bit)
//   [DW 2-3]  Destination list address (64-bit) - array of destination pointers
//   [DW 4  ]  Number of destinations (32-bit)
//   [DW 5  ]  Copy size per destination (32-bit)

.text

.macro V_ADD_CO_U32 vdst, src0, vsrc1
  .if (.amdgcn.gfx_generation_number >= 10)
     v_add_co_u32        \vdst, vcc_lo, \src0, \vsrc1
  .elseif (.amdgcn.gfx_generation_number >= 9)
    v_add_co_u32        \vdst, vcc, \src0, \vsrc1
  .else
    v_add_u32           \vdst, vcc, \src0, \vsrc1
  .endif
.endm

.macro V_ADD_CO_CI_U32 vdst, src0, vsrc1
  .if (.amdgcn.gfx_generation_number >= 10)
    v_add_co_ci_u32     \vdst, vcc_lo, \src0, \vsrc1, vcc_lo
  .elseif (.amdgcn.gfx_generation_number >= 9)
    v_addc_co_u32       \vdst, vcc, \src0, \vsrc1, vcc
  .else
    v_addc_u32          \vdst, vcc, \src0, \vsrc1, vcc
  .endif
.endm

.macro V_CMP_LT_U64 src0, vsrc1
  .if (.amdgcn.gfx_generation_number >= 10)
    v_cmp_lt_u64        vcc_lo, \src0, \vsrc1
  .else
    v_cmp_lt_u64        vcc, \src0, \vsrc1
  .endif
.endm

.macro S_AND_B64 sdst, src0, ssrc1
  .if (.amdgcn.gfx_generation_number >= 10)
    s_and_b32           \sdst, \src0, \ssrc1
  .else
    s_and_b64           \sdst, \src0, \ssrc1
  .endif
.endm

// GFX12 replaces the packed s_waitcnt encoding with one instruction per
// counter. The two S_WAITCNT_ macros below gate on the generation alone: the
// per-counter forms exist from GFX12.0, so there is no reason to keep GFX12.0
// on an encoding that the compiler itself no longer emits for that generation.
.macro S_WAITCNT_KMCNT
  .if (.amdgcn.gfx_generation_number >= 12)
    s_wait_kmcnt        0
  .else
    s_waitcnt           lgkmcnt(0)
  .endif
.endm

.macro S_WAITCNT_LOADCNT
  .if (.amdgcn.gfx_generation_number >= 12)
    s_wait_loadcnt      0
  .else
    s_waitcnt           vmcnt(0)
  .endif
.endm

// GFX12.5 adds a counter that is incremented when a memory instruction issues
// and decremented when the address translation for that instruction completes.
// Until that point the instruction still holds a claim on its own source
// registers - its address, and for a store its data - and nothing interlocks a
// write to them, so the wait belongs to software. The compiler models it the
// same way: it scores every source operand of a memory instruction against this
// counter and makes any later definition of one consult it.
//
// A wait on another memory counter is documented to imply this one, but the
// compiler does not treat the load counter as covering it while a store is
// outstanding. This shader issues no store-completion wait at all, so from the
// second iteration onward a store is always outstanding and the load waits
// below can never stand in. Every copy step advances the pointers it has just
// handed to a store, so every one of them needs this.
//
// The predicate names generation 12 rather than accepting later ones. GFX13
// inherits the GFX12.5 instruction set but not this counter, and the wait would
// not assemble there. Earlier generations have no such counter and need none.
.macro S_WAIT_XCNT
  .if (.amdgcn.gfx_generation_number == 12 && .amdgcn.gfx_generation_minor >= 5)
    s_wait_xcnt         0x0
  .endif
.endm

// No store wait follows a store in this shader, deliberately.
//
// The only thing such a wait could protect here is the reuse of a store's own
// address and data VGPRs, which the next unrolled iteration overwrites. From
// GFX7 through GFX12.0 that needs no wait: the store captures its operands at
// issue. The counter that once tracked operand release was expcnt, and
// GCNSubtarget::vmemWriteNeedsExpWaitcnt() returns getGeneration() < SEA_ISLANDS,
// so it has not applied since GFX6.
//
// GFX12.5 is the exception and reintroduces exactly that exposure through a
// separate counter, which is why S_WAIT_XCNT above exists and follows every
// store. That wait is not a store-completion wait and does not order the write
// against memory; it only releases the source registers.
//
// This shader never reads back what it wrote, and no store wait is required
// before s_endpgm. A store-completion wait would therefore order nothing that
// needs ordering, while serialising every 16-byte chunk of every destination
// against memory completion.

.set kBroadcastCopyVecWidth, 4
.set kBroadcastCopyUnroll, 4

// The shader occupies s0-s13 (14 SGPRs); the extra six leave room for the
// FLAT_SCRATCH / XNACK_MASK / VCC pairs the hardware aliases into the top of
// the wave's GFX9 allocation.
//
// Only the .text section of this file is extracted, so the compute_pgm_rsrc1
// assignments below never reach the GPU: the runtime builds the kernel
// descriptor itself from its own copy of these counts. Keep the two in sync.
.set kBroadcastCopyNumSGPRs, 20

// VGPR allocation: local_id is held in v16 on ALL architectures.
// The vectorized copy loop reloads source data into v[12:15] on every iteration
// (flat_load_dwordx4 v[12:15]), so any local_id register in that range is
// overwritten mid-loop and the byte-tail address later derived from it is
// corrupted. v16 sits above the load range and survives the loop, so the tail
// is always addressed correctly. The runtime allocates 17 VGPRs (v0..v16) for
// this kernel on every architecture.
.set kBroadcastCopyNumVGPRs, 17
.set LOCAL_ID_VREG, 16

.set BroadcastCopyRsrc1SGPRs, (kBroadcastCopyNumSGPRs - 1) / 8
  .if BroadcastCopyRsrc1SGPRs < 0
    .set BroadcastCopyRsrc1SGPRs, 0
  .endif

.set BroadcastCopyRsrc1VGPRs, (kBroadcastCopyNumVGPRs - 1) / 4
  .if BroadcastCopyRsrc1VGPRs < 0
    .set BroadcastCopyRsrc1VGPRs, 0
  .endif

.p2align 8

BroadcastCopy:

    compute_pgm_rsrc1_sgprs = BroadcastCopyRsrc1SGPRs
    compute_pgm_rsrc1_vgprs = BroadcastCopyRsrc1VGPRs
    compute_pgm_rsrc2_user_sgpr = 2
    compute_pgm_rsrc2_tgid_x_en = 1
    enable_sgpr_kernarg_segment_ptr = 1

    // Load kernel arguments
    // s[4:5] = src_addr, s[6:7] = dst_list_addr, s[8:9] = {num_destinations, copy_size}.
    // The grid is sized by the host from those two counts, so the shader derives
    // its bounds from them and loads nothing further; s[10:11] is scratch below.
    s_load_dwordx4      s[4:7], s[0:1], 0x0
    s_load_dwordx2      s[8:9], s[0:1], 0x10
    S_WAITCNT_KMCNT

    // Save local work-item ID (v0 at kernel entry contains local_id).
    // v16 is used on every architecture: it sits above the v[12:15] range that
    // flat_load_dwordx4 reloads each vectorized iteration, so the value survives
    // the copy loop and the byte-tail address derived from it stays correct.
    v_mov_b32           v16, v0

    // Get workgroup ID (destination index)
    // Workgroup-ID source per architecture (must match the other blit shaders):
    //   GFX12:  ttmp9 holds workgroup_id_x
    //   others: s2, the architected tgid_x SGPR enabled by
    //           compute_pgm_rsrc2_tgid_x_en above
    .if (.amdgcn.gfx_generation_number == 12)
      v_mov_b32         v0, ttmp9
    .else
      v_mov_b32         v0, s2
    .endif

    // Load destination address from dst_list[workgroup_id]
    // v1 = workgroup_id * 8 (each address is 8 bytes)
    v_lshlrev_b32       v1, 3, v0
    v_mov_b32           v3, s7
    V_ADD_CO_U32        v2, v1, s6
    V_ADD_CO_CI_U32     v3, v3, 0x0

    flat_load_dwordx2   v[4:5], v[2:3]
    S_WAITCNT_LOADCNT

    // For broadcast: each workgroup copies FULL data to its destination
    // Use local_id (v16 on all archs) for src/dst addressing
    // This ensures all workgroups read the same source data

    // Main Loop: 16-byte vectorized copy
    // s12 = 64 * 16 = 1024 (stride for workgroup, not total workitems)
    // s13 = 3 * s12 = 3072 (for unroll bounds adjustment)
    s_mov_b32           s12, 64 * 16
    s_mov_b32           s13, 64 * 16 * 3

    // v[6:7] = src_addr + local_id * 16
    v_lshlrev_b32       v1, 4, v16
    v_mov_b32           v7, s5
    V_ADD_CO_U32        v6, v1, s4
    V_ADD_CO_CI_U32     v7, v7, 0x0

    // v[8:9] = dst_addr + local_id * 16
    V_ADD_CO_U32        v8, v1, v4
    V_ADD_CO_CI_U32     v9, v5, 0x0

    // v[10:11] = src_addr + (copy_size & ~0xF) (end of vectorized region)
    v_and_b32           v10, 0xFFFFFFF0, s9
    v_mov_b32           v11, s5
    V_ADD_CO_U32        v10, v10, s4
    V_ADD_CO_CI_U32     v11, v11, 0x0

    // Phase 1: Unrolled loop (4 iterations per check)
    // Adjusted end: v[10:11] - 3*stride ensures 4 iterations are always safe
    // We compare against (end - 3*stride) so when ptr < adjusted_end,
    // ptr, ptr+stride, ptr+2*stride, ptr+3*stride are all < end

    // Compute adjusted_end = end - 3*stride in v[0:1] (must be 64-bit aligned for gfx1250)
    // Use v_sub for 64-bit subtraction
    v_mov_b32           v0, s13             // v0 = 3*stride
    v_mov_b32           v1, 0
    // v[0:1] = v[10:11] - v[0:1] (end - 3*stride)
    .if (.amdgcn.gfx_generation_number >= 10)
      v_sub_co_u32      v0, vcc_lo, v10, v0
      v_sub_co_ci_u32   v1, vcc_lo, v11, v1, vcc_lo
    .elseif (.amdgcn.gfx_generation_number >= 9)
      v_sub_co_u32      v0, vcc, v10, v0
      v_subb_co_u32     v1, vcc, v11, v1, vcc
    .else
      v_sub_u32         v0, vcc, v10, v0
      v_subb_u32        v1, vcc, v11, v1, vcc
    .endif

    // Save the full EXEC mask before the vectorized phase. The single-step
    // cleanup loop below narrows EXEC to the lanes still in bounds and does not
    // restore it per iteration, so the mask is restored once at
    // L_BROADCAST_VEC_DONE before the byte tail, which re-derives its own lane
    // participation and therefore needs all workitems active again.
    .if (.amdgcn.gfx_generation_number >= 10)
      s_mov_b32         s3, exec_lo
    .else
      s_mov_b64         s[2:3], exec
    .endif

L_BROADCAST_VEC_UNROLL:
    // Enter the unrolled body only when EVERY active lane can complete all four
    // iterations in bounds (ptr < end - 3*stride). If any active lane would
    // overrun, fall through to the masked single-step cleanup instead. Testing
    // (vcc & exec) == exec (all active lanes set) rather than "any lane set"
    // stops out-of-range lanes from issuing a 16-byte access past the buffer.
    V_CMP_LT_U64        v[6:7], v[0:1]
    .if (.amdgcn.gfx_generation_number >= 10)
      s_and_b32         s11, vcc_lo, exec_lo
      s_cmp_lg_u32      s11, exec_lo
    .else
      s_and_b64         s[10:11], vcc, exec
      s_cmp_lg_u64      s[10:11], exec
    .endif
    s_cbranch_scc1      L_BROADCAST_VEC_SINGLE

    // Unrolled vectorized copy (4 iterations, bounds guaranteed safe)
    flat_load_dwordx4   v[12:15], v[6:7]
    S_WAITCNT_LOADCNT
    flat_store_dwordx4  v[8:9], v[12:15]
    S_WAIT_XCNT

    V_ADD_CO_U32        v6, v6, s12
    V_ADD_CO_CI_U32     v7, v7, 0x0
    V_ADD_CO_U32        v8, v8, s12
    V_ADD_CO_CI_U32     v9, v9, 0x0

    flat_load_dwordx4   v[12:15], v[6:7]
    S_WAITCNT_LOADCNT
    flat_store_dwordx4  v[8:9], v[12:15]
    S_WAIT_XCNT

    V_ADD_CO_U32        v6, v6, s12
    V_ADD_CO_CI_U32     v7, v7, 0x0
    V_ADD_CO_U32        v8, v8, s12
    V_ADD_CO_CI_U32     v9, v9, 0x0

    flat_load_dwordx4   v[12:15], v[6:7]
    S_WAITCNT_LOADCNT
    flat_store_dwordx4  v[8:9], v[12:15]
    S_WAIT_XCNT

    V_ADD_CO_U32        v6, v6, s12
    V_ADD_CO_CI_U32     v7, v7, 0x0
    V_ADD_CO_U32        v8, v8, s12
    V_ADD_CO_CI_U32     v9, v9, 0x0

    flat_load_dwordx4   v[12:15], v[6:7]
    S_WAITCNT_LOADCNT
    flat_store_dwordx4  v[8:9], v[12:15]
    S_WAIT_XCNT

    V_ADD_CO_U32        v6, v6, s12
    V_ADD_CO_CI_U32     v7, v7, 0x0
    V_ADD_CO_U32        v8, v8, s12
    V_ADD_CO_CI_U32     v9, v9, 0x0

    s_branch            L_BROADCAST_VEC_UNROLL

    // Phase 2: Single-iteration cleanup (handles remaining 0-3 iterations).
    // EXEC is narrowed to the lanes still in bounds so out-of-range lanes do not
    // read or write past the end of the region. The mask is restored after the
    // loop at L_BROADCAST_VEC_DONE.
L_BROADCAST_VEC_SINGLE:
    V_CMP_LT_U64        v[6:7], v[10:11]
    s_cbranch_vccz      L_BROADCAST_VEC_DONE
    .if (.amdgcn.gfx_generation_number >= 10)
      s_and_b32         exec_lo, exec_lo, vcc_lo
    .else
      s_and_b64         exec, exec, vcc
    .endif

    flat_load_dwordx4   v[12:15], v[6:7]
    S_WAITCNT_LOADCNT
    flat_store_dwordx4  v[8:9], v[12:15]
    S_WAIT_XCNT

    V_ADD_CO_U32        v6, v6, s12
    V_ADD_CO_CI_U32     v7, v7, 0x0
    V_ADD_CO_U32        v8, v8, s12
    V_ADD_CO_CI_U32     v9, v9, 0x0

    s_branch            L_BROADCAST_VEC_SINGLE

L_BROADCAST_VEC_DONE:
    // Restore the full EXEC mask saved before the vectorized phase; the byte
    // tail re-derives its own participation from all workitems.
    .if (.amdgcn.gfx_generation_number >= 10)
      s_mov_b32         exec_lo, s3
    .else
      s_mov_b64         exec, s[2:3]
    .endif

    // Tail Loop: Byte-by-byte copy for remaining bytes after vectorized portion
    // Start at: (copy_size & ~0xF) + local_id
    //
    // The offset is built as a 64-bit value. copy_size is a 32-bit byte count
    // whose permitted maximum leaves less than one workgroup of headroom below
    // 2^32, so (copy_size & ~0xF) + local_id overflows 32 bits on the largest
    // transfers. A truncated offset points back near the start of the buffer,
    // where it still compares below the end address, so the loop guard admits
    // it and the tail copies from the wrong place instead of exiting.
    //
    // v[2:3] = (copy_size & ~0xF) + local_id, zero-extended to 64 bits
    v_and_b32           v2, 0xFFFFFFF0, s9
    v_mov_b32           v3, 0
    V_ADD_CO_U32        v2, v2, v16
    V_ADD_CO_CI_U32     v3, v3, 0x0

    // v[6:7] = src_addr + v[2:3]
    v_mov_b32           v7, s5
    V_ADD_CO_U32        v6, v2, s4
    V_ADD_CO_CI_U32     v7, v7, v3

    // v[8:9] = dst_addr + v[2:3]
    V_ADD_CO_U32        v8, v2, v4
    V_ADD_CO_CI_U32     v9, v5, v3

    // v[10:11] = src_addr + copy_size (end address)
    v_mov_b32           v10, s9
    v_mov_b32           v11, s5
    V_ADD_CO_U32        v10, v10, s4
    V_ADD_CO_CI_U32     v11, v11, 0x0

L_BROADCAST_BYTE_LOOP:
    V_CMP_LT_U64        v[6:7], v[10:11]
    s_cbranch_vccz      L_BROADCAST_BYTE_DONE
    // EXEC mask progressively narrowed - not restored since s_endpgm follows
    .if (.amdgcn.gfx_generation_number >= 10)
      s_and_b32         exec_lo, exec_lo, vcc_lo
    .else
      s_and_b64         exec, exec, vcc
    .endif

    flat_load_ubyte     v1, v[6:7]
    S_WAITCNT_LOADCNT
    flat_store_byte     v[8:9], v1
    S_WAIT_XCNT

    // Stride by 64 (workgroup size), not total workitems
    V_ADD_CO_U32        v6, v6, 64
    V_ADD_CO_CI_U32     v7, v7, 0x0
    V_ADD_CO_U32        v8, v8, 64
    V_ADD_CO_CI_U32     v9, v9, 0x0

    s_branch            L_BROADCAST_BYTE_LOOP

L_BROADCAST_BYTE_DONE:
    s_endpgm
