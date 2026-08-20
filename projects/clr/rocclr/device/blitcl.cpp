/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

namespace amd::device {

#define BLIT_KERNELS(...) #__VA_ARGS__

const char* BlitLinearSourceCode = BLIT_KERNELS(
    // Extern
    extern void __amd_fillBufferAligned2D(__global uchar*, __global ushort*, __global uint*,
                                          __global ulong*, __constant uchar*, uint, ulong, ulong,
                                          ulong, ulong);

    extern void __amd_copyBuffer(__global uchar*, __global uchar*, ulong, ulong, ulong, uint);

    extern void __amd_copyBufferAligned(__global uint*, __global uint*, ulong, ulong, ulong, uint);

    extern void __amd_copyBufferRect(__global uchar*, __global uchar*, ulong4, ulong4, ulong4);

    extern void __amd_copyBufferRectAligned(__global uint*, __global uint*, ulong4, ulong4, ulong4);

    extern void __amd_streamOpsWrite(__global uint*, __global ulong*, ulong);

    extern void __amd_streamOpsIncrement(__global uint*, __global ulong*, ulong);

    extern void __amd_streamOpsDecrement(__global uint*, __global ulong*, ulong);

    extern void __amd_streamOpsWait(__global uint*, __global ulong*, ulong, ulong, ulong);

    extern void __amd_batchMemOp(__global void*, uint count);

    extern void __ockl_dm_init_v1(ulong, ulong, uint, uint);

    extern void __amd_fillBufferUnAligned(
        __global void* __restrict buf, __constant uchar* __restrict pattern,
        ulong2 body_tile_pattern, ulong body_pattern, ulong body_tail_pattern,
        ulong body_tile_count, ulong body_tile_passes, ulong stride,
        ulong pattern_size, ulong tail_offset, __global uchar* __restrict body_ptr,
        __global uchar* __restrict body_tail_ptr, __global uchar* __restrict tail_ptr,
        __global ulong2* __restrict element_tiled, ushort4 counts);

    __kernel void __amd_rocclr_fillBufferUnAligned(
        __global void* __restrict buf, __constant uchar* __restrict pattern,
        ulong2 body_tile_pattern, ulong body_pattern, ulong body_tail_pattern,
        ulong body_tile_count, ulong body_tile_passes, ulong stride,
        ulong pattern_size, ulong tail_offset, __global uchar* __restrict body_ptr,
        __global uchar* __restrict body_tail_ptr, __global uchar* __restrict tail_ptr,
        __global ulong2* __restrict element_tiled, ushort4 counts) {
      ulong id = get_global_id(0);

      // Cleanup region: lanes 0..15 of group 0 wave 0 handle head/body/body_tail/tail.
      // Body and body_tail are always uint64 stores (always either 0 or 1 element).
      // Aligned-buffer case: counts are all zero, predicates fall through with no work.
      // body_pattern and body_tail_pattern are host-rotated u64 payloads (rotated by their
      // byte-offset-from-fill-start mod patternSize), so the unaligned-base case is byte-correct.
      if (id < 16) {
        __global uchar* head_ptr = (__global uchar*)buf;
        const uint lane = (uint)id;
        const uint head_end = (uint)counts.s0;
        const uint body_end = head_end + (uint)counts.s1;
        const uint body_tail_end = body_end + (uint)counts.s2;
        const uint tail_end = body_tail_end + (uint)counts.s3;

        if (lane < head_end) {
          head_ptr[lane] = pattern[lane & (pattern_size - 1)];
        } else if (lane < body_end) {
          *(__global ulong*)body_ptr = body_pattern;
        } else if (lane < body_tail_end) {
          *(__global ulong*)body_tail_ptr = body_tail_pattern;
        } else if (lane < tail_end) {
          const ulong tail_byte_idx = (ulong)(lane - body_tail_end);
          tail_ptr[tail_byte_idx] =
              pattern[(tail_offset + tail_byte_idx) & (pattern_size - 1)];
        }
      }

      // Tile region: split-last-pass. Bulk passes have unconditional stores
      // (host guarantees they are in-bounds); only the tail pass is per-lane guarded.
      // body_tile_passes is a CPU-known bound for codegen.
      ulong j = 0;
      ulong idx = id;
      for (; j + 1 < body_tile_passes; ++j, idx += stride) {
        element_tiled[idx] = body_tile_pattern;
      }
      if (j < body_tile_passes && idx < body_tile_count) {
        element_tiled[idx] = body_tile_pattern;
      }
    }

    __kernel void __amd_rocclr_fillBufferAligned2D(
        __global uchar* bufUChar, __global ushort* bufUShort, __global uint* bufUInt,
        __global ulong* bufULong, __constant uchar* pattern, uint patternSize, ulong offset,
        ulong width, ulong height, ulong pitch) {
      __amd_fillBufferAligned2D(bufUChar, bufUShort, bufUInt, bufULong, pattern, patternSize,
                                offset, width, height, pitch);
    }

    __kernel void __amd_rocclr_copyBuffer(__global uchar* src, __global uchar* dst, ulong size,
                                          uint remainder, uint aligned_size, ulong end_ptr,
                                          uint next_chunk, uint workgroup_size) {
      uint l = __builtin_amdgcn_workitem_id_x();
      uint g = __builtin_amdgcn_workgroup_id_x();
      ulong id = (g * workgroup_size + l);
      ulong id_remainder = id;

      if (aligned_size == sizeof(ulong2)) {
        __global ulong2* srcD = (__global ulong2*)(src);
        __global ulong2* dstD = (__global ulong2*)(dst);
        while ((ulong)(&dstD[id]) < end_ptr) {
          dstD[id] = srcD[id];
          id += next_chunk;
        }
      } else {
        __global uint* srcD = (__global uint*)(src);
        __global uint* dstD = (__global uint*)(dst);
        while ((ulong)(&dstD[id]) < end_ptr) {
          dstD[id] = srcD[id];
          id += next_chunk;
        }
      }
      if ((remainder != 0) && (id_remainder == 0)) {
        for (ulong i = size - remainder; i < size; ++i) {
          dst[i] = src[i];
        }
      }
    }

    typedef struct CopyBufferBatchDescriptor {
      ulong source_address;
      ulong destination_address;
      ulong aligned_element_count;
      uint aligned_element_size;
      uint trailing_byte_count;
    } CopyBufferBatchDescriptor;

    __kernel void __amd_rocclr_copyBufferBatch(
        __global const CopyBufferBatchDescriptor *descriptors,
        uint workgroup_size,
        uint copy_stride) {
      uint work_item_id = __builtin_amdgcn_workitem_id_x();
      uint group_ordinal = __builtin_amdgcn_workgroup_id_x();
      uint descriptor_index = __builtin_amdgcn_workgroup_id_y();

      CopyBufferBatchDescriptor descriptor = descriptors[descriptor_index];
      __global uchar *source = (__global uchar *)descriptor.source_address;
      __global uchar *destination =
          (__global uchar *)descriptor.destination_address;
      ulong copy_index = ((ulong)group_ordinal * workgroup_size) + work_item_id;

      if (descriptor.aligned_element_size == sizeof(ulong2)) {
        __global ulong2 *source_data = (__global ulong2 *)(source);
        __global ulong2 *destination_data = (__global ulong2 *)(destination);
        while (copy_index < descriptor.aligned_element_count) {
          destination_data[copy_index] = source_data[copy_index];
          copy_index += copy_stride;
        }
      } else {
        __global uint *source_data = (__global uint *)(source);
        __global uint *destination_data = (__global uint *)(destination);
        while (copy_index < descriptor.aligned_element_count) {
          destination_data[copy_index] = source_data[copy_index];
          copy_index += copy_stride;
        }
      }
      if ((descriptor.trailing_byte_count != 0) && (group_ordinal == 0) &&
          (work_item_id == 0)) {
        ulong tail_start =
            descriptor.aligned_element_count * descriptor.aligned_element_size;
        ulong tail_end = tail_start + descriptor.trailing_byte_count;
        for (ulong i = tail_start; i < tail_end; ++i) {
          destination[i] = source[i];
        }
      }
    }

    __kernel void __amd_rocclr_copyBufferAligned(__global uint* src, __global uint* dst,
                                                 ulong srcOrigin, ulong dstOrigin, ulong size,
                                                 uint alignment) {
      __amd_copyBufferAligned(src, dst, srcOrigin, dstOrigin, size, alignment);
    }

    __kernel void __amd_rocclr_copyBufferRect(__global uchar* src, __global uchar* dst,
                                              ulong4 srcRect, ulong4 dstRect, ulong4 size) {
      __amd_copyBufferRect(src, dst, srcRect, dstRect, size);
    }

    __kernel void __amd_rocclr_copyBufferRectAligned(__global uint* src, __global uint* dst,
                                                     ulong4 srcRect, ulong4 dstRect, ulong4 size) {
      __amd_copyBufferRectAligned(src, dst, srcRect, dstRect, size);
    }

    // TODO: Once the sequential for-loop fix lands in llvm-project/amd/device-libs/opencl/src/misc/amdblit.cl
    // (replacing get_global_id(0) with a for loop), revert this back to:
    //   __amd_batchMemOp(params, count);
    //
    // Local definitions mirroring BatchMemOpType and BatchMemOpParams from
    // amd/device-libs/opencl/src/misc/amdblit.cl. Defined here because blitcl.cpp
    // kernels are JIT-compiled by COMGR in OpenCL C 1.x mode, which does not have
    // access to amdblit.cl's types. Values match hipStreamBatchMemOpType in hip_runtime_api.h.
    typedef enum {
      ROCCLR_STREAM_WAIT_VALUE_32  = 0x1,
      ROCCLR_STREAM_WRITE_VALUE_32 = 0x2,
      ROCCLR_STREAM_WAIT_VALUE_64  = 0x4,
      ROCCLR_STREAM_WRITE_VALUE_64 = 0x5,
    } RocclrBatchMemOpType;

    // Mirrors BatchMemOpParams in amdblit.cl — uses __global ulong* instead of
    // atomic_ulong* since atomic_ulong requires OpenCL C 2.0 unavailable here.
    typedef union {
      RocclrBatchMemOpType operation;
      struct {
        RocclrBatchMemOpType  operation;
        __global ulong*       address;
        union { uint value; ulong value64; };
        uint                  flags;
        __global ulong*       alias;
      } waitValue;
      struct {
        RocclrBatchMemOpType  operation;
        __global ulong*       address;
        union { uint value; ulong value64; };
        uint                  flags;
        __global ulong*       alias;
      } writeValue;
      ulong pad[6];
    } RocclrBatchMemOpParams;
    __kernel void __amd_rocclr_batchMemOp(__global void* params, uint count) {
      __global RocclrBatchMemOpParams* param = (__global RocclrBatchMemOpParams*)params;
      for (uint i = 0; i < count; i++) {
        switch (param[i].operation) {
          case ROCCLR_STREAM_WAIT_VALUE_32:
            __amd_streamOpsWait((__global uint*)param[i].waitValue.address, NULL,
                                (uint)param[i].waitValue.value, (uint)param[i].waitValue.flags,
                                (ulong)~0UL);
            break;
          case ROCCLR_STREAM_WRITE_VALUE_32:
            __amd_streamOpsWrite((__global uint*)param[i].writeValue.address, NULL,
                                 (uint)param[i].writeValue.value);
            break;
          case ROCCLR_STREAM_WAIT_VALUE_64:
            __amd_streamOpsWait(NULL, (__global ulong*)param[i].waitValue.address,
                                param[i].waitValue.value64,
                                (uint)param[i].waitValue.flags, (ulong)~0UL);
            break;
          case ROCCLR_STREAM_WRITE_VALUE_64:
            __amd_streamOpsWrite(NULL, (__global ulong*)param[i].writeValue.address,
                                 param[i].writeValue.value64);
            break;
          default:
            break;
        }
      }
    });

const char* HipExtraSourceCode = BLIT_KERNELS(
    __kernel void __amd_rocclr_streamOpsWrite(__global uint* ptrInt, __global ulong* ptrUlong,
                                              ulong value) {
      __amd_streamOpsWrite(ptrInt, ptrUlong, value);
    }

    __kernel void __amd_rocclr_streamOpsIncrement(__global uint* ptrInt, __global ulong* ptrUlong,
                                                  ulong value) {
      __amd_streamOpsIncrement(ptrInt, ptrUlong, value);
    }

    __kernel void __amd_rocclr_streamOpsDecrement(__global uint* ptrInt, __global ulong* ptrUlong,
                                                  ulong value) {
      __amd_streamOpsDecrement(ptrInt, ptrUlong, value);
    }

    __kernel void __amd_rocclr_streamOpsWait(__global uint* ptrInt, __global ulong* ptrUlong,
                                             ulong value, ulong flags, ulong mask) {
      __amd_streamOpsWait(ptrInt, ptrUlong, value, flags, mask);
    }

    __kernel void __amd_rocclr_initHeap(ulong heap_to_initialize, ulong initial_blocks,
                                        uint heap_size, uint number_of_initial_blocks) {
      __ockl_dm_init_v1(heap_to_initialize, initial_blocks, heap_size, number_of_initial_blocks);
    }

    __kernel void __amd_rocclr_gwsInit(uint value) { __builtin_amdgcn_ds_gws_init(value, 0); }

    __kernel void __amd_rocclr_graphScheduler(__global void* params) {
      __global ulong* p = (__global ulong*)params;
      __global uchar* cmd_buf = (__global uchar*)p[0];
      uint pkt_count          = (uint)p[1];
      __global uchar* base    = (__global uchar*)p[2];
      uint q_size             = (uint)p[3];
      uint q_mask             = q_size - 1;
      volatile __global ulong* wr_id = (volatile __global ulong*)p[4];
      volatile __global ulong* rd_id = (volatile __global ulong*)p[5];
      volatile __global long* dbell  = (volatile __global long*)p[6];
      ulong wr = atom_add(wr_id, (ulong)pkt_count);
      while (wr + pkt_count - *rd_id > q_size) {}
      for (uint i = 0; i < pkt_count; i++) {
        ulong s = (wr + i) & (ulong)q_mask;
        __global ulong* d = (__global ulong*)(base + s * 64);
        __global ulong* r = (__global ulong*)(cmd_buf + i * 64);
        d[0]=r[0]; d[1]=r[1]; d[2]=r[2]; d[3]=r[3];
        d[4]=r[4]; d[5]=r[5]; d[6]=r[6]; d[7]=r[7];
      }
      mem_fence(CLK_GLOBAL_MEM_FENCE);
      atom_xchg((__global uint*)(base + (wr & (ulong)q_mask) * 64),
                *((__global uint*)cmd_buf));
      *dbell = (long)(wr + pkt_count - 1);
    });

const char* HipExtraSourceCodeNoGWS = BLIT_KERNELS(
    __kernel void __amd_rocclr_streamOpsWrite(__global uint* ptrInt, __global ulong* ptrUlong,
                                              ulong value) {
      __amd_streamOpsWrite(ptrInt, ptrUlong, value);
    }

    __kernel void __amd_rocclr_streamOpsIncrement(__global uint* ptrInt, __global ulong* ptrUlong,
                                                  ulong value) {
      __amd_streamOpsIncrement(ptrInt, ptrUlong, value);
    }

    __kernel void __amd_rocclr_streamOpsDecrement(__global uint* ptrInt, __global ulong* ptrUlong,
                                                  ulong value) {
      __amd_streamOpsDecrement(ptrInt, ptrUlong, value);
    }

    __kernel void __amd_rocclr_streamOpsWait(__global uint* ptrInt, __global ulong* ptrUlong,
                                             ulong value, ulong flags, ulong mask) {
      __amd_streamOpsWait(ptrInt, ptrUlong, value, flags, mask);
    }

    __kernel void __amd_rocclr_initHeap(ulong heap_to_initialize, ulong initial_blocks,
                                        uint heap_size, uint number_of_initial_blocks) {
      __ockl_dm_init_v1(heap_to_initialize, initial_blocks, heap_size, number_of_initial_blocks);
    }

    __kernel void __amd_rocclr_graphScheduler(__global void* params) {
      __global ulong* p = (__global ulong*)params;
      __global uchar* cmd_buf = (__global uchar*)p[0];
      uint pkt_count          = (uint)p[1];
      __global uchar* base    = (__global uchar*)p[2];
      uint q_size             = (uint)p[3];
      uint q_mask             = q_size - 1;
      volatile __global ulong* wr_id = (volatile __global ulong*)p[4];
      volatile __global ulong* rd_id = (volatile __global ulong*)p[5];
      volatile __global long* dbell  = (volatile __global long*)p[6];
      ulong wr = atom_add(wr_id, (ulong)pkt_count);
      while (wr + pkt_count - *rd_id > q_size) {}
      for (uint i = 0; i < pkt_count; i++) {
        ulong s = (wr + i) & (ulong)q_mask;
        __global ulong* d = (__global ulong*)(base + s * 64);
        __global ulong* r = (__global ulong*)(cmd_buf + i * 64);
        d[0]=r[0]; d[1]=r[1]; d[2]=r[2]; d[3]=r[3];
        d[4]=r[4]; d[5]=r[5]; d[6]=r[6]; d[7]=r[7];
      }
      mem_fence(CLK_GLOBAL_MEM_FENCE);
      atom_xchg((__global uint*)(base + (wr & (ulong)q_mask) * 64),
                *((__global uint*)cmd_buf));
      *dbell = (long)(wr + pkt_count - 1);
    });

// OpenCL C (CL2.0) port of the block-based graph-scheduler kernels
// (__amd_rocclr_graphCondBranchWhile, __amd_rocclr_graphBlockIssue,
// __amd_rocclr_graphBranch, __amd_rocclr_graphCondBranch,
// __amd_rocclr_graphSwitchBranch, __amd_rocclr_graphReturn,
// __amd_rocclr_graphWhileLoop). Compiled as part of the same blit program as
// the kernels above (via BlitProgram::create's extraKernel argument, see
// Device::createBlitProgram in rocdevice.cpp), so it's resolved from
// device()->blitProgram() rather than a separate amd::Program. Requires
// -cl-std=CL2.0 (for atomic_work_item_fence / atomic_*_explicit with
// explicit memory_scope_*); rocdevice.cpp's createBlitProgram() passes that
// for the HIP extraKernel build.
//
// __amd_rocclr_graphSchedulerHIP (the legacy flat/single-buffer scheduler
// above, and its OpenCL analog __amd_rocclr_graphScheduler) is NOT the same
// mechanism as this block-based scheduler: it has no callers left in the
// runtime (superseded by the block-based scheduler below).
//
// Kernels are migrated off the HIP source (graph_scheduler_kernel.hip) here;
// when HIP_GRAPH_SCHED_CL is set, the loader resolves every kernel present in
// this string from the blit program instead of the HIP one, falling back
// per-kernel to the HIP symbol on any failure.
//
// This is hand-maintained (not generated). The struct layout and queue
// offsets MUST stay in sync with graph_scheduler_kernel.hip and the
// host-side kernarg packing.
//
// Memory-model mapping (HIP builtin -> OpenCL C11), verified to lower to the
// same gfx9xx cache-control bits (agent -> sc1, system -> sc0 sc1):
//   fence(RELEASE,"agent")     -> atomic_work_item_fence(GLOBAL, release, device)
//   fence(RELEASE,"workgroup") -> atomic_work_item_fence(LOCAL|GLOBAL, release, work_group)
//   fence(RELEASE,"")          -> atomic_work_item_fence(GLOBAL, release, all_svm_devices)
//   __atomic_load_n(RELAXED)   -> atomic_load_explicit(relaxed, device)
//   __atomic_store_n(RELEASE)  -> atomic_store_explicit(release, device)
//
// Pointer fields inside ExecutionState/BlockDescriptor/SubBlock are kept as
// plain uint64_t (matching the HIP struct exactly) and reinterpreted to a
// concrete `global` pointer type at each use site. This sidesteps OpenCL's
// address-space-qualified-pointer rules entirely (no pointer-typed struct
// fields to worry about) and keeps the struct layout byte-for-byte identical
// to the host-filled HIP version.
const char* GraphSchedulerCLSourceCode = R"CLSRC(
typedef uint  uint32_t;
typedef ulong uint64_t;
typedef long  int64_t;
typedef uchar uint8_t;

#define QUEUE_BASE_ADDR_OFF 8
#define QUEUE_SIZE_OFF      24
#define QUEUE_WRITE_ID_OFF  56
#define QUEUE_READ_ID_OFF   128

#define COND_EQ 0u
#define COND_NE 1u
#define COND_LT 2u
#define COND_LE 3u
#define COND_GT 4u
#define COND_GE 5u

// Terminator kinds for the walk-loop interpreter's BlockTerminator table,
// matching the host-side FlatBlock::TermType enum order.
#define TERM_BRANCH      0u
#define TERM_COND_BRANCH 1u
#define TERM_RETURN      2u
#define TERM_SWITCH      3u

// BlockTerminator::flags -- see graph_scheduler_kernel.hip for the rationale.
// WAIT: detect the producer by spinning on the condition itself (cheap, WHILE
// latches only) rather than draining a completion signal. ARM: stamp
// COND_SENTINEL when re-entering the loop body.
#define TERM_FLAG_SENTINEL_WAIT 0x1u
#define TERM_FLAG_SENTINEL_ARM  0x2u

// When a branch target equals this sentinel, signal completion inline instead
// of issuing another block.
#define kInlineReturn 0xFFFFFFFFu

// Sentinel written to a WHILE loop's cond_ptr before issuing its body; seeing
// anything else means the body kernel has overwritten it with 0/1.
#define COND_SENTINEL 0xFFFFFFFFFFFFFFFFUL

// AQL packet-type values in the low byte of the 32-bit header word, used by
// the staged fused-WHILE experiment (state->staged_while != 0).
#define AQL_HEADER_INVALID      0x1u
#define AQL_HEADER_BARRIER_NOOP 0x3u

typedef struct {
    uint64_t cmd_buffer_base;
    uint64_t block_table_ptr;
    uint32_t block_count;
    uint32_t current_block;
    uint64_t queue_ptr;
    uint64_t completion_signal_ptr;
    uint64_t kernarg_pool_ptr;
    uint64_t doorbell_ptr;
    uint64_t work_queue_ptr;
    uint64_t work_doorbell_ptr;
    uint64_t cond_init_ptr;
    uint64_t cond_init_value;
    uint64_t cached_queue_base;
    uint32_t cached_queue_size;
    uint32_t num_lane_queues;
    uint64_t subblock_table_ptr;
    uint64_t lane_queue_ptrs;
    uint64_t lane_doorbell_ptrs;
    uint64_t fj_reset_ptr;
    uint32_t fj_reset_count;
    uint32_t staged_while;
    uint64_t prof_cond;
    uint64_t prof_reserve;
    uint64_t prof_memcpy;
    uint64_t prof_doorbell;
    uint64_t cond_init_table_ptr;
    uint32_t cond_init_count;
    uint32_t _pad_cond_init;
    uint64_t term_table_ptr;
    uint32_t entry_block;
    uint32_t _pad_walk;
} ExecutionState;

// One per block: the control-flow decision to make after issuing the block's
// packets. Layout must stay byte-identical to BlockTerminator in
// graph_scheduler_kernel.hip and hip_graph_internal.hpp (64 bytes).
typedef struct {
    uint32_t kind;
    uint32_t branch_target;
    uint32_t true_target;
    uint32_t false_target;
    uint32_t cond_op;
    uint32_t num_cases;
    uint32_t default_target;
    uint32_t flags;
    uint64_t cond_ptr;
    uint64_t cond_value;
    uint64_t signal_addr;
    uint64_t case_targets_ptr;
} BlockTerminator;

// One dispatchable block: a byte range of AQL packets in the shared command
// buffer, plus optional fork/join sub-block fan-out.
typedef struct {
    uint32_t packet_offset;
    uint32_t packet_count;
    uint32_t subblock_index;
    uint32_t subblock_count;
} BlockDescriptor;

// One per-queue packet range within a fork/join block.
typedef struct {
    uint32_t queue_index;
    uint32_t packet_offset;
    uint32_t packet_count;
    uint32_t _pad;
} SubBlock;

static inline bool evaluate_cond(uint64_t lhs, uint32_t cond, uint64_t rhs) {
  switch (cond) {
    case COND_EQ: return lhs == rhs;
    case COND_NE: return lhs != rhs;
    case COND_LT: return (int64_t)lhs <  (int64_t)rhs;
    case COND_LE: return (int64_t)lhs <= (int64_t)rhs;
    case COND_GT: return (int64_t)lhs >  (int64_t)rhs;
    case COND_GE: return (int64_t)lhs >= (int64_t)rhs;
    default:      return false;
  }
}

// Cooperative copy in 16-byte (uint4) chunks across the workgroup's threads.
static inline void copy_coop(uint8_t* dst, const uint8_t* src,
                             uint32_t bytes, uint32_t tid, uint32_t nt) {
  uint32_t n16 = bytes >> 4;
  uint4*       d = (uint4*)dst;
  const uint4* s = (const uint4*)src;
  for (uint32_t i = tid; i < n16; i += nt) {
    d[i] = s[i];
  }
}

// Signal graph-chain completion (shared by graphReturn, the inline-return
// paths of graphCondBranch/graphSwitchBranch/graphCondBranchWhile, and
// graphWhileLoop's loop-exit). Matches HIP's `fence(RELEASE,"")` (system-wide
// scope) + relaxed-to-release store to the completion signal's value field.
static inline void signal_return(__global ExecutionState* state) {
  if (state->completion_signal_ptr) {
    atomic_work_item_fence(CLK_GLOBAL_MEM_FENCE, memory_order_release,
                           memory_scope_all_svm_devices);
    atomic_store_explicit((volatile global atomic_long*)state->completion_signal_ptr, 0,
                          memory_order_release, memory_scope_all_svm_devices);
  }
}

// Reset every conditional handle in the launch table to its default value.
static inline void reset_cond_table(__global ExecutionState* state) {
  if (!state->cond_init_table_ptr || state->cond_init_count == 0) return;
  __global const uint64_t* tbl = (__global const uint64_t*)state->cond_init_table_ptr;
  for (uint32_t i = 0; i < state->cond_init_count; ++i) {
    uint64_t p = tbl[2 * i];
    if (p) *(volatile global uint64_t*)p = tbl[2 * i + 1];
  }
}

// Single-threaded issue of one packet range into a HW queue's ring buffer.
// Zeroes the source packet's header before the memcpy (so the copied-in
// destination header starts invalid), restores the source header afterward
// (so a re-issued block, e.g. inside a WHILE loop, sees its template intact),
// then publishes the destination header last with a release store -- the CP
// can only observe a fully-formed packet.
static inline void issue_block_to_queue(
    __global uint8_t* src, uint32_t pkt_count, __global uint8_t* qptr,
    volatile __global uint64_t* doorbell) {
  if (pkt_count == 0) return;

  __global uint8_t* base = (__global uint8_t*)(*(__global uint64_t*)(qptr + QUEUE_BASE_ADDR_OFF));
  uint32_t q_size = *(__global uint32_t*)(qptr + QUEUE_SIZE_OFF);
  uint32_t q_mask = q_size - 1;
  volatile global atomic_ulong* wr_id = (volatile global atomic_ulong*)(qptr + QUEUE_WRITE_ID_OFF);
  volatile global atomic_ulong* rd_id = (volatile global atomic_ulong*)(qptr + QUEUE_READ_ID_OFF);

  uint64_t wr = atomic_fetch_add_explicit(wr_id, (uint64_t)pkt_count,
                                          memory_order_relaxed, memory_scope_device);

  while (wr + pkt_count - atomic_load_explicit(rd_id, memory_order_relaxed,
                                               memory_scope_device) > q_size) {}

  uint32_t saved_header = *(volatile __global uint32_t*)src;
  *(volatile __global uint32_t*)src = 0;

  uint32_t start_slot        = (uint32_t)(wr & q_mask);
  uint32_t slots_before_wrap = q_size - start_slot;
  uint32_t first_count  = (pkt_count <= slots_before_wrap) ? pkt_count : slots_before_wrap;
  uint32_t second_count = pkt_count - first_count;

  __builtin_memcpy(base + start_slot * 64, src, first_count * 64);
  if (second_count > 0) {
    __builtin_memcpy(base, src + first_count * 64, second_count * 64);
  }

  *(volatile __global uint32_t*)src = saved_header;

  atomic_work_item_fence(CLK_GLOBAL_MEM_FENCE, memory_order_release, memory_scope_device);

  atomic_store_explicit((volatile global atomic_uint*)(base + start_slot * 64), saved_header,
                        memory_order_release, memory_scope_device);

  if (doorbell) {
    *doorbell = wr + pkt_count - 1;
  }
}

// Issue a whole block. Single-queue blocks (subblock_count == 0) go to the
// control queue. Fork/join blocks fan their lane sub-blocks out to distinct
// HW queues; the control sub-block (queue_index 0) carries the join barrier
// packets plus this block's CFG terminator and is issued last so its
// terminator runs only after the lane packets are already in flight.
static inline void issue_block(__global ExecutionState* state, uint32_t blk) {
  __global const BlockDescriptor* table = (__global const BlockDescriptor*)state->block_table_ptr;
  BlockDescriptor bd = table[blk];
  __global uint8_t* cmd_base = (__global uint8_t*)state->cmd_buffer_base;

  if (bd.subblock_count == 0) {
    issue_block_to_queue(cmd_base + bd.packet_offset, bd.packet_count,
                         (__global uint8_t*)state->queue_ptr,
                         (volatile __global uint64_t*)state->doorbell_ptr);
    return;
  }

  // Re-arm this diamond's fork/join barrier signals to 1 before issuing (see
  // graph_scheduler_kernel.hip for the full rationale: WHILE loops re-issue
  // the same diamond block every iteration and must not see stale signals).
  if (state->fj_reset_count) {
    __global const uint64_t* rs = (__global const uint64_t*)state->fj_reset_ptr;
    for (uint32_t i = 0; i < state->fj_reset_count; ++i) {
      atomic_store_explicit((volatile global atomic_long*)rs[i], 1,
                            memory_order_release, memory_scope_device);
    }
    atomic_work_item_fence(CLK_GLOBAL_MEM_FENCE, memory_order_release, memory_scope_device);
  }

  __global const SubBlock* subs = (__global const SubBlock*)state->subblock_table_ptr;
  __global const uint64_t* qptrs = (__global const uint64_t*)state->lane_queue_ptrs;
  __global const uint64_t* dbs   = (__global const uint64_t*)state->lane_doorbell_ptrs;

  // Lane sub-blocks first (queue_index != 0) ...
  for (uint32_t i = 0; i < bd.subblock_count; ++i) {
    SubBlock sb = subs[bd.subblock_index + i];
    if (sb.queue_index == 0) continue;
    issue_block_to_queue(cmd_base + sb.packet_offset, sb.packet_count,
                         (__global uint8_t*)qptrs[sb.queue_index],
                         (volatile __global uint64_t*)dbs[sb.queue_index]);
  }
  // ... control sub-block(s) last (carries join barriers + terminator).
  for (uint32_t i = 0; i < bd.subblock_count; ++i) {
    SubBlock sb = subs[bd.subblock_index + i];
    if (sb.queue_index != 0) continue;
    issue_block_to_queue(cmd_base + sb.packet_offset, sb.packet_count,
                         (__global uint8_t*)state->queue_ptr,
                         (volatile __global uint64_t*)state->doorbell_ptr);
  }
}

// Cooperative issue of a single (control-queue, no sub-block) block: the
// whole workgroup copies the block's packets into the queue ring in parallel;
// only thread 0 reserves the slot range, publishes write_index and rings the
// doorbell. Sole producer of the control queue at this point, so a plain
// load + release store replaces the atomic RMW used by issue_block_to_queue.
// OpenCL C forbids __local variables inside non-kernel functions, so the
// scratch slots live in the caller's (kernel) frame and are passed in here
// by local-address-space pointer.
static inline void issue_block_coop(__global ExecutionState* state, uint32_t blk,
                                    uint32_t tid, uint32_t nt,
                                    __local uint64_t* s_wr,
                                    __local uint32_t* s_start_byte,
                                    __local uint32_t* s_first_bytes,
                                    __local uint32_t* s_second_bytes) {
  __global const BlockDescriptor* table = (__global const BlockDescriptor*)state->block_table_ptr;
  BlockDescriptor bd = table[blk];
  __global const uint8_t* src = (__global const uint8_t*)state->cmd_buffer_base + bd.packet_offset;
  uint32_t pkt_count = bd.packet_count;
  if (pkt_count == 0) return;

  __global uint8_t* qptr = (__global uint8_t*)state->queue_ptr;
  __global uint8_t* base = (__global uint8_t*)(*(__global uint64_t*)(qptr + QUEUE_BASE_ADDR_OFF));
  uint32_t q_size = *(__global uint32_t*)(qptr + QUEUE_SIZE_OFF);
  uint32_t q_mask = q_size - 1;
  volatile global atomic_ulong* wr_id = (volatile global atomic_ulong*)(qptr + QUEUE_WRITE_ID_OFF);
  volatile global atomic_ulong* rd_id = (volatile global atomic_ulong*)(qptr + QUEUE_READ_ID_OFF);

  if (tid == 0) {
    uint64_t wr     = atomic_load_explicit(wr_id, memory_order_relaxed, memory_scope_device);
    uint64_t new_wr = wr + pkt_count;
    while (new_wr - atomic_load_explicit(rd_id, memory_order_relaxed, memory_scope_device) > q_size) {}
    uint32_t start_slot = (uint32_t)(wr & q_mask);
    uint32_t sbw        = q_size - start_slot;
    uint32_t first      = (pkt_count <= sbw) ? pkt_count : sbw;
    *s_wr           = wr;
    *s_start_byte   = start_slot * 64;
    *s_first_bytes  = first * 64;
    *s_second_bytes = (pkt_count - first) * 64;
  }
  atomic_work_item_fence(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE,
                         memory_order_release, memory_scope_work_group);
  work_group_barrier(CLK_LOCAL_MEM_FENCE);
  atomic_work_item_fence(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE,
                         memory_order_acquire, memory_scope_work_group);

  copy_coop(base + *s_start_byte, src, *s_first_bytes, tid, nt);
  if (*s_second_bytes > 0) {
    copy_coop(base, src + *s_first_bytes, *s_second_bytes, tid, nt);
  }
  atomic_work_item_fence(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE,
                         memory_order_release, memory_scope_work_group);
  work_group_barrier(CLK_LOCAL_MEM_FENCE);

  if (tid != 0) return;
  atomic_work_item_fence(CLK_GLOBAL_MEM_FENCE, memory_order_release, memory_scope_device);
  atomic_store_explicit(wr_id, *s_wr + pkt_count, memory_order_release, memory_scope_device);
  *((volatile __global uint64_t*)state->doorbell_ptr) = *s_wr + pkt_count - 1;
}

// ============================================================================
// CFG: Fast-path conditional branch for WHILE self-loops.
// ============================================================================

__attribute__((amdgpu_flat_work_group_size(1, 64)))
__kernel void __amd_rocclr_graphCondBranchWhile(
    __global ExecutionState* state,
    __global const uint64_t* restrict ref_ptr,
    uint32_t cond,
    uint64_t value,
    __global const uint8_t* restrict body_src,
    uint32_t body_pkt_count) {

  const uint32_t tid = get_local_id(0);
  const uint32_t nt  = 64;

  uint64_t ref_val = *ref_ptr;
  if (!evaluate_cond(ref_val, cond, value)) {
    if (tid == 0) {
      signal_return(state);
    }
    return;
  }

  uint8_t*  base   = (uint8_t*)state->cached_queue_base;
  uint32_t  q_size = state->cached_queue_size;
  uint32_t  q_mask = q_size - 1;
  uint8_t*  qptr   = (uint8_t*)state->queue_ptr;
  volatile global atomic_ulong* wr_id = (volatile global atomic_ulong*)(qptr + QUEUE_WRITE_ID_OFF);
  volatile global atomic_ulong* rd_id = (volatile global atomic_ulong*)(qptr + QUEUE_READ_ID_OFF);

  __local uint64_t s_wr;
  __local uint32_t s_first_bytes;
  __local uint32_t s_second_bytes;
  __local uint32_t s_start_byte;

  if (tid == 0) {
    uint64_t wr     = atomic_load_explicit(wr_id, memory_order_relaxed, memory_scope_device);
    uint64_t new_wr = wr + body_pkt_count;
    while (new_wr - atomic_load_explicit(rd_id, memory_order_relaxed, memory_scope_device) > q_size) {}

    uint32_t start_slot        = (uint32_t)(wr & q_mask);
    uint32_t slots_before_wrap = q_size - start_slot;
    uint32_t first_count  = (body_pkt_count <= slots_before_wrap)
                              ? body_pkt_count : slots_before_wrap;
    s_wr           = wr;
    s_start_byte   = start_slot * 64;
    s_first_bytes  = first_count * 64;
    s_second_bytes = (body_pkt_count - first_count) * 64;
  }
  atomic_work_item_fence(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE,
                         memory_order_release, memory_scope_work_group);
  work_group_barrier(CLK_LOCAL_MEM_FENCE);
  atomic_work_item_fence(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE,
                         memory_order_acquire, memory_scope_work_group);

  copy_coop(base + s_start_byte, body_src, s_first_bytes, tid, nt);
  if (s_second_bytes > 0) {
    copy_coop(base, body_src + s_first_bytes, s_second_bytes, tid, nt);
  }
  atomic_work_item_fence(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE,
                         memory_order_release, memory_scope_work_group);
  work_group_barrier(CLK_LOCAL_MEM_FENCE);

  if (tid != 0) return;

  uint64_t wr     = s_wr;
  uint64_t new_wr = wr + body_pkt_count;
  atomic_work_item_fence(CLK_GLOBAL_MEM_FENCE, memory_order_release, memory_scope_device);
  atomic_store_explicit(wr_id, new_wr, memory_order_release, memory_scope_device);

  *((volatile global uint64_t*)state->doorbell_ptr) = wr + body_pkt_count - 1;
}

// ============================================================================
// Block-based scheduler: initial entry point dispatched from host.
// ============================================================================

__attribute__((amdgpu_flat_work_group_size(1, 64)))
__kernel void __amd_rocclr_graphBlockIssue(__global ExecutionState* state) {
  const uint32_t tid = get_local_id(0);
  const uint32_t nt  = 64;
  __local uint64_t s_wr;
  __local uint32_t s_start_byte, s_first_bytes, s_second_bytes;

  // Initialize conditional variable on device before first block dispatch.
  if (tid == 0) {
    if (state->cond_init_ptr) {
      *(volatile __global uint64_t*)state->cond_init_ptr = state->cond_init_value;
    }
    reset_cond_table(state);
  }

  __global const BlockDescriptor* table = (__global const BlockDescriptor*)state->block_table_ptr;
  if (table[state->current_block].subblock_count == 0) {
    issue_block_coop(state, state->current_block, tid, nt,
                     &s_wr, &s_start_byte, &s_first_bytes, &s_second_bytes);
  } else if (tid == 0) {
    issue_block(state, state->current_block);
  }
}

// ============================================================================
// CFG: Unconditional branch — set current_block and issue next block
// ============================================================================

__attribute__((amdgpu_flat_work_group_size(1, 1)))
__kernel void __amd_rocclr_graphBranch(
    __global ExecutionState* state,
    uint32_t target_block) {
  state->current_block = target_block;
  issue_block(state, target_block);
}

// ============================================================================
// CFG: Conditional branch — evaluate *ref_ptr <cond> value, pick block
// ============================================================================

__attribute__((amdgpu_flat_work_group_size(1, 1)))
__kernel void __amd_rocclr_graphCondBranch(
    __global ExecutionState* state,
    __global const uint64_t* restrict ref_ptr,
    uint32_t cond,
    uint64_t value,
    uint32_t true_block,
    uint32_t false_block) {
  uint64_t ref_val = *ref_ptr;
  uint32_t next = evaluate_cond(ref_val, cond, value) ? true_block : false_block;

  if (next == kInlineReturn) {
    signal_return(state);
    return;
  }

  state->current_block = next;
  issue_block(state, next);
}

// ============================================================================
// CFG: Switch branch — read integer *ref_ptr and jump to case_targets[value].
// Native O(1) SWITCH terminator (vs. an O(N) chain of graphCondBranch checks).
// ============================================================================

__attribute__((amdgpu_flat_work_group_size(1, 1)))
__kernel void __amd_rocclr_graphSwitchBranch(
    __global ExecutionState* state,
    __global const uint64_t* restrict ref_ptr,
    __global const uint32_t* restrict case_targets,
    uint32_t num_cases,
    uint32_t default_target) {
  uint64_t v = *ref_ptr;
  uint32_t next = (v < (uint64_t)num_cases) ? case_targets[v] : default_target;

  if (next == kInlineReturn) {
    signal_return(state);
    return;
  }

  state->current_block = next;
  issue_block(state, next);
}

// ============================================================================
// CFG: Return — signal completion, end the GPU-side chain
// ============================================================================

__attribute__((amdgpu_flat_work_group_size(1, 1)))
__kernel void __amd_rocclr_graphReturn(__global ExecutionState* state) {
  signal_return(state);
}

// ============================================================================
// Fused WHILE loop: persistent kernel that issues body packets, waits for
// completion by polling read_id, checks the condition, and loops. Eliminates
// 2 AQL dispatch round-trips (graphCondBranch + graphBranch) per iteration by
// keeping the control flow entirely within this single kernel.
// ============================================================================

__attribute__((amdgpu_flat_work_group_size(1, 1)))
__kernel void __amd_rocclr_graphWhileLoop(
    __global ExecutionState* state,
    volatile __global uint64_t* restrict cond_ptr,
    uint32_t body_block_idx) {

  __global const BlockDescriptor* table = (__global const BlockDescriptor*)state->block_table_ptr;
  __global const uint8_t* cmd_base = (__global const uint8_t*)state->cmd_buffer_base;

  // Body packets go to Queue B (work queue), NOT the control queue.
  __global uint8_t* wq = (__global uint8_t*)state->work_queue_ptr;
  volatile global uint64_t* w_doorbell = (volatile global uint64_t*)state->work_doorbell_ptr;

  __global uint8_t* q_base = (__global uint8_t*)(*(__global uint64_t*)(wq + QUEUE_BASE_ADDR_OFF));
  uint32_t q_size = *(__global uint32_t*)(wq + QUEUE_SIZE_OFF);
  uint32_t q_mask = q_size - 1;
  volatile global atomic_ulong* wr_id = (volatile global atomic_ulong*)(wq + QUEUE_WRITE_ID_OFF);
  volatile global atomic_ulong* rd_id = (volatile global atomic_ulong*)(wq + QUEUE_READ_ID_OFF);

  __global const uint8_t* body_src = cmd_base + table[body_block_idx].packet_offset;
  uint32_t body_pkt_count = table[body_block_idx].packet_count;

  // Initialize the condition variable to its default before the first test,
  // so the loop starts armed on every launch.
  if (state->cond_init_ptr) {
    *(volatile __global uint64_t*)state->cond_init_ptr = state->cond_init_value;
    reset_cond_table(state);
    atomic_work_item_fence(CLK_GLOBAL_MEM_FENCE, memory_order_release, memory_scope_device);
  }

  // ---------------------------------------------------------------------
  // Staged fused-WHILE (experimental, state->staged_while != 0). See
  // graph_scheduler_kernel.hip for the full rationale: pre-stage N INVALID
  // body packets and ring the doorbell once, then flip headers one at a time
  // to release each iteration without a per-iteration doorbell/copy.
  // ---------------------------------------------------------------------
  if (state->staged_while != 0 && body_pkt_count == 1) {
    uint32_t N = state->staged_while;
    if (N > q_size - 1) N = q_size - 1;
    if (N == 0) N = 1;

    const uint32_t real_header = *(volatile __global const uint32_t*)body_src;

    bool finished = false;
    while (!finished && *cond_ptr != 0) {
      uint64_t wr = atomic_load_explicit(wr_id, memory_order_relaxed, memory_scope_device);
      while (wr + N - atomic_load_explicit(rd_id, memory_order_relaxed, memory_scope_device) > q_size) {
        __builtin_amdgcn_s_sleep(1);
      }

      // Stage N slots: header INVALID first, then the body payload.
      for (uint32_t i = 0; i < N; ++i) {
        __global uint8_t* slot = q_base + ((uint32_t)((wr + i) & q_mask)) * 64;
        *(volatile __global uint32_t*)slot = AQL_HEADER_INVALID;
        __builtin_memcpy(slot + 4, body_src + 4, 60);
      }
      atomic_work_item_fence(CLK_GLOBAL_MEM_FENCE, memory_order_release, memory_scope_device);

      atomic_store_explicit(wr_id, wr + N, memory_order_release, memory_scope_device);
      if (w_doorbell) {
        *w_doorbell = wr + N - 1;
      }

      // Release the staged bodies one at a time, gated on the condition.
      uint32_t consumed = 0;
      for (uint32_t i = 0; i < N; ++i) {
        if (*cond_ptr == 0) break;

        *cond_ptr = COND_SENTINEL;
        atomic_work_item_fence(CLK_GLOBAL_MEM_FENCE, memory_order_release, memory_scope_device);

        __global uint8_t* slot = q_base + ((uint32_t)((wr + i) & q_mask)) * 64;
        atomic_store_explicit((volatile global atomic_uint*)slot, real_header,
                              memory_order_release, memory_scope_device);

        while (*cond_ptr == COND_SENTINEL) {
          __builtin_amdgcn_s_sleep(1);
        }
        atomic_work_item_fence(CLK_GLOBAL_MEM_FENCE, memory_order_acquire, memory_scope_device);
        consumed = i + 1;
        if (*cond_ptr == 0) break;
      }

      // Drain any staged-but-unreleased slots with no-op barrier packets.
      if (consumed < N) {
        for (uint32_t j = consumed; j < N; ++j) {
          __global uint8_t* slot = q_base + ((uint32_t)((wr + j) & q_mask)) * 64;
          for (uint32_t w = 1; w < 16; ++w) {
            ((volatile __global uint32_t*)slot)[w] = 0;
          }
          atomic_work_item_fence(CLK_GLOBAL_MEM_FENCE, memory_order_release, memory_scope_device);
          atomic_store_explicit((volatile global atomic_uint*)slot, AQL_HEADER_BARRIER_NOOP,
                                memory_order_release, memory_scope_device);
        }
        if (w_doorbell) {
          *w_doorbell = wr + N - 1;
        }
        finished = true;
      }
    }

    signal_return(state);
    return;
  }

  // Sole producer on the work queue: track write index in a register and
  // cache the read index, re-reading rd_id only when the cached bound says
  // the ring might actually be full.
  uint64_t local_wr  = atomic_load_explicit(wr_id, memory_order_relaxed, memory_scope_device);
  uint64_t cached_rd = atomic_load_explicit(rd_id, memory_order_relaxed, memory_scope_device);

  while (*cond_ptr != 0) {
    // Set sentinel before issuing body; body kernel overwrites with 0/1.
    *cond_ptr = COND_SENTINEL;
    atomic_work_item_fence(CLK_GLOBAL_MEM_FENCE, memory_order_release, memory_scope_device);

    uint64_t wr     = local_wr;
    uint64_t new_wr = wr + body_pkt_count;

    if (new_wr - cached_rd > q_size) {
      do {
        __builtin_amdgcn_s_sleep(1);
        cached_rd = atomic_load_explicit(rd_id, memory_order_relaxed, memory_scope_device);
      } while (new_wr - cached_rd > q_size);
    }
    local_wr = new_wr;

    uint32_t start_slot        = (uint32_t)(wr & q_mask);
    uint32_t slots_before_wrap = q_size - start_slot;
    uint32_t first_count  = (body_pkt_count <= slots_before_wrap)
                              ? body_pkt_count : slots_before_wrap;
    uint32_t second_count = body_pkt_count - first_count;

    // Copy the full packet(s) INCLUDING the header; write_index only
    // advances (below) after the copy, so the CP cannot observe the slot
    // until the packet is fully formed.
    __builtin_memcpy(q_base + start_slot * 64, body_src, first_count * 64);
    if (second_count > 0) {
      __builtin_memcpy(q_base, body_src + first_count * 64, second_count * 64);
    }

    atomic_work_item_fence(CLK_GLOBAL_MEM_FENCE, memory_order_release, memory_scope_device);
    atomic_store_explicit(wr_id, new_wr, memory_order_release, memory_scope_device);

    if (w_doorbell) {
      *w_doorbell = wr + body_pkt_count - 1;
    }

    // Tight-poll cond_ptr. Body kernel writes 0 or 1 as its last op.
    while (*cond_ptr == COND_SENTINEL) {
      __builtin_amdgcn_s_sleep(1);
    }

    atomic_work_item_fence(CLK_GLOBAL_MEM_FENCE, memory_order_acquire, memory_scope_device);
  }

  signal_return(state);
}

// ============================================================================
// CFG-walking interpreter: one persistent kernel that walks the whole flattened
// block CFG, issuing each block's packets to the work queue and evaluating
// terminators itself. Generalizes graphWhileLoop (which hardcodes a single
// WHILE around a single body block) to arbitrary nested WHILE/IF/SWITCH, and
// replaces the standard path's per-terminator dispatch round trip.
//
// One workgroup (== one CU, same residency as the 1-thread graphWhileLoop,
// which already reserves a full wavefront): thread 0 owns the ring accounting
// and every control-flow decision, all 64 threads cooperate on the packet copy.
//
// Sync model: a block whose completion can still be pending when a condition is
// read carries a trailing barrier packet that decrements a shared signal. The
// walk drains that signal before evaluating ANY condition -- including from an
// empty block such as a WHILE latch, whose condition was written by the body
// block issued one step earlier. The host only arms blocks that need it, so the
// armed block is always drained before anything can re-arm.
//
// Exit is RETURN-triggered, never condition-triggered: a condition going false
// is an ordinary edge. Only TERM_RETURN ends the walk.
// ============================================================================

__attribute__((amdgpu_flat_work_group_size(1, 64)))
__kernel void __amd_rocclr_graphWalkLoop(__global ExecutionState* state) {
  const uint32_t tid = get_local_id(0);
  const uint32_t nt  = 64;

  __global const BlockDescriptor* table = (__global const BlockDescriptor*)state->block_table_ptr;
  __global const BlockTerminator* terms = (__global const BlockTerminator*)state->term_table_ptr;
  __global const uint8_t* cmd_base = (__global const uint8_t*)state->cmd_buffer_base;
  __global uint8_t* wq = (__global uint8_t*)state->work_queue_ptr;

  // Both are set by the launch path for this path; bail out rather than fault.
  if (terms == 0 || wq == 0) {
    if (tid == 0) signal_return(state);
    return;
  }

  volatile global uint64_t* w_doorbell = (volatile global uint64_t*)state->work_doorbell_ptr;
  __global uint8_t* q_base = (__global uint8_t*)(*(__global uint64_t*)(wq + QUEUE_BASE_ADDR_OFF));
  uint32_t q_size = *(__global uint32_t*)(wq + QUEUE_SIZE_OFF);
  uint32_t q_mask = q_size - 1;
  volatile global atomic_ulong* wr_id = (volatile global atomic_ulong*)(wq + QUEUE_WRITE_ID_OFF);
  volatile global atomic_ulong* rd_id = (volatile global atomic_ulong*)(wq + QUEUE_READ_ID_OFF);

  // Shared: thread 0 publishes, every thread consumes.
  __local uint32_t s_cur;
  __local uint32_t s_start_byte, s_first_bytes, s_second_bytes;

  // Private; only thread 0's copy is meaningful. Sole producer on the work
  // queue, so the write index lives in a register and the read index is a
  // cached lower bound re-read only when the ring looks full.
  uint64_t local_wr  = 0;
  uint64_t cached_rd = 0;
  uint64_t pending   = 0;  // signal value addr of the most recent armed block

  if (tid == 0) {
    // Arm every conditional handle for this launch before the first test.
    if (state->cond_init_ptr) {
      *(volatile __global uint64_t*)state->cond_init_ptr = state->cond_init_value;
    }
    reset_cond_table(state);
    atomic_work_item_fence(CLK_GLOBAL_MEM_FENCE, memory_order_release, memory_scope_device);

    local_wr  = atomic_load_explicit(wr_id, memory_order_relaxed, memory_scope_device);
    cached_rd = atomic_load_explicit(rd_id, memory_order_relaxed, memory_scope_device);
    s_cur = state->entry_block;
  }
  work_group_barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

  for (;;) {
    // Uniform across the workgroup: same block index, same table reads.
    uint32_t cur = s_cur;
    BlockTerminator t = terms[cur];
    uint32_t pkt_count = table[cur].packet_count;

    if (pkt_count != 0) {
      __global const uint8_t* src = cmd_base + table[cur].packet_offset;

      if (tid == 0) {
        // Re-arm before issuing: a block inside a loop is issued many times,
        // and the previous iteration's barrier already drove this to 0.
        if (t.signal_addr) {
          atomic_store_explicit((volatile global atomic_long*)t.signal_addr, 1,
                                memory_order_release, memory_scope_device);
        }

        uint64_t wr     = local_wr;
        uint64_t new_wr = wr + pkt_count;
        // cached_rd <= true rd (rd only advances), so passing with the cached
        // value is conservative; only re-read when it fails.
        if (new_wr - cached_rd > q_size) {
          do {
            __builtin_amdgcn_s_sleep(1);
            cached_rd = atomic_load_explicit(rd_id, memory_order_relaxed, memory_scope_device);
          } while (new_wr - cached_rd > q_size);
        }

        uint32_t start_slot        = (uint32_t)(wr & q_mask);
        uint32_t slots_before_wrap = q_size - start_slot;
        uint32_t first_count  = (pkt_count <= slots_before_wrap) ? pkt_count : slots_before_wrap;
        s_start_byte   = start_slot * 64;
        s_first_bytes  = first_count * 64;
        s_second_bytes = (pkt_count - first_count) * 64;
      }
      atomic_work_item_fence(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE,
                             memory_order_release, memory_scope_work_group);
      work_group_barrier(CLK_LOCAL_MEM_FENCE);
      atomic_work_item_fence(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE,
                             memory_order_acquire, memory_scope_work_group);

      // Copy full packets INCLUDING headers: write_index only advances after
      // the copy, so the CP cannot observe a partially-formed slot.
      copy_coop(q_base + s_start_byte, src, s_first_bytes, tid, nt);
      if (s_second_bytes > 0) {
        copy_coop(q_base, src + s_first_bytes, s_second_bytes, tid, nt);
      }
      atomic_work_item_fence(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE,
                             memory_order_release, memory_scope_work_group);
      work_group_barrier(CLK_LOCAL_MEM_FENCE);

      if (tid == 0) {
        uint64_t wr     = local_wr;
        uint64_t new_wr = wr + pkt_count;
        atomic_work_item_fence(CLK_GLOBAL_MEM_FENCE, memory_order_release, memory_scope_device);
        atomic_store_explicit(wr_id, new_wr, memory_order_release, memory_scope_device);
        if (w_doorbell) {
          *w_doorbell = new_wr - 1;
        }
        local_wr = new_wr;
        if (t.signal_addr) {
          pending = t.signal_addr;
        }
      }
    }

    // Terminator: thread 0 decides and publishes the next block.
    if (tid == 0) {
      uint32_t next = kInlineReturn;
      if (t.kind == TERM_BRANCH) {
        next = t.branch_target;
      } else if (t.kind == TERM_COND_BRANCH || t.kind == TERM_SWITCH) {
        // A condition is only readable once the block that writes it has
        // retired. Drain the outstanding block, if any, then clear it.
        if (t.flags & TERM_FLAG_SENTINEL_WAIT) {
          // Watch the condition itself. Falls through immediately when this
          // latch was not armed (first visit of a fresh launch).
          while (*(volatile __global const uint64_t*)t.cond_ptr == COND_SENTINEL) {
            __builtin_amdgcn_s_sleep(1);
          }
        } else if (pending) {
          while (atomic_load_explicit((volatile global atomic_long*)pending,
                                      memory_order_relaxed, memory_scope_device) > 0) {
            __builtin_amdgcn_s_sleep(1);
          }
          pending = 0;
        }
        atomic_work_item_fence(CLK_GLOBAL_MEM_FENCE, memory_order_acquire, memory_scope_device);

        uint64_t v = *(volatile __global const uint64_t*)t.cond_ptr;
        if (t.kind == TERM_COND_BRANCH) {
          next = evaluate_cond(v, t.cond_op, t.cond_value) ? t.true_target : t.false_target;
        } else {
          __global const uint32_t* cases = (__global const uint32_t*)t.case_targets_ptr;
          next = (v < (uint64_t)t.num_cases) ? cases[v] : t.default_target;
        }

        // Re-entering the loop body: stamp the sentinel so the latch closing
        // this iteration can tell the body's store apart from the value just
        // read. Must happen before the body's packets are issued, which is why
        // it lives here and not on the issuing block.
        if ((t.flags & TERM_FLAG_SENTINEL_ARM) && next == t.true_target) {
          *(volatile __global uint64_t*)t.cond_ptr = COND_SENTINEL;
          atomic_work_item_fence(CLK_GLOBAL_MEM_FENCE, memory_order_release, memory_scope_device);
        }
      }
      s_cur = next;
    }
    atomic_work_item_fence(CLK_LOCAL_MEM_FENCE, memory_order_release, memory_scope_work_group);
    work_group_barrier(CLK_LOCAL_MEM_FENCE);
    atomic_work_item_fence(CLK_LOCAL_MEM_FENCE, memory_order_acquire, memory_scope_work_group);

    if (s_cur == kInlineReturn) break;
  }

  if (tid == 0) {
    signal_return(state);
  }
}
)CLSRC";

const char* BlitImageSourceCode = BLIT_KERNELS(
    // Extern
    extern void __amd_fillImage(__write_only image2d_array_t, float4, int4, uint4, int4, int4,
                                uint);

    extern void __amd_copyImage(__read_only image2d_array_t, __write_only image2d_array_t, int4,
                                int4, int4);

    extern void __amd_copyImage1DA(__read_only image2d_array_t, __write_only image2d_array_t, int4,
                                   int4, int4);

    extern void __amd_copyBufferToImage(__global uint*, __write_only image2d_array_t, ulong4, int4,
                                        int4, uint4, ulong4);

    extern void __amd_copyImageToBuffer(__read_only image2d_array_t, __global uint*,
                                        __global ushort*, __global uchar*, int4, ulong4, int4,
                                        uint4, ulong4);

    __kernel void __amd_rocclr_fillImage(__write_only image2d_array_t image, float4 patternFLOAT4,
                                         int4 patternINT4, uint4 patternUINT4, int4 origin,
                                         int4 size, uint type) {
      __amd_fillImage(image, patternFLOAT4, patternINT4, patternUINT4, origin, size, type);
    }

    __kernel void __amd_rocclr_copyImage(
        __read_only image2d_array_t src, __write_only image2d_array_t dst, int4 srcOrigin,
        int4 dstOrigin, int4 size) { __amd_copyImage(src, dst, srcOrigin, dstOrigin, size); }

    __kernel void __amd_rocclr_copyImage1DA(
        __read_only image2d_array_t src, __write_only image2d_array_t dst, int4 srcOrigin,
        int4 dstOrigin, int4 size) { __amd_copyImage1DA(src, dst, srcOrigin, dstOrigin, size); }

    __kernel void __amd_rocclr_copyBufferToImage(
        __global uint* src, __write_only image2d_array_t dst, ulong4 srcOrigin, int4 dstOrigin,
        int4 size, uint4 format, ulong4 pitch) {
      __amd_copyBufferToImage(src, dst, srcOrigin, dstOrigin, size, format, pitch);
    }

    __kernel void __amd_rocclr_copyImageToBuffer(
        __read_only image2d_array_t src, __global uint* dstUInt, __global ushort* dstUShort,
        __global uchar* dstUChar, int4 srcOrigin, ulong4 dstOrigin, int4 size, uint4 format,
        ulong4 pitch) {
      __amd_copyImageToBuffer(src, dstUInt, dstUShort, dstUChar, srcOrigin, dstOrigin, size, format,
                              pitch);
    });

}  // namespace amd::device
