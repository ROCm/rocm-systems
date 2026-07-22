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

    // Single unified 1D swap dispatch (SoA arguments). Work is distributed
    // globally across all copies proportional to each copy's body-op count.
    // body_prefix[] is a dense exclusive prefix sum of per-copy body-op counts
    // with length n_batches+1; body_prefix[n_batches] == total_ops.
    __kernel void __amd_rocclr_swapBufferBatch(
        __global const ulong *srcs,        // base src address per copy
        __global const ulong *dsts,        // base dst address per copy
        __global const ulong *body_prefix, // exclusive prefix of body-op counts
        __global const uchar *op_size,     // per-copy transfer width: 1,2,4,8,16
        __global const uchar *head_size,   // leading bytes to reach body alignment
        __global const uchar *tail_size,   // trailing bytes after body
        ulong n_batches,
        ulong total_ops,                   // == body_prefix[n_batches]
        ulong copy_stride) {
      ulong gid = get_global_id(0);

      // Body loop: strided walk over the global body-op index space. Each global
      // op index g maps to (copy c, local element) via a binary search over the
      // prefix-sum array, with the last-found copy index cached across strides.
      ulong c = 0;
      // Per-batch metadata cache. All of these are loop-invariant per copy, so
      // they are loaded once per batch change and held in registers across the
      // strided iterations that land in the same copy. Sentinel bounds
      // (lo_bound=1, hi_bound=0) form an empty range so the first iteration
      // always misses and performs the initial refresh; this also avoids
      // speculative loads when the loop body never executes (gid >= total_ops).
      ulong lo_bound = 1;  // cached body_prefix[c]
      ulong hi_bound = 0;  // cached body_prefix[c + 1]
      ulong src_c = 0;     // cached srcs[c]
      ulong dst_c = 0;     // cached dsts[c]
      uchar s = 0;         // cached op_size[c]
      uchar head_c = 0;    // cached head_size[c]
      for (ulong g = gid; g < total_ops; g += copy_stride) {
        // Resolve copy index c = largest index with body_prefix[c] <= g, and
        // refresh the metadata cache, only when g leaves the cached copy range.
        if (!(g >= lo_bound && g < hi_bound)) {
          ulong lo = 0;
          ulong hi = n_batches;  // upper_bound over body_prefix[0..n_batches]
          while (lo < hi) {
            ulong mid = lo + ((hi - lo) >> 1);
            if (body_prefix[mid] <= g) {
              lo = mid + 1;
            } else {
              hi = mid;
            }
          }
          c = lo - 1;  // upper_bound - 1
          lo_bound = body_prefix[c];
          hi_bound = body_prefix[c + 1];
          src_c = srcs[c];
          dst_c = dsts[c];
          s = op_size[c];
          head_c = head_size[c];
        }

        ulong elem = g - lo_bound;
        __global uchar *sp =
            (__global uchar *)(src_c + head_c) + elem * (ulong)s;
        __global uchar *dp =
            (__global uchar *)(dst_c + head_c) + elem * (ulong)s;

        // Swap one element of width s. Addresses are aligned to s by the CPU
        // side head computation.
        if (s == 16) {
          __global ulong2 *a = (__global ulong2 *)sp;
          __global ulong2 *b = (__global ulong2 *)dp;
          ulong2 tmp = *a;
          *a = *b;
          *b = tmp;
        } else if (s == 8) {
          __global ulong *a = (__global ulong *)sp;
          __global ulong *b = (__global ulong *)dp;
          ulong tmp = *a;
          *a = *b;
          *b = tmp;
        } else if (s == 4) {
          __global uint *a = (__global uint *)sp;
          __global uint *b = (__global uint *)dp;
          uint tmp = *a;
          *a = *b;
          *b = tmp;
        } else if (s == 2) {
          __global ushort *a = (__global ushort *)sp;
          __global ushort *b = (__global ushort *)dp;
          ushort tmp = *a;
          *a = *b;
          *b = tmp;
        } else {
          uchar tmp = *sp;
          *sp = *dp;
          *dp = tmp;
        }
      }

      // Head / tail: handled only by the first workgroup. Thread t owns copies
      // t, t+blockDim, ... and swaps their leading head_size bytes and trailing
      // tail_size bytes. Regions are disjoint from the body and from each other.
      if (get_group_id(0) == 0) {
        ulong t = get_local_id(0);
        ulong block_dim = get_local_size(0);
        for (ulong ci = t; ci < n_batches; ci += block_dim) {
          __global uchar *src_base = (__global uchar *)srcs[ci];
          __global uchar *dst_base = (__global uchar *)dsts[ci];
          uchar head = head_size[ci];
          uchar tail = tail_size[ci];

          // Head bytes: [0, head).
          for (ulong i = 0; i < head; ++i) {
            uchar tmp = src_base[i];
            src_base[i] = dst_base[i];
            dst_base[i] = tmp;
          }

          // Tail bytes: [tail_start, tail_start + tail), after head + body.
          ulong body_count = body_prefix[ci + 1] - body_prefix[ci];
          ulong tail_start = (ulong)head + body_count * (ulong)op_size[ci];
          for (ulong i = 0; i < tail; ++i) {
            uchar tmp = src_base[tail_start + i];
            src_base[tail_start + i] = dst_base[tail_start + i];
            dst_base[tail_start + i] = tmp;
          }
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

    __kernel void __amd_rocclr_batchMemOp(__global void* params, uint count) {
      __amd_batchMemOp(params, count);
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

    __kernel void __amd_rocclr_gwsInit(uint value) { __builtin_amdgcn_ds_gws_init(value, 0); });

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
    });

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
