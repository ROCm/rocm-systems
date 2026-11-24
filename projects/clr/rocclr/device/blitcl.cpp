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

    // copy_mode is the element size (16 or 4) for a cached copy. Above 16 it selects a 16-byte
    // copy with nontemporal stores (17) or nontemporal loads and stores (18). The range check
    // keeps the compiler from turning the selection into a switch, so the cached 16-byte loop
    // stays first. Each mode has its own loop: with a shared loop the compiler merges the plain
    // and nontemporal accesses and drops the hint.
    __kernel void __amd_rocclr_copyBuffer(__global uchar* src, __global uchar* dst, ulong size,
                                          uint remainder, uint copy_mode, ulong end_ptr,
                                          uint next_chunk, uint workgroup_size) {
      uint l = __builtin_amdgcn_workitem_id_x();
      uint g = __builtin_amdgcn_workgroup_id_x();
      ulong id = (g * workgroup_size + l);
      ulong id_remainder = id;

      if (copy_mode == sizeof(ulong2)) {
        __global ulong2* srcD = (__global ulong2*)(src);
        __global ulong2* dstD = (__global ulong2*)(dst);
        while ((ulong)(&dstD[id]) < end_ptr) {
          dstD[id] = srcD[id];
          id += next_chunk;
        }
      } else if (copy_mode > sizeof(ulong2)) {
        __global ulong2* srcD = (__global ulong2*)(src);
        __global ulong2* dstD = (__global ulong2*)(dst);
        if (copy_mode == sizeof(ulong2) + 1) {
          while ((ulong)(&dstD[id]) < end_ptr) {
            __builtin_nontemporal_store(srcD[id], &dstD[id]);
            id += next_chunk;
          }
        } else {
          while ((ulong)(&dstD[id]) < end_ptr) {
            __builtin_nontemporal_store(__builtin_nontemporal_load(&srcD[id]), &dstD[id]);
            id += next_chunk;
          }
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

    // The rect kernels below give each work-item one 16-byte chunk of a destination row,
    // counted from the row's first byte rounded down to 16. A chunk inside the row is one
    // aligned 16-byte store; the partial chunks at either end of the row store less. The first
    // grid dimension runs over the chunks of every row of a slice, the second over slices.
    // Rects are {row pitch, slice pitch, start, 0} in bytes, and size is {row bytes, rows,
    // slices, chunks per row}.
    typedef uchar16 __attribute__((aligned(1))) unaligned_uchar16;

    // Moves the bytes of a little-endian 16-byte value up by byte_count, 1 to 15
    ulong2 __amd_rocclr_bytesUp(ulong2 value, uint byte_count) {
      uint bits = 8 * byte_count;
      if (bits >= 64) {
        return (ulong2)(0, value.s0 << (bits - 64));
      }
      return (ulong2)(value.s0 << bits, (value.s1 << bits) | (value.s0 >> (64 - bits)));
    }

    // Moves the bytes of a little-endian 16-byte value down by byte_count, 1 to 15
    ulong2 __amd_rocclr_bytesDown(ulong2 value, uint byte_count) {
      uint bits = 8 * byte_count;
      if (bits >= 64) {
        return (ulong2)(value.s1 >> (bits - 64), 0);
      }
      return (ulong2)((value.s0 >> bits) | (value.s1 << (64 - bits)), value.s1 >> bits);
    }

    // Stores bytes position to position + store_size - 1 of a chunk-aligned 16-byte value
    void __amd_rocclr_storeChunkPiece(ulong chunk_start, ulong2 value, uint position,
                                      uint store_size) {
      ulong word = ((position < 8) ? value.s0 : value.s1) >> (8 * (position & 7));
      if (store_size == 1) {
        *(__global uchar*)(chunk_start + position) = (uchar)word;
      } else if (store_size == 2) {
        *(__global ushort*)(chunk_start + position) = (ushort)word;
      } else if (store_size == 4) {
        *(__global uint*)(chunk_start + position) = (uint)word;
      } else {
        *(__global ulong*)(chunk_start + position) = word;
      }
    }

    // Stores bytes first to last - 1 of a chunk-aligned 16-byte value at chunk_start with
    // naturally aligned stores: growing ones up to an 8-byte boundary, then shrinking ones
    void __amd_rocclr_storeChunkBytes(ulong chunk_start, ulong2 value, uint first, uint last) {
      uint position = first;
      for (uint store_size = 1; store_size <= 8; store_size *= 2) {
        if (((position & store_size) != 0) && (position + store_size <= last)) {
          __amd_rocclr_storeChunkPiece(chunk_start, value, position, store_size);
          position += store_size;
        }
      }
      for (uint store_size = 8; store_size > 0; store_size /= 2) {
        if (position + store_size <= last) {
          __amd_rocclr_storeChunkPiece(chunk_start, value, position, store_size);
          position += store_size;
        }
      }
    }

    // Rows of at least 16 bytes with any alignment. Each chunk loads the 16 source bytes of the
    // row nearest to it, unaligned (HSA enables unaligned global access), so the partial chunks
    // at either end of the row also wait for a single load. Stores the partial chunks; returns
    // true with the source and destination of a chunk inside the row, which the kernels copy
    // with their own cache policy. Each policy is its own kernel: the compiler merges accesses
    // that differ only in the nontemporal hint and drops it.
    bool __amd_rocclr_copyRectChunk(__global uchar* src, __global uchar* dst, ulong4 srcRect,
                                    ulong4 dstRect, ulong4 size,
                                    __global const unaligned_uchar16** source,
                                    __global ulong2** target) {
      uint chunks_per_row = (uint)size.w;
      uint slice_chunk = (uint)get_global_id(0);
      uint row = slice_chunk / chunks_per_row;
      ulong slice = get_global_id(1);
      if ((row >= size.y) || (slice >= size.z)) {
        return false;
      }
      uint chunk = slice_chunk - row * chunks_per_row;
      ulong src_row = (ulong)src + srcRect.z + slice * srcRect.y + row * srcRect.x;
      ulong dst_row = (ulong)dst + dstRect.z + slice * dstRect.y + row * dstRect.x;
      ulong dst_end = dst_row + size.x;
      ulong chunk_start = (dst_row & ~15UL) + (ulong)chunk * 16;
      if (chunk_start >= dst_end) {
        return false;
      }
      ulong load_offset = min(max(chunk_start, dst_row), dst_end - 16) - dst_row;
      *source = (__global const unaligned_uchar16*)(src_row + load_offset);
      if (chunk_start < dst_row) {
        uint first = (uint)(dst_row - chunk_start);
        __amd_rocclr_storeChunkBytes(chunk_start, __amd_rocclr_bytesUp(as_ulong2(**source), first),
                                     first, 16);
        return false;
      }
      if (chunk_start + 16 > dst_end) {
        __amd_rocclr_storeChunkBytes(
            chunk_start,
            __amd_rocclr_bytesDown(as_ulong2(**source),
                                   (uint)(chunk_start - dst_row - load_offset)),
            0, (uint)(dst_end - chunk_start));
        return false;
      }
      *target = (__global ulong2*)chunk_start;
      return true;
    }

    __kernel void __amd_rocclr_copyBufferRectUnaligned(__global uchar* src, __global uchar* dst,
                                                       ulong4 srcRect, ulong4 dstRect,
                                                       ulong4 size) {
      __global const unaligned_uchar16* source;
      __global ulong2* target;
      if (__amd_rocclr_copyRectChunk(src, dst, srcRect, dstRect, size, &source, &target)) {
        *target = as_ulong2(*source);
      }
    }

    // Nontemporal loads and stores, for a copy at least as large as the MALL, where neither the
    // source nor the destination stays resident
    __kernel void __amd_rocclr_copyBufferRectStream(__global uchar* src, __global uchar* dst,
                                                    ulong4 srcRect, ulong4 dstRect, ulong4 size) {
      __global const unaligned_uchar16* source;
      __global ulong2* target;
      if (__amd_rocclr_copyRectChunk(src, dst, srcRect, dstRect, size, &source, &target)) {
        __builtin_nontemporal_store(as_ulong2(__builtin_nontemporal_load(source)), target);
      }
    }

    // pattern holds the pattern repeated to 16 bytes, and dstRect.w the pattern size, a power of
    // two up to 16. The pattern restarts at every row, so with the pattern rotated by the row's
    // first address modulo its size, the byte at address A takes byte A % 16 of it.
    __kernel void __amd_rocclr_fillBufferRect(__global uchar* dst, ulong2 pattern,
                                              ulong4 dstRect, ulong4 size) {
      uint chunks_per_row = (uint)size.w;
      uint slice_chunk = (uint)get_global_id(0);
      uint row = slice_chunk / chunks_per_row;
      ulong slice = get_global_id(1);
      if ((row >= size.y) || (slice >= size.z)) {
        return;
      }
      uint chunk = slice_chunk - row * chunks_per_row;
      ulong dst_row = (ulong)dst + dstRect.z + slice * dstRect.y + row * dstRect.x;
      ulong dst_end = dst_row + size.x;
      ulong chunk_start = (dst_row & ~15UL) + (ulong)chunk * 16;
      if (chunk_start >= dst_end) {
        return;
      }
      uint rotation = (uint)(dst_row & (dstRect.w - 1));
      ulong2 row_pattern = (rotation == 0) ? pattern
                                           : (__amd_rocclr_bytesUp(pattern, rotation) |
                                              __amd_rocclr_bytesDown(pattern, 16 - rotation));
      if ((chunk_start >= dst_row) && (chunk_start + 16 <= dst_end)) {
        *(__global ulong2*)chunk_start = row_pattern;
        return;
      }
      __amd_rocclr_storeChunkBytes(chunk_start, row_pattern,
                                   (uint)(max(dst_row, chunk_start) - chunk_start),
                                   (uint)(min(dst_end, chunk_start + 16) - chunk_start));
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
