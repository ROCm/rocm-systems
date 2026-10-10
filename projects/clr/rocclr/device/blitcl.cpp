/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

namespace amd::device {

#define BLIT_KERNELS(...) #__VA_ARGS__

// Blits Moved from device libs to clr
// Requires CL2.0 standard

const char* BlitLinearSourceCode = BLIT_KERNELS(
    extern void __ockl_dm_init_v1(ulong, ulong, uint, uint);

    typedef enum BatchMemOpType {
      STREAM_WAIT_VALUE_32 = 0x1,
      STREAM_WRITE_VALUE_32 = 0x2,
      STREAM_WAIT_VALUE_64 = 0x4,
      STREAM_WRITE_VALUE_64 = 0x5,
      STREAM_MEM_OP_BARRIER = 0x6,            // Currently not supported
      STREAM_MEM_OP_FLUSH_REMOTE_WRITES = 0x3 // Currently not supported
    } BatchMemOpType;

    typedef union streamBatchMemOpParams_union {
      BatchMemOpType operation;
      struct streamMemOpWaitValueParams_t{
        BatchMemOpType operation;
        atomic_ulong* address;
        union {
          uint value;
          ulong value64;
        };
        uint flags;
        atomic_ulong* alias; // Not valid for AMD backend
      } waitValue;
      struct streamMemOpWriteValueParams_t{
        BatchMemOpType operation;
        atomic_ulong* address;
        union {
          uint value;
          ulong value64;
        };
        uint flags;
        atomic_ulong* alias; // Not valid for AMD backend
      } writeValue;
      struct streamMemOpFlushRemoteWritesParams_t{ // Currently not supported
        BatchMemOpType operation;
        uint flags;
      } flushRemoteWrites;
      struct streamMemOpMemoryBarrierParams_t{ // Currently not supported
        BatchMemOpType operation;
        uint flags;
      } memoryBarrier;
      ulong pad[6];
    } BatchMemOpParams;

    typedef struct CopyBufferBatchDescriptor {
      ulong source_address;
      ulong destination_address;
      ulong aligned_element_count;
      uint aligned_element_size;
      uint trailing_byte_count;
    } CopyBufferBatchDescriptor;

    __attribute__((always_inline)) static void __amd_fillBufferAligned2D(__global uchar* bufUChar,
                              __global ushort* bufUShort,
                              __global uint* bufUInt,
                              __global ulong* bufULong,
                              __constant uchar* pattern,
                              uint patternSize,
                              ulong origin,
                              ulong width,
                              ulong height,
                              ulong pitch)
    {
      ulong tid_x = get_global_id(0);
      ulong tid_y = get_global_id(1);

      if (tid_x >= width || tid_y >= height) {
        return;
      }

      ulong offset = (tid_y * pitch + tid_x);

      if (bufULong) {
        __global ulong* element = &bufULong[origin + offset];
        __constant ulong* pt = (__constant ulong*)pattern;
        for (uint i = 0; i < patternSize; ++i) {
          element[i] = pt[i];
        }
      } else if (bufUInt) {
        __global uint* element = &bufUInt[origin + offset];
        __constant uint* pt = (__constant uint*)pattern;
        for (uint i = 0; i < patternSize; ++i) {
          element[i] = pt[i];
        }
      } else if (bufUShort) {
        __global ushort* element = &bufUShort[origin + offset];
        __constant ushort* pt = (__constant ushort*)pattern;
        for (uint i = 0; i < patternSize; ++i) {
          element[i] = pt[i];
        }
      } else if (bufUChar) {
        __global uchar* element = &bufUChar[origin + offset];
        __constant uchar* pt = (__constant uchar*)pattern;
        for (uint i = 0; i < patternSize; ++i) {
          element[i] = pt[i];
        }
      }
    }

    __attribute__((always_inline)) static void __amd_copyBufferAligned(
        __global uint* src,
        __global uint* dst,
        ulong srcOrigin,
        ulong dstOrigin,
        ulong size,
        uint alignment)
    {
        ulong id = get_global_id(0);

        if (id >= size) {
            return;
        }

        ulong   offsSrc = id + srcOrigin;
        ulong   offsDst = id + dstOrigin;

        if (alignment == 16) {
            __global uint4* src4 = (__global uint4*)src;
            __global uint4* dst4 = (__global uint4*)dst;
            dst4[offsDst] = src4[offsSrc];
        }
        else {
            dst[offsDst] = src[offsSrc];
        }
    }

    __attribute__((always_inline)) static void __amd_copyBufferRect(
        __global uchar* src,
        __global uchar* dst,
        ulong4 srcRect,
        ulong4 dstRect,
        ulong4 size)
    {
        ulong x = get_global_id(0);
        ulong y = get_global_id(1);
        ulong z = get_global_id(2);

        if ((x >= size.x) ||
            (y >= size.y) ||
            (z >= size.z)) {
            return;
        }

        ulong offsSrc = srcRect.z + x + y * srcRect.x + z * srcRect.y;
        ulong offsDst = dstRect.z + x + y * dstRect.x + z * dstRect.y;

        dst[offsDst] = src[offsSrc];
    }

    __attribute__((always_inline)) static void __amd_copyBufferRectAligned(
        __global uint* src,
        __global uint* dst,
        ulong4 srcRect,
        ulong4 dstRect,
        ulong4 size)
    {
        ulong x = get_global_id(0);
        ulong y = get_global_id(1);
        ulong z = get_global_id(2);

        if ((x >= size.x) ||
            (y >= size.y) ||
            (z >= size.z)) {
            return;
        }

        ulong offsSrc = srcRect.z + x + y * srcRect.x + z * srcRect.y;
        ulong offsDst = dstRect.z + x + y * dstRect.x + z * dstRect.y;

        if (size.w == 16) {
            __global uint4* src4 = (__global uint4*)src;
            __global uint4* dst4 = (__global uint4*)dst;
            dst4[offsDst] = src4[offsSrc];
        }
        else {
            dst[offsDst] = src[offsSrc];
        }
    }

    __attribute__((always_inline)) static void __amd_streamOpsWrite(
        __global atomic_uint* ptrUint,
        __global atomic_ulong* ptrUlong,
        ulong value) {

      // The launch parameters for this shader is a 1 grid work-item

      // 32-bit write
      if (ptrUint) {
        atomic_store_explicit(ptrUint, (uint)value, memory_order_relaxed, memory_scope_all_svm_devices);
      }
      // 64-bit write
      else {
        atomic_store_explicit(ptrUlong, value, memory_order_relaxed, memory_scope_all_svm_devices);
      }
    }

    __attribute__((always_inline)) static void __amd_streamOpsIncrement(
        __global atomic_uint* ptrUint,
        __global atomic_ulong* ptrUlong,
        ulong value) {

        if (ptrUint) {
          atomic_fetch_add_explicit (ptrUint, value,  memory_order_relaxed, memory_scope_all_svm_devices);
        } else {
          atomic_fetch_add_explicit  (ptrUlong, value,  memory_order_relaxed, memory_scope_all_svm_devices);
        }
    }

    __attribute__((always_inline)) static void __amd_streamOpsDecrement(
        __global atomic_uint* ptrUint,
        __global atomic_ulong* ptrUlong,
        ulong value) {

        __attribute__((atomic(remote_memory, fine_grained_memory)))
        {
          if (ptrUint) {
            __scoped_atomic_fetch_sub((volatile uint*)ptrUint, (uint)value, memory_order_relaxed, __MEMORY_SCOPE_SYSTEM);
          } else {
            __scoped_atomic_fetch_sub((volatile ulong*)ptrUlong, value, memory_order_relaxed, __MEMORY_SCOPE_SYSTEM);
          }
        }
    }

    __attribute__((always_inline)) static void __amd_streamOpsWait(
        __global atomic_uint* ptrUint,
        __global atomic_ulong* ptrUlong,
        ulong value, ulong compareOp, ulong mask) {

        // The launch parameters for this shader is a 1 grid work-item

        switch (compareOp) {
        case 0: //GEQ
          if (ptrUint) {
            while ((int)(atomic_load_explicit(ptrUint, memory_order_relaxed,
                        memory_scope_all_svm_devices) & (uint)mask) < (uint)value) {
              __builtin_amdgcn_s_sleep(1);
            }
          }
          else {
            while ((long)(atomic_load_explicit(ptrUlong, memory_order_relaxed,
                        memory_scope_all_svm_devices) & mask) < value) {
              __builtin_amdgcn_s_sleep(1);
            }
          }
          break;

        case 1: // EQ
          if (ptrUint) {
            while ((atomic_load_explicit(ptrUint, memory_order_relaxed,
                       memory_scope_all_svm_devices) & (uint)mask) != (uint)value) {
              __builtin_amdgcn_s_sleep(1);
            }
          }
          else {
            while ((atomic_load_explicit(ptrUlong, memory_order_relaxed,
                       memory_scope_all_svm_devices) & mask) != value) {
              __builtin_amdgcn_s_sleep(1);
            }
          }
          break;

        case 2: //AND
          if (ptrUint) {
            while (!((atomic_load_explicit(ptrUint, memory_order_relaxed,
                       memory_scope_all_svm_devices) & (uint)mask) & (uint)value)) {
              __builtin_amdgcn_s_sleep(1);
            }
          }
          else {
            while (!((atomic_load_explicit(ptrUlong, memory_order_relaxed,
                       memory_scope_all_svm_devices) & mask) & value)) {
              __builtin_amdgcn_s_sleep(1);
            }
          }
          break;

        case 3: //NOR
          if (ptrUint) {
            while (((atomic_load_explicit(ptrUint, memory_order_relaxed,
                     memory_scope_all_svm_devices) | (uint)value) & (uint)mask) == (uint)mask) {
              __builtin_amdgcn_s_sleep(1);
            }
          }
          else {
            while (((atomic_load_explicit(ptrUlong, memory_order_relaxed,
                         memory_scope_all_svm_devices) | value) & mask) == mask) {
              __builtin_amdgcn_s_sleep(1);
            }
          }
          break;
        }
    }

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

    // DEVIATION from amdblit.cl: __amd_batchMemOp there indexes param[get_global_id(0)].
    // KernelBlitManager::batchMemOps dispatches globalWorkSize=1 (rocblit.cpp), so that form would
    // execute only param[0]. CUDA requires the ops run in array order, hence the sequential loop.
    // Revert to a verbatim call once amdblit.cl replaces get_global_id(0) with a for loop.
    __kernel void __amd_rocclr_batchMemOp(__global void* params, uint count) {
      __global BatchMemOpParams* param = (__global BatchMemOpParams*)params;
      for (uint i = 0; i < count; i++) {
        switch (param[i].operation) {
          case STREAM_WAIT_VALUE_32:
            __amd_streamOpsWait((__global atomic_uint*)param[i].waitValue.address, NULL,
                                (uint)param[i].waitValue.value, (uint)param[i].waitValue.flags,
                                (ulong)~0UL);
            break;
          case STREAM_WRITE_VALUE_32:
            __amd_streamOpsWrite((__global atomic_uint*)param[i].writeValue.address, NULL,
                                 (uint)param[i].writeValue.value);
            break;
          case STREAM_WAIT_VALUE_64:
            __amd_streamOpsWait(NULL, (__global atomic_ulong*)param[i].waitValue.address,
                                param[i].waitValue.value64,
                                (uint)param[i].waitValue.flags, (ulong)~0UL);
            break;
          case STREAM_WRITE_VALUE_64:
            __amd_streamOpsWrite(NULL, (__global atomic_ulong*)param[i].writeValue.address,
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
      __amd_streamOpsWrite((__global atomic_uint*)ptrInt, (__global atomic_ulong*)ptrUlong, value);
    }

    __kernel void __amd_rocclr_streamOpsIncrement(__global uint* ptrInt, __global ulong* ptrUlong,
                                                  ulong value) {
      __amd_streamOpsIncrement((__global atomic_uint*)ptrInt, (__global atomic_ulong*)ptrUlong,
                               value);
    }

    __kernel void __amd_rocclr_streamOpsDecrement(__global uint* ptrInt, __global ulong* ptrUlong,
                                                  ulong value) {
      __amd_streamOpsDecrement((__global atomic_uint*)ptrInt, (__global atomic_ulong*)ptrUlong,
                               value);
    }

    __kernel void __amd_rocclr_streamOpsWait(__global uint* ptrInt, __global ulong* ptrUlong,
                                             ulong value, ulong flags, ulong mask) {
      __amd_streamOpsWait((__global atomic_uint*)ptrInt, (__global atomic_ulong*)ptrUlong, value,
                          flags, mask);
    }

    __kernel void __amd_rocclr_initHeap(ulong heap_to_initialize, ulong initial_blocks,
                                        uint heap_size, uint number_of_initial_blocks) {
      __ockl_dm_init_v1(heap_to_initialize, initial_blocks, heap_size, number_of_initial_blocks);
    }

    __kernel void __amd_rocclr_gwsInit(uint value) { __builtin_amdgcn_ds_gws_init(value, 0); });

const char* HipExtraSourceCodeNoGWS = BLIT_KERNELS(
    __kernel void __amd_rocclr_streamOpsWrite(__global uint* ptrInt, __global ulong* ptrUlong,
                                              ulong value) {
      __amd_streamOpsWrite((__global atomic_uint*)ptrInt, (__global atomic_ulong*)ptrUlong, value);
    }

    __kernel void __amd_rocclr_streamOpsIncrement(__global uint* ptrInt, __global ulong* ptrUlong,
                                                  ulong value) {
      __amd_streamOpsIncrement((__global atomic_uint*)ptrInt, (__global atomic_ulong*)ptrUlong,
                               value);
    }

    __kernel void __amd_rocclr_streamOpsDecrement(__global uint* ptrInt, __global ulong* ptrUlong,
                                                  ulong value) {
      __amd_streamOpsDecrement((__global atomic_uint*)ptrInt, (__global atomic_ulong*)ptrUlong,
                               value);
    }

    __kernel void __amd_rocclr_streamOpsWait(__global uint* ptrInt, __global ulong* ptrUlong,
                                             ulong value, ulong flags, ulong mask) {
      __amd_streamOpsWait((__global atomic_uint*)ptrInt, (__global atomic_ulong*)ptrUlong, value,
                          flags, mask);
    }

    __kernel void __amd_rocclr_initHeap(ulong heap_to_initialize, ulong initial_blocks,
                                        uint heap_size, uint number_of_initial_blocks) {
      __ockl_dm_init_v1(heap_to_initialize, initial_blocks, heap_size, number_of_initial_blocks);
    });

const char* BlitImageSourceCode = BLIT_KERNELS(
    __constant uint SplitCount = 3;

    __attribute__((always_inline)) static void __amd_fillImage(
        __write_only image2d_array_t image,
        float4 patternFLOAT4,
        int4 patternINT4,
        uint4 patternUINT4,
        int4 origin,
        int4 size,
        uint type)
    {
        int4  coords;

        coords.x = get_global_id(0);
        coords.y = get_global_id(1);
        coords.z = get_global_id(2);
        coords.w = 0;

        if ((coords.x >= size.x) ||
            (coords.y >= size.y) ||
            (coords.z >= size.z)) {
            return;
        }

        coords += origin;

        int SizeX = get_global_size(0);
        int AdjustedSizeX = size.x + origin.x;

        for (uint i = 0; i < SplitCount; ++i) {
            // Check components
            switch (type) {
            case 0:
                write_imagef(image, coords, patternFLOAT4);
                break;
            case 1:
                write_imagei(image, coords, patternINT4);
                break;
            case 2:
                write_imageui(image, coords, patternUINT4);
                break;
            }
            coords.x += SizeX;
            if (coords.x >= AdjustedSizeX) return;
        }
    }

    __attribute__((always_inline)) static void __amd_copyImage(
        __read_only image2d_array_t src,
        __write_only image2d_array_t dst,
        int4 srcOrigin,
        int4 dstOrigin,
        int4 size)
    {
        int4    coordsDst;
        int4    coordsSrc;

        coordsDst.x = get_global_id(0);
        coordsDst.y = get_global_id(1);
        coordsDst.z = get_global_id(2);
        coordsDst.w = 0;

        if ((coordsDst.x >= size.x) ||
            (coordsDst.y >= size.y) ||
            (coordsDst.z >= size.z)) {
            return;
        }

        coordsSrc = srcOrigin + coordsDst;
        coordsDst += dstOrigin;

        uint4  texel;
        texel = read_imageui(src, coordsSrc);
        write_imageui(dst, coordsDst, texel);
    }

    __attribute__((always_inline)) static void __amd_copyImage1DA(
        __read_only image2d_array_t src,
        __write_only image2d_array_t dst,
        int4 srcOrigin,
        int4 dstOrigin,
        int4 size)
    {
        int4 coordsDst;
        int4 coordsSrc;

        coordsDst.x = get_global_id(0);
        coordsDst.y = get_global_id(1);
        coordsDst.z = get_global_id(2);
        coordsDst.w = 0;

        if ((coordsDst.x >= size.x) ||
            (coordsDst.y >= size.y) ||
            (coordsDst.z >= size.z)) {
            return;
        }

        coordsSrc = srcOrigin + coordsDst;
        coordsDst += dstOrigin;
        if (srcOrigin.w != 0) {
           coordsSrc.z = coordsSrc.y;
           coordsSrc.y = 0;
        }
        if (dstOrigin.w != 0) {
           coordsDst.z = coordsDst.y;
           coordsDst.y = 0;
        }

        uint4  texel;
        texel = read_imageui(src, coordsSrc);
        write_imageui(dst, coordsDst, texel);
    }

    __attribute__((always_inline)) static void __amd_copyBufferToImage(
        __global uint *src,
        __write_only image2d_array_t dst,
        ulong4 srcOrigin,
        int4 dstOrigin,
        int4 size,
        uint4 format,
        ulong4 pitch)
    {
        ulong idxSrc;
        int4 coordsDst;
        uint4 pixel;
        __global uint* srcUInt = src;
        __global ushort* srcUShort = (__global ushort*)src;
        __global uchar* srcUChar  = (__global uchar*)src;
        ushort tmpUShort;
        uint tmpUInt;

        coordsDst.x = get_global_id(0);
        coordsDst.y = get_global_id(1);
        coordsDst.z = get_global_id(2);
        coordsDst.w = 0;

        if ((coordsDst.x >= size.x) ||
            (coordsDst.y >= size.y) ||
            (coordsDst.z >= size.z)) {
            return;
        }

        idxSrc = (coordsDst.z * pitch.y +
           coordsDst.y * pitch.x + coordsDst.x) *
           format.z + srcOrigin.x;

        coordsDst.x += dstOrigin.x;
        coordsDst.y += dstOrigin.y;
        coordsDst.z += dstOrigin.z;

        // Check components
        switch (format.x) {
        case 1:
            // Check size
            if (format.y == 1) {
                pixel.x = (uint)srcUChar[idxSrc];
            }
            else if (format.y == 2) {
                pixel.x = (uint)srcUShort[idxSrc];
            }
            else {
                pixel.x = srcUInt[idxSrc];
            }
        break;
        case 2:
            // Check size
            if (format.y == 1) {
                tmpUShort = srcUShort[idxSrc];
                pixel.x = (uint)(tmpUShort & 0xff);
                pixel.y = (uint)(tmpUShort >> 8);
            }
            else if (format.y == 2) {
                tmpUInt = srcUInt[idxSrc];
                pixel.x = (tmpUInt & 0xffff);
                pixel.y = (tmpUInt >> 16);
            }
            else {
                pixel.x = srcUInt[idxSrc++];
                pixel.y = srcUInt[idxSrc];
            }
        break;
        case 4:
            // Check size
            if (format.y == 1) {
                tmpUInt = srcUInt[idxSrc];
                pixel.x = tmpUInt & 0xff;
                pixel.y = (tmpUInt >> 8) & 0xff;
                pixel.z = (tmpUInt >> 16) & 0xff;
                pixel.w = (tmpUInt >> 24) & 0xff;
            }
            else if (format.y == 2) {
                tmpUInt = srcUInt[idxSrc++];
                pixel.x = tmpUInt & 0xffff;
                pixel.y = (tmpUInt >> 16);
                tmpUInt = srcUInt[idxSrc];
                pixel.z = tmpUInt & 0xffff;
                pixel.w = (tmpUInt >> 16);
            }
            else {
                pixel.x = srcUInt[idxSrc++];
                pixel.y = srcUInt[idxSrc++];
                pixel.z = srcUInt[idxSrc++];
                pixel.w = srcUInt[idxSrc];
            }
        break;
        }
        // Write the final pixel
        write_imageui(dst, coordsDst, pixel);
    }

    __attribute__((always_inline)) static void __amd_copyImageToBuffer(
        __read_only image2d_array_t src,
        __global uint* dstUInt,
        __global ushort* dstUShort,
        __global uchar* dstUChar,
        int4 srcOrigin,
        ulong4 dstOrigin,
        int4 size,
        uint4 format,
        ulong4 pitch)
    {
        ulong idxDst;
        int4 coordsSrc;
        uint4 texel;

        coordsSrc.x = get_global_id(0);
        coordsSrc.y = get_global_id(1);
        coordsSrc.z = get_global_id(2);
        coordsSrc.w = 0;

        if ((coordsSrc.x >= size.x) ||
            (coordsSrc.y >= size.y) ||
            (coordsSrc.z >= size.z)) {
            return;
        }

        idxDst = (coordsSrc.z * pitch.y + coordsSrc.y * pitch.x +
            coordsSrc.x) * format.z + dstOrigin.x;

        coordsSrc.x += srcOrigin.x;
        coordsSrc.y += srcOrigin.y;
        coordsSrc.z += srcOrigin.z;

        texel = read_imageui(src, coordsSrc);

        // Check components
        switch (format.x) {
        case 1:
            // Check size
            switch (format.y) {
            case 1:
                dstUChar[idxDst] = (uchar)texel.x;
                break;
            case 2:
                dstUShort[idxDst] = (ushort)texel.x;
                break;
            case 4:
                dstUInt[idxDst] = texel.x;
                break;
            }
        break;
        case 2:
            // Check size
            switch (format.y) {
            case 1:
                dstUShort[idxDst] = (ushort)texel.x |
                   ((ushort)texel.y << 8);
                break;
            case 2:
                dstUInt[idxDst] = texel.x | (texel.y << 16);
                break;
            case 4:
                dstUInt[idxDst++] = texel.x;
                dstUInt[idxDst] = texel.y;
                break;
            }
        break;
        case 4:
            // Check size
            switch (format.y) {
            case 1:
                dstUInt[idxDst] = (uint)texel.x |
                   (texel.y << 8) |
                   (texel.z << 16) |
                   (texel.w << 24);
                break;
            case 2:
                dstUInt[idxDst++] = texel.x | (texel.y << 16);
                dstUInt[idxDst] = texel.z | (texel.w << 16);
                break;
            case 4:
                dstUInt[idxDst++] = texel.x;
                dstUInt[idxDst++] = texel.y;
                dstUInt[idxDst++] = texel.z;
                dstUInt[idxDst] = texel.w;
                break;
            }
        break;
        }
    }

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
