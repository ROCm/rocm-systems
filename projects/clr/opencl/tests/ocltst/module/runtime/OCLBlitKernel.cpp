/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include "OCLBlitKernel.h"

#include <Timer.h>
#include <assert.h>
#include <stdio.h>
#include <string.h>

#include "CL/cl.h"

const static cl_uint Stages = 4;
const static cl_uint ThreadsForCheck = 1 << Stages;

#define KERNEL_CODE(...) #__VA_ARGS__

const static char* strKernel =
    KERNEL_CODE(
    \n
    \x23 if OCL20
    \n
    extern void __amd_scheduler(__global void *, __global void *, uint);
    \n
    \x23 endif
    \n
    static const uint SplitCount = 3;

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

    __attribute__((always_inline)) static void __amd_copyBuffer(
        __global uchar* srcI,
        __global uchar* dstI,
        ulong srcOrigin,
        ulong dstOrigin,
        ulong size,
        uint remain)
    {
        ulong id = get_global_id(0);

        if (id >= size) {
            return;
        }

        __global uchar* src = srcI + srcOrigin;
        __global uchar* dst = dstI + dstOrigin;

        if (remain == 8) {
            dst[id] = src[id];
        }
        else {
            if (id < (size - 1)) {
                __global uint* srcD = (__global uint*)(src);
                __global uint* dstD = (__global uint*)(dst);
                dstD[id] = srcD[id];
            }
            else {
                for (uint i = 0; i < remain; ++i) {
                    dst[id * 4 + i] = src[id * 4 + i];
                }
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

    __attribute__((always_inline)) static void __amd_fillBuffer(
        __global uchar* bufUChar,
        __global uint* bufUInt,
        __constant uchar* pattern,
        uint patternSize,
        ulong offset,
        ulong size)
    {
        ulong id = get_global_id(0);

        if (id >= size) {
            return;
        }

        if (bufUInt) {
           __global uint* element = &bufUInt[offset + id * patternSize];
           __constant uint*  pt = (__constant uint*)pattern;

            for (uint i = 0; i < patternSize; ++i) {
                element[i] = pt[i];
            }
        }
        else {
            __global uchar* element = &bufUChar[offset + id * patternSize];

            for (uint i = 0; i < patternSize; ++i) {
                element[i] = pattern[i];
            }
        }
    }

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

    __kernel void copyBufferToImage(
        __global uint* src,
        __write_only image2d_array_t dst,
        ulong4 srcOrigin,
        int4 dstOrigin,
        int4 size,
        uint4 format,
        ulong4 pitch)
    {
        __amd_copyBufferToImage(src, dst, srcOrigin, dstOrigin, size, format, pitch);
    }

    __kernel void copyImageToBuffer(
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
        __amd_copyImageToBuffer(src, dstUInt, dstUShort, dstUChar,
                                  srcOrigin, dstOrigin, size, format, pitch);
    }

    __kernel void copyImage(
        __read_only  image2d_array_t src,
        __write_only image2d_array_t dst,
        int4 srcOrigin,
        int4 dstOrigin,
        int4 size)
    {
        __amd_copyImage(src, dst, srcOrigin, dstOrigin, size);
    }

    __kernel void copyImage1DA(
        __read_only image2d_array_t src,
        __write_only image2d_array_t dst,
        int4 srcOrigin,
        int4 dstOrigin,
        int4 size)
    {
        __amd_copyImage1DA(src, dst, srcOrigin, dstOrigin, size);
    }

    __kernel void copyBufferRect(
        __global uchar* src,
        __global uchar* dst,
        ulong4 srcRect,
        ulong4 dstRect,
        ulong4 size)
    {
        __amd_copyBufferRect(src, dst, srcRect, dstRect, size);
    }

    __kernel void copyBufferRectAligned(
        __global uint* src,
        __global uint* dst,
        ulong4 srcRect,
        ulong4 dstRect,
        ulong4 size)
    {
        __amd_copyBufferRectAligned(src, dst, srcRect, dstRect, size);
    }

    __kernel void copyBuffer(
        __global uchar* srcI,
        __global uchar* dstI,
        ulong srcOrigin,
        ulong dstOrigin,
        ulong size,
        uint remain)
    {
        __amd_copyBuffer(srcI, dstI, srcOrigin, dstOrigin, size, remain);
    }

    __kernel void copyBufferAligned(
        __global uint* src,
        __global uint* dst,
        ulong srcOrigin,
        ulong dstOrigin,
        ulong size,
        uint alignment)
    {
        __amd_copyBufferAligned(src, dst, srcOrigin, dstOrigin, size, alignment);
    }

    __kernel void fillBuffer(
        __global uchar* bufUChar,
        __global uint* bufUInt,
        __constant uchar* pattern,
        uint patternSize,
        ulong offset,
        ulong size)
    {
        __amd_fillBuffer(bufUChar, bufUInt, pattern, patternSize, offset, size);
    }

    __kernel void fillImage(
        __write_only image2d_array_t image,
        float4 patternFLOAT4,
        int4 patternINT4,
        uint4 patternUINT4,
        int4 origin,
        int4 size,
        uint type)
    {
        __amd_fillImage(image, patternFLOAT4, patternINT4, patternUINT4,
                          origin, size, type);
    }
    \n
    \x23 if OCL20
    \n
    typedef struct _HsaAqlDispatchPacket {
        uint    mix;
        ushort  workgroup_size[3];
        ushort  reserved2;
        uint    grid_size[3];
        uint    private_segment_size_bytes;
        uint    group_segment_size_bytes;
        ulong   kernel_object_address;
        ulong   kernel_arg_address;
        ulong   reserved3;
        ulong   completion_signal;
    } HsaAqlDispatchPacket;
    \n
    // This is an OpenCLized hsa_control_directives_t
    typedef struct _AmdControlDirectives {
        ulong   enabled_control_directives;
        ushort  enable_break_exceptions;
        ushort  enable_detect_exceptions;
        uint    max_dynamic_group_size;
        ulong   max_flat_grid_size;
        uint    max_flat_workgroup_size;
        uchar   required_dim;
        uchar   reserved1[3];
        ulong   required_grid_size[3];
        uint    required_workgroup_size[3];
        uchar   reserved2[60];
    } AmdControlDirectives;
    \n
    // This is an OpenCLized amd_kernel_code_t
    typedef struct _AmdKernelCode {
        uint    amd_kernel_code_version_major;
        uint    amd_kernel_code_version_minor;
        ushort  amd_machine_kind;
        ushort  amd_machine_version_major;
        ushort  amd_machine_version_minor;
        ushort  amd_machine_version_stepping;
        long    kernel_code_entry_byte_offset;
        long    kernel_code_prefetch_byte_offset;
        ulong   kernel_code_prefetch_byte_size;
        ulong   max_scratch_backing_memory_byte_size;
        uint    compute_pgm_rsrc1;
        uint    compute_pgm_rsrc2;
        uint    kernel_code_properties;
        uint    workitem_private_segment_byte_size;
        uint    workgroup_group_segment_byte_size;
        uint    gds_segment_byte_size;
        ulong   kernarg_segment_byte_size;
        uint    workgroup_fbarrier_count;
        ushort  wavefront_sgpr_count;
        ushort  workitem_vgpr_count;
        ushort  reserved_vgpr_first;
        ushort  reserved_vgpr_count;
        ushort  reserved_sgpr_first;
        ushort  reserved_sgpr_count;
        ushort  debug_wavefront_private_segment_offset_sgpr;
        ushort  debug_private_segment_buffer_sgpr;
        uchar   kernarg_segment_alignment;
        uchar   group_segment_alignment;
        uchar   private_segment_alignment;
        uchar   wavefront_size;
        int     call_convention;
        uchar   reserved1[12];
        ulong   runtime_loader_kernel_symbol;
        AmdControlDirectives control_directives;
    } AmdKernelCode;
    \n
    typedef struct _HwDispatchHeader {
        uint    writeData0;     // CP WRITE_DATA write to rewind for memory
        uint    writeData1;
        uint    writeData2;
        uint    writeData3;
        uint    rewind;         // REWIND execution
        uint    startExe;       // valid bit
        uint    condExe0;       // 0xC0032200 -- TYPE 3, COND_EXEC
        uint    condExe1;       // 0x00000204 ----
        uint    condExe2;       // 0x00000000 ----
        uint    condExe3;       // 0x00000000 ----
        uint    condExe4;       // 0x00000000 ----
    } HwDispatchHeader;
    \n
    typedef struct _HwDispatch {
        uint    packet0;        // 0xC0067602 -- TYPE 3, SET_SH_REG, TYPE:COMPUTE (6 values)
        uint    offset0;        // 0x00000204 ---- OFFSET
        uint    startX;         // 0x00000000 ---- COMPUTE_START_X: START = 0x0
        uint    startY;         // 0x00000000 ---- COMPUTE_START_Y: START = 0x0
        uint    startZ;         // 0x00000000 ---- COMPUTE_START_Z: START = 0x0
        uint    wrkGrpSizeX;    // 0x00000000 ---- COMPUTE_NUM_THREAD_X: NUM_THREAD_FULL = 0x0, NUM_THREAD_PARTIAL = 0x0
        uint    wrkGrpSizeY;    // 0x00000000 ---- COMPUTE_NUM_THREAD_Y: NUM_THREAD_FULL = 0x0, NUM_THREAD_PARTIAL = 0x0
        uint    wrkGrpSizeZ;    // 0x00000000 ---- COMPUTE_NUM_THREAD_Z: NUM_THREAD_FULL = 0x0, NUM_THREAD_PARTIAL = 0x0
        uint    packet1;        // 0xC0027602 -- TYPE 3, SET_SH_REG, TYPE:COMPUTE (2 values)
        uint    offset1;        // 0x0000020C ---- OFFSET
        uint    isaLo;          // 0x00000000 ---- COMPUTE_PGM_LO: DATA = 0x0
        uint    isaHi;          // 0x00000000 ---- COMPUTE_PGM_HI: DATA = 0x0, INST_ATC__CI__VI = 0x0
        uint    packet2;        // 0xC0027602 -- TYPE 3, SET_SH_REG, TYPE:COMPUTE (2 values)
        uint    offset2;        // 0x00000212 ---- OFFSET
        uint    resource1;      // 0x00000000 ---- COMPUTE_PGM_RSRC1
        uint    resource2;      // 0x00000000 ---- COMPUTE_PGM_RSRC2
        uint    packet3;        // 0xc0017602 -- TYPE 3, SET_SH_REG, TYPE:COMPUTE (1 value)
        uint    offset3;        // 0x00000215 ---- OFFSET
        uint    pad31;          // 0x000003ff ---- COMPUTE_RESOURCE_LIMITS
        uint    packet31;       // 0xC0067602 -- TYPE 3, SET_SH_REG, TYPE:COMPUTE (1 value)
        uint    offset31;       // 0x00000218 ---- OFFSET
        uint    ringSize;       // 0x00000000 ---- COMPUTE_TMPRING_SIZE: WAVES = 0x0, WAVESIZE = 0x0
        uint    user0;          // 0xC0047602 -- TYPE 3, SET_SH_REG, TYPE:COMPUTE (4 values)
        uint    offsUser0;      // 0x00000240 ---- OFFSET
        uint    scratchLo;      // 0x00000000 ---- COMPUTE_USER_DATA_0: DATA = 0x0
        uint    scratchHi;      // 0x80000000 ---- COMPUTE_USER_DATA_1: DATA = 0x80000000
        uint    scratchSize;    // 0x00000000 ---- COMPUTE_USER_DATA_2: DATA = 0x0
        uint    padUser;        // 0x00EA7FAC ---- COMPUTE_USER_DATA_3: DATA = 0xEA7FAC
        uint    user1;          // 0xC0027602 -- TYPE 3, SET_SH_REG, TYPE:COMPUTE (2 values)
        uint    offsUser1;      // 0x00000244 ---- OFFSET
        uint    aqlPtrLo;       // 0x00000000 ---- COMPUTE_USER_DATA_4: DATA = 0x0
        uint    aqlPtrHi;       // 0x00000000 ---- COMPUTE_USER_DATA_5: DATA = 0x0
        uint    user2;          // 0xC0027602 -- TYPE 3, SET_SH_REG, TYPE:COMPUTE (2 values)
        uint    offsUser2;      // 0x00000246 ---- OFFSET
        uint    hsaQueueLo;     // 0x00000000 ---- COMPUTE_USER_DATA_6: DATA = 0x0
        uint    hsaQueueHi;     // 0x00000000 ---- COMPUTE_USER_DATA_7: DATA = 0x0
        uint    user3;          // 0xC0027602 -- TYPE 3, SET_SH_REG, TYPE:COMPUTE (2 values)
        uint    offsUser3;      // 0x00000246 ---- OFFSET
        uint    argsLo;         // 0x00000000 ---- COMPUTE_USER_DATA_8: DATA = 0x0
        uint    argsHi;         // 0x00000000 ---- COMPUTE_USER_DATA_9: DATA = 0x0
        uint    copyData;       // 0xC0044000 -- TYPE 3, COPY_DATA
        uint    copyDataFlags;  // 0x00000405 ---- srcSel 0x5, destSel 0x4, countSel 0x0, wrConfirm 0x0, engineSel 0x0
        uint    scratchAddrLo;  // 0x000201C4 ---- srcAddressLo
        uint    scratchAddrHi;  // 0x00000000 ---- srcAddressHi
        uint    shPrivateLo;    // 0x00002580 ---- dstAddressLo
        uint    shPrivateHi;    // 0x00000000 ---- dstAddressHi
        uint    user4;          // 0xC0027602 -- TYPE 3, SET_SH_REG, TYPE:COMPUTE (2 values)
        uint    offsUser4;      // 0x00000248 ---- OFFSET
        uint    scratchOffs;    // 0x00000000 ---- COMPUTE_USER_DATA_10: DATA = 0x0
        uint    privSize;       // 0x00000030 ---- COMPUTE_USER_DATA_11: DATA = 0x30
        uint    packet4;        // 0xC0031502 -- TYPE 3, DISPATCH_DIRECT, TYPE:COMPUTE
        uint    glbSizeX;       // 0x00000000
        uint    glbSizeY;       // 0x00000000
        uint    glbSizeZ;       // 0x00000000
        uint    padd41;         // 0x00000021
    } HwDispatch;
    \n
    static const uint WavefrontSize     = 64;
    static const uint MaxWaveSize       = 0x400;
    static const uint UsrRegOffset      = 0x240;
    static const uint Pm4Nop            = 0xC0001002;
    static const uint Pm4UserRegs       = 0xC0007602;
    static const uint Pm4CopyReg        = 0xC0044000;
    static const uint PrivateSegEna     = 0x1;
    static const uint DispatchEna       = 0x2;
    static const uint QueuePtrEna       = 0x4;
    static const uint KernelArgEna      = 0x8;
    static const uint FlatScratchEna    = 0x20;
    \n
    uint GetCmdTemplateHeaderSize() { return sizeof(HwDispatchHeader); }
    \n
    uint GetCmdTemplateDispatchSize() { return sizeof(HwDispatch); }
    \n
    void EmptyCmdTemplateDispatch(ulong cmdBuf)
    {
        volatile __global HwDispatch* dispatch = (volatile __global HwDispatch*)cmdBuf;
        dispatch->glbSizeX = 0;
        dispatch->glbSizeY = 0;
        dispatch->glbSizeZ = 0;
    }
    \n
    void RunCmdTemplateDispatch(
        ulong   cmdBuf,
        __global HsaAqlDispatchPacket* aqlPkt,
        ulong   scratch,
        ulong   hsaQueue,
        uint    scratchSize,
        uint    scratchOffset,
        uint    numMaxWaves,
        uint    useATC)
    \n
    {
        volatile __global HwDispatch* dispatch = (volatile __global HwDispatch*)cmdBuf;
        uint usrRegCnt = 0;

        // Program workgroup size
        dispatch->wrkGrpSizeX = aqlPkt->workgroup_size[0];
        dispatch->wrkGrpSizeY = aqlPkt->workgroup_size[1];
        dispatch->wrkGrpSizeZ = aqlPkt->workgroup_size[2];

        // ISA address
        __global AmdKernelCode* kernelObj = (__global AmdKernelCode*)aqlPkt->kernel_object_address;
        ulong isa = aqlPkt->kernel_object_address + kernelObj->kernel_code_entry_byte_offset;

        dispatch->isaLo = (uint)(isa >> 8);
        dispatch->isaHi = (uint)(isa >> 40) | (useATC ? 0x100 : 0);

        // Program PGM resource registers
        dispatch->resource1 = kernelObj->compute_pgm_rsrc1;
        dispatch->resource2 = kernelObj->compute_pgm_rsrc2;

        uint    flags = kernelObj->kernel_code_properties;
        uint    privateSize = kernelObj->workitem_private_segment_byte_size;

        uint ldsSize = aqlPkt->group_segment_size_bytes +
            kernelObj->workgroup_group_segment_byte_size;

        // Align up the LDS blocks 128 * 4(in DWORDs)
        uint ldsBlocks = (ldsSize + 511) >> 9;

        dispatch->resource2 |= (ldsBlocks << 15);

        // Private/scratch segment was enabled
        if (flags & PrivateSegEna) {
            uint    waveSize = privateSize * WavefrontSize;
            // 256 DWRODs is the minimum for SQ
            waveSize = max(MaxWaveSize, waveSize);

            uint numWaves = scratchSize / waveSize;

            numWaves = min(numWaves, numMaxWaves);

            dispatch->ringSize = numWaves;
            dispatch->ringSize |= (waveSize >> 10) << 12;
            dispatch->user0 = Pm4UserRegs | (4 << 16);
            dispatch->scratchLo = (uint)scratch;
            dispatch->scratchHi = ((uint)(scratch >> 32)) | 0x80000000; // Enables swizzle
            dispatch->scratchSize = scratchSize;
            usrRegCnt += 4;
        }
        else {
            dispatch->ringSize = 0;
            dispatch->user0 = Pm4Nop | (4 << 16);
        }

        // Pointer to the AQL dispatch packet
        dispatch->user1 = (flags & DispatchEna) ? (Pm4UserRegs | (2 << 16)) : (Pm4Nop | (2 << 16));
        dispatch->offsUser1 = UsrRegOffset + usrRegCnt;
        usrRegCnt += (flags & DispatchEna) ? 2 : 0;
        ulong  gpuAqlPtr = (ulong)aqlPkt;
        dispatch->aqlPtrLo = (uint)gpuAqlPtr;
        dispatch->aqlPtrHi = (uint)(gpuAqlPtr >> 32);

        // Pointer to the AQL queue header
        if (flags & QueuePtrEna) {
            dispatch->user2 = Pm4UserRegs | (2 << 16);
            dispatch->offsUser2 = UsrRegOffset + usrRegCnt;
            usrRegCnt += 2;
            dispatch->hsaQueueLo = (uint)hsaQueue;
            dispatch->hsaQueueHi = (uint)(hsaQueue >> 32);
        }
        else {
            dispatch->user2 = Pm4Nop | (2 << 16);
        }

        // Pointer to the AQL kernel arguments
        dispatch->user3 = (flags & KernelArgEna) ? (Pm4UserRegs | (2 << 16)) : (Pm4Nop | (2 << 16));
        dispatch->offsUser3 = UsrRegOffset + usrRegCnt;
        usrRegCnt += (flags & KernelArgEna) ? 2 : 0;
        dispatch->argsLo = (uint)aqlPkt->kernel_arg_address;
        dispatch->argsHi = (uint)(aqlPkt->kernel_arg_address >> 32);

        // Provide pointer to the private/scratch buffer for the flat address
        if (flags & FlatScratchEna) {
            dispatch->copyData = Pm4CopyReg;
            dispatch->scratchAddrLo = (uint)((scratch - scratchOffset) >> 16);
            dispatch->offsUser4 = UsrRegOffset + usrRegCnt;
            dispatch->scratchOffs = scratchOffset;
            dispatch->privSize = privateSize;
        }
        else {
            dispatch->copyData = Pm4Nop | (8 << 16);
        }

        // Update the global launch grid
        dispatch->glbSizeX = aqlPkt->grid_size[0];
        dispatch->glbSizeY = aqlPkt->grid_size[1];
        dispatch->glbSizeZ = aqlPkt->grid_size[2];
    }
    \n
    __kernel void scheduler(
        __global void * queue,
        __global void * params,
        uint paramIdx)
    {
        __amd_scheduler(queue, params, paramIdx);
    }
    \n
    \x23 endif
    \n
    );

enum {
  BlitCopyImage = 0,
  BlitCopyImage1DA,
  BlitCopyImageToBuffer,
  BlitCopyBufferToImage,
  BlitCopyBufferRect,
  BlitCopyBufferRectAligned,
  BlitCopyBuffer,
  BlitCopyBufferAligned,
  FillBuffer,
  FillImage,
  Scheduler,
  BlitTotal
};

static const char* BlitName[BlitTotal] = {
    "copyImage",         "copyImage1DA",      "copyImageToBuffer",
    "copyBufferToImage", "copyBufferRect",    "copyBufferRectAligned",
    "copyBuffer",        "copyBufferAligned", "fillBuffer",
    "fillImage",         "scheduler",
};

OCLBlitKernel::OCLBlitKernel() { _numSubTests = 1; }

OCLBlitKernel::~OCLBlitKernel() {}

void OCLBlitKernel::open(unsigned int test, char* units, double& conversion,
                         unsigned int deviceId) {
  OCLTestImp::open(test, units, conversion, deviceId);
  CHECK_RESULT((error_ != CL_SUCCESS), "Error opening test");
  char dbuffer[1024] = {0};
  CPerfCounter timer;
  int sub = 0;
  std::string options = "-cl-std=CL2.0 -DOCL20=1";

  cl_device_type deviceType;
  error_ = _wrapper->clGetDeviceInfo(devices_[deviceId], CL_DEVICE_TYPE, sizeof(deviceType),
                                     &deviceType, NULL);
  CHECK_RESULT((error_ != CL_SUCCESS), "CL_DEVICE_TYPE failed");

  if (!(deviceType & CL_DEVICE_TYPE_GPU)) {
    testDescString = "GPU device is required for this test!\n";
    return;
  }

  size_t param_size = 0;
  char* strVersion = 0;
  error_ = _wrapper->clGetDeviceInfo(devices_[_deviceId], CL_DEVICE_VERSION, 0, 0, &param_size);
  CHECK_RESULT(error_ != CL_SUCCESS, "clGetDeviceInfo failed");
  strVersion = new char[param_size];
  error_ =
      _wrapper->clGetDeviceInfo(devices_[_deviceId], CL_DEVICE_VERSION, param_size, strVersion, 0);
  CHECK_RESULT(error_ != CL_SUCCESS, "clGetDeviceInfo failed");
  if (strVersion[7] < '2') {
    options = "-DOCL20=0";
    sub = 1;
    delete strVersion;
    testDescString = "Currently it works for OCL20 devices only!\n";
    return;
  }
  delete strVersion;

  error_ = _wrapper->clGetDeviceInfo(devices_[_deviceId], CL_DRIVER_VERSION, 0, 0, &param_size);
  CHECK_RESULT(error_ != CL_SUCCESS, "clGetDeviceInfo failed");
  strVersion = new char[param_size];
  error_ =
      _wrapper->clGetDeviceInfo(devices_[_deviceId], CL_DRIVER_VERSION, param_size, strVersion, 0);
  CHECK_RESULT(error_ != CL_SUCCESS, "clGetDeviceInfo failed");

  std::string sch = strKernel;
  static const char AmdScheduler[] = "amd_scheduler";
  static const char AmdSchedulerPal[] = "amd_scheduler_pal";
  static const char AmdSchedulerROCm[] = "amd_scheduler_rocm";
  const char* AmdSchedulerPatch = NULL;
  size_t loc = 0;

  if (NULL != strstr(strVersion, "LC")) {
    if (NULL != strstr(strVersion, "PAL")) {
      AmdSchedulerPatch = AmdSchedulerPal;
    } else if (NULL != strstr(strVersion, "HSA")) {
      AmdSchedulerPatch = AmdSchedulerROCm;
    }
  }
  delete strVersion;

  if (NULL != AmdSchedulerPatch) {
    loc = sch.find(AmdScheduler);
    sch.replace(loc, strlen(AmdScheduler), AmdSchedulerPatch);
    loc = sch.find(AmdScheduler, (loc + strlen(AmdSchedulerPatch)));
    sch.replace(loc, strlen(AmdScheduler), AmdSchedulerPatch);
  }

  timer.Reset();
  timer.Start();

  const char* strProgram = sch.c_str();
  program_ = _wrapper->clCreateProgramWithSource(context_, 1, &strProgram, NULL, &error_);
  CHECK_RESULT((error_ != CL_SUCCESS), "clCreateProgramWithSource()  failed");
  error_ = _wrapper->clBuildProgram(program_, 1, &devices_[deviceId], options.c_str(), NULL, NULL);
  if (error_ != CL_SUCCESS) {
    char programLog[1024];
    _wrapper->clGetProgramBuildInfo(program_, devices_[deviceId], CL_PROGRAM_BUILD_LOG, 1024,
                                    programLog, 0);
    printf("\n%s\n", programLog);
    fflush(stdout);
  }
  CHECK_RESULT((error_ != CL_SUCCESS), "clBuildProgram() failed");

  cl_kernel kernels[BlitTotal];
  for (int i = 0; i < BlitTotal - sub; ++i) {
    kernels[i] = _wrapper->clCreateKernel(program_, BlitName[i], &error_);
    CHECK_RESULT((error_ != CL_SUCCESS), "clCreateKernel() failed");
  }
  timer.Stop();
  double sec = timer.GetElapsedTime();

  time_ = (float)sec * 1000.f;
  testDescString = "Blit kernel compilaiton time (ms):";

  for (int i = 0; i < BlitTotal - sub; ++i) {
    _wrapper->clReleaseKernel(kernels[i]);
  }
}

void OCLBlitKernel::run(void) { _perfInfo = time_; }

unsigned int OCLBlitKernel::close(void) { return OCLTestImp::close(); }
