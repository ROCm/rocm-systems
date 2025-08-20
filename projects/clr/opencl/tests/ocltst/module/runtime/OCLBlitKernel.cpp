/* Copyright (c) 2010 - 2021 Advanced Micro Devices, Inc.

 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in
 all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE. */

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
    extern void __amd_copyBufferToImage(
        __global uint*, __write_only image2d_array_t, ulong4,
        int4, int4, uint4, ulong4);

    extern void __amd_copyImageToBuffer(
        __read_only image2d_array_t, __global uint*, __global ushort*,
        __global uchar*, int4, ulong4, int4, uint4, ulong4);

    extern void __amd_copyImage(
        __read_only image2d_array_t, __write_only image2d_array_t,
        int4, int4, int4);

    extern void __amd_copyImage1DA(
        __read_only image2d_array_t, __write_only image2d_array_t,
        int4, int4, int4);

    extern void __amd_copyBufferRect(
        __global uchar*, __global uchar*,
        ulong4, ulong4, ulong4);

    extern void __amd_copyBufferRectAligned(
        __global uint*, __global uint*,
        ulong4, ulong4, ulong4);

    extern void __amd_copyBuffer(
        __global uchar*, __global uchar*,
        ulong, ulong, ulong, uint);

    extern void __amd_copyBufferAligned(
        __global uint*, __global uint*,
        ulong, ulong, ulong, uint);

    extern void __amd_fillBuffer(
        __global uchar*, __global uint*, __constant uchar*,
        uint, ulong, ulong);

    extern void __amd_fillImage(
        __write_only image2d_array_t,
        float4, int4, uint4, int4, int4, uint);

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
