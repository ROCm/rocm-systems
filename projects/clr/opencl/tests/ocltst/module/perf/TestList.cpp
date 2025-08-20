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

#include "OCLTestListImp.h"

//
// Includes for tests
//
#include "OCLPerfAES256.h"
#include "OCLPerfAtomicSpeed.h"
#include "OCLPerfBufferCopyOverhead.h"
#include "OCLPerfBufferCopySpeed.h"
#include "OCLPerfBufferReadSpeed.h"
#include "OCLPerfBufferWriteSpeed.h"
#include "OCLPerfCPUMemSpeed.h"
#include "OCLPerfCommandQueue.h"
#include "OCLPerfConcurrency.h"
#include "OCLPerfDevMemReadSpeed.h"
#include "OCLPerfDevMemWriteSpeed.h"
#include "OCLPerfDeviceConcurrency.h"
#include "OCLPerfDeviceEnqueue.h"
#include "OCLPerfDispatchSpeed.h"
#include "OCLPerfDoubleDMA.h"
#include "OCLPerfDoubleDMASeq.h"
#include "OCLPerfFillBuffer.h"
#include "OCLPerfFillImage.h"
#include "OCLPerfFlush.h"
#include "OCLPerfGenericBandwidth.h"
#include "OCLPerfGenoilSiaMiner.h"
#include "OCLPerfImageCopyCorners.h"
#include "OCLPerfImageCopySpeed.h"
#include "OCLPerfImageMapUnmap.h"
#include "OCLPerfImageReadSpeed.h"
#include "OCLPerfImageSampleRate.h"
#include "OCLPerfImageWriteSpeed.h"
#include "OCLPerfKernelArguments.h"
#include "OCLPerfLDSLatency.h"
#include "OCLPerfLDSReadSpeed.h"
#include "OCLPerfMandelbrot.h"
#include "OCLPerfMapBufferReadSpeed.h"
#include "OCLPerfMapBufferWriteSpeed.h"
#include "OCLPerfMapImageReadSpeed.h"
#include "OCLPerfMapImageWriteSpeed.h"
#include "OCLPerfMatrixTranspose.h"
#include "OCLPerfMemCombine.h"
#include "OCLPerfMemCreate.h"
#include "OCLPerfMemLatency.h"
#include "OCLPerfPinnedBufferReadSpeed.h"
#include "OCLPerfPinnedBufferWriteSpeed.h"
#include "OCLPerfPipeCopySpeed.h"
#include "OCLPerfSHA256.h"
#include "OCLPerfSampleRate.h"
#include "OCLPerfScalarReplArrayElem.h"
#include "OCLPerfSdiP2PCopy.h"
#include "OCLPerfTextureMemLatency.h"
#include "OCLPerfUAVReadSpeed.h"
#include "OCLPerfUAVReadSpeedHostMem.h"
#include "OCLPerfUAVWriteSpeedHostMem.h"
#include "OCLPerfVerticalFetch.h"
// 2.0
#include "OCLPerf3DImageWriteSpeed.h"
#include "OCLPerfAtomicSpeed20.h"
#include "OCLPerfDeviceEnqueue2.h"
#include "OCLPerfDeviceEnqueueEvent.h"
#include "OCLPerfDeviceEnqueueSier.h"
#include "OCLPerfImageCreate.h"
#include "OCLPerfImageReadWrite.h"
#include "OCLPerfImageReadsRGBA.h"
#include "OCLPerfProgramGlobalRead.h"
#include "OCLPerfProgramGlobalWrite.h"
#include "OCLPerfSVMAlloc.h"
#include "OCLPerfSVMKernelArguments.h"
#include "OCLPerfSVMMap.h"
#include "OCLPerfSVMMemFill.h"
#include "OCLPerfSVMMemcpy.h"
#include "OCLPerfSVMSampleRate.h"
#include "OCLPerfUncoalescedRead.h"

//
//  Helper macro for adding tests
//
template <typename T> static void* dictionary_CreateTestFunc(void) { return new T(); }

#define TEST(name) {#name, &dictionary_CreateTestFunc<name>}

TestEntry TestList[] = {
    TEST(OCLPerfUAVReadSpeed),
    TEST(OCLPerfUAVReadSpeedHostMem),
    TEST(OCLPerfUAVWriteSpeedHostMem),
    TEST(OCLPerfLDSReadSpeed),
    TEST(OCLPerfDispatchSpeed),
    TEST(OCLPerfMapBufferReadSpeed),
    TEST(OCLPerfMapBufferWriteSpeed),
    TEST(OCLPerfBufferReadSpeed),
    TEST(OCLPerfBufferReadRectSpeed),
    TEST(OCLPerfPinnedBufferReadSpeed),
    TEST(OCLPerfPinnedBufferReadRectSpeed),
    TEST(OCLPerfBufferWriteSpeed),
    TEST(OCLPerfBufferWriteRectSpeed),
    TEST(OCLPerfPinnedBufferWriteSpeed),
    TEST(OCLPerfPinnedBufferWriteRectSpeed),
    TEST(OCLPerfBufferCopySpeed),
    TEST(OCLPerfBufferCopyRectSpeed),
    TEST(OCLPerfMapImageReadSpeed),
    TEST(OCLPerfMapImageWriteSpeed),
    TEST(OCLPerfMemCombine),
    TEST(OCLPerfImageReadSpeed),
    TEST(OCLPerfPinnedImageReadSpeed),
    TEST(OCLPerfImageWriteSpeed),
    TEST(OCLPerfPinnedImageWriteSpeed),
    TEST(OCLPerfImageCopySpeed),
    TEST(OCLPerfCPUMemSpeed),
    TEST(OCLPerfMandelbrot),
    TEST(OCLPerfAsyncMandelbrot),
    TEST(OCLPerfConcurrency),
    TEST(OCLPerfDeviceConcurrency),
    TEST(OCLPerfAES256),
    TEST(OCLPerfSHA256),
    TEST(OCLPerfAtomicSpeed),
    TEST(OCLPerfMatrixTranspose),
    TEST(OCLPerfImageCopyCorners),
    TEST(OCLPerfScalarReplArrayElem),
    TEST(OCLPerfSdiP2PCopy),
    TEST(OCLPerfFlush),
    TEST(OCLPerfMemCreate),
    TEST(OCLPerfImageMapUnmap),
    TEST(OCLPerfCommandQueue),
    TEST(OCLPerfKernelArguments),
    TEST(OCLPerfDoubleDMA),
    TEST(OCLPerfDoubleDMASeq),
    TEST(OCLPerfMemLatency),
    TEST(OCLPerfTextureMemLatency),
    TEST(OCLPerfSampleRate),
    TEST(OCLPerfImageSampleRate),
    TEST(OCLPerfBufferCopyOverhead),
    TEST(OCLPerfMapDispatchSpeed),
    TEST(OCLPerfDeviceEnqueue),
    TEST(OCLPerfPipeCopySpeed),
    TEST(OCLPerfGenericBandwidth),
    TEST(OCLPerfLDSLatency),
    TEST(OCLPerfDeviceEnqueue2),
    TEST(OCLPerfSVMAlloc),
    TEST(OCLPerfSVMMap),
    TEST(OCLPerfDeviceEnqueueEvent),
    TEST(OCLPerfSVMKernelArguments),
    TEST(OCLPerfDeviceEnqueueSier),
    TEST(OCLPerfProgramGlobalRead),
    TEST(OCLPerfProgramGlobalWrite),
    TEST(OCLPerfAtomicSpeed20),
    TEST(OCLPerfSVMSampleRate),
    TEST(OCLPerfImageCreate),
    TEST(OCLPerfImageReadsRGBA),
    TEST(OCLPerf3DImageWriteSpeed),
    TEST(OCLPerfImageReadWrite),
    TEST(OCLPerfSVMMemcpy),
    TEST(OCLPerfSVMMemFill),
    TEST(OCLPerfFillBuffer),
    TEST(OCLPerfFillImage),
    TEST(OCLPerfUncoalescedRead),
    TEST(OCLPerfGenoilSiaMiner),
    TEST(OCLPerfDevMemReadSpeed),
    TEST(OCLPerfDevMemWriteSpeed),
    TEST(OCLPerfVerticalFetch),
};

unsigned int TestListCount = sizeof(TestList) / sizeof(TestList[0]);
unsigned int TestLibVersion = 0;
const char* TestLibName = "oclperf";
