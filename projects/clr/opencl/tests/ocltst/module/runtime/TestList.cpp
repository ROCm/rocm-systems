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
#include "OCLAsyncMap.h"
#include "OCLAsyncTransfer.h"
#include "OCLAtomicCounter.h"
#include "OCLBlitKernel.h"
#include "OCLBufferFromImage.h"
#include "OCLCPUGuardPages.h"
#include "OCLCreateBuffer.h"
#include "OCLCreateContext.h"
#include "OCLCreateImage.h"
#include "OCLDeviceAtomic.h"
#include "OCLDeviceQueries.h"
#include "OCLDynamic.h"
#include "OCLDynamicBLines.h"
#include "OCLGenericAddressSpace.h"
#include "OCLGetQueueThreadID.h"
#include "OCLGlobalOffset.h"
#include "OCLImage2DFromBuffer.h"
#include "OCLImageCopyPartial.h"
#include "OCLKernelBinary.h"
#include "OCLLDS32K.h"
#include "OCLLinearFilter.h"
#include "OCLMapCount.h"
#include "OCLMemDependency.h"
#include "OCLMemObjs.h"
#include "OCLMemoryInfo.h"
#include "OCLMultiQueue.h"
#include "OCLOfflineCompilation.h"
#include "OCLP2PBuffer.h"
#include "OCLPartialWrkgrp.h"
#include "OCLPerfCounters.h"
#include "OCLPersistent.h"
#include "OCLPinnedMemory.h"
#include "OCLPlatformAtomics.h"
#include "OCLProgramScopeVariables.h"
#include "OCLRTQueue.h"
#include "OCLReadWriteImage.h"
#include "OCLSDI.h"
#include "OCLSVM.h"
#include "OCLSemaphore.h"
#include "OCLStablePState.h"
#include "OCLThreadTrace.h"
#include "OCLUnalignedCopy.h"
#include "OCLCreatePipe.h"

//
//  Helper macro for adding tests
//
template <typename T> static void* dictionary_CreateTestFunc(void) { return new T(); }

#define TEST(name) {#name, &dictionary_CreateTestFunc<name>}

TestEntry TestList[] = {
    TEST(OCLCreateContext),
    TEST(OCLAtomicCounter),
    TEST(OCLKernelBinary),
    TEST(OCLGlobalOffset),
    TEST(OCLLinearFilter),
    TEST(OCLAsyncTransfer),
    TEST(OCLLDS32K),
    TEST(OCLMemObjs),
    TEST(OCLSemaphore),
    TEST(OCLPartialWrkgrp),
    TEST(OCLCreateBuffer),
    TEST(OCLCreateImage),
    TEST(OCLCPUGuardPages),
    TEST(OCLMapCount),
    TEST(OCLMemoryInfo),
    TEST(OCLOfflineCompilation),
    TEST(OCLMemDependency),
    TEST(OCLGetQueueThreadID),
    TEST(OCLDeviceQueries),
    TEST(OCLSDI),
    TEST(OCLThreadTrace),
    TEST(OCLMultiQueue),
    TEST(OCLImage2DFromBuffer),
    TEST(OCLBufferFromImage),
    TEST(OCLPerfCounters),
    TEST(OCLSVM),
    TEST(OCLProgramScopeVariables),
    TEST(OCLGenericAddressSpace),
    TEST(OCLDynamic),
    TEST(OCLPlatformAtomics),
    TEST(OCLDeviceAtomic),
    TEST(OCLDynamicBLines),
    TEST(OCLUnalignedCopy),
    TEST(OCLBlitKernel),
    TEST(OCLRTQueue),
    TEST(OCLAsyncMap),
    TEST(OCLPinnedMemory),
    TEST(OCLReadWriteImage),
    TEST(OCLStablePState),
    TEST(OCLP2PBuffer),

    // Disabled until new Windows driver release that contains the
    // clCreatePipe changes
    // TEST(OCLCreatePipe),

    // Failures in Linux. IOL doesn't support tiling aperture and Cypress linear
    // image writes TEST(OCLPersistent),
};

unsigned int TestListCount = sizeof(TestList) / sizeof(TestList[0]);
unsigned int TestLibVersion = 0;
const char* TestLibName = "oclruntime";
