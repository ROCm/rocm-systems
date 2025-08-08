/*
 * Copyright (C) 2021 Advanced Micro Devices, Inc. All Rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
 * OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 *
 */

#ifndef _SHADERSTORE_H_
#define _SHADERSTORE_H_

#include <vector>

/* KFDASMTest List */
extern const std::vector<const char*> ShaderList;

/* Common */
extern const char *NoopIsa;
extern const char *CopyDwordIsa;
extern const char *InfiniteLoopIsa;
extern const char *AtomicIncIsa;

/* KFDMemoryTest */
extern const char *ScratchCopyDwordIsa;
extern const char *PollMemoryIsa;
extern const char *PollNCMemoryIsa;
extern const char *CopyOnSignalIsa;
extern const char *PollAndCopyIsa;
extern const char *WriteFlagAndValueIsa;
extern const char *WriteAndSignalIsa;
extern const char *WriteReleaseVectorIsa;
extern const char *WriteReleaseScalarIsa;
extern const char *ReadAcquireVectorIsa;
extern const char *ReadAcquireScalarIsa;
extern const char *FlushBufferForAcquireReleaseIsa;

/* KFDQMTest */
extern const char *LoopIsa;
extern const char *CheckCuMaskIsa;

/* KFDCWSRTest */
extern const char *PersistentIterateIsa;

/* KFDEvictTest */
extern const char *ReadMemoryIsa;

/* KFDGWSTest */
extern const char *GwsInitIsa;
extern const char *GwsAtomicIncreaseIsa;

/* HitTrapEvent */
extern const char *JumpToTrapIsa;
extern const char *TrapHandlerIsa;

/* HitWatchPointEvent */
extern const char *WatchReadIsa;
extern const char *WatchWriteIsa;
extern const char *WatchAtomicIsa;

#endif  // _SHADERSTORE_H_
