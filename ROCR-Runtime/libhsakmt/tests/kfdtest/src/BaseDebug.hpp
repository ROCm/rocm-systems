/*
 * Copyright (C) 2023 Advanced Micro Devices, Inc. All Rights Reserved.
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

#ifndef __KFD_BASE_DEBUG__H__
#define __KFD_BASE_DEBUG__H__

#include "hsakmt/hsakmt.h"
#include <poll.h>
#include <stdlib.h>

// @class BaseDebug
class BaseDebug {
 public:
    BaseDebug(void);
    virtual ~BaseDebug(void);

    HSAKMT_STATUS Attach(struct kfd_runtime_info *rInfo,
                         int rInfoSize,
                         unsigned int pid,
                         uint64_t exceptionEnable);

    void Detach(void);
    HSAKMT_STATUS SendRuntimeEvent(uint64_t exceptions, int gpuId, int queueId);
    HSAKMT_STATUS QueryDebugEvent(uint64_t *exceptions,
                                  uint32_t *gpuId, uint32_t *queueId,
                                  int timeoutMsec);
    void SetExceptionsEnabled(uint64_t exceptions);
    HSAKMT_STATUS SuspendQueues(unsigned int *numQueues, HSA_QUEUEID *queues, uint32_t *queueIds,
                                uint64_t exceptionsToClear);
    HSAKMT_STATUS ResumeQueues(unsigned int *numQueues, HSA_QUEUEID *queues, uint32_t *queueIds);
    HSAKMT_STATUS QueueSnapshot(uint64_t exceptionsToClear, uint64_t snapshotBufAddr,
                                uint32_t *numSnapshots);
    HSAKMT_STATUS DeviceSnapshot(uint64_t exceptionsToClear, uint64_t snapshotBuffAddr,
                                 uint32_t *numSnapshots);
    HSAKMT_STATUS SetWaveLaunchOverride(int mode, uint32_t *enableMask, uint32_t *supportMask);
    HSAKMT_STATUS SetAddressWatch(uint64_t address, int mode, uint64_t mask, uint32_t gpuId, uint32_t *id);
    HSAKMT_STATUS ClearAddressWatch(uint32_t gpuId, uint32_t id);
    HSAKMT_STATUS SetFlags(uint32_t *flags);

 private:
    unsigned int m_Pid;
    struct pollfd m_Fd;
    const char *m_Fd_Name = "/tmp/dbg_fifo";
};

#endif  // __KFD_BASE_DEBUG__H__
