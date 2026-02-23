/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
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

#ifndef __KFD_IO_URING_TEST__H__
#define __KFD_IO_URING_TEST__H__

#include <linux/io_uring.h>

#include "KFDBaseComponentTest.hpp"

class KFDIoUringTest : public KFDBaseComponentTest {
 public:
    KFDIoUringTest() {}
    ~KFDIoUringTest() {}

    // Signal write tests
    void WriteSet(int gpuNode);
    void WriteInc(int gpuNode);
    void WriteDec(int gpuNode);
    void WriteDoorbell(int gpuNode);

    // Signal wait tests
    void WaitEq(int gpuNode);
    void WaitNe(int gpuNode);
    void WaitLtLeGtGe(int gpuNode);
    void WaitTimeout(int gpuNode);
    void WaitAny(int gpuNode);
    void WaitAll(int gpuNode);

    // Combined write + wait
    void WriteTriggersWait(int gpuNode);

    virtual void SetUp() override;
    virtual void TearDown() override;

 protected:
    static const unsigned int EVENT_TIMEOUT = 5000;  // 5 seconds

 private:
    struct IoUring;

    static constexpr size_t SQE128_SIZE = 128;

    // Raw io_uring helpers (no liburing dependency)
    static int IoUringSetup(uint32_t entries, struct io_uring_params* p);
    static int IoUringEnter(int ring_fd, uint32_t to_submit,
                          uint32_t min_complete, uint32_t flags);
    int IoUringInit(IoUring* ring, uint32_t entries);
    void IoUringDestroy(IoUring* ring);
    struct io_uring_sqe* UringGetSqe(IoUring* ring);
    void UringPrepCmd(IoUring* ring, struct io_uring_sqe* sqe, int fd,
                      uint32_t cmd_op, const void* payload, size_t payload_sz);
    int UringSubmitAndWait(IoUring* ring);
    int UringSubmitNw(IoUring* ring);
    int UringWaitCqe(IoUring* ring);
    int SubmitUringCmdSync(uint32_t cmd_op, const void* payload,
                           size_t payload_sz);

    // Signal memory helpers
    bool AllocSignalMemory(volatile int64_t** signal_ptr, int* dmabuf_fd,
                           uint64_t* dmabuf_offset, int64_t initial_value);
    void FreeSignalMemory(volatile int64_t* signal_ptr);

    int m_KfdFd;
    bool m_PrerequisitesMet;
};

#endif  // __KFD_IO_URING_TEST__H__
