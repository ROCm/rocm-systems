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

#include "KFDIoUringTest.hpp"

#include <linux/io_uring.h>
#include <linux/kfd_ioctl.h>
#include <stdatomic.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <cerrno>
#include <chrono>
#include <cstdint>
#include <thread>

#include "kfdcontext.h"

extern HsaKFDContext hsakmt_primary_kfd_ctx;

// ============================================================================
// Raw io_uring helpers (no liburing dependency)
// ============================================================================

struct KFDIoUringTest::IoUring {
    int ring_fd;
    // SQ ring mmap
    void* sq_ring_ptr;
    size_t sq_ring_sz;
    uint32_t* sq_head;
    uint32_t* sq_tail;
    uint32_t* sq_ring_mask;
    uint32_t* sq_array;
    // CQ ring mmap
    void* cq_ring_ptr;
    size_t cq_ring_sz;
    uint32_t* cq_head;
    uint32_t* cq_tail;
    uint32_t* cq_ring_mask;
    struct io_uring_cqe* cqes;
    // SQEs mmap
    struct io_uring_sqe* sqes;
    size_t sqes_sz;
};

int KFDIoUringTest::IoUringSetup(uint32_t entries, struct io_uring_params* p) {
    return (int)syscall(__NR_io_uring_setup, entries, p);
}

int KFDIoUringTest::IoUringEnter(int ring_fd, uint32_t to_submit,
                                uint32_t min_complete, uint32_t flags) {
    return (int)syscall(__NR_io_uring_enter, ring_fd, to_submit, min_complete,
                        flags, nullptr, 0);
}

int KFDIoUringTest::IoUringInit(IoUring* ring, uint32_t entries) {
    struct io_uring_params params;
    memset(&params, 0, sizeof(params));
    params.flags = IORING_SETUP_SQE128;

    int fd = IoUringSetup(entries, &params);
    if (fd < 0) return -errno;

    ring->ring_fd = fd;

    // Map SQ ring
    ring->sq_ring_sz = params.sq_off.array +
                       params.sq_entries * sizeof(uint32_t);
    ring->sq_ring_ptr =
        mmap(nullptr, ring->sq_ring_sz, PROT_READ | PROT_WRITE,
             MAP_SHARED | MAP_POPULATE, fd, IORING_OFF_SQ_RING);
    if (ring->sq_ring_ptr == MAP_FAILED) {
        close(fd);
        return -errno;
    }

    ring->sq_head =
        (uint32_t*)((char*)ring->sq_ring_ptr + params.sq_off.head);
    ring->sq_tail =
        (uint32_t*)((char*)ring->sq_ring_ptr + params.sq_off.tail);
    ring->sq_ring_mask =
        (uint32_t*)((char*)ring->sq_ring_ptr + params.sq_off.ring_mask);
    ring->sq_array =
        (uint32_t*)((char*)ring->sq_ring_ptr + params.sq_off.array);

    // Map CQ ring
    ring->cq_ring_sz = params.cq_off.cqes +
                       params.cq_entries * sizeof(struct io_uring_cqe);
    ring->cq_ring_ptr =
        mmap(nullptr, ring->cq_ring_sz, PROT_READ | PROT_WRITE,
             MAP_SHARED | MAP_POPULATE, fd, IORING_OFF_CQ_RING);
    if (ring->cq_ring_ptr == MAP_FAILED) {
        munmap(ring->sq_ring_ptr, ring->sq_ring_sz);
        close(fd);
        return -errno;
    }

    ring->cq_head =
        (uint32_t*)((char*)ring->cq_ring_ptr + params.cq_off.head);
    ring->cq_tail =
        (uint32_t*)((char*)ring->cq_ring_ptr + params.cq_off.tail);
    ring->cq_ring_mask =
        (uint32_t*)((char*)ring->cq_ring_ptr + params.cq_off.ring_mask);
    ring->cqes = (struct io_uring_cqe*)((char*)ring->cq_ring_ptr +
                                         params.cq_off.cqes);

    // Map SQEs (128 bytes each with SQE128)
    ring->sqes_sz = params.sq_entries * SQE128_SIZE;
    ring->sqes = (struct io_uring_sqe*)mmap(
        nullptr, ring->sqes_sz, PROT_READ | PROT_WRITE,
        MAP_SHARED | MAP_POPULATE, fd, IORING_OFF_SQES);
    if (ring->sqes == MAP_FAILED) {
        munmap(ring->cq_ring_ptr, ring->cq_ring_sz);
        munmap(ring->sq_ring_ptr, ring->sq_ring_sz);
        close(fd);
        return -errno;
    }

    return 0;
}

void KFDIoUringTest::IoUringDestroy(IoUring* ring) {
    if (ring->sqes)
        munmap(ring->sqes, ring->sqes_sz);
    if (ring->cq_ring_ptr)
        munmap(ring->cq_ring_ptr, ring->cq_ring_sz);
    if (ring->sq_ring_ptr)
        munmap(ring->sq_ring_ptr, ring->sq_ring_sz);
    if (ring->ring_fd >= 0)
        close(ring->ring_fd);
    memset(ring, 0, sizeof(*ring));
    ring->ring_fd = -1;
}

struct io_uring_sqe* KFDIoUringTest::UringGetSqe(IoUring* ring) {
    uint32_t tail = *ring->sq_tail;
    uint32_t idx = tail & *ring->sq_ring_mask;
    struct io_uring_sqe* sqe =
        (struct io_uring_sqe*)((char*)ring->sqes + idx * SQE128_SIZE);
    return sqe;
}

void KFDIoUringTest::UringPrepCmd(IoUring* ring,
                                   struct io_uring_sqe* sqe, int fd,
                                   uint32_t cmd_op, const void* payload,
                                   size_t payload_sz) {
    memset(sqe, 0, SQE128_SIZE);
    sqe->opcode = IORING_OP_URING_CMD;
    sqe->fd = fd;
    sqe->cmd_op = cmd_op;
    sqe->user_data = 0xABCD;
    if (payload && payload_sz > 0) {
        memcpy(sqe->cmd, payload, payload_sz);
    }
    ring->sq_array[*ring->sq_tail & *ring->sq_ring_mask] =
        *ring->sq_tail & *ring->sq_ring_mask;
    __atomic_store_n(ring->sq_tail, *ring->sq_tail + 1, __ATOMIC_RELEASE);
}

int KFDIoUringTest::UringSubmitAndWait(IoUring* ring) {
    int ret = IoUringEnter(ring->ring_fd, 1, 1, IORING_ENTER_GETEVENTS);
    if (ret < 0) return -errno;

    uint32_t head = __atomic_load_n(ring->cq_head, __ATOMIC_ACQUIRE);
    if (head == *ring->cq_tail) return -EAGAIN;

    uint32_t idx = head & *ring->cq_ring_mask;
    int result = ring->cqes[idx].res;

    __atomic_store_n(ring->cq_head, head + 1, __ATOMIC_RELEASE);
    return result;
}

int KFDIoUringTest::UringSubmitNw(IoUring* ring) {
    int ret = IoUringEnter(ring->ring_fd, 1, 0, 0);
    if (ret < 0) return -errno;
    return 0;
}

int KFDIoUringTest::UringWaitCqe(IoUring* ring) {
    uint32_t head = __atomic_load_n(ring->cq_head, __ATOMIC_ACQUIRE);
    while (head == __atomic_load_n(ring->cq_tail, __ATOMIC_ACQUIRE)) {
        int ret = IoUringEnter(ring->ring_fd, 0, 1, IORING_ENTER_GETEVENTS);
        if (ret < 0) return -errno;
        head = __atomic_load_n(ring->cq_head, __ATOMIC_ACQUIRE);
    }

    uint32_t idx = head & *ring->cq_ring_mask;
    int result = ring->cqes[idx].res;
    __atomic_store_n(ring->cq_head, head + 1, __ATOMIC_RELEASE);
    return result;
}

int KFDIoUringTest::SubmitUringCmdSync(uint32_t cmd_op,
                                        const void* payload,
                                        size_t payload_sz) {
    IoUring ring;
    int ret = IoUringInit(&ring, 4);
    if (ret < 0) return ret;

    struct io_uring_sqe* sqe = UringGetSqe(&ring);
    UringPrepCmd(&ring, sqe, m_KfdFd, cmd_op, payload, payload_sz);

    ret = UringSubmitAndWait(&ring);
    IoUringDestroy(&ring);
    return ret;
}

// ============================================================================
// Signal memory helpers
// ============================================================================

bool KFDIoUringTest::AllocSignalMemory(volatile int64_t** signal_ptr,
                                        int* dmabuf_fd,
                                        uint64_t* dmabuf_offset,
                                        int64_t initial_value) {
    HsaMemFlags memFlags = GetHsaMemFlags();
    memFlags.ui32.NonPaged = 1;

    void* buf = nullptr;
    HSAKMT_STATUS status = hsaKmtAllocMemory(0, PAGE_SIZE, memFlags, &buf);
    if (status != HSAKMT_STATUS_SUCCESS)
        return false;

    status = hsaKmtMapMemoryToGPU(buf, PAGE_SIZE, nullptr);
    if (status != HSAKMT_STATUS_SUCCESS) {
        hsaKmtFreeMemory(buf, PAGE_SIZE);
        return false;
    }

    int fd = -1;
    uint64_t offset = 0;
    status = hsaKmtExportDMABufHandle(buf, sizeof(int64_t), &fd, &offset);
    if (status != HSAKMT_STATUS_SUCCESS) {
        hsaKmtUnmapMemoryToGPU(buf);
        hsaKmtFreeMemory(buf, PAGE_SIZE);
        return false;
    }

    *(volatile int64_t*)buf = initial_value;

    *signal_ptr = (volatile int64_t*)buf;
    *dmabuf_fd = fd;
    *dmabuf_offset = offset;
    return true;
}

void KFDIoUringTest::FreeSignalMemory(volatile int64_t* signal_ptr) {
    if (signal_ptr) {
        hsaKmtUnmapMemoryToGPU(const_cast<int64_t*>(signal_ptr));
        hsaKmtFreeMemory(const_cast<int64_t*>(signal_ptr), PAGE_SIZE);
    }
}

// ============================================================================
// KFDIoUringTest implementation
// ============================================================================

void KFDIoUringTest::SetUp() {
    ROUTINE_START

    KFDBaseComponentTest::SetUp();

    m_KfdFd = -1;
    m_PrerequisitesMet = false;

    // Use the thunk's KFD fd so io_uring commands operate on the same
    // process context as hsaKmt* API calls (events, queues, memory).
    m_KfdFd = hsakmt_primary_kfd_ctx.fd;
    if (m_KfdFd < 0) {
        LOG() << "KFD not opened by thunk" << std::endl;
        return;
    }

    // Probe SQE128 io_uring support
    IoUring ring;
    int ret = IoUringInit(&ring, 4);
    if (ret < 0) {
        LOG() << "io_uring with SQE128 not supported: "
              << strerror(-ret) << std::endl;
        return;
    }
    IoUringDestroy(&ring);

    m_PrerequisitesMet = true;

    ROUTINE_END
}

void KFDIoUringTest::TearDown() {
    ROUTINE_START

    m_KfdFd = -1;

    KFDBaseComponentTest::TearDown();

    ROUTINE_END
}

// ============================================================================
// Signal Write Tests
// ============================================================================

void KFDIoUringTest::WriteSet(int gpuNode) {
    if (!m_PrerequisitesMet) {
        LOG() << "Prerequisites not met for io_uring tests" << std::endl;
        return;
    }

    volatile int64_t* signal;
    int dmabuf_fd;
    uint64_t dmabuf_offset;
    const int64_t initial = 100;
    const int64_t write_value = 42;

    ASSERT_TRUE(AllocSignalMemory(&signal, &dmabuf_fd,
                                   &dmabuf_offset, initial));

    HsaEvent* event = nullptr;
    ASSERT_SUCCESS_GPU(CreateQueueTypeEvent(false, false, gpuNode, &event),
                       gpuNode);

    struct kfd_uring_cmd_signal_write cmd;
    memset(&cmd, 0, sizeof(cmd));
    cmd.event_id = event->EventId;
    cmd.dmabuf_fd = dmabuf_fd;
    cmd.dmabuf_offset = dmabuf_offset;
    cmd.signal_write_value = write_value;
    cmd.flags = 0;
    cmd.memory_op = KFD_SIGNAL_WRITE_SET;

    int result = SubmitUringCmdSync(KFD_URING_CMD_WRITE_SIGNAL,
                                    &cmd, sizeof(cmd));
    ASSERT_EQ_GPU(0, result, gpuNode)
        << "uring write SET failed: " << strerror(-result);

    int64_t current = *signal;
    EXPECT_EQ_GPU(write_value, current, gpuNode)
        << "Signal should be " << write_value << " after SET, got " << current;

    EXPECT_SUCCESS(hsaKmtDestroyEvent(event));
    close(dmabuf_fd);
    FreeSignalMemory(signal);
}

TEST_F(KFDIoUringTest, WriteSet) {
    TEST_START(TESTPROFILE_RUNALL);

    ASSERT_SUCCESS(KFDTestLaunch([this](int gpuNode) {
        this->WriteSet(gpuNode);
    }));

    TEST_END
}

void KFDIoUringTest::WriteInc(int gpuNode) {
    if (!m_PrerequisitesMet) {
        LOG() << "Prerequisites not met for io_uring tests" << std::endl;
        return;
    }

    volatile int64_t* signal;
    int dmabuf_fd;
    uint64_t dmabuf_offset;
    const int64_t initial = 10;
    const int64_t inc_value = 5;

    ASSERT_TRUE(AllocSignalMemory(&signal, &dmabuf_fd,
                                   &dmabuf_offset, initial));

    HsaEvent* event = nullptr;
    ASSERT_SUCCESS_GPU(CreateQueueTypeEvent(false, false, gpuNode, &event),
                       gpuNode);

    struct kfd_uring_cmd_signal_write cmd;
    memset(&cmd, 0, sizeof(cmd));
    cmd.event_id = event->EventId;
    cmd.dmabuf_fd = dmabuf_fd;
    cmd.dmabuf_offset = dmabuf_offset;
    cmd.signal_write_value = inc_value;
    cmd.flags = 0;
    cmd.memory_op = KFD_SIGNAL_WRITE_INC;

    int result = SubmitUringCmdSync(KFD_URING_CMD_WRITE_SIGNAL,
                                    &cmd, sizeof(cmd));
    ASSERT_EQ_GPU(0, result, gpuNode)
        << "uring write INC failed: " << strerror(-result);

    int64_t current = *signal;
    EXPECT_EQ_GPU(initial + inc_value, current, gpuNode)
        << "Signal should be " << (initial + inc_value) << " after INC";

    EXPECT_SUCCESS(hsaKmtDestroyEvent(event));
    close(dmabuf_fd);
    FreeSignalMemory(signal);
}

TEST_F(KFDIoUringTest, WriteInc) {
    TEST_START(TESTPROFILE_RUNALL);

    ASSERT_SUCCESS(KFDTestLaunch([this](int gpuNode) {
        this->WriteInc(gpuNode);
    }));

    TEST_END
}

void KFDIoUringTest::WriteDec(int gpuNode) {
    if (!m_PrerequisitesMet) {
        LOG() << "Prerequisites not met for io_uring tests" << std::endl;
        return;
    }

    volatile int64_t* signal;
    int dmabuf_fd;
    uint64_t dmabuf_offset;
    const int64_t initial = 100;
    const int64_t dec_value = 30;

    ASSERT_TRUE(AllocSignalMemory(&signal, &dmabuf_fd,
                                   &dmabuf_offset, initial));

    HsaEvent* event = nullptr;
    ASSERT_SUCCESS_GPU(CreateQueueTypeEvent(false, false, gpuNode, &event),
                       gpuNode);

    struct kfd_uring_cmd_signal_write cmd;
    memset(&cmd, 0, sizeof(cmd));
    cmd.event_id = event->EventId;
    cmd.dmabuf_fd = dmabuf_fd;
    cmd.dmabuf_offset = dmabuf_offset;
    cmd.signal_write_value = dec_value;
    cmd.flags = 0;
    cmd.memory_op = KFD_SIGNAL_WRITE_DEC;

    int result = SubmitUringCmdSync(KFD_URING_CMD_WRITE_SIGNAL,
                                    &cmd, sizeof(cmd));
    ASSERT_EQ_GPU(0, result, gpuNode)
        << "uring write DEC failed: " << strerror(-result);

    int64_t current = *signal;
    EXPECT_EQ_GPU(initial - dec_value, current, gpuNode)
        << "Signal should be " << (initial - dec_value) << " after DEC";

    EXPECT_SUCCESS(hsaKmtDestroyEvent(event));
    close(dmabuf_fd);
    FreeSignalMemory(signal);
}

TEST_F(KFDIoUringTest, WriteDec) {
    TEST_START(TESTPROFILE_RUNALL);

    ASSERT_SUCCESS(KFDTestLaunch([this](int gpuNode) {
        this->WriteDec(gpuNode);
    }));

    TEST_END
}

void KFDIoUringTest::WriteDoorbell(int gpuNode) {
    if (!m_PrerequisitesMet) {
        LOG() << "Prerequisites not met for io_uring tests" << std::endl;
        return;
    }

    // Allocate AQL queue ring buffer (zero-initialized, executable, uncached)
    const unsigned int queueSize = PAGE_SIZE;
    HsaMemoryBuffer ringBuf(queueSize, gpuNode, true/*zero*/, false/*local*/,
                             true/*exec*/, false/*isScratch*/, false/*isReadOnly*/,
                             true/*isUncached*/);

    // Allocate read/write pointer buffer for AQL queue
    HsaMemoryBuffer pointers(PAGE_SIZE, gpuNode, true/*zero*/, false/*local*/,
                              false/*exec*/, false/*isScratch*/, false/*isReadOnly*/,
                              false/*isUncached*/,
                              NeedNonPagedWptr(gpuNode));

    // Allocate completion signal memory (mimics amd_signal_t, 64-byte aligned).
    // Layout: kind (uint64_t, offset 0) | value (int64_t, offset 8) | ...
    // CP firmware decrements value by 1 when the packet completes.
    HsaMemoryBuffer signalBuf(PAGE_SIZE, gpuNode, true/*zero*/, false/*local*/,
                               false/*exec*/, false/*isScratch*/, false/*isReadOnly*/,
                               true/*isUncached*/);
    volatile uint64_t* signalMem = signalBuf.As<volatile uint64_t*>();
    signalMem[0] = 1;  // kind = AMD_SIGNAL_KIND_USER
    unsigned int initialSignalVal = (rand() % 100) + 1;  // random value in [1, 100]
    signalMem[1] = initialSignalVal;  // CP decrements by 1 on completion

    // Set up queue resources with AQL-style pointers
    HSAuint64* ptrBuf = pointers.As<HSAuint64*>();
    HsaQueueResource res;
    memset(&res, 0, sizeof(res));
    res.Queue_read_ptr_aql = &ptrBuf[0];
    res.Queue_write_ptr_aql = &ptrBuf[1];

    ASSERT_SUCCESS_GPU(hsaKmtCreateQueue(gpuNode,
                                          HSA_QUEUE_COMPUTE_AQL,
                                          100,
                                          HSA_QUEUE_PRIORITY_NORMAL,
                                          ringBuf.As<unsigned int*>(),
                                          ringBuf.Size(),
                                          NULL,
                                          &res), gpuNode);

    // Write an AQL barrier-AND packet (64 bytes) directly into the ring buffer.
    // A barrier-AND with no dependencies completes immediately.
    //
    // hsa_barrier_and_packet_t layout:
    //   offset  0: uint16_t header (type=3 barrier-AND)
    //   offset  8: dep_signal[5] (all 0 = no dependencies)
    //   offset 56: completion_signal.handle (pointer to amd_signal_t)
    uint8_t* pkt = ringBuf.As<uint8_t*>();

    // Set completion_signal.handle at offset 56 to our signal object
    uint64_t signalHandle = (uint64_t)(uintptr_t)signalMem;
    memcpy(pkt + 56, &signalHandle, sizeof(signalHandle));

    // Set header last with atomic release (makes packet visible to CP)
    const uint16_t AQL_BARRIER_AND_HEADER = 3;
    __atomic_store_n((uint16_t*)pkt, AQL_BARRIER_AND_HEADER, __ATOMIC_RELEASE);

    // Update write pointer: AQL wptr counts packets (not dwords)
    const uint64_t wptr = 1;
    MemoryBarrier();
    *res.Queue_write_ptr_aql = wptr;

    // Ring the doorbell via io_uring instead of writing to it directly.
    // res.QueueId is a pointer to the thunk's internal struct queue cast to
    // uint64. The real kernel queue_id is the first field (uint32_t at offset 0).
    struct kfd_uring_cmd_signal_write cmd;
    memset(&cmd, 0, sizeof(cmd));
    cmd.queue_id = *(uint32_t*)(uintptr_t)res.QueueId;
    cmd.signal_write_value = (int64_t)wptr;
    cmd.flags = KFD_WRITE_SIGNAL_DOORBELL;
    cmd.memory_op = KFD_SIGNAL_WRITE_SET;

    int result = SubmitUringCmdSync(KFD_URING_CMD_WRITE_SIGNAL,
                                    &cmd, sizeof(cmd));
    ASSERT_EQ_GPU(0, result, gpuNode)
        << "uring doorbell write failed: " << strerror(-result);

    // Wait for CP to process the packet: signal value decremented by 1
    EXPECT_TRUE_GPU(WaitOnValue((unsigned int*)&signalMem[1],
                                initialSignalVal - 1), gpuNode);

    EXPECT_SUCCESS_GPU(hsaKmtDestroyQueue(res.QueueId), gpuNode);
}

TEST_F(KFDIoUringTest, WriteDoorbell) {
    TEST_START(TESTPROFILE_RUNALL);

    ASSERT_SUCCESS(KFDTestLaunch([this](int gpuNode) {
        this->WriteDoorbell(gpuNode);
    }));

    TEST_END
}

// ============================================================================
// Signal Wait Tests
// ============================================================================

void KFDIoUringTest::WaitEq(int gpuNode) {
    if (!m_PrerequisitesMet) {
        LOG() << "Prerequisites not met for io_uring tests" << std::endl;
        return;
    }

    volatile int64_t* signal;
    int dmabuf_fd;
    uint64_t dmabuf_offset;
    const int64_t initial = 0;
    const int64_t target = 42;

    ASSERT_TRUE(AllocSignalMemory(&signal, &dmabuf_fd,
                                   &dmabuf_offset, initial));

    HsaEvent* event = nullptr;
    ASSERT_SUCCESS_GPU(CreateQueueTypeEvent(false, false, gpuNode, &event),
                       gpuNode);

    struct kfd_uring_event_wait_signal waiter;
    memset(&waiter, 0, sizeof(waiter));
    waiter.event_id = event->EventId;
    waiter.dmabuf_fd = dmabuf_fd;
    waiter.dmabuf_offset = dmabuf_offset;
    waiter.test_value = target;
    waiter.compare_op = KFD_EVENT_WAIT_SIGNAL_CMP_EQ;

    struct kfd_uring_cmd_event cmd;
    memset(&cmd, 0, sizeof(cmd));
    cmd.event_waiter_array = (uint64_t)&waiter;
    cmd.num_events = 1;
    cmd.flags = 0;
    cmd.timeout_ns = 5000000000ULL;  // 5s

    // Submit wait asynchronously
    IoUring ring;
    ASSERT_EQ_GPU(0, IoUringInit(&ring, 4), gpuNode);

    struct io_uring_sqe* sqe = UringGetSqe(&ring);
    UringPrepCmd(&ring, sqe, m_KfdFd, KFD_URING_CMD_EVENT_WAIT_SIGNAL,
                 &cmd, sizeof(cmd));

    int ret = UringSubmitNw(&ring);
    ASSERT_EQ_GPU(0, ret, gpuNode)
        << "Failed to submit wait: " << strerror(-ret);

    // Write the target value and signal the event
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    *signal = target;

    EXPECT_SUCCESS_GPU(hsaKmtSetEvent(event), gpuNode);

    int result = UringWaitCqe(&ring);
    EXPECT_EQ_GPU(0, result, gpuNode)
        << "Wait EQ should succeed, got: " << result;

    IoUringDestroy(&ring);
    EXPECT_SUCCESS(hsaKmtDestroyEvent(event));
    close(dmabuf_fd);
    FreeSignalMemory(signal);
}

TEST_F(KFDIoUringTest, WaitEq) {
    TEST_START(TESTPROFILE_RUNALL);

    ASSERT_SUCCESS(KFDTestLaunch([this](int gpuNode) {
        this->WaitEq(gpuNode);
    }));

    TEST_END
}

void KFDIoUringTest::WaitNe(int gpuNode) {
    if (!m_PrerequisitesMet) {
        LOG() << "Prerequisites not met for io_uring tests" << std::endl;
        return;
    }

    volatile int64_t* signal;
    int dmabuf_fd;
    uint64_t dmabuf_offset;
    const int64_t initial = 0;

    ASSERT_TRUE(AllocSignalMemory(&signal, &dmabuf_fd,
                                   &dmabuf_offset, initial));

    HsaEvent* event = nullptr;
    ASSERT_SUCCESS_GPU(CreateQueueTypeEvent(false, false, gpuNode, &event),
                       gpuNode);

    struct kfd_uring_event_wait_signal waiter;
    memset(&waiter, 0, sizeof(waiter));
    waiter.event_id = event->EventId;
    waiter.dmabuf_fd = dmabuf_fd;
    waiter.dmabuf_offset = dmabuf_offset;
    waiter.test_value = 0;  // wait for signal != 0
    waiter.compare_op = KFD_EVENT_WAIT_SIGNAL_CMP_NE;

    struct kfd_uring_cmd_event cmd;
    memset(&cmd, 0, sizeof(cmd));
    cmd.event_waiter_array = (uint64_t)&waiter;
    cmd.num_events = 1;
    cmd.flags = 0;
    cmd.timeout_ns = 5000000000ULL;

    IoUring ring;
    ASSERT_EQ_GPU(0, IoUringInit(&ring, 4), gpuNode);

    struct io_uring_sqe* sqe = UringGetSqe(&ring);
    UringPrepCmd(&ring, sqe, m_KfdFd, KFD_URING_CMD_EVENT_WAIT_SIGNAL,
                 &cmd, sizeof(cmd));
    ASSERT_EQ_GPU(0, UringSubmitNw(&ring), gpuNode);

    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    *signal = 99;

    EXPECT_SUCCESS_GPU(hsaKmtSetEvent(event), gpuNode);

    int result = UringWaitCqe(&ring);
    EXPECT_EQ_GPU(0, result, gpuNode)
        << "Wait NE should succeed, got: " << result;

    IoUringDestroy(&ring);
    EXPECT_SUCCESS(hsaKmtDestroyEvent(event));
    close(dmabuf_fd);
    FreeSignalMemory(signal);
}

TEST_F(KFDIoUringTest, WaitNe) {
    TEST_START(TESTPROFILE_RUNALL);

    ASSERT_SUCCESS(KFDTestLaunch([this](int gpuNode) {
        this->WaitNe(gpuNode);
    }));

    TEST_END
}

void KFDIoUringTest::WaitLtLeGtGe(int gpuNode) {
    if (!m_PrerequisitesMet) {
        LOG() << "Prerequisites not met for io_uring tests" << std::endl;
        return;
    }

    struct CompareTest {
        uint8_t compare_op;
        const char* name;
        int64_t initial;
        int64_t test_value;
        int64_t satisfy_value;
    } tests[] = {
        {KFD_EVENT_WAIT_SIGNAL_CMP_LT, "LT", 50, 30, 20},
        {KFD_EVENT_WAIT_SIGNAL_CMP_LE, "LE", 50, 30, 30},
        {KFD_EVENT_WAIT_SIGNAL_CMP_GT, "GT", 10, 50, 60},
        {KFD_EVENT_WAIT_SIGNAL_CMP_GE, "GE", 10, 50, 50},
    };

    for (const auto& t : tests) {
        LOG() << "  Testing compare op: " << t.name << std::endl;

        volatile int64_t* signal;
        int dmabuf_fd;
        uint64_t dmabuf_offset;

        ASSERT_TRUE(AllocSignalMemory(&signal, &dmabuf_fd,
                                       &dmabuf_offset, t.initial));

        HsaEvent* event = nullptr;
        ASSERT_SUCCESS_GPU(CreateQueueTypeEvent(false, false, gpuNode, &event),
                           gpuNode);

        struct kfd_uring_event_wait_signal waiter;
        memset(&waiter, 0, sizeof(waiter));
        waiter.event_id = event->EventId;
        waiter.dmabuf_fd = dmabuf_fd;
        waiter.dmabuf_offset = dmabuf_offset;
        waiter.test_value = t.test_value;
        waiter.compare_op = t.compare_op;

        struct kfd_uring_cmd_event cmd;
        memset(&cmd, 0, sizeof(cmd));
        cmd.event_waiter_array = (uint64_t)&waiter;
        cmd.num_events = 1;
        cmd.flags = 0;
        cmd.timeout_ns = 5000000000ULL;

        IoUring ring;
        ASSERT_EQ_GPU(0, IoUringInit(&ring, 4), gpuNode);

        struct io_uring_sqe* sqe = UringGetSqe(&ring);
        UringPrepCmd(&ring, sqe, m_KfdFd, KFD_URING_CMD_EVENT_WAIT_SIGNAL,
                     &cmd, sizeof(cmd));
        ASSERT_EQ_GPU(0, UringSubmitNw(&ring), gpuNode);

        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        *signal = t.satisfy_value;

        EXPECT_SUCCESS_GPU(hsaKmtSetEvent(event), gpuNode);

        int result = UringWaitCqe(&ring);
        EXPECT_EQ_GPU(0, result, gpuNode)
            << "Wait " << t.name << " should succeed, got: " << result;

        IoUringDestroy(&ring);
        EXPECT_SUCCESS(hsaKmtDestroyEvent(event));
        close(dmabuf_fd);
        FreeSignalMemory(signal);
    }
}

TEST_F(KFDIoUringTest, WaitLtLeGtGe) {
    TEST_START(TESTPROFILE_RUNALL);

    ASSERT_SUCCESS(KFDTestLaunch([this](int gpuNode) {
        this->WaitLtLeGtGe(gpuNode);
    }));

    TEST_END
}

void KFDIoUringTest::WaitTimeout(int gpuNode) {
    if (!m_PrerequisitesMet) {
        LOG() << "Prerequisites not met for io_uring tests" << std::endl;
        return;
    }

    volatile int64_t* signal;
    int dmabuf_fd;
    uint64_t dmabuf_offset;

    ASSERT_TRUE(AllocSignalMemory(&signal, &dmabuf_fd,
                                   &dmabuf_offset, 0));

    HsaEvent* event = nullptr;
    ASSERT_SUCCESS_GPU(CreateQueueTypeEvent(false, false, gpuNode, &event),
                       gpuNode);

    // Wait for signal == 999 -- will never be satisfied
    struct kfd_uring_event_wait_signal waiter;
    memset(&waiter, 0, sizeof(waiter));
    waiter.event_id = event->EventId;
    waiter.dmabuf_fd = dmabuf_fd;
    waiter.dmabuf_offset = dmabuf_offset;
    waiter.test_value = 999;
    waiter.compare_op = KFD_EVENT_WAIT_SIGNAL_CMP_EQ;

    struct kfd_uring_cmd_event cmd;
    memset(&cmd, 0, sizeof(cmd));
    cmd.event_waiter_array = (uint64_t)&waiter;
    cmd.num_events = 1;
    cmd.flags = 0;
    cmd.timeout_ns = 200000000ULL;  // 200ms

    int result = SubmitUringCmdSync(KFD_URING_CMD_EVENT_WAIT_SIGNAL,
                                    &cmd, sizeof(cmd));
    EXPECT_EQ_GPU(-ETIMEDOUT, result, gpuNode)
        << "Wait should timeout, got: " << result
        << " (" << strerror(-result) << ")";

    EXPECT_SUCCESS(hsaKmtDestroyEvent(event));
    close(dmabuf_fd);
    FreeSignalMemory(signal);
}

TEST_F(KFDIoUringTest, WaitTimeout) {
    TEST_START(TESTPROFILE_RUNALL);

    ASSERT_SUCCESS(KFDTestLaunch([this](int gpuNode) {
        this->WaitTimeout(gpuNode);
    }));

    TEST_END
}

void KFDIoUringTest::WaitAny(int gpuNode) {
    if (!m_PrerequisitesMet) {
        LOG() << "Prerequisites not met for io_uring tests" << std::endl;
        return;
    }

    const int NUM_SIGNALS = 3;
    volatile int64_t* signals[NUM_SIGNALS];
    int dmabuf_fds[NUM_SIGNALS];
    uint64_t dmabuf_offsets[NUM_SIGNALS];
    HsaEvent* events[NUM_SIGNALS];
    struct kfd_uring_event_wait_signal waiters[NUM_SIGNALS];

    for (int i = 0; i < NUM_SIGNALS; i++) {
        ASSERT_TRUE(AllocSignalMemory(&signals[i], &dmabuf_fds[i],
                                       &dmabuf_offsets[i], 0));
        events[i] = nullptr;
        ASSERT_SUCCESS_GPU(CreateQueueTypeEvent(false, false, gpuNode,
                                                &events[i]), gpuNode);

        memset(&waiters[i], 0, sizeof(waiters[i]));
        waiters[i].event_id = events[i]->EventId;
        waiters[i].dmabuf_fd = dmabuf_fds[i];
        waiters[i].dmabuf_offset = dmabuf_offsets[i];
        waiters[i].test_value = 1;
        waiters[i].compare_op = KFD_EVENT_WAIT_SIGNAL_CMP_EQ;
    }

    struct kfd_uring_cmd_event cmd;
    memset(&cmd, 0, sizeof(cmd));
    cmd.event_waiter_array = (uint64_t)waiters;
    cmd.num_events = NUM_SIGNALS;
    cmd.flags = KFD_EVENT_WAIT_SIGNAL_ANY;
    cmd.timeout_ns = 5000000000ULL;

    IoUring ring;
    ASSERT_EQ_GPU(0, IoUringInit(&ring, 4), gpuNode);

    struct io_uring_sqe* sqe = UringGetSqe(&ring);
    UringPrepCmd(&ring, sqe, m_KfdFd, KFD_URING_CMD_EVENT_WAIT_SIGNAL,
                 &cmd, sizeof(cmd));
    ASSERT_EQ_GPU(0, UringSubmitNw(&ring), gpuNode);

    // Satisfy only the second signal
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    *signals[1] = 1;

    EXPECT_SUCCESS_GPU(hsaKmtSetEvent(events[1]), gpuNode);

    int result = UringWaitCqe(&ring);
    EXPECT_EQ_GPU(0, result, gpuNode)
        << "Wait ANY should succeed when 1 of 3 satisfied";

    IoUringDestroy(&ring);
    for (int i = 0; i < NUM_SIGNALS; i++) {
        EXPECT_SUCCESS(hsaKmtDestroyEvent(events[i]));
        close(dmabuf_fds[i]);
        FreeSignalMemory(signals[i]);
    }
}

TEST_F(KFDIoUringTest, WaitAny) {
    TEST_START(TESTPROFILE_RUNALL);

    ASSERT_SUCCESS(KFDTestLaunch([this](int gpuNode) {
        this->WaitAny(gpuNode);
    }));

    TEST_END
}

void KFDIoUringTest::WaitAll(int gpuNode) {
    if (!m_PrerequisitesMet) {
        LOG() << "Prerequisites not met for io_uring tests" << std::endl;
        return;
    }

    const int NUM_SIGNALS = 3;
    volatile int64_t* signals[NUM_SIGNALS];
    int dmabuf_fds[NUM_SIGNALS];
    uint64_t dmabuf_offsets[NUM_SIGNALS];
    HsaEvent* events[NUM_SIGNALS];
    struct kfd_uring_event_wait_signal waiters[NUM_SIGNALS];

    for (int i = 0; i < NUM_SIGNALS; i++) {
        ASSERT_TRUE(AllocSignalMemory(&signals[i], &dmabuf_fds[i],
                                       &dmabuf_offsets[i], 0));
        events[i] = nullptr;
        ASSERT_SUCCESS_GPU(CreateQueueTypeEvent(false, false, gpuNode,
                                                &events[i]), gpuNode);

        memset(&waiters[i], 0, sizeof(waiters[i]));
        waiters[i].event_id = events[i]->EventId;
        waiters[i].dmabuf_fd = dmabuf_fds[i];
        waiters[i].dmabuf_offset = dmabuf_offsets[i];
        waiters[i].test_value = 1;
        waiters[i].compare_op = KFD_EVENT_WAIT_SIGNAL_CMP_EQ;
    }

    struct kfd_uring_cmd_event cmd;
    memset(&cmd, 0, sizeof(cmd));
    cmd.event_waiter_array = (uint64_t)waiters;
    cmd.num_events = NUM_SIGNALS;
    cmd.flags = 0;  // wait for ALL
    cmd.timeout_ns = 5000000000ULL;

    IoUring ring;
    ASSERT_EQ_GPU(0, IoUringInit(&ring, 4), gpuNode);

    struct io_uring_sqe* sqe = UringGetSqe(&ring);
    UringPrepCmd(&ring, sqe, m_KfdFd, KFD_URING_CMD_EVENT_WAIT_SIGNAL,
                 &cmd, sizeof(cmd));
    ASSERT_EQ_GPU(0, UringSubmitNw(&ring), gpuNode);

    // Satisfy all signals one by one
    for (int i = 0; i < NUM_SIGNALS; i++) {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        *signals[i] = 1;

        EXPECT_SUCCESS_GPU(hsaKmtSetEvent(events[i]), gpuNode);
    }

    int result = UringWaitCqe(&ring);
    EXPECT_EQ_GPU(0, result, gpuNode)
        << "Wait ALL should succeed when all satisfied";

    IoUringDestroy(&ring);
    for (int i = 0; i < NUM_SIGNALS; i++) {
        EXPECT_SUCCESS(hsaKmtDestroyEvent(events[i]));
        close(dmabuf_fds[i]);
        FreeSignalMemory(signals[i]);
    }
}

TEST_F(KFDIoUringTest, WaitAll) {
    TEST_START(TESTPROFILE_RUNALL);

    ASSERT_SUCCESS(KFDTestLaunch([this](int gpuNode) {
        this->WaitAll(gpuNode);
    }));

    TEST_END
}

// ============================================================================
// Combined Write + Wait Test
// ============================================================================

void KFDIoUringTest::WriteTriggersWait(int gpuNode) {
    if (!m_PrerequisitesMet) {
        LOG() << "Prerequisites not met for io_uring tests" << std::endl;
        return;
    }

    volatile int64_t* signal;
    int dmabuf_fd;
    uint64_t dmabuf_offset;
    const int64_t initial = 0;
    const int64_t target = 77;

    ASSERT_TRUE(AllocSignalMemory(&signal, &dmabuf_fd,
                                   &dmabuf_offset, initial));

    HsaEvent* event = nullptr;
    ASSERT_SUCCESS_GPU(CreateQueueTypeEvent(false, false, gpuNode, &event),
                       gpuNode);

    // Submit wait asynchronously: wait for signal == 77
    struct kfd_uring_event_wait_signal waiter;
    memset(&waiter, 0, sizeof(waiter));
    waiter.event_id = event->EventId;
    waiter.dmabuf_fd = dmabuf_fd;
    waiter.dmabuf_offset = dmabuf_offset;
    waiter.test_value = target;
    waiter.compare_op = KFD_EVENT_WAIT_SIGNAL_CMP_EQ;

    struct kfd_uring_cmd_event wait_cmd;
    memset(&wait_cmd, 0, sizeof(wait_cmd));
    wait_cmd.event_waiter_array = (uint64_t)&waiter;
    wait_cmd.num_events = 1;
    wait_cmd.flags = 0;
    wait_cmd.timeout_ns = 5000000000ULL;

    IoUring wait_ring;
    ASSERT_EQ_GPU(0, IoUringInit(&wait_ring, 4), gpuNode);

    struct io_uring_sqe* sqe = UringGetSqe(&wait_ring);
    UringPrepCmd(&wait_ring, sqe, m_KfdFd,
                 KFD_URING_CMD_EVENT_WAIT_SIGNAL, &wait_cmd,
                 sizeof(wait_cmd));
    ASSERT_EQ_GPU(0, UringSubmitNw(&wait_ring), gpuNode);

    // Let the wait get registered
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    // Now submit a write via uring that sets signal to 77 and notifies event
    struct kfd_uring_cmd_signal_write write_cmd;
    memset(&write_cmd, 0, sizeof(write_cmd));
    write_cmd.event_id = event->EventId;
    write_cmd.dmabuf_fd = dmabuf_fd;
    write_cmd.dmabuf_offset = dmabuf_offset;
    write_cmd.signal_write_value = target;
    write_cmd.flags = 0;
    write_cmd.memory_op = KFD_SIGNAL_WRITE_SET;

    int write_result = SubmitUringCmdSync(
        KFD_URING_CMD_WRITE_SIGNAL, &write_cmd, sizeof(write_cmd));
    ASSERT_EQ_GPU(0, write_result, gpuNode) << "Write command failed";

    // The wait should now complete
    int wait_result = UringWaitCqe(&wait_ring);
    EXPECT_EQ_GPU(0, wait_result, gpuNode)
        << "Wait should succeed after uring write";

    // Verify signal value
    int64_t current = *signal;
    EXPECT_EQ_GPU(target, current, gpuNode);

    IoUringDestroy(&wait_ring);
    EXPECT_SUCCESS(hsaKmtDestroyEvent(event));
    close(dmabuf_fd);
    FreeSignalMemory(signal);
}

TEST_F(KFDIoUringTest, WriteTriggersWait) {
    TEST_START(TESTPROFILE_RUNALL);

    ASSERT_SUCCESS(KFDTestLaunch([this](int gpuNode) {
        this->WriteTriggersWait(gpuNode);
    }));

    TEST_END
}
