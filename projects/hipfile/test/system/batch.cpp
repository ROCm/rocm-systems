/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

#include "hipfile-data-ops.h"
#include "hipfile-literals.h"
#include "hipfile-warnings.h"
#include "hipfile.h"
#include "test-common.h"
#include "test-options.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <csignal>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <gtest/gtest.h>
#include <hip/hip_runtime_api.h>
#include <memory>
#include <sys/resource.h>
#include <sys/types.h>
#include <unistd.h>
#include <utility>
#include <vector>

extern SystemTestOptions test_env;

namespace {

struct BatchOpCookie {
    size_t index;
};

/// Owns a device allocation that is registered with hipFile for the lifetime of the object.
class RegisteredDeviceBuffer {
public:
    RegisteredDeviceBuffer()                                          = default;
    RegisteredDeviceBuffer(const RegisteredDeviceBuffer &)            = delete;
    RegisteredDeviceBuffer &operator=(const RegisteredDeviceBuffer &) = delete;
    RegisteredDeviceBuffer(RegisteredDeviceBuffer &&other) noexcept
        : ptr{std::exchange(other.ptr, nullptr)}, size{std::exchange(other.size, 0)}
    {
    }
    RegisteredDeviceBuffer &operator=(RegisteredDeviceBuffer &&other) noexcept
    {
        if (this != &other) {
            reset();
            ptr  = std::exchange(other.ptr, nullptr);
            size = std::exchange(other.size, 0);
        }
        return *this;
    }

    ~RegisteredDeviceBuffer()
    {
        reset();
    }

    void allocate(size_t bytes)
    {
        ASSERT_EQ(hipMalloc(&ptr, bytes), hipSuccess);
        ASSERT_EQ(hipMemset(ptr, 0, bytes), hipSuccess);
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
        ASSERT_EQ(hipFileBufRegister(ptr, bytes, 0), HIPFILE_SUCCESS);
        size = bytes;
    }

    void reset()
    {
        if (ptr == nullptr) {
            return;
        }
        EXPECT_EQ(hipFileBufDeregister(ptr), HIPFILE_SUCCESS);
        EXPECT_EQ(hipFree(ptr), hipSuccess);
        ptr  = nullptr;
        size = 0;
    }

    void *get() const
    {
        return ptr;
    }
    size_t bytes() const
    {
        return size;
    }

private:
    void  *ptr{};
    size_t size{};
};

/// Owns a device allocation that is deliberately never registered with hipFile.
class UnregisteredDeviceBuffer {
public:
    UnregisteredDeviceBuffer()                                            = default;
    UnregisteredDeviceBuffer(const UnregisteredDeviceBuffer &)            = delete;
    UnregisteredDeviceBuffer &operator=(const UnregisteredDeviceBuffer &) = delete;
    UnregisteredDeviceBuffer(UnregisteredDeviceBuffer &&other) noexcept
        : ptr{std::exchange(other.ptr, nullptr)}, size{std::exchange(other.size, 0)}
    {
    }
    UnregisteredDeviceBuffer &operator=(UnregisteredDeviceBuffer &&other) noexcept
    {
        if (this != &other) {
            reset();
            ptr  = std::exchange(other.ptr, nullptr);
            size = std::exchange(other.size, 0);
        }
        return *this;
    }

    ~UnregisteredDeviceBuffer()
    {
        reset();
    }

    void allocate(size_t bytes)
    {
        ASSERT_EQ(hipMalloc(&ptr, bytes), hipSuccess);
        ASSERT_EQ(hipMemset(ptr, 0, bytes), hipSuccess);
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
        size = bytes;
    }

    void reset()
    {
        if (ptr == nullptr) {
            return;
        }
        EXPECT_EQ(hipFree(ptr), hipSuccess);
        ptr  = nullptr;
        size = 0;
    }

    void *get() const
    {
        return ptr;
    }
    size_t bytes() const
    {
        return size;
    }

private:
    void  *ptr{};
    size_t size{};
};

struct BatchTest : public testing::Test {
    Tmpfile         tmpfile{test_env.ais_capable_dir};
    hipFileHandle_t file_handle{};
    void           *device_buffer{};
    size_t          op_size{4096};
    unsigned        op_count{4};
    size_t          file_size{op_size * op_count};

    size_t                     device_buffer_size{file_size};
    hipFileBatchHandle_t       batch_handle{};
    std::vector<uint8_t>       host_buffer;
    std::vector<BatchOpCookie> cookies;

    /// One separately allocated and registered device buffer per operation.
    std::vector<RegisteredDeviceBuffer> op_buffers;

    void SetUp() override
    {
        ASSERT_EQ(ftruncate(tmpfile.fd, static_cast<off_t>(file_size)), 0);

        hipFileDescr_t descr{};
        descr.type      = hipFileHandleTypeOpaqueFD;
        descr.handle.fd = tmpfile.fd;
        ASSERT_EQ(hipFileHandleRegister(&file_handle, &descr), HIPFILE_SUCCESS);

        ASSERT_EQ(hipMalloc(&device_buffer, device_buffer_size), hipSuccess);
        ASSERT_EQ(hipMemset(device_buffer, 0, device_buffer_size), hipSuccess);
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
        ASSERT_EQ(hipFileBufRegister(device_buffer, device_buffer_size, 0), HIPFILE_SUCCESS);

        host_buffer.assign(device_buffer_size, 0);
        cookies.resize(op_count);
        for (size_t i = 0; i < cookies.size(); ++i) {
            cookies[i].index = i;
        }
    }

    void TearDown() override
    {
        if (batch_handle != nullptr) {
            hipFileBatchIODestroy(batch_handle);
            batch_handle = nullptr;
        }
        op_buffers.clear();
        if (device_buffer != nullptr) {
            EXPECT_EQ(hipFileBufDeregister(device_buffer), HIPFILE_SUCCESS);
            EXPECT_EQ(hipFree(device_buffer), hipSuccess);
            device_buffer = nullptr;
        }
        if (file_handle != nullptr) {
            hipFileHandleDeregister(file_handle);
            file_handle = nullptr;
        }
        while (hipFileUseCount() > 0) {
            EXPECT_EQ(hipFileDriverClose(), HIPFILE_SUCCESS);
        }
    }

    void setupBatch(unsigned capacity)
    {
        ASSERT_EQ(hipFileBatchIOSetUp(&batch_handle, capacity), HIPFILE_SUCCESS);
        ASSERT_NE(batch_handle, nullptr);
    }

    hipFileIOParams_t makeOp(size_t index, hipFileOpcode_t opcode)
    {
        hipFileIOParams_t op{};
        op.mode                  = hipFileBatch;
        op.u.batch.devPtr_base   = device_buffer;
        op.u.batch.file_offset   = static_cast<int64_t>(index * op_size);
        op.u.batch.devPtr_offset = static_cast<int64_t>(index * op_size);
        op.u.batch.size          = op_size;
        op.fh                    = file_handle;
        op.opcode                = opcode;
        op.cookie                = &cookies[index];
        return op;
    }

    /// Allocates and registers one device buffer of @p op_size for each of @p count operations.
    void setupOpBuffers(size_t count)
    {
        op_buffers.clear();
        op_buffers.resize(count);
        for (auto &buffer : op_buffers) {
            ASSERT_NO_FATAL_FAILURE(buffer.allocate(op_size));
        }
    }

    /// Builds an op that reads/writes into the buffer dedicated to @p index.
    hipFileIOParams_t makeOpWithOwnBuffer(size_t index, hipFileOpcode_t opcode)
    {
        auto op                  = makeOp(index, opcode);
        op.u.batch.devPtr_base   = op_buffers.at(index).get();
        op.u.batch.devPtr_offset = 0;
        return op;
    }

    std::vector<hipFileIOEvents_t> waitForEvents(unsigned expected)
    {
        std::vector<hipFileIOEvents_t> events(expected);
        unsigned                       nr = expected;
        EXPECT_EQ(hipFileBatchIOGetStatus(batch_handle, expected, &nr, events.data(), nullptr),
                  HIPFILE_SUCCESS);
        events.resize(nr);
        return events;
    }

    void expectCompleteEvents(const std::vector<hipFileIOEvents_t> &events, unsigned expected)
    {
        ASSERT_EQ(events.size(), expected);
        std::vector<size_t> seen;
        seen.reserve(events.size());
        for (const auto &event : events) {
            ASSERT_NE(event.cookie, nullptr);
            const auto *cookie = static_cast<const BatchOpCookie *>(event.cookie);
            EXPECT_LT(cookie->index, op_count);
            seen.push_back(cookie->index);
            EXPECT_EQ(event.status, hipFileComplete);
            EXPECT_EQ(event.ret, op_size);
        }
        std::sort(seen.begin(), seen.end());
        ASSERT_EQ(std::adjacent_find(seen.begin(), seen.end()), seen.end());
    }
};

struct BatchWriteFailureTest : public BatchTest {
    void SetUp() override
    {
        file_size          = 1_MiB;
        device_buffer_size = 1_MiB;
#if defined(__HIP_PLATFORM_NVIDIA__)
        // This fixture has a test that constrains process file writes. Redirect cuFile logging to
        // a character device, which RLIMIT_FSIZE does not apply to, so cuFile's own logging cannot
        // fail alongside the target I/O. Only this fixture is affected; cuFile logging stays
        // enabled for every other test.
        ASSERT_EQ(setenv("CUFILE_LOGFILE_PATH", "/dev/null", 1), 0);
#endif
        BatchTest::SetUp();
    }
};

struct BatchCancelTest : public BatchTest {
    void SetUp() override
    {
        op_count           = 128;
        file_size          = op_size * op_count;
        device_buffer_size = file_size;
        BatchTest::SetUp();
    }
};

struct ScopedSignalAction {
    int ignore(int signal_number_)
    {
        signal_number = signal_number_;
        old_handler   = std::signal(signal_number, SIG_IGN);
        active        = old_handler != SIG_ERR;
        return active ? 0 : -1;
    }

    ~ScopedSignalAction()
    {
        if (active) {
            (void)std::signal(signal_number, old_handler);
        }
    }

    using SignalHandler = void (*)(int);

    int           signal_number{};
    SignalHandler old_handler{SIG_DFL};
    bool          active{};
};

struct ScopedFileSizeLimit {
    int limit(rlim_t soft_limit)
    {
        if (getrlimit(RLIMIT_FSIZE, &old_limit) != 0) {
            return -1;
        }

        struct rlimit new_limit = old_limit;
        new_limit.rlim_cur      = soft_limit;
        const int status        = setrlimit(RLIMIT_FSIZE, &new_limit);
        active                  = status == 0;
        return status;
    }

    ~ScopedFileSizeLimit()
    {
        if (active) {
            (void)setrlimit(RLIMIT_FSIZE, &old_limit);
        }
    }

    struct rlimit old_limit {};
    bool          active{};
};

} // namespace

HIPFILE_WARN_NO_GLOBAL_CTOR_OFF

TEST_F(BatchTest, BatchReadSingleOperation)
{
    HipFileDataOps::zeroMemoryRegion(device_buffer, 0, device_buffer_size);
    HipFileDataOps::randomizeFileRegion(tmpfile.fd, op_size);

    setupBatch(1);
    auto op = makeOp(0, hipFileBatchRead);
    ASSERT_EQ(hipFileBatchIOSubmit(batch_handle, 1, &op, 0), HIPFILE_SUCCESS);

    const auto events = waitForEvents(1);
    expectCompleteEvents(events, 1);

    HipFileDataOps::assertFileAndMemoryRegionsMatch(device_buffer, 0, tmpfile.fd, 0, op_size);
}

TEST_F(BatchTest, BatchReadMultipleOperations)
{
    HipFileDataOps::zeroMemoryRegion(device_buffer, 0, device_buffer_size);
    HipFileDataOps::randomizeFileRegion(tmpfile.fd, file_size);

    setupBatch(op_count);
    std::vector<hipFileIOParams_t> ops(op_count);
    for (size_t i = 0; i < ops.size(); ++i) {
        ops[i] = makeOp(i, hipFileBatchRead);
    }
    ASSERT_EQ(hipFileBatchIOSubmit(batch_handle, static_cast<unsigned>(ops.size()), ops.data(), 0),
              HIPFILE_SUCCESS);

    const auto events = waitForEvents(static_cast<unsigned>(ops.size()));
    expectCompleteEvents(events, static_cast<unsigned>(ops.size()));

    HipFileDataOps::assertFileAndMemoryRegionsMatch(device_buffer, 0, tmpfile.fd, 0, file_size);
}

TEST_F(BatchTest, BatchWriteSingleOperation)
{
    HipFileDataOps::zeroFileRegion(tmpfile.fd, op_size);
    HipFileDataOps::randomizeMemoryRegion(device_buffer, 0, op_size);

    setupBatch(1);
    auto op = makeOp(0, hipFileBatchWrite);
    ASSERT_EQ(hipFileBatchIOSubmit(batch_handle, 1, &op, 0), HIPFILE_SUCCESS);

    const auto events = waitForEvents(1);
    expectCompleteEvents(events, 1);
    HipFileDataOps::assertFileAndMemoryRegionsMatch(device_buffer, 0, tmpfile.fd, 0, op_size);
}

TEST_F(BatchTest, BatchWriteMultipleOperations)
{
    HipFileDataOps::zeroFileRegion(tmpfile.fd, file_size);
    HipFileDataOps::randomizeMemoryRegion(device_buffer, 0, file_size);

    setupBatch(op_count);
    std::vector<hipFileIOParams_t> ops(op_count);
    for (size_t i = 0; i < ops.size(); ++i) {
        ops[i] = makeOp(i, hipFileBatchWrite);
    }
    ASSERT_EQ(hipFileBatchIOSubmit(batch_handle, static_cast<unsigned>(ops.size()), ops.data(), 0),
              HIPFILE_SUCCESS);

    const auto events = waitForEvents(static_cast<unsigned>(ops.size()));
    expectCompleteEvents(events, static_cast<unsigned>(ops.size()));
    HipFileDataOps::assertFileAndMemoryRegionsMatch(device_buffer, 0, tmpfile.fd, 0, file_size);
}

TEST_F(BatchTest, ReusedBatchHandleAcceptsSequentialFullBatches)
{
    HipFileDataOps::randomizeFileRegion(tmpfile.fd, file_size);

    setupBatch(op_count);

    for (int round = 0; round < 2; ++round) {
        HipFileDataOps::zeroMemoryRegion(device_buffer, 0, device_buffer_size);

        std::vector<hipFileIOParams_t> ops(op_count);
        for (size_t i = 0; i < ops.size(); ++i) {
            ops[i] = makeOp(i, hipFileBatchRead);
        }
        ASSERT_EQ(hipFileBatchIOSubmit(batch_handle, static_cast<unsigned>(ops.size()), ops.data(), 0),
                  HIPFILE_SUCCESS)
            << "submit failed on round " << round;

        const auto events = waitForEvents(static_cast<unsigned>(ops.size()));
        expectCompleteEvents(events, static_cast<unsigned>(ops.size()));

        HipFileDataOps::assertFileAndMemoryRegionsMatch(device_buffer, 0, tmpfile.fd, 0, file_size);
    }
}

TEST_F(BatchWriteFailureTest, FailedBatchWriteReportsErrorInEventRet)
{
    ScopedSignalAction xfsz;
    ASSERT_EQ(xfsz.ignore(SIGXFSZ), 0);

    ScopedFileSizeLimit file_size_limit;
    ASSERT_EQ(file_size_limit.limit(static_cast<rlim_t>(file_size)), 0);

    // The failing operation deliberately uses an unregistered buffer. cuFile only reports the
    // underlying errno for operations serviced through its bounce-buffer path, which registered
    // buffers bypass; a registered buffer reports a generic -1 instead. That path also rejects an
    // operation at submit time once its size exceeds the per-buffer cache, so keep the write small.
    UnregisteredDeviceBuffer unregistered_buffer;
    ASSERT_NO_FATAL_FAILURE(unregistered_buffer.allocate(op_size));

    setupBatch(1);
    auto op                  = makeOp(0, hipFileBatchWrite);
    op.u.batch.devPtr_base   = unregistered_buffer.get();
    op.u.batch.devPtr_offset = 0;
    // Writing past the file size limit fails with EFBIG regardless of how small the write is.
    op.u.batch.file_offset = static_cast<hoff_t>(file_size);

    ASSERT_EQ(hipFileBatchIOSubmit(batch_handle, 1, &op, 0), HIPFILE_SUCCESS);

    const auto events = waitForEvents(1);
    ASSERT_EQ(events.size(), 1);

    const auto &event = events[0];
    ASSERT_EQ(event.cookie, &cookies[0]);
    EXPECT_EQ(event.status, hipFileFailed);

    const auto expected_ret = -static_cast<ssize_t>(EFBIG);
    EXPECT_EQ(static_cast<ssize_t>(event.ret), expected_ret);
    EXPECT_EQ(event.ret, static_cast<size_t>(expected_ret));
}

TEST_F(BatchTest, GetStatusWithSmallEventBufferReturnsRemainingEventsLater)
{
    HipFileDataOps::randomizeFileRegion(tmpfile.fd, file_size);

    setupBatch(op_count);
    std::vector<hipFileIOParams_t> ops(op_count);
    for (size_t i = 0; i < ops.size(); ++i) {
        ops[i] = makeOp(i, hipFileBatchRead);
    }
    ASSERT_EQ(hipFileBatchIOSubmit(batch_handle, static_cast<unsigned>(ops.size()), ops.data(), 0),
              HIPFILE_SUCCESS);

    std::vector<hipFileIOEvents_t> all_events;
    while (all_events.size() < op_count) {
        std::array<hipFileIOEvents_t, 2> events{};
        unsigned                         nr = events.size();
        ASSERT_EQ(hipFileBatchIOGetStatus(batch_handle, 1, &nr, events.data(), nullptr), HIPFILE_SUCCESS);
        ASSERT_GT(nr, 0);
        ASSERT_LE(nr, events.size());
        all_events.insert(all_events.end(), events.begin(), events.begin() + nr);
    }
    expectCompleteEvents(all_events, op_count);
}

TEST_F(BatchTest, GetStatusNoOutstandingReturnsZero)
{
    setupBatch(1);
    hipFileIOEvents_t event{};
    unsigned          nr = 1;
    struct timespec   timeout {
        1, 0
    };

    ASSERT_EQ(hipFileBatchIOGetStatus(batch_handle, 1, &nr, &event, &timeout), HIPFILE_SUCCESS);
    ASSERT_EQ(nr, 0);
}

TEST_F(BatchTest, GetStatusMinNrLargerThanSubmittedReturnsCompletedOps)
{
    HipFileDataOps::randomizeFileRegion(tmpfile.fd, op_size);

    setupBatch(op_count);
    auto op = makeOp(0, hipFileBatchRead);
    ASSERT_EQ(hipFileBatchIOSubmit(batch_handle, 1, &op, 0), HIPFILE_SUCCESS);

    // Ask for more events than were submitted; the wait is bounded to the outstanding ops.
    std::vector<hipFileIOEvents_t> events(op_count);
    unsigned                       nr = op_count;
    ASSERT_EQ(hipFileBatchIOGetStatus(batch_handle, op_count, &nr, events.data(), nullptr), HIPFILE_SUCCESS);
    ASSERT_EQ(nr, 1u);

    events.resize(nr);
    expectCompleteEvents(events, 1);
}

TEST_F(BatchTest, CancelEmptyBatchSucceeds)
{
    setupBatch(1);
    ASSERT_EQ(hipFileBatchIOCancel(batch_handle), HIPFILE_SUCCESS);

    hipFileIOEvents_t event{};
    unsigned          nr = 1;
    ASSERT_EQ(hipFileBatchIOGetStatus(batch_handle, 0, &nr, &event, nullptr), HIPFILE_SUCCESS);
    ASSERT_EQ(nr, 0);
}

TEST_F(BatchCancelTest, CancelFullBatchReportsTerminalStatus)
{
    HipFileDataOps::randomizeFileRegion(tmpfile.fd, file_size);

    setupBatch(op_count);
    std::vector<hipFileIOParams_t> ops(op_count);
    for (size_t i = 0; i < ops.size(); ++i) {
        ops[i] = makeOp(i, hipFileBatchRead);
    }
    ASSERT_EQ(hipFileBatchIOSubmit(batch_handle, static_cast<unsigned>(ops.size()), ops.data(), 0),
              HIPFILE_SUCCESS);

    ASSERT_EQ(hipFileBatchIOCancel(batch_handle), HIPFILE_SUCCESS);

    const auto events = waitForEvents(op_count);
#if defined(__HIP_PLATFORM_NVIDIA__)
    ASSERT_EQ(events.size(), 0);
    return;
#endif
    ASSERT_EQ(events.size(), op_count);
    std::vector<size_t> seen;
    seen.reserve(events.size());
    for (const auto &event : events) {
        ASSERT_NE(event.cookie, nullptr);
        const auto *cookie = static_cast<const BatchOpCookie *>(event.cookie);
        EXPECT_LT(cookie->index, op_count);
        seen.push_back(cookie->index);
        EXPECT_TRUE(event.status == hipFileCanceled || event.status == hipFileComplete)
            << "unexpected status " << event.status;
    }
    std::sort(seen.begin(), seen.end());
    ASSERT_EQ(std::adjacent_find(seen.begin(), seen.end()), seen.end());
}

TEST_F(BatchTest, DestroyEmptyBatchDoesNotCrash)
{
    setupBatch(1);
    hipFileBatchIODestroy(batch_handle);
    batch_handle = nullptr;

    hipFileBatchIODestroy(batch_handle);
}

TEST_F(BatchTest, SubmitRejectsInvalidArguments)
{
    setupBatch(1);
    auto                             op = makeOp(0, hipFileBatchRead);
    std::array<hipFileIOParams_t, 2> ops{op, makeOp(1, hipFileBatchRead)};

#if defined(__HIP_PLATFORM_NVIDIA__)
    constexpr auto invalid_submit_arg_error      = HipFileOpError(hipFileInternalError);
    constexpr auto invalid_submit_capacity_error = HipFileOpError(hipFileInternalError);
#else
    constexpr auto invalid_submit_arg_error      = HipFileOpError(hipFileInvalidValue);
    constexpr auto invalid_submit_capacity_error = HipFileOpError(hipFileBatchFull);
#endif
    ASSERT_EQ(hipFileBatchIOSubmit(batch_handle, 0, &op, 0), invalid_submit_arg_error);
    ASSERT_EQ(hipFileBatchIOSubmit(batch_handle, 1, nullptr, 0), invalid_submit_arg_error);
    ASSERT_EQ(hipFileBatchIOSubmit(batch_handle, ops.size(), ops.data(), 0), invalid_submit_capacity_error);
}

TEST_F(BatchTest, SubmitRejectsInvalidSizeInNonFirstOperation)
{
    setupBatch(op_count);
    ASSERT_NO_FATAL_FAILURE(setupOpBuffers(op_count));

    std::vector<hipFileIOParams_t> ops(op_count);
    for (size_t i = 0; i < ops.size(); ++i) {
        ops[i] = makeOpWithOwnBuffer(i, hipFileBatchRead);
    }
    // Give a non-first op a size that overruns the registered device buffer so validation must
    // inspect ops beyond the first to catch it.
    ops[1].u.batch.size = op_buffers[1].bytes() + op_size;

#if defined(__HIP_PLATFORM_NVIDIA__)
    constexpr auto retval = HipFileOpError(hipFileInternalError);
#else
    constexpr auto retval = HipFileOpError(hipFileInvalidValue);
#endif

    // Both backends validate every op at submit time and reject the whole batch, queuing nothing.
    // They differ only in the reported error code.
    ASSERT_EQ(hipFileBatchIOSubmit(batch_handle, static_cast<unsigned>(ops.size()), ops.data(), 0), retval);

    hipFileIOEvents_t event{};
    unsigned          nr = 1;
    struct timespec   timeout {
        1, 0
    };
    ASSERT_EQ(hipFileBatchIOGetStatus(batch_handle, 0, &nr, &event, &timeout), HIPFILE_SUCCESS);
    ASSERT_EQ(nr, 0);
}

TEST_F(BatchTest, SubmitRejectsMixedBatchWithUnregisteredFileHandle)
{
    setupBatch(2);
    ASSERT_NO_FATAL_FAILURE(setupOpBuffers(2));
    std::array<hipFileIOParams_t, 2> ops{makeOpWithOwnBuffer(0, hipFileBatchRead),
                                         makeOpWithOwnBuffer(1, hipFileBatchRead)};

    Tmpfile         unregistered_tmpfile{test_env.ais_capable_dir};
    hipFileDescr_t  unregistered_descr{};
    hipFileHandle_t unregistered_handle{};
    unregistered_descr.type      = hipFileHandleTypeOpaqueFD;
    unregistered_descr.handle.fd = unregistered_tmpfile.fd;
    ASSERT_EQ(ftruncate(unregistered_tmpfile.fd, static_cast<off_t>(op_size)), 0);
    ASSERT_EQ(hipFileHandleRegister(&unregistered_handle, &unregistered_descr), HIPFILE_SUCCESS);
    ops[1].fh = unregistered_handle;

    // Leave the first operation's handle registered and invalidate only the second operation.
    hipFileHandleDeregister(unregistered_handle);

#if defined(__HIP_PLATFORM_NVIDIA__)
    constexpr auto retval = HipFileOpError(hipFileInternalError);
#else
    constexpr auto retval = HipFileOpError(hipFileHandleNotRegistered);
#endif

    ASSERT_EQ(hipFileBatchIOSubmit(batch_handle, static_cast<unsigned>(ops.size()), ops.data(), 0), retval);

    std::array<hipFileIOEvents_t, 2> events{};
    unsigned                         nr = static_cast<unsigned>(events.size());
    struct timespec                  timeout {
        1, 0
    };
    ASSERT_EQ(hipFileBatchIOGetStatus(batch_handle, 0, &nr, events.data(), &timeout), HIPFILE_SUCCESS);
    ASSERT_EQ(nr, 0);
}

TEST_F(BatchTest, SubmitMixedBatchWithHostMemoryBuffer)
{
    setupBatch(2);
    ASSERT_NO_FATAL_FAILURE(setupOpBuffers(2));
    std::array<hipFileIOParams_t, 2> ops{makeOpWithOwnBuffer(0, hipFileBatchRead),
                                         makeOpWithOwnBuffer(1, hipFileBatchRead)};

    // Point only the second operation at host memory instead of the registered device buffer.
    ops[1].u.batch.devPtr_base = host_buffer.data();

#if defined(__HIP_PLATFORM_NVIDIA__)
    // cuFile falls back to a host-memory path instead of rejecting the buffer, so both ops
    // complete normally.
    ASSERT_EQ(hipFileBatchIOSubmit(batch_handle, static_cast<unsigned>(ops.size()), ops.data(), 0),
              HIPFILE_SUCCESS);
    const auto events = waitForEvents(static_cast<unsigned>(ops.size()));
    ASSERT_NO_FATAL_FAILURE(expectCompleteEvents(events, static_cast<unsigned>(ops.size())));
#else
    ASSERT_EQ(hipFileBatchIOSubmit(batch_handle, static_cast<unsigned>(ops.size()), ops.data(), 0),
              HipFileOpError(hipFileHipMemoryTypeInvalid));

    std::array<hipFileIOEvents_t, 2> events{};
    unsigned                         nr = static_cast<unsigned>(events.size());
    struct timespec                  timeout {
        1, 0
    };
    ASSERT_EQ(hipFileBatchIOGetStatus(batch_handle, 0, &nr, events.data(), &timeout), HIPFILE_SUCCESS);
    ASSERT_EQ(nr, 0);
#endif
}

TEST_F(BatchTest, SetUpRejectsZeroCapacity)
{
    hipFileBatchHandle_t handle = nullptr;
#if defined(__HIP_PLATFORM_NVIDIA__)
    constexpr auto retval = HipFileOpError(hipFileInternalError);
#else
    constexpr auto retval = HipFileOpError(hipFileInvalidValue);
#endif
    ASSERT_EQ(hipFileBatchIOSetUp(&handle, 0), retval);
    ASSERT_EQ(handle, nullptr);
}

TEST_F(BatchTest, SetUpRejectsCapacityAboveMaximum)
{
    // The batch context is limited to 128 outstanding ops; 129 must be rejected.
    hipFileBatchHandle_t handle = nullptr;
#if defined(__HIP_PLATFORM_NVIDIA__)
    constexpr auto retval = HipFileOpError(hipFileInternalError);
#else
    constexpr auto retval = HipFileOpError(hipFileInvalidValue);
#endif
    ASSERT_EQ(hipFileBatchIOSetUp(&handle, 129), retval);
    ASSERT_EQ(handle, nullptr);
}

TEST_F(BatchTest, GetStatusRejectsInvalidArguments)
{
    setupBatch(1);
    hipFileIOEvents_t event{};
    unsigned          nr = 1;

#if defined(__HIP_PLATFORM_NVIDIA__)
    constexpr auto invalid_status_arg_error = HipFileOpError(hipFileInternalError);
#else
    constexpr auto invalid_status_arg_error = HipFileOpError(hipFileInvalidValue);
#endif
    ASSERT_EQ(hipFileBatchIOGetStatus(batch_handle, 0, nullptr, &event, nullptr), invalid_status_arg_error);
    ASSERT_EQ(hipFileBatchIOGetStatus(batch_handle, 0, &nr, nullptr, nullptr), invalid_status_arg_error);
    ASSERT_EQ(hipFileBatchIOGetStatus(batch_handle, 2, &nr, &event, nullptr), invalid_status_arg_error);
    nr = 0;
    ASSERT_EQ(hipFileBatchIOGetStatus(batch_handle, 0, &nr, &event, nullptr), invalid_status_arg_error);
}

HIPFILE_WARN_NO_GLOBAL_CTOR_ON
