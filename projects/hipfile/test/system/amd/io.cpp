/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

#include "context.h"
#include "configuration.h"
#include "hipfile-literals.h"
#include "hipfile-warnings.h"
#include "hipfile.h"

#include "ais-capability.h"
#include "test-common.h"
#include "test-options.h"

#include <array>
#include <algorithm>
#include <bit>
#include <chrono>
#include <compare>
#include <cstddef>
#include <cstdint>
#include <errno.h>
#include <fcntl.h>
#include <gtest/gtest.h>
#include <hip/hip_runtime_api.h>
#include <latch>
#include <linux/stat.h>
#include <span>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>
#include <vector>

extern SystemTestOptions test_env;

using namespace hipFile;

namespace {

// Gate fastpath-only tests on AIS capability.
void
enforceFastpathGate(hipFileHandle_t handle, void *device_buffer)
{
    hipFile::test::AisCapability ais_capability{test_env.allow_skip_fastpath};

    const auto decision = ais_capability.populate(handle, device_buffer);

    if (decision == hipFile::test::AisCapability::GateDecision::Run) {
        return;
    }

    if (decision == hipFile::test::AisCapability::GateDecision::Skip) {
        // Keep this marker synchronized with test/CMakeLists.txt SKIP_REGULAR_EXPRESSION.
        GTEST_SKIP() << "fastpath not available in this environment\n" << ais_capability.report();
    }

    FAIL() << "Fastpath Validation Failed!\n" << ais_capability.report() << "\n" << ais_capability.skipHint();
}

}

HIPFILE_WARN_NO_GLOBAL_CTOR_OFF

enum class IoTestBackend {
    Fastpath,
    Fallback,
};

struct IoTestParam {
    IoTestBackend backend;
    std::string   name;
};

HIPFILE_WARN_NO_EXIT_DTOR_OFF
static std::array<IoTestParam, 2> io_test_params{
    {{IoTestBackend::Fastpath, "Fastpath"}, {IoTestBackend::Fallback, "Fallback"}}};
HIPFILE_WARN_NO_EXIT_DTOR_ON

struct HipFileIo : public testing::TestWithParam<IoTestParam> {

    Tmpfile              tmpfile;
    size_t               tmpfile_size;
    uint32_t             tmpfile_dio_offset_align;
    hipFileHandle_t      tmpfile_handle;
    void                *unregistered_device_buffer;
    size_t               unregistered_device_buffer_size;
    std::vector<uint8_t> host_buffer;

    HipFileIo()
        : tmpfile{test_env.ais_capable_dir}, tmpfile_size{1_MiB}, tmpfile_dio_offset_align{4_KiB},
          tmpfile_handle{nullptr}, unregistered_device_buffer{nullptr},
          unregistered_device_buffer_size{tmpfile_size}, host_buffer(tmpfile_size)
    {
#if defined(STATX_DIOALIGN)
        struct statx stx {};
        if (0 == statx(tmpfile.fd, "", AT_EMPTY_PATH, STATX_DIOALIGN, &stx) &&
            stx.stx_mask & STATX_DIOALIGN) {
            if (std::popcount(stx.stx_dio_offset_align) != 1) {
                throw std::runtime_error("Invalid statx dio offset");
            }
            tmpfile_dio_offset_align = stx.stx_dio_offset_align;
        }
#endif
    }

    void SetUp() override
    {
        // Disable all backends
        Context<Configuration>::get()->fastpath(false);
        Context<Configuration>::get()->fallback(false);

        // Enable the desired backend
        switch (GetParam().backend) {
            case IoTestBackend::Fastpath:
                Context<Configuration>::get()->fastpath(true);
                break;

            case IoTestBackend::Fallback:
                Context<Configuration>::get()->fallback(true);
                break;

            default:
                FAIL() << "Unsupported IoTestBackend";
        }

        ASSERT_EQ(0, ftruncate(tmpfile.fd, static_cast<off_t>(tmpfile_size)));

        hipFileDescr_t descr{};
        descr.type      = hipFileHandleTypeOpaqueFD;
        descr.handle.fd = tmpfile.fd;

        ASSERT_EQ(HIPFILE_SUCCESS, hipFileHandleRegister(&tmpfile_handle, &descr));

        ASSERT_EQ(hipSuccess, hipMalloc(&unregistered_device_buffer, unregistered_device_buffer_size));

        if (GetParam().backend == IoTestBackend::Fastpath) {
            enforceFastpathGate(tmpfile_handle, unregistered_device_buffer);
        }
    }

    void TearDown() override
    {
        ASSERT_EQ(hipSuccess, hipFree(unregistered_device_buffer));

        hipFileHandleDeregister(tmpfile_handle);
    }
};

TEST_P(HipFileIo, ReadToUnregisteredBufferAtOffset)
{
    hoff_t io_buffer_offset{4096};
    size_t io_size{unregistered_device_buffer_size - static_cast<size_t>(io_buffer_offset)};

    ASSERT_EQ(io_size, hipFileRead(tmpfile_handle, unregistered_device_buffer, io_size, 0, io_buffer_offset));
}

TEST_P(HipFileIo, ReadToUnregisteredBufferAtOffsetReturnsErrorIfOverflow)
{
    hoff_t io_buffer_offset{4096};
    size_t io_size{unregistered_device_buffer_size};

    ASSERT_EQ(-hipFileInvalidValue,
              hipFileRead(tmpfile_handle, unregistered_device_buffer, io_size, 0, io_buffer_offset));
}

TEST_P(HipFileIo, readAtNegativeFileOffsetReturnsEINVAL)
{
    errno = 0;
    ASSERT_EQ(-1, pread(tmpfile.fd, host_buffer.data(), host_buffer.size(), -1));
    ASSERT_EQ(EINVAL, errno);

    errno = 0;
    ASSERT_EQ(
        -1, hipFileRead(tmpfile_handle, unregistered_device_buffer, unregistered_device_buffer_size, -1, 0));
    ASSERT_EQ(EINVAL, errno);
}

TEST_P(HipFileIo, writeAtNegativeFileOffsetReturnsEINVAL)
{
    errno = 0;
    ASSERT_EQ(-1, pwrite(tmpfile.fd, host_buffer.data(), host_buffer.size(), -1));
    ASSERT_EQ(EINVAL, errno);

    errno = 0;
    ASSERT_EQ(
        -1, hipFileWrite(tmpfile_handle, unregistered_device_buffer, unregistered_device_buffer_size, -1, 0));
    ASSERT_EQ(EINVAL, errno);
}

TEST_P(HipFileIo, readAtNegativeBufferOffsetReturnsEINVAL)
{
    errno = 0;
    ASSERT_EQ(
        -1, hipFileRead(tmpfile_handle, unregistered_device_buffer, unregistered_device_buffer_size, 0, -1));
    ASSERT_EQ(EINVAL, errno);
}

TEST_P(HipFileIo, writeAtNegativeBufferOffsetReturnsEINVAL)
{
    errno = 0;
    ASSERT_EQ(
        -1, hipFileWrite(tmpfile_handle, unregistered_device_buffer, unregistered_device_buffer_size, 0, -1));
    ASSERT_EQ(EINVAL, errno);
}

// This test ensures that a hipFileRead into GPU memory is observable by the
// CPU. On XGMI-connected systems, the CPU may cache GPU memory, so this test
// verifies that CPU caches are properly invalidated. On PCIe-connected systems,
// the CPU does not cache GPU memory, so all reads from the CPU will go directly
// to GPU memory.
TEST_P(HipFileIo, ReadIntoVramIsObservableByCpu)
{
    // Initialize GPU data
    constexpr uint16_t    initial_value{0xFFFF};
    std::vector<uint16_t> test_buffer(unregistered_device_buffer_size / sizeof(initial_value), initial_value);
    ASSERT_EQ(hipSuccess, hipMemcpy(unregistered_device_buffer, test_buffer.data(),
                                    unregistered_device_buffer_size, hipMemcpyHostToDevice));
    ASSERT_EQ(hipSuccess, hipDeviceSynchronize());

    // Determine the CPU cache line size
    const long _cache_line_size{sysconf(_SC_LEVEL1_DCACHE_LINESIZE)};
    ASSERT_LT(0U, _cache_line_size);
    const size_t cache_line_size{static_cast<size_t>(_cache_line_size)};
    ASSERT_EQ(reinterpret_cast<uintptr_t>(unregistered_device_buffer) % cache_line_size, 0);

    // Initialize file data so that when read into GPU memory, each cache line
    // will begin with a unique value
    ASSERT_EQ(cache_line_size % sizeof(uint16_t), 0);
    const size_t cache_line_stride{cache_line_size / sizeof(uint16_t)};
    for (size_t i{0}; i * cache_line_stride < test_buffer.size(); i++) {
        test_buffer[i * cache_line_stride] = static_cast<uint16_t>(i);
    }
    {
        auto nbytes{std::span(test_buffer).size_bytes()};
        ASSERT_EQ(static_cast<ssize_t>(nbytes), pwrite(tmpfile.fd, test_buffer.data(), nbytes, 0));
    }

    // Number of watchers observing GPU memory (ideally one per core)
    const size_t watcher_count{
        std::min(std::max(1U, std::thread::hardware_concurrency()),
                 static_cast<unsigned>(unregistered_device_buffer_size / cache_line_size))};
    ASSERT_LT(0U, watcher_count);

    // Used by watcher threads to signal the main thread that they are ready to observe the read
    std::latch watchers_ready{static_cast<std::ptrdiff_t>(watcher_count)};

    // Watcher timeout for observing the new value in GPU memory
    constexpr auto watcher_timeout{std::chrono::seconds{5}};

    // Time to wait after all watchers are ready before initiating the read into GPU memory
    constexpr auto wait_after_watchers_ready{std::chrono::milliseconds{500}};
    ASSERT_LT(wait_after_watchers_ready, watcher_timeout);

    {
        std::vector<std::jthread> watcher_threads;

        for (size_t i{0}; i < watcher_count; i++) {
            // Each watcher observes a value in GPU memory which maps to a unique cache line
            auto value{static_cast<volatile uint16_t *>(unregistered_device_buffer) + i * cache_line_stride};

            watcher_threads.emplace_back([i, value, &watchers_ready, watcher_timeout]() {
                if (*value != initial_value) {
                    ADD_FAILURE() << "Watcher[" << i << "] did not observe initial value";
                }

                watchers_ready.arrive_and_wait();

                std::chrono::steady_clock::time_point start_time{std::chrono::steady_clock::now()};
                do {
                    for (size_t j{0}; j < 1'000'000; j++) {
                        auto observed{*value};
                        if (observed == static_cast<uint16_t>(i)) {
                            return;
                        }
                        else if (observed != initial_value) {
                            ADD_FAILURE() << "Watcher[" << i
                                          << "] observed an unexpected value: " << static_cast<int>(observed);
                            return;
                        }
                    }
                } while (std::chrono::steady_clock::now() - start_time < watcher_timeout);
                ADD_FAILURE() << "Watcher[" << i << "] did not observe the new byte within timeout";
            });
        }

        // Once all watchers are ready, give them a moment to start observing
        watchers_ready.wait();
        std::this_thread::sleep_for(wait_after_watchers_ready);

        // Read the file into GPU memory
        ASSERT_EQ(unregistered_device_buffer_size, hipFileRead(tmpfile_handle, unregistered_device_buffer,
                                                               unregistered_device_buffer_size, 0, 0));
    }
}

// Zero-sized IO tests require >= ROCm 7.14
#if HIP_VERSION_MAJOR > 7 || (HIP_VERSION_MAJOR == 7 && HIP_VERSION_MINOR >= 14)
TEST_P(HipFileIo, zeroSizedReadAtAlignedFileOffsetReturnsZero)
{
    for (hoff_t offset = 0; offset < static_cast<hoff_t>(tmpfile_size); offset += tmpfile_dio_offset_align) {
        ASSERT_EQ(0, pread(tmpfile.fd, host_buffer.data(), 0, offset));
        ASSERT_EQ(0, hipFileRead(tmpfile_handle, unregistered_device_buffer, 0, offset, 0));
    }
}

TEST_P(HipFileIo, zeroSizedWriteAtAlignedFileOffsetReturnsZero)
{
    for (hoff_t offset = 0; offset < static_cast<hoff_t>(tmpfile_size); offset += tmpfile_dio_offset_align) {
        ASSERT_EQ(0, pwrite(tmpfile.fd, host_buffer.data(), 0, offset));
        ASSERT_EQ(0, hipFileWrite(tmpfile_handle, unregistered_device_buffer, 0, offset, 0));
    }
}

TEST_P(HipFileIo, zeroSizedReadAtAlignedOffsetBeyondEndOfFileReturnsZero)
{
    size_t aligned_eof{align_up(tmpfile_size + 1, tmpfile_dio_offset_align)};
    for (size_t i = 0; i < 10; i++) {
        hoff_t offset{static_cast<hoff_t>(aligned_eof * i)};
        ASSERT_EQ(0, pread(tmpfile.fd, host_buffer.data(), 0, offset));
        ASSERT_EQ(0, hipFileRead(tmpfile_handle, unregistered_device_buffer, 0, offset, 0));

        // ensure the file size has not increased
        struct stat st {};
        ASSERT_EQ(0, fstat(tmpfile.fd, &st));
        ASSERT_EQ(tmpfile_size, static_cast<size_t>(st.st_size));
    }
}

TEST_P(HipFileIo, zeroSizedWriteAtAlignedOffsetBeyondEndOfFileReturnsZero)
{
    size_t aligned_eof{align_up(tmpfile_size + 1, tmpfile_dio_offset_align)};
    for (size_t i = 0; i < 10; i++) {
        hoff_t offset{static_cast<hoff_t>(aligned_eof * i)};
        ASSERT_EQ(0, pwrite(tmpfile.fd, host_buffer.data(), 0, static_cast<off_t>(offset)));
        ASSERT_EQ(0, hipFileWrite(tmpfile_handle, unregistered_device_buffer, 0, offset, 0));

        // ensure the file size has not increased
        struct stat st {};
        ASSERT_EQ(0, fstat(tmpfile.fd, &st));
        ASSERT_EQ(tmpfile_size, static_cast<size_t>(st.st_size));
    }
}
#endif

INSTANTIATE_TEST_SUITE_P(, HipFileIo, testing::ValuesIn(io_test_params),
                         [](const testing::TestParamInfo<HipFileIo::ParamType> &param_info) {
                             return param_info.param.name;
                         });

struct HipFileIoHipInit : public testing::Test {

    Tmpfile         tmpfile;
    size_t          tmpfile_size;
    hipFileHandle_t tmpfile_handle;
    void           *registered_device_buffer;
    size_t          registered_device_buffer_size;

    HipFileIoHipInit()
        : tmpfile{test_env.ais_capable_dir}, tmpfile_size{1024 * 1024}, tmpfile_handle{nullptr},
          registered_device_buffer{nullptr}, registered_device_buffer_size{1024 * 1024}
    {
    }

    void SetUp() override
    {
        Context<Configuration>::get()->fastpath(true);
        Context<Configuration>::get()->fallback(false);

        ASSERT_EQ(0, ftruncate(tmpfile.fd, static_cast<off_t>(tmpfile_size)));

        hipFileDescr_t descr{};
        descr.type      = hipFileHandleTypeOpaqueFD;
        descr.handle.fd = tmpfile.fd;

        ASSERT_EQ(HIPFILE_SUCCESS, hipFileHandleRegister(&tmpfile_handle, &descr));
        ASSERT_EQ(hipSuccess, hipMalloc(&registered_device_buffer, registered_device_buffer_size));
        ASSERT_EQ(HIPFILE_SUCCESS,
                  hipFileBufRegister(registered_device_buffer, registered_device_buffer_size, 0));

        enforceFastpathGate(tmpfile_handle, registered_device_buffer);
    }

    void TearDown() override
    {
        ASSERT_EQ(HIPFILE_SUCCESS, hipFileBufDeregister(registered_device_buffer));
        ASSERT_EQ(hipSuccess, hipFree(registered_device_buffer));
        hipFileHandleDeregister(tmpfile_handle);
    }
};

TEST_F(HipFileIoHipInit, spawnedThreadReadRunsWithoutSegfault)
{
    size_t  io_size{registered_device_buffer_size};
    ssize_t res{};
    std::thread([&]() { res = hipFileRead(tmpfile_handle, registered_device_buffer, io_size, 0, 0); }).join();
    ASSERT_EQ(io_size, res);
}

TEST_F(HipFileIoHipInit, spawnedThreadWriteRunsWithoutSegfault)
{
    size_t  io_size{registered_device_buffer_size};
    ssize_t res{};
    std::thread([&]() {
        res = hipFileWrite(tmpfile_handle, registered_device_buffer, io_size, 0, 0);
    }).join();
    ASSERT_EQ(io_size, res);
}

HIPFILE_WARN_NO_GLOBAL_CTOR_ON
