/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include "backend/fallback.h"
#include "hipfile-literals.h"
#include "hipfile-warnings.h"
#include "hipfile.h"

#include "verify-kernel.h"
#include "verify-pattern.h"
#include "verify-pattern-bytes.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <gtest/gtest.h>
#include <hip/hip_runtime_api.h>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

namespace hipFileTest {

// ---------------------------------------------------------------------------
// Transfer sizes and file offsets shared by the I/O verification tests.
// ---------------------------------------------------------------------------
constexpr size_t  kChunkBytes           = hipFile::Fallback::DefaultChunkSize;
constexpr int32_t kDefaultWorkgroupSize = 256;
constexpr hoff_t  kCombinedFileOff =
    static_cast<hoff_t>(kChunkBytes + 4_KiB); // exercises chunking and save room for file sentinel
constexpr hoff_t kChunkOff   = static_cast<hoff_t>(kChunkBytes);
constexpr hoff_t kFourKiBOff = static_cast<hoff_t>(4_KiB);

// ---------------------------------------------------------------------------
// Device sentinel region: values and sizing.
//
// Sentinel values for writing to device buffer regions to ensure that hipFile does not transfer data to
// device memory regions where it is not supposed to.
// ---------------------------------------------------------------------------
constexpr int32_t kSentinel     = -1;
constexpr uint8_t kSentinelByte = 0xFF;

// Number of elements in a sentinel region (sentinel region always 4KiB).
inline constexpr size_t
slackElems()
{
    return 4_KiB / sizeof(int32_t);
}

// ---------------------------------------------------------------------------
// Transfer-size test parameters.
// ---------------------------------------------------------------------------
struct SizeParam {
    size_t      bytes;
    std::string name;
};

HIPFILE_WARN_NO_EXIT_DTOR_OFF
inline const std::array<SizeParam, 3> combined_sizes{{
    {4_KiB, "sub_chunk"},
    {kChunkBytes + 4_KiB, "cross_chunk"},
    {2 * kChunkBytes, "multi_chunk"},
}};
HIPFILE_WARN_NO_EXIT_DTOR_ON

// ---------------------------------------------------------------------------
// File-extension cases.
//
// An extending write covers [file_off, file_off + io_bytes) of a file whose length before the write is
// base_len. hipFileWrite should be able to extend the length of a file, and when hipFileWrite targets a
// file_off that is beyond the existing end of the file (file_end), then [file_end, file_off) should be filled
// with zeros.
// ---------------------------------------------------------------------------
struct ExtendCase {
    hoff_t base_len; // file length before the extending write
    hoff_t file_off; // where the extending write starts
};

// Append to a zero-length file.
inline constexpr ExtendCase
appendFromEmpty()
{
    return {0, 0};
}

// Append exactly at the EOF of a file.
inline constexpr ExtendCase
appendAt(hoff_t eof)
{
    return {eof, eof};
}

// Append at `gap` many bytes past the EOF of a file. [base_len, base_len + gap) should be filled with zeros.
inline constexpr ExtendCase
holeAfter(hoff_t base_len, hoff_t gap)
{
    return {base_len, base_len + gap};
}

// Write to a `file_off` that appears before the EOF, but also extend the file with this write.
inline constexpr ExtendCase
overwriteAppend(hoff_t file_off, hoff_t overlap)
{
    return {file_off + overlap, file_off};
}

// ---------------------------------------------------------------------------
// Kernel launch geometry.
// ---------------------------------------------------------------------------
inline dim3
defaultGrid(size_t n)
{
    constexpr unsigned wg_size    = kDefaultWorkgroupSize;
    const size_t       workgroups = (n + wg_size - 1) / wg_size;
    const unsigned     grid       = static_cast<unsigned>(std::min<size_t>(workgroups, 65535));
    return dim3(grid == 0 ? 1 : grid);
}

// ---------------------------------------------------------------------------
// Host-side file inspection.
// ---------------------------------------------------------------------------
inline hoff_t
fileSize(int fd)
{
    struct stat st {};
    EXPECT_EQ(0, fstat(fd, &st));
    return static_cast<hoff_t>(st.st_size);
}

inline void
assertHoleZero(int fd, hoff_t from, hoff_t to)
{
    ASSERT_LE(from, to);
    const size_t n = static_cast<size_t>(to - from);
    if (n == 0) {
        return;
    }
    std::vector<uint8_t> hole(n, 0xEE); // pre-fill non-zero so a short pread is caught
    ssize_t              rv = pread(fd, hole.data(), n, from);
    ASSERT_EQ(static_cast<ssize_t>(n), rv);
    for (size_t i = 0; i < n; ++i) {
        ASSERT_EQ(0, hole[i]) << "hole byte non-zero at file offset " << (from + static_cast<hoff_t>(i));
    }
}

// ---------------------------------------------------------------------------
// Kernel error reporting.
// ---------------------------------------------------------------------------
// RAII for flag in device memory the kernel uses to report the first bad index.
struct BadIdxFlag {
    int32_t *dev{nullptr};

    BadIdxFlag()
    {
        if (hipMalloc(&dev, sizeof(int32_t)) != hipSuccess) {
            dev = nullptr;
            ADD_FAILURE() << "hipMalloc for the kernel bad-index flag failed";
            return;
        }
        reset();
    }
    ~BadIdxFlag()
    {
        if (dev != nullptr) {
            (void)hipFree(dev);
        }
    }
    // Callers must check this before launching a kernel that dereferences dev.
    bool allocated() const
    {
        return dev != nullptr;
    }
    void reset()
    {
        if (dev == nullptr) {
            return;
        }
        int32_t init = -1;
        EXPECT_EQ(hipSuccess, hipMemcpy(dev, &init, sizeof(int32_t), hipMemcpyHostToDevice));
    }
    int32_t value() const
    {
        int32_t v = -2;
        if (dev == nullptr) {
            return v;
        }
        EXPECT_EQ(hipSuccess, hipMemcpy(&v, dev, sizeof(int32_t), hipMemcpyDeviceToHost));
        return v;
    }
};

// ---------------------------------------------------------------------------
// int32 (pattern) helpers
// ---------------------------------------------------------------------------
inline std::vector<int32_t>
readbackInts(void *device_buffer, hoff_t byte_offset, size_t n)
{
    std::vector<int32_t> host(n);
    EXPECT_EQ(hipSuccess, hipMemcpy(host.data(),
                                    reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(device_buffer) +
                                                             static_cast<size_t>(byte_offset)),
                                    n * sizeof(int32_t), hipMemcpyDeviceToHost));
    return host;
}

inline std::vector<int32_t>
readFileInts(int fd, hoff_t byte_offset, size_t n)
{
    std::vector<int32_t> file(n);
    ssize_t              rv = pread(fd, file.data(), n * sizeof(int32_t), byte_offset);
    EXPECT_EQ(static_cast<ssize_t>(n * sizeof(int32_t)), rv);
    return file;
}

inline void
seedDevicePattern(void *device_buffer, hoff_t byte_offset, size_t n)
{
    std::vector<int32_t> host(n);
    fillIndexPattern(host.data(), n); // host[i] == i+1
    ASSERT_EQ(hipSuccess, hipMemcpy(reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(device_buffer) +
                                                             static_cast<size_t>(byte_offset)),
                                    host.data(), n * sizeof(int32_t), hipMemcpyHostToDevice));
}

inline void
seedFilePattern(int fd, hoff_t byte_offset, size_t n)
{
    std::vector<int32_t> host(n);
    fillIndexPattern(host.data(), n); // host[i] == i+1
    ssize_t rv = pwrite(fd, host.data(), n * sizeof(int32_t), byte_offset);
    ASSERT_EQ(static_cast<ssize_t>(n * sizeof(int32_t)), rv);
}

inline void
seedFileConstant(int fd, hoff_t byte_offset, size_t n, int32_t value)
{
    std::vector<int32_t> host(n, value);
    ssize_t              rv = pwrite(fd, host.data(), n * sizeof(int32_t), byte_offset);
    ASSERT_EQ(static_cast<ssize_t>(n * sizeof(int32_t)), rv);
}

inline void
assertConstant(const int32_t *arr, size_t from, size_t to, int32_t value)
{
    for (size_t i = from; i < to; ++i) {
        ASSERT_EQ(value, arr[i]) << "sentinel changed at index " << i;
    }
}

// Launches the int32 verify+modify kernel and asserts neither the payload pattern nor the
// device sentinel region was corrupted.
inline void
assertVerifyAndModify(int32_t *start, size_t alloc_n, int32_t *arr, size_t n, dim3 grid, dim3 workgroup,
                      size_t modify_stride = 1)
{
    ASSERT_NE(0U, modify_stride) << "modify_stride must be >= 1";
    BadIdxFlag bad;
    BadIdxFlag bad_slack;
    ASSERT_TRUE(bad.allocated() && bad_slack.allocated()) << "kernel bad-index flag allocation failed";
    ASSERT_EQ(hipSuccess, launchVerifyAndModify(start, alloc_n, arr, n, kPatternBase, bad.dev, kSentinel,
                                                bad_slack.dev, grid, workgroup, modify_stride));
    ASSERT_EQ(-1, bad.value()) << "payload pattern corrupted";
    ASSERT_EQ(-1, bad_slack.value()) << "untouched device sentinel region was clobbered";
}

// ---------------------------------------------------------------------------
// byte-granular helpers
// ---------------------------------------------------------------------------
inline void
seedFileBytesConstant(int fd, hoff_t byte_offset, size_t n, uint8_t value)
{
    std::vector<uint8_t> host(n, value);
    ssize_t              rv = pwrite(fd, host.data(), n, byte_offset);
    ASSERT_EQ(static_cast<ssize_t>(n), rv);
}

inline std::vector<uint8_t>
readFileBytes(int fd, hoff_t byte_offset, size_t n)
{
    std::vector<uint8_t> file(n);
    ssize_t              rv = pread(fd, file.data(), n, byte_offset);
    EXPECT_EQ(static_cast<ssize_t>(n), rv);
    return file;
}

// Launches the byte verify+modify kernel and asserts neither the payload bytes nor the
// device sentinel region was corrupted.
inline void
assertVerifyAndModifyBytes(uint8_t *start, size_t alloc_bytes, uint8_t *arr, size_t n, dim3 grid,
                           dim3 workgroup, size_t modify_stride)
{
    ASSERT_NE(0U, modify_stride) << "modify_stride must be >= 1";
    BadIdxFlag bad;
    BadIdxFlag bad_slack;
    ASSERT_TRUE(bad.allocated() && bad_slack.allocated()) << "kernel bad-index flag allocation failed";
    ASSERT_EQ(hipSuccess,
              launchVerifyAndModifyBytes(start, alloc_bytes, arr, n, kByteEntry, kByteModified, bad.dev,
                                         kByteDevSlack, bad_slack.dev, grid, workgroup, modify_stride));
    ASSERT_EQ(-1, bad.value()) << "payload bytes corrupted";
    ASSERT_EQ(-1, bad_slack.value()) << "untouched device sentinel region was clobbered";
}

} // namespace hipFileTest
