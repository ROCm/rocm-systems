/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

#include "backend/fallback.h"
#include "configuration.h"
#include "context.h"
#include "hipfile-literals.h"
#include "hipfile-warnings.h"
#include "hipfile.h"

#include "io-data-modification-kernel.h"
#include "io-test.h"
#include "test-common.h"
#include "test-options.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <hip/hip_runtime_api.h>
#include <optional>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

extern SystemTestOptions test_env;

namespace hipFileTest {

// ===========================================================================================================
// Constants and helper functions for setting up sentinel regions of files and device buffers, as well as
// verifying that the GPU correctly modified the targeted data without clobbering sentinel regions.
// ===========================================================================================================
inline constexpr int32_t kPatternBase = 1;

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
// Host-side file preparation.
// ---------------------------------------------------------------------------
inline void
fillIndexPattern(int32_t *arr, size_t n)
{
    for (size_t i = 0; i < n; ++i) {
        arr[i] = kPatternBase + static_cast<int32_t>(i);
    }
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
seedFileBytesConstant(int fd, hoff_t byte_offset, size_t n, uint8_t value)
{
    std::vector<uint8_t> host(n, value);
    ssize_t              rv = pwrite(fd, host.data(), n, byte_offset);
    ASSERT_EQ(static_cast<ssize_t>(n), rv);
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

inline std::vector<int32_t>
readFileInts(int fd, hoff_t byte_offset, size_t n)
{
    std::vector<int32_t> file(n);
    ssize_t              rv = pread(fd, file.data(), n * sizeof(int32_t), byte_offset);
    EXPECT_EQ(static_cast<ssize_t>(n * sizeof(int32_t)), rv);
    return file;
}

inline std::vector<uint8_t>
readFileBytes(int fd, hoff_t byte_offset, size_t n)
{
    std::vector<uint8_t> file(n);
    ssize_t              rv = pread(fd, file.data(), n, byte_offset);
    EXPECT_EQ(static_cast<ssize_t>(n), rv);
    return file;
}

// ---------------------------------------------------------------------------
// Host-side memory verification. Can be memory read from either a file or device buffer.
// ---------------------------------------------------------------------------
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

// Asserts every element was doubled: arr[i] == 2 * (kPatternBase + i).
inline void
assertDoubledPattern(const int32_t *arr, size_t n)
{
    for (size_t i = 0; i < n; ++i) {
        ASSERT_EQ(2 * (kPatternBase + static_cast<int32_t>(i)), arr[i]) << "doubled mismatch at index " << i;
    }
}

// Asserts the every-Nth-element modify policy: element i is doubled iff
// (i % modify_stride == 0), otherwise it still holds the un-doubled seed.
inline void
assertModifiedPattern(const int32_t *arr, size_t n, size_t modify_stride)
{
    ASSERT_NE(0U, modify_stride) << "modify_stride must be >= 1";
    for (size_t i = 0; i < n; ++i) {
        const int32_t seed = kPatternBase + static_cast<int32_t>(i);
        const int32_t want = (i % modify_stride == 0) ? 2 * seed : seed;
        ASSERT_EQ(want, arr[i]) << "modify-policy mismatch at index " << i;
    }
}

inline void
assertConstant(const int32_t *arr, size_t from, size_t to, int32_t value)
{
    ASSERT_LE(from, to);
    for (size_t i = from; i < to; ++i) {
        ASSERT_EQ(value, arr[i]) << "sentinel changed at index " << i;
    }
}

inline constexpr uint8_t kByteEntry     = 0xFF; // Initial value
inline constexpr uint8_t kByteModified  = 0x22;
inline constexpr uint8_t kByteDevSlack  = 0xAA;
inline constexpr uint8_t kByteFileSlack = 0x55;

// Asserts the every-Nth-byte modify policy over n Data bytes: byte i equals
// kByteModified iff (i % modify_stride == 0), otherwise it still holds kByteEntry.
inline void
assertBytesModified(const uint8_t *arr, size_t n, size_t modify_stride)
{
    ASSERT_NE(0U, modify_stride) << "modify_stride must be >= 1";
    for (size_t i = 0; i < n; ++i) {
        const uint8_t want = (i % modify_stride == 0) ? kByteModified : kByteEntry;
        ASSERT_EQ(want, arr[i]) << "byte modify-policy mismatch at index " << i;
    }
}

// Asserts bytes in [from, to) all equal `value` to verify untouched data was truly untouched.
inline void
assertBytesConstant(const uint8_t *arr, size_t from, size_t to, uint8_t value)
{
    ASSERT_LE(from, to);
    for (size_t i = from; i < to; ++i) {
        ASSERT_EQ(value, arr[i]) << "byte constant region changed at index " << i;
    }
}

// ---------------------------------------------------------------------------
// Device-side buffer inspection.
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

// ---------------------------------------------------------------------------
// Device-side buffer preparation.
// ---------------------------------------------------------------------------
inline void
seedDevicePattern(void *device_buffer, hoff_t byte_offset, size_t n)
{
    std::vector<int32_t> host(n);
    fillIndexPattern(host.data(), n); // host[i] == i+1
    ASSERT_EQ(hipSuccess, hipMemcpy(reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(device_buffer) +
                                                             static_cast<size_t>(byte_offset)),
                                    host.data(), n * sizeof(int32_t), hipMemcpyHostToDevice));
}

// ===========================================================================================================
// Default or frequently used sizes and offsets.
// ===========================================================================================================
constexpr size_t  kChunkBytes           = hipFile::Fallback::DefaultChunkSize;
constexpr int32_t kDefaultWorkgroupSize = 256;
constexpr hoff_t  kChunkOff             = static_cast<hoff_t>(kChunkBytes);

// Where a suite places its data to be modified in a file or buffer by default.
constexpr hoff_t kFileOffBase = static_cast<hoff_t>(kChunkBytes + 4_KiB);
constexpr hoff_t kBufOffBase  = static_cast<hoff_t>(4_KiB);

// Matrix for unaligned test cases.
constexpr size_t kSmallAligned   = 4_KiB;
constexpr size_t kSmallUnaligned = 4_KiB + 1;
constexpr size_t kLargeAligned   = kChunkBytes + 4_KiB;
constexpr size_t kLargeUnaligned = kChunkBytes + 4_KiB + 1;

// ===========================================================================================================
// Named cases where a test is extending the length of a file by using hipFileWrite.
// ===========================================================================================================
// An extending write covers [file_off, file_off + io_bytes) of a file whose length before the write is
// base_len. hipFileWrite should be able to extend the length of a file, and when hipFileWrite targets a
// file_off that is beyond the existing end of the file (file_end), then [file_end, file_off) should be filled
// with zeros.
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

// ===========================================================================================================
// Common kernel launch configurations, helpers for launching kernels and checking results for errors.
// ===========================================================================================================
enum class GridMode { Default, OneWorkgroup, ManyWorkgroups };

constexpr unsigned kManyWorkgroups = 300;

inline dim3
defaultGrid(size_t n)
{
    constexpr unsigned wg_size    = kDefaultWorkgroupSize;
    const size_t       workgroups = (n + wg_size - 1) / wg_size;
    const unsigned     grid       = static_cast<unsigned>(std::min<size_t>(workgroups, 65535));
    return dim3(grid == 0 ? 1 : grid);
}

inline dim3
gridFor(GridMode mode, size_t n)
{
    switch (mode) {
        case GridMode::OneWorkgroup:
            return dim3(1);
        case GridMode::ManyWorkgroups:
            return dim3(kManyWorkgroups);
        case GridMode::Default:
        default:
            return defaultGrid(n);
    }
}

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
// Kernel launch and error checking helpers.
// ---------------------------------------------------------------------------
// Launches the int32 verify+modify kernel and asserts neither the payload pattern nor the
// device sentinel region was corrupted.
inline void
assertVerifyAndModify(int32_t *start, size_t alloc_n, size_t data_start, size_t n, dim3 grid, dim3 workgroup,
                      size_t modify_stride = 1)
{
    ASSERT_NE(0U, modify_stride) << "modify_stride must be >= 1";
    BadIdxFlag bad;
    BadIdxFlag bad_slack;
    ASSERT_TRUE(bad.allocated() && bad_slack.allocated()) << "kernel bad-index flag allocation failed";
    ASSERT_EQ(hipSuccess, launchVerifyAndModify(start, alloc_n, data_start, n, kPatternBase, bad.dev,
                                                kSentinel, bad_slack.dev, grid, workgroup, modify_stride));
    ASSERT_EQ(-1, bad.value()) << "payload pattern corrupted";
    ASSERT_EQ(-1, bad_slack.value()) << "untouched device sentinel region was clobbered";
}

// Launches the byte verify+modify kernel and asserts neither the payload bytes nor the
// device sentinel region was corrupted.
inline void
assertVerifyAndModifyBytes(uint8_t *start, size_t alloc_bytes, size_t data_start, size_t n, dim3 grid,
                           dim3 workgroup, size_t modify_stride)
{
    ASSERT_NE(0U, modify_stride) << "modify_stride must be >= 1";
    BadIdxFlag bad;
    BadIdxFlag bad_slack;
    ASSERT_TRUE(bad.allocated() && bad_slack.allocated()) << "kernel bad-index flag allocation failed";
    ASSERT_EQ(hipSuccess, launchVerifyAndModifyBytes(start, alloc_bytes, data_start, n, kByteEntry,
                                                     kByteModified, bad.dev, kByteDevSlack, bad_slack.dev,
                                                     grid, workgroup, modify_stride));
    ASSERT_EQ(-1, bad.value()) << "payload bytes corrupted";
    ASSERT_EQ(-1, bad_slack.value()) << "untouched device sentinel region was clobbered";
}

// ===========================================================================================================
// Common struct containing test parameters that will be reused by all tests with helpers for creating
// complex matrices of tests.
// ===========================================================================================================
struct IoTestScenario {
    std::string               name{};                             // the full gtest case name
    IoTestBackend             backend  = IoTestBackend::Fallback; // backend fulfilling the I/O request
    size_t                    io_bytes = 4_KiB;                   // bytes transferred through the hipFile API
    hoff_t                    file_off = 0;                       // where the data sits in the file
    hoff_t                    buf_off  = 0;                       // where the data sits in the device buffer
    size_t                    stride   = 1;                       // 1 == modify every element
    GridMode                  grid     = GridMode::Default;       // workgroup count the kernel launches with
    std::optional<ExtendCase> ext      = std::nullopt;            // set only for extending writes
};

// A value of one of the parameters of an `IoTestScenario` and the string to reflect
// it with in the test name.
template <class T> struct Axis {
    T           value;
    std::string name;
};

// ---------------------------------------------------------------------------
// This class is responsible for building a series of IoTestScenarios. Allowing you
// to use a single class/object to represent all of the parameters of a test instance,
// as opposed to many small classes.
//
// This class contains functions that mimic functionality of GTest's `param_generator`s
// (e.g., `over` for `testing::Combine`, `add` for `testing::Values`).
// ---------------------------------------------------------------------------
class IoTestScenarioSet {
public:
    explicit IoTestScenarioSet(const IoTestScenario &base) : rows_{base}
    {
    }

    // This function is like GTest's `testing::Combine`. You specify a `field` of `IoTestScenario`
    // that you would like to vary over a range of values defined by `axis`, and `over`
    // multiplies each existing `IoTestScenario` in `rows_` to have an instance that takes on
    // each of the values in `axis`.
    template <class F, class Range> IoTestScenarioSet &over(F IoTestScenario::*field, const Range &axis)
    {
        std::vector<IoTestScenario> expanded;
        for (const IoTestScenario &row : rows_) {
            for (const auto &a : axis) {
                IoTestScenario next = row;
                next.*field         = a.value;
                next.name           = next.name.empty() ? a.name : next.name + "_" + a.name;
                expanded.push_back(next);
            }
        }
        rows_ = expanded;
        return *this;
    }

    // If you were to repeatedly use this function, this function is like GTest's `testing::Values`,
    // except you can append to an existing set of `IoTestScenario`.
    // This function should be used for adding bespoke test parameter scenarios that cannot be
    // expressed by a matrix of individual parameters that are combined by `over`.
    IoTestScenarioSet &add(const IoTestScenario &one)
    {
        rows_.push_back(one);
        return *this;
    }

    // This function combines two sets of `IoTestScenario`, which typically isn't possible in GTest,
    // unless you used multiple `INSTANTIATE_TEST_SUITE_P` calls.
    IoTestScenarioSet &add(const IoTestScenarioSet &other)
    {
        rows_.insert(rows_.end(), other.rows_.begin(), other.rows_.end());
        return *this;
    }

    std::vector<IoTestScenario> build() const
    {
        return rows_;
    }

private:
    std::vector<IoTestScenario> rows_;
};

inline std::string
ioTestScenarioName(const testing::TestParamInfo<IoTestScenario> &info)
{
    return info.param.name;
}

// ---------------------------------------------------------------------------
// Axis values that are common between many test suites.
// ---------------------------------------------------------------------------
HIPFILE_WARN_NO_EXIT_DTOR_OFF
inline const std::array<Axis<IoTestBackend>, 2> kBackends{{
    {IoTestBackend::Fastpath, "Fastpath"},
    {IoTestBackend::Fallback, "Fallback"},
}};

inline const std::array<Axis<size_t>, 3> kCombinedSizes{{
    {4_KiB, "sub_chunk"},
    {kChunkBytes + 4_KiB, "cross_chunk"},
    {2 * kChunkBytes, "multi_chunk"},
}};
HIPFILE_WARN_NO_EXIT_DTOR_ON

// ===========================================================================================================
// Common helper functions used in the tests, specialized for given data types.
// ===========================================================================================================
// Element policies that determine the datatype that the GPU kernels will operate on.
struct IntElementPolicy {
    using Elem = int32_t;

    static constexpr uint8_t kDevFill = kSentinelByte;

    static constexpr size_t elems(size_t bytes)
    {
        return bytes / sizeof(Elem);
    }

    static void seedFileData(int fd, hoff_t off, size_t n)
    {
        seedFilePattern(fd, off, n);
    }

    static void seedFileSlack(int fd, hoff_t off, size_t n)
    {
        seedFileConstant(fd, off, n, kSentinel);
    }

    static void seedDeviceData(void *buf, hoff_t off, size_t n)
    {
        seedDevicePattern(buf, off, n);
    }

    static std::vector<Elem> readFile(int fd, hoff_t off, size_t n)
    {
        return readFileInts(fd, off, n);
    }

    static void assertFileSlack(const Elem *arr, size_t from, size_t to)
    {
        assertConstant(arr, from, to, kSentinel);
    }

    static void assertModified(const Elem *arr, size_t n, size_t stride)
    {
        assertModifiedPattern(arr, n, stride);
    }

    static void verifyAndModify(Elem *start, size_t alloc_n, size_t data_start, size_t n, dim3 grid,
                                dim3 workgroup, size_t stride)
    {
        assertVerifyAndModify(start, alloc_n, data_start, n, grid, workgroup, stride);
    }
};

struct ByteElementPolicy {
    using Elem = uint8_t;

    static constexpr uint8_t kDevFill = kByteDevSlack;

    static constexpr size_t elems(size_t bytes)
    {
        return bytes;
    }

    static void seedFileData(int fd, hoff_t off, size_t n)
    {
        seedFileBytesConstant(fd, off, n, kByteEntry);
    }

    static void seedFileSlack(int fd, hoff_t off, size_t n)
    {
        seedFileBytesConstant(fd, off, n, kByteFileSlack);
    }

    static void seedDeviceData(void *buf, hoff_t off, size_t n)
    {
        ASSERT_EQ(hipSuccess,
                  hipMemset(static_cast<uint8_t *>(buf) + static_cast<size_t>(off), kByteEntry, n));
    }

    static std::vector<Elem> readFile(int fd, hoff_t off, size_t n)
    {
        return readFileBytes(fd, off, n);
    }

    static void assertFileSlack(const Elem *arr, size_t from, size_t to)
    {
        assertBytesConstant(arr, from, to, kByteFileSlack);
    }

    static void assertModified(const Elem *arr, size_t n, size_t stride)
    {
        assertBytesModified(arr, n, stride);
    }

    static void verifyAndModify(Elem *start, size_t alloc_bytes, size_t data_start, size_t n, dim3 grid,
                                dim3 workgroup, size_t stride)
    {
        assertVerifyAndModifyBytes(start, alloc_bytes, data_start, n, grid, workgroup, stride);
    }
};

// ===========================================================================================================
// Base test fixture for common file and buffer allocation and modification for test suites to reuse.
// ===========================================================================================================
// Individual suites should use ftruncate to size the file's initial length before
// calling this SetUp.
template <class Policy> struct DataModificationBase : public testing::TestWithParam<IoTestScenario> {

    const IoTestBackend backend{GetParam().backend};   // backend fulfilling the I/O request
    const size_t        io_bytes{GetParam().io_bytes}; // bytes transferred using the hipFile API
    // Over-allocate a device sentinel region on each side of the data.
    // Device layout (each sentinel region 4_KiB, data io_bytes):
    // [head device sentinel region][data][tail device sentinel region].
    const size_t buffer_bytes{io_bytes + 2 * 4_KiB};        // total device buffer size
    const size_t io_elems{Policy::elems(io_bytes)};         // transferred elements
    const size_t buffer_elems{Policy::elems(buffer_bytes)}; // elements spanning the whole buffer

    Tmpfile                tmpfile;
    hipFileHandle_t        tmpfile_handle{nullptr};
    void                  *device_buffer{nullptr}; // allocated by SetUp
    typename Policy::Elem *buffer_start{nullptr};  // typed view of device_buffer

    DataModificationBase() : tmpfile{test_env.ais_capable_dir}
    {
    }

    void SetUp() override
    {
        hipFile::Context<hipFile::Configuration>::get()->fastpath(false);
        hipFile::Context<hipFile::Configuration>::get()->fallback(false);

        switch (backend) {
            case IoTestBackend::Fastpath:
                hipFile::Context<hipFile::Configuration>::get()->fastpath(true);
                break;
            case IoTestBackend::Fallback:
                hipFile::Context<hipFile::Configuration>::get()->fallback(true);
                break;
            default:
                FAIL() << "Unsupported IoTestBackend";
        }

        hipFileDescr_t descr{};
        descr.type      = hipFileHandleTypeOpaqueFD;
        descr.handle.fd = tmpfile.fd;
        ASSERT_EQ(HIPFILE_SUCCESS, hipFileHandleRegister(&tmpfile_handle, &descr));

        ASSERT_EQ(hipSuccess, hipMalloc(&device_buffer, buffer_bytes));
        buffer_start = static_cast<typename Policy::Elem *>(device_buffer);
        ASSERT_EQ(hipSuccess, hipMemset(device_buffer, Policy::kDevFill, buffer_bytes));
        // hipMemset is not synchronous w.r.t. the host, and hipFileRead is not ordered w.r.t. the stream.
        ASSERT_EQ(hipSuccess, hipDeviceSynchronize());

        if (backend == IoTestBackend::Fastpath) {
            enforceFastpathGate(tmpfile_handle, device_buffer);
        }
    }

    void TearDown() override
    {
        if (device_buffer) {
            ASSERT_EQ(hipSuccess, hipFree(device_buffer));
        }
        hipFileHandleDeregister(tmpfile_handle);
    }
};

// ===========================================================================================================
// Shared test bodies that are reused across multiple test suites.
// ===========================================================================================================
// These bodies need to be called with ASSERT_NO_FATAL_FAILURE from a TEST_P in order for
// failures to propagate.

// Round trip with both the device sentinel regions and the file sentinel regions guarded
// (checked to make sure they are not erroneously modified).
template <class Policy>
void
runAllRegionsTest(DataModificationBase<Policy> &f)
{
    using Elem = typename Policy::Elem;

    const IoTestScenario &s = f.GetParam();

    const size_t n        = f.io_elems;
    const size_t slack_n  = Policy::elems(4_KiB);
    const hoff_t buf_off  = s.buf_off;
    const hoff_t file_off = s.file_off;
    const hoff_t head_off = file_off - static_cast<hoff_t>(4_KiB); // head file sentinel region
    const hoff_t tail_off = file_off + static_cast<hoff_t>(f.io_bytes);

    // File layout (each sentinel region slack_n elements, data n elements; data begins at
    // file_off past the chunk boundary):
    // [unwritten hole = 0][head file sentinel region][data][tail file sentinel region].
    Policy::seedFileSlack(f.tmpfile.fd, head_off, slack_n);
    Policy::seedFileData(f.tmpfile.fd, file_off, n);
    Policy::seedFileSlack(f.tmpfile.fd, tail_off, slack_n);

    ASSERT_EQ(static_cast<ssize_t>(f.io_bytes),
              hipFileRead(f.tmpfile_handle, f.device_buffer, f.io_bytes, file_off, buf_off));

    // Device layout (each sentinel region slack_n elements, data n elements):
    // [head device sentinel region][data][tail device sentinel region].
    Policy::verifyAndModify(f.buffer_start, f.buffer_elems, Policy::elems(static_cast<size_t>(buf_off)), n,
                            gridFor(s.grid, n), dim3(kDefaultWorkgroupSize), s.stride);

    ASSERT_EQ(static_cast<ssize_t>(f.io_bytes),
              hipFileWrite(f.tmpfile_handle, f.device_buffer, f.io_bytes, file_off, buf_off));

    // Start of file that was supposed to be untouched remains untouched.
    assertHoleZero(f.tmpfile.fd, 0, head_off);
    // Existing data before write is fully intact.
    std::vector<Elem> head = Policy::readFile(f.tmpfile.fd, head_off, slack_n);
    Policy::assertFileSlack(head.data(), 0, slack_n);
    // Modified data is correctly modified.
    std::vector<Elem> body = Policy::readFile(f.tmpfile.fd, file_off, n);
    Policy::assertModified(body.data(), n, s.stride);
    // Existing data after write is fully intact.
    std::vector<Elem> tail = Policy::readFile(f.tmpfile.fd, tail_off, slack_n);
    Policy::assertFileSlack(tail.data(), 0, slack_n);
}

// A write that extends the file size. If hipFileWrite is made past the current EOF, that
// gap should be zero-filled.
template <class Policy>
void
runExtendTest(DataModificationBase<Policy> &f)
{
    using Elem = typename Policy::Elem;

    const IoTestScenario &s = f.GetParam();
    ASSERT_TRUE(s.ext.has_value()) << "runExtendTest requires IoTestScenario::ext to be set";

    const size_t n          = f.io_elems;
    const hoff_t base_len   = s.ext->base_len;
    const hoff_t file_off   = s.ext->file_off;
    const hoff_t buf_off    = s.buf_off;
    const size_t data_start = Policy::elems(static_cast<size_t>(buf_off));

    // Device layout (data begins at buf_off, after the head sentinel region):
    // [head device sentinel region][data][tail device sentinel region].
    Policy::seedDeviceData(f.device_buffer, buf_off, n);

    Policy::verifyAndModify(f.buffer_start, f.buffer_elems, data_start, n, gridFor(s.grid, n),
                            dim3(kDefaultWorkgroupSize), s.stride);

    // File layout after the write (the hole spans [base_len, file_off) and is empty when
    // we append to the file):
    // [preserved][hole = 0][data = stride-modified], when file_off >= base_len.
    // [preserved][data = stride-modified],           when file_off <  base_len.
    const hoff_t preserved_end = (file_off < base_len) ? file_off : base_len;
    const size_t preserved_n   = Policy::elems(static_cast<size_t>(preserved_end));
    if (preserved_n > 0) {
        Policy::seedFileSlack(f.tmpfile.fd, 0, preserved_n);
    }

    ASSERT_EQ(static_cast<ssize_t>(f.io_bytes),
              hipFileWrite(f.tmpfile_handle, f.device_buffer, f.io_bytes, file_off, buf_off));

    std::vector<Elem> body = Policy::readFile(f.tmpfile.fd, file_off, n);
    Policy::assertModified(body.data(), n, s.stride);

    // Hole was zero-filled.
    if (file_off > base_len) {
        assertHoleZero(f.tmpfile.fd, base_len, file_off);
    }

    // Final size correct.
    ASSERT_EQ(file_off + static_cast<hoff_t>(f.io_bytes), fileSize(f.tmpfile.fd));

    // Existing data below the write is fully intact.
    if (preserved_n > 0) {
        std::vector<Elem> head = Policy::readFile(f.tmpfile.fd, 0, preserved_n);
        Policy::assertFileSlack(head.data(), 0, preserved_n);
    }
}

} // namespace hipFileTest

using namespace hipFile;
using namespace hipFileTest;

HIPFILE_WARN_NO_GLOBAL_CTOR_OFF

// ===========================================================================================================
// Beginning of test suites.
// ===========================================================================================================

// This test suite exercises the ability of hipFileRead and hipFileWrite while varying the size of I/O and the
// backend that is used to fulfill the I/O request.
//
// We verify hipFileRead and hipFileWrite behave as expected by guarding the targeted regions of the
// device memory allocation and the region of the file that data will be read from and written to, surrounding
// them with poisoned memory containing sentinel values that would tell us if hipFile ever read or wrote data
// to a location that the user did not specify.
struct HipFileVerify : public DataModificationBase<IntElementPolicy> {
    void SetUp() override
    {
        // Size the file to hold io_bytes of data plus a leading empty chunk.
        ASSERT_EQ(0, ftruncate(tmpfile.fd, static_cast<off_t>(io_bytes + kChunkBytes)));
        DataModificationBase::SetUp();
    }
};

// Isolated hipFileWrite test.
TEST_P(HipFileVerify, WritePersistsDoubledData)
{
    const size_t n = io_elems;
    seedDevicePattern(device_buffer, 0, n);

    assertVerifyAndModify(buffer_start, buffer_elems, 0, n, defaultGrid(n), dim3(kDefaultWorkgroupSize));

    ASSERT_EQ(static_cast<ssize_t>(io_bytes), hipFileWrite(tmpfile_handle, device_buffer, io_bytes, 0, 0));

    std::vector<int32_t> file = readFileInts(tmpfile.fd, 0, n);
    assertDoubledPattern(file.data(), n);
}

// Isolated hipFileRead test.
TEST_P(HipFileVerify, ReadDeliversDoubledData)
{
    const size_t n = io_elems;
    seedFilePattern(tmpfile.fd, 0, n);

    ASSERT_EQ(static_cast<ssize_t>(io_bytes), hipFileRead(tmpfile_handle, device_buffer, io_bytes, 0, 0));

    assertVerifyAndModify(buffer_start, buffer_elems, 0, n, defaultGrid(n), dim3(kDefaultWorkgroupSize));

    std::vector<int32_t> buf = readbackInts(device_buffer, 0, n);
    assertDoubledPattern(buf.data(), n);
}

// hipFileRead + hipFileWrite at offset into device buffer (places sentinels before and after data).
// i.e., verify that data from hipFileRead does not clobber outside specified device buffer.
TEST_P(HipFileVerify, RoundTripGuardsDeviceSlack)
{
    const size_t n       = io_elems;
    const hoff_t buf_off = static_cast<hoff_t>(4_KiB); // one head device sentinel region

    seedFilePattern(tmpfile.fd, 0, n);
    ASSERT_EQ(static_cast<ssize_t>(io_bytes),
              hipFileRead(tmpfile_handle, device_buffer, io_bytes, 0, buf_off));

    // Device layout (each sentinel region slackElems() ints, data n ints):
    // [head device sentinel region][data][tail device sentinel region]
    assertVerifyAndModify(buffer_start, buffer_elems, slackElems(), n, defaultGrid(n),
                          dim3(kDefaultWorkgroupSize));

    ASSERT_EQ(static_cast<ssize_t>(io_bytes),
              hipFileWrite(tmpfile_handle, device_buffer, io_bytes, 0, buf_off));

    std::vector<int32_t> file = readFileInts(tmpfile.fd, 0, n);
    assertDoubledPattern(file.data(), n);
}

// hipFileRead + hipFileWrite at offset into file buffer (places sentinels before and after data).
// i.e., verify that data from hipFileWrite does not clobber outside specified file buffer.
TEST_P(HipFileVerify, RoundTripGuardsFileSlack)
{
    const size_t n         = io_elems;
    const size_t bracket_n = 4_KiB / sizeof(int32_t);    // file sentinel region head/tail each
    const hoff_t file_off  = static_cast<hoff_t>(4_KiB); // data starts after head
    const size_t total_n   = bracket_n + n + bracket_n;  // head + data + tail
    const hoff_t tail_off  = static_cast<hoff_t>((bracket_n + n) * sizeof(int32_t));

    // File layout (each sentinel region bracket_n ints, data n ints):
    // [head file sentinel region = -1][data = i+1][tail file sentinel region = -1].
    seedFileConstant(tmpfile.fd, 0, bracket_n, kSentinel);
    seedFilePattern(tmpfile.fd, file_off, n);
    seedFileConstant(tmpfile.fd, tail_off, bracket_n, kSentinel);

    ASSERT_EQ(static_cast<ssize_t>(io_bytes),
              hipFileRead(tmpfile_handle, device_buffer, io_bytes, file_off, 0));

    assertVerifyAndModify(buffer_start, buffer_elems, 0, n, defaultGrid(n), dim3(kDefaultWorkgroupSize));

    ASSERT_EQ(static_cast<ssize_t>(io_bytes),
              hipFileWrite(tmpfile_handle, device_buffer, io_bytes, file_off, 0));

    std::vector<int32_t> file = readFileInts(tmpfile.fd, 0, total_n);
    assertConstant(file.data(), 0, bracket_n, kSentinel);
    assertDoubledPattern(file.data() + bracket_n, n);
    assertConstant(file.data(), bracket_n + n, total_n, kSentinel);
}

HIPFILE_WARN_NO_EXIT_DTOR_OFF
const std::array<Axis<size_t>, 5> kVerifySizes{{
    {4_KiB, "sub_chunk"},
    {kChunkBytes - 4_KiB, "near_chunk"},
    {kChunkBytes, "exact_chunk"},
    {kChunkBytes + 4_KiB, "cross_chunk"},
    {2 * kChunkBytes, "multi_chunk"},
}};
HIPFILE_WARN_NO_EXIT_DTOR_ON

INSTANTIATE_TEST_SUITE_P(, HipFileVerify,
                         testing::ValuesIn(IoTestScenarioSet{IoTestScenario{}}
                                               .over(&IoTestScenario::backend, kBackends)
                                               .over(&IoTestScenario::io_bytes, kVerifySizes)
                                               .build()),
                         ioTestScenarioName);

// This test suite performs the same combined hipFileRead + hipFileWrite test that the hipFileVerify test
// suite exercises. We use the same poisoned memory region mechanism to verify hipFile transferred the
// specified data. We also varying the following new parameters:
//   - the stride between cache lines that the modified data touches.
//   - the number of workgroups that modify the data (this can also vary the number of different CUs or XCDs
//   on the GPU that touch the data).
// Additionally, this test suite places the targeted data in the file at an offset past the fallback path's
// chunking boundary to ensure the fallback path operates on the specified chunks.
struct HipFileVerifyCombined : public DataModificationBase<IntElementPolicy> {
    void SetUp() override
    {
        // File layout (each sentinel region 4_KiB, data io_bytes; data begins at file
        // offset kFileOffBase past the chunk boundary):
        // [head file sentinel region][data][tail file sentinel region].
        const hoff_t tail_off = GetParam().file_off + static_cast<hoff_t>(io_bytes);
        ASSERT_EQ(0, ftruncate(tmpfile.fd, tail_off + static_cast<hoff_t>(4_KiB)));
        DataModificationBase::SetUp();
    }
};

TEST_P(HipFileVerifyCombined, RoundTripGuardsAllRegions)
{
    ASSERT_NO_FATAL_FAILURE(runAllRegionsTest(*this));
}

HIPFILE_WARN_NO_EXIT_DTOR_OFF
const std::array<Axis<GridMode>, 2> kGridSweep{{
    {GridMode::OneWorkgroup, "1wg"},
    {GridMode::ManyWorkgroups, "300wg"},
}};

const std::array<Axis<size_t>, 3> kStrides{{{2, "stride2"}, {32, "stride32"}, {64, "stride64"}}};
const std::array<Axis<size_t>, 2> kStridesWide{{{2, "stride2"}, {512, "stride512"}}};
const std::array<Axis<size_t>, 1> kMultiChunkOnly{{{2 * kChunkBytes, "multi_chunk"}}};
HIPFILE_WARN_NO_EXIT_DTOR_ON

INSTANTIATE_TEST_SUITE_P(, HipFileVerifyCombined,
                         testing::ValuesIn(IoTestScenarioSet{
                             IoTestScenario{.file_off = kFileOffBase, .buf_off = kBufOffBase}}
                                               .over(&IoTestScenario::backend, kBackends)
                                               .over(&IoTestScenario::io_bytes, kCombinedSizes)
                                               .over(&IoTestScenario::stride, kStrides)
                                               // A large sweep with many workgroups, exercising the behaviour
                                               // of only some CUs modifying the data.
                                               .add(IoTestScenarioSet{IoTestScenario{.file_off = kFileOffBase,
                                                                                     .buf_off  = kBufOffBase}}
                                                        .over(&IoTestScenario::backend, kBackends)
                                                        .over(&IoTestScenario::io_bytes, kMultiChunkOnly)
                                                        .over(&IoTestScenario::grid, kGridSweep)
                                                        .over(&IoTestScenario::stride, kStridesWide))
                                               .build()),
                         ioTestScenarioName);

// This test suite exercises the same hipFileRead + hipFileWrite behaviour, in combination with the behaviour
// of extending the file length to ensure that untouched regions of a file that is extended as a result of
// hipFileWrite are either the previously present data, or a hole of 0-initialized data, matching the POSIX
// behaviour.
struct HipFileExtend : public DataModificationBase<IntElementPolicy> {
    void SetUp() override
    {
        // Every scenario must actually extend the file; the final-size assertion in the
        // body depends on it.
        ASSERT_TRUE(GetParam().ext.has_value());
        ASSERT_GT(GetParam().ext->file_off + static_cast<hoff_t>(io_bytes), GetParam().ext->base_len);

        // Size file to base_len to be extended.
        ASSERT_EQ(0, ftruncate(tmpfile.fd, static_cast<off_t>(GetParam().ext->base_len)));
        DataModificationBase::SetUp();
    }
};

TEST_P(HipFileExtend, Extends)
{
    ASSERT_NO_FATAL_FAILURE(runExtendTest(*this));
}

HIPFILE_WARN_NO_EXIT_DTOR_OFF
const std::array<Axis<ExtendCase>, 5> kExtendCases{{
    {appendFromEmpty(), "extend_empty_contiguous"},
    {appendAt(kChunkOff), "append_aligned"},
    {holeAfter(0, kChunkOff), "hole_from_empty"},
    {holeAfter(kChunkOff, kChunkOff), "hole_from_aligned"},
    {overwriteAppend(kChunkOff, static_cast<hoff_t>(4_KiB) / 2), "overwrite_and_append"},
}};

const std::array<Axis<size_t>, 2> kExtendSizes{{{4_KiB, "small"}, {kChunkBytes + 4_KiB, "large"}}};
HIPFILE_WARN_NO_EXIT_DTOR_ON

INSTANTIATE_TEST_SUITE_P(, HipFileExtend,
                         testing::ValuesIn(IoTestScenarioSet{
                             IoTestScenario{.buf_off = kBufOffBase, .stride = 2}}
                                               .over(&IoTestScenario::backend, kBackends)
                                               .over(&IoTestScenario::ext, kExtendCases)
                                               .over(&IoTestScenario::io_bytes, kExtendSizes)
                                               .build()),
                         ioTestScenarioName);

// This test suite exercises the ability of hipFileRead and hipFileWrite while varying the size of I/O and the
// backend that is used to fulfill the I/O request. Additionally, we test the ability of hipFileRead and
// hipFileWrite to correctly transfer data when data is modified at a byte granularity. It additionally
// exercises unaligned I/O, varying the source of unaligned-ness (device buffer offset vs. file offset vs.
// size).
//
// NOTE: unaligned I/O only uses the fallback path.
//
// We verify hipFileRead and hipFileWrite behave as expected by guarding the targeted regions of the
// device memory allocation and the region of the file that data will be read from and written to, surrounding
// them with poisoned memory containing sentinel values that would tell us if hipFile ever read or wrote data
// to a location that the user did not specify.
struct HipFileVerifyBytes : public DataModificationBase<ByteElementPolicy> {
    void SetUp() override
    {
        // File layout (each sentinel region 4_KiB, data io_bytes; data begins at file
        // offset GetParam().file_off past the chunk boundary). Size through the tail bracket:
        // [head file sentinel region][data][tail file sentinel region]
        const hoff_t tail_off = GetParam().file_off + static_cast<hoff_t>(io_bytes);
        ASSERT_EQ(0, ftruncate(tmpfile.fd, tail_off + static_cast<hoff_t>(4_KiB)));
        DataModificationBase::SetUp();
    }
};

TEST_P(HipFileVerifyBytes, RoundTripGuardsAllRegions)
{
    ASSERT_NO_FATAL_FAILURE(runAllRegionsTest(*this));
}

HIPFILE_WARN_NO_EXIT_DTOR_OFF
const std::array<Axis<hoff_t>, 2> kFileOffs{{
    {kFileOffBase, "file_aligned"},
    {kFileOffBase + 1, "file_unaligned"},
}};

const std::array<Axis<hoff_t>, 2> kBufOffs{{
    {kBufOffBase, "buffer_aligned"},
    {kBufOffBase + 1, "buffer_unaligned"},
}};

const std::array<Axis<size_t>, 4> kUnalignedSizes{{
    {kSmallAligned, "small_aligned"},
    {kSmallUnaligned, "small_unaligned"},
    {kLargeAligned, "large_aligned"},
    {kLargeUnaligned, "large_unaligned"},
}};
HIPFILE_WARN_NO_EXIT_DTOR_ON

INSTANTIATE_TEST_SUITE_P(, HipFileVerifyBytes,
                         testing::ValuesIn(IoTestScenarioSet{
                             IoTestScenario{.file_off = kFileOffBase, .buf_off = kBufOffBase, .stride = 2}}
                                               .over(&IoTestScenario::backend, kBackends)
                                               .over(&IoTestScenario::io_bytes, kCombinedSizes)
                                               .build()),
                         ioTestScenarioName);

// Roughly a matrix of {device buffer aligned, unaligned} x {file offset aligned, unaligned} x {size
// aligned, unaligned}. Unaligned I/O only uses the fallback path, which is the IoTestScenario default.
INSTANTIATE_TEST_SUITE_P(Unaligned, HipFileVerifyBytes,
                         testing::ValuesIn(IoTestScenarioSet{IoTestScenario{.stride = 2}}
                                               .over(&IoTestScenario::file_off, kFileOffs)
                                               .over(&IoTestScenario::buf_off, kBufOffs)
                                               .over(&IoTestScenario::io_bytes, kUnalignedSizes)
                                               .build()),
                         ioTestScenarioName);

// This test suite exercises the behaviour of extending the length of a file with an unaligned hipFileWrite,
// to ensure that untouched regions of a file that is extended are either the previously present data, or a
// hole of 0-initialized data, matching the POSIX behaviour. The length of the file before the write, the file
// offset of the write, the size of the write, and the device buffer offset each vary between aligned and one
// byte past a boundary.
//
// NOTE: unaligned I/O only uses the fallback path.
struct HipFileExtendUnaligned : public DataModificationBase<ByteElementPolicy> {
    void SetUp() override
    {
        // Every scenario must actually extend the file.
        ASSERT_TRUE(GetParam().ext.has_value());
        ASSERT_GT(GetParam().ext->file_off + static_cast<hoff_t>(io_bytes), GetParam().ext->base_len);

        ASSERT_EQ(0, ftruncate(tmpfile.fd, static_cast<off_t>(GetParam().ext->base_len)));
        DataModificationBase::SetUp();
    }
};

TEST_P(HipFileExtendUnaligned, Extends)
{
    ASSERT_NO_FATAL_FAILURE(runExtendTest(*this));
}

// The axis values every scenario is built from. Each unaligned value is its aligned counterpart
// pushed one byte past the boundary, so a scenario's name reads as the list of axes it misaligns.
constexpr hoff_t kBaseEmpty     = 0; // file length before the extending write
constexpr hoff_t kBaseAligned   = kChunkOff;
constexpr hoff_t kBaseUnaligned = kChunkOff + 1;

constexpr hoff_t kGapAligned =
    static_cast<hoff_t>(4_KiB); // distance from EOF to the start of a write that leaves a hole
constexpr hoff_t kGapUnaligned = static_cast<hoff_t>(4_KiB) + 1;

// The bytes from the file that a write that also extends the file will overlap.
constexpr hoff_t kOverlap = static_cast<hoff_t>(4_KiB) / 2;

constexpr hoff_t kBufAligned   = 0; // where the data starts within the device buffer
constexpr hoff_t kBufUnaligned = 1;

constexpr size_t kUnalignedStride = 2; // every scenario modifies every other element

static IoTestScenario
extendScenario(const char *name, ExtendCase ext, size_t io_bytes = kSmallAligned,
               hoff_t buf_off = kBufAligned)
{
    return IoTestScenario{
        .name = name, .io_bytes = io_bytes, .buf_off = buf_off, .stride = kUnalignedStride, .ext = ext};
}

// These are hard to express using `testing::Combine` or `over`, so we use a helper function.
HIPFILE_WARN_NO_EXIT_DTOR_OFF
const std::array<IoTestScenario, 16> kExtendUnalignedScenarios{{
    // Append to empty file.
    extendScenario("empty_contiguous_aligned_size", appendFromEmpty()),
    extendScenario("empty_contiguous_unaligned_size", appendFromEmpty(), kSmallUnaligned),
    // Contiguous append onto a non-empty file.
    extendScenario("append_unaligned_base", appendAt(kBaseUnaligned)),
    extendScenario("append_aligned_base_unaligned_size", appendAt(kBaseAligned), kSmallUnaligned),
    // Write past EOF, creating a hole from the existing EOF of the file to the start of the write.
    extendScenario("hole_from_empty_aligned_off", holeAfter(kBaseEmpty, kGapAligned)),
    extendScenario("hole_from_empty_unaligned_off", holeAfter(kBaseEmpty, kGapUnaligned)),
    extendScenario("hole_from_unaligned_base", holeAfter(kBaseUnaligned, kGapUnaligned)),
    extendScenario("hole_unaligned_off_unaligned_size", holeAfter(kBaseAligned, kGapUnaligned),
                   kSmallUnaligned),
    // Unaligned device buffer.
    extendScenario("append_unaligned_buffer", appendAt(kBaseAligned), kSmallAligned, kBufUnaligned),
    extendScenario("hole_unaligned_buffer", holeAfter(kBaseEmpty, kGapAligned), kSmallAligned, kBufUnaligned),
    extendScenario("append_unaligned_base_unaligned_buffer", appendAt(kBaseUnaligned), kSmallUnaligned,
                   kBufUnaligned),
    extendScenario("hole_all_unaligned", holeAfter(kBaseUnaligned, kGapUnaligned), kSmallUnaligned,
                   kBufUnaligned),
    // Transfers larger than the fallback chunking size.
    extendScenario("large_hole_cross_chunk", holeAfter(kBaseEmpty, kGapAligned), kLargeAligned),
    extendScenario("large_append_unaligned_base", appendAt(kBaseUnaligned), kLargeUnaligned),
    // Write before the existing EOF, but extend the length of the file with the same write.
    extendScenario("overwrite_append_aligned_base", overwriteAppend(kBaseAligned, kOverlap)),
    extendScenario("overwrite_append_all_unaligned", overwriteAppend(kBaseUnaligned, kOverlap),
                   kSmallUnaligned, kBufUnaligned),
}};
HIPFILE_WARN_NO_EXIT_DTOR_ON

INSTANTIATE_TEST_SUITE_P(, HipFileExtendUnaligned, testing::ValuesIn(kExtendUnalignedScenarios),
                         ioTestScenarioName);

HIPFILE_WARN_NO_GLOBAL_CTOR_ON
