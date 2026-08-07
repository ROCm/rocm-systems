/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

#include "context.h"
#include "configuration.h"
#include "hipfile-warnings.h"
#include "hipfile.h"

#include "io.hpp"
#include "io-verify.hpp"
#include "test-common.h"
#include "test-options.h"

#include <array>
#include <cstdint>
#include <gtest/gtest.h>
#include <hip/hip_runtime_api.h>
#include <string>
#include <tuple>
#include <unistd.h>
#include <vector>

extern SystemTestOptions test_env;

using namespace hipFile;
using namespace hipFileTest;

HIPFILE_WARN_NO_GLOBAL_CTOR_OFF

// ---------------------------------------------------------------------------
// This test suite exercises the ability of hipFileRead and hipFileWrite while varying the size of I/O and the
// backend that is used to fulfill the I/O request.
//
// We verify hipFileRead and hipFileWrite behave as expected by guarding the targeted regions of the
// device memory allocation and the region of the file that data will be read from and written to, surrounding
// them with poisoned memory containing sentinel values that would tell us if hipFile ever read or wrote data
// to a location that the user did not specify.
// ---------------------------------------------------------------------------
struct HipFileVerify : public testing::TestWithParam<std::tuple<IoTestParam, SizeParam>> {

    Tmpfile         tmpfile;
    hipFileHandle_t tmpfile_handle{nullptr};
    void           *device_buffer{nullptr};
    size_t          io_bytes{0};     // bytes transferred using the hipFile API
    size_t          buffer_bytes{0}; // total device buffer size

    HipFileVerify() : tmpfile{test_env.ais_capable_dir}
    {
    }

    IoTestBackend backend() const
    {
        return std::get<0>(GetParam()).backend;
    }

    void SetUp() override
    {
        Context<Configuration>::get()->fastpath(false);
        Context<Configuration>::get()->fallback(false);

        switch (backend()) {
            case IoTestBackend::Fastpath:
                Context<Configuration>::get()->fastpath(true);
                break;
            case IoTestBackend::Fallback:
                Context<Configuration>::get()->fallback(true);
                break;
            default:
                FAIL() << "Unsupported IoTestBackend";
        }

        io_bytes = std::get<1>(GetParam()).bytes;
        // The data modification kernel will also verify that the sentinel regions of the device buffer are
        // unmodified. Device layout (each sentinel region is kFourKiB, data is io_bytes): [head device
        // sentinel region][data][tail device sentinel region]
        buffer_bytes = io_bytes + 2 * kFourKiB;

        // Size the file to hold io_bytes of data plus a leading empty chunk.
        ASSERT_EQ(0, ftruncate(tmpfile.fd, static_cast<off_t>(io_bytes + kChunkBytes)));

        hipFileDescr_t descr{};
        descr.type      = hipFileHandleTypeOpaqueFD;
        descr.handle.fd = tmpfile.fd;
        ASSERT_EQ(HIPFILE_SUCCESS, hipFileHandleRegister(&tmpfile_handle, &descr));

        ASSERT_EQ(hipSuccess, hipMalloc(&device_buffer, buffer_bytes));
        ASSERT_EQ(hipSuccess, hipMemset(device_buffer, kSentinelByte, buffer_bytes));
        // hipMemset is not synchronous w.r.t. the host, and hipFileRead is not ordered w.r.t. the stream.
        ASSERT_EQ(hipSuccess, hipDeviceSynchronize());

        if (backend() == IoTestBackend::Fastpath) {
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

    size_t elems() const
    {
        return io_bytes / sizeof(int32_t);
    }

    size_t bufferElems() const
    {
        return buffer_bytes / sizeof(int32_t);
    }

    int32_t *bufferStart() const
    {
        return static_cast<int32_t *>(device_buffer);
    }
};

// Isolated hipFileWrite test.
TEST_P(HipFileVerify, WritePersistsDoubledData)
{
    const size_t n = elems();
    seedDevicePattern(device_buffer, 0, n);

    launchAndVerify(bufferStart(), bufferElems(), bufferStart(), n, defaultGrid(n),
                    dim3(kDefaultWorkgroupSize));

    ASSERT_EQ(static_cast<ssize_t>(io_bytes), hipFileWrite(tmpfile_handle, device_buffer, io_bytes, 0, 0));

    std::vector<int32_t> file = readFileInts(tmpfile.fd, 0, n);
    assertDoubledPattern(file.data(), n);
}

// Isolated hipFileRead test.
TEST_P(HipFileVerify, ReadDeliversDoubledData)
{
    const size_t n = elems();
    seedFilePattern(tmpfile.fd, 0, n);

    ASSERT_EQ(static_cast<ssize_t>(io_bytes), hipFileRead(tmpfile_handle, device_buffer, io_bytes, 0, 0));

    launchAndVerify(bufferStart(), bufferElems(), bufferStart(), n, defaultGrid(n),
                    dim3(kDefaultWorkgroupSize));

    std::vector<int32_t> buf = readbackInts(device_buffer, 0, n);
    assertDoubledPattern(buf.data(), n);
}

// hipFileRead + hipFileWrite at offset into device buffer (places sentinels before and after data).
// i.e., verify that data from hipFileRead does not clobber outside specified device buffer.
TEST_P(HipFileVerify, RoundTripGuardsDeviceSlack)
{
    const size_t n       = elems();
    const hoff_t buf_off = static_cast<hoff_t>(kFourKiB); // one head device sentinel region

    seedFilePattern(tmpfile.fd, 0, n);
    ASSERT_EQ(static_cast<ssize_t>(io_bytes),
              hipFileRead(tmpfile_handle, device_buffer, io_bytes, 0, buf_off));

    // Device layout (each sentinel region slackElems() ints, data n ints):
    // [head device sentinel region][data][tail device sentinel region]
    int32_t *data = bufferStart() + slackElems();
    launchAndVerify(bufferStart(), bufferElems(), data, n, defaultGrid(n), dim3(kDefaultWorkgroupSize));

    ASSERT_EQ(static_cast<ssize_t>(io_bytes),
              hipFileWrite(tmpfile_handle, device_buffer, io_bytes, 0, buf_off));

    std::vector<int32_t> file = readFileInts(tmpfile.fd, 0, n);
    assertDoubledPattern(file.data(), n);
}

// hipFileRead + hipFileWrite at offset into file buffer (places sentinels before and after data).
// i.e., verify that data from hipFileWrite does not clobber outside specified file buffer.
TEST_P(HipFileVerify, RoundTripGuardsFileSlack)
{
    const size_t n         = elems();
    const size_t bracket_n = kFourKiB / sizeof(int32_t);    // file sentinel region head/tail each
    const hoff_t file_off  = static_cast<hoff_t>(kFourKiB); // data starts after head
    const size_t total_n   = bracket_n + n + bracket_n;     // head + data + tail
    const hoff_t tail_off  = static_cast<hoff_t>((bracket_n + n) * sizeof(int32_t));

    // File layout (each sentinel region bracket_n ints, data n ints):
    // [head file sentinel region = -1][data = i+1][tail file sentinel region = -1].
    seedFileConstant(tmpfile.fd, 0, bracket_n, kSentinel);
    seedFilePattern(tmpfile.fd, file_off, n);
    seedFileConstant(tmpfile.fd, tail_off, bracket_n, kSentinel);

    ASSERT_EQ(static_cast<ssize_t>(io_bytes),
              hipFileRead(tmpfile_handle, device_buffer, io_bytes, file_off, 0));

    launchAndVerify(bufferStart(), bufferElems(), bufferStart(), n, defaultGrid(n),
                    dim3(kDefaultWorkgroupSize));

    ASSERT_EQ(static_cast<ssize_t>(io_bytes),
              hipFileWrite(tmpfile_handle, device_buffer, io_bytes, file_off, 0));

    std::vector<int32_t> file = readFileInts(tmpfile.fd, 0, total_n);
    assertConstant(file.data(), 0, bracket_n, kSentinel);
    assertDoubledPattern(file.data() + bracket_n, n);
    assertConstant(file.data(), bracket_n + n, total_n, kSentinel);
}

static std::string
verifyName(const testing::TestParamInfo<HipFileVerify::ParamType> &info)
{
    return std::get<0>(info.param).name + "_" + std::get<1>(info.param).name;
}

INSTANTIATE_TEST_SUITE_P(, HipFileVerify,
                         testing::Combine(testing::ValuesIn(io_test_params),
                                          testing::Values(SizeParam{kFourKiB, "sub_chunk"},
                                                          SizeParam{kChunkBytes - kFourKiB, "near_chunk"},
                                                          SizeParam{kChunkBytes, "exact_chunk"},
                                                          SizeParam{kChunkBytes + kFourKiB, "cross_chunk"},
                                                          SizeParam{2 * kChunkBytes, "multi_chunk"})),
                         verifyName);

// ---------------------------------------------------------------------------
// This test suite performs the same combined hipFileRead + hipFileWrite test that the hipFileVerify test
// suite exercises. We use the same poisoned memory region mechanism to verify hipFile transferred the
// specified data. We also varying the following new parameters:
//   - the stride between cache lines that the modified data touches.
//   - the number of workgroups that modify the data (this can also vary the number of different CUs or XCDs
//   on the GPU that touch the data).
// Additionally, this test suite places the targeted data in the file at an offset past the fallback path's
// chunking boundary to ensure the fallback path operates on the specified chunks.
// ---------------------------------------------------------------------------
namespace {

enum class GridMode { Default, OneWorkgroup, ManyWorkgroups };
struct WorkgroupParam {
    GridMode    mode;
    std::string name;
};

constexpr unsigned kManyWorkgroups = 300;

dim3
gridFor(GridMode mode, size_t n)
{
    switch (mode) {
        case GridMode::OneWorkgroup:
            return dim3(1);
        case GridMode::ManyWorkgroups:
            return dim3(kManyWorkgroups);
        case GridMode::Default:
            return defaultGrid(n);
        default:
            return defaultGrid(n);
    }
}

struct StrideParam {
    size_t      stride;
    std::string name;
};

} // namespace

struct HipFileVerifyCombined
    : public testing::TestWithParam<std::tuple<IoTestParam, SizeParam, WorkgroupParam, StrideParam>> {

    Tmpfile         tmpfile;
    hipFileHandle_t tmpfile_handle{nullptr};
    void           *device_buffer{nullptr};
    size_t          io_bytes{0};     // bytes transferred using the hipFile API
    size_t          buffer_bytes{0}; // total device buffer size

    HipFileVerifyCombined() : tmpfile{test_env.ais_capable_dir}
    {
    }

    IoTestBackend backend() const
    {
        return std::get<0>(GetParam()).backend;
    }

    GridMode gridMode() const
    {
        return std::get<2>(GetParam()).mode;
    }

    size_t stride() const
    {
        return std::get<3>(GetParam()).stride;
    }

    void SetUp() override
    {
        Context<Configuration>::get()->fastpath(false);
        Context<Configuration>::get()->fallback(false);

        switch (backend()) {
            case IoTestBackend::Fastpath:
                Context<Configuration>::get()->fastpath(true);
                break;
            case IoTestBackend::Fallback:
                Context<Configuration>::get()->fallback(true);
                break;
            default:
                FAIL() << "Unsupported IoTestBackend";
        }

        io_bytes = std::get<1>(GetParam()).bytes;
        // Over-allocate a device sentinel region on each side of the data.
        // Device layout (each sentinel region kFourKiB, data io_bytes):
        // [head device sentinel region][data][tail device sentinel region].
        buffer_bytes = io_bytes + 2 * kFourKiB;

        // File layout (each sentinel region kFourKiB, data io_bytes; data begins at file
        // offset kCombinedFileOff past the chunk boundary):
        // [head file sentinel region][data][tail file sentinel region].
        const hoff_t tail_off = kCombinedFileOff + static_cast<hoff_t>(io_bytes);
        ASSERT_EQ(0, ftruncate(tmpfile.fd, tail_off + static_cast<hoff_t>(kFourKiB)));

        hipFileDescr_t descr{};
        descr.type      = hipFileHandleTypeOpaqueFD;
        descr.handle.fd = tmpfile.fd;
        ASSERT_EQ(HIPFILE_SUCCESS, hipFileHandleRegister(&tmpfile_handle, &descr));

        ASSERT_EQ(hipSuccess, hipMalloc(&device_buffer, buffer_bytes));
        ASSERT_EQ(hipSuccess, hipMemset(device_buffer, kSentinelByte, buffer_bytes));
        // hipMemset is not synchronous w.r.t. the host, and hipFileRead is not ordered w.r.t. the stream.
        ASSERT_EQ(hipSuccess, hipDeviceSynchronize());

        if (backend() == IoTestBackend::Fastpath) {
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

    size_t elems() const
    {
        return io_bytes / sizeof(int32_t);
    }

    size_t bufferElems() const
    {
        return buffer_bytes / sizeof(int32_t);
    }

    int32_t *bufferStart() const
    {
        return static_cast<int32_t *>(device_buffer);
    }
};

TEST_P(HipFileVerifyCombined, RoundTripGuardsAllRegions)
{
    const size_t n        = elems();
    const size_t slack_n  = kFourKiB / sizeof(int32_t);
    const hoff_t buf_off  = static_cast<hoff_t>(kFourKiB); // device head device sentinel region
    const hoff_t file_off = kCombinedFileOff;
    const hoff_t head_off = file_off - static_cast<hoff_t>(kFourKiB); // head file sentinel region
    const hoff_t tail_off = file_off + static_cast<hoff_t>(io_bytes);
    const size_t kStride  = stride();

    // File layout (hole is kChunkBytes, each sentinel region slack_n ints, data n ints):
    // [unwritten hole = 0][head file sentinel region = -1][data = i+1][tail file sentinel region = -1].
    seedFileConstant(tmpfile.fd, head_off, slack_n, kSentinel);
    seedFilePattern(tmpfile.fd, file_off, n);
    seedFileConstant(tmpfile.fd, tail_off, slack_n, kSentinel);

    ASSERT_EQ(static_cast<ssize_t>(io_bytes),
              hipFileRead(tmpfile_handle, device_buffer, io_bytes, file_off, buf_off));

    // Device layout (each sentinel region slackElems() ints, data n ints):
    // [head device sentinel region][data][tail device sentinel region].
    int32_t *data = bufferStart() + slackElems();
    launchAndVerify(bufferStart(), bufferElems(), data, n, gridFor(gridMode(), n),
                    dim3(kDefaultWorkgroupSize), kStride);

    ASSERT_EQ(static_cast<ssize_t>(io_bytes),
              hipFileWrite(tmpfile_handle, device_buffer, io_bytes, file_off, buf_off));

    assertHoleZero(tmpfile.fd, 0, head_off);
    std::vector<int32_t> head = readFileInts(tmpfile.fd, head_off, slack_n);
    assertConstant(head.data(), 0, slack_n, kSentinel);
    std::vector<int32_t> body = readFileInts(tmpfile.fd, file_off, n);
    assertModifiedPattern(body.data(), n, kStride);
    std::vector<int32_t> tail = readFileInts(tmpfile.fd, tail_off, slack_n);
    assertConstant(tail.data(), 0, slack_n, kSentinel);
}

static std::string
combinedName(const testing::TestParamInfo<HipFileVerifyCombined::ParamType> &info)
{
    return std::get<0>(info.param).name + "_" + std::get<1>(info.param).name + "_" +
           std::get<2>(info.param).name + "_" + std::get<3>(info.param).name;
}

INSTANTIATE_TEST_SUITE_P(Sizes, HipFileVerifyCombined,
                         testing::Combine(testing::ValuesIn(io_test_params),
                                          testing::ValuesIn(combined_sizes),
                                          testing::Values(WorkgroupParam{GridMode::Default, "auto"}),
                                          testing::Values(StrideParam{2, "stride2"},
                                                          StrideParam{32, "stride32"},
                                                          StrideParam{64, "stride64"})),
                         combinedName);

// Use a large sweep in combination with many workgroups to exercise the behaviour of only some CUs modify
// data.
INSTANTIATE_TEST_SUITE_P(Workgroups, HipFileVerifyCombined,
                         testing::Combine(testing::ValuesIn(io_test_params),
                                          testing::Values(SizeParam{2 * kChunkBytes, "multi_chunk"}),
                                          testing::Values(WorkgroupParam{GridMode::OneWorkgroup, "1wg"},
                                                          WorkgroupParam{GridMode::ManyWorkgroups, "300wg"}),
                                          testing::Values(StrideParam{2, "stride2"},
                                                          StrideParam{512, "stride512"})),
                         combinedName);

// ---------------------------------------------------------------------------
// This test suite exercises the same hipFileRead + hipFileWrite behaviour, in combination with the behaviour
// of extending the file length to ensure that untouched regions of a file that is extended as a result of
// hipFileWrite are either the previously present data, or a hole of 0-initialized data, matching the POSIX
// behaviour.
// ---------------------------------------------------------------------------
namespace {

struct ScenarioParam {
    ExtendCase  ext;
    std::string name;
};

HIPFILE_WARN_NO_EXIT_DTOR_OFF
const std::array<ScenarioParam, 5> extend_scenarios{{
    {appendFromEmpty(), "extend_empty_contiguous"},
    {appendAt(kChunkOff), "append_aligned"},
    {holeAfter(0, kChunkOff), "hole_from_empty"},
    {holeAfter(kChunkOff, kChunkOff), "hole_from_aligned"},
    {overwriteAppend(kChunkOff, kFourKiBOff / 2), "overwrite_and_append"},
}};
HIPFILE_WARN_NO_EXIT_DTOR_ON

} // namespace

struct HipFileExtend : public testing::TestWithParam<std::tuple<IoTestParam, ScenarioParam, SizeParam>> {

    Tmpfile         tmpfile;
    hipFileHandle_t tmpfile_handle{nullptr};
    void           *device_buffer{nullptr};
    size_t          io_bytes{0};
    size_t          buffer_bytes{0};
    hoff_t          base_len{0};
    hoff_t          file_off{0};

    HipFileExtend() : tmpfile{test_env.ais_capable_dir}
    {
    }

    IoTestBackend backend() const
    {
        return std::get<0>(GetParam()).backend;
    }

    void SetUp() override
    {
        Context<Configuration>::get()->fastpath(false);
        Context<Configuration>::get()->fallback(false);
        switch (backend()) {
            case IoTestBackend::Fastpath:
                Context<Configuration>::get()->fastpath(true);
                break;
            case IoTestBackend::Fallback:
                Context<Configuration>::get()->fallback(true);
                break;
            default:
                FAIL() << "Unsupported IoTestBackend";
        }

        const ScenarioParam &sc = std::get<1>(GetParam());
        base_len                = sc.ext.base_len;
        file_off                = sc.ext.file_off;
        io_bytes                = std::get<2>(GetParam()).bytes;

        // Every scenario must actually extend the file; the final-size assertion below depends on it.
        ASSERT_GT(file_off + static_cast<hoff_t>(io_bytes), base_len);

        // Size file to base_len to be extended.
        ASSERT_EQ(0, ftruncate(tmpfile.fd, static_cast<off_t>(base_len)));

        // One head sentinel region + data + one tail sentinel region.
        // [head device sentinel region][data][tail device sentinel region].
        buffer_bytes = io_bytes + 2 * kFourKiB;

        hipFileDescr_t descr{};
        descr.type      = hipFileHandleTypeOpaqueFD;
        descr.handle.fd = tmpfile.fd;
        ASSERT_EQ(HIPFILE_SUCCESS, hipFileHandleRegister(&tmpfile_handle, &descr));

        ASSERT_EQ(hipSuccess, hipMalloc(&device_buffer, buffer_bytes));
        ASSERT_EQ(hipSuccess, hipMemset(device_buffer, kSentinelByte, buffer_bytes));

        if (backend() == IoTestBackend::Fastpath) {
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

    size_t elems() const
    {
        return io_bytes / sizeof(int32_t);
    }
    size_t bufferElems() const
    {
        return buffer_bytes / sizeof(int32_t);
    }
    int32_t *bufferStart() const
    {
        return static_cast<int32_t *>(device_buffer);
    }
};

TEST_P(HipFileExtend, Extends)
{
    const size_t     n       = elems();
    const size_t     slack_n = slackElems();
    const hoff_t     buf_off = static_cast<hoff_t>(kFourKiB);
    constexpr size_t kStride = 2;

    // Device layout (each sentinel region slack_n ints, data n ints):
    // [head device sentinel region][data][tail device sentinel region].
    // data starts after the head device sentinel region; seed the index pattern there.
    int32_t *data = bufferStart() + slack_n;
    seedDevicePattern(device_buffer, buf_off, n);

    launchAndVerify(bufferStart(), bufferElems(), data, n, defaultGrid(n), dim3(kDefaultWorkgroupSize),
                    kStride);

    // File layout after the write (preserved region is preserved_n ints, data is n ints, hole spans
    // [base_len, file_off) and is empty when we append to the file):
    // [preserved = -1][hole = 0][data = stride-modified i+1], when file_off >= base_len.
    // [preserved = -1][data = stride-modified i+1], when file_off < base_len.
    const hoff_t preserved_end = (file_off < base_len) ? file_off : base_len;
    const size_t preserved_n   = static_cast<size_t>(preserved_end) / sizeof(int32_t);
    if (preserved_n > 0) {
        seedFileConstant(tmpfile.fd, 0, preserved_n, kSentinel);
    }

    ASSERT_EQ(static_cast<ssize_t>(io_bytes),
              hipFileWrite(tmpfile_handle, device_buffer, io_bytes, file_off, buf_off));

    std::vector<int32_t> body = readFileInts(tmpfile.fd, file_off, n);
    assertModifiedPattern(body.data(), n, kStride);

    // Hole was zero-filled.
    if (file_off > base_len) {
        assertHoleZero(tmpfile.fd, base_len, file_off);
    }

    // Final size correct.
    ASSERT_EQ(file_off + static_cast<hoff_t>(io_bytes), fileSize(tmpfile.fd));

    // Existing data below the write is fully intact.
    if (preserved_n > 0) {
        std::vector<int32_t> head = readFileInts(tmpfile.fd, 0, preserved_n);
        assertConstant(head.data(), 0, preserved_n, kSentinel);
    }
}

static std::string
extendName(const testing::TestParamInfo<HipFileExtend::ParamType> &info)
{
    return std::get<0>(info.param).name + "_" + std::get<1>(info.param).name + "_" +
           std::get<2>(info.param).name;
}

INSTANTIATE_TEST_SUITE_P(, HipFileExtend,
                         testing::Combine(testing::ValuesIn(io_test_params),
                                          testing::ValuesIn(extend_scenarios),
                                          testing::Values(SizeParam{kFourKiB, "small"},
                                                          SizeParam{kChunkBytes + kFourKiB, "large"})),
                         extendName);

HIPFILE_WARN_NO_GLOBAL_CTOR_ON
