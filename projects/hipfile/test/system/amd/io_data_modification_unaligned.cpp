/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

#include "context.h"
#include "configuration.h"
#include "hipfile-warnings.h"
#include "hipfile.h"

#include "io-verify.hpp"
#include "test-common.h"
#include "test-options.h"

#include <array>
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
// This test suite exercises the ability of hipFileRead and hipFileWrite on unaligned I/O while varying the
// size of I/O, and the source of unaligned-ness (device buffer offset vs. file offset vs. size).
//
// NOTE: unaligned I/O only uses the fallback path.
//
// We verify hipFileRead and hipFileWrite behave as expected by guarding the targeted regions of the
// device memory allocation and the region of the file that data will be read from and written to, surrounding
// them with poisoned memory containing sentinel values that would tell us if hipFile ever read or wrote data
// to a location that the user did not specify.
// ---------------------------------------------------------------------------
namespace {

constexpr hoff_t kFileOffBase = static_cast<hoff_t>(kChunkBytes + kFourKiB);
constexpr hoff_t kBufOffBase  = static_cast<hoff_t>(kFourKiB);

// Data starts at `base+delta`, where `base` is one of `kFileOffBase` or `kBufOffBase`.
struct OffParam {
    hoff_t      delta;
    std::string name;
};

} // namespace

struct HipFileVerifyUnaligned : public testing::TestWithParam<std::tuple<OffParam, OffParam, SizeParam>> {

    Tmpfile         tmpfile;
    hipFileHandle_t tmpfile_handle{nullptr};
    void           *device_buffer{nullptr};
    size_t          io_bytes{0};     // bytes transferred using the hipFile API
    size_t          buffer_bytes{0}; // total device buffer size
    hoff_t          file_off{0};     // (possibly unaligned) file offset of the data
    hoff_t          buf_off{0};      // (possibly unaligned) device buffer offset of the data

    HipFileVerifyUnaligned() : tmpfile{test_env.ais_capable_dir}
    {
    }

    hoff_t fileDelta() const
    {
        return std::get<0>(GetParam()).delta;
    }
    hoff_t bufferDelta() const
    {
        return std::get<1>(GetParam()).delta;
    }

    void SetUp() override
    {
        Context<Configuration>::get()->fastpath(false);
        Context<Configuration>::get()->fallback(true); // fallback only

        io_bytes = std::get<2>(GetParam()).bytes;
        file_off = kFileOffBase + fileDelta();
        buf_off  = kBufOffBase + bufferDelta();

        // Device layout (each sentinel region ~kFourKiB (+-1), data io_bytes):
        // [head device sentinel region][data][tail device sentinel region]
        buffer_bytes = io_bytes + 2 * kFourKiB;

        // File layout (each sentinel region kFourKiB, data io_bytes; data begins at file
        // offset file_off past the chunk boundary). Size through the tail bracket:
        // [head file sentinel region][data][tail file sentinel region]
        const hoff_t tail_off = file_off + static_cast<hoff_t>(io_bytes);
        ASSERT_EQ(0, ftruncate(tmpfile.fd, tail_off + static_cast<hoff_t>(kFourKiB)));

        hipFileDescr_t descr{};
        descr.type      = hipFileHandleTypeOpaqueFD;
        descr.handle.fd = tmpfile.fd;
        ASSERT_EQ(HIPFILE_SUCCESS, hipFileHandleRegister(&tmpfile_handle, &descr));

        ASSERT_EQ(hipSuccess, hipMalloc(&device_buffer, buffer_bytes));
        ASSERT_EQ(hipSuccess, hipMemset(device_buffer, kByteDevSlack, buffer_bytes));
        // hipMemset is not synchronous w.r.t. the host, and hipFileRead is not ordered w.r.t. the stream.
        ASSERT_EQ(hipSuccess, hipDeviceSynchronize());
    }

    void TearDown() override
    {
        if (device_buffer) {
            ASSERT_EQ(hipSuccess, hipFree(device_buffer));
        }
        hipFileHandleDeregister(tmpfile_handle);
    }

    unsigned char *bufferStart() const
    {
        return static_cast<unsigned char *>(device_buffer);
    }
};

TEST_P(HipFileVerifyUnaligned, RoundTripGuardsAllRegions)
{
    const size_t n        = io_bytes; // one byte per element
    const size_t slack_n  = kFourKiB; // file sentinel region bracket size in bytes
    const hoff_t head_off = file_off - static_cast<hoff_t>(kFourKiB); // head file sentinel region byte offset
    const hoff_t tail_off = file_off + static_cast<hoff_t>(io_bytes);
    constexpr size_t kStride = 2; // every-other byte

    // File layout (each sentinel region slack_n bytes, data n bytes):
    // [unwritten sparse hole = 0][head file sentinel region = 0x55][data = 0xFF][tail file sentinel region =
    // 0x55].
    seedFileBytesConstant(tmpfile.fd, head_off, slack_n, kByteFileSlack);
    seedFileBytesConstant(tmpfile.fd, file_off, n, kByteEntry);
    seedFileBytesConstant(tmpfile.fd, tail_off, slack_n, kByteFileSlack);

    ASSERT_EQ(static_cast<ssize_t>(io_bytes),
              hipFileRead(tmpfile_handle, device_buffer, io_bytes, file_off, buf_off));

    // Device layout (each sentinel region ~kFourKiB (+-1), data n bytes):
    // [head device sentinel region][data][tail device sentinel region]
    unsigned char *data = bufferStart() + static_cast<size_t>(buf_off);
    launchAndVerifyBytes(bufferStart(), buffer_bytes, data, n, defaultGrid(n), dim3(kDefaultWorkgroupSize),
                         kStride);

    ASSERT_EQ(static_cast<ssize_t>(io_bytes),
              hipFileWrite(tmpfile_handle, device_buffer, io_bytes, file_off, buf_off));

    assertHoleZero(tmpfile.fd, 0, head_off);
    std::vector<unsigned char> head = readFileBytes(tmpfile.fd, head_off, slack_n);
    assertBytesConstant(head.data(), 0, slack_n, kByteFileSlack);
    std::vector<unsigned char> body = readFileBytes(tmpfile.fd, file_off, n);
    assertBytesModified(body.data(), n, kStride);
    std::vector<unsigned char> tail = readFileBytes(tmpfile.fd, tail_off, slack_n);
    assertBytesConstant(tail.data(), 0, slack_n, kByteFileSlack);
}

static std::string
unalignedName(const testing::TestParamInfo<HipFileVerifyUnaligned::ParamType> &info)
{
    return std::get<0>(info.param).name + "_" + std::get<1>(info.param).name + "_" +
           std::get<2>(info.param).name;
}

// Roughly a matrix of {device buffer aligned, unaligned} x {file offset aligned, unaligned} x {size
// aligned, unaligned}.
INSTANTIATE_TEST_SUITE_P(
    , HipFileVerifyUnaligned,
    testing::Combine(testing::Values(OffParam{0, "file_aligned"}, OffParam{1, "file_unaligned"}),
                     testing::Values(OffParam{0, "buffer_aligned"}, OffParam{1, "buffer_unaligned"}),
                     testing::Values(SizeParam{kFourKiB, "small_aligned"},
                                     SizeParam{kFourKiB + 1, "small_unaligned"},
                                     SizeParam{kChunkBytes + kFourKiB, "large_aligned"},
                                     SizeParam{kChunkBytes + kFourKiB + 1, "large_unaligned"})),
    unalignedName);

// ---------------------------------------------------------------------------
// This test suite exercises the behaviour of extending the length of a file with an unaligned hipFileWrite,
// to ensure that untouched regions of a file that is extended are either the previously present data, or a
// hole of 0-initialized data, matching the POSIX behaviour. The length of the file before the write, the file
// offset of the write, the size of the write, and the device buffer offset each vary between aligned and one
// byte past a boundary.
//
// NOTE: unaligned I/O only uses the fallback path.
// ---------------------------------------------------------------------------
namespace {

// Determines the combination of size, offset into the device buffer, offset into the file, and initial length
// of the file.
struct ExtendScenarioParam {
    ExtendCase  ext;
    size_t      io_bytes;
    hoff_t      buf_off;
    std::string name;
};

constexpr hoff_t kBaseEmpty     = 0; // file length before the extending write
constexpr hoff_t kBaseAligned   = kChunkOff;
constexpr hoff_t kBaseUnaligned = kChunkOff + 1;
constexpr hoff_t kGapAligned   = kFourKiBOff; // distance from EOF to the start of a write that creates a hole
constexpr hoff_t kGapUnaligned = kFourKiBOff + 1;
constexpr hoff_t kOverlap =
    kFourKiBOff / 2; // the bytes from the file that a write that also extends the file will overlap
constexpr size_t kSizeAligned    = kFourKiB; // io_bytes
constexpr size_t kSizeUnaligned  = kFourKiB + 1;
constexpr size_t kLargeAligned   = kChunkBytes + kFourKiB; // spans more than one chunk
constexpr size_t kLargeUnaligned = kChunkBytes + kFourKiB + 1;
constexpr hoff_t kBufAligned     = 0; // where the data starts within the device buffer
constexpr hoff_t kBufUnaligned   = 1;

HIPFILE_WARN_NO_EXIT_DTOR_OFF
const std::array<ExtendScenarioParam, 16> extend_scenarios{{
    // Append to empty file.
    {appendFromEmpty(), kSizeAligned, kBufAligned, "empty_contiguous_aligned_size"},
    {appendFromEmpty(), kSizeUnaligned, kBufAligned, "empty_contiguous_unaligned_size"},
    // Contiguous append onto a non-empty file.
    {appendAt(kBaseUnaligned), kSizeAligned, kBufAligned, "append_unaligned_base"},
    {appendAt(kBaseAligned), kSizeUnaligned, kBufAligned, "append_aligned_base_unaligned_size"},
    // Write past EOF, creating a hole from the existing EOF of the file to the start of the write.
    {holeAfter(kBaseEmpty, kGapAligned), kSizeAligned, kBufAligned, "hole_from_empty_aligned_off"},
    {holeAfter(kBaseEmpty, kGapUnaligned), kSizeAligned, kBufAligned, "hole_from_empty_unaligned_off"},
    {holeAfter(kBaseUnaligned, kGapUnaligned), kSizeAligned, kBufAligned, "hole_from_unaligned_base"},
    {holeAfter(kBaseAligned, kGapUnaligned), kSizeUnaligned, kBufAligned,
     "hole_unaligned_off_unaligned_size"},
    // Unaligned device buffer.
    {appendAt(kBaseAligned), kSizeAligned, kBufUnaligned, "append_unaligned_buffer"},
    {holeAfter(kBaseEmpty, kGapAligned), kSizeAligned, kBufUnaligned, "hole_unaligned_buffer"},
    {appendAt(kBaseUnaligned), kSizeUnaligned, kBufUnaligned, "append_unaligned_base_unaligned_buffer"},
    {holeAfter(kBaseUnaligned, kGapUnaligned), kSizeUnaligned, kBufUnaligned, "hole_all_unaligned"},
    // Transfers larger than the fallback chunking size.
    {holeAfter(kBaseEmpty, kGapAligned), kLargeAligned, kBufAligned, "large_hole_cross_chunk"},
    {appendAt(kBaseUnaligned), kLargeUnaligned, kBufAligned, "large_append_unaligned_base"},
    // Write before the existing EOF, but extend the length of the file with the same write.
    {overwriteAppend(kBaseAligned, kOverlap), kSizeAligned, kBufAligned, "overwrite_append_aligned_base"},
    {overwriteAppend(kBaseUnaligned, kOverlap), kSizeUnaligned, kBufUnaligned,
     "overwrite_append_all_unaligned"},
}};
HIPFILE_WARN_NO_EXIT_DTOR_ON

} // namespace

struct HipFileExtendUnaligned : public testing::TestWithParam<ExtendScenarioParam> {

    Tmpfile         tmpfile;
    hipFileHandle_t tmpfile_handle{nullptr};
    void           *device_buffer{nullptr};
    size_t          buffer_bytes{0};

    HipFileExtendUnaligned() : tmpfile{test_env.ais_capable_dir}
    {
    }

    void SetUp() override
    {
        // Fallback only.
        Context<Configuration>::get()->fastpath(false);
        Context<Configuration>::get()->fallback(true);

        const ExtendScenarioParam &r = GetParam();

        // Every scenario must actually extend the file.
        ASSERT_GT(r.ext.file_off + static_cast<hoff_t>(r.io_bytes), r.ext.base_len);

        ASSERT_EQ(0, ftruncate(tmpfile.fd, static_cast<off_t>(r.ext.base_len)));

        // Device layout (each sentinel region ~kFourKiB, data r.io_bytes; data begins at
        // buffer offset r.buf_off, which may be +1 unaligned within the head sentinel region):
        // [head device sentinel region][data][tail device sentinel region]
        buffer_bytes = r.io_bytes + 2 * kFourKiB;

        hipFileDescr_t descr{};
        descr.type      = hipFileHandleTypeOpaqueFD;
        descr.handle.fd = tmpfile.fd;
        ASSERT_EQ(HIPFILE_SUCCESS, hipFileHandleRegister(&tmpfile_handle, &descr));

        ASSERT_EQ(hipSuccess, hipMalloc(&device_buffer, buffer_bytes));
        ASSERT_EQ(hipSuccess, hipMemset(device_buffer, kByteDevSlack, buffer_bytes));
    }

    void TearDown() override
    {
        if (device_buffer) {
            ASSERT_EQ(hipSuccess, hipFree(device_buffer));
        }
        hipFileHandleDeregister(tmpfile_handle);
    }

    unsigned char *bufferStart() const
    {
        return static_cast<unsigned char *>(device_buffer);
    }
};

TEST_P(HipFileExtendUnaligned, Extends)
{
    const ExtendScenarioParam &r       = GetParam();
    const size_t               n       = r.io_bytes;
    constexpr size_t           kStride = 2;

    unsigned char *data = bufferStart() + static_cast<size_t>(r.buf_off);
    ASSERT_EQ(hipSuccess, hipMemset(data, kByteEntry, n));

    launchAndVerifyBytes(bufferStart(), buffer_bytes, data, n, defaultGrid(n), dim3(kDefaultWorkgroupSize),
                         kStride);

    // File layout after the write (preserved region is preserved_n ints, data is n ints, hole spans
    // [base_len, file_off) and is empty when we append to the file):
    // [preserved = -1][hole = 0][data = stride-modified i+1], when file_off >= base_len.
    // [preserved = -1][data = stride-modified i+1], when file_off < base_len.
    const hoff_t preserved_end = (r.ext.file_off < r.ext.base_len) ? r.ext.file_off : r.ext.base_len;
    const size_t preserved_n   = static_cast<size_t>(preserved_end);
    if (preserved_n > 0) {
        seedFileBytesConstant(tmpfile.fd, 0, preserved_n, kByteFileSlack);
    }

    ASSERT_EQ(static_cast<ssize_t>(r.io_bytes),
              hipFileWrite(tmpfile_handle, device_buffer, r.io_bytes, r.ext.file_off, r.buf_off));

    std::vector<unsigned char> body = readFileBytes(tmpfile.fd, r.ext.file_off, n);
    assertBytesModified(body.data(), n, kStride);

    // Hole was zero-filled.
    if (r.ext.file_off > r.ext.base_len) {
        assertHoleZero(tmpfile.fd, r.ext.base_len, r.ext.file_off);
    }

    // Final size.
    ASSERT_EQ(r.ext.file_off + static_cast<hoff_t>(r.io_bytes), fileSize(tmpfile.fd));

    // Existing data below the write is fully intact.
    if (preserved_n > 0) {
        std::vector<unsigned char> head = readFileBytes(tmpfile.fd, 0, preserved_n);
        assertBytesConstant(head.data(), 0, preserved_n, kByteFileSlack);
    }
}

static std::string
extendScenarioName(const testing::TestParamInfo<HipFileExtendUnaligned::ParamType> &info)
{
    return info.param.name;
}

INSTANTIATE_TEST_SUITE_P(, HipFileExtendUnaligned, testing::ValuesIn(extend_scenarios), extendScenarioName);

HIPFILE_WARN_NO_GLOBAL_CTOR_ON
