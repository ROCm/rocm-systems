/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

#include "hipfile-warnings.h"
#include "hipfile.h"

#include "io-scenario.h"
#include "io-verify.h"
#include "test-options.h"

#include <array>
#include <cstdint>
#include <gtest/gtest.h>
#include <hip/hip_runtime_api.h>
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
struct HipFileVerifyCombined : public DataModificationBase<IntElementPolicy> {
    void SetUp() override
    {
        // File layout (each sentinel region 4_KiB, data io_bytes; data begins at file
        // offset kCombinedFileOff past the chunk boundary):
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

INSTANTIATE_TEST_SUITE_P(
    , HipFileVerifyCombined,
    testing::ValuesIn(IoTestScenarioSet{IoTestScenario{.file_off = kCombinedFileOff, .buf_off = kFourKiBOff}}
                          .over(&IoTestScenario::backend, kBackends)
                          .over(&IoTestScenario::io_bytes, kCombinedSizes)
                          .over(&IoTestScenario::stride, kStrides)
                          // A large sweep with many workgroups, exercising the behaviour of only some CUs
                          // modifying the data.
                          .add(IoTestScenarioSet{
                              IoTestScenario{.file_off = kCombinedFileOff, .buf_off = kFourKiBOff}}
                                   .over(&IoTestScenario::backend, kBackends)
                                   .over(&IoTestScenario::io_bytes, kMultiChunkOnly)
                                   .over(&IoTestScenario::grid, kGridSweep)
                                   .over(&IoTestScenario::stride, kStridesWide))
                          .build()),
    ioTestScenarioName);

// ---------------------------------------------------------------------------
// This test suite exercises the same hipFileRead + hipFileWrite behaviour, in combination with the behaviour
// of extending the file length to ensure that untouched regions of a file that is extended as a result of
// hipFileWrite are either the previously present data, or a hole of 0-initialized data, matching the POSIX
// behaviour.
// ---------------------------------------------------------------------------
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
    {overwriteAppend(kChunkOff, kFourKiBOff / 2), "overwrite_and_append"},
}};

const std::array<Axis<size_t>, 2> kExtendSizes{{{4_KiB, "small"}, {kChunkBytes + 4_KiB, "large"}}};
HIPFILE_WARN_NO_EXIT_DTOR_ON

INSTANTIATE_TEST_SUITE_P(, HipFileExtend,
                         testing::ValuesIn(IoTestScenarioSet{
                             IoTestScenario{.buf_off = kFourKiBOff, .stride = 2}}
                                               .over(&IoTestScenario::backend, kBackends)
                                               .over(&IoTestScenario::ext, kExtendCases)
                                               .over(&IoTestScenario::io_bytes, kExtendSizes)
                                               .build()),
                         ioTestScenarioName);

HIPFILE_WARN_NO_GLOBAL_CTOR_ON
