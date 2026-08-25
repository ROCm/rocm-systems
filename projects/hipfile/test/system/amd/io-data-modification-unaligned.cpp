/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

#include "context.h"
#include "configuration.h"
#include "hipfile-warnings.h"
#include "hipfile.h"

#include "io-scenario.h"
#include "io-verify.h"
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
struct HipFileVerifyUnaligned : public DataModificationBase<ByteElementPolicy> {
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

TEST_P(HipFileVerifyUnaligned, RoundTripGuardsAllRegions)
{
    ASSERT_NO_FATAL_FAILURE(runAllRegionsTest(*this));
}

constexpr hoff_t kFileOffBase = static_cast<hoff_t>(kChunkBytes + 4_KiB);
constexpr hoff_t kBufOffBase  = static_cast<hoff_t>(4_KiB);

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
    {4_KiB, "small_aligned"},
    {4_KiB + 1, "small_unaligned"},
    {kChunkBytes + 4_KiB, "large_aligned"},
    {kChunkBytes + 4_KiB + 1, "large_unaligned"},
}};
HIPFILE_WARN_NO_EXIT_DTOR_ON

// Roughly a matrix of {device buffer aligned, unaligned} x {file offset aligned, unaligned} x {size
// aligned, unaligned}.
INSTANTIATE_TEST_SUITE_P(, HipFileVerifyUnaligned,
                         testing::ValuesIn(IoTestScenarioSet{IoTestScenario{.stride = 2}}
                                               .over(&IoTestScenario::file_off, kFileOffs)
                                               .over(&IoTestScenario::buf_off, kBufOffs)
                                               .over(&IoTestScenario::io_bytes, kUnalignedSizes)
                                               .build()),
                         ioTestScenarioName);

// ---------------------------------------------------------------------------
// This test suite exercises the behaviour of extending the length of a file with an unaligned hipFileWrite,
// to ensure that untouched regions of a file that is extended are either the previously present data, or a
// hole of 0-initialized data, matching the POSIX behaviour. The length of the file before the write, the file
// offset of the write, the size of the write, and the device buffer offset each vary between aligned and one
// byte past a boundary.
//
// NOTE: unaligned I/O only uses the fallback path.
// ---------------------------------------------------------------------------
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

constexpr hoff_t kGapAligned   = kFourKiBOff; // distance from EOF to the start of a write that leaves a hole
constexpr hoff_t kGapUnaligned = kFourKiBOff + 1;

// The bytes from the file that a write that also extends the file will overlap.
constexpr hoff_t kOverlap = kFourKiBOff / 2;

constexpr size_t kSizeAligned    = 4_KiB; // io_bytes
constexpr size_t kSizeUnaligned  = 4_KiB + 1;
constexpr size_t kLargeAligned   = kChunkBytes + 4_KiB; // spans more than one chunk
constexpr size_t kLargeUnaligned = kChunkBytes + 4_KiB + 1;

constexpr hoff_t kBufAligned   = 0; // where the data starts within the device buffer
constexpr hoff_t kBufUnaligned = 1;

constexpr size_t kStride = 2; // every scenario modifies every other element

static IoTestScenario
extendScenario(const char *name, ExtendCase ext, size_t io_bytes = kSizeAligned, hoff_t buf_off = kBufAligned)
{
    return IoTestScenario{
        .name = name, .io_bytes = io_bytes, .buf_off = buf_off, .stride = kStride, .ext = ext};
}

// These are hard to express using `testing::Combine` or `over`, so we use a helper function.
HIPFILE_WARN_NO_EXIT_DTOR_OFF
const std::array<IoTestScenario, 16> kExtendUnalignedScenarios{{
    // Append to empty file.
    extendScenario("empty_contiguous_aligned_size", appendFromEmpty()),
    extendScenario("empty_contiguous_unaligned_size", appendFromEmpty(), kSizeUnaligned),
    // Contiguous append onto a non-empty file.
    extendScenario("append_unaligned_base", appendAt(kBaseUnaligned)),
    extendScenario("append_aligned_base_unaligned_size", appendAt(kBaseAligned), kSizeUnaligned),
    // Write past EOF, creating a hole from the existing EOF of the file to the start of the write.
    extendScenario("hole_from_empty_aligned_off", holeAfter(kBaseEmpty, kGapAligned)),
    extendScenario("hole_from_empty_unaligned_off", holeAfter(kBaseEmpty, kGapUnaligned)),
    extendScenario("hole_from_unaligned_base", holeAfter(kBaseUnaligned, kGapUnaligned)),
    extendScenario("hole_unaligned_off_unaligned_size", holeAfter(kBaseAligned, kGapUnaligned),
                   kSizeUnaligned),
    // Unaligned device buffer.
    extendScenario("append_unaligned_buffer", appendAt(kBaseAligned), kSizeAligned, kBufUnaligned),
    extendScenario("hole_unaligned_buffer", holeAfter(kBaseEmpty, kGapAligned), kSizeAligned, kBufUnaligned),
    extendScenario("append_unaligned_base_unaligned_buffer", appendAt(kBaseUnaligned), kSizeUnaligned,
                   kBufUnaligned),
    extendScenario("hole_all_unaligned", holeAfter(kBaseUnaligned, kGapUnaligned), kSizeUnaligned,
                   kBufUnaligned),
    // Transfers larger than the fallback chunking size.
    extendScenario("large_hole_cross_chunk", holeAfter(kBaseEmpty, kGapAligned), kLargeAligned),
    extendScenario("large_append_unaligned_base", appendAt(kBaseUnaligned), kLargeUnaligned),
    // Write before the existing EOF, but extend the length of the file with the same write.
    extendScenario("overwrite_append_aligned_base", overwriteAppend(kBaseAligned, kOverlap)),
    extendScenario("overwrite_append_all_unaligned", overwriteAppend(kBaseUnaligned, kOverlap),
                   kSizeUnaligned, kBufUnaligned),
}};
HIPFILE_WARN_NO_EXIT_DTOR_ON

INSTANTIATE_TEST_SUITE_P(, HipFileExtendUnaligned, testing::ValuesIn(kExtendUnalignedScenarios),
                         ioTestScenarioName);

HIPFILE_WARN_NO_GLOBAL_CTOR_ON
