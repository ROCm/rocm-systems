/*
Copyright (c) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

// Table-driven unit test for the chroma-format -> surface-format selection in
// RocVideoDecoder (roc_video_dec.cpp). Builds host-only (no HIP / no GPU) against
// the mirror in surface_format_utils.h.

#include <cctype>
#include <string>

#include <gtest/gtest.h>

#include "surface_format_utils.h"

struct SurfaceFormatCase {
    const char*              description;       // what this row exercises
    rocDecVideoChromaFormat  chroma_format;     // input: chroma subsampling
    uint8_t                  bitdepth_minus_8;  // input: 0 = 8-bit, >0 = higher
    rocDecVideoSurfaceFormat expected;          // expected outcome
};

class SelectSurfaceFormatTest : public ::testing::TestWithParam<SurfaceFormatCase> {};

TEST_P(SelectSurfaceFormatTest, ReturnsExpectedFormat) {
    const auto& tc = GetParam();
    EXPECT_EQ(SelectSurfaceFormat(tc.chroma_format, tc.bitdepth_minus_8), tc.expected)
        << "case: " << tc.description;
}

static const SurfaceFormatCase kCases[] = {
    // ---- positive: 4:2:0 (8/10/12-bit) ----
    {"positive: 4:2:0 8-bit -> NV12",   rocDecVideoChromaFormat_420, 0, rocDecVideoSurfaceFormat_NV12},
    {"positive: 4:2:0 10-bit -> P016",  rocDecVideoChromaFormat_420, 2, rocDecVideoSurfaceFormat_P016},
    {"positive: 4:2:0 12-bit -> P016",  rocDecVideoChromaFormat_420, 4, rocDecVideoSurfaceFormat_P016},

    // ---- bug-catching: Monochrome must map like 4:2:0 (NV12/P016) ----
    // These fail pre-fix: roc_video_dec.cpp:309 drops Monochrome (returns Native).
    {"bug: Monochrome 8-bit -> NV12",   rocDecVideoChromaFormat_Monochrome, 0, rocDecVideoSurfaceFormat_NV12},
    {"bug: Monochrome 10-bit -> P016",  rocDecVideoChromaFormat_Monochrome, 2, rocDecVideoSurfaceFormat_P016},
    {"bug: Monochrome 12-bit -> P016",  rocDecVideoChromaFormat_Monochrome, 4, rocDecVideoSurfaceFormat_P016},

    // ---- positive: 4:4:4 (8/10/12-bit) ----
    {"positive: 4:4:4 8-bit -> YUV444",         rocDecVideoChromaFormat_444, 0, rocDecVideoSurfaceFormat_YUV444},
    {"positive: 4:4:4 10-bit -> YUV444_16Bit",  rocDecVideoChromaFormat_444, 2, rocDecVideoSurfaceFormat_YUV444_16Bit},
    {"positive: 4:4:4 12-bit -> YUV444_16Bit",  rocDecVideoChromaFormat_444, 4, rocDecVideoSurfaceFormat_YUV444_16Bit},

    // ---- positive: 4:2:2 (8/10/12-bit) ----
    {"positive: 4:2:2 8-bit -> YUV422",         rocDecVideoChromaFormat_422, 0, rocDecVideoSurfaceFormat_YUV422},
    {"positive: 4:2:2 10-bit -> YUV422_16Bit",  rocDecVideoChromaFormat_422, 2, rocDecVideoSurfaceFormat_YUV422_16Bit},

    // ---- boundary: bit-depth 0 vs 1 (the only branch point) ----
    {"boundary: 4:2:0 bitdepth_minus_8=0 -> NV12 (8-bit path)", rocDecVideoChromaFormat_420, 0, rocDecVideoSurfaceFormat_NV12},
    {"boundary: 4:2:0 bitdepth_minus_8=1 -> P016 (>8-bit path)", rocDecVideoChromaFormat_420, 1, rocDecVideoSurfaceFormat_P016},

    // ---- corner/negative: unrecognized chroma -> Native (defensive sentinel) ----
    {"corner: unknown chroma 99 -> Native",  static_cast<rocDecVideoChromaFormat>(99), 0, rocDecVideoSurfaceFormat_Native},
    {"corner: unknown chroma -1 -> Native",  static_cast<rocDecVideoChromaFormat>(-1), 0, rocDecVideoSurfaceFormat_Native},
};

INSTANTIATE_TEST_SUITE_P(
    SurfaceFormat,
    SelectSurfaceFormatTest,
    ::testing::ValuesIn(kCases),
    [](const ::testing::TestParamInfo<SurfaceFormatCase>& info) {
        std::string name;
        for (char c : std::string(info.param.description)) {
            if (std::isalnum(static_cast<unsigned char>(c))) name += c;
            else if (!name.empty() && name.back() != '_') name += '_';
        }
        while (!name.empty() && name.back() == '_') name.pop_back();
        return name;
    });
