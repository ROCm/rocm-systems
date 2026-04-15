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

#include <cctype>
#include <string>

#include <gtest/gtest.h>
#include "../../utils/rocvideodecode/surface_format_utils.h"

struct SurfaceFormatTestCase {
    const char*              description;
    rocDecVideoChromaFormat  chroma_format;
    uint8_t                  bitdepth_minus_8;
    rocDecVideoSurfaceFormat expected;
};

class SelectSurfaceFormatTest : public ::testing::TestWithParam<SurfaceFormatTestCase> {};

TEST_P(SelectSurfaceFormatTest, ReturnsExpectedFormat) {
    const auto& tc = GetParam();
    EXPECT_EQ(SelectSurfaceFormat(tc.chroma_format, tc.bitdepth_minus_8), tc.expected)
        << "Failed for: " << tc.description;
}

static const SurfaceFormatTestCase kTestCases[] = {
    // ── 4:2:0 ───────────────────────────────────────────────────────────
    {"4:2:0  8-bit  → NV12",
     rocDecVideoChromaFormat_420, 0, rocDecVideoSurfaceFormat_NV12},
    {"4:2:0  10-bit → P016",
     rocDecVideoChromaFormat_420, 2, rocDecVideoSurfaceFormat_P016},
    {"4:2:0  12-bit → P016",
     rocDecVideoChromaFormat_420, 4, rocDecVideoSurfaceFormat_P016},

    // ── Monochrome ──────────────────────────────────────────────────────
    // Bug roc_video_dec.cpp:308: `|| rocDecVideoChromaFormat_Monochrome`
    // (enum value 0) is always false, so Monochrome falls through unhandled.
    {"Mono   8-bit  → NV12",
     rocDecVideoChromaFormat_Monochrome, 0, rocDecVideoSurfaceFormat_NV12},
    {"Mono   10-bit → P016  (fails with bug: gets Native/unset)",
     rocDecVideoChromaFormat_Monochrome, 2, rocDecVideoSurfaceFormat_P016},
    {"Mono   12-bit → P016",
     rocDecVideoChromaFormat_Monochrome, 4, rocDecVideoSurfaceFormat_P016},

    // ── 4:4:4 ───────────────────────────────────────────────────────────
    {"4:4:4  8-bit  → YUV444",
     rocDecVideoChromaFormat_444, 0, rocDecVideoSurfaceFormat_YUV444},
    {"4:4:4  10-bit → YUV444_16Bit",
     rocDecVideoChromaFormat_444, 2, rocDecVideoSurfaceFormat_YUV444_16Bit},
    {"4:4:4  12-bit → YUV444_16Bit",
     rocDecVideoChromaFormat_444, 4, rocDecVideoSurfaceFormat_YUV444_16Bit},

    // ── 4:2:2 ───────────────────────────────────────────────────────────
    {"4:2:2  8-bit  → YUV422",
     rocDecVideoChromaFormat_422, 0, rocDecVideoSurfaceFormat_YUV422},
    {"4:2:2  10-bit → YUV422_16Bit",
     rocDecVideoChromaFormat_422, 2, rocDecVideoSurfaceFormat_YUV422_16Bit},
    {"4:2:2  12-bit → YUV422_16Bit",
     rocDecVideoChromaFormat_422, 4, rocDecVideoSurfaceFormat_YUV422_16Bit},

    // ── Unknown / future chroma formats ─────────────────────────────────
    // Defence in depth: unrecognized chroma format returns Native (sentinel)
    // rather than silently picking NV12 — the "incomplete dispatch" bug class.
    {"Unknown(99)  8-bit  → Native (error sentinel)",
     static_cast<rocDecVideoChromaFormat>(99), 0, rocDecVideoSurfaceFormat_Native},
    {"Unknown(99)  10-bit → Native (error sentinel)",
     static_cast<rocDecVideoChromaFormat>(99), 2, rocDecVideoSurfaceFormat_Native},
    {"Unknown(-1)  8-bit  → Native (error sentinel)",
     static_cast<rocDecVideoChromaFormat>(-1), 0, rocDecVideoSurfaceFormat_Native},
};

INSTANTIATE_TEST_SUITE_P(
    SurfaceFormat,
    SelectSurfaceFormatTest,
    ::testing::ValuesIn(kTestCases),
    [](const ::testing::TestParamInfo<SurfaceFormatTestCase>& info) {
        // Generate a test name from the description, replacing non-alphanumeric chars.
        std::string name;
        for (char c : std::string(info.param.description)) {
            if (std::isalnum(c)) name += c;
            else if (c == ' ' && !name.empty() && name.back() != '_') name += '_';
        }
        return name;
    }
);
