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

// ==========================================================================
// Unit tests for the video layer pointer dispatch logic in GetVideoFrame().
//
// These tests demonstrate bugs in the ORIGINAL code (roc_decoder.cpp lines
// 177-185 as of commit 18f165e424) and verify the FIXED replacement.  Both
// implementations are tested with the same table-driven inputs so the
// behavioral difference is immediately visible in the test output.
//
// No GPU hardware is required — these test pure pointer arithmetic.
//
// To build and run:
//
//   cd projects/rocdecode/test/unit
//   mkdir -p build && cd build
//   cmake .. -DCMAKE_BUILD_TYPE=Debug
//   make -j$(nproc)
//   ./rocdecode_unit_tests
//
// Requires only cmake, a C++17 compiler, and googletest.
// ==========================================================================

#include <gtest/gtest.h>
#include <cstdint>
#include "../../src/rocdecode/video_layer_utils.h"

// -------------------------------------------------------------------------
// Sentinel values to detect untouched output slots
// -------------------------------------------------------------------------
static constexpr uintptr_t kSentinelPtr   = 0xDEADBEEFDEADBEEF;
static constexpr uint32_t  kSentinelPitch = 0xCAFECAFE;

// Both PopulateLayerPointers (fixed) and OriginalPopulateLayerPointers
// (buggy, preserved for comparison) come from video_layer_utils.h.

// -------------------------------------------------------------------------
// Test helpers
// -------------------------------------------------------------------------

// Build a VideoLayerInfo with deterministic, non-zero test values.
// Each layer gets offset = (layer+1)*0x1000, pitch = (layer+1)*256.
static VideoLayerInfo makeTestLayerInfo(
        uint32_t num_layers,
        uint8_t *base_addr = reinterpret_cast<uint8_t*>(0x100000)) {
    VideoLayerInfo info{};
    info.base_ptr = base_addr;
    info.num_layers = num_layers;
    for (uint32_t i = 0; i < kMaxVideoLayers; i++) {
        info.offset[i] = (i + 1) * 0x1000;
        info.pitch[i]  = (i + 1) * 256;
    }
    return info;
}

// Fill output arrays with sentinel values so untouched slots are detectable.
static void initOutputs(void *dev_mem_ptr[kMaxVideoLayers],
                        uint32_t horizontal_pitch[kMaxVideoLayers]) {
    for (uint32_t i = 0; i < kMaxVideoLayers; i++) {
        dev_mem_ptr[i]      = reinterpret_cast<void*>(kSentinelPtr);
        horizontal_pitch[i] = kSentinelPitch;
    }
}

// =========================================================================
// Unified test table
//
// Every test scenario is in ONE table.  To add a new layer count (e.g. when
// hardware adds 4-plane support), add a row here — the test body handles
// both valid and invalid cases automatically.
//
//   expect_success:
//     true  = PopulateLayerPointers should return kVideoLayerSuccess and
//             set exactly num_layers output slots.
//     false = PopulateLayerPointers should return kVideoLayerInvalidParam
//             and leave all output slots untouched.
//
//   original_bug:
//     Brief description of what the original code does wrong for this case,
//     or "" if the original code handles it correctly.  This field exists
//     so a reviewer can see the before/after at a glance.
// =========================================================================

struct LayerTestCase {
    const char *name;
    uint32_t    num_layers;
    bool        expect_success;
    const char *description;
    const char *original_bug;
};

// clang-format off
static const LayerTestCase kAllTestCases[] = {
    //  name                 layers  ok?    description                              original code behavior
    // ----------------------------------------------------------------------------------------------------------------------------------------------------------
    // Valid surface types — these are the real VA-API layer counts
    {"monochrome_Y800",          1, true,  "Y plane only (grayscale)",              ""},
    {"semi_planar_NV12",         2, true,  "Y + interleaved UV (most common: NV12, P010, P012)", ""},
    {"planar_YUV444",            3, true,  "Y + U + V separate planes",             "layer 1 (U) never set — else-if skips it"},

    // Invalid: below minimum
    {"zero_layers",              0, false, "no layers — should never happen",       "silently writes layer 0 anyway, no error returned"},

    // Invalid: above kMaxVideoLayers (currently 3)
    // These ensure future layer counts are explicitly rejected until
    // PopulateLayerPointers is updated to handle them.  Test failures here
    // are a signal that the function AND these tests need updating.
    {"four",                     4, false, "just above current max",                "silently ignored, no error returned"},
    {"eight",                    8, false, "hypothetical multi-view",               "silently ignored, no error returned"},
    {"sixteen",                 16, false, "generous headroom",                     "silently ignored, no error returned"},
    {"max_uint8",              255, false, "uint8 boundary",                        "silently ignored, no error returned"},
    {"max_uint32",      UINT32_MAX, false, "overflow guard",                        "silently ignored, no error returned"},
};
// clang-format on

// =========================================================================
// PART 1: Test PopulateLayerPointers() — bounded loop with validation
// =========================================================================

class BoundedLoopDispatchTest : public ::testing::TestWithParam<LayerTestCase> {};

TEST_P(BoundedLoopDispatchTest, AllScenarios) {
    const auto &tc = GetParam();
    auto info = makeTestLayerInfo(tc.num_layers);

    void *dev_mem_ptr[kMaxVideoLayers];
    uint32_t horizontal_pitch[kMaxVideoLayers];
    initOutputs(dev_mem_ptr, horizontal_pitch);

    int status = PopulateLayerPointers(info, dev_mem_ptr, horizontal_pitch);

    if (tc.expect_success) {
        // --- Valid case: all num_layers slots must be correctly set ---
        ASSERT_EQ(status, kVideoLayerSuccess)
            << "num_layers=" << tc.num_layers << " (" << tc.description << ")";

        // Layer 0 — always base address
        EXPECT_EQ(dev_mem_ptr[0], info.base_ptr);
        EXPECT_EQ(horizontal_pitch[0], info.pitch[0]);

        // Layers 1..num_layers-1 — base + offset[i]
        for (uint32_t i = 1; i < tc.num_layers; i++) {
            EXPECT_EQ(dev_mem_ptr[i], info.base_ptr + info.offset[i])
                << "Layer " << i << " pointer";
            EXPECT_EQ(horizontal_pitch[i], info.pitch[i])
                << "Layer " << i << " pitch";
        }

        // Slots beyond num_layers — untouched at sentinel
        for (uint32_t i = tc.num_layers; i < kMaxVideoLayers; i++) {
            EXPECT_EQ(dev_mem_ptr[i], reinterpret_cast<void*>(kSentinelPtr))
                << "Layer " << i << " should be untouched";
            EXPECT_EQ(horizontal_pitch[i], kSentinelPitch)
                << "Pitch " << i << " should be untouched";
        }
    } else {
        // --- Invalid case: must return error, all outputs untouched ---
        EXPECT_EQ(status, kVideoLayerInvalidParam)
            << "num_layers=" << tc.num_layers << " (" << tc.description
            << ") must be rejected";

        for (uint32_t i = 0; i < kMaxVideoLayers; i++) {
            EXPECT_EQ(dev_mem_ptr[i], reinterpret_cast<void*>(kSentinelPtr))
                << "Layer " << i << " pointer modified on error path";
            EXPECT_EQ(horizontal_pitch[i], kSentinelPitch)
                << "Pitch " << i << " modified on error path";
        }
    }
}

INSTANTIATE_TEST_SUITE_P(
    LayerDispatch,
    BoundedLoopDispatchTest,
    ::testing::ValuesIn(kAllTestCases),
    [](const ::testing::TestParamInfo<LayerTestCase> &info) {
        return info.param.name;
    }
);

// =========================================================================
// PART 2: Test OriginalPopulateLayerPointers() — if/else-if chain
//         (original code from commit 18f165e424, preserved for comparison)
//
// These tests DEMONSTRATE the bugs.  They use EXPECT (not ASSERT) so all
// failures are reported in a single run.  The original code has no return
// value, so we can only check output state.
// =========================================================================

class IfElseChainDispatchTest : public ::testing::TestWithParam<LayerTestCase> {};

TEST_P(IfElseChainDispatchTest, AllScenarios) {
    const auto &tc = GetParam();
    auto info = makeTestLayerInfo(tc.num_layers);

    void *dev_mem_ptr[kMaxVideoLayers];
    uint32_t horizontal_pitch[kMaxVideoLayers];
    initOutputs(dev_mem_ptr, horizontal_pitch);

    OriginalPopulateLayerPointers(info, dev_mem_ptr, horizontal_pitch);

    if (tc.expect_success) {
        // --- Valid case: check what the original code actually does ---

        // Layer 0 — original code always sets this (correct)
        EXPECT_EQ(dev_mem_ptr[0], info.base_ptr)
            << "Layer 0 should be set for num_layers=" << tc.num_layers;

        // Layers 1..num_layers-1
        for (uint32_t i = 1; i < tc.num_layers; i++) {
            if (tc.original_bug[0] != '\0') {
                // This case has a known bug in the original code.
                // We assert the bug EXISTS (sentinel = never written) so this
                // test will FAIL if someone fixes the original code without
                // updating this test — keeping the bug documentation honest.
                //
                // For num_layers==3: layer 1 is the missing one.
                if (tc.num_layers == 3 && i == 1) {
                    EXPECT_EQ(dev_mem_ptr[i], reinterpret_cast<void*>(kSentinelPtr))
                        << "BUG CONFIRMED (original code): layer " << i
                        << " never set for " << tc.name
                        << ". " << tc.original_bug;
                }
            } else {
                // No known bug — original code should handle this correctly
                EXPECT_EQ(dev_mem_ptr[i], info.base_ptr + info.offset[i])
                    << "Layer " << i << " pointer for " << tc.name;
                EXPECT_EQ(horizontal_pitch[i], info.pitch[i])
                    << "Layer " << i << " pitch for " << tc.name;
            }
        }
    } else {
        // --- Invalid case: original code has no error return ---
        // It silently writes layer 0 regardless.  We confirm this behavior
        // to document it, not to endorse it.
        if (tc.num_layers != UINT32_MAX) {
            // (skip UINT32_MAX — original code still writes layer 0)
            EXPECT_NE(dev_mem_ptr[0], reinterpret_cast<void*>(kSentinelPtr))
                << "BUG CONFIRMED (original code): layer 0 written even for "
                << "invalid num_layers=" << tc.num_layers
                << ", no error returned. " << tc.original_bug;
        }
    }
}

INSTANTIATE_TEST_SUITE_P(
    LayerDispatch,
    IfElseChainDispatchTest,
    ::testing::ValuesIn(kAllTestCases),
    [](const ::testing::TestParamInfo<LayerTestCase> &info) {
        return info.param.name;
    }
);

// =========================================================================
// Edge cases (not part of the main table)
// =========================================================================

TEST(BoundedLoopDispatch_Edge, NullBasePointerArithmeticIsCorrect) {
    auto info = makeTestLayerInfo(2, nullptr);
    void *dev_mem_ptr[kMaxVideoLayers] = {};
    uint32_t horizontal_pitch[kMaxVideoLayers] = {};

    int status = PopulateLayerPointers(info, dev_mem_ptr, horizontal_pitch);
    ASSERT_EQ(status, kVideoLayerSuccess);

    EXPECT_EQ(dev_mem_ptr[0], nullptr);
    EXPECT_EQ(dev_mem_ptr[1],
              reinterpret_cast<void*>(static_cast<uintptr_t>(info.offset[1])));
}
