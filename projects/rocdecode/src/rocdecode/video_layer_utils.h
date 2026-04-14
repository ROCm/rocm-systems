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

#pragma once

#include <stdint.h>

// Maximum number of planes in a video surface (Y, U/UV, V).
static constexpr uint32_t kMaxVideoLayers = 3;

// Return codes — mirrors rocDecStatus values so this header is self-contained
// (no dependency on HIP or the full rocdecode API header).
static constexpr int kVideoLayerSuccess = 0;           // ROCDEC_SUCCESS
static constexpr int kVideoLayerInvalidParam = -5;     // ROCDEC_INVALID_PARAMETER

// Minimal surface description for layer pointer dispatch.
struct VideoLayerInfo {
    uint8_t* base_ptr;                   // Mapped device memory base pointer
    uint32_t offset[kMaxVideoLayers];    // Offset of each plane from base_ptr
    uint32_t pitch[kMaxVideoLayers];     // Pitch (row stride) of each plane
    uint32_t num_layers;                 // Number of layers making up the surface
};

// Populate output device memory pointers and pitches from a VideoLayerInfo.
// Returns kVideoLayerSuccess on success, kVideoLayerInvalidParam if num_layers
// is 0 or exceeds kMaxVideoLayers.
//
// Extracted from RocDecoder::GetVideoFrame() for testability.
inline int PopulateLayerPointers(const VideoLayerInfo &layer_info,
                                 void *dev_mem_ptr[kMaxVideoLayers],
                                 uint32_t horizontal_pitch[kMaxVideoLayers]) {
    if (layer_info.num_layers == 0 || layer_info.num_layers > kMaxVideoLayers) {
        return kVideoLayerInvalidParam;
    }
    // Layer 0 (Y plane) — always present
    dev_mem_ptr[0] = layer_info.base_ptr;
    horizontal_pitch[0] = layer_info.pitch[0];
    // Remaining layers (U/UV, V) — set for each layer present
    for (uint32_t i = 1; i < layer_info.num_layers; i++) {
        dev_mem_ptr[i] = layer_info.base_ptr + layer_info.offset[i];
        horizontal_pitch[i] = layer_info.pitch[i];
    }
    return kVideoLayerSuccess;
}

// -------------------------------------------------------------------------
// ORIGINAL CODE — preserved for testing only.  Can be removed once the fix
// is accepted and the regression tests are established.
//
// This is a verbatim copy of the layer dispatch logic from
// RocDecoder::GetVideoFrame() (roc_decoder.cpp lines 177-185, commit
// 18f165e424, 2024-02-05).  It has the following bugs:
//
//   1. num_layers==3: the else-if means layer 1 (U plane) is NEVER set.
//      The caller receives an uninitialized pointer for the U plane.
//
//   2. num_layers==0 or >3: no error is returned.  Layer 0 is
//      unconditionally written and remaining slots are left uninitialized.
//
// The unit tests run BOTH this function and PopulateLayerPointers() against
// the same table-driven inputs, so the behavioral difference is immediately
// visible in the test output.
// -------------------------------------------------------------------------
inline void OriginalPopulateLayerPointers(const VideoLayerInfo &layer_info,
                                          void *dev_mem_ptr[kMaxVideoLayers],
                                          uint32_t horizontal_pitch[kMaxVideoLayers]) {
    // Line 177-178: always set layer 0
    dev_mem_ptr[0] = layer_info.base_ptr;
    horizontal_pitch[0] = layer_info.pitch[0];

    // Line 179-185: the if/else-if chain
    if (layer_info.num_layers == 2) {
        dev_mem_ptr[1] = layer_info.base_ptr + layer_info.offset[1];
        horizontal_pitch[1] = layer_info.pitch[1];
    } else if (layer_info.num_layers == 3) {
        // BUG: enters this branch instead of the num_layers==2 branch,
        //      so layer 1 (U plane) is NEVER set.
        dev_mem_ptr[2] = layer_info.base_ptr + layer_info.offset[2];
        horizontal_pitch[2] = layer_info.pitch[2];
    }
    // BUG: num_layers==0 or >3 — no error, caller gets partial output.
}
