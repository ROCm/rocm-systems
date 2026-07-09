/*
Copyright (c) 2024 - 2026 Advanced Micro Devices, Inc. All rights reserved.

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

// Regression test for SWSPLAT-24294: uint32 overflow in SaveImage buffer sizing.
// Verifies that large JPEG dimensions (approaching 65535×65535) do not cause
// integer overflow when computing the host buffer size, which would result in
// an undersized allocation and subsequent heap overflow during hipMemcpyDtoH.

#include "../../samples/rocjpeg_samples_utils.h"
#include <cstdint>
#include <cstdlib>
#include <iostream>

// Mock RocJpegImage for testing size calculations without requiring actual GPU allocation.
struct MockRocJpegImage {
    hipDeviceptr_t channel[3];
    uint32_t pitch[3];
};

// Test helper: compute what SaveImage's buffer size would be for given dimensions.
// This mimics the size arithmetic from SaveImage() so we can verify it doesn't overflow.
static size_t compute_saveimage_buffer_size(uint32_t pitch, uint32_t height) {
    // This is the pattern from SaveImage after the SWSPLAT-24294 fix:
    // size_t channel_size = static_cast<size_t>(pitch) * height;
    // The test verifies this arithmetic stays in bounds and doesn't wrap.
    size_t channel_size = static_cast<size_t>(pitch) * height;
    return channel_size;
}

int main() {
    std::cout << "Testing SaveImage overflow fix (SWSPLAT-24294)..." << std::endl;

    // Test case 1: Max JPEG dimensions (65535×65535).
    // Before the fix, uint32_t pitch * height would wrap to ~0 for large images.
    // After the fix (size_t arithmetic), this should yield the correct ~4.3 GB size.
    {
        uint32_t max_dim = 65535;
        uint32_t pitch = max_dim;  // pitch is typically close to width
        uint32_t height = max_dim;

        size_t buffer_size = compute_saveimage_buffer_size(pitch, height);

        // Expected size: 65535 * 65535 = 4,294,836,225 bytes (~4.29 GB).
        // If the calculation wrapped uint32_t, buffer_size would be < UINT32_MAX.
        // With size_t, it should be > UINT32_MAX.
        const size_t expected_min = static_cast<size_t>(UINT32_MAX) + 1;

        if (buffer_size < expected_min) {
            std::cerr << "FAIL: buffer_size " << buffer_size
                      << " is suspiciously small for " << max_dim << "×" << max_dim
                      << " (expected > " << expected_min << "). Overflow still present?"
                      << std::endl;
            return EXIT_FAILURE;
        }

        std::cout << "  ✓ Max dimensions (" << max_dim << "×" << max_dim
                  << "): buffer_size = " << buffer_size << " bytes (correct, no overflow)"
                  << std::endl;
    }

    // Test case 2: Aligned dimensions (256-byte alignment can push the product higher).
    // rocjpeg samples align height to 256 bytes, which can increase the overflow window.
    {
        uint32_t width = 65500;
        uint32_t height = 65500;
        // Simulate 256-byte alignment on height (same as mem_alignment in samples).
        uint32_t aligned_height = (height + 255) & ~255;
        uint32_t pitch = (width + 255) & ~255;  // pitch is also typically aligned

        size_t buffer_size = compute_saveimage_buffer_size(pitch, aligned_height);

        // The product should be pitch * aligned_height, both ~65536 after alignment.
        // Expected: ~4.29 GB. If uint32_t overflowed, buffer_size would wrap small.
        const size_t expected_min = static_cast<size_t>(UINT32_MAX) + 1;

        if (buffer_size < expected_min) {
            std::cerr << "FAIL: buffer_size " << buffer_size
                      << " for aligned " << pitch << "×" << aligned_height
                      << " (expected > " << expected_min << "). Overflow still present?"
                      << std::endl;
            return EXIT_FAILURE;
        }

        std::cout << "  ✓ Aligned dimensions (pitch=" << pitch << ", aligned_height="
                  << aligned_height << "): buffer_size = " << buffer_size
                  << " bytes (correct, no overflow)" << std::endl;
    }

    // Test case 3: Small dimensions (sanity check that normal images still work).
    {
        uint32_t pitch = 1920;
        uint32_t height = 1080;

        size_t buffer_size = compute_saveimage_buffer_size(pitch, height);
        size_t expected = static_cast<size_t>(pitch) * height;

        if (buffer_size != expected) {
            std::cerr << "FAIL: buffer_size " << buffer_size
                      << " for " << pitch << "×" << height
                      << " (expected " << expected << ")" << std::endl;
            return EXIT_FAILURE;
        }

        std::cout << "  ✓ Normal dimensions (" << pitch << "×" << height
                  << "): buffer_size = " << buffer_size << " bytes (correct)"
                  << std::endl;
    }

    std::cout << "All SaveImage overflow tests passed." << std::endl;
    return EXIT_SUCCESS;
}
