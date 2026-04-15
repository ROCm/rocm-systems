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

// Forward-declare the enum types from rocdecode.h to avoid pulling in HIP.
// Values must stay in sync with rocdecode/rocdecode.h.
#ifndef ROCDECODE_H
typedef enum rocDecVideoChromaFormat_enum {
    rocDecVideoChromaFormat_Monochrome = 0,
    rocDecVideoChromaFormat_420,
    rocDecVideoChromaFormat_422,
    rocDecVideoChromaFormat_444
} rocDecVideoChromaFormat;

typedef enum rocDecVideoSurfaceFormat_enum {
    rocDecVideoSurfaceFormat_NV12 = 0,
    rocDecVideoSurfaceFormat_P016 = 1,
    rocDecVideoSurfaceFormat_YUV444 = 2,
    rocDecVideoSurfaceFormat_YUV444_16Bit = 3,
    rocDecVideoSurfaceFormat_YUV420 = 4,
    rocDecVideoSurfaceFormat_YUV420_16Bit = 5,
    rocDecVideoSurfaceFormat_YUV422 = 6,
    rocDecVideoSurfaceFormat_YUV422_16Bit = 7,
    rocDecVideoSurfaceFormat_Native = 8
} rocDecVideoSurfaceFormat;
#endif

/**
 * @brief Select the appropriate output surface format for a given chroma format and bit depth.
 *
 * This is the pure-function extraction of the surface format selection logic from
 * RocVideoDecoder::HandleVideoSequence() and ReconfigureDecoder().
 *
 * @param chroma_format The video chroma subsampling format.
 * @param bitdepth_minus_8 The bit depth minus 8 (0 for 8-bit, >0 for higher).
 * @return The corresponding rocDecVideoSurfaceFormat, or rocDecVideoSurfaceFormat_Native
 *         if the chroma format is not recognized (callers should treat this as an error).
 */
inline rocDecVideoSurfaceFormat SelectSurfaceFormat(rocDecVideoChromaFormat chroma_format, uint8_t bitdepth_minus_8) {
    switch (chroma_format) {
    case rocDecVideoChromaFormat_420:
    case rocDecVideoChromaFormat_Monochrome:
        return bitdepth_minus_8 ? rocDecVideoSurfaceFormat_P016
                                : rocDecVideoSurfaceFormat_NV12;
    case rocDecVideoChromaFormat_444:
        return bitdepth_minus_8 ? rocDecVideoSurfaceFormat_YUV444_16Bit
                                : rocDecVideoSurfaceFormat_YUV444;
    case rocDecVideoChromaFormat_422:
        return bitdepth_minus_8 ? rocDecVideoSurfaceFormat_YUV422_16Bit
                                : rocDecVideoSurfaceFormat_YUV422;
    default:
        return rocDecVideoSurfaceFormat_Native;  // unrecognized chroma format
    }
}
