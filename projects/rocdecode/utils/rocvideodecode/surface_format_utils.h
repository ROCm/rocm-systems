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

// The production build pulls the chroma/surface enums (and HIP) from rocdecode.h.
// Unit tests define ROCDECODE_SURFACE_FORMAT_UTILS_STANDALONE to compile this
// header host-only, using the fallback enum definitions below (kept in sync with
// api/rocdecode/rocdecode.h) so the *same* SelectSurfaceFormat used in production
// can be exercised without the HIP toolchain or a GPU.
#ifdef ROCDECODE_SURFACE_FORMAT_UTILS_STANDALONE
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
#else
#include "rocdecode/rocdecode.h"
#endif

// Select the output surface format for a chroma format + bit depth. Monochrome
// selects the same format as 4:2:0 (NV12/P016). Unrecognized chroma formats
// return rocDecVideoSurfaceFormat_Native (the "decoder chooses" sentinel).
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
