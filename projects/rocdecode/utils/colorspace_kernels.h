
/*
Copyright (c) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.

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
#include <hip/hip_runtime.h>

/*!
 * \file
 * \brief The AMD Color Space Standards for VCN Decode Library.
 *
 * \defgroup group_amd_vcn_colorspace colorSpace: AMD VCN Color Space API
 * \brief AMD The vcnDECODE Color Space API.
 */

typedef enum ColorSpaceStandard_ {
    ColorSpaceStandard_BT709 = 1,
    ColorSpaceStandard_Unspecified = 2,
    ColorSpaceStandard_Reserved = 3,
    ColorSpaceStandard_FCC = 4,
    ColorSpaceStandard_BT470 = 5,
    ColorSpaceStandard_BT601 = 6,
    ColorSpaceStandard_SMPTE240M = 7,
    ColorSpaceStandard_YCgCo = 8,
    ColorSpaceStandard_BT2020 = 9,
    ColorSpaceStandard_BT2020C = 10
} ColorSpaceStandard;

union BGR24 {
    uchar3 v;
    struct {
        uint8_t b, g, r;
    } c;
};

union RGB24 {
    uchar3 v;
    struct {
        uint8_t r, g, b;
    } c;
};

union BGR48 {
    ushort3 v;
    struct {
        uint16_t b, g, r;
    } c;
};

union RGB48 {
    ushort3 v;
    struct {
        uint16_t r, g, b;
    } c;
};

union BGRA32 {
    uint32_t d;
    uchar4 v;
    struct {
        uint8_t b, g, r, a;
    } c;
};

union RGBA32 {
    uint32_t d;
    uchar4 v;
    struct {
        uint8_t r, g, b, a;
    } c;
};

union BGRA64 {
    uint64_t d;
    ushort4 v;
    struct {
        uint16_t b, g, r, a;
    } c;
};

union RGBA64 {
    uint64_t d;
    ushort4 v;
    struct {
        uint16_t r, g, b, a;
    } c;
};

// color-convert hip kernel function definitions
template <class COLOR32>
void YUV444ToColor32(uint8_t *dp_yuv_444, int pitch, uint8_t *dp_bgra, int bgra_pitch, int width, int height, int v_pitch, int col_standard, hipStream_t hip_stream);
template <class COLOR64>
void YUV444ToColor64(uint8_t *dp_yuv_444, int pitch, uint8_t *dp_bgra, int bgra_pitch, int width, int height, int v_pitch, int col_standard, hipStream_t hip_stream);
template <class COLOR24>
void YUV444ToColor24(uint8_t *dp_yuv_444, int pitch, uint8_t *dp_bgr, int bgr_pitch, int width, int height, int v_pitch, int col_standard, hipStream_t hip_stream);
template <class COLOR48>
void YUV444ToColor48(uint8_t *dp_yuv_444, int pitch, uint8_t *dp_bgr, int bgr_pitch, int width, int height, int v_pitch, int col_standard, hipStream_t hip_stream);

template <class COLOR24>
void Nv12ToColor24(uint8_t *dp_nv12, int nv12_pitch, uint8_t *dp_bgr, int bgr_pitch, int width, int height, int v_pitch, int col_standard, hipStream_t hip_stream);
template <class COLOR32>
void Nv12ToColor32(uint8_t *dp_nv12, int nv12_pitch, uint8_t *dp_bgra, int bgra_pitch, int width, int height, int v_pitch, int col_standard, hipStream_t hip_stream);
template <class COLOR48>
void Nv12ToColor48(uint8_t *dp_nv12, int nv12_pitch, uint8_t *dp_bgr, int bgr_pitch, int width, int height, int v_pitch, int col_standard, hipStream_t hip_stream);
template <class COLOR64>
void Nv12ToColor64(uint8_t *dp_nv12, int nv12_pitch, uint8_t *dp_bgra, int bgra_pitch, int width, int height, int v_pitch, int col_standard, hipStream_t hip_stream);
template <class COLOR24>
void YUV444P16ToColor24(uint8_t *dp_yuv_444, int pitch, uint8_t *dp_bgra, int bgra_pitch, int width, int height, int v_pitch, int col_standard, hipStream_t hip_stream);
template <class COLOR48>
void YUV444P16ToColor48(uint8_t *dp_yuv_444, int pitch, uint8_t *dp_bgr, int bgr_pitch, int width, int height, int v_pitch, int col_standard, hipStream_t hip_stream);
template <class COLOR32>
void YUV444P16ToColor32(uint8_t *dp_yuv_444, int pitch, uint8_t *dp_bgra, int bgra_pitch, int width, int height, int v_pitch, int col_standard, hipStream_t hip_stream);
template <class COLOR64>
void YUV444P16ToColor64(uint8_t *dp_yuv_444, int pitch, uint8_t *dp_bgra, int bgra_pitch, int width, int height, int v_pitch, int col_standard, hipStream_t hip_stream);
template <class COLOR32>
void P016ToColor32(uint8_t *dp_p016, int p016_pitch, uint8_t *dp_bgra, int bgra_pitch, int width, int height, int v_pitch, int col_standard, hipStream_t hip_stream);
template <class COLOR64>
void P016ToColor64(uint8_t *dp_p016, int p016_pitch, uint8_t *dp_bgra, int bgra_pitch, int width, int height, int v_pitch, int col_standard, hipStream_t hip_stream);
template <class COLOR24>
void P016ToColor24(uint8_t *dp_p016, int p016_pitch, uint8_t *dp_bgr, int bgr_pitch, int width, int height, int v_pitch, int col_standard, hipStream_t hip_stream);
template <class COLOR48>
void P016ToColor48(uint8_t *dp_p016, int p016_pitch, uint8_t *dp_bgr, int bgr_pitch, int width, int height, int v_pitch, int col_standard, hipStream_t hip_stream);

