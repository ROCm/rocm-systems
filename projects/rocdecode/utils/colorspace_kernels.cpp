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

#include "colorspace_kernels.h"
#include "roc_video_dec.h"
 
__constant__ float yuv_to_rgb_mat[3][3];
__constant__ float rgb_to_yuv_mat[3][3];


void inline GetColMatCoefficients(int col_standard, float &wr, float &wb, int &black, int &white, int &max) {
    black = 16; white = 235;
    max = 255;

    switch (col_standard)
    {
    case ColorSpaceStandard_BT709:
    default:
        wr = 0.2126f; wb = 0.0722f;
        break;

    case ColorSpaceStandard_FCC:
        wr = 0.30f; wb = 0.11f;
        break;

    case ColorSpaceStandard_BT470:
    case ColorSpaceStandard_BT601:
        wr = 0.2990f; wb = 0.1140f;
        break;

    case ColorSpaceStandard_SMPTE240M:
        wr = 0.212f; wb = 0.087f;
        break;

    case ColorSpaceStandard_BT2020:
    case ColorSpaceStandard_BT2020C:
        wr = 0.2627f; wb = 0.0593f;
        // 10-bit only
        black = 64 << 6; white = 940 << 6;
        max = (1 << 16) - 1;
        break;
    }
}

void SetMatYuv2Rgb(int col_standard) {
    float wr, wb;
    int black, white, max;
    GetColMatCoefficients(col_standard, wr, wb, black, white, max);
    float mat[3][3] = {
        1.0f, 0.0f, (1.0f - wr) / 0.5f,
        1.0f, -wb * (1.0f - wb) / 0.5f / (1 - wb - wr), -wr * (1 - wr) / 0.5f / (1 - wb - wr),
        1.0f, (1.0f - wb) / 0.5f, 0.0f,
    };
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            mat[i][j] = (float)(1.0 * max / (white - black) * mat[i][j]);
        }
    }
    HIP_API_CALL(hipMemcpyToSymbol(yuv_to_rgb_mat, mat, sizeof(mat)));
}

void SetMatRgb2Yuv(int col_standard) {
    float wr, wb;
    int black, white, max;
    GetColMatCoefficients(col_standard, wr, wb, black, white, max);
    float mat[3][3] = {
        wr, 1.0f - wb - wr, wb,
        -0.5f * wr / (1.0f - wb), -0.5f * (1 - wb - wr) / (1.0f - wb), 0.5f,
        0.5f, -0.5f * (1.0f - wb - wr) / (1.0f - wr), -0.5f * wb / (1.0f - wr),
    };
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            mat[i][j] = (float)(1.0 * (white - black) / max * mat[i][j]);
        }
    }
    HIP_API_CALL(hipMemcpyToSymbol(rgb_to_yuv_mat, mat, sizeof(mat)));
}

template<class T>
__device__ static T Clamp(T x, T lower, T upper) {
    return x < lower ? lower : (x > upper ? upper : x);
}

template<class Rgb, class YuvUnit>
__device__ inline Rgb YuvToRgbForPixel(YuvUnit y, YuvUnit u, YuvUnit v) {
    const int 
        low = 1 << (sizeof(YuvUnit) * 8 - 4),
        mid = 1 << (sizeof(YuvUnit) * 8 - 1);
    float fy = (int)y - low, fu = (int)u - mid, fv = (int)v - mid;
    const float maxf = (1 << sizeof(YuvUnit) * 8) - 1.0f;
    YuvUnit 
        r = (YuvUnit)Clamp(yuv_to_rgb_mat[0][0] * fy + yuv_to_rgb_mat[0][1] * fu + yuv_to_rgb_mat[0][2] * fv, 0.0f, maxf),
        g = (YuvUnit)Clamp(yuv_to_rgb_mat[1][0] * fy + yuv_to_rgb_mat[1][1] * fu + yuv_to_rgb_mat[1][2] * fv, 0.0f, maxf),
        b = (YuvUnit)Clamp(yuv_to_rgb_mat[2][0] * fy + yuv_to_rgb_mat[2][1] * fu + yuv_to_rgb_mat[2][2] * fv, 0.0f, maxf);
    
    Rgb rgb{};
    const int nShift = abs((int)sizeof(YuvUnit) - (int)sizeof(rgb.c.r)) * 8;
    if (sizeof(YuvUnit) >= sizeof(rgb.c.r)) {
        rgb.c.r = r >> nShift;
        rgb.c.g = g >> nShift;
        rgb.c.b = b >> nShift;
    } else {
        rgb.c.r = r << nShift;
        rgb.c.g = g << nShift;
        rgb.c.b = b << nShift;
    }
    return rgb;
}

// yuv to RGBA (32/64 bit)
template<class YuvUnitx2, class Rgb, class RgbIntx2>
__global__ static void YuvToRgbaKernel(uint8_t *dp_yuv, int yuv_pitch, uint8_t *dp_rgb, int rgb_pitch, int width, int height, int v_pitch) {
    int x = (threadIdx.x + blockIdx.x * blockDim.x) * 2;
    int y = (threadIdx.y + blockIdx.y * blockDim.y) * 2;
    if (x + 1 >= width || y + 1 >= height) {
        return;
    }

    uint8_t *p_src = dp_yuv + x * sizeof(YuvUnitx2) / 2 + y * yuv_pitch;
    uint8_t *p_dst = dp_rgb + x * sizeof(Rgb) + y * rgb_pitch;

    YuvUnitx2 l0 = *(YuvUnitx2 *)p_src;
    YuvUnitx2 l1 = *(YuvUnitx2 *)(p_src + yuv_pitch);
    YuvUnitx2 ch = *(YuvUnitx2 *)(p_src + (v_pitch - y / 2) * yuv_pitch);

    *(RgbIntx2 *)p_dst = RgbIntx2 {
        YuvToRgbForPixel<Rgb>(l0.x, ch.x, ch.y).d,
        YuvToRgbForPixel<Rgb>(l0.y, ch.x, ch.y).d,
    };
    *(RgbIntx2 *)(p_dst + rgb_pitch) = RgbIntx2 {
        YuvToRgbForPixel<Rgb>(l1.x, ch.x, ch.y).d, 
        YuvToRgbForPixel<Rgb>(l1.y, ch.x, ch.y).d,
    };
}

// yuv to RGB (24/48 bit)
template<class YuvUnitx2, class Rgb, class RgbInt1, class RgbInt2>
__global__ static void YuvToRgbKernel(uint8_t *dp_yuv, int yuv_pitch, uint8_t *dp_rgb, int rgb_pitch, int width, int height, int v_pitch) {
    int x = (threadIdx.x + blockIdx.x * blockDim.x) * 2;
    int y = (threadIdx.y + blockIdx.y * blockDim.y) * 2;
    if (x + 1 >= width || y + 1 >= height) {
        return;
    }

    uint8_t *p_src = dp_yuv + x * sizeof(YuvUnitx2) / 2 + y * yuv_pitch;
    uint8_t *p_dst = dp_rgb + x * sizeof(Rgb) + y * rgb_pitch;
    uint8_t *p_dst1 = p_dst + rgb_pitch;

    YuvUnitx2 l0 = *(YuvUnitx2 *)p_src;
    YuvUnitx2 l1 = *(YuvUnitx2 *)(p_src + yuv_pitch);
    YuvUnitx2 ch = *(YuvUnitx2 *)(p_src + (v_pitch - y / 2) * yuv_pitch);
    Rgb rgb0 = YuvToRgbForPixel<Rgb>(l0.x, ch.x, ch.y),
        rgb1 = YuvToRgbForPixel<Rgb>(l0.y, ch.x, ch.y),
        rgb2 = YuvToRgbForPixel<Rgb>(l1.x, ch.x, ch.y),
        rgb3 = YuvToRgbForPixel<Rgb>(l1.y, ch.x, ch.y);

    *(RgbInt1 *)p_dst = RgbInt1 { rgb0.v.x, rgb0.v.y, rgb0.v.z, rgb1.v.x };
    *(RgbInt2 *)(p_dst + sizeof(RgbInt1)) = RgbInt2 { rgb1.v.y, rgb1.v.z };
    *(RgbInt1 *)(p_dst1) = RgbInt1 { rgb2.v.x, rgb2.v.y, rgb2.v.z, rgb3.v.x };
    *(RgbInt2 *)(p_dst1 + sizeof(RgbInt1)) = RgbInt2 { rgb3.v.y, rgb3.v.z };
}

// yuv444 to RGBA (32/64 bit)
template<class YuvUnitx2, class Rgb, class RgbIntx2>
__global__ static void Yuv444ToRgbaKernel(uint8_t *dp_yuv, int yuv_pitch, uint8_t *dp_rgb, int rgb_pitch, int width, int height, int v_pitch) {
    int x = (threadIdx.x + blockIdx.x * blockDim.x) * 2;
    int y = (threadIdx.y + blockIdx.y * blockDim.y);
    if (x + 1 >= width || y  >= height) {
        return;
    }

    uint8_t *p_src = dp_yuv + x * sizeof(YuvUnitx2) / 2 + y * yuv_pitch;
    uint8_t *p_dst = dp_rgb + x * sizeof(Rgb) + y * rgb_pitch;

    YuvUnitx2 l0 = *(YuvUnitx2 *)p_src;
    YuvUnitx2 ch1 = *(YuvUnitx2 *)(p_src + (v_pitch * yuv_pitch));
    YuvUnitx2 ch2 = *(YuvUnitx2 *)(p_src + (2 * v_pitch * yuv_pitch));

    *(RgbIntx2 *)p_dst = RgbIntx2{
        YuvToRgbForPixel<Rgb>(l0.x, ch1.x, ch2.x).d,
        YuvToRgbForPixel<Rgb>(l0.y, ch1.y, ch2.y).d,
    };
}

// yuv444 to RGB (24/48 bit)
template<class YuvUnitx2, class Rgb, class RgbInt1, class RgbInt2>
__global__ static void Yuv444ToRgbKernel(uint8_t *dp_yuv, int yuv_pitch, uint8_t *dp_rgb, int rgb_pitch, int width, int height, int v_pitch) {
    int x = (threadIdx.x + blockIdx.x * blockDim.x) * 2;
    int y = (threadIdx.y + blockIdx.y * blockDim.y);
    if (x + 1 >= width || y  >= height) {
        return;
    }

    uint8_t *p_src = dp_yuv + x * sizeof(YuvUnitx2) / 2 + y * yuv_pitch;
    uint8_t *p_dst = dp_rgb + x * sizeof(Rgb) + y * rgb_pitch;

    YuvUnitx2 l0 = *(YuvUnitx2 *)p_src;
    YuvUnitx2 ch1 = *(YuvUnitx2 *)(p_src + (v_pitch * yuv_pitch));
    YuvUnitx2 ch2 = *(YuvUnitx2 *)(p_src + (2 * v_pitch * yuv_pitch));
    Rgb rgb0 = YuvToRgbForPixel<Rgb>(l0.x, ch1.x, ch2.x),
        rgb1 = YuvToRgbForPixel<Rgb>(l0.y, ch1.y, ch2.y);

    *(RgbInt1 *)p_dst = RgbInt1 { rgb0.v.x, rgb0.v.y, rgb0.v.z, rgb1.v.x };
    *(RgbInt2 *)(p_dst + sizeof(RgbInt1)) = RgbInt2 { rgb1.v.y, rgb1.v.z };
}


template<class YuvUnitx2, class Rgb, class RgbUnitx2>
__global__ static void YuvToRgbPlanarKernel(uint8_t *dp_yuv, int yuv_pitch, uint8_t *dp_rgbp, int nRgbpPitch, int width, int height, int v_pitch) {
    int x = (threadIdx.x + blockIdx.x * blockDim.x) * 2;
    int y = (threadIdx.y + blockIdx.y * blockDim.y) * 2;
    if (x + 1 >= width || y + 1 >= height) {
        return;
    }

    uint8_t *p_src = dp_yuv + x * sizeof(YuvUnitx2) / 2 + y * yuv_pitch;

    YuvUnitx2 l0 = *(YuvUnitx2 *)p_src;
    YuvUnitx2 l1 = *(YuvUnitx2 *)(p_src + yuv_pitch);
    YuvUnitx2 ch = *(YuvUnitx2 *)(p_src + (v_pitch - y / 2) * yuv_pitch);

    Rgb rgb0 = YuvToRgbForPixel<Rgb>(l0.x, ch.x, ch.y),
        rgb1 = YuvToRgbForPixel<Rgb>(l0.y, ch.x, ch.y),
        rgb2 = YuvToRgbForPixel<Rgb>(l1.x, ch.x, ch.y),
        rgb3 = YuvToRgbForPixel<Rgb>(l1.y, ch.x, ch.y);

    uint8_t *p_dst = dp_rgbp + x * sizeof(RgbUnitx2) / 2 + y * nRgbpPitch;
    *(RgbUnitx2 *)p_dst = RgbUnitx2 {rgb0.v.x, rgb1.v.x};
    *(RgbUnitx2 *)(p_dst + nRgbpPitch) = RgbUnitx2 {rgb2.v.x, rgb3.v.x};
    p_dst += nRgbpPitch * height;
    *(RgbUnitx2 *)p_dst = RgbUnitx2 {rgb0.v.y, rgb1.v.y};
    *(RgbUnitx2 *)(p_dst + nRgbpPitch) = RgbUnitx2 {rgb2.v.y, rgb3.v.y};
    p_dst += nRgbpPitch * height;
    *(RgbUnitx2 *)p_dst = RgbUnitx2 {rgb0.v.z, rgb1.v.z};
    *(RgbUnitx2 *)(p_dst + nRgbpPitch) = RgbUnitx2 {rgb2.v.z, rgb3.v.z};
}

template<class YuvUnitx2, class Rgb, class RgbUnitx2>
__global__ static void Yuv444ToRgbPlanarKernel(uint8_t *dp_yuv, int yuv_pitch, uint8_t *dp_rgbp, int nRgbpPitch, int width, int height, int v_pitch) {
    int x = (threadIdx.x + blockIdx.x * blockDim.x) * 2;
    int y = (threadIdx.y + blockIdx.y * blockDim.y);
    if (x + 1 >= width || y >= height) {
        return;
    }

    uint8_t *p_src = dp_yuv + x * sizeof(YuvUnitx2) / 2 + y * yuv_pitch;

    YuvUnitx2 l0 = *(YuvUnitx2 *)p_src;
    YuvUnitx2 ch1 = *(YuvUnitx2 *)(p_src + (v_pitch * yuv_pitch));
    YuvUnitx2 ch2 = *(YuvUnitx2 *)(p_src + (2 * v_pitch * yuv_pitch));

    Rgb rgb0 = YuvToRgbForPixel<Rgb>(l0.x, ch1.x, ch2.x),
        rgb1 = YuvToRgbForPixel<Rgb>(l0.y, ch1.y, ch2.y);


    uint8_t *p_dst = dp_rgbp + x * sizeof(RgbUnitx2) / 2 + y * nRgbpPitch;
    *(RgbUnitx2 *)p_dst = RgbUnitx2{ rgb0.v.x, rgb1.v.x };

    p_dst += nRgbpPitch * height;
    *(RgbUnitx2 *)p_dst = RgbUnitx2{ rgb0.v.y, rgb1.v.y };

    p_dst += nRgbpPitch * height;
    *(RgbUnitx2 *)p_dst = RgbUnitx2{ rgb0.v.z, rgb1.v.z };
}

template <class COLOR32>
void Nv12ToColor32(uint8_t *dp_nv12, int nv12_pitch, uint8_t *dp_bgra, int bgra_pitch, int width, int height, int v_pitch, int col_standard, hipStream_t hip_stream) {
    SetMatYuv2Rgb(col_standard);
    YuvToRgbaKernel<uchar2, COLOR32, uint2>
        <<<dim3((width + 63) / 32 / 2, (height + 3) / 2 / 2), dim3(32, 2), 0, hip_stream>>>
        (dp_nv12, nv12_pitch, dp_bgra, bgra_pitch, width, height, v_pitch);
}

template <class COLOR64>
void Nv12ToColor64(uint8_t *dp_nv12, int nv12_pitch, uint8_t *dp_bgra, int bgra_pitch, int width, int height, int v_pitch, int col_standard, hipStream_t hip_stream) {
    SetMatYuv2Rgb(col_standard);
    YuvToRgbaKernel<uchar2, COLOR64, ulonglong2>
        <<<dim3((width + 63) / 32 / 2, (height + 3) / 2 / 2), dim3(32, 2), 0, hip_stream>>>
        (dp_nv12, nv12_pitch, dp_bgra, bgra_pitch, width, height, v_pitch);
}

template <class COLOR32>
void YUV444ToColor32(uint8_t *dp_yuv_444, int pitch, uint8_t *dp_bgra, int bgra_pitch, int width, int height, int v_pitch, int col_standard, hipStream_t hip_stream) {
    SetMatYuv2Rgb(col_standard);
    Yuv444ToRgbaKernel<uchar2, COLOR32, uint2>
        <<<dim3((width + 63) / 32 / 2, (height + 3) / 2), dim3(32, 2), 0, hip_stream>>>
        (dp_yuv_444, pitch, dp_bgra, bgra_pitch, width, height, v_pitch);
}

template <class COLOR64>
void YUV444ToColor64(uint8_t *dp_yuv_444, int pitch, uint8_t *dp_bgra, int bgra_pitch, int width, int height, int v_pitch, int col_standard, hipStream_t hip_stream) {
    SetMatYuv2Rgb(col_standard);
    Yuv444ToRgbaKernel<uchar2, COLOR64, ulonglong2>
        <<<dim3((width + 63) / 32 / 2, (height + 3) / 2), dim3(32, 2), 0, hip_stream>>>
        (dp_yuv_444, pitch, dp_bgra, bgra_pitch, width, height, v_pitch);
}

template <class COLOR32>
void P016ToColor32(uint8_t *dp_p016, int p016_pitch, uint8_t *dp_bgra, int bgra_pitch, int width, int height, int v_pitch, int col_standard, hipStream_t hip_stream) {
    SetMatYuv2Rgb(col_standard);
    YuvToRgbaKernel<ushort2, COLOR32, uint2>
        <<<dim3((width + 63) / 32 / 2, (height + 3) / 2 / 2), dim3(32, 2), 0, hip_stream>>>
        (dp_p016, p016_pitch, dp_bgra, bgra_pitch, width, height, v_pitch);
}

template <class COLOR64>
void P016ToColor64(uint8_t *dp_p016, int p016_pitch, uint8_t *dp_bgra, int bgra_pitch, int width, int height, int v_pitch, int col_standard, hipStream_t hip_stream) {
    SetMatYuv2Rgb(col_standard);
    YuvToRgbaKernel<ushort2, COLOR64, ulonglong2>
        <<<dim3((width + 63) / 32 / 2, (height + 3) / 2 / 2), dim3(32, 2), 0, hip_stream>>>
        (dp_p016, p016_pitch, dp_bgra, bgra_pitch, width, height, v_pitch);
}

template <class COLOR32>
void YUV444P16ToColor32(uint8_t *dp_yuv_444, int pitch, uint8_t *dp_bgra, int bgra_pitch, int width, int height, int v_pitch, int col_standard, hipStream_t hip_stream) {
    SetMatYuv2Rgb(col_standard);
    Yuv444ToRgbaKernel<ushort2, COLOR32, uint2>
        <<<dim3((width + 63) / 32 / 2, (height + 3) / 2), dim3(32, 2), 0, hip_stream>>>
        (dp_yuv_444, pitch, dp_bgra, bgra_pitch, width, height, v_pitch);
}

template <class COLOR64>
void YUV444P16ToColor64(uint8_t *dp_yuv_444, int pitch, uint8_t *dp_bgra, int bgra_pitch, int width, int height, int v_pitch, int col_standard, hipStream_t hip_stream) {
    SetMatYuv2Rgb(col_standard);
    Yuv444ToRgbaKernel<ushort2, COLOR64, ulonglong2>
        <<<dim3((width + 63) / 32 / 2, (height + 3) / 2), dim3(32, 2), 0, hip_stream>>>
        (dp_yuv_444, pitch, dp_bgra, bgra_pitch, width, height, v_pitch);
}

template <class COLOR32>
void Nv12ToColorPlanar(uint8_t *dp_nv12, int nv12_pitch, uint8_t *dp_bgrp, int nBgrpPitch, int width, int height, int v_pitch, int col_standard, hipStream_t hip_stream) {
    SetMatYuv2Rgb(col_standard);
    YuvToRgbPlanarKernel<uchar2, COLOR32, uchar2>
        <<<dim3((width + 63) / 32 / 2, (height + 3) / 2 / 2), dim3(32, 2), 0, hip_stream>>>
        (dp_nv12, nv12_pitch, dp_bgrp, nBgrpPitch, width, height, v_pitch);
}

template <class COLOR32>
void P016ToColorPlanar(uint8_t *dp_p016, int p016_pitch, uint8_t *dp_bgrp, int nBgrpPitch, int width, int height, int v_pitch, int col_standard, hipStream_t hip_stream) {
    SetMatYuv2Rgb(col_standard);
    YuvToRgbPlanarKernel<ushort2, COLOR32, uchar2>
        <<<dim3((width + 63) / 32 / 2, (height + 3) / 2 / 2), dim3(32, 2), 0, hip_stream>>>
        (dp_p016, p016_pitch, dp_bgrp, nBgrpPitch, width, height, v_pitch);
}

template <class COLOR32>
void YUV444ToColorPlanar(uint8_t *dp_yuv_444, int pitch, uint8_t *dp_bgrp, int nBgrpPitch, int width, int height, int v_pitch, int col_standard, hipStream_t hip_stream) {
    SetMatYuv2Rgb(col_standard);
    Yuv444ToRgbPlanarKernel<uchar2, COLOR32, uchar2>
        <<<dim3((width + 63) / 32 / 2, (height + 3) / 2), dim3(32, 2), 0, hip_stream>>>
        (dp_yuv_444, pitch, dp_bgrp, nBgrpPitch, width, height, v_pitch);
}

template <class COLOR32>
void YUV444P16ToColorPlanar(uint8_t *dp_yuv_444, int pitch, uint8_t *dp_bgrp, int nBgrpPitch, int width, int height, int v_pitch, int col_standard, hipStream_t hip_stream) {
    SetMatYuv2Rgb(col_standard);
    Yuv444ToRgbPlanarKernel<ushort2, COLOR32, uchar2>
        <<<dim3((width + 63) / 32 / 2, (height + 3) / 2), dim3(32, 2), 0, hip_stream>>>
        (dp_yuv_444, pitch, dp_bgrp, nBgrpPitch, width, height, v_pitch);
}

// Explicit Instantiation: for RGB32/BGR32 and RGB64/BGR64 formats
template void Nv12ToColor32<BGRA32>(uint8_t *dp_nv12, int nv12_pitch, uint8_t *dp_bgra, int bgra_pitch, int width, int height, int v_pitch, int col_standard, hipStream_t hip_stream);
template void Nv12ToColor32<RGBA32>(uint8_t *dp_nv12, int nv12_pitch, uint8_t *dp_bgra, int bgra_pitch, int width, int height, int v_pitch, int col_standard, hipStream_t hip_stream);
template void Nv12ToColor64<BGRA64>(uint8_t *dp_nv12, int nv12_pitch, uint8_t *dp_bgra, int bgra_pitch, int width, int height, int v_pitch, int col_standard, hipStream_t hip_stream);
template void Nv12ToColor64<RGBA64>(uint8_t *dp_nv12, int nv12_pitch, uint8_t *dp_bgra, int bgra_pitch, int width, int height, int v_pitch, int col_standard, hipStream_t hip_stream);
template void YUV444ToColor32<BGRA32>(uint8_t *dp_yuv_444, int pitch, uint8_t *dp_bgra, int bgra_pitch, int width, int height, int v_pitch, int col_standard, hipStream_t hip_stream);
template void YUV444ToColor32<RGBA32>(uint8_t *dp_yuv_444, int pitch, uint8_t *dp_bgra, int bgra_pitch, int width, int height, int v_pitch, int col_standard, hipStream_t hip_stream);
template void YUV444ToColor64<BGRA64>(uint8_t *dp_yuv_444, int pitch, uint8_t *dp_bgra, int bgra_pitch, int width, int height, int v_pitch, int col_standard, hipStream_t hip_stream);
template void YUV444ToColor64<RGBA64>(uint8_t *dp_yuv_444, int pitch, uint8_t *dp_bgra, int bgra_pitch, int width, int height, int v_pitch, int col_standard, hipStream_t hip_stream);
template void P016ToColor32<BGRA32>(uint8_t *dp_p016, int p016_pitch, uint8_t *dp_bgra, int bgra_pitch, int width, int height, int v_pitch, int col_standard, hipStream_t hip_stream);
template void P016ToColor32<RGBA32>(uint8_t *dp_p016, int p016_pitch, uint8_t *dp_bgra, int bgra_pitch, int width, int height, int v_pitch, int col_standard, hipStream_t hip_stream);
template void P016ToColor64<BGRA64>(uint8_t *dp_p016, int p016_pitch, uint8_t *dp_bgra, int bgra_pitch, int width, int height, int v_pitch, int col_standard, hipStream_t hip_stream);
template void P016ToColor64<RGBA64>(uint8_t *dp_p016, int p016_pitch, uint8_t *dp_bgra, int bgra_pitch, int width, int height, int v_pitch, int col_standard, hipStream_t hip_stream);
template void YUV444P16ToColor32<BGRA32>(uint8_t *dp_yuv_444, int pitch, uint8_t *dp_bgra, int bgra_pitch, int width, int height, int v_pitch, int col_standard, hipStream_t hip_stream);
template void YUV444P16ToColor32<RGBA32>(uint8_t *dp_yuv_444, int pitch, uint8_t *dp_bgra, int bgra_pitch, int width, int height, int v_pitch, int col_standard, hipStream_t hip_stream);
template void YUV444P16ToColor64<BGRA64>(uint8_t *dp_yuv_444, int pitch, uint8_t *dp_bgra, int bgra_pitch, int width, int height, int v_pitch, int col_standard, hipStream_t hip_stream);
template void YUV444P16ToColor64<RGBA64>(uint8_t *dp_yuv_444, int pitch, uint8_t *dp_bgra, int bgra_pitch, int width, int height, int v_pitch, int col_standard, hipStream_t hip_stream);
template void Nv12ToColorPlanar<BGRA32>(uint8_t *dp_nv12, int nv12_pitch, uint8_t *dp_bgrp, int nBgrpPitch, int width, int height, int v_pitch, int col_standard, hipStream_t hip_stream);
template void Nv12ToColorPlanar<RGBA32>(uint8_t *dp_nv12, int nv12_pitch, uint8_t *dp_bgrp, int nBgrpPitch, int width, int height, int v_pitch, int col_standard, hipStream_t hip_stream);
template void P016ToColorPlanar<BGRA32>(uint8_t *dp_p016, int p016_pitch, uint8_t *dp_bgrp, int nBgrpPitch, int width, int height, int v_pitch, int col_standard, hipStream_t hip_stream);
template void P016ToColorPlanar<RGBA32>(uint8_t *dp_p016, int p016_pitch, uint8_t *dp_bgrp, int nBgrpPitch, int width, int height, int v_pitch, int col_standard, hipStream_t hip_stream);
template void YUV444ToColorPlanar<BGRA32>(uint8_t *dp_yuv_444, int pitch, uint8_t *dp_bgrp, int nBgrpPitch, int width, int height, int v_pitch, int col_standard, hipStream_t hip_stream);
template void YUV444ToColorPlanar<RGBA32>(uint8_t *dp_yuv_444, int pitch, uint8_t *dp_bgrp, int nBgrpPitch, int width, int height, int v_pitch, int col_standard, hipStream_t hip_stream);
template void YUV444P16ToColorPlanar<BGRA32>(uint8_t *dp_yuv_444, int pitch, uint8_t *dp_bgrp, int nBgrpPitch, int width, int height, int v_pitch, int col_standard, hipStream_t hip_stream);
template void YUV444P16ToColorPlanar<RGBA32>(uint8_t *dp_yuv_444, int pitch, uint8_t *dp_bgrp, int nBgrpPitch, int width, int height, int v_pitch, int col_standard, hipStream_t hip_stream);

template <class COLOR24>
void Nv12ToColor24(uint8_t *dp_nv12, int nv12_pitch, uint8_t *dp_bgr, int bgr_pitch, int width, int height, int v_pitch, int col_standard, hipStream_t hip_stream) {
    SetMatYuv2Rgb(col_standard);
    YuvToRgbKernel<uchar2, COLOR24, uchar4, uchar2>
        <<<dim3((width + 63) / 32 / 2, (height + 3) / 2 / 2), dim3(32, 2), 0, hip_stream>>>
        (dp_nv12, nv12_pitch, dp_bgr, bgr_pitch, width, height, v_pitch);
}

template <class COLOR48>
void Nv12ToColor48(uint8_t *dp_nv12, int nv12_pitch, uint8_t *dp_bgr, int bgr_pitch, int width, int height, int v_pitch, int col_standard, hipStream_t hip_stream) {
    SetMatYuv2Rgb(col_standard);
    YuvToRgbKernel<uchar2, COLOR48, ushort4, ushort2>
        <<<dim3((width + 63) / 32 / 2, (height + 3) / 2 / 2), dim3(32, 2), 0, hip_stream>>>
        (dp_nv12, nv12_pitch, dp_bgr, bgr_pitch, width, height, v_pitch);
}

template <class COLOR24>
void YUV444ToColor24(uint8_t *dp_yuv_444, int pitch, uint8_t *dp_bgr, int bgr_pitch, int width, int height, int v_pitch, int col_standard, hipStream_t hip_stream) {
    SetMatYuv2Rgb(col_standard);
    Yuv444ToRgbKernel<uchar2, COLOR24, uchar4, uchar2>
        <<<dim3((width + 63) / 32 / 2, (height + 3) / 2), dim3(32, 2), 0, hip_stream>>>
        (dp_yuv_444, pitch, dp_bgr, bgr_pitch, width, height, v_pitch);
}

template <class COLOR48>
void YUV444ToColor48(uint8_t *dp_yuv_444, int pitch, uint8_t *dp_bgr, int bgr_pitch, int width, int height, int v_pitch, int col_standard, hipStream_t hip_stream) {
    SetMatYuv2Rgb(col_standard);
    Yuv444ToRgbKernel<uchar2, COLOR48, ushort4, ushort2>
        <<<dim3((width + 63) / 32 / 2, (height + 3) / 2), dim3(32, 2), 0, hip_stream>>>
        (dp_yuv_444, pitch, dp_bgr, bgr_pitch, width, height, v_pitch);
}

template <class COLOR24>
void P016ToColor24(uint8_t *dp_p016, int p016_pitch, uint8_t *dp_bgr, int bgr_pitch, int width, int height, int v_pitch, int col_standard, hipStream_t hip_stream) {
    SetMatYuv2Rgb(col_standard);
    YuvToRgbKernel<ushort2, COLOR24, uchar4, uchar2>
        <<<dim3((width + 63) / 32 / 2, (height + 3) / 2 / 2), dim3(32, 2), 0, hip_stream>>>
        (dp_p016, p016_pitch, dp_bgr, bgr_pitch, width, height, v_pitch);
}

template <class COLOR48>
void P016ToColor48(uint8_t *dp_p016, int p016_pitch, uint8_t *dp_bgr, int bgr_pitch, int width, int height, int v_pitch, int col_standard, hipStream_t hip_stream) {
    SetMatYuv2Rgb(col_standard);
    YuvToRgbKernel<ushort2, COLOR48, ushort4, ushort2>
        <<<dim3((width + 63) / 32 / 2, (height + 3) / 2 / 2), dim3(32, 2), 0, hip_stream>>>
        (dp_p016, p016_pitch, dp_bgr, bgr_pitch, width, height, v_pitch);
}

template <class COLOR24>
void YUV444P16ToColor24(uint8_t *dp_yuv_444, int pitch, uint8_t *dp_bgra, int bgra_pitch, int width, int height, int v_pitch, int col_standard, hipStream_t hip_stream) {
    SetMatYuv2Rgb(col_standard);
    Yuv444ToRgbKernel<ushort2, COLOR24, uchar4, uchar2>
        <<<dim3((width + 63) / 32 / 2, (height + 3) / 2), dim3(32, 2), 0, hip_stream>>>
        (dp_yuv_444, pitch, dp_bgra, bgra_pitch, width, height, v_pitch);
}

template <class COLOR48>
void YUV444P16ToColor48(uint8_t *dp_yuv_444, int pitch, uint8_t *dp_bgr, int bgr_pitch, int width, int height, int v_pitch, int col_standard, hipStream_t hip_stream) {
    SetMatYuv2Rgb(col_standard);
    Yuv444ToRgbKernel<ushort2, COLOR48, ushort4, ushort2>
        <<<dim3((width + 63) / 32 / 2, (height + 3) / 2), dim3(32, 2), 0, hip_stream>>>
        (dp_yuv_444, pitch, dp_bgr, bgr_pitch, width, height, v_pitch);
}


// Explicit Instantiation: for RGB24/BGR24 and RGB48/BGR48 formats
template void Nv12ToColor24<BGR24>(uint8_t *dp_nv12, int nv12_pitch, uint8_t *p_bgr, int p_bgrPitch, int width, int height, int v_pitch, int col_standard, hipStream_t hip_stream);
template void Nv12ToColor24<RGB24>(uint8_t *dp_nv12, int nv12_pitch, uint8_t *p_bgr, int p_bgrPitch, int width, int height, int v_pitch, int col_standard, hipStream_t hip_stream);
template void Nv12ToColor48<BGR48>(uint8_t *dp_nv12, int nv12_pitch, uint8_t *p_bgr, int p_bgrPitch, int width, int height, int v_pitch, int col_standard, hipStream_t hip_stream);
template void Nv12ToColor48<RGB48>(uint8_t *dp_nv12, int nv12_pitch, uint8_t *p_bgr, int p_bgrPitch, int width, int height, int v_pitch, int col_standard, hipStream_t hip_stream);
template void YUV444ToColor24<BGR24>(uint8_t *dp_yuv_444, int pitch, uint8_t *p_bgr, int p_bgrPitch, int width, int height, int v_pitch, int col_standard, hipStream_t hip_stream);
template void YUV444ToColor24<RGB24>(uint8_t *dp_yuv_444, int pitch, uint8_t *p_bgr, int p_bgrPitch, int width, int height, int v_pitch, int col_standard, hipStream_t hip_stream);
template void YUV444ToColor48<BGR48>(uint8_t *dp_yuv_444, int pitch, uint8_t *p_bgr, int p_bgrPitch, int width, int height, int v_pitch, int col_standard, hipStream_t hip_stream);
template void YUV444ToColor48<RGB48>(uint8_t *dp_yuv_444, int pitch, uint8_t *p_bgr, int p_bgrPitch, int width, int height, int v_pitch, int col_standard, hipStream_t hip_stream);
template void P016ToColor24<BGR24>(uint8_t *dp_p016, int p016_pitch, uint8_t *p_bgr, int p_bgrPitch, int width, int height, int v_pitch, int col_standard, hipStream_t hip_stream);
template void P016ToColor24<RGB24>(uint8_t *dp_p016, int p016_pitch, uint8_t *p_bgr, int p_bgrPitch, int width, int height, int v_pitch, int col_standard, hipStream_t hip_stream);
template void P016ToColor48<BGR48>(uint8_t *dp_p016, int p016_pitch, uint8_t *p_bgr, int p_bgrPitch, int width, int height, int v_pitch, int col_standard, hipStream_t hip_stream);
template void P016ToColor48<RGB48>(uint8_t *dp_p016, int p016_pitch, uint8_t *p_bgr, int p_bgrPitch, int width, int height, int v_pitch, int col_standard, hipStream_t hip_stream);
template void YUV444P16ToColor24<BGR24>(uint8_t *dp_yuv_444, int pitch, uint8_t *p_bgr, int p_bgrPitch, int width, int height, int v_pitch, int col_standard, hipStream_t hip_stream);
template void YUV444P16ToColor24<RGB24>(uint8_t *dp_yuv_444, int pitch, uint8_t *p_bgr, int p_bgrPitch, int width, int height, int v_pitch, int col_standard, hipStream_t hip_stream);
template void YUV444P16ToColor48<BGR48>(uint8_t *dp_yuv_444, int pitch, uint8_t *p_bgr, int p_bgrPitch, int width, int height, int v_pitch, int col_standard, hipStream_t hip_stream);
template void YUV444P16ToColor48<RGB48>(uint8_t *dp_yuv_444, int pitch, uint8_t *p_bgr, int p_bgrPitch, int width, int height, int v_pitch, int col_standard, hipStream_t hip_stream);


template<class YuvUnit, class RgbUnit>
__device__ inline YuvUnit RgbToY(RgbUnit r, RgbUnit g, RgbUnit b) {
    const YuvUnit low = 1 << (sizeof(YuvUnit) * 8 - 4);
    return rgb_to_yuv_mat[0][0] * r + rgb_to_yuv_mat[0][1] * g + rgb_to_yuv_mat[0][2] * b + low;
}

template<class YuvUnit, class RgbUnit>
__device__ inline YuvUnit RgbToU(RgbUnit r, RgbUnit g, RgbUnit b) {
    const YuvUnit mid = 1 << (sizeof(YuvUnit) * 8 - 1);
    return rgb_to_yuv_mat[1][0] * r + rgb_to_yuv_mat[1][1] * g + rgb_to_yuv_mat[1][2] * b + mid;
}

template<class YuvUnit, class RgbUnit>
__device__ inline YuvUnit RgbToV(RgbUnit r, RgbUnit g, RgbUnit b) {
    const YuvUnit mid = 1 << (sizeof(YuvUnit) * 8 - 1);
    return rgb_to_yuv_mat[2][0] * r + rgb_to_yuv_mat[2][1] * g + rgb_to_yuv_mat[2][2] * b + mid;
}

template<class YuvUnitx2, class Rgb, class RgbIntx2>
__global__ static void RgbaToYuvKernel(uint8_t *dp_rgb, int rgba_pitch, uint8_t *dp_yuv, int yuv_pitch, int width, int height) {
    int x = (threadIdx.x + blockIdx.x * blockDim.x) * 2;
    int y = (threadIdx.y + blockIdx.y * blockDim.y) * 2;
    if (x + 1 >= width || y + 1 >= height) {
        return;
    }

    uint8_t *p_src = dp_rgb + x * sizeof(Rgb) + y * rgba_pitch;
    RgbIntx2 int2a = *(RgbIntx2 *)p_src;
    RgbIntx2 int2b = *(RgbIntx2 *)(p_src + rgba_pitch);

    Rgb rgb[4] = {int2a.x, int2a.y, int2b.x, int2b.y};
    decltype(Rgb::c.r)
        r = (rgb[0].c.r + rgb[1].c.r + rgb[2].c.r + rgb[3].c.r) / 4,
        g = (rgb[0].c.g + rgb[1].c.g + rgb[2].c.g + rgb[3].c.g) / 4,
        b = (rgb[0].c.b + rgb[1].c.b + rgb[2].c.b + rgb[3].c.b) / 4;

    uint8_t *p_dst = dp_yuv + x * sizeof(YuvUnitx2) / 2 + y * yuv_pitch;
    *(YuvUnitx2 *)p_dst = YuvUnitx2 {
        RgbToY<decltype(YuvUnitx2::x)>(rgb[0].c.r, rgb[0].c.g, rgb[0].c.b),
        RgbToY<decltype(YuvUnitx2::x)>(rgb[1].c.r, rgb[1].c.g, rgb[1].c.b),
    };
    *(YuvUnitx2 *)(p_dst + yuv_pitch) = YuvUnitx2 {
        RgbToY<decltype(YuvUnitx2::x)>(rgb[2].c.r, rgb[2].c.g, rgb[2].c.b),
        RgbToY<decltype(YuvUnitx2::x)>(rgb[3].c.r, rgb[3].c.g, rgb[3].c.b),
    };
    *(YuvUnitx2 *)(p_dst + (height - y / 2) * yuv_pitch) = YuvUnitx2 {
        RgbToU<decltype(YuvUnitx2::x)>(r, g, b), 
        RgbToV<decltype(YuvUnitx2::x)>(r, g, b),
    };
}

void Bgra64ToP016(uint8_t *dp_bgra, int bgra_pitch, uint8_t *dp_p016, int p016_pitch, int width, int height, int col_standard, hipStream_t hip_stream) {
    SetMatRgb2Yuv(col_standard);
    RgbaToYuvKernel<ushort2, BGRA64, ulonglong2>
        <<<dim3((width + 63) / 32 / 2, (height + 3) / 2 / 2), dim3(32, 2), 0, hip_stream>>>
        (dp_bgra, bgra_pitch, dp_p016, p016_pitch, width, height);
}

template<class YuvUnitx2, class Rgb, class RgbInt1, class RgbInt2>
__global__ static void RgbToYuvKernel(uint8_t *dp_rgb, int rgb_pitch, uint8_t *dp_yuv, int yuv_pitch, int width, int height) {
    int x = (threadIdx.x + blockIdx.x * blockDim.x) * 2;
    int y = (threadIdx.y + blockIdx.y * blockDim.y) * 2;
    if (x + 1 >= width || y + 1 >= height) {
        return;
    }

    uint8_t *p_src = dp_rgb + x * sizeof(Rgb) + y * rgb_pitch;
    RgbInt1 int1a = *(RgbInt1 *)p_src;
    RgbInt2 int2a = *(RgbInt2 *)(p_src + sizeof(RgbInt1));
    RgbInt1 int1b = *(RgbInt1 *)(p_src + rgb_pitch);
    RgbInt2 int2b = *(RgbInt2 *)(p_src + rgb_pitch + sizeof(RgbInt1));

    Rgb rgb[4];
        rgb[0].v = {int1a.x, int1a.y, int1a.z},
        rgb[1].v = {int1a.w, int2a.x, int2a.y},
        rgb[2].v = {int1b.x, int1b.y, int1b.z},
        rgb[3].v = {int1b.w, int2b.x, int2b.y};
    decltype(Rgb::c.r)
        r = (rgb[0].c.r + rgb[1].c.r + rgb[2].c.r + rgb[3].c.r) / 4,
        g = (rgb[0].c.g + rgb[1].c.g + rgb[2].c.g + rgb[3].c.g) / 4,
        b = (rgb[0].c.b + rgb[1].c.b + rgb[2].c.b + rgb[3].c.b) / 4;

    uint8_t *p_dst = dp_yuv + x * sizeof(YuvUnitx2) / 2 + y * yuv_pitch;
    *(YuvUnitx2 *)p_dst = YuvUnitx2 {
        RgbToY<decltype(YuvUnitx2::x)>(rgb[0].c.r, rgb[0].c.g, rgb[0].c.b),
        RgbToY<decltype(YuvUnitx2::x)>(rgb[1].c.r, rgb[1].c.g, rgb[1].c.b),
    };
    *(YuvUnitx2 *)(p_dst + yuv_pitch) = YuvUnitx2 {
        RgbToY<decltype(YuvUnitx2::x)>(rgb[2].c.r, rgb[2].c.g, rgb[2].c.b),
        RgbToY<decltype(YuvUnitx2::x)>(rgb[3].c.r, rgb[3].c.g, rgb[3].c.b),
    };
    *(YuvUnitx2 *)(p_dst + (height - y / 2) * yuv_pitch) = YuvUnitx2 {
        RgbToU<decltype(YuvUnitx2::x)>(r, g, b), 
        RgbToV<decltype(YuvUnitx2::x)>(r, g, b),
    };
}

void Bgr48ToP016(uint8_t *p_bgr, int bgr_pitch, uint8_t *dp_p016, int p016_pitch, int width, int height, int col_standard, hipStream_t hip_stream) {
    SetMatRgb2Yuv(col_standard);
    RgbToYuvKernel<ushort2, BGR48, ushort4, ushort2>
        <<<dim3((width + 63) / 32 / 2, (height + 3) / 2 / 2), dim3(32, 2), 0, hip_stream>>>
        (p_bgr, bgr_pitch, dp_p016, p016_pitch, width, height);
}
