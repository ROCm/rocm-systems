/*
Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.

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

#include "rocjpeg_hip_kernels.h"

__device__ __forceinline__ uint32_t hipPack(float4 src) {
    return __builtin_amdgcn_cvt_pk_u8_f32(src.w, 3,
           __builtin_amdgcn_cvt_pk_u8_f32(src.z, 2,
           __builtin_amdgcn_cvt_pk_u8_f32(src.y, 1,
           __builtin_amdgcn_cvt_pk_u8_f32(src.x, 0, 0))));
}

__device__ __forceinline__ float hipUnpack0(uint32_t src) {
    return (float)(src & 0xFF);
}

__device__ __forceinline__ float hipUnpack1(uint32_t src) {
    return (float)((src >> 8) & 0xFF);
}

__device__ __forceinline__ float hipUnpack2(uint32_t src) {
    return (float)((src >> 16) & 0xFF);
}

__device__ __forceinline__ float hipUnpack3(uint32_t src) {
    return (float)((src >> 24) & 0xFF);
}

__device__ __forceinline__ float4 hipUnpack(uint32_t src) {
    return make_float4(hipUnpack0(src), hipUnpack1(src), hipUnpack2(src), hipUnpack3(src));
}

__global__ void ColorConvertYUV444ToRGBKernel(uint8_t *dst_image, uint32_t dst_image_stride_in_bytes,
    uint32_t dst_image_stride_in_bytes_comp, const uint8_t *src_y_image, const uint8_t *src_u_image, const uint8_t *src_v_image,
    uint32_t src_yuv_image_stride_in_bytes, uint32_t dst_width_comp, uint32_t dst_height_comp, uint32_t src_yuv_image_stride_in_bytes_comp) {

    uint32_t x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    uint32_t y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    if ((x < dst_width_comp) && (y < dst_height_comp)) {
        uint32_t src_y0_idx = y * src_yuv_image_stride_in_bytes_comp + (x << 3);
        uint32_t src_y1_idx = src_y0_idx + src_yuv_image_stride_in_bytes;


        uint2 y0 = *((uint2 *)(&src_y_image[src_y0_idx]));
        uint2 y1 = *((uint2 *)(&src_y_image[src_y1_idx]));

        uint2 u0 = *((uint2 *)(&src_u_image[src_y0_idx]));
        uint2 u1 = *((uint2 *)(&src_u_image[src_y1_idx]));

        uint2 v0 = *((uint2 *)(&src_v_image[src_y0_idx]));
        uint2 v1 = *((uint2 *)(&src_v_image[src_y1_idx]));

        uint32_t rgb0_idx = y * dst_image_stride_in_bytes_comp + (x * 24);
        uint32_t rgb1_idx = rgb0_idx + dst_image_stride_in_bytes;

        float2 cr = make_float2( 0.0000f,  1.5748f);
        float2 cg = make_float2(-0.1873f, -0.4681f);
        float2 cb = make_float2( 1.8556f,  0.0000f);
        float3 yuv;
        DUINT6 rgb0, rgb1;
        float4 f;

        yuv = make_float3(hipUnpack0(y0.x), hipUnpack0(u0.x), hipUnpack0(v0.x));
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.x = fmaf(cr.y, yuv.z, yuv.x);
        f.y = fmaf(cg.x, yuv.y, yuv.x);
        f.y = fmaf(cg.y, yuv.z, f.y);
        f.z = fmaf(cb.x, yuv.y, yuv.x);
        yuv = make_float3(hipUnpack1(y0.x), hipUnpack1(u0.x), hipUnpack1(v0.x));
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.w = fmaf(cr.y, yuv.z, yuv.x);
        rgb0.data[0] = hipPack(f);

        f.x = fmaf(cg.x, yuv.y, yuv.x);
        f.x = fmaf(cg.y, yuv.z, f.x);
        f.y = fmaf(cb.x, yuv.y, yuv.x);
        yuv = make_float3(hipUnpack2(y0.x), hipUnpack2(u0.x), hipUnpack2(v0.x));
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.z = fmaf(cr.y, yuv.z, yuv.x);
        f.w = fmaf(cg.x, yuv.y, yuv.x);
        f.w = fmaf(cg.y, yuv.z, f.w);
        rgb0.data[1] = hipPack(f);

        f.x = fmaf(cb.x, yuv.y, yuv.x);
        yuv = make_float3(hipUnpack3(y0.x), hipUnpack3(u0.x), hipUnpack3(v0.x));
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.y = fmaf(cr.y, yuv.z, yuv.x);
        f.z = fmaf(cg.x, yuv.y, yuv.x);
        f.z = fmaf(cg.y, yuv.z, f.z);
        f.w = fmaf(cb.x, yuv.y, yuv.x);
        rgb0.data[2] = hipPack(f);

        yuv = make_float3(hipUnpack0(y0.y), hipUnpack0(u0.y), hipUnpack0(v0.y));
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.x = fmaf(cr.y, yuv.z, yuv.x);
        f.y = fmaf(cg.x, yuv.y, yuv.x);
        f.y = fmaf(cg.y, yuv.z, f.y);
        f.z = fmaf(cb.x, yuv.y, yuv.x);
        yuv = make_float3(hipUnpack1(y0.y), hipUnpack1(u0.y), hipUnpack1(v0.y));
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.w = fmaf(cr.y, yuv.z, yuv.x);
        rgb0.data[3] = hipPack(f);

        f.x = fmaf(cg.x, yuv.y, yuv.x);
        f.x = fmaf(cg.y, yuv.z, f.x);
        f.y = fmaf(cb.x, yuv.y, yuv.x);
        yuv = make_float3(hipUnpack2(y0.y), hipUnpack2(u0.y), hipUnpack2(v0.y));
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.z = fmaf(cr.y, yuv.z, yuv.x);
        f.w = fmaf(cg.x, yuv.y, yuv.x);
        f.w = fmaf(cg.y, yuv.z, f.w);
        rgb0.data[4] = hipPack(f);

        f.x = fmaf(cb.x, yuv.y, yuv.x);
        yuv = make_float3(hipUnpack3(y0.y), hipUnpack3(u0.y), hipUnpack3(v0.y));
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.y = fmaf(cr.y, yuv.z, yuv.x);
        f.z = fmaf(cg.x, yuv.y, yuv.x);
        f.z = fmaf(cg.y, yuv.z, f.z);
        f.w = fmaf(cb.x, yuv.y, yuv.x);
        rgb0.data[5] = hipPack(f);

        yuv = make_float3(hipUnpack0(y1.x), hipUnpack0(u1.x), hipUnpack0(v1.x));
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.x = fmaf(cr.y, yuv.z, yuv.x);
        f.y = fmaf(cg.x, yuv.y, yuv.x);
        f.y = fmaf(cg.y, yuv.z, f.y);
        f.z = fmaf(cb.x, yuv.y, yuv.x);
        yuv = make_float3(hipUnpack1(y1.x), hipUnpack1(u1.x), hipUnpack1(v1.x));
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.w = fmaf(cr.y, yuv.z, yuv.x);
        rgb1.data[0] = hipPack(f);

        f.x = fmaf(cg.x, yuv.y, yuv.x);
        f.x = fmaf(cg.y, yuv.z, f.x);
        f.y = fmaf(cb.x, yuv.y, yuv.x);
        yuv = make_float3(hipUnpack2(y1.x), hipUnpack2(u1.x), hipUnpack2(v1.x));
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.z = fmaf(cr.y, yuv.z, yuv.x);
        f.w = fmaf(cg.x, yuv.y, yuv.x);
        f.w = fmaf(cg.y, yuv.z, f.w);
        rgb1.data[1] = hipPack(f);

        f.x = fmaf(cb.x, yuv.y, yuv.x);
        yuv = make_float3(hipUnpack3(y1.x), hipUnpack3(u1.x), hipUnpack3(v1.x));
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.y = fmaf(cr.y, yuv.z, yuv.x);
        f.z = fmaf(cg.x, yuv.y, yuv.x);
        f.z = fmaf(cg.y, yuv.z, f.z);
        f.w = fmaf(cb.x, yuv.y, yuv.x);
        rgb1.data[2] = hipPack(f);

        yuv = make_float3(hipUnpack0(y1.y), hipUnpack0(u1.y), hipUnpack0(v1.y));
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.x = fmaf(cr.y, yuv.z, yuv.x);
        f.y = fmaf(cg.x, yuv.y, yuv.x);
        f.y = fmaf(cg.y, yuv.z, f.y);
        f.z = fmaf(cb.x, yuv.y, yuv.x);
        yuv = make_float3(hipUnpack1(y1.y), hipUnpack1(u1.y), hipUnpack1(v1.y));
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.w = fmaf(cr.y, yuv.z, yuv.x);
        rgb1.data[3] = hipPack(f);

        f.x = fmaf(cg.x, yuv.y, yuv.x);
        f.x = fmaf(cg.y, yuv.z, f.x);
        f.y = fmaf(cb.x, yuv.y, yuv.x);
        yuv = make_float3(hipUnpack2(y1.y), hipUnpack2(u1.y), hipUnpack2(v1.y));
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.z = fmaf(cr.y, yuv.z, yuv.x);
        f.w = fmaf(cg.x, yuv.y, yuv.x);
        f.w = fmaf(cg.y, yuv.z, f.w);
        rgb1.data[4] = hipPack(f);

        f.x = fmaf(cb.x, yuv.y, yuv.x);
        yuv = make_float3(hipUnpack3(y1.y), hipUnpack3(u1.y), hipUnpack3(v1.y));
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.y = fmaf(cr.y, yuv.z, yuv.x);
        f.z = fmaf(cg.x, yuv.y, yuv.x);
        f.z = fmaf(cg.y, yuv.z, f.z);
        f.w = fmaf(cb.x, yuv.y, yuv.x);
        rgb1.data[5] = hipPack(f);

        *((DUINT6 *)(&dst_image[rgb0_idx])) = rgb0;
        *((DUINT6 *)(&dst_image[rgb1_idx])) = rgb1;
    }
}

/**
 * @brief Converts YUV444 image to RGB image.
 *
 * This function takes a YUV444 image and converts it to an RGB image using the ColorConvertYUV444ToRGBKernel HIP kernel.
 *
 * @param stream The HIP stream used for asynchronous execution of the kernel.
 * @param dst_width The width of the destination RGB image.
 * @param dst_height The height of the destination RGB image.
 * @param dst_image Pointer to the destination RGB image buffer.
 * @param dst_image_stride_in_bytes The stride (in bytes) of the destination RGB image buffer.
 * @param src_yuv_image Pointer to the source YUV444 image buffer.
 * @param src_yuv_image_stride_in_bytes The stride (in bytes) of the source YUV444 image buffer.
 * @param src_u_image_offset The offset (in bytes) to the U component in the source YUV444 image buffer.
 */
void ColorConvertYUV444ToRGB(hipStream_t stream, uint32_t dst_width, uint32_t dst_height,
    uint8_t *dst_image, uint32_t dst_image_stride_in_bytes, const uint8_t *src_yuv_image,
    uint32_t src_yuv_image_stride_in_bytes, uint32_t src_u_image_offset, uint32_t src_v_image_offset) {

    int32_t local_threads_x = 16;
    int32_t local_threads_y = 4;
    int32_t global_threads_x = (dst_width + 7) >> 3;
    int32_t global_threads_y = (dst_height + 1) >> 1;

    uint32_t dst_width_comp = (dst_width + 7) / 8;
    uint32_t dst_height_comp = (dst_height + 1) / 2;
    uint32_t dst_image_stride_in_bytes_comp = dst_image_stride_in_bytes * 2;
    uint32_t src_yuv_image_stride_in_bytes_comp = src_yuv_image_stride_in_bytes * 2;

    ColorConvertYUV444ToRGBKernel<<<dim3(ceil(static_cast<float>(global_threads_x) / local_threads_x), ceil(static_cast<float>(global_threads_y) / local_threads_y)),
                        dim3(local_threads_x, local_threads_y), 0, stream>>>(dst_image,
                        dst_image_stride_in_bytes, dst_image_stride_in_bytes_comp, src_yuv_image, src_yuv_image + src_u_image_offset,
                        src_yuv_image + src_v_image_offset, src_yuv_image_stride_in_bytes,
                        dst_width_comp, dst_height_comp, src_yuv_image_stride_in_bytes_comp);
}

__global__ void ColorConvertYUV444ToRGBPlanarKernel(uint8_t *dst_image_r, uint8_t *dst_image_g, uint8_t *dst_image_b, uint32_t dst_image_stride_in_bytes,
    uint32_t dst_image_stride_in_bytes_comp, const uint8_t *src_y_image, const uint8_t *src_u_image, const uint8_t *src_v_image,
    uint32_t src_yuv_image_stride_in_bytes, uint32_t dst_width_comp, uint32_t dst_height_comp, uint32_t src_yuv_image_stride_in_bytes_comp) {

    uint32_t x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    uint32_t y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    if ((x < dst_width_comp) && (y < dst_height_comp)) {
        uint32_t src_y0_idx = y * src_yuv_image_stride_in_bytes_comp + (x << 3);
        uint32_t src_y1_idx = src_y0_idx + src_yuv_image_stride_in_bytes;


        uint2 y0 = *((uint2 *)(&src_y_image[src_y0_idx]));
        uint2 y1 = *((uint2 *)(&src_y_image[src_y1_idx]));

        uint2 u0 = *((uint2 *)(&src_u_image[src_y0_idx]));
        uint2 u1 = *((uint2 *)(&src_u_image[src_y1_idx]));

        uint2 v0 = *((uint2 *)(&src_v_image[src_y0_idx]));
        uint2 v1 = *((uint2 *)(&src_v_image[src_y1_idx]));

        uint32_t rgb0_idx = y * dst_image_stride_in_bytes_comp + (x * 8);
        uint32_t rgb1_idx = rgb0_idx + dst_image_stride_in_bytes;

        float2 cr = make_float2( 0.0000f,  1.5748f);
        float2 cg = make_float2(-0.1873f, -0.4681f);
        float2 cb = make_float2( 1.8556f,  0.0000f);
        float3 yuv;
        DUINT6 rgb0, rgb1;
        float4 f;

        yuv = make_float3(hipUnpack0(y0.x), hipUnpack0(u0.x), hipUnpack0(v0.x));
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.x = fmaf(cr.y, yuv.z, yuv.x);
        f.y = fmaf(cg.x, yuv.y, yuv.x);
        f.y = fmaf(cg.y, yuv.z, f.y);
        f.z = fmaf(cb.x, yuv.y, yuv.x);
        yuv = make_float3(hipUnpack1(y0.x), hipUnpack1(u0.x), hipUnpack1(v0.x));
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.w = fmaf(cr.y, yuv.z, yuv.x);
        rgb0.data[0] = hipPack(f);

        f.x = fmaf(cg.x, yuv.y, yuv.x);
        f.x = fmaf(cg.y, yuv.z, f.x);
        f.y = fmaf(cb.x, yuv.y, yuv.x);
        yuv = make_float3(hipUnpack2(y0.x), hipUnpack2(u0.x), hipUnpack2(v0.x));
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.z = fmaf(cr.y, yuv.z, yuv.x);
        f.w = fmaf(cg.x, yuv.y, yuv.x);
        f.w = fmaf(cg.y, yuv.z, f.w);
        rgb0.data[1] = hipPack(f);

        f.x = fmaf(cb.x, yuv.y, yuv.x);
        yuv = make_float3(hipUnpack3(y0.x), hipUnpack3(u0.x), hipUnpack3(v0.x));
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.y = fmaf(cr.y, yuv.z, yuv.x);
        f.z = fmaf(cg.x, yuv.y, yuv.x);
        f.z = fmaf(cg.y, yuv.z, f.z);
        f.w = fmaf(cb.x, yuv.y, yuv.x);
        rgb0.data[2] = hipPack(f);

        yuv = make_float3(hipUnpack0(y0.y), hipUnpack0(u0.y), hipUnpack0(v0.y));
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.x = fmaf(cr.y, yuv.z, yuv.x);
        f.y = fmaf(cg.x, yuv.y, yuv.x);
        f.y = fmaf(cg.y, yuv.z, f.y);
        f.z = fmaf(cb.x, yuv.y, yuv.x);
        yuv = make_float3(hipUnpack1(y0.y), hipUnpack1(u0.y), hipUnpack1(v0.y));
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.w = fmaf(cr.y, yuv.z, yuv.x);
        rgb0.data[3] = hipPack(f);

        f.x = fmaf(cg.x, yuv.y, yuv.x);
        f.x = fmaf(cg.y, yuv.z, f.x);
        f.y = fmaf(cb.x, yuv.y, yuv.x);
        yuv = make_float3(hipUnpack2(y0.y), hipUnpack2(u0.y), hipUnpack2(v0.y));
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.z = fmaf(cr.y, yuv.z, yuv.x);
        f.w = fmaf(cg.x, yuv.y, yuv.x);
        f.w = fmaf(cg.y, yuv.z, f.w);
        rgb0.data[4] = hipPack(f);

        f.x = fmaf(cb.x, yuv.y, yuv.x);
        yuv = make_float3(hipUnpack3(y0.y), hipUnpack3(u0.y), hipUnpack3(v0.y));
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.y = fmaf(cr.y, yuv.z, yuv.x);
        f.z = fmaf(cg.x, yuv.y, yuv.x);
        f.z = fmaf(cg.y, yuv.z, f.z);
        f.w = fmaf(cb.x, yuv.y, yuv.x);
        rgb0.data[5] = hipPack(f);

        yuv = make_float3(hipUnpack0(y1.x), hipUnpack0(u1.x), hipUnpack0(v1.x));
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.x = fmaf(cr.y, yuv.z, yuv.x);
        f.y = fmaf(cg.x, yuv.y, yuv.x);
        f.y = fmaf(cg.y, yuv.z, f.y);
        f.z = fmaf(cb.x, yuv.y, yuv.x);
        yuv = make_float3(hipUnpack1(y1.x), hipUnpack1(u1.x), hipUnpack1(v1.x));
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.w = fmaf(cr.y, yuv.z, yuv.x);
        rgb1.data[0] = hipPack(f);

        f.x = fmaf(cg.x, yuv.y, yuv.x);
        f.x = fmaf(cg.y, yuv.z, f.x);
        f.y = fmaf(cb.x, yuv.y, yuv.x);
        yuv = make_float3(hipUnpack2(y1.x), hipUnpack2(u1.x), hipUnpack2(v1.x));
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.z = fmaf(cr.y, yuv.z, yuv.x);
        f.w = fmaf(cg.x, yuv.y, yuv.x);
        f.w = fmaf(cg.y, yuv.z, f.w);
        rgb1.data[1] = hipPack(f);

        f.x = fmaf(cb.x, yuv.y, yuv.x);
        yuv = make_float3(hipUnpack3(y1.x), hipUnpack3(u1.x), hipUnpack3(v1.x));
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.y = fmaf(cr.y, yuv.z, yuv.x);
        f.z = fmaf(cg.x, yuv.y, yuv.x);
        f.z = fmaf(cg.y, yuv.z, f.z);
        f.w = fmaf(cb.x, yuv.y, yuv.x);
        rgb1.data[2] = hipPack(f);

        yuv = make_float3(hipUnpack0(y1.y), hipUnpack0(u1.y), hipUnpack0(v1.y));
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.x = fmaf(cr.y, yuv.z, yuv.x);
        f.y = fmaf(cg.x, yuv.y, yuv.x);
        f.y = fmaf(cg.y, yuv.z, f.y);
        f.z = fmaf(cb.x, yuv.y, yuv.x);
        yuv = make_float3(hipUnpack1(y1.y), hipUnpack1(u1.y), hipUnpack1(v1.y));
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.w = fmaf(cr.y, yuv.z, yuv.x);
        rgb1.data[3] = hipPack(f);

        f.x = fmaf(cg.x, yuv.y, yuv.x);
        f.x = fmaf(cg.y, yuv.z, f.x);
        f.y = fmaf(cb.x, yuv.y, yuv.x);
        yuv = make_float3(hipUnpack2(y1.y), hipUnpack2(u1.y), hipUnpack2(v1.y));
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.z = fmaf(cr.y, yuv.z, yuv.x);
        f.w = fmaf(cg.x, yuv.y, yuv.x);
        f.w = fmaf(cg.y, yuv.z, f.w);
        rgb1.data[4] = hipPack(f);

        f.x = fmaf(cb.x, yuv.y, yuv.x);
        yuv = make_float3(hipUnpack3(y1.y), hipUnpack3(u1.y), hipUnpack3(v1.y));
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.y = fmaf(cr.y, yuv.z, yuv.x);
        f.z = fmaf(cg.x, yuv.y, yuv.x);
        f.z = fmaf(cg.y, yuv.z, f.z);
        f.w = fmaf(cb.x, yuv.y, yuv.x);
        rgb1.data[5] = hipPack(f);

        uint2 red0, red1, green0, green1, blue0, blue1;
        red0.x = hipPack(make_float4(hipUnpack0(rgb0.data[0]), hipUnpack3(rgb0.data[0]), hipUnpack2(rgb0.data[1]), hipUnpack1(rgb0.data[2])));
        red0.y = hipPack(make_float4(hipUnpack0(rgb0.data[3]), hipUnpack3(rgb0.data[3]), hipUnpack2(rgb0.data[4]), hipUnpack1(rgb0.data[5])));
        red1.x = hipPack(make_float4(hipUnpack0(rgb1.data[0]), hipUnpack3(rgb1.data[0]), hipUnpack2(rgb1.data[1]), hipUnpack1(rgb1.data[2])));
        red1.y = hipPack(make_float4(hipUnpack0(rgb1.data[3]), hipUnpack3(rgb1.data[3]), hipUnpack2(rgb1.data[4]), hipUnpack1(rgb1.data[5])));

        green0.x = hipPack(make_float4(hipUnpack1(rgb0.data[0]), hipUnpack0(rgb0.data[1]), hipUnpack3(rgb0.data[1]), hipUnpack2(rgb0.data[2])));
        green0.y = hipPack(make_float4(hipUnpack1(rgb0.data[3]), hipUnpack0(rgb0.data[4]), hipUnpack3(rgb0.data[4]), hipUnpack2(rgb0.data[5])));
        green1.x = hipPack(make_float4(hipUnpack1(rgb1.data[0]), hipUnpack0(rgb1.data[1]), hipUnpack3(rgb1.data[1]), hipUnpack2(rgb1.data[2])));
        green1.y = hipPack(make_float4(hipUnpack1(rgb1.data[3]), hipUnpack0(rgb1.data[4]), hipUnpack3(rgb1.data[4]), hipUnpack2(rgb1.data[5])));

        blue0.x = hipPack(make_float4(hipUnpack2(rgb0.data[0]), hipUnpack1(rgb0.data[1]), hipUnpack0(rgb0.data[2]), hipUnpack3(rgb0.data[2])));
        blue0.y = hipPack(make_float4(hipUnpack2(rgb0.data[3]), hipUnpack1(rgb0.data[4]), hipUnpack0(rgb0.data[5]), hipUnpack3(rgb0.data[5])));
        blue1.x = hipPack(make_float4(hipUnpack2(rgb1.data[0]), hipUnpack1(rgb1.data[1]), hipUnpack0(rgb1.data[2]), hipUnpack3(rgb1.data[2])));
        blue1.y = hipPack(make_float4(hipUnpack2(rgb1.data[3]), hipUnpack1(rgb1.data[4]), hipUnpack0(rgb1.data[5]), hipUnpack3(rgb1.data[5])));

        *((uint2 *)(&dst_image_r[rgb0_idx])) = red0;
        *((uint2 *)(&dst_image_r[rgb1_idx])) = red1;
        *((uint2 *)(&dst_image_g[rgb0_idx])) = green0;
        *((uint2 *)(&dst_image_g[rgb1_idx])) = green1;
        *((uint2 *)(&dst_image_b[rgb0_idx])) = blue0;
        *((uint2 *)(&dst_image_b[rgb1_idx])) = blue1;
    }
}


/**
 * @brief Converts YUV444 image to RGB planar format.
 *
 * This function takes a YUV444 image and converts it to RGB planar format using the ColorConvertYUV444ToRGBPlanarKernel HIP kernel.
 *
 * @param stream The HIP stream to be used for the kernel launch.
 * @param dst_width The width of the destination RGB image.
 * @param dst_height The height of the destination RGB image.
 * @param dst_image_r Pointer to the destination red channel image.
 * @param dst_image_g Pointer to the destination green channel image.
 * @param dst_image_b Pointer to the destination blue channel image.
 * @param dst_image_stride_in_bytes The stride (in bytes) of the destination image.
 * @param src_yuv_image Pointer to the source YUV image.
 * @param src_yuv_image_stride_in_bytes The stride (in bytes) of the source YUV image.
 * @param src_u_image_offset The offset (in bytes) to the U channel in the source YUV image.
 */
void ColorConvertYUV444ToRGBPlanar(hipStream_t stream, uint32_t dst_width, uint32_t dst_height,
    uint8_t *dst_image_r, uint8_t *dst_image_g, uint8_t *dst_image_b, uint32_t dst_image_stride_in_bytes, const uint8_t *src_yuv_image,
    uint32_t src_yuv_image_stride_in_bytes, uint32_t src_u_image_offset, uint32_t src_v_image_offset) {

    int32_t local_threads_x = 16;
    int32_t local_threads_y = 4;
    int32_t global_threads_x = (dst_width + 7) >> 3;
    int32_t global_threads_y = (dst_height + 1) >> 1;

    uint32_t dst_width_comp = (dst_width + 7) / 8;
    uint32_t dst_height_comp = (dst_height + 1) / 2;
    uint32_t dst_image_stride_in_bytes_comp = dst_image_stride_in_bytes * 2;
    uint32_t src_yuv_image_stride_in_bytes_comp = src_yuv_image_stride_in_bytes * 2;

    ColorConvertYUV444ToRGBPlanarKernel<<<dim3(ceil(static_cast<float>(global_threads_x) / local_threads_x), ceil(static_cast<float>(global_threads_y) / local_threads_y)),
                        dim3(local_threads_x, local_threads_y), 0, stream>>>(dst_image_r, dst_image_g, dst_image_b,
                        dst_image_stride_in_bytes, dst_image_stride_in_bytes_comp, src_yuv_image, src_yuv_image + src_u_image_offset,
                        src_yuv_image + src_v_image_offset, src_yuv_image_stride_in_bytes,
                        dst_width_comp, dst_height_comp, src_yuv_image_stride_in_bytes_comp);
}

__global__ void ColorConvertYUV440ToRGBKernel(uint8_t *dst_image, uint32_t dst_image_stride_in_bytes,
    uint32_t dst_image_stride_in_bytes_comp, const uint8_t *src_y_image, const uint8_t *src_u_image, const uint8_t *src_v_image,
    uint32_t src_yuv_image_stride_in_bytes, uint32_t dst_width_comp, uint32_t dst_height_comp, uint32_t src_yuv_image_stride_in_bytes_comp) {

    uint32_t x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    uint32_t y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    if ((x < dst_width_comp) && (y < dst_height_comp)) {
        uint32_t src_y0_idx = y * src_yuv_image_stride_in_bytes_comp + (x << 3);
        uint32_t src_y1_idx = src_y0_idx + src_yuv_image_stride_in_bytes;
        uint32_t src_chroma_idx = y * src_yuv_image_stride_in_bytes + (x << 3);

        uint2 y0 = *((uint2 *)(&src_y_image[src_y0_idx]));
        uint2 y1 = *((uint2 *)(&src_y_image[src_y1_idx]));

        uint2 u0 = *((uint2 *)(&src_u_image[src_chroma_idx]));
        uint2 v0 = *((uint2 *)(&src_v_image[src_chroma_idx]));

        uint32_t rgb0_idx = y * dst_image_stride_in_bytes_comp + (x * 24);
        uint32_t rgb1_idx = rgb0_idx + dst_image_stride_in_bytes;

        float2 cr = make_float2( 0.0000f,  1.5748f);
        float2 cg = make_float2(-0.1873f, -0.4681f);
        float2 cb = make_float2( 1.8556f,  0.0000f);
        float3 yuv;
        DUINT6 rgb0, rgb1;
        float4 f;

        yuv = make_float3(hipUnpack0(y0.x), hipUnpack0(u0.x), hipUnpack0(v0.x));
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.x = fmaf(cr.y, yuv.z, yuv.x);
        f.y = fmaf(cg.x, yuv.y, yuv.x);
        f.y = fmaf(cg.y, yuv.z, f.y);
        f.z = fmaf(cb.x, yuv.y, yuv.x);
        yuv = make_float3(hipUnpack1(y0.x), hipUnpack1(u0.x), hipUnpack1(v0.x));
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.w = fmaf(cr.y, yuv.z, yuv.x);
        rgb0.data[0] = hipPack(f);

        f.x = fmaf(cg.x, yuv.y, yuv.x);
        f.x = fmaf(cg.y, yuv.z, f.x);
        f.y = fmaf(cb.x, yuv.y, yuv.x);
        yuv = make_float3(hipUnpack2(y0.x), hipUnpack2(u0.x), hipUnpack2(v0.x));
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.z = fmaf(cr.y, yuv.z, yuv.x);
        f.w = fmaf(cg.x, yuv.y, yuv.x);
        f.w = fmaf(cg.y, yuv.z, f.w);
        rgb0.data[1] = hipPack(f);

        f.x = fmaf(cb.x, yuv.y, yuv.x);
        yuv = make_float3(hipUnpack3(y0.x), hipUnpack3(u0.x), hipUnpack3(v0.x));
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.y = fmaf(cr.y, yuv.z, yuv.x);
        f.z = fmaf(cg.x, yuv.y, yuv.x);
        f.z = fmaf(cg.y, yuv.z, f.z);
        f.w = fmaf(cb.x, yuv.y, yuv.x);
        rgb0.data[2] = hipPack(f);

        yuv = make_float3(hipUnpack0(y0.y), hipUnpack0(u0.y), hipUnpack0(v0.y));
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.x = fmaf(cr.y, yuv.z, yuv.x);
        f.y = fmaf(cg.x, yuv.y, yuv.x);
        f.y = fmaf(cg.y, yuv.z, f.y);
        f.z = fmaf(cb.x, yuv.y, yuv.x);
        yuv = make_float3(hipUnpack1(y0.y), hipUnpack1(u0.y), hipUnpack1(v0.y));
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.w = fmaf(cr.y, yuv.z, yuv.x);
        rgb0.data[3] = hipPack(f);

        f.x = fmaf(cg.x, yuv.y, yuv.x);
        f.x = fmaf(cg.y, yuv.z, f.x);
        f.y = fmaf(cb.x, yuv.y, yuv.x);
        yuv = make_float3(hipUnpack2(y0.y), hipUnpack2(u0.y), hipUnpack2(v0.y));
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.z = fmaf(cr.y, yuv.z, yuv.x);
        f.w = fmaf(cg.x, yuv.y, yuv.x);
        f.w = fmaf(cg.y, yuv.z, f.w);
        rgb0.data[4] = hipPack(f);

        f.x = fmaf(cb.x, yuv.y, yuv.x);
        yuv = make_float3(hipUnpack3(y0.y), hipUnpack3(u0.y), hipUnpack3(v0.y));
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.y = fmaf(cr.y, yuv.z, yuv.x);
        f.z = fmaf(cg.x, yuv.y, yuv.x);
        f.z = fmaf(cg.y, yuv.z, f.z);
        f.w = fmaf(cb.x, yuv.y, yuv.x);
        rgb0.data[5] = hipPack(f);

        yuv = make_float3(hipUnpack0(y1.x), hipUnpack0(u0.x), hipUnpack0(v0.x));
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.x = fmaf(cr.y, yuv.z, yuv.x);
        f.y = fmaf(cg.x, yuv.y, yuv.x);
        f.y = fmaf(cg.y, yuv.z, f.y);
        f.z = fmaf(cb.x, yuv.y, yuv.x);
        yuv = make_float3(hipUnpack1(y1.x), hipUnpack1(u0.x), hipUnpack1(v0.x));
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.w = fmaf(cr.y, yuv.z, yuv.x);
        rgb1.data[0] = hipPack(f);

        f.x = fmaf(cg.x, yuv.y, yuv.x);
        f.x = fmaf(cg.y, yuv.z, f.x);
        f.y = fmaf(cb.x, yuv.y, yuv.x);
        yuv = make_float3(hipUnpack2(y1.x), hipUnpack2(u0.x), hipUnpack2(v0.x));
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.z = fmaf(cr.y, yuv.z, yuv.x);
        f.w = fmaf(cg.x, yuv.y, yuv.x);
        f.w = fmaf(cg.y, yuv.z, f.w);
        rgb1.data[1] = hipPack(f);

        f.x = fmaf(cb.x, yuv.y, yuv.x);
        yuv = make_float3(hipUnpack3(y1.x), hipUnpack3(u0.x), hipUnpack3(v0.x));
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.y = fmaf(cr.y, yuv.z, yuv.x);
        f.z = fmaf(cg.x, yuv.y, yuv.x);
        f.z = fmaf(cg.y, yuv.z, f.z);
        f.w = fmaf(cb.x, yuv.y, yuv.x);
        rgb1.data[2] = hipPack(f);

        yuv = make_float3(hipUnpack0(y1.y), hipUnpack0(u0.y), hipUnpack0(v0.y));
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.x = fmaf(cr.y, yuv.z, yuv.x);
        f.y = fmaf(cg.x, yuv.y, yuv.x);
        f.y = fmaf(cg.y, yuv.z, f.y);
        f.z = fmaf(cb.x, yuv.y, yuv.x);
        yuv = make_float3(hipUnpack1(y1.y), hipUnpack1(u0.y), hipUnpack1(v0.y));
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.w = fmaf(cr.y, yuv.z, yuv.x);
        rgb1.data[3] = hipPack(f);

        f.x = fmaf(cg.x, yuv.y, yuv.x);
        f.x = fmaf(cg.y, yuv.z, f.x);
        f.y = fmaf(cb.x, yuv.y, yuv.x);
        yuv = make_float3(hipUnpack2(y1.y), hipUnpack2(u0.y), hipUnpack2(v0.y));
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.z = fmaf(cr.y, yuv.z, yuv.x);
        f.w = fmaf(cg.x, yuv.y, yuv.x);
        f.w = fmaf(cg.y, yuv.z, f.w);
        rgb1.data[4] = hipPack(f);

        f.x = fmaf(cb.x, yuv.y, yuv.x);
        yuv = make_float3(hipUnpack3(y1.y), hipUnpack3(u0.y), hipUnpack3(v0.y));
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.y = fmaf(cr.y, yuv.z, yuv.x);
        f.z = fmaf(cg.x, yuv.y, yuv.x);
        f.z = fmaf(cg.y, yuv.z, f.z);
        f.w = fmaf(cb.x, yuv.y, yuv.x);
        rgb1.data[5] = hipPack(f);

        *((DUINT6 *)(&dst_image[rgb0_idx])) = rgb0;
        *((DUINT6 *)(&dst_image[rgb1_idx])) = rgb1;
    }
}

/**
 * @brief Converts YUV440 image to RGB image.
 *
 * This function takes a YUV440 image and converts it to an RGB image using the ColorConvertYUV444ToRGBKernel HIP kernel.
 *
 * @param stream The HIP stream used for asynchronous execution of the kernel.
 * @param dst_width The width of the destination RGB image.
 * @param dst_height The height of the destination RGB image.
 * @param dst_image Pointer to the destination RGB image buffer.
 * @param dst_image_stride_in_bytes The stride (in bytes) of the destination RGB image buffer.
 * @param src_yuv_image Pointer to the source YUV440 image buffer.
 * @param src_yuv_image_stride_in_bytes The stride (in bytes) of the source YUV440 image buffer.
 * @param src_u_image_offset The offset (in bytes) to the U component in the source YUV440 image buffer.
 */
void ColorConvertYUV440ToRGB(hipStream_t stream, uint32_t dst_width, uint32_t dst_height,
    uint8_t *dst_image, uint32_t dst_image_stride_in_bytes, const uint8_t *src_yuv_image,
    uint32_t src_yuv_image_stride_in_bytes, uint32_t src_u_image_offset, uint32_t src_v_image_offset) {

    int32_t local_threads_x = 16;
    int32_t local_threads_y = 4;
    int32_t global_threads_x = (dst_width + 7) >> 3;
    int32_t global_threads_y = (dst_height + 1) >> 1;

    uint32_t dst_width_comp = (dst_width + 7) / 8;
    uint32_t dst_height_comp = (dst_height + 1) / 2;
    uint32_t dst_image_stride_in_bytes_comp = dst_image_stride_in_bytes * 2;
    uint32_t src_yuv_image_stride_in_bytes_comp = src_yuv_image_stride_in_bytes * 2;

    ColorConvertYUV440ToRGBKernel<<<dim3(ceil(static_cast<float>(global_threads_x) / local_threads_x), ceil(static_cast<float>(global_threads_y) / local_threads_y)),
                        dim3(local_threads_x, local_threads_y), 0, stream>>>(dst_image,
                        dst_image_stride_in_bytes, dst_image_stride_in_bytes_comp, src_yuv_image, src_yuv_image + src_u_image_offset,
                        src_yuv_image + src_v_image_offset, src_yuv_image_stride_in_bytes,
                        dst_width_comp, dst_height_comp, src_yuv_image_stride_in_bytes_comp);
}

__global__ void ColorConvertYUV440ToRGBPlanarKernel(uint8_t *dst_image_r, uint8_t *dst_image_g, uint8_t *dst_image_b, uint32_t dst_image_stride_in_bytes,
    uint32_t dst_image_stride_in_bytes_comp, const uint8_t *src_y_image, const uint8_t *src_u_image, const uint8_t *src_v_image,
    uint32_t src_yuv_image_stride_in_bytes, uint32_t dst_width_comp, uint32_t dst_height_comp, uint32_t src_yuv_image_stride_in_bytes_comp) {

    uint32_t x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    uint32_t y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    if ((x < dst_width_comp) && (y < dst_height_comp)) {
        uint32_t src_y0_idx = y * src_yuv_image_stride_in_bytes_comp + (x << 3);
        uint32_t src_y1_idx = src_y0_idx + src_yuv_image_stride_in_bytes;
        uint32_t src_chroma_idx = y * src_yuv_image_stride_in_bytes + (x << 3);

        uint2 y0 = *((uint2 *)(&src_y_image[src_y0_idx]));
        uint2 y1 = *((uint2 *)(&src_y_image[src_y1_idx]));

        uint2 u0 = *((uint2 *)(&src_u_image[src_chroma_idx]));
        uint2 v0 = *((uint2 *)(&src_v_image[src_chroma_idx]));

        uint32_t rgb0_idx = y * dst_image_stride_in_bytes_comp + (x * 8);
        uint32_t rgb1_idx = rgb0_idx + dst_image_stride_in_bytes;

        float2 cr = make_float2( 0.0000f,  1.5748f);
        float2 cg = make_float2(-0.1873f, -0.4681f);
        float2 cb = make_float2( 1.8556f,  0.0000f);
        float3 yuv;
        DUINT6 rgb0, rgb1;
        float4 f;

        yuv = make_float3(hipUnpack0(y0.x), hipUnpack0(u0.x), hipUnpack0(v0.x));
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.x = fmaf(cr.y, yuv.z, yuv.x);
        f.y = fmaf(cg.x, yuv.y, yuv.x);
        f.y = fmaf(cg.y, yuv.z, f.y);
        f.z = fmaf(cb.x, yuv.y, yuv.x);
        yuv = make_float3(hipUnpack1(y0.x), hipUnpack1(u0.x), hipUnpack1(v0.x));
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.w = fmaf(cr.y, yuv.z, yuv.x);
        rgb0.data[0] = hipPack(f);

        f.x = fmaf(cg.x, yuv.y, yuv.x);
        f.x = fmaf(cg.y, yuv.z, f.x);
        f.y = fmaf(cb.x, yuv.y, yuv.x);
        yuv = make_float3(hipUnpack2(y0.x), hipUnpack2(u0.x), hipUnpack2(v0.x));
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.z = fmaf(cr.y, yuv.z, yuv.x);
        f.w = fmaf(cg.x, yuv.y, yuv.x);
        f.w = fmaf(cg.y, yuv.z, f.w);
        rgb0.data[1] = hipPack(f);

        f.x = fmaf(cb.x, yuv.y, yuv.x);
        yuv = make_float3(hipUnpack3(y0.x), hipUnpack3(u0.x), hipUnpack3(v0.x));
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.y = fmaf(cr.y, yuv.z, yuv.x);
        f.z = fmaf(cg.x, yuv.y, yuv.x);
        f.z = fmaf(cg.y, yuv.z, f.z);
        f.w = fmaf(cb.x, yuv.y, yuv.x);
        rgb0.data[2] = hipPack(f);

        yuv = make_float3(hipUnpack0(y0.y), hipUnpack0(u0.y), hipUnpack0(v0.y));
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.x = fmaf(cr.y, yuv.z, yuv.x);
        f.y = fmaf(cg.x, yuv.y, yuv.x);
        f.y = fmaf(cg.y, yuv.z, f.y);
        f.z = fmaf(cb.x, yuv.y, yuv.x);
        yuv = make_float3(hipUnpack1(y0.y), hipUnpack1(u0.y), hipUnpack1(v0.y));
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.w = fmaf(cr.y, yuv.z, yuv.x);
        rgb0.data[3] = hipPack(f);

        f.x = fmaf(cg.x, yuv.y, yuv.x);
        f.x = fmaf(cg.y, yuv.z, f.x);
        f.y = fmaf(cb.x, yuv.y, yuv.x);
        yuv = make_float3(hipUnpack2(y0.y), hipUnpack2(u0.y), hipUnpack2(v0.y));
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.z = fmaf(cr.y, yuv.z, yuv.x);
        f.w = fmaf(cg.x, yuv.y, yuv.x);
        f.w = fmaf(cg.y, yuv.z, f.w);
        rgb0.data[4] = hipPack(f);

        f.x = fmaf(cb.x, yuv.y, yuv.x);
        yuv = make_float3(hipUnpack3(y0.y), hipUnpack3(u0.y), hipUnpack3(v0.y));
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.y = fmaf(cr.y, yuv.z, yuv.x);
        f.z = fmaf(cg.x, yuv.y, yuv.x);
        f.z = fmaf(cg.y, yuv.z, f.z);
        f.w = fmaf(cb.x, yuv.y, yuv.x);
        rgb0.data[5] = hipPack(f);

        yuv = make_float3(hipUnpack0(y1.x), hipUnpack0(u0.x), hipUnpack0(v0.x));
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.x = fmaf(cr.y, yuv.z, yuv.x);
        f.y = fmaf(cg.x, yuv.y, yuv.x);
        f.y = fmaf(cg.y, yuv.z, f.y);
        f.z = fmaf(cb.x, yuv.y, yuv.x);
        yuv = make_float3(hipUnpack1(y1.x), hipUnpack1(u0.x), hipUnpack1(v0.x));
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.w = fmaf(cr.y, yuv.z, yuv.x);
        rgb1.data[0] = hipPack(f);

        f.x = fmaf(cg.x, yuv.y, yuv.x);
        f.x = fmaf(cg.y, yuv.z, f.x);
        f.y = fmaf(cb.x, yuv.y, yuv.x);
        yuv = make_float3(hipUnpack2(y1.x), hipUnpack2(u0.x), hipUnpack2(v0.x));
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.z = fmaf(cr.y, yuv.z, yuv.x);
        f.w = fmaf(cg.x, yuv.y, yuv.x);
        f.w = fmaf(cg.y, yuv.z, f.w);
        rgb1.data[1] = hipPack(f);

        f.x = fmaf(cb.x, yuv.y, yuv.x);
        yuv = make_float3(hipUnpack3(y1.x), hipUnpack3(u0.x), hipUnpack3(v0.x));
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.y = fmaf(cr.y, yuv.z, yuv.x);
        f.z = fmaf(cg.x, yuv.y, yuv.x);
        f.z = fmaf(cg.y, yuv.z, f.z);
        f.w = fmaf(cb.x, yuv.y, yuv.x);
        rgb1.data[2] = hipPack(f);

        yuv = make_float3(hipUnpack0(y1.y), hipUnpack0(u0.y), hipUnpack0(v0.y));
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.x = fmaf(cr.y, yuv.z, yuv.x);
        f.y = fmaf(cg.x, yuv.y, yuv.x);
        f.y = fmaf(cg.y, yuv.z, f.y);
        f.z = fmaf(cb.x, yuv.y, yuv.x);
        yuv = make_float3(hipUnpack1(y1.y), hipUnpack1(u0.y), hipUnpack1(v0.y));
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.w = fmaf(cr.y, yuv.z, yuv.x);
        rgb1.data[3] = hipPack(f);

        f.x = fmaf(cg.x, yuv.y, yuv.x);
        f.x = fmaf(cg.y, yuv.z, f.x);
        f.y = fmaf(cb.x, yuv.y, yuv.x);
        yuv = make_float3(hipUnpack2(y1.y), hipUnpack2(u0.y), hipUnpack2(v0.y));
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.z = fmaf(cr.y, yuv.z, yuv.x);
        f.w = fmaf(cg.x, yuv.y, yuv.x);
        f.w = fmaf(cg.y, yuv.z, f.w);
        rgb1.data[4] = hipPack(f);

        f.x = fmaf(cb.x, yuv.y, yuv.x);
        yuv = make_float3(hipUnpack3(y1.y), hipUnpack3(u0.y), hipUnpack3(v0.y));
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.y = fmaf(cr.y, yuv.z, yuv.x);
        f.z = fmaf(cg.x, yuv.y, yuv.x);
        f.z = fmaf(cg.y, yuv.z, f.z);
        f.w = fmaf(cb.x, yuv.y, yuv.x);
        rgb1.data[5] = hipPack(f);

        uint2 red0, red1, green0, green1, blue0, blue1;
        red0.x = hipPack(make_float4(hipUnpack0(rgb0.data[0]), hipUnpack3(rgb0.data[0]), hipUnpack2(rgb0.data[1]), hipUnpack1(rgb0.data[2])));
        red0.y = hipPack(make_float4(hipUnpack0(rgb0.data[3]), hipUnpack3(rgb0.data[3]), hipUnpack2(rgb0.data[4]), hipUnpack1(rgb0.data[5])));
        red1.x = hipPack(make_float4(hipUnpack0(rgb1.data[0]), hipUnpack3(rgb1.data[0]), hipUnpack2(rgb1.data[1]), hipUnpack1(rgb1.data[2])));
        red1.y = hipPack(make_float4(hipUnpack0(rgb1.data[3]), hipUnpack3(rgb1.data[3]), hipUnpack2(rgb1.data[4]), hipUnpack1(rgb1.data[5])));

        green0.x = hipPack(make_float4(hipUnpack1(rgb0.data[0]), hipUnpack0(rgb0.data[1]), hipUnpack3(rgb0.data[1]), hipUnpack2(rgb0.data[2])));
        green0.y = hipPack(make_float4(hipUnpack1(rgb0.data[3]), hipUnpack0(rgb0.data[4]), hipUnpack3(rgb0.data[4]), hipUnpack2(rgb0.data[5])));
        green1.x = hipPack(make_float4(hipUnpack1(rgb1.data[0]), hipUnpack0(rgb1.data[1]), hipUnpack3(rgb1.data[1]), hipUnpack2(rgb1.data[2])));
        green1.y = hipPack(make_float4(hipUnpack1(rgb1.data[3]), hipUnpack0(rgb1.data[4]), hipUnpack3(rgb1.data[4]), hipUnpack2(rgb1.data[5])));

        blue0.x = hipPack(make_float4(hipUnpack2(rgb0.data[0]), hipUnpack1(rgb0.data[1]), hipUnpack0(rgb0.data[2]), hipUnpack3(rgb0.data[2])));
        blue0.y = hipPack(make_float4(hipUnpack2(rgb0.data[3]), hipUnpack1(rgb0.data[4]), hipUnpack0(rgb0.data[5]), hipUnpack3(rgb0.data[5])));
        blue1.x = hipPack(make_float4(hipUnpack2(rgb1.data[0]), hipUnpack1(rgb1.data[1]), hipUnpack0(rgb1.data[2]), hipUnpack3(rgb1.data[2])));
        blue1.y = hipPack(make_float4(hipUnpack2(rgb1.data[3]), hipUnpack1(rgb1.data[4]), hipUnpack0(rgb1.data[5]), hipUnpack3(rgb1.data[5])));

        *((uint2 *)(&dst_image_r[rgb0_idx])) = red0;
        *((uint2 *)(&dst_image_r[rgb1_idx])) = red1;
        *((uint2 *)(&dst_image_g[rgb0_idx])) = green0;
        *((uint2 *)(&dst_image_g[rgb1_idx])) = green1;
        *((uint2 *)(&dst_image_b[rgb0_idx])) = blue0;
        *((uint2 *)(&dst_image_b[rgb1_idx])) = blue1;
    }
}


/**
 * @brief Converts YUV440 image to RGB planar format.
 *
 * This function takes a YUV440 image and converts it to RGB planar format using the ColorConvertYUV444ToRGBPlanarKernel HIP kernel.
 *
 * @param stream The HIP stream to be used for the kernel launch.
 * @param dst_width The width of the destination RGB image.
 * @param dst_height The height of the destination RGB image.
 * @param dst_image_r Pointer to the destination red channel image.
 * @param dst_image_g Pointer to the destination green channel image.
 * @param dst_image_b Pointer to the destination blue channel image.
 * @param dst_image_stride_in_bytes The stride (in bytes) of the destination image.
 * @param src_yuv_image Pointer to the source YUV image.
 * @param src_yuv_image_stride_in_bytes The stride (in bytes) of the source YUV image.
 * @param src_u_image_offset The offset (in bytes) to the U channel in the source YUV image.
 */
void ColorConvertYUV440ToRGBPlanar(hipStream_t stream, uint32_t dst_width, uint32_t dst_height,
    uint8_t *dst_image_r, uint8_t *dst_image_g, uint8_t *dst_image_b, uint32_t dst_image_stride_in_bytes, const uint8_t *src_yuv_image,
    uint32_t src_yuv_image_stride_in_bytes, uint32_t src_u_image_offset, uint32_t src_v_image_offset) {

    int32_t local_threads_x = 16;
    int32_t local_threads_y = 4;
    int32_t global_threads_x = (dst_width + 7) >> 3;
    int32_t global_threads_y = (dst_height + 1) >> 1;

    uint32_t dst_width_comp = (dst_width + 7) / 8;
    uint32_t dst_height_comp = (dst_height + 1) / 2;
    uint32_t dst_image_stride_in_bytes_comp = dst_image_stride_in_bytes * 2;
    uint32_t src_yuv_image_stride_in_bytes_comp = src_yuv_image_stride_in_bytes * 2;

    ColorConvertYUV440ToRGBPlanarKernel<<<dim3(ceil(static_cast<float>(global_threads_x) / local_threads_x), ceil(static_cast<float>(global_threads_y) / local_threads_y)),
                        dim3(local_threads_x, local_threads_y), 0, stream>>>(dst_image_r, dst_image_g, dst_image_b,
                        dst_image_stride_in_bytes, dst_image_stride_in_bytes_comp, src_yuv_image, src_yuv_image + src_u_image_offset,
                        src_yuv_image + src_v_image_offset, src_yuv_image_stride_in_bytes,
                        dst_width_comp, dst_height_comp, src_yuv_image_stride_in_bytes_comp);
}

__global__ void ColorConvertYUYVToRGBKernel(
    uint8_t *dst_image, uint32_t dst_image_stride_in_bytes, uint32_t dst_image_stride_in_bytes_comp,
    const uint8_t *src_image, uint32_t src_image_stride_in_bytes, uint32_t src_image_stride_in_bytes_comp,
    uint32_t dst_width_comp, uint32_t dst_height_comp) {

    uint32_t x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    uint32_t y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    if ((x < dst_width_comp) && (y < dst_height_comp)) {
        uint32_t l0_idx = y * src_image_stride_in_bytes_comp + (x << 4);
        uint32_t l1_idx = l0_idx + src_image_stride_in_bytes;
        uint4 l0 = *((uint4 *)(&src_image[l0_idx]));
        uint4 l1 = *((uint4 *)(&src_image[l1_idx]));

        uint32_t rgb0_idx = y * dst_image_stride_in_bytes_comp + (x * 24);
        uint32_t rgb1_idx = rgb0_idx + dst_image_stride_in_bytes;

        float4 f;

        uint2 py0, py1;
        uint2 pu0, pu1;
        uint2 pv0, pv1;

        py0.x = hipPack(make_float4(hipUnpack0(l0.x), hipUnpack2(l0.x), hipUnpack0(l0.y), hipUnpack2(l0.y)));
        py0.y = hipPack(make_float4(hipUnpack0(l0.z), hipUnpack2(l0.z), hipUnpack0(l0.w), hipUnpack2(l0.w)));
        py1.x = hipPack(make_float4(hipUnpack0(l1.x), hipUnpack2(l1.x), hipUnpack0(l1.y), hipUnpack2(l1.y)));
        py1.y = hipPack(make_float4(hipUnpack0(l1.z), hipUnpack2(l1.z), hipUnpack0(l1.w), hipUnpack2(l1.w)));
        pu0.x = hipPack(make_float4(hipUnpack1(l0.x), hipUnpack1(l0.x), hipUnpack1(l0.y), hipUnpack1(l0.y)));
        pu0.y = hipPack(make_float4(hipUnpack1(l0.z), hipUnpack1(l0.z), hipUnpack1(l0.w), hipUnpack1(l0.w)));
        pu1.x = hipPack(make_float4(hipUnpack1(l1.x), hipUnpack1(l1.x), hipUnpack1(l1.y), hipUnpack1(l1.y)));
        pu1.y = hipPack(make_float4(hipUnpack1(l1.z), hipUnpack1(l1.z), hipUnpack1(l1.w), hipUnpack1(l1.w)));
        pv0.x = hipPack(make_float4(hipUnpack3(l0.x), hipUnpack3(l0.x), hipUnpack3(l0.y), hipUnpack3(l0.y)));
        pv0.y = hipPack(make_float4(hipUnpack3(l0.z), hipUnpack3(l0.z), hipUnpack3(l0.w), hipUnpack3(l0.w)));
        pv1.x = hipPack(make_float4(hipUnpack3(l1.x), hipUnpack3(l1.x), hipUnpack3(l1.y), hipUnpack3(l1.y)));
        pv1.y = hipPack(make_float4(hipUnpack3(l1.z), hipUnpack3(l1.z), hipUnpack3(l1.w), hipUnpack3(l1.w)));

        float2 cr = make_float2( 0.0000f,  1.5748f);
        float2 cg = make_float2(-0.1873f, -0.4681f);
        float2 cb = make_float2( 1.8556f,  0.0000f);
        float3 yuv;
        DUINT6 prgb0, prgb1;

        yuv = make_float3(hipUnpack0(py0.x), hipUnpack0(pu0.x), hipUnpack0(pv0.x));
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.x = fmaf(cr.y, yuv.z, yuv.x);
        f.y = fmaf(cg.x, yuv.y, yuv.x);
        f.y = fmaf(cg.y, yuv.z, f.y);
        f.z = fmaf(cb.x, yuv.y, yuv.x);
        yuv = make_float3(hipUnpack1(py0.x), hipUnpack1(pu0.x), hipUnpack1(pv0.x));
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.w = fmaf(cr.y, yuv.z, yuv.x);
        prgb0.data[0] = hipPack(f);

        f.x = fmaf(cg.x, yuv.y, yuv.x);
        f.x = fmaf(cg.y, yuv.z, f.x);
        f.y = fmaf(cb.x, yuv.y, yuv.x);
        yuv = make_float3(hipUnpack2(py0.x), hipUnpack2(pu0.x), hipUnpack2(pv0.x));
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.z = fmaf(cr.y, yuv.z, yuv.x);
        f.w = fmaf(cg.x, yuv.y, yuv.x);
        f.w = fmaf(cg.y, yuv.z, f.w);
        prgb0.data[1] = hipPack(f);

        f.x = fmaf(cb.x, yuv.y, yuv.x);
        yuv = make_float3(hipUnpack3(py0.x), hipUnpack3(pu0.x), hipUnpack3(pv0.x));
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.y = fmaf(cr.y, yuv.z, yuv.x);
        f.z = fmaf(cg.x, yuv.y, yuv.x);
        f.z = fmaf(cg.y, yuv.z, f.z);
        f.w = fmaf(cb.x, yuv.y, yuv.x);
        prgb0.data[2] = hipPack(f);

        yuv = make_float3(hipUnpack0(py0.y), hipUnpack0(pu0.y), hipUnpack0(pv0.y));
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.x = fmaf(cr.y, yuv.z, yuv.x);
        f.y = fmaf(cg.x, yuv.y, yuv.x);
        f.y = fmaf(cg.y, yuv.z, f.y);
        f.z = fmaf(cb.x, yuv.y, yuv.x);
        yuv = make_float3(hipUnpack1(py0.y), hipUnpack1(pu0.y), hipUnpack1(pv0.y));
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.w = fmaf(cr.y, yuv.z, yuv.x);
        prgb0.data[3] = hipPack(f);

        f.x = fmaf(cg.x, yuv.y, yuv.x);
        f.x = fmaf(cg.y, yuv.z, f.x);
        f.y = fmaf(cb.x, yuv.y, yuv.x);
        yuv = make_float3(hipUnpack2(py0.y), hipUnpack2(pu0.y), hipUnpack2(pv0.y));
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.z = fmaf(cr.y, yuv.z, yuv.x);
        f.w = fmaf(cg.x, yuv.y, yuv.x);
        f.w = fmaf(cg.y, yuv.z, f.w);
        prgb0.data[4] = hipPack(f);

        f.x = fmaf(cb.x, yuv.y, yuv.x);
        yuv = make_float3(hipUnpack3(py0.y), hipUnpack3(pu0.y), hipUnpack3(pv0.y));
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.y = fmaf(cr.y, yuv.z, yuv.x);
        f.z = fmaf(cg.x, yuv.y, yuv.x);
        f.z = fmaf(cg.y, yuv.z, f.z);
        f.w = fmaf(cb.x, yuv.y, yuv.x);
        prgb0.data[5] = hipPack(f);

        yuv = make_float3(hipUnpack0(py1.x), hipUnpack0(pu1.x), hipUnpack0(pv1.x));
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.x = fmaf(cr.y, yuv.z, yuv.x);
        f.y = fmaf(cg.x, yuv.y, yuv.x);
        f.y = fmaf(cg.y, yuv.z, f.y);
        f.z = fmaf(cb.x, yuv.y, yuv.x);
        yuv = make_float3(hipUnpack1(py1.x), hipUnpack1(pu1.x), hipUnpack1(pv1.x));
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.w = fmaf(cr.y, yuv.z, yuv.x);
        prgb1.data[0] = hipPack(f);

        f.x = fmaf(cg.x, yuv.y, yuv.x);
        f.x = fmaf(cg.y, yuv.z, f.x);
        f.y = fmaf(cb.x, yuv.y, yuv.x);
        yuv = make_float3(hipUnpack2(py1.x), hipUnpack2(pu1.x), hipUnpack2(pv1.x));
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.z = fmaf(cr.y, yuv.z, yuv.x);
        f.w = fmaf(cg.x, yuv.y, yuv.x);
        f.w = fmaf(cg.y, yuv.z, f.w);
        prgb1.data[1] = hipPack(f);

        f.x = fmaf(cb.x, yuv.y, yuv.x);
        yuv = make_float3(hipUnpack3(py1.x), hipUnpack3(pu1.x), hipUnpack3(pv1.x));
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.y = fmaf(cr.y, yuv.z, yuv.x);
        f.z = fmaf(cg.x, yuv.y, yuv.x);
        f.z = fmaf(cg.y, yuv.z, f.z);
        f.w = fmaf(cb.x, yuv.y, yuv.x);
        prgb1.data[2] = hipPack(f);

        yuv = make_float3(hipUnpack0(py1.y), hipUnpack0(pu1.y), hipUnpack0(pv1.y));
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.x = fmaf(cr.y, yuv.z, yuv.x);
        f.y = fmaf(cg.x, yuv.y, yuv.x);
        f.y = fmaf(cg.y, yuv.z, f.y);
        f.z = fmaf(cb.x, yuv.y, yuv.x);
        yuv = make_float3(hipUnpack1(py1.y), hipUnpack1(pu1.y), hipUnpack1(pv1.y));
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.w = fmaf(cr.y, yuv.z, yuv.x);
        prgb1.data[3] = hipPack(f);

        f.x = fmaf(cg.x, yuv.y, yuv.x);
        f.x = fmaf(cg.y, yuv.z, f.x);
        f.y = fmaf(cb.x, yuv.y, yuv.x);
        yuv = make_float3(hipUnpack2(py1.y), hipUnpack2(pu1.y), hipUnpack2(pv1.y));
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.z = fmaf(cr.y, yuv.z, yuv.x);
        f.w = fmaf(cg.x, yuv.y, yuv.x);
        f.w = fmaf(cg.y, yuv.z, f.w);
        prgb1.data[4] = hipPack(f);

        f.x = fmaf(cb.x, yuv.y, yuv.x);
        yuv = make_float3(hipUnpack3(py1.y), hipUnpack3(pu1.y), hipUnpack3(pv1.y));
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.y = fmaf(cr.y, yuv.z, yuv.x);
        f.z = fmaf(cg.x, yuv.y, yuv.x);
        f.z = fmaf(cg.y, yuv.z, f.z);
        f.w = fmaf(cb.x, yuv.y, yuv.x);
        prgb1.data[5] = hipPack(f);

        *((DUINT6 *)(&dst_image[rgb0_idx])) = prgb0;
        *((DUINT6 *)(&dst_image[rgb1_idx])) = prgb1;
    }
}

/**
 * @brief Converts YUYV image format to RGB image format.
 *
 * This function takes a YUYV image and converts it to RGB image format using ColorConvertYUYVToRGBKernel HIP kernel
 *
 * @param stream The HIP stream to associate the kernel launch with.
 * @param dst_width The width of the destination RGB image.
 * @param dst_height The height of the destination RGB image.
 * @param dst_image Pointer to the destination RGB image buffer.
 * @param dst_image_stride_in_bytes The stride (in bytes) of the destination RGB image buffer.
 * @param src_image Pointer to the source YUYV image buffer.
 * @param src_image_stride_in_bytes The stride (in bytes) of the source YUYV image buffer.
 */
void ColorConvertYUYVToRGB(hipStream_t stream, uint32_t dst_width, uint32_t dst_height,
    uint8_t *dst_image, uint32_t dst_image_stride_in_bytes,
    const uint8_t *src_image, uint32_t src_image_stride_in_bytes) {
    int32_t local_threads_x = 16;
    int32_t local_threads_y = 4;
    int32_t global_threads_x = (dst_width + 7) >> 3;
    int32_t global_threads_y = (dst_height + 1) >> 1;

    uint32_t dst_width_comp = (dst_width + 7) / 8;
    uint32_t dst_height_comp = (dst_height + 1) / 2;
    uint32_t dst_image_stride_in_bytes_comp = dst_image_stride_in_bytes * 2;
    uint32_t src_image_stride_in_bytes_comp = src_image_stride_in_bytes * 2;

    ColorConvertYUYVToRGBKernel<<<dim3(ceil(static_cast<float>(global_threads_x) / local_threads_x), ceil(static_cast<float>(global_threads_y) / local_threads_y)),
                                   dim3(local_threads_x, local_threads_y), 0, stream>>>(dst_image,
                                   dst_image_stride_in_bytes, dst_image_stride_in_bytes_comp, src_image, src_image_stride_in_bytes,
                                   src_image_stride_in_bytes_comp, dst_width_comp, dst_height_comp);
}

__global__ void ColorConvertYUYVToRGBPlanarKernel(
    uint8_t *dst_image_r, uint8_t *dst_image_g, uint8_t *dst_image_b, uint32_t dst_image_stride_in_bytes, uint32_t dst_image_stride_in_bytes_comp,
    const uint8_t *src_image, uint32_t src_image_stride_in_bytes, uint32_t src_image_stride_in_bytes_comp,
    uint32_t dst_width_comp, uint32_t dst_height_comp) {
    uint32_t x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    uint32_t y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    if ((x < dst_width_comp) && (y < dst_height_comp)) {
        uint32_t l0_idx = y * src_image_stride_in_bytes_comp + (x << 4);
        uint32_t l1_idx = l0_idx + src_image_stride_in_bytes;
        uint4 l0 = *((uint4 *)(&src_image[l0_idx]));
        uint4 l1 = *((uint4 *)(&src_image[l1_idx]));

        uint32_t rgb0_idx = y * dst_image_stride_in_bytes_comp + (x * 8);
        uint32_t rgb1_idx = rgb0_idx + dst_image_stride_in_bytes;

        float4 f;

        uint2 py0, py1;
        uint2 pu0, pu1;
        uint2 pv0, pv1;

        py0.x = hipPack(make_float4(hipUnpack0(l0.x), hipUnpack2(l0.x), hipUnpack0(l0.y), hipUnpack2(l0.y)));
        py0.y = hipPack(make_float4(hipUnpack0(l0.z), hipUnpack2(l0.z), hipUnpack0(l0.w), hipUnpack2(l0.w)));
        py1.x = hipPack(make_float4(hipUnpack0(l1.x), hipUnpack2(l1.x), hipUnpack0(l1.y), hipUnpack2(l1.y)));
        py1.y = hipPack(make_float4(hipUnpack0(l1.z), hipUnpack2(l1.z), hipUnpack0(l1.w), hipUnpack2(l1.w)));
        pu0.x = hipPack(make_float4(hipUnpack1(l0.x), hipUnpack1(l0.x), hipUnpack1(l0.y), hipUnpack1(l0.y)));
        pu0.y = hipPack(make_float4(hipUnpack1(l0.z), hipUnpack1(l0.z), hipUnpack1(l0.w), hipUnpack1(l0.w)));
        pu1.x = hipPack(make_float4(hipUnpack1(l1.x), hipUnpack1(l1.x), hipUnpack1(l1.y), hipUnpack1(l1.y)));
        pu1.y = hipPack(make_float4(hipUnpack1(l1.z), hipUnpack1(l1.z), hipUnpack1(l1.w), hipUnpack1(l1.w)));
        pv0.x = hipPack(make_float4(hipUnpack3(l0.x), hipUnpack3(l0.x), hipUnpack3(l0.y), hipUnpack3(l0.y)));
        pv0.y = hipPack(make_float4(hipUnpack3(l0.z), hipUnpack3(l0.z), hipUnpack3(l0.w), hipUnpack3(l0.w)));
        pv1.x = hipPack(make_float4(hipUnpack3(l1.x), hipUnpack3(l1.x), hipUnpack3(l1.y), hipUnpack3(l1.y)));
        pv1.y = hipPack(make_float4(hipUnpack3(l1.z), hipUnpack3(l1.z), hipUnpack3(l1.w), hipUnpack3(l1.w)));

        float2 cr = make_float2( 0.0000f,  1.5748f);
        float2 cg = make_float2(-0.1873f, -0.4681f);
        float2 cb = make_float2( 1.8556f,  0.0000f);
        float3 yuv;
        DUINT6 prgb0, prgb1;

        yuv = make_float3(hipUnpack0(py0.x), hipUnpack0(pu0.x), hipUnpack0(pv0.x));
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.x = fmaf(cr.y, yuv.z, yuv.x);
        f.y = fmaf(cg.x, yuv.y, yuv.x);
        f.y = fmaf(cg.y, yuv.z, f.y);
        f.z = fmaf(cb.x, yuv.y, yuv.x);
        yuv = make_float3(hipUnpack1(py0.x), hipUnpack1(pu0.x), hipUnpack1(pv0.x));
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.w = fmaf(cr.y, yuv.z, yuv.x);
        prgb0.data[0] = hipPack(f);

        f.x = fmaf(cg.x, yuv.y, yuv.x);
        f.x = fmaf(cg.y, yuv.z, f.x);
        f.y = fmaf(cb.x, yuv.y, yuv.x);
        yuv = make_float3(hipUnpack2(py0.x), hipUnpack2(pu0.x), hipUnpack2(pv0.x));
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.z = fmaf(cr.y, yuv.z, yuv.x);
        f.w = fmaf(cg.x, yuv.y, yuv.x);
        f.w = fmaf(cg.y, yuv.z, f.w);
        prgb0.data[1] = hipPack(f);

        f.x = fmaf(cb.x, yuv.y, yuv.x);
        yuv = make_float3(hipUnpack3(py0.x), hipUnpack3(pu0.x), hipUnpack3(pv0.x));
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.y = fmaf(cr.y, yuv.z, yuv.x);
        f.z = fmaf(cg.x, yuv.y, yuv.x);
        f.z = fmaf(cg.y, yuv.z, f.z);
        f.w = fmaf(cb.x, yuv.y, yuv.x);
        prgb0.data[2] = hipPack(f);

        yuv = make_float3(hipUnpack0(py0.y), hipUnpack0(pu0.y), hipUnpack0(pv0.y));
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.x = fmaf(cr.y, yuv.z, yuv.x);
        f.y = fmaf(cg.x, yuv.y, yuv.x);
        f.y = fmaf(cg.y, yuv.z, f.y);
        f.z = fmaf(cb.x, yuv.y, yuv.x);
        yuv = make_float3(hipUnpack1(py0.y), hipUnpack1(pu0.y), hipUnpack1(pv0.y));
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.w = fmaf(cr.y, yuv.z, yuv.x);
        prgb0.data[3] = hipPack(f);

        f.x = fmaf(cg.x, yuv.y, yuv.x);
        f.x = fmaf(cg.y, yuv.z, f.x);
        f.y = fmaf(cb.x, yuv.y, yuv.x);
        yuv = make_float3(hipUnpack2(py0.y), hipUnpack2(pu0.y), hipUnpack2(pv0.y));
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.z = fmaf(cr.y, yuv.z, yuv.x);
        f.w = fmaf(cg.x, yuv.y, yuv.x);
        f.w = fmaf(cg.y, yuv.z, f.w);
        prgb0.data[4] = hipPack(f);

        f.x = fmaf(cb.x, yuv.y, yuv.x);
        yuv = make_float3(hipUnpack3(py0.y), hipUnpack3(pu0.y), hipUnpack3(pv0.y));
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.y = fmaf(cr.y, yuv.z, yuv.x);
        f.z = fmaf(cg.x, yuv.y, yuv.x);
        f.z = fmaf(cg.y, yuv.z, f.z);
        f.w = fmaf(cb.x, yuv.y, yuv.x);
        prgb0.data[5] = hipPack(f);

        yuv = make_float3(hipUnpack0(py1.x), hipUnpack0(pu1.x), hipUnpack0(pv1.x));
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.x = fmaf(cr.y, yuv.z, yuv.x);
        f.y = fmaf(cg.x, yuv.y, yuv.x);
        f.y = fmaf(cg.y, yuv.z, f.y);
        f.z = fmaf(cb.x, yuv.y, yuv.x);
        yuv = make_float3(hipUnpack1(py1.x), hipUnpack1(pu1.x), hipUnpack1(pv1.x));
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.w = fmaf(cr.y, yuv.z, yuv.x);
        prgb1.data[0] = hipPack(f);

        f.x = fmaf(cg.x, yuv.y, yuv.x);
        f.x = fmaf(cg.y, yuv.z, f.x);
        f.y = fmaf(cb.x, yuv.y, yuv.x);
        yuv = make_float3(hipUnpack2(py1.x), hipUnpack2(pu1.x), hipUnpack2(pv1.x));
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.z = fmaf(cr.y, yuv.z, yuv.x);
        f.w = fmaf(cg.x, yuv.y, yuv.x);
        f.w = fmaf(cg.y, yuv.z, f.w);
        prgb1.data[1] = hipPack(f);

        f.x = fmaf(cb.x, yuv.y, yuv.x);
        yuv = make_float3(hipUnpack3(py1.x), hipUnpack3(pu1.x), hipUnpack3(pv1.x));
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.y = fmaf(cr.y, yuv.z, yuv.x);
        f.z = fmaf(cg.x, yuv.y, yuv.x);
        f.z = fmaf(cg.y, yuv.z, f.z);
        f.w = fmaf(cb.x, yuv.y, yuv.x);
        prgb1.data[2] = hipPack(f);

        yuv = make_float3(hipUnpack0(py1.y), hipUnpack0(pu1.y), hipUnpack0(pv1.y));
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.x = fmaf(cr.y, yuv.z, yuv.x);
        f.y = fmaf(cg.x, yuv.y, yuv.x);
        f.y = fmaf(cg.y, yuv.z, f.y);
        f.z = fmaf(cb.x, yuv.y, yuv.x);
        yuv = make_float3(hipUnpack1(py1.y), hipUnpack1(pu1.y), hipUnpack1(pv1.y));
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.w = fmaf(cr.y, yuv.z, yuv.x);
        prgb1.data[3] = hipPack(f);

        f.x = fmaf(cg.x, yuv.y, yuv.x);
        f.x = fmaf(cg.y, yuv.z, f.x);
        f.y = fmaf(cb.x, yuv.y, yuv.x);
        yuv = make_float3(hipUnpack2(py1.y), hipUnpack2(pu1.y), hipUnpack2(pv1.y));
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.z = fmaf(cr.y, yuv.z, yuv.x);
        f.w = fmaf(cg.x, yuv.y, yuv.x);
        f.w = fmaf(cg.y, yuv.z, f.w);
        prgb1.data[4] = hipPack(f);

        f.x = fmaf(cb.x, yuv.y, yuv.x);
        yuv = make_float3(hipUnpack3(py1.y), hipUnpack3(pu1.y), hipUnpack3(pv1.y));
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.y = fmaf(cr.y, yuv.z, yuv.x);
        f.z = fmaf(cg.x, yuv.y, yuv.x);
        f.z = fmaf(cg.y, yuv.z, f.z);
        f.w = fmaf(cb.x, yuv.y, yuv.x);
        prgb1.data[5] = hipPack(f);

        uint2 red0, red1, green0, green1, blue0, blue1;
        red0.x = hipPack(make_float4(hipUnpack0(prgb0.data[0]), hipUnpack3(prgb0.data[0]), hipUnpack2(prgb0.data[1]), hipUnpack1(prgb0.data[2])));
        red0.y = hipPack(make_float4(hipUnpack0(prgb0.data[3]), hipUnpack3(prgb0.data[3]), hipUnpack2(prgb0.data[4]), hipUnpack1(prgb0.data[5])));
        red1.x = hipPack(make_float4(hipUnpack0(prgb1.data[0]), hipUnpack3(prgb1.data[0]), hipUnpack2(prgb1.data[1]), hipUnpack1(prgb1.data[2])));
        red1.y = hipPack(make_float4(hipUnpack0(prgb1.data[3]), hipUnpack3(prgb1.data[3]), hipUnpack2(prgb1.data[4]), hipUnpack1(prgb1.data[5])));

        green0.x = hipPack(make_float4(hipUnpack1(prgb0.data[0]), hipUnpack0(prgb0.data[1]), hipUnpack3(prgb0.data[1]), hipUnpack2(prgb0.data[2])));
        green0.y = hipPack(make_float4(hipUnpack1(prgb0.data[3]), hipUnpack0(prgb0.data[4]), hipUnpack3(prgb0.data[4]), hipUnpack2(prgb0.data[5])));
        green1.x = hipPack(make_float4(hipUnpack1(prgb1.data[0]), hipUnpack0(prgb1.data[1]), hipUnpack3(prgb1.data[1]), hipUnpack2(prgb1.data[2])));
        green1.y = hipPack(make_float4(hipUnpack1(prgb1.data[3]), hipUnpack0(prgb1.data[4]), hipUnpack3(prgb1.data[4]), hipUnpack2(prgb1.data[5])));

        blue0.x = hipPack(make_float4(hipUnpack2(prgb0.data[0]), hipUnpack1(prgb0.data[1]), hipUnpack0(prgb0.data[2]), hipUnpack3(prgb0.data[2])));
        blue0.y = hipPack(make_float4(hipUnpack2(prgb0.data[3]), hipUnpack1(prgb0.data[4]), hipUnpack0(prgb0.data[5]), hipUnpack3(prgb0.data[5])));
        blue1.x = hipPack(make_float4(hipUnpack2(prgb1.data[0]), hipUnpack1(prgb1.data[1]), hipUnpack0(prgb1.data[2]), hipUnpack3(prgb1.data[2])));
        blue1.y = hipPack(make_float4(hipUnpack2(prgb1.data[3]), hipUnpack1(prgb1.data[4]), hipUnpack0(prgb1.data[5]), hipUnpack3(prgb1.data[5])));

        *((uint2 *)(&dst_image_r[rgb0_idx])) = red0;
        *((uint2 *)(&dst_image_r[rgb1_idx])) = red1;
        *((uint2 *)(&dst_image_g[rgb0_idx])) = green0;
        *((uint2 *)(&dst_image_g[rgb1_idx])) = green1;
        *((uint2 *)(&dst_image_b[rgb0_idx])) = blue0;
        *((uint2 *)(&dst_image_b[rgb1_idx])) = blue1;
    }
}

/**
 * @brief Converts YUYV image format to RGB planar image format.
 *
 * This function takes a YUYV image and converts it to RGB planar image format
 * using ColorConvertYUYVToRGBPlanarKernel HIP kernel.
 * The YUYV image is represented as a packed format where each pixel consists of
 * one luminance (Y) component and two chrominance (U and V) components. The RGB
 * planar image format represents each color component (R, G, and B) in separate
 * planes.
 *
 * @param stream The HIP stream to execute the kernel on.
 * @param dst_width The width of the destination RGB planar image.
 * @param dst_height The height of the destination RGB planar image.
 * @param dst_image_r Pointer to the destination red (R) component image plane.
 * @param dst_image_g Pointer to the destination green (G) component image plane.
 * @param dst_image_b Pointer to the destination blue (B) component image plane.
 * @param dst_image_stride_in_bytes The stride (in bytes) between consecutive rows of the destination image.
 * @param src_image Pointer to the source YUYV image.
 * @param src_image_stride_in_bytes The stride (in bytes) between consecutive rows of the source image.
 */
void ColorConvertYUYVToRGBPlanar(hipStream_t stream, uint32_t dst_width, uint32_t dst_height,
    uint8_t *dst_image_r, uint8_t *dst_image_g, uint8_t *dst_image_b, uint32_t dst_image_stride_in_bytes,
    const uint8_t *src_image, uint32_t src_image_stride_in_bytes) {

    int32_t local_threads_x = 16;
    int32_t local_threads_y = 4;
    int32_t global_threads_x = (dst_width + 7) >> 3;
    int32_t global_threads_y = (dst_height + 1) >> 1;

    uint32_t dst_width_comp = (dst_width + 7) / 8;
    uint32_t dst_height_comp = (dst_height + 1) / 2;
    uint32_t dst_image_stride_in_bytes_comp = dst_image_stride_in_bytes * 2;
    uint32_t src_image_stride_in_bytes_comp = src_image_stride_in_bytes * 2;

    ColorConvertYUYVToRGBPlanarKernel<<<dim3(ceil(static_cast<float>(global_threads_x) / local_threads_x), ceil(static_cast<float>(global_threads_y) / local_threads_y)),
                                   dim3(local_threads_x, local_threads_y), 0, stream>>>(dst_image_r, dst_image_g, dst_image_b,
                                   dst_image_stride_in_bytes, dst_image_stride_in_bytes_comp, src_image, src_image_stride_in_bytes,
                                   src_image_stride_in_bytes_comp, dst_width_comp, dst_height_comp);
}

__global__ void ColorConvertNV12ToRGBKernel(
    uint8_t *dst_image, uint32_t dst_image_stride_in_bytes, uint32_t dst_image_stride_in_bytes_comp,
    const uint8_t *src_luma_image, uint32_t src_luma_image_stride_in_bytes,
    const uint8_t *src_chroma_image, uint32_t src_chroma_image_stride_in_bytes,
    uint32_t dst_width_comp, uint32_t dst_height_comp, uint32_t src_luma_image_stride_in_bytes_comp) {

    uint32_t x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    uint32_t y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    if ((x < dst_width_comp) && (y < dst_height_comp)) {
        uint32_t src_y0_idx = y * src_luma_image_stride_in_bytes_comp + (x << 3);
        uint32_t src_y1_idx = src_y0_idx + src_luma_image_stride_in_bytes;
        uint32_t src_uv_idx = y * src_chroma_image_stride_in_bytes + (x << 3);
        uint2 y0 = *((uint2 *)(&src_luma_image[src_y0_idx]));
        uint2 y1 = *((uint2 *)(&src_luma_image[src_y1_idx]));
        uint2 uv = *((uint2 *)(&src_chroma_image[src_uv_idx]));

        uint32_t rgb0_idx = y * dst_image_stride_in_bytes_comp + (x * 24);
        uint32_t rgb1_idx = rgb0_idx + dst_image_stride_in_bytes;

        float4 f;
        uint2 u0, u1;
        uint2 v0, v1;

        f.x = hipUnpack0(uv.x);
        f.y = f.x;
        f.z = hipUnpack2(uv.x);
        f.w = f.z;
        u0.x = hipPack(f);

        f.x = hipUnpack0(uv.y);
        f.y = f.x;
        f.z = hipUnpack2(uv.y);
        f.w = f.z;
        u0.y = hipPack(f);

        u1.x = u0.x;
        u1.y = u0.y;

        f.x = hipUnpack1(uv.x);
        f.y = f.x;
        f.z = hipUnpack3(uv.x);
        f.w = f.z;
        v0.x = hipPack(f);

        f.x = hipUnpack1(uv.y);
        f.y = f.x;
        f.z = hipUnpack3(uv.y);
        f.w = f.z;
        v0.y = hipPack(f);

        v1.x = v0.x;
        v1.y = v0.y;

        float2 cr = make_float2( 0.0000f,  1.5748f);
        float2 cg = make_float2(-0.1873f, -0.4681f);
        float2 cb = make_float2( 1.8556f,  0.0000f);
        float3 yuv;
        DUINT6 rgb0, rgb1;

        yuv = make_float3(hipUnpack0(y0.x), hipUnpack0(u0.x), hipUnpack0(v0.x));
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.x = fmaf(cr.y, yuv.z, yuv.x);
        f.y = fmaf(cg.x, yuv.y, yuv.x);
        f.y = fmaf(cg.y, yuv.z, f.y);
        f.z = fmaf(cb.x, yuv.y, yuv.x);
        yuv = make_float3(hipUnpack1(y0.x), hipUnpack1(u0.x), hipUnpack1(v0.x));
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.w = fmaf(cr.y, yuv.z, yuv.x);
        rgb0.data[0] = hipPack(f);

        f.x = fmaf(cg.x, yuv.y, yuv.x);
        f.x = fmaf(cg.y, yuv.z, f.x);
        f.y = fmaf(cb.x, yuv.y, yuv.x);
        yuv = make_float3(hipUnpack2(y0.x), hipUnpack2(u0.x), hipUnpack2(v0.x));
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.z = fmaf(cr.y, yuv.z, yuv.x);
        f.w = fmaf(cg.x, yuv.y, yuv.x);
        f.w = fmaf(cg.y, yuv.z, f.w);
        rgb0.data[1] = hipPack(f);

        f.x = fmaf(cb.x, yuv.y, yuv.x);
        yuv = make_float3(hipUnpack3(y0.x), hipUnpack3(u0.x), hipUnpack3(v0.x));
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.y = fmaf(cr.y, yuv.z, yuv.x);
        f.z = fmaf(cg.x, yuv.y, yuv.x);
        f.z = fmaf(cg.y, yuv.z, f.z);
        f.w = fmaf(cb.x, yuv.y, yuv.x);
        rgb0.data[2] = hipPack(f);

        yuv = make_float3(hipUnpack0(y0.y), hipUnpack0(u0.y), hipUnpack0(v0.y));
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.x = fmaf(cr.y, yuv.z, yuv.x);
        f.y = fmaf(cg.x, yuv.y, yuv.x);
        f.y = fmaf(cg.y, yuv.z, f.y);
        f.z = fmaf(cb.x, yuv.y, yuv.x);
        yuv = make_float3(hipUnpack1(y0.y), hipUnpack1(u0.y), hipUnpack1(v0.y));
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.w = fmaf(cr.y, yuv.z, yuv.x);
        rgb0.data[3] = hipPack(f);

        f.x = fmaf(cg.x, yuv.y, yuv.x);
        f.x = fmaf(cg.y, yuv.z, f.x);
        f.y = fmaf(cb.x, yuv.y, yuv.x);
        yuv = make_float3(hipUnpack2(y0.y), hipUnpack2(u0.y), hipUnpack2(v0.y));
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.z = fmaf(cr.y, yuv.z, yuv.x);
        f.w = fmaf(cg.x, yuv.y, yuv.x);
        f.w = fmaf(cg.y, yuv.z, f.w);
        rgb0.data[4] = hipPack(f);

        f.x = fmaf(cb.x, yuv.y, yuv.x);
        yuv = make_float3(hipUnpack3(y0.y), hipUnpack3(u0.y), hipUnpack3(v0.y));
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.y = fmaf(cr.y, yuv.z, yuv.x);
        f.z = fmaf(cg.x, yuv.y, yuv.x);
        f.z = fmaf(cg.y, yuv.z, f.z);
        f.w = fmaf(cb.x, yuv.y, yuv.x);
        rgb0.data[5] = hipPack(f);

        yuv = make_float3(hipUnpack0(y1.x), hipUnpack0(u1.x), hipUnpack0(v1.x));
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.x = fmaf(cr.y, yuv.z, yuv.x);
        f.y = fmaf(cg.x, yuv.y, yuv.x);
        f.y = fmaf(cg.y, yuv.z, f.y);
        f.z = fmaf(cb.x, yuv.y, yuv.x);
        yuv = make_float3(hipUnpack1(y1.x), hipUnpack1(u1.x), hipUnpack1(v1.x));
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.w = fmaf(cr.y, yuv.z, yuv.x);
        rgb1.data[0] = hipPack(f);

        f.x = fmaf(cg.x, yuv.y, yuv.x);
        f.x = fmaf(cg.y, yuv.z, f.x);
        f.y = fmaf(cb.x, yuv.y, yuv.x);
        yuv = make_float3(hipUnpack2(y1.x), hipUnpack2(u1.x), hipUnpack2(v1.x));
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.z = fmaf(cr.y, yuv.z, yuv.x);
        f.w = fmaf(cg.x, yuv.y, yuv.x);
        f.w = fmaf(cg.y, yuv.z, f.w);
        rgb1.data[1] = hipPack(f);

        f.x = fmaf(cb.x, yuv.y, yuv.x);
        yuv = make_float3(hipUnpack3(y1.x), hipUnpack3(u1.x), hipUnpack3(v1.x));
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.y = fmaf(cr.y, yuv.z, yuv.x);
        f.z = fmaf(cg.x, yuv.y, yuv.x);
        f.z = fmaf(cg.y, yuv.z, f.z);
        f.w = fmaf(cb.x, yuv.y, yuv.x);
        rgb1.data[2] = hipPack(f);

        yuv = make_float3(hipUnpack0(y1.y), hipUnpack0(u1.y), hipUnpack0(v1.y));
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.x = fmaf(cr.y, yuv.z, yuv.x);
        f.y = fmaf(cg.x, yuv.y, yuv.x);
        f.y = fmaf(cg.y, yuv.z, f.y);
        f.z = fmaf(cb.x, yuv.y, yuv.x);
        yuv = make_float3(hipUnpack1(y1.y), hipUnpack1(u1.y), hipUnpack1(v1.y));
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.w = fmaf(cr.y, yuv.z, yuv.x);
        rgb1.data[3] = hipPack(f);

        f.x = fmaf(cg.x, yuv.y, yuv.x);
        f.x = fmaf(cg.y, yuv.z, f.x);
        f.y = fmaf(cb.x, yuv.y, yuv.x);
        yuv = make_float3(hipUnpack2(y1.y), hipUnpack2(u1.y), hipUnpack2(v1.y));
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.z = fmaf(cr.y, yuv.z, yuv.x);
        f.w = fmaf(cg.x, yuv.y, yuv.x);
        f.w = fmaf(cg.y, yuv.z, f.w);
        rgb1.data[4] = hipPack(f);

        f.x = fmaf(cb.x, yuv.y, yuv.x);
        yuv = make_float3(hipUnpack3(y1.y), hipUnpack3(u1.y), hipUnpack3(v1.y));
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.y = fmaf(cr.y, yuv.z, yuv.x);
        f.z = fmaf(cg.x, yuv.y, yuv.x);
        f.z = fmaf(cg.y, yuv.z, f.z);
        f.w = fmaf(cb.x, yuv.y, yuv.x);
        rgb1.data[5] = hipPack(f);

        *((DUINT6 *)(&dst_image[rgb0_idx])) = rgb0;
        *((DUINT6 *)(&dst_image[rgb1_idx])) = rgb1;
    }
}

/**
 * @brief Converts an NV12 image to RGB format.
 *
 * This function takes an NV12 image and converts it to RGB format using the ColorConvertNV12ToRGBKernel HIP kernel.
 *
 * @param stream The CUDA stream to use for the kernel execution.
 * @param dst_width The width of the destination RGB image.
 * @param dst_height The height of the destination RGB image.
 * @param dst_image Pointer to the destination RGB image buffer.
 * @param dst_image_stride_in_bytes The stride (in bytes) of the destination RGB image buffer.
 * @param src_luma_image Pointer to the source luma (Y) image buffer.
 * @param src_luma_image_stride_in_bytes The stride (in bytes) of the source luma (Y) image buffer.
 * @param src_chroma_image Pointer to the source chroma (UV) image buffer.
 * @param src_chroma_image_stride_in_bytes The stride (in bytes) of the source chroma (UV) image buffer.
 */
void ColorConvertNV12ToRGB(hipStream_t stream, uint32_t dst_width, uint32_t dst_height,
    uint8_t *dst_image, uint32_t dst_image_stride_in_bytes,
    const uint8_t *src_luma_image, uint32_t src_luma_image_stride_in_bytes,
    const uint8_t *src_chroma_image, uint32_t src_chroma_image_stride_in_bytes) {
    int32_t local_threads_x = 16;
    int32_t local_threads_y = 4;
    int32_t global_threads_x = (dst_width + 7) >> 3;
    int32_t global_threads_y = (dst_height + 1) >> 1;

    uint32_t dst_width_comp = (dst_width + 7) / 8;
    uint32_t dst_height_comp = (dst_height + 1) / 2;
    uint32_t dst_image_stride_in_bytes_comp = dst_image_stride_in_bytes * 2;
    uint32_t src_luma_image_stride_in_bytes_comp = src_luma_image_stride_in_bytes * 2;

    ColorConvertNV12ToRGBKernel<<<dim3(ceil(static_cast<float>(global_threads_x) / local_threads_x), ceil(static_cast<float>(global_threads_y) / local_threads_y)),
                        dim3(local_threads_x, local_threads_y), 0, stream>>>(dst_image, dst_image_stride_in_bytes,
                        dst_image_stride_in_bytes_comp, src_luma_image, src_luma_image_stride_in_bytes, src_chroma_image,
                        src_chroma_image_stride_in_bytes, dst_width_comp, dst_height_comp, src_luma_image_stride_in_bytes_comp);
}

__global__ void ColorConvertNV12ToRGBPlanarKernel(
    uint8_t *dst_image_r, uint8_t *dst_image_g, uint8_t *dst_image_b, uint32_t dst_image_stride_in_bytes, uint32_t dst_image_stride_in_bytes_comp,
    const uint8_t *src_luma_image, uint32_t src_luma_image_stride_in_bytes,
    const uint8_t *src_chroma_image, uint32_t src_chroma_image_stride_in_bytes,
    uint32_t dst_width_comp, uint32_t dst_height_comp, uint32_t src_luma_image_stride_in_bytes_comp) {

    uint32_t x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    uint32_t y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    if ((x < dst_width_comp) && (y < dst_height_comp)) {
        uint32_t src_y0_idx = y * src_luma_image_stride_in_bytes_comp + (x << 3);
        uint32_t src_y1_idx = src_y0_idx + src_luma_image_stride_in_bytes;
        uint32_t src_uv_idx = y * src_chroma_image_stride_in_bytes + (x << 3);
        uint2 y0 = *((uint2 *)(&src_luma_image[src_y0_idx]));
        uint2 y1 = *((uint2 *)(&src_luma_image[src_y1_idx]));
        uint2 uv = *((uint2 *)(&src_chroma_image[src_uv_idx]));

        uint32_t rgb0_idx = y * dst_image_stride_in_bytes_comp + (x * 8);
        uint32_t rgb1_idx = rgb0_idx + dst_image_stride_in_bytes;

        float4 f;
        uint2 u0, u1;
        uint2 v0, v1;

        f.x = hipUnpack0(uv.x);
        f.y = f.x;
        f.z = hipUnpack2(uv.x);
        f.w = f.z;
        u0.x = hipPack(f);

        f.x = hipUnpack0(uv.y);
        f.y = f.x;
        f.z = hipUnpack2(uv.y);
        f.w = f.z;
        u0.y = hipPack(f);

        u1.x = u0.x;
        u1.y = u0.y;

        f.x = hipUnpack1(uv.x);
        f.y = f.x;
        f.z = hipUnpack3(uv.x);
        f.w = f.z;
        v0.x = hipPack(f);

        f.x = hipUnpack1(uv.y);
        f.y = f.x;
        f.z = hipUnpack3(uv.y);
        f.w = f.z;
        v0.y = hipPack(f);

        v1.x = v0.x;
        v1.y = v0.y;

        float2 cr = make_float2( 0.0000f,  1.5748f);
        float2 cg = make_float2(-0.1873f, -0.4681f);
        float2 cb = make_float2( 1.8556f,  0.0000f);
        float3 yuv;
        DUINT6 rgb0, rgb1;

        yuv = make_float3(hipUnpack0(y0.x), hipUnpack0(u0.x), hipUnpack0(v0.x));
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.x = fmaf(cr.y, yuv.z, yuv.x);
        f.y = fmaf(cg.x, yuv.y, yuv.x);
        f.y = fmaf(cg.y, yuv.z, f.y);
        f.z = fmaf(cb.x, yuv.y, yuv.x);
        yuv = make_float3(hipUnpack1(y0.x), hipUnpack1(u0.x), hipUnpack1(v0.x));
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.w = fmaf(cr.y, yuv.z, yuv.x);
        rgb0.data[0] = hipPack(f);

        f.x = fmaf(cg.x, yuv.y, yuv.x);
        f.x = fmaf(cg.y, yuv.z, f.x);
        f.y = fmaf(cb.x, yuv.y, yuv.x);
        yuv = make_float3(hipUnpack2(y0.x), hipUnpack2(u0.x), hipUnpack2(v0.x));
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.z = fmaf(cr.y, yuv.z, yuv.x);
        f.w = fmaf(cg.x, yuv.y, yuv.x);
        f.w = fmaf(cg.y, yuv.z, f.w);
        rgb0.data[1] = hipPack(f);

        f.x = fmaf(cb.x, yuv.y, yuv.x);
        yuv = make_float3(hipUnpack3(y0.x), hipUnpack3(u0.x), hipUnpack3(v0.x));
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.y = fmaf(cr.y, yuv.z, yuv.x);
        f.z = fmaf(cg.x, yuv.y, yuv.x);
        f.z = fmaf(cg.y, yuv.z, f.z);
        f.w = fmaf(cb.x, yuv.y, yuv.x);
        rgb0.data[2] = hipPack(f);

        yuv = make_float3(hipUnpack0(y0.y), hipUnpack0(u0.y), hipUnpack0(v0.y));
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.x = fmaf(cr.y, yuv.z, yuv.x);
        f.y = fmaf(cg.x, yuv.y, yuv.x);
        f.y = fmaf(cg.y, yuv.z, f.y);
        f.z = fmaf(cb.x, yuv.y, yuv.x);
        yuv = make_float3(hipUnpack1(y0.y), hipUnpack1(u0.y), hipUnpack1(v0.y));
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.w = fmaf(cr.y, yuv.z, yuv.x);
        rgb0.data[3] = hipPack(f);

        f.x = fmaf(cg.x, yuv.y, yuv.x);
        f.x = fmaf(cg.y, yuv.z, f.x);
        f.y = fmaf(cb.x, yuv.y, yuv.x);
        yuv = make_float3(hipUnpack2(y0.y), hipUnpack2(u0.y), hipUnpack2(v0.y));
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.z = fmaf(cr.y, yuv.z, yuv.x);
        f.w = fmaf(cg.x, yuv.y, yuv.x);
        f.w = fmaf(cg.y, yuv.z, f.w);
        rgb0.data[4] = hipPack(f);

        f.x = fmaf(cb.x, yuv.y, yuv.x);
        yuv = make_float3(hipUnpack3(y0.y), hipUnpack3(u0.y), hipUnpack3(v0.y));
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.y = fmaf(cr.y, yuv.z, yuv.x);
        f.z = fmaf(cg.x, yuv.y, yuv.x);
        f.z = fmaf(cg.y, yuv.z, f.z);
        f.w = fmaf(cb.x, yuv.y, yuv.x);
        rgb0.data[5] = hipPack(f);

        yuv = make_float3(hipUnpack0(y1.x), hipUnpack0(u1.x), hipUnpack0(v1.x));
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.x = fmaf(cr.y, yuv.z, yuv.x);
        f.y = fmaf(cg.x, yuv.y, yuv.x);
        f.y = fmaf(cg.y, yuv.z, f.y);
        f.z = fmaf(cb.x, yuv.y, yuv.x);
        yuv = make_float3(hipUnpack1(y1.x), hipUnpack1(u1.x), hipUnpack1(v1.x));
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.w = fmaf(cr.y, yuv.z, yuv.x);
        rgb1.data[0] = hipPack(f);

        f.x = fmaf(cg.x, yuv.y, yuv.x);
        f.x = fmaf(cg.y, yuv.z, f.x);
        f.y = fmaf(cb.x, yuv.y, yuv.x);
        yuv = make_float3(hipUnpack2(y1.x), hipUnpack2(u1.x), hipUnpack2(v1.x));
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.z = fmaf(cr.y, yuv.z, yuv.x);
        f.w = fmaf(cg.x, yuv.y, yuv.x);
        f.w = fmaf(cg.y, yuv.z, f.w);
        rgb1.data[1] = hipPack(f);

        f.x = fmaf(cb.x, yuv.y, yuv.x);
        yuv = make_float3(hipUnpack3(y1.x), hipUnpack3(u1.x), hipUnpack3(v1.x));
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.y = fmaf(cr.y, yuv.z, yuv.x);
        f.z = fmaf(cg.x, yuv.y, yuv.x);
        f.z = fmaf(cg.y, yuv.z, f.z);
        f.w = fmaf(cb.x, yuv.y, yuv.x);
        rgb1.data[2] = hipPack(f);

        yuv = make_float3(hipUnpack0(y1.y), hipUnpack0(u1.y), hipUnpack0(v1.y));
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.x = fmaf(cr.y, yuv.z, yuv.x);
        f.y = fmaf(cg.x, yuv.y, yuv.x);
        f.y = fmaf(cg.y, yuv.z, f.y);
        f.z = fmaf(cb.x, yuv.y, yuv.x);
        yuv = make_float3(hipUnpack1(y1.y), hipUnpack1(u1.y), hipUnpack1(v1.y));
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.w = fmaf(cr.y, yuv.z, yuv.x);
        rgb1.data[3] = hipPack(f);

        f.x = fmaf(cg.x, yuv.y, yuv.x);
        f.x = fmaf(cg.y, yuv.z, f.x);
        f.y = fmaf(cb.x, yuv.y, yuv.x);
        yuv = make_float3(hipUnpack2(y1.y), hipUnpack2(u1.y), hipUnpack2(v1.y));
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.z = fmaf(cr.y, yuv.z, yuv.x);
        f.w = fmaf(cg.x, yuv.y, yuv.x);
        f.w = fmaf(cg.y, yuv.z, f.w);
        rgb1.data[4] = hipPack(f);

        f.x = fmaf(cb.x, yuv.y, yuv.x);
        yuv = make_float3(hipUnpack3(y1.y), hipUnpack3(u1.y), hipUnpack3(v1.y));
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.y = fmaf(cr.y, yuv.z, yuv.x);
        f.z = fmaf(cg.x, yuv.y, yuv.x);
        f.z = fmaf(cg.y, yuv.z, f.z);
        f.w = fmaf(cb.x, yuv.y, yuv.x);
        rgb1.data[5] = hipPack(f);

        uint2 red0, red1, green0, green1, blue0, blue1;
        red0.x = hipPack(make_float4(hipUnpack0(rgb0.data[0]), hipUnpack3(rgb0.data[0]), hipUnpack2(rgb0.data[1]), hipUnpack1(rgb0.data[2])));
        red0.y = hipPack(make_float4(hipUnpack0(rgb0.data[3]), hipUnpack3(rgb0.data[3]), hipUnpack2(rgb0.data[4]), hipUnpack1(rgb0.data[5])));
        red1.x = hipPack(make_float4(hipUnpack0(rgb1.data[0]), hipUnpack3(rgb1.data[0]), hipUnpack2(rgb1.data[1]), hipUnpack1(rgb1.data[2])));
        red1.y = hipPack(make_float4(hipUnpack0(rgb1.data[3]), hipUnpack3(rgb1.data[3]), hipUnpack2(rgb1.data[4]), hipUnpack1(rgb1.data[5])));

        green0.x = hipPack(make_float4(hipUnpack1(rgb0.data[0]), hipUnpack0(rgb0.data[1]), hipUnpack3(rgb0.data[1]), hipUnpack2(rgb0.data[2])));
        green0.y = hipPack(make_float4(hipUnpack1(rgb0.data[3]), hipUnpack0(rgb0.data[4]), hipUnpack3(rgb0.data[4]), hipUnpack2(rgb0.data[5])));
        green1.x = hipPack(make_float4(hipUnpack1(rgb1.data[0]), hipUnpack0(rgb1.data[1]), hipUnpack3(rgb1.data[1]), hipUnpack2(rgb1.data[2])));
        green1.y = hipPack(make_float4(hipUnpack1(rgb1.data[3]), hipUnpack0(rgb1.data[4]), hipUnpack3(rgb1.data[4]), hipUnpack2(rgb1.data[5])));

        blue0.x = hipPack(make_float4(hipUnpack2(rgb0.data[0]), hipUnpack1(rgb0.data[1]), hipUnpack0(rgb0.data[2]), hipUnpack3(rgb0.data[2])));
        blue0.y = hipPack(make_float4(hipUnpack2(rgb0.data[3]), hipUnpack1(rgb0.data[4]), hipUnpack0(rgb0.data[5]), hipUnpack3(rgb0.data[5])));
        blue1.x = hipPack(make_float4(hipUnpack2(rgb1.data[0]), hipUnpack1(rgb1.data[1]), hipUnpack0(rgb1.data[2]), hipUnpack3(rgb1.data[2])));
        blue1.y = hipPack(make_float4(hipUnpack2(rgb1.data[3]), hipUnpack1(rgb1.data[4]), hipUnpack0(rgb1.data[5]), hipUnpack3(rgb1.data[5])));

        *((uint2 *)(&dst_image_r[rgb0_idx])) = red0;
        *((uint2 *)(&dst_image_r[rgb1_idx])) = red1;
        *((uint2 *)(&dst_image_g[rgb0_idx])) = green0;
        *((uint2 *)(&dst_image_g[rgb1_idx])) = green1;
        *((uint2 *)(&dst_image_b[rgb0_idx])) = blue0;
        *((uint2 *)(&dst_image_b[rgb1_idx])) = blue1;
    }
}

/**
 * @brief Converts an NV12 image to RGB planar format.
 *
 * This function takes an NV12 image and converts it to RGB planar format using the ColorConvertNV12ToRGBPlanarKernel HIP kernel.
 * The resulting RGB planar image is stored in separate R, G, and B channels.
 *
 * @param stream The HIP stream to use for the kernel execution.
 * @param dst_width The width of the destination RGB planar image.
 * @param dst_height The height of the destination RGB planar image.
 * @param dst_image_r Pointer to the destination R channel of the RGB planar image.
 * @param dst_image_g Pointer to the destination G channel of the RGB planar image.
 * @param dst_image_b Pointer to the destination B channel of the RGB planar image.
 * @param dst_image_stride_in_bytes The stride (in bytes) of the destination RGB planar image.
 * @param src_luma_image Pointer to the source luma (Y) channel of the NV12 image.
 * @param src_luma_image_stride_in_bytes The stride (in bytes) of the source luma (Y) channel.
 * @param src_chroma_image Pointer to the source chroma (UV) channel of the NV12 image.
 * @param src_chroma_image_stride_in_bytes The stride (in bytes) of the source chroma (UV) channel.
 */
void ColorConvertNV12ToRGBPlanar(hipStream_t stream, uint32_t dst_width, uint32_t dst_height,
    uint8_t *dst_image_r, uint8_t *dst_image_g, uint8_t *dst_image_b, uint32_t dst_image_stride_in_bytes,
    const uint8_t *src_luma_image, uint32_t src_luma_image_stride_in_bytes,
    const uint8_t *src_chroma_image, uint32_t src_chroma_image_stride_in_bytes) {

    int32_t local_threads_x = 16;
    int32_t local_threads_y = 4;
    int32_t global_threads_x = (dst_width + 7) >> 3;
    int32_t global_threads_y = (dst_height + 1) >> 1;

    uint32_t dst_width_comp = (dst_width + 7) / 8;
    uint32_t dst_height_comp = (dst_height + 1) / 2;
    uint32_t dst_image_stride_in_bytes_comp = dst_image_stride_in_bytes * 2;
    uint32_t src_luma_image_stride_in_bytes_comp = src_luma_image_stride_in_bytes * 2;

    ColorConvertNV12ToRGBPlanarKernel<<<dim3(ceil(static_cast<float>(global_threads_x) / local_threads_x), ceil(static_cast<float>(global_threads_y) / local_threads_y)),
                        dim3(local_threads_x, local_threads_y), 0, stream>>>(dst_image_r, dst_image_g, dst_image_b, dst_image_stride_in_bytes,
                        dst_image_stride_in_bytes_comp, src_luma_image, src_luma_image_stride_in_bytes, src_chroma_image,
                        src_chroma_image_stride_in_bytes, dst_width_comp, dst_height_comp, src_luma_image_stride_in_bytes_comp);

}

__global__ void ColorConvertYUV400ToRGBKernel(
    uint8_t *dst_image, uint32_t dst_image_stride_in_bytes, uint32_t dst_image_stride_in_bytes_comp,
    const uint8_t *src_luma_image, uint32_t src_luma_image_stride_in_bytes,
    uint32_t dst_width_comp, uint32_t dst_height_comp, uint32_t src_luma_image_stride_in_bytes_comp) {

    uint32_t x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    uint32_t y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    if ((x < dst_width_comp) && (y < dst_height_comp)) {
        uint32_t src_y0_idx = y * src_luma_image_stride_in_bytes_comp + (x << 3);
        uint32_t src_y1_idx = src_y0_idx + src_luma_image_stride_in_bytes;

        uint2 y0 = *((uint2 *)(&src_luma_image[src_y0_idx]));
        uint2 y1 = *((uint2 *)(&src_luma_image[src_y1_idx]));

        uint32_t rgb0_idx = y * dst_image_stride_in_bytes_comp + (x * 24);
        uint32_t rgb1_idx = rgb0_idx + dst_image_stride_in_bytes;

        DUINT6 rgb0, rgb1;

        uint8_t y0_b0, y0_b1, y0_b2, y0_b3, y0_b4, y0_b5, y0_b6, y0_b7;
        uint8_t y1_b0, y1_b1, y1_b2, y1_b3, y1_b4, y1_b5, y1_b6, y1_b7;

        y0_b0 = hipUnpack0(y0.x);
        y0_b1 = hipUnpack1(y0.x);
        y0_b2 = hipUnpack2(y0.x);
        y0_b3 = hipUnpack3(y0.x);
        y0_b4 = hipUnpack0(y0.y);
        y0_b5 = hipUnpack1(y0.y);
        y0_b6 = hipUnpack2(y0.y);
        y0_b7 = hipUnpack3(y0.y);

        y1_b0 = hipUnpack0(y1.x);
        y1_b1 = hipUnpack1(y1.x);
        y1_b2 = hipUnpack2(y1.x);
        y1_b3 = hipUnpack3(y1.x);
        y1_b4 = hipUnpack0(y1.y);
        y1_b5 = hipUnpack1(y1.y);
        y1_b6 = hipUnpack2(y1.y);
        y1_b7 = hipUnpack3(y1.y);

        rgb0.data[0] = hipPack(make_float4(y0_b0, y0_b0, y0_b0, y0_b1));
        rgb0.data[1] = hipPack(make_float4(y0_b1, y0_b1, y0_b2, y0_b2));
        rgb0.data[2] = hipPack(make_float4(y0_b2, y0_b3, y0_b3, y0_b3));
        rgb0.data[3] = hipPack(make_float4(y0_b4, y0_b4, y0_b4, y0_b5));
        rgb0.data[4] = hipPack(make_float4(y0_b5, y0_b5, y0_b6, y0_b6));
        rgb0.data[5] = hipPack(make_float4(y0_b6, y0_b7, y0_b7, y0_b7));

        rgb1.data[0] = hipPack(make_float4(y1_b0, y1_b0, y1_b0, y1_b1));
        rgb1.data[1] = hipPack(make_float4(y1_b1, y1_b1, y1_b2, y1_b2));
        rgb1.data[2] = hipPack(make_float4(y1_b2, y1_b3, y1_b3, y1_b3));
        rgb1.data[3] = hipPack(make_float4(y1_b4, y1_b4, y1_b4, y1_b5));
        rgb1.data[4] = hipPack(make_float4(y1_b5, y1_b5, y1_b6, y1_b6));
        rgb1.data[5] = hipPack(make_float4(y1_b6, y1_b7, y1_b7, y1_b7));

        *((DUINT6 *)(&dst_image[rgb0_idx])) = rgb0;
        *((DUINT6 *)(&dst_image[rgb1_idx])) = rgb1;
    }
}

/**
 * @brief Converts a YUV400 image to RGB format.
 *
 * This function takes a YUV400 image and converts it to RGB format using the ColorConvertYUV400ToRGBKernel HIP kernel.
 *
 * @param stream The HIP stream to be used for the kernel execution.
 * @param dst_width The width of the destination RGB image.
 * @param dst_height The height of the destination RGB image.
 * @param dst_image Pointer to the destination RGB image buffer.
 * @param dst_image_stride_in_bytes The stride (in bytes) of the destination RGB image buffer.
 * @param src_luma_image Pointer to the source YUV400 luma image buffer.
 * @param src_luma_image_stride_in_bytes The stride (in bytes) of the source YUV400 luma image buffer.
 */
void ColorConvertYUV400ToRGB(hipStream_t stream, uint32_t dst_width, uint32_t dst_height,
    uint8_t *dst_image, uint32_t dst_image_stride_in_bytes,
    const uint8_t *src_luma_image, uint32_t src_luma_image_stride_in_bytes){

    int32_t local_threads_x = 16;
    int32_t local_threads_y = 4;
    int32_t global_threads_x = (dst_width + 7) >> 3;
    int32_t global_threads_y = (dst_height + 1) >> 1;

    uint32_t dst_width_comp = (dst_width + 7) / 8;
    uint32_t dst_height_comp = (dst_height + 1) / 2;
    uint32_t dst_image_stride_in_bytes_comp = dst_image_stride_in_bytes * 2;
    uint32_t src_luma_image_stride_in_bytes_comp = src_luma_image_stride_in_bytes * 2;

    ColorConvertYUV400ToRGBKernel<<<dim3(ceil(static_cast<float>(global_threads_x) / local_threads_x), ceil(static_cast<float>(global_threads_y) / local_threads_y)),
                        dim3(local_threads_x, local_threads_y), 0, stream>>>(dst_image, dst_image_stride_in_bytes,
                        dst_image_stride_in_bytes_comp, src_luma_image, src_luma_image_stride_in_bytes, dst_width_comp, dst_height_comp,
                        src_luma_image_stride_in_bytes_comp);

}

__global__ void ColorConvertYUV400ToRGBPlanarKernel(
    uint8_t *dst_image_r, uint8_t *dst_image_g, uint8_t *dst_image_b, uint32_t dst_image_stride_in_bytes, uint32_t dst_image_stride_in_bytes_comp,
    const uint8_t *src_luma_image, uint32_t src_luma_image_stride_in_bytes,
    uint32_t dst_width_comp, uint32_t dst_height_comp, uint32_t src_luma_image_stride_in_bytes_comp) {

    uint32_t x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    uint32_t y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    if ((x < dst_width_comp) && (y < dst_height_comp)) {
        uint32_t src_y0_idx = y * src_luma_image_stride_in_bytes_comp + (x << 3);
        uint32_t src_y1_idx = src_y0_idx + src_luma_image_stride_in_bytes;

        uint2 y0 = *((uint2 *)(&src_luma_image[src_y0_idx]));
        uint2 y1 = *((uint2 *)(&src_luma_image[src_y1_idx]));

        uint32_t rgb0_idx = y * dst_image_stride_in_bytes_comp + (x * 8);
        uint32_t rgb1_idx = rgb0_idx + dst_image_stride_in_bytes;

        *((uint2 *)(&dst_image_r[rgb0_idx])) = y0;
        *((uint2 *)(&dst_image_r[rgb1_idx])) = y1;
        *((uint2 *)(&dst_image_g[rgb0_idx])) = y0;
        *((uint2 *)(&dst_image_g[rgb1_idx])) = y1;
        *((uint2 *)(&dst_image_b[rgb0_idx])) = y0;
        *((uint2 *)(&dst_image_b[rgb1_idx])) = y1;
    }
}

/**
 * @brief Converts a YUV400 image to RGB planar format.
 *
 * This function takes a YUV400 image and converts it to RGB planar format using the ColorConvertYUV400ToRGBPlanarKernel HIP kernel.
 *
 * @param stream The HIP stream on which the kernel function will be executed.
 * @param dst_width The width of the destination RGB image.
 * @param dst_height The height of the destination RGB image.
 * @param dst_image_r Pointer to the destination red channel image.
 * @param dst_image_g Pointer to the destination green channel image.
 * @param dst_image_b Pointer to the destination blue channel image.
 * @param dst_image_stride_in_bytes The stride (in bytes) of the destination image.
 * @param src_luma_image Pointer to the source YUV400 luma image.
 * @param src_luma_image_stride_in_bytes The stride (in bytes) of the source luma image.
 */
void ColorConvertYUV400ToRGBPlanar(hipStream_t stream, uint32_t dst_width, uint32_t dst_height,
    uint8_t *dst_image_r, uint8_t *dst_image_g, uint8_t *dst_image_b, uint32_t dst_image_stride_in_bytes,
    const uint8_t *src_luma_image, uint32_t src_luma_image_stride_in_bytes) {

    int32_t local_threads_x = 16;
    int32_t local_threads_y = 4;
    int32_t global_threads_x = (dst_width + 7) >> 3;
    int32_t global_threads_y = (dst_height + 1) >> 1;

    uint32_t dst_width_comp = (dst_width + 7) / 8;
    uint32_t dst_height_comp = (dst_height + 1) / 2;
    uint32_t dst_image_stride_in_bytes_comp = dst_image_stride_in_bytes * 2;
    uint32_t src_luma_image_stride_in_bytes_comp = src_luma_image_stride_in_bytes * 2;

    ColorConvertYUV400ToRGBPlanarKernel<<<dim3(ceil(static_cast<float>(global_threads_x) / local_threads_x), ceil(static_cast<float>(global_threads_y) / local_threads_y)),
                        dim3(local_threads_x, local_threads_y), 0, stream>>>(dst_image_r, dst_image_g, dst_image_b, dst_image_stride_in_bytes,
                        dst_image_stride_in_bytes_comp, src_luma_image, src_luma_image_stride_in_bytes, dst_width_comp, dst_height_comp,
                        src_luma_image_stride_in_bytes_comp);

}

__global__ void ColorConvertRGBAToRGBKernel(uint32_t dst_width, uint32_t dst_height, uint8_t *dst_image, uint32_t dst_image_stride_in_bytes,
    const uint8_t *src_image, uint32_t src_image_stride_in_bytes) {

    uint32_t x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 8;
    uint32_t y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    if (x >= dst_width || y >= dst_height) {
        return;
    }

    uint32_t src_idx = y * src_image_stride_in_bytes + (x << 2);
    uint32_t dst_idx  = y * dst_image_stride_in_bytes + (x * 3);

    DUINT8 src = *((DUINT8 *)(&src_image[src_idx]));
    DUINT6 dst;

    dst.data[0] = hipPack(make_float4(hipUnpack0(src.data[0]), hipUnpack1(src.data[0]), hipUnpack2(src.data[0]), hipUnpack0(src.data[1])));
    dst.data[1] = hipPack(make_float4(hipUnpack1(src.data[1]), hipUnpack2(src.data[1]), hipUnpack0(src.data[2]), hipUnpack1(src.data[2])));
    dst.data[2] = hipPack(make_float4(hipUnpack2(src.data[2]), hipUnpack0(src.data[3]), hipUnpack1(src.data[3]), hipUnpack2(src.data[3])));
    dst.data[3] = hipPack(make_float4(hipUnpack0(src.data[4]), hipUnpack1(src.data[4]), hipUnpack2(src.data[4]), hipUnpack0(src.data[5])));
    dst.data[4] = hipPack(make_float4(hipUnpack1(src.data[5]), hipUnpack2(src.data[5]), hipUnpack0(src.data[6]), hipUnpack1(src.data[6])));
    dst.data[5] = hipPack(make_float4(hipUnpack2(src.data[6]), hipUnpack0(src.data[7]), hipUnpack1(src.data[7]), hipUnpack2(src.data[7])));

    *((DUINT6 *)(&dst_image[dst_idx])) = dst;
}

/**
 * @brief Converts an RGBA image to an RGB image.
 *
 * This function takes an RGBA image and converts it to an RGB image using the ColorConvertRGBAToRGBKernel HIP kernel
 *
 * @param stream The HIP stream to execute the kernel on.
 * @param dst_width The width of the destination image.
 * @param dst_height The height of the destination image.
 * @param dst_image Pointer to the destination image buffer.
 * @param dst_image_stride_in_bytes The stride (in bytes) of the destination image buffer.
 * @param src_image Pointer to the source image buffer.
 * @param src_image_stride_in_bytes The stride (in bytes) of the source image buffer.
 */
void ColorConvertRGBAToRGB(hipStream_t stream, uint32_t dst_width, uint32_t dst_height, uint8_t *dst_image, uint32_t dst_image_stride_in_bytes,
    const uint8_t *src_image, uint32_t src_image_stride_in_bytes) {
    int localThreads_x = 16;
    int localThreads_y = 16;
    int globalThreads_x = (dst_width + 7) >> 3;
    int globalThreads_y = dst_height;

    ColorConvertRGBAToRGBKernel<<<dim3(ceil(static_cast<float>(globalThreads_x) / localThreads_x), ceil(static_cast<float>(globalThreads_y) / localThreads_y)),
                        dim3(localThreads_x, localThreads_y), 0, stream >>>(dst_width, dst_height, dst_image, dst_image_stride_in_bytes,
                        src_image, src_image_stride_in_bytes);
}

__global__ void ConvertInterleavedUVToPlanarUVKernel(uint32_t dst_width, uint32_t dst_height,
    uint8_t *dst_image1, uint8_t *dst_image2, uint32_t dst_image_stride_in_bytes,
    const uint8_t *src_image, uint32_t src_image_stride_in_bytes) {

    uint32_t x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 8;
    uint32_t y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    if (x >= dst_width || y >= dst_height) {
        return;
    }

    uint32_t src_idx = y * src_image_stride_in_bytes + x + x;
    uint32_t dst_idx = y * dst_image_stride_in_bytes + x;

    uint4 src = *((uint4 *)(&src_image[src_idx]));
    uint2 dst1, dst2;

    dst1.x = hipPack(make_float4(hipUnpack0(src.x), hipUnpack2(src.x), hipUnpack0(src.y), hipUnpack2(src.y)));
    dst1.y = hipPack(make_float4(hipUnpack0(src.z), hipUnpack2(src.z), hipUnpack0(src.w), hipUnpack2(src.w)));
    dst2.x = hipPack(make_float4(hipUnpack1(src.x), hipUnpack3(src.x), hipUnpack1(src.y), hipUnpack3(src.y)));
    dst2.y = hipPack(make_float4(hipUnpack1(src.z), hipUnpack3(src.z), hipUnpack1(src.w), hipUnpack3(src.w)));

    *((uint2 *)(&dst_image1[dst_idx])) = dst1;
    *((uint2 *)(&dst_image2[dst_idx])) = dst2;

}
/**
 * @brief Converts interleaved UV data to planar UV data.
 *
 * This function takes interleaved UV data and converts it to planar UV data
 * using the ConvertInterleavedUVToPlanarUVKernel HIP kernel.
 *
 * @param stream The HIP stream to use for the kernel execution.
 * @param dst_width The width of the destination image.
 * @param dst_height The height of the destination image.
 * @param dst_image1 Pointer to the destination image buffer for the first plane.
 * @param dst_image2 Pointer to the destination image buffer for the second plane.
 * @param dst_image_stride_in_bytes The stride (in bytes) of the destination image buffer.
 * @param src_image1 Pointer to the source image buffer containing interleaved UV data.
 * @param src_image1_stride_in_bytes The stride (in bytes) of the source image buffer.
 */
void ConvertInterleavedUVToPlanarUV(hipStream_t stream, uint32_t dst_width, uint32_t dst_height,
    uint8_t *dst_image1, uint8_t *dst_image2, uint32_t dst_image_stride_in_bytes,
    const uint8_t *src_image1, uint32_t src_image1_stride_in_bytes) {
    int32_t local_threads_x = 16, local_threads_y = 16;
    int32_t global_threads_x = (dst_width + 7) >> 3;
    int32_t global_threads_y = dst_height;

    ConvertInterleavedUVToPlanarUVKernel<<<dim3(ceil(static_cast<float>(global_threads_x) / local_threads_x), ceil(static_cast<float>(global_threads_y) / local_threads_y)),
                                    dim3(local_threads_x, local_threads_y), 0, stream>>>(dst_width, dst_height, dst_image1, dst_image2,
                                    dst_image_stride_in_bytes, src_image1, src_image1_stride_in_bytes);

}

__global__ void ExtractYFromPackedYUYVKernel(uint32_t dst_height,
    uint8_t *destination_y, uint32_t dst_luma_stride_in_bytes,
    const uint8_t *src_image, uint32_t src_image_stride_in_bytes,
    uint32_t dst_width_comp) {

    uint32_t x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    uint32_t y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    if (x < dst_width_comp && y < dst_height) {
        uint32_t src_idx = y * src_image_stride_in_bytes + (x << 4);
        uint32_t dst_idx = y * dst_luma_stride_in_bytes + (x << 3);

        uint4 src = *((uint4 *)(&src_image[src_idx]));
        uint2 dst_y;
        dst_y.x = hipPack(make_float4(hipUnpack0(src.x), hipUnpack2(src.x), hipUnpack0(src.y), hipUnpack2(src.y)));
        dst_y.y = hipPack(make_float4(hipUnpack0(src.z), hipUnpack2(src.z), hipUnpack0(src.w), hipUnpack2(src.w)));

        *((uint2 *)(&destination_y[dst_idx])) = dst_y;
    }
}

/**
 * @brief Extracts the Y component from a packed YUYV image and stores it in a separate buffer.
 *
 * This function takes a packed YUYV image and extracts the Y component (luma) from it. The extracted Y component is then
 * stored in a separate buffer. The function operates on the GPU using HIP
 * and requires a HIP stream for asynchronous execution.
 *
 * @param stream The HIP stream to be used for the kernel execution.
 * @param dst_width The width of the destination buffer (in pixels).
 * @param dst_height The height of the destination buffer (in pixels).
 * @param destination_y Pointer to the destination buffer where the extracted Y component will be stored.
 * @param dst_luma_stride_in_bytes The stride (in bytes) of the destination buffer.
 * @param src_image Pointer to the source packed YUYV image.
 * @param src_image_stride_in_bytes The stride (in bytes) of the source image.
 */
void ExtractYFromPackedYUYV(hipStream_t stream, uint32_t dst_width, uint32_t dst_height,
    uint8_t *destination_y, uint32_t dst_luma_stride_in_bytes, const uint8_t *src_image, uint32_t src_image_stride_in_bytes) {
    int32_t local_threads_x = 16;
    int32_t local_threads_y = 4;
    int32_t global_threads_x = (dst_width + 7) >> 3;
    int32_t global_threads_y = dst_height;

    uint32_t dst_width_comp = (dst_width + 7) / 8;

    ExtractYFromPackedYUYVKernel<<<dim3(ceil(static_cast<float>(global_threads_x) / local_threads_x), ceil(static_cast<float>(global_threads_y) / local_threads_y)),
                                  dim3(local_threads_x, local_threads_y), 0, stream>>>(dst_height, destination_y,
                                  dst_luma_stride_in_bytes, src_image, src_image_stride_in_bytes, dst_width_comp);
}

__global__ void ConvertPackedYUYVToPlanarYUVKernel(uint32_t dst_height,
    uint8_t *destination_y, uint8_t *destination_u, uint8_t *destination_v, uint32_t dst_luma_stride_in_bytes, uint32_t dst_chroma_stride_in_bytes,
    const uint8_t *src_image, uint32_t src_image_stride_in_bytes,
    uint32_t dst_width_comp) {

    uint32_t x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    uint32_t y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    if ((x < dst_width_comp && y < dst_height)) {
        uint32_t src_idx = y * src_image_stride_in_bytes + (x << 4);
        uint32_t dst_y_idx = y * dst_luma_stride_in_bytes + (x << 3);
        uint32_t dst_uv_idx = y * dst_chroma_stride_in_bytes + (x << 2);

        uint4 src = *((uint4 *)(&src_image[src_idx]));
        uint2 dst_y;
        uint32_t dst_u, dst_v;

        dst_y.x = hipPack(make_float4(hipUnpack0(src.x), hipUnpack2(src.x), hipUnpack0(src.y), hipUnpack2(src.y)));
        dst_y.y = hipPack(make_float4(hipUnpack0(src.z), hipUnpack2(src.z), hipUnpack0(src.w), hipUnpack2(src.w)));
        dst_u = hipPack(make_float4(hipUnpack1(src.x), hipUnpack1(src.y), hipUnpack1(src.z), hipUnpack1(src.w)));
        dst_v = hipPack(make_float4(hipUnpack3(src.x), hipUnpack3(src.y), hipUnpack3(src.z), hipUnpack3(src.w)));

        *((uint2 *)(&destination_y[dst_y_idx])) = dst_y;
        *((uint32_t *)(&destination_u[dst_uv_idx])) = dst_u;
        *((uint32_t *)(&destination_v[dst_uv_idx])) = dst_v;
    }
}

/**
 * @brief Converts a packed YUYV image to planar YUV format.
 *
 * This function takes a packed YUYV image and converts it to planar YUV format
 * using the ConvertPackedYUYVToPlanarYUVKernel HIP kernel.
 * The packed YUYV image consists of interleaved Y, U, Y, V samples, while the
 * planar YUV format separates the Y, U, and V samples into separate planes.
 *
 * @param stream The HIP stream to associate the kernel launch with.
 * @param dst_width The width of the destination image in pixels.
 * @param dst_height The height of the destination image in pixels.
 * @param destination_y Pointer to the destination Y plane.
 * @param destination_u Pointer to the destination U plane.
 * @param destination_v Pointer to the destination V plane.
 * @param dst_luma_stride_in_bytes The stride (in bytes) of the destination luma plane.
 * @param dst_chroma_stride_in_bytes The stride (in bytes) of the destination chroma planes.
 * @param src_image Pointer to the source packed YUYV image.
 * @param src_image_stride_in_bytes The stride (in bytes) of the source image.
 */
void ConvertPackedYUYVToPlanarYUV(hipStream_t stream, uint32_t dst_width, uint32_t dst_height,
    uint8_t *destination_y, uint8_t *destination_u, uint8_t *destination_v, uint32_t dst_luma_stride_in_bytes, uint32_t dst_chroma_stride_in_bytes,
    const uint8_t *src_image, uint32_t src_image_stride_in_bytes) {

    int32_t local_threads_x = 16;
    int32_t local_threads_y = 4;
    int32_t global_threads_x = (dst_width + 7) >> 3;
    int32_t global_threads_y = dst_height;
    uint32_t dst_width_comp = (dst_width + 7) / 8;

    ConvertPackedYUYVToPlanarYUVKernel<<<dim3(ceil(static_cast<float>(global_threads_x) / local_threads_x), ceil(static_cast<float>(global_threads_y) / local_threads_y)),
                                    dim3(local_threads_x, local_threads_y), 0, stream>>>(dst_height, destination_y, destination_u,
                                    destination_v, dst_luma_stride_in_bytes, dst_chroma_stride_in_bytes, src_image, src_image_stride_in_bytes, dst_width_comp);
}