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

#include "resize_kernels.h"
#include "roc_video_dec.h"

/**
 * @brief low level HIP kernel for Resize using nearest neighbor interpolation
 * 
 * @tparam YuvUnitx2 
 * @param p_src    - src Y pointer
 * @param p_src_uv  - src UV pointer
 * @param src_pitch - src pitch
 * @param p_dst     - dst Y pointer
 * @param p_dst_uv  - dst UV pointer
 * @param pitch     - dst pitch
 * @param width     - dst width
 * @param height    - dst height
 * @param fx_scale  - xscale
 * @param fy_scale  - yscale
 * @return 
 */

template<typename YuvUnitx2>
static __global__ void ResizeHip(uint8_t *p_src, uint8_t *p_src_uv, int src_pitch,
                        uint8_t *p_dst, uint8_t *p_dst_uv, int pitch, int width, int height, float fx_scale, float fy_scale) {

    int ix = blockIdx.x * blockDim.x + threadIdx.x,
        iy = blockIdx.y * blockDim.y + threadIdx.y;

    if (ix >= width / 2 || iy >= height / 2) {
        return;
    }

    int x = ix * 2, y = iy * 2;
    typedef decltype(YuvUnitx2::x) YuvUnit;
    uint8_t *p_src_y = p_src + src_pitch * static_cast<uint32_t>(fmaf(y, fy_scale, 0.5 * fy_scale));
    *(YuvUnitx2 *)(p_dst + y * pitch + x * sizeof(YuvUnit)) = YuvUnitx2 {
        *(YuvUnit *)(p_src_y + static_cast<uint>(fmaf(x, fx_scale, 0.5 * fx_scale)) * sizeof(YuvUnit)),
        *(YuvUnit *)(p_src_y + static_cast<uint>(fmaf(x + 1, fx_scale, 0.5 * fx_scale) * sizeof(YuvUnit)))
    };
    y++;
    p_src_y = p_src + src_pitch * static_cast<uint32_t>(fmaf(y, fy_scale, 0.5 * fy_scale));
    *(YuvUnitx2 *)(p_dst + y * pitch + x * sizeof(YuvUnit)) = YuvUnitx2 {
        *(YuvUnit *)(p_src_y + static_cast<uint>(fmaf(x, fx_scale, 0.5 * fx_scale)) * sizeof(YuvUnit)),
        *(YuvUnit *)(p_src_y + static_cast<uint>(fmaf(x + 1, fx_scale, 0.5 * fx_scale)) * sizeof(YuvUnit))
    };
    YuvUnit *p_uv = (YuvUnit *) (p_src_uv + static_cast<uint>(fmaf(ix, fx_scale, fx_scale * 0.5)) * sizeof(YuvUnit) * 2 + 
                            src_pitch * static_cast<uint>(fmaf(iy, fy_scale, 0.5 * fy_scale)));
    *(YuvUnitx2 *)(p_dst_uv + iy * pitch + ix * 2 * sizeof(YuvUnit)) = YuvUnitx2{ (YuvUnit)p_uv[0], (YuvUnit)p_uv[1] };
}


template <typename YuvUnitx2>
static void Resize(unsigned char *p_dst, unsigned char* p_dst_uv, int dst_pitch, int dst_width, int dst_height, 
                    unsigned char *p_src, unsigned char *p_src_uv, int src_pitch, int src_width, int src_height, hipStream_t hip_stream) {                        
    ResizeHip<YuvUnitx2> <<<dim3((dst_width + 31) / 32, (dst_height + 31) / 32), dim3(16, 16), 0, hip_stream >>>(p_src, p_src_uv, src_pitch, p_dst, p_dst_uv,
        dst_pitch, dst_width, dst_height, 1.0f * src_width / dst_width, 1.0f * src_height / dst_height);
}

void ResizeNv12(unsigned char *p_dst_nv12, int dst_pitch, int dst_width, int dst_height, unsigned char *p_src_nv12, 
                int src_pitch, int src_width, int src_height, unsigned char* p_src_nv12_uv, unsigned char* p_dst_nv12_uv, hipStream_t hip_stream)
{
    unsigned char* p_src_uv = p_src_nv12_uv ? p_src_nv12_uv : p_src_nv12 + (src_pitch*src_height);
    unsigned char* p_dst_uv = p_dst_nv12_uv ? p_dst_nv12_uv : p_dst_nv12 + (dst_pitch*dst_height);
    return Resize<uchar2>(p_dst_nv12, p_dst_uv, dst_pitch, dst_width, dst_height, p_src_nv12, p_src_uv, src_pitch, src_width, src_height, hip_stream);
}


void ResizeP016(unsigned char *p_dst_p016, int dst_pitch, int dst_width, int dst_height, unsigned char *p_src_p016,
               int src_pitch, int src_width, int src_height, unsigned char* p_src_p016_uv, unsigned char* p_dst_p016_uv, hipStream_t hip_stream)
{
    unsigned char* p_src_uv = p_src_p016_uv ? p_src_p016_uv : p_src_p016 + (src_pitch*src_height);
    unsigned char* p_dst_uv = p_dst_p016_uv ? p_dst_p016_uv : p_dst_p016 + (dst_pitch*dst_height);
    return Resize<ushort2>(p_dst_p016, p_dst_uv, dst_pitch, dst_width, dst_height, p_src_p016, p_src_uv, src_pitch, src_width, src_height, hip_stream);
}

static __global__ void Scale(uint8_t *p_src, int src_pitch, uint8_t *p_dst, int pitch, int width, 
                            int height, float fx_scale, float fy_scale) {
    int x = blockIdx.x * blockDim.x + threadIdx.x,
        y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height){
        return;
    }

    // do nearest neighbor interpolation
    uint8_t *p_src_xy = p_src + src_pitch * static_cast<uint>(fmaf(y, fy_scale, 0.5 * fy_scale)) + static_cast<uint>(fmaf(x, fx_scale, 0.5*fx_scale));
    *(uint8_t*)(p_dst + (y * pitch) + x) = *p_src_xy;
}

static __global__ void Scale_UV(uint8_t *p_src, int src_pitch, uint8_t *p_dst, int pitch, int width,
                                int height, float fx_scale, float fy_scale) {
    int x = blockIdx.x * blockDim.x + threadIdx.x,
        y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) {
        return;
    }
    // do nearest neighbor interpolation
    uint8_t *p_src_uv = p_src + src_pitch * static_cast<uint>(fmaf(y , fy_scale, 0.5 * fy_scale)) + static_cast<uint>(fmaf(x, fx_scale, 0.5 * fx_scale)) * 2;
    uchar2 dst_uv = uchar2{ p_src_uv[0], p_src_uv[1] };
    *(uchar2*)(p_dst + (y * pitch) + 2 * x) = dst_uv;
}

/**
 * @brief Resize a single plane of Y/U/V or UV interleaved (reserved for future)
 * 
 * @param dp_dst    - dest pointer
 * @param dst_pitch - Pitch of the dst plane
 * @param dst_width - Width of the dst plane
 * @param dst_height - Height of the dst plane
 * @param dp_src     - source pointer
 * @param src_pitch  - source pitch
 * @param src_width  - source width
 * @param src_height - source height
 * @param b_resize_uv - to resize UV plance or not
 * @param hip_stream    - Stream for launching the kernel   
 */
void ResizeYUVHipLaunchKernel(uint8_t *dp_dst, int dst_pitch, int dst_width, int dst_height, uint8_t *dp_src, int src_pitch, 
                                    int src_width, int src_height, bool b_resize_uv, hipStream_t hip_stream) {

    dim3 blockSize(16, 16, 1);
    dim3 gridSize(((uint32_t)dst_width + blockSize.x - 1) / blockSize.x, ((uint32_t)dst_height + blockSize.y - 1) / blockSize.y, 1);

    if (b_resize_uv) {
        Scale_UV <<<gridSize, blockSize, 0, hip_stream >>>(dp_src, src_pitch, dp_dst,
            dst_pitch, dst_width, dst_height, 1.0f * src_width / dst_width, 1.0f * src_height / dst_height);
    }
    else {
        Scale <<<gridSize, blockSize, 0, hip_stream >>>(dp_src, src_pitch, dp_dst,
            dst_pitch, dst_width, dst_height, 1.0f * src_width / dst_width, 1.0f * src_height / dst_height);
    }
}

void ResizeYUV420(uint8_t *p_dst_y,
                uint8_t* p_dst_u,
                uint8_t* p_dst_v,
                int dst_pitch_y,
                int dst_pitch_uv,
                int dst_width,
                int dst_height,
                uint8_t *p_src_y,
                uint8_t* p_src_u,
                uint8_t* p_src_v,
                int src_pitch_y,
                int src_pitch_uv,
                int src_width,
                int src_height,
                bool b_nv12,
                hipStream_t hip_stream) {

    int uv_width_dst = (dst_width + 1) >> 1;
    int uv_height_dst = (dst_width + 1) >> 1;
    int uv_width_src = (src_width + 1) >> 1;
    int uv_height_src = (src_height + 1) >> 1;

    // Scale Y plane
    ResizeYUVHipLaunchKernel(p_dst_y, dst_pitch_y, dst_width, dst_height, p_src_y, src_pitch_y, src_width, src_height, 0, hip_stream);
    if (b_nv12) {
        ResizeYUVHipLaunchKernel(p_dst_u, dst_pitch_uv, uv_width_dst, uv_height_dst, p_src_u, src_pitch_uv, uv_width_src, uv_height_src, b_nv12, hip_stream);
    } else {
        ResizeYUVHipLaunchKernel(p_dst_u, dst_pitch_uv, uv_width_dst, uv_height_dst, p_src_u, src_pitch_uv, uv_width_src, uv_height_src, b_nv12, hip_stream);
        ResizeYUVHipLaunchKernel(p_dst_v, dst_pitch_uv, uv_width_dst, uv_height_dst, p_src_v, src_pitch_uv, uv_width_src, uv_height_src, b_nv12, hip_stream);
    }
}

