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


/**
 * @brief Function to resize both planes of an NV12 image
 * 
 * 
 * @param p_dst_nv12 - destination pointer Y plane
 * @param dst_pitch - destination pitch
 * @param dst_width - destination width
 * @param dst_height - destination height
 * @param p_src_nv12 - source pointer
 * @param src_pitch - source pitch
 * @param src_width - source width
 * @param src_height - source height
 * @param p_src_nv12_uv - source pointer of UV plane
 * @param hip_stream    - Stream for launching the kernel
 */
void ResizeNv12(uint8_t *p_dst_nv12, int dst_pitch, int dst_width, int dst_height, uint8_t *p_src_nv12, 
                int src_pitch, int src_width, int src_height, unsigned char* p_src_nv12_uv, unsigned char* p_dst_nv12_uv, hipStream_t hip_stream);

/**
 * @brief 
 * 
 * @param p_dst_p016 
 * @param dst_pitch 
 * @param dst_width 
 * @param dst_height 
 * @param p_src_p016 
 * @param src_pitch 
 * @param src_width 
 * @param src_height 
 * @param p_src_p016_uv 
 * @param p_dst_p016_uv 
 * @param hip_stream    - Stream for launching the kernel
 */
void ResizeP016(uint8_t *p_dst_p016, int dst_pitch, int dst_width, int dst_height, uint8_t *p_src_p016, int src_pitch, 
            int src_width, int src_height, unsigned char* p_src_p016_uv, unsigned char* p_dst_p016_uv, hipStream_t hip_stream);

/**
 * @brief Function to resize 420 YUV image
 * 
 * @param p_dst_y  - Destination Y plane pointer
 * @param p_dst_u  - Destination U plane pointer
 * @param p_dst_v  - Destination V plane pointer
 * @param dst_pitch_y   - Destination Pitch Y
 * @param dst_pitch_uv  - Destination Pitch UV
 * @param dst_width     - Destination Width
 * @param dst_height    - Destination Height
 * @param p_src_y       - Src Y plane pointer
 * @param p_src_u       - Src U plane pointer
 * @param p_src_v       - Src V plane pointer
 * @param src_pitch_y   - Src Pitch Y
 * @param src_pitch_uv  - Src Pitch UV
 * @param src_width     - Src Width
 * @param src_height    - Src Height
 * @param b_nv12        - Is uv interleaved?                   
 * @param hip_stream    - Stream for launching the kernel   
 */
void ResizeYUV420(uint8_t *p_dst_y, uint8_t* p_dst_u, uint8_t* p_dst_v, int dst_pitch_y, int dst_pitch_uv, 
                int dst_width, int dst_height, uint8_t *p_src_y, uint8_t* p_src_u, uint8_t* p_src_v,
                int src_pitch_y, int src_pitch_uv, int src_width, int src_height, bool b_nv12 = false, hipStream_t hip_stream = nullptr);

/**
 * @brief The function to launch ResizeYUV HIP kernel
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
                                    int src_width, int src_height, bool b_resize_uv = false, hipStream_t hip_stream = nullptr);
