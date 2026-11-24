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

// ============================================================================
// Batched kernels: one kernel launch converts a group of images. hipBlockIdx_z
// selects the per-image parameter struct, whose fields are aliased to the
// per-image args at the top of each body. A single-image conversion is just a
// group of one, so these are the only conversion kernels in the library.
// ============================================================================

// Writes the first `num_bytes` bytes of `src` to `dst`. Each thread produces a fixed-size block of
// pixels, so the last block of a row can extend past the image width; those blocks store only
// the bytes inside the image instead of writing into the row padding, the next row, or past the
// end of the destination buffer.
__device__ __forceinline__ void StorePartial(uint8_t *dst, const void *src, uint32_t num_bytes) {
    const uint8_t *src_bytes = reinterpret_cast<const uint8_t *>(src);
    for (uint32_t i = 0; i < num_bytes; i++) {
        dst[i] = src_bytes[i];
    }
}

// Reads the first `num_bytes` (at most 16) bytes at `src` into a uint4 whose other bytes are 0. The odd-left YUYV
// kernels read their source from one pixel before the crop, so in the last block of a row they read only the
// macropixels holding its pixels: a whole 16-byte block could run past the end of the decoded surface.
__device__ __forceinline__ uint4 LoadPartialUint4(const uint8_t *src, uint32_t num_bytes) {
    uint32_t words[4] = {0, 0, 0, 0};
    for (uint32_t i = 0; i < num_bytes; i++) {
        words[i >> 2] |= static_cast<uint32_t>(src[i]) << ((i & 3) * 8);
    }
    return make_uint4(words[0], words[1], words[2], words[3]);
}

__global__ void ColorConvertRGBAToRGBBatchedKernel(const RGBAToRGBBatchParams *params, uint32_t n_images) {
    if (hipBlockIdx_z >= n_images) return;
    const RGBAToRGBBatchParams &p = params[hipBlockIdx_z];
    uint32_t dst_width = p.dst_width;
    uint32_t dst_height = p.dst_height;
    uint8_t *dst_image = p.dst_image;
    uint32_t dst_image_stride_in_bytes = p.dst_image_stride_in_bytes;
    const uint8_t *src_image = p.src_image;
    uint32_t src_image_stride_in_bytes = p.src_image_stride_in_bytes;

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

    uint32_t num_pixels = dst_width - x;
    if (num_pixels >= 8) {
        *((DUINT6 *)(&dst_image[dst_idx])) = dst;
    } else {
        StorePartial(&dst_image[dst_idx], &dst, num_pixels * 3);
    }
}

void ColorConvertRGBAToRGBBatched(hipStream_t stream, uint32_t max_gx, uint32_t max_gy,
    const RGBAToRGBBatchParams *d_params, uint32_t n_images) {
    uint32_t grid_x = static_cast<uint32_t>(ceil(static_cast<float>(max_gx) / 16));
    uint32_t grid_y = static_cast<uint32_t>(ceil(static_cast<float>(max_gy) / 16));
    ColorConvertRGBAToRGBBatchedKernel<<<dim3(grid_x, grid_y, n_images),
        dim3(16, 16, 1), 0, stream>>>(d_params, n_images);
}

__global__ void ConvertInterleavedUVToPlanarUVBatchedKernel(const InterleavedUVToPlanarUVBatchParams *params, uint32_t n_images) {
    if (hipBlockIdx_z >= n_images) return;
    const InterleavedUVToPlanarUVBatchParams &p = params[hipBlockIdx_z];
    uint32_t dst_width = p.dst_width;
    uint32_t dst_height = p.dst_height;
    uint8_t *dst_image1 = p.dst_image1;
    uint8_t *dst_image2 = p.dst_image2;
    uint32_t dst_image_stride_in_bytes = p.dst_image_stride_in_bytes;
    const uint8_t *src_image = p.src_image;
    uint32_t src_image_stride_in_bytes = p.src_image_stride_in_bytes;

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

    uint32_t num_pixels = dst_width - x;
    if (num_pixels >= 8) {
        *((uint2 *)(&dst_image1[dst_idx])) = dst1;
        *((uint2 *)(&dst_image2[dst_idx])) = dst2;
    } else {
        StorePartial(&dst_image1[dst_idx], &dst1, num_pixels);
        StorePartial(&dst_image2[dst_idx], &dst2, num_pixels);
    }
}

void ConvertInterleavedUVToPlanarUVBatched(hipStream_t stream, uint32_t max_gx, uint32_t max_gy,
    const InterleavedUVToPlanarUVBatchParams *d_params, uint32_t n_images) {
    uint32_t grid_x = static_cast<uint32_t>(ceil(static_cast<float>(max_gx) / 16));
    uint32_t grid_y = static_cast<uint32_t>(ceil(static_cast<float>(max_gy) / 16));
    ConvertInterleavedUVToPlanarUVBatchedKernel<<<dim3(grid_x, grid_y, n_images),
        dim3(16, 16, 1), 0, stream>>>(d_params, n_images);
}

__global__ void ExtractYFromPackedYUYVBatchedKernel(const YFromPackedYUYVBatchParams *params, uint32_t n_images) {
    if (hipBlockIdx_z >= n_images) return;
    const YFromPackedYUYVBatchParams &p = params[hipBlockIdx_z];
    uint32_t dst_height = p.dst_height;
    uint8_t *destination_y = p.destination_y;
    uint32_t dst_luma_stride_in_bytes = p.dst_luma_stride_in_bytes;
    const uint8_t *src_image = p.src_image;
    uint32_t src_image_stride_in_bytes = p.src_image_stride_in_bytes;
    uint32_t dst_width_in_8px_blocks = p.dst_width_in_8px_blocks;

    uint32_t x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    uint32_t y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    if (x < dst_width_in_8px_blocks && y < dst_height) {
        uint32_t src_idx = y * src_image_stride_in_bytes + (x << 4);
        uint32_t dst_idx = y * dst_luma_stride_in_bytes + (x << 3);

        uint4 src = *((uint4 *)(&src_image[src_idx]));
        uint2 dst_y;
        dst_y.x = hipPack(make_float4(hipUnpack0(src.x), hipUnpack2(src.x), hipUnpack0(src.y), hipUnpack2(src.y)));
        dst_y.y = hipPack(make_float4(hipUnpack0(src.z), hipUnpack2(src.z), hipUnpack0(src.w), hipUnpack2(src.w)));

        uint32_t num_pixels = p.dst_width - (x << 3);
        if (num_pixels >= 8) {
            *((uint2 *)(&destination_y[dst_idx])) = dst_y;
        } else {
            StorePartial(&destination_y[dst_idx], &dst_y, num_pixels);
        }
    }
}

void ExtractYFromPackedYUYVBatched(hipStream_t stream, uint32_t max_gx, uint32_t max_gy,
    const YFromPackedYUYVBatchParams *d_params, uint32_t n_images) {
    uint32_t grid_x = static_cast<uint32_t>(ceil(static_cast<float>(max_gx) / 16));
    uint32_t grid_y = static_cast<uint32_t>(ceil(static_cast<float>(max_gy) / 4));
    ExtractYFromPackedYUYVBatchedKernel<<<dim3(grid_x, grid_y, n_images),
        dim3(16, 4, 1), 0, stream>>>(d_params, n_images);
}

// kOddLeft: the first destination pixel is the second pixel of the source macropixel (a crop with an odd left edge).
template <bool kOddLeft>
__global__ void ConvertPackedYUYVToPlanarYUVBatchedKernel(const PackedYUYVToPlanarYUVBatchParams *params, uint32_t n_images) {
    if (hipBlockIdx_z >= n_images) return;
    const PackedYUYVToPlanarYUVBatchParams &p = params[hipBlockIdx_z];
    uint32_t dst_height = p.dst_height;
    uint8_t *destination_y = p.destination_y;
    uint8_t *destination_u = p.destination_u;
    uint8_t *destination_v = p.destination_v;
    uint32_t dst_luma_stride_in_bytes = p.dst_luma_stride_in_bytes;
    uint32_t dst_chroma_stride_in_bytes = p.dst_chroma_stride_in_bytes;
    const uint8_t *src_image = p.src_image;
    uint32_t src_image_stride_in_bytes = p.src_image_stride_in_bytes;
    uint32_t dst_width_in_8px_blocks = p.dst_width_in_8px_blocks;

    uint32_t x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    uint32_t y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    if ((x < dst_width_in_8px_blocks && y < dst_height)) {
        uint32_t src_idx = y * src_image_stride_in_bytes + (x << 4);
        uint32_t dst_y_idx = y * dst_luma_stride_in_bytes + (x << 3);
        uint32_t dst_uv_idx = y * dst_chroma_stride_in_bytes + (x << 2);

        uint4 src;
        if (kOddLeft && p.dst_width - (x << 3) < 8) {
            // Pixel i of the block is the luma at byte 2 * i + 2, so the last block of a row reads only the
            // macropixels up to the one holding its last pixel.
            src = LoadPartialUint4(&src_image[src_idx], ((p.dst_width - (x << 3)) >> 1) * 4 + 4);
        } else {
            src = *((uint4 *)(&src_image[src_idx]));
        }
        uint2 dst_y;
        uint32_t dst_u, dst_v;

        if (kOddLeft) {
            // Luma is bytes 2, 4, ..., 16; chroma still comes from the macropixels holding pixels 0, 2, 4 and 6.
            // Byte 16 is the luma of pixel 7, so it is read only when the block has all 8 pixels.
            float y8 = (p.dst_width - (x << 3) >= 8) ? static_cast<float>(src_image[src_idx + 16]) : 0.0f;
            dst_y.x = hipPack(make_float4(hipUnpack2(src.x), hipUnpack0(src.y), hipUnpack2(src.y), hipUnpack0(src.z)));
            dst_y.y = hipPack(make_float4(hipUnpack2(src.z), hipUnpack0(src.w), hipUnpack2(src.w), y8));
        } else {
            dst_y.x = hipPack(make_float4(hipUnpack0(src.x), hipUnpack2(src.x), hipUnpack0(src.y), hipUnpack2(src.y)));
            dst_y.y = hipPack(make_float4(hipUnpack0(src.z), hipUnpack2(src.z), hipUnpack0(src.w), hipUnpack2(src.w)));
        }
        dst_u = hipPack(make_float4(hipUnpack1(src.x), hipUnpack1(src.y), hipUnpack1(src.z), hipUnpack1(src.w)));
        dst_v = hipPack(make_float4(hipUnpack3(src.x), hipUnpack3(src.y), hipUnpack3(src.z), hipUnpack3(src.w)));

        uint32_t num_pixels = p.dst_width - (x << 3);
        if (num_pixels >= 8) {
            *((uint2 *)(&destination_y[dst_y_idx])) = dst_y;
            *((uint32_t *)(&destination_u[dst_uv_idx])) = dst_u;
            *((uint32_t *)(&destination_v[dst_uv_idx])) = dst_v;
        } else {
            // The chroma planes are (dst_width >> 1) wide, so this block has (dst_width >> 1) - 4 * x chroma samples left.
            uint32_t num_chroma = (p.dst_width >> 1) - (x << 2);
            StorePartial(&destination_y[dst_y_idx], &dst_y, num_pixels);
            StorePartial(&destination_u[dst_uv_idx], &dst_u, num_chroma);
            StorePartial(&destination_v[dst_uv_idx], &dst_v, num_chroma);
        }
    }
}

void ConvertPackedYUYVToPlanarYUVBatched(hipStream_t stream, uint32_t max_gx, uint32_t max_gy,
    const PackedYUYVToPlanarYUVBatchParams *d_params, uint32_t n_images) {
    uint32_t grid_x = static_cast<uint32_t>(ceil(static_cast<float>(max_gx) / 16));
    uint32_t grid_y = static_cast<uint32_t>(ceil(static_cast<float>(max_gy) / 4));
    ConvertPackedYUYVToPlanarYUVBatchedKernel<false><<<dim3(grid_x, grid_y, n_images),
        dim3(16, 4, 1), 0, stream>>>(d_params, n_images);
}

void ConvertPackedYUYVToPlanarYUVOddLeftBatched(hipStream_t stream, uint32_t max_gx, uint32_t max_gy,
    const PackedYUYVToPlanarYUVBatchParams *d_params, uint32_t n_images) {
    uint32_t grid_x = static_cast<uint32_t>(ceil(static_cast<float>(max_gx) / 16));
    uint32_t grid_y = static_cast<uint32_t>(ceil(static_cast<float>(max_gy) / 4));
    ConvertPackedYUYVToPlanarYUVBatchedKernel<true><<<dim3(grid_x, grid_y, n_images),
        dim3(16, 4, 1), 0, stream>>>(d_params, n_images);
}

__global__ void CopyYUYVOddLeftBatchedKernel(const YUYVOddLeftBatchParams *params, uint32_t n_images) {
    if (hipBlockIdx_z >= n_images) return;
    const YUYVOddLeftBatchParams &p = params[hipBlockIdx_z];
    uint32_t x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 8;
    uint32_t y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if (x >= p.dst_width || y >= p.dst_height) {
        return;
    }
    // Destination macropixel k is (Y, U, Y, V) = source bytes (4k + 2, 4k + 1, 4k + 4, 4k + 3), so the first n pixels
    // of a block use source bytes up to 2n.
    uint32_t src_idx = y * p.src_image_stride_in_bytes + (x << 1);
    uint32_t dst_idx = y * p.dst_image_stride_in_bytes + (x << 1);
    uint32_t num_pixels = p.dst_width - x;
    uint4 src;
    float y8 = 0.0f;
    if (num_pixels >= 8) {
        src = *((const uint4 *)(&p.src_image[src_idx]));
        y8 = static_cast<float>(p.src_image[src_idx + 16]);
    } else {
        // The last block of a row reads only the bytes its pixels use. Loading the whole block could read past the
        // end of the image, which the copy of a crop with an even left edge never does.
        uint32_t words[4] = {0, 0, 0, 0};
        for (uint32_t i = 0; i <= 2 * num_pixels; i++) {
            words[i >> 2] |= static_cast<uint32_t>(p.src_image[src_idx + i]) << ((i & 3) * 8);
        }
        src = make_uint4(words[0], words[1], words[2], words[3]);
    }
    uint4 dst;
    dst.x = hipPack(make_float4(hipUnpack2(src.x), hipUnpack1(src.x), hipUnpack0(src.y), hipUnpack3(src.x)));
    dst.y = hipPack(make_float4(hipUnpack2(src.y), hipUnpack1(src.y), hipUnpack0(src.z), hipUnpack3(src.y)));
    dst.z = hipPack(make_float4(hipUnpack2(src.z), hipUnpack1(src.z), hipUnpack0(src.w), hipUnpack3(src.z)));
    dst.w = hipPack(make_float4(hipUnpack2(src.w), hipUnpack1(src.w), y8, hipUnpack3(src.w)));
    if (num_pixels >= 8) {
        *((uint4 *)(&p.dst_image[dst_idx])) = dst;
    } else {
        StorePartial(&p.dst_image[dst_idx], &dst, num_pixels * 2);
    }
}

void CopyYUYVOddLeftBatched(hipStream_t stream, uint32_t max_gx, uint32_t max_gy,
    const YUYVOddLeftBatchParams *d_params, uint32_t n_images) {
    uint32_t grid_x = static_cast<uint32_t>(ceil(static_cast<float>(max_gx) / 16));
    uint32_t grid_y = static_cast<uint32_t>(ceil(static_cast<float>(max_gy) / 4));
    CopyYUYVOddLeftBatchedKernel<<<dim3(grid_x, grid_y, n_images),
        dim3(16, 4, 1), 0, stream>>>(d_params, n_images);
}

// ============================================================================
// Unified YUV->RGB batched kernels. A per-image `layout` tag (block-uniform,
// since hipBlockIdx_z fixes the image for a whole block) selects the source-load
// path; the color-conversion and the write are shared. Together these two kernels
// cover every supported YUV surface layout for both packed and planar RGB output.
// ============================================================================

// Reads the first `num_bytes` (at most 8) bytes at `src` into a uint2 whose other bytes are 0. In the last block of a
// row, the odd-crop chroma loads read only the bytes their pixels use: with an odd top edge they also read the chroma
// row below the ones the plain kernels read, which can be the last row of the surface, so a full load could read past
// its end.
__device__ __forceinline__ uint2 LoadPartialUint2(const uint8_t *src, uint32_t num_bytes) {
    uint32_t words[2] = {0, 0};
    for (uint32_t i = 0; i < num_bytes; i++) {
        words[i >> 2] |= static_cast<uint32_t>(src[i]) << ((i & 3) * 8);
    }
    return make_uint2(words[0], words[1]);
}

// Odd-crop loads for one 8-pixel row of an NV12 block whose first pixel is the second pixel of the U,V pair at `src`
// (x_phase 1): pixels 0..7 take their chroma from pairs 0, 1, 1, 2, 2, 3, 3, 4. With x_phase 0 they take it from
// pairs 0, 0, 1, 1, 2, 2, 3, 3, as the plain kernels do. Only the pairs of the first num_pixels pixels are read.
__device__ __forceinline__ void LoadNV12ChromaRowOddCrop(const uint8_t *src, uint32_t x_phase, uint32_t num_pixels, uint2 &u, uint2 &v) {
    uint2 uv;
    uint32_t uv4 = 0;
    if (num_pixels >= 8) {
        uv = *((const uint2 *)src);
        if (x_phase) {
            uv4 = *((const uint16_t *)(src + 8));
        }
    } else {
        // Pixel i takes its chroma from pair (i + x_phase) >> 1, so the last pixel needs pairs up to this one.
        uint32_t num_pairs = ((num_pixels - 1 + x_phase) >> 1) + 1;
        uv = LoadPartialUint2(src, num_pairs * 2);
    }
    if (x_phase) {
        u.x = hipPack(make_float4(hipUnpack0(uv.x), hipUnpack2(uv.x), hipUnpack2(uv.x), hipUnpack0(uv.y)));
        u.y = hipPack(make_float4(hipUnpack0(uv.y), hipUnpack2(uv.y), hipUnpack2(uv.y), hipUnpack0(uv4)));
        v.x = hipPack(make_float4(hipUnpack1(uv.x), hipUnpack3(uv.x), hipUnpack3(uv.x), hipUnpack1(uv.y)));
        v.y = hipPack(make_float4(hipUnpack1(uv.y), hipUnpack3(uv.y), hipUnpack3(uv.y), hipUnpack1(uv4)));
    } else {
        u.x = hipPack(make_float4(hipUnpack0(uv.x), hipUnpack0(uv.x), hipUnpack2(uv.x), hipUnpack2(uv.x)));
        u.y = hipPack(make_float4(hipUnpack0(uv.y), hipUnpack0(uv.y), hipUnpack2(uv.y), hipUnpack2(uv.y)));
        v.x = hipPack(make_float4(hipUnpack1(uv.x), hipUnpack1(uv.x), hipUnpack3(uv.x), hipUnpack3(uv.x)));
        v.y = hipPack(make_float4(hipUnpack1(uv.y), hipUnpack1(uv.y), hipUnpack3(uv.y), hipUnpack3(uv.y)));
    }
}

// Odd-crop loads for one 8-pixel row of a YUYV block whose first pixel is the second pixel of the macropixel at `src`
// (the odd-crop kernels only see YUYV with x_phase 1): luma is bytes 2, 4, ..., 16 and pixel i takes its chroma from
// macropixel (i + 1) >> 1. Only the macropixels holding the first num_pixels pixels are read, so the fifth one is read
// only for a block of 8 pixels.
__device__ __forceinline__ void LoadYUYVRowOddLeft(const uint8_t *src, uint32_t num_pixels, uint2 &yv, uint2 &u, uint2 &v) {
    uint4 l;
    uint32_t e = 0;
    if (num_pixels >= 8) {
        l = *((const uint4 *)src);
        e = *((const uint32_t *)(src + 16));
    } else {
        // The last pixel takes its chroma from macropixel num_pixels >> 1, which ends within the first 16 bytes.
        l = LoadPartialUint4(src, (num_pixels >> 1) * 4 + 4);
    }
    yv.x = hipPack(make_float4(hipUnpack2(l.x), hipUnpack0(l.y), hipUnpack2(l.y), hipUnpack0(l.z)));
    yv.y = hipPack(make_float4(hipUnpack2(l.z), hipUnpack0(l.w), hipUnpack2(l.w), hipUnpack0(e)));
    u.x = hipPack(make_float4(hipUnpack1(l.x), hipUnpack1(l.y), hipUnpack1(l.y), hipUnpack1(l.z)));
    u.y = hipPack(make_float4(hipUnpack1(l.z), hipUnpack1(l.w), hipUnpack1(l.w), hipUnpack1(e)));
    v.x = hipPack(make_float4(hipUnpack3(l.x), hipUnpack3(l.y), hipUnpack3(l.y), hipUnpack3(l.z)));
    v.y = hipPack(make_float4(hipUnpack3(l.z), hipUnpack3(l.w), hipUnpack3(l.w), hipUnpack3(e)));
}

// Fills the six per-2x8 sample vectors (y0,y1,u0,u1,v0,v1) from whichever source
// layout the `layout` tag denotes. All five layouts share the same 8-luma-per-thread,
// 2-rows-per-thread geometry, so only the load differs.
// kOddCrop: the image is cropped with a left or top edge inside a subsampled chroma sample, and x_phase / y_phase are
// its YUVToRGBBatchParams::src_x_phase / src_y_phase. num_pixels is the number of pixels of the block inside the
// destination (8 except in the last block of a row), and has_row1 is false for the last row pair of an odd-height
// destination, whose second row is outside the image: the odd-crop chroma loads read only chroma samples covering
// destination pixels.
template <bool kOddCrop>
__device__ __forceinline__ void LoadYUVSamples(uint32_t layout, uint32_t x_phase, uint32_t y_phase, uint32_t num_pixels, bool has_row1,
    const uint8_t *src_y_image, uint32_t src_y_stride, uint32_t src_y_row_pair_stride,
    const uint8_t *src_u_image, const uint8_t *src_v_image,
    const uint8_t *src_chroma_image, uint32_t src_chroma_stride,
    uint32_t x, uint32_t y,
    uint2 &y0, uint2 &y1, uint2 &u0, uint2 &u1, uint2 &v0, uint2 &v1) {

    if (kOddCrop && layout == YUV_LAYOUT_YUYV && x_phase) {
        uint32_t l0_idx = y * src_y_row_pair_stride + (x << 4);
        // Without the block's second row (has_row1 false), its samples are not stored, so the first source row is read
        // again instead of a row that can lie past the end of the decoded surface.
        uint32_t l1_idx = l0_idx + (has_row1 ? src_y_stride : 0);
        LoadYUYVRowOddLeft(&src_y_image[l0_idx], num_pixels, y0, u0, v0);
        LoadYUYVRowOddLeft(&src_y_image[l1_idx], num_pixels, y1, u1, v1);
        return;
    }
    if (layout == YUV_LAYOUT_YUYV) {
        uint32_t l0_idx = y * src_y_row_pair_stride + (x << 4);
        uint32_t l1_idx = l0_idx + src_y_stride;
        uint4 l0 = *((uint4 *)(&src_y_image[l0_idx]));
        uint4 l1 = *((uint4 *)(&src_y_image[l1_idx]));
        y0.x = hipPack(make_float4(hipUnpack0(l0.x), hipUnpack2(l0.x), hipUnpack0(l0.y), hipUnpack2(l0.y)));
        y0.y = hipPack(make_float4(hipUnpack0(l0.z), hipUnpack2(l0.z), hipUnpack0(l0.w), hipUnpack2(l0.w)));
        y1.x = hipPack(make_float4(hipUnpack0(l1.x), hipUnpack2(l1.x), hipUnpack0(l1.y), hipUnpack2(l1.y)));
        y1.y = hipPack(make_float4(hipUnpack0(l1.z), hipUnpack2(l1.z), hipUnpack0(l1.w), hipUnpack2(l1.w)));
        u0.x = hipPack(make_float4(hipUnpack1(l0.x), hipUnpack1(l0.x), hipUnpack1(l0.y), hipUnpack1(l0.y)));
        u0.y = hipPack(make_float4(hipUnpack1(l0.z), hipUnpack1(l0.z), hipUnpack1(l0.w), hipUnpack1(l0.w)));
        u1.x = hipPack(make_float4(hipUnpack1(l1.x), hipUnpack1(l1.x), hipUnpack1(l1.y), hipUnpack1(l1.y)));
        u1.y = hipPack(make_float4(hipUnpack1(l1.z), hipUnpack1(l1.z), hipUnpack1(l1.w), hipUnpack1(l1.w)));
        v0.x = hipPack(make_float4(hipUnpack3(l0.x), hipUnpack3(l0.x), hipUnpack3(l0.y), hipUnpack3(l0.y)));
        v0.y = hipPack(make_float4(hipUnpack3(l0.z), hipUnpack3(l0.z), hipUnpack3(l0.w), hipUnpack3(l0.w)));
        v1.x = hipPack(make_float4(hipUnpack3(l1.x), hipUnpack3(l1.x), hipUnpack3(l1.y), hipUnpack3(l1.y)));
        v1.y = hipPack(make_float4(hipUnpack3(l1.z), hipUnpack3(l1.z), hipUnpack3(l1.w), hipUnpack3(l1.w)));
        return;
    }

    // Y / luma plane load, common to NV12 / YUV444 / YUV440 / YUV400.
    uint32_t src_y0_idx = y * src_y_row_pair_stride + (x << 3);
    uint32_t src_y1_idx = src_y0_idx + src_y_stride;
    y0 = *((uint2 *)(&src_y_image[src_y0_idx]));
    y1 = *((uint2 *)(&src_y_image[src_y1_idx]));

    if (layout == YUV_LAYOUT_YUV444) {
        u0 = *((uint2 *)(&src_u_image[src_y0_idx]));
        u1 = *((uint2 *)(&src_u_image[src_y1_idx]));
        v0 = *((uint2 *)(&src_v_image[src_y0_idx]));
        v1 = *((uint2 *)(&src_v_image[src_y1_idx]));
    } else if (kOddCrop && layout == YUV_LAYOUT_YUV440) {
        // With y_phase, the block's second row takes its chroma from the next chroma row.
        uint32_t src_chroma0_idx = y * src_y_stride + (x << 3);
        uint32_t src_chroma1_idx = src_chroma0_idx + ((y_phase && has_row1) ? src_y_stride : 0);
        if (num_pixels >= 8) {
            u0 = *((uint2 *)(&src_u_image[src_chroma0_idx]));
            v0 = *((uint2 *)(&src_v_image[src_chroma0_idx]));
            u1 = *((uint2 *)(&src_u_image[src_chroma1_idx]));
            v1 = *((uint2 *)(&src_v_image[src_chroma1_idx]));
        } else {
            u0 = LoadPartialUint2(&src_u_image[src_chroma0_idx], num_pixels);
            v0 = LoadPartialUint2(&src_v_image[src_chroma0_idx], num_pixels);
            u1 = LoadPartialUint2(&src_u_image[src_chroma1_idx], num_pixels);
            v1 = LoadPartialUint2(&src_v_image[src_chroma1_idx], num_pixels);
        }
    } else if (kOddCrop && layout == YUV_LAYOUT_NV12) {
        // With y_phase, the block's second row takes its chroma from the next chroma row.
        uint32_t src_uv0_idx = y * src_chroma_stride + (x << 3);
        uint32_t src_uv1_idx = src_uv0_idx + ((y_phase && has_row1) ? src_chroma_stride : 0);
        LoadNV12ChromaRowOddCrop(&src_chroma_image[src_uv0_idx], x_phase, num_pixels, u0, v0);
        LoadNV12ChromaRowOddCrop(&src_chroma_image[src_uv1_idx], x_phase, num_pixels, u1, v1);
    } else if (layout == YUV_LAYOUT_YUV440) {
        uint32_t src_chroma_idx = y * src_y_stride + (x << 3);
        u0 = *((uint2 *)(&src_u_image[src_chroma_idx]));
        v0 = *((uint2 *)(&src_v_image[src_chroma_idx]));
        u1 = u0;
        v1 = v0;
    } else if (layout == YUV_LAYOUT_NV12) {
        uint32_t src_uv_idx = y * src_chroma_stride + (x << 3);
        uint2 uv = *((uint2 *)(&src_chroma_image[src_uv_idx]));
        float4 f;
        f.x = hipUnpack0(uv.x); f.y = f.x; f.z = hipUnpack2(uv.x); f.w = f.z; u0.x = hipPack(f);
        f.x = hipUnpack0(uv.y); f.y = f.x; f.z = hipUnpack2(uv.y); f.w = f.z; u0.y = hipPack(f);
        u1.x = u0.x; u1.y = u0.y;
        f.x = hipUnpack1(uv.x); f.y = f.x; f.z = hipUnpack3(uv.x); f.w = f.z; v0.x = hipPack(f);
        f.x = hipUnpack1(uv.y); f.y = f.x; f.z = hipUnpack3(uv.y); f.w = f.z; v0.y = hipPack(f);
        v1.x = v0.x; v1.y = v0.y;
    } else {  // YUV_LAYOUT_YUV400: no chroma -> neutral 128 gives R=G=B=Y.
        uint2 neutral;
        neutral.x = hipPack(make_float4(128.0f, 128.0f, 128.0f, 128.0f));
        neutral.y = neutral.x;
        u0 = neutral; u1 = neutral; v0 = neutral; v1 = neutral;
    }
}

// Shared YUV->RGB conversion for one 8-pixel-wide by 2-row block.
__device__ __forceinline__ void ConvertYUVToRGB(
    uint2 y0, uint2 y1, uint2 u0, uint2 u1, uint2 v0, uint2 v1,
    DUINT6 &rgb0, DUINT6 &rgb1) {
        float2 cr = make_float2(CC_CR0, CC_CR1);
        float2 cg = make_float2(CC_CG0, CC_CG1);
        float2 cb = make_float2(CC_CB0, CC_CB1);
        float3 yuv;
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
}

template <bool kOddCrop>
__global__ void ColorConvertYUVToRGBBatchedKernel(const YUVToRGBBatchParams *params, uint32_t n_images) {
    if (hipBlockIdx_z >= n_images) return;
    const YUVToRGBBatchParams &p = params[hipBlockIdx_z];

    uint32_t x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    uint32_t y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    if ((x < p.dst_width_in_8px_blocks) && (y < p.dst_height_in_2row_blocks)) {
        uint2 y0, y1, u0, u1, v0, v1;
        // The plain instantiation passes constants for the odd-crop arguments, so it compiles to the same code as without them.
        LoadYUVSamples<kOddCrop>(p.layout, kOddCrop ? p.src_x_phase : 0, kOddCrop ? p.src_y_phase : 0,
                          kOddCrop ? p.dst_width - (x << 3) : 8, !kOddCrop || (y * 2 + 1) < p.dst_height,
                          p.src_y_image, p.src_y_image_stride_in_bytes, p.src_y_image_row_pair_stride_in_bytes,
                          p.src_u_image, p.src_v_image, p.src_chroma_image, p.src_chroma_image_stride_in_bytes,
                          x, y, y0, y1, u0, u1, v0, v1);
        DUINT6 rgb0, rgb1;
        ConvertYUVToRGB(y0, y1, u0, u1, v0, v1, rgb0, rgb1);
        uint32_t rgb0_idx = y * p.dst_image_row_pair_stride_in_bytes + (x * 24);
        uint32_t rgb1_idx = rgb0_idx + p.dst_image_stride_in_bytes;
        // With an odd height, the last row pair has only its first row inside the image.
        bool has_row1 = (y * 2 + 1) < p.dst_height;
        uint32_t num_pixels = p.dst_width - (x << 3);
        if (num_pixels >= 8) {
            *((DUINT6 *)(&p.dst_image[rgb0_idx])) = rgb0;
            if (has_row1) *((DUINT6 *)(&p.dst_image[rgb1_idx])) = rgb1;
        } else {
            StorePartial(&p.dst_image[rgb0_idx], &rgb0, num_pixels * 3);
            if (has_row1) StorePartial(&p.dst_image[rgb1_idx], &rgb1, num_pixels * 3);
        }
    }
}

void ColorConvertYUVToRGBBatched(hipStream_t stream, uint32_t max_gx, uint32_t max_gy,
    const YUVToRGBBatchParams *d_params, uint32_t n_images) {
    uint32_t grid_x = static_cast<uint32_t>(ceil(static_cast<float>(max_gx) / 16));
    uint32_t grid_y = static_cast<uint32_t>(ceil(static_cast<float>(max_gy) / 4));
    ColorConvertYUVToRGBBatchedKernel<false><<<dim3(grid_x, grid_y, n_images),
        dim3(16, 4, 1), 0, stream>>>(d_params, n_images);
}

void ColorConvertYUVToRGBOddCropBatched(hipStream_t stream, uint32_t max_gx, uint32_t max_gy,
    const YUVToRGBBatchParams *d_params, uint32_t n_images) {
    uint32_t grid_x = static_cast<uint32_t>(ceil(static_cast<float>(max_gx) / 16));
    uint32_t grid_y = static_cast<uint32_t>(ceil(static_cast<float>(max_gy) / 4));
    ColorConvertYUVToRGBBatchedKernel<true><<<dim3(grid_x, grid_y, n_images),
        dim3(16, 4, 1), 0, stream>>>(d_params, n_images);
}

template <bool kOddCrop>
__global__ void ColorConvertYUVToRGBPlanarBatchedKernel(const YUVToRGBPlanarBatchParams *params, uint32_t n_images) {
    if (hipBlockIdx_z >= n_images) return;
    const YUVToRGBPlanarBatchParams &p = params[hipBlockIdx_z];

    uint32_t x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    uint32_t y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    if ((x < p.dst_width_in_8px_blocks) && (y < p.dst_height_in_2row_blocks)) {
        uint2 y0, y1, u0, u1, v0, v1;
        // The plain instantiation passes constants for the odd-crop arguments, so it compiles to the same code as without them.
        LoadYUVSamples<kOddCrop>(p.layout, kOddCrop ? p.src_x_phase : 0, kOddCrop ? p.src_y_phase : 0,
                          kOddCrop ? p.dst_width - (x << 3) : 8, !kOddCrop || (y * 2 + 1) < p.dst_height,
                          p.src_y_image, p.src_y_image_stride_in_bytes, p.src_y_image_row_pair_stride_in_bytes,
                          p.src_u_image, p.src_v_image, p.src_chroma_image, p.src_chroma_image_stride_in_bytes,
                          x, y, y0, y1, u0, u1, v0, v1);
        DUINT6 rgb0, rgb1;
        ConvertYUVToRGB(y0, y1, u0, u1, v0, v1, rgb0, rgb1);

        uint8_t *dst_image_r = p.dst_image_r;
        uint8_t *dst_image_g = p.dst_image_g;
        uint8_t *dst_image_b = p.dst_image_b;
        uint32_t rgb0_idx = y * p.dst_image_row_pair_stride_in_bytes + (x * 8);
        uint32_t rgb1_idx = rgb0_idx + p.dst_image_stride_in_bytes;
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

        // With an odd height, the last row pair has only its first row inside the image.
        bool has_row1 = (y * 2 + 1) < p.dst_height;
        uint32_t num_pixels = p.dst_width - (x << 3);
        if (num_pixels >= 8) {
            *((uint2 *)(&dst_image_r[rgb0_idx])) = red0;
            *((uint2 *)(&dst_image_g[rgb0_idx])) = green0;
            *((uint2 *)(&dst_image_b[rgb0_idx])) = blue0;
            if (has_row1) {
                *((uint2 *)(&dst_image_r[rgb1_idx])) = red1;
                *((uint2 *)(&dst_image_g[rgb1_idx])) = green1;
                *((uint2 *)(&dst_image_b[rgb1_idx])) = blue1;
            }
        } else {
            StorePartial(&dst_image_r[rgb0_idx], &red0, num_pixels);
            StorePartial(&dst_image_g[rgb0_idx], &green0, num_pixels);
            StorePartial(&dst_image_b[rgb0_idx], &blue0, num_pixels);
            if (has_row1) {
                StorePartial(&dst_image_r[rgb1_idx], &red1, num_pixels);
                StorePartial(&dst_image_g[rgb1_idx], &green1, num_pixels);
                StorePartial(&dst_image_b[rgb1_idx], &blue1, num_pixels);
            }
        }
    }
}

void ColorConvertYUVToRGBPlanarBatched(hipStream_t stream, uint32_t max_gx, uint32_t max_gy,
    const YUVToRGBPlanarBatchParams *d_params, uint32_t n_images) {
    uint32_t grid_x = static_cast<uint32_t>(ceil(static_cast<float>(max_gx) / 16));
    uint32_t grid_y = static_cast<uint32_t>(ceil(static_cast<float>(max_gy) / 4));
    ColorConvertYUVToRGBPlanarBatchedKernel<false><<<dim3(grid_x, grid_y, n_images),
        dim3(16, 4, 1), 0, stream>>>(d_params, n_images);
}

void ColorConvertYUVToRGBPlanarOddCropBatched(hipStream_t stream, uint32_t max_gx, uint32_t max_gy,
    const YUVToRGBPlanarBatchParams *d_params, uint32_t n_images) {
    uint32_t grid_x = static_cast<uint32_t>(ceil(static_cast<float>(max_gx) / 16));
    uint32_t grid_y = static_cast<uint32_t>(ceil(static_cast<float>(max_gy) / 4));
    ColorConvertYUVToRGBPlanarBatchedKernel<true><<<dim3(grid_x, grid_y, n_images),
        dim3(16, 4, 1), 0, stream>>>(d_params, n_images);
}