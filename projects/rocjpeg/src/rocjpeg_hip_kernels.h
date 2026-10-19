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

#ifndef ROC_JPEG_HIP_KERNELS_H_
#define ROC_JPEG_HIP_KERNELS_H_

#pragma once

#include <hip/hip_runtime.h>

/**
 * @brief Compile-time selection of the YUV-to-RGB color conversion standard.
 *
 * Define CC_STANDARD at build time to choose the coefficient set:
 *   CC_STANDARD_BT601 (default) - ITU-R BT.601 full-range (JFIF/JPEG)
 *   CC_STANDARD_BT709           - ITU-R BT.709 full-range
 *
 * The coefficients expand to float literals, so the compiler keeps them as
 * immediate operands in the fmaf instructions (no memory load, no register
 * pressure, identical codegen to hardcoded constants).
 */
#define CC_STANDARD_BT601 601
#define CC_STANDARD_BT709 709

#ifndef CC_STANDARD
#define CC_STANDARD CC_STANDARD_BT601
#endif

#if CC_STANDARD == CC_STANDARD_BT601
#define CC_CR0  0.0000f
#define CC_CR1  1.4020f
#define CC_CG0 -0.3441f
#define CC_CG1 -0.7141f
#define CC_CB0  1.7720f
#define CC_CB1  0.0000f
#elif CC_STANDARD == CC_STANDARD_BT709
#define CC_CR0  0.0000f
#define CC_CR1  1.5748f
#define CC_CG0 -0.1873f
#define CC_CG1 -0.4681f
#define CC_CB0  1.8556f
#define CC_CB1  0.0000f
#else
#error "Unsupported CC_STANDARD: use CC_STANDARD_BT601 or CC_STANDARD_BT709"
#endif

/**
 * ---- Batched kernel param structs + prototypes ----
 *
 * Every conversion kernel is batched: one launch processes `n_images` images,
 * each described by one entry of the `d_params` device array and selected inside
 * the kernel by hipBlockIdx_z. `max_gx` / `max_gy` are the maximum per-image grid
 * extents across the group. A single-image conversion is simply n_images == 1.
 */

struct RGBAToRGBBatchParams {
    uint32_t dst_width;
    uint32_t dst_height;
    uint8_t *dst_image;
    uint32_t dst_image_stride_in_bytes;
    const uint8_t *src_image;
    uint32_t src_image_stride_in_bytes;
};
void ColorConvertRGBAToRGBBatched(hipStream_t stream, uint32_t max_gx, uint32_t max_gy,
    const RGBAToRGBBatchParams *d_params, uint32_t n_images);

struct InterleavedUVToPlanarUVBatchParams {
    uint32_t dst_width;
    uint32_t dst_height;
    uint8_t *dst_image1;
    uint8_t *dst_image2;
    uint32_t dst_image_stride_in_bytes;
    const uint8_t *src_image;
    uint32_t src_image_stride_in_bytes;
};
void ConvertInterleavedUVToPlanarUVBatched(hipStream_t stream, uint32_t max_gx, uint32_t max_gy,
    const InterleavedUVToPlanarUVBatchParams *d_params, uint32_t n_images);

struct YFromPackedYUYVBatchParams {
    uint32_t dst_height;
    uint8_t *destination_y;
    uint32_t dst_luma_stride_in_bytes;
    const uint8_t *src_image;
    uint32_t src_image_stride_in_bytes;
    uint32_t dst_width_in_8px_blocks; // Destination width in 8-pixel-wide blocks; each thread writes 8 luma samples of one row.
};
void ExtractYFromPackedYUYVBatched(hipStream_t stream, uint32_t max_gx, uint32_t max_gy,
    const YFromPackedYUYVBatchParams *d_params, uint32_t n_images);

struct PackedYUYVToPlanarYUVBatchParams {
    uint32_t dst_height;
    uint8_t *destination_y;
    uint8_t *destination_u;
    uint8_t *destination_v;
    uint32_t dst_luma_stride_in_bytes;
    uint32_t dst_chroma_stride_in_bytes;
    const uint8_t *src_image;
    uint32_t src_image_stride_in_bytes;
    uint32_t dst_width_in_8px_blocks; // Destination width in 8-pixel-wide blocks; each thread writes 8 luma samples of one row.
};
void ConvertPackedYUYVToPlanarYUVBatched(hipStream_t stream, uint32_t max_gx, uint32_t max_gy,
    const PackedYUYVToPlanarYUVBatchParams *d_params, uint32_t n_images);

// ---- Unified YUV->RGB batched kernels (cover NV12/YUV444/YUV440/YUYV/YUV400) ----

// Per-image source layout selector for the unified kernels.
enum YUVSurfaceLayout : uint32_t {
    YUV_LAYOUT_NV12 = 0,   // 2-plane: luma + interleaved UV (4:2:0)
    YUV_LAYOUT_YUV444,     // 3-plane 4:4:4
    YUV_LAYOUT_YUV440,     // 3-plane 4:4:0 (422V), chroma half-height
    YUV_LAYOUT_YUYV,       // 1-plane packed YUYV (4:2:2)
    YUV_LAYOUT_YUV400,     // 1-plane luma only (grayscale)
};

// Superset of per-image source params; which fields are read depends on `layout`.
// Each thread of the unified kernels converts an 8-pixel-wide by 2-row block, so the
// "*_in_8px_blocks" / "*_in_2row_blocks" extents are the per-image grid bounds and the
// "*_row_pair_stride_in_bytes" values are the byte distance between consecutive row pairs.
struct YUVToRGBBatchParams {
    uint32_t       layout;
    uint8_t       *dst_image;
    uint32_t       dst_image_stride_in_bytes;
    uint32_t       dst_image_row_pair_stride_in_bytes; // dst_image_stride_in_bytes * 2 (one thread writes 2 rows)
    const uint8_t *src_y_image;
    uint32_t       src_y_image_stride_in_bytes;
    uint32_t       src_y_image_row_pair_stride_in_bytes; // src_y_image_stride_in_bytes * 2 (one thread reads 2 rows)
    const uint8_t *src_u_image;                 // YUV444 / YUV440
    const uint8_t *src_v_image;                 // YUV444 / YUV440
    const uint8_t *src_chroma_image;            // NV12 interleaved UV
    uint32_t       src_chroma_image_stride_in_bytes; // NV12
    uint32_t       dst_width_in_8px_blocks;  // ceil(dst_width / 8)
    uint32_t       dst_height_in_2row_blocks; // ceil(dst_height / 2)
};

struct YUVToRGBPlanarBatchParams {
    uint32_t       layout;
    uint8_t       *dst_image_r;
    uint8_t       *dst_image_g;
    uint8_t       *dst_image_b;
    uint32_t       dst_image_stride_in_bytes;
    uint32_t       dst_image_row_pair_stride_in_bytes; // dst_image_stride_in_bytes * 2 (one thread writes 2 rows)
    const uint8_t *src_y_image;
    uint32_t       src_y_image_stride_in_bytes;
    uint32_t       src_y_image_row_pair_stride_in_bytes; // src_y_image_stride_in_bytes * 2 (one thread reads 2 rows)
    const uint8_t *src_u_image;
    const uint8_t *src_v_image;
    const uint8_t *src_chroma_image;
    uint32_t       src_chroma_image_stride_in_bytes;
    uint32_t       dst_width_in_8px_blocks;  // ceil(dst_width / 8)
    uint32_t       dst_height_in_2row_blocks; // ceil(dst_height / 2)
};

void ColorConvertYUVToRGBBatched(hipStream_t stream, uint32_t max_gx, uint32_t max_gy,
    const YUVToRGBBatchParams *d_params, uint32_t n_images);
void ColorConvertYUVToRGBPlanarBatched(hipStream_t stream, uint32_t max_gx, uint32_t max_gy,
    const YUVToRGBPlanarBatchParams *d_params, uint32_t n_images);

/**
 * @brief Structure representing an array of 6 unsigned integers.
 *
 * This structure is used to store an array of 6 unsigned integers.
 * The `data` member is an array of size 6 that holds the integer values.
 */
typedef struct UINT6TYPE {
  uint data[6];
} DUINT6;

/**
 * @brief Represents a struct that holds an array of 8 unsigned integers.
 *
 * This struct is used to store an array of 8 unsigned integers in the `data` member.
 * It is typically used in the context of the `rocjpeg_hip_kernels` module.
 */
typedef struct UINT8TYPE {
  uint data[8];
} DUINT8;

#endif //ROC_JPEG_HIP_KERNELS_H_