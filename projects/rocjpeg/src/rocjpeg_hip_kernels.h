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

#ifndef ROC_JPEG_HIP_KERNELS_H_
#define ROC_JPEG_HIP_KERNELS_H_

#pragma once

#include <hip/hip_runtime.h>

/**
 * @brief Converts YUV444 image to RGB image.
 *
 * This function takes a YUV444 image and converts it to an RGB image.
 *
 * @param stream The HIP stream to be used for the conversion.
 * @param dst_width The width of the destination RGB image.
 * @param dst_height The height of the destination RGB image.
 * @param dst_image Pointer to the destination RGB image buffer.
 * @param dst_image_stride_in_bytes The stride (in bytes) of the destination RGB image buffer.
 * @param src_yuv_image Pointer to the source YUV444 image buffer.
 * @param src_yuv_image_stride_in_bytes The stride (in bytes) of the source YUV444 image buffer.
 * @param src_u_image_offset The offset (in bytes) of the U component in the source YUV444 image buffer.
 */
void ColorConvertYUV444ToRGB(hipStream_t stream, uint32_t dst_width, uint32_t dst_height,
    uint8_t *dst_image, uint32_t dst_image_stride_in_bytes, const uint8_t *src_yuv_image,
    uint32_t src_yuv_image_stride_in_bytes, uint32_t src_u_image_offset, uint32_t src_v_image_offset);

/**
 * @brief Converts YUV440 image to RGB image.
 *
 * This function takes a YUV440 image and converts it to an RGB image.
 *
 * @param stream The HIP stream to be used for the conversion.
 * @param dst_width The width of the destination RGB image.
 * @param dst_height The height of the destination RGB image.
 * @param dst_image Pointer to the destination RGB image buffer.
 * @param dst_image_stride_in_bytes The stride (in bytes) of the destination RGB image buffer.
 * @param src_yuv_image Pointer to the source YUV440 image buffer.
 * @param src_yuv_image_stride_in_bytes The stride (in bytes) of the source YUV440 image buffer.
 * @param src_u_image_offset The offset (in bytes) of the U component in the source YUV440 image buffer.
 */
void ColorConvertYUV440ToRGB(hipStream_t stream, uint32_t dst_width, uint32_t dst_height,
    uint8_t *dst_image, uint32_t dst_image_stride_in_bytes, const uint8_t *src_yuv_image,
    uint32_t src_yuv_image_stride_in_bytes, uint32_t src_u_image_offset, uint32_t src_v_image_offset);

/**
 * @brief Converts an image in YUYV format to RGB format.
 *
 * This function takes an image in YUYV format and converts it to RGB format.
 * The converted image is stored in the destination buffer.
 *
 * @param stream The HIP stream to be used for the conversion.
 * @param dst_width The width of the destination image in pixels.
 * @param dst_height The height of the destination image in pixels.
 * @param dst_image Pointer to the destination image buffer.
 * @param dst_image_stride_in_bytes The stride (in bytes) of the destination image buffer.
 * @param src_image Pointer to the source image buffer in YUYV format.
 * @param src_image_stride_in_bytes The stride (in bytes) of the source image buffer.
 */
void ColorConvertYUYVToRGB(hipStream_t stream, uint32_t dst_width, uint32_t dst_height,
    uint8_t *dst_image, uint32_t dst_image_stride_in_bytes,
    const uint8_t *src_image, uint32_t src_image_stride_in_bytes);

/**
 * @brief Converts an NV12 image to RGB format.
 *
 * This function takes an NV12 image, consisting of a luma (Y) plane and a chroma (UV) plane,
 * and converts it to RGB format. The resulting RGB image is stored in the destination image buffer.
 *
 * @param stream The HIP stream to be used for the conversion.
 * @param dst_width The width of the destination RGB image.
 * @param dst_height The height of the destination RGB image.
 * @param dst_image Pointer to the destination RGB image buffer.
 * @param dst_image_stride_in_bytes The stride (in bytes) of the destination RGB image buffer.
 * @param src_luma_image Pointer to the source luma (Y) plane of the NV12 image.
 * @param src_luma_image_stride_in_bytes The stride (in bytes) of the source luma (Y) plane.
 * @param src_chroma_image Pointer to the source chroma (UV) plane of the NV12 image.
 * @param src_chroma_image_stride_in_bytes The stride (in bytes) of the source chroma (UV) plane.
 */
void ColorConvertNV12ToRGB(hipStream_t stream, uint32_t dst_width, uint32_t dst_height,
    uint8_t *dst_image, uint32_t dst_image_stride_in_bytes,
    const uint8_t *src_luma_image, uint32_t src_luma_image_stride_in_bytes,
    const uint8_t *src_chroma_image, uint32_t src_chroma_image_stride_in_bytes);

/**
 * @brief Converts a YUV400 image to RGB format.
 *
 * This function converts a YUV400 image to RGB format using the specified stream and parameters.
 *
 * @param stream The HIP stream to be used for the conversion.
 * @param dst_width The width of the destination RGB image.
 * @param dst_height The height of the destination RGB image.
 * @param dst_image Pointer to the destination RGB image buffer.
 * @param dst_image_stride_in_bytes The stride (in bytes) of the destination RGB image buffer.
 * @param src_luma_image Pointer to the source YUV400 luma image buffer.
 * @param src_luma_image_stride_in_bytes The stride (in bytes) of the source YUV400 luma image buffer.
 */
void ColorConvertYUV400ToRGB(hipStream_t stream, uint32_t dst_width, uint32_t dst_height,
    uint8_t *dst_image, uint32_t dst_image_stride_in_bytes,
    const uint8_t *src_luma_image, uint32_t src_luma_image_stride_in_bytes);

/**
 * @brief Converts an RGBA image to an RGB image.
 *
 * This function takes an input RGBA image and converts it to an output RGB image.
 * The conversion is performed in parallel using the HIP framework.
 *
 * @param stream The HIP stream to be used for the conversion.
 * @param dst_width The width of the output RGB image.
 * @param dst_height The height of the output RGB image.
 * @param dst_image Pointer to the output RGB image buffer.
 * @param dst_image_stride_in_bytes The stride (in bytes) of the output RGB image buffer.
 * @param src_image Pointer to the input RGBA image buffer.
 * @param src_image_stride_in_bytes The stride (in bytes) of the input RGBA image buffer.
 */
void ColorConvertRGBAToRGB(hipStream_t stream, uint32_t dst_width, uint32_t dst_height,
    uint8_t *dst_image, uint32_t dst_image_stride_in_bytes,
    const uint8_t *src_image, uint32_t src_image_stride_in_bytes);

/**
 * @brief Converts YUV444 image to RGB planar format.
 *
 * This function takes a YUV444 image and converts it to RGB planar format.
 * The resulting RGB image is stored in separate R, G, and B planes.
 *
 * @param stream The HIP stream to be used for the kernel execution.
 * @param dst_width The width of the destination RGB image.
 * @param dst_height The height of the destination RGB image.
 * @param dst_image_r Pointer to the destination R plane of the RGB image.
 * @param dst_image_g Pointer to the destination G plane of the RGB image.
 * @param dst_image_b Pointer to the destination B plane of the RGB image.
 * @param dst_image_stride_in_bytes The stride (in bytes) of the destination RGB image.
 * @param src_yuv_image Pointer to the source YUV444 image.
 * @param src_yuv_image_stride_in_bytes The stride (in bytes) of the source YUV444 image.
 * @param src_u_image_offset The offset (in bytes) of the U plane in the source YUV444 image.
 */
void ColorConvertYUV444ToRGBPlanar(hipStream_t stream, uint32_t dst_width, uint32_t dst_height,
    uint8_t *dst_image_r, uint8_t *dst_image_g, uint8_t *dst_image_b, uint32_t dst_image_stride_in_bytes, const uint8_t *src_yuv_image,
    uint32_t src_yuv_image_stride_in_bytes, uint32_t src_u_image_offset, uint32_t src_v_image_offset);

/**
 * @brief Converts YUV440 image to RGB planar format.
 *
 * This function takes a YUV440 image and converts it to RGB planar format.
 * The resulting RGB image is stored in separate R, G, and B planes.
 *
 * @param stream The HIP stream to be used for the kernel execution.
 * @param dst_width The width of the destination RGB image.
 * @param dst_height The height of the destination RGB image.
 * @param dst_image_r Pointer to the destination R plane of the RGB image.
 * @param dst_image_g Pointer to the destination G plane of the RGB image.
 * @param dst_image_b Pointer to the destination B plane of the RGB image.
 * @param dst_image_stride_in_bytes The stride (in bytes) of the destination RGB image.
 * @param src_yuv_image Pointer to the source YUV440 image.
 * @param src_yuv_image_stride_in_bytes The stride (in bytes) of the source YUV440 image.
 * @param src_u_image_offset The offset (in bytes) of the U plane in the source YUV440 image.
 */
void ColorConvertYUV440ToRGBPlanar(hipStream_t stream, uint32_t dst_width, uint32_t dst_height,
    uint8_t *dst_image_r, uint8_t *dst_image_g, uint8_t *dst_image_b, uint32_t dst_image_stride_in_bytes, const uint8_t *src_yuv_image,
    uint32_t src_yuv_image_stride_in_bytes, uint32_t src_u_image_offset, uint32_t src_v_image_offset);

/**
 * Converts a YUYV image to RGB planar format.
 *
 * This function takes a YUYV image and converts it to RGB planar format. The resulting RGB image
 * is stored in separate planes for red, green, and blue channels.
 *
 * @param stream The HIP stream to use for the conversion.
 * @param dst_width The width of the destination RGB image.
 * @param dst_height The height of the destination RGB image.
 * @param dst_image_r Pointer to the destination red channel plane.
 * @param dst_image_g Pointer to the destination green channel plane.
 * @param dst_image_b Pointer to the destination blue channel plane.
 * @param dst_image_stride_in_bytes The stride (in bytes) between consecutive rows in the destination image planes.
 * @param src_image Pointer to the source YUYV image.
 * @param src_image_stride_in_bytes The stride (in bytes) between consecutive rows in the source image.
 */
void ColorConvertYUYVToRGBPlanar(hipStream_t stream, uint32_t dst_width, uint32_t dst_height,
    uint8_t *dst_image_r, uint8_t *dst_image_g, uint8_t *dst_image_b, uint32_t dst_image_stride_in_bytes,
    const uint8_t *src_image, uint32_t src_image_stride_in_bytes);

/**
 * @brief Converts an NV12 image to RGB planar format.
 *
 * This function takes an NV12 image and converts it to RGB planar format. The NV12 image consists of a luma (Y) plane followed by a packed chroma (UV) plane.
 * The RGB planar format consists of separate R, G, and B planes.
 *
 * @param stream The HIP stream to be used for the conversion.
 * @param dst_width The width of the destination RGB image.
 * @param dst_height The height of the destination RGB image.
 * @param dst_image_r Pointer to the destination R plane.
 * @param dst_image_g Pointer to the destination G plane.
 * @param dst_image_b Pointer to the destination B plane.
 * @param dst_image_stride_in_bytes The stride (in bytes) of the destination RGB image.
 * @param src_luma_image Pointer to the source luma (Y) plane.
 * @param src_luma_image_stride_in_bytes The stride (in bytes) of the source luma (Y) plane.
 * @param src_chroma_image Pointer to the source chroma (UV) plane.
 * @param src_chroma_image_stride_in_bytes The stride (in bytes) of the source chroma (UV) plane.
 */
void ColorConvertNV12ToRGBPlanar(hipStream_t stream, uint32_t dst_width, uint32_t dst_height,
    uint8_t *dst_image_r, uint8_t *dst_image_g, uint8_t *dst_image_b, uint32_t dst_image_stride_in_bytes,
    const uint8_t *src_luma_image, uint32_t src_luma_image_stride_in_bytes,
    const uint8_t *src_chroma_image, uint32_t src_chroma_image_stride_in_bytes);

/**
 * @brief Converts a YUV400 image to RGB planar format.
 *
 * This function takes a YUV400 image and converts it to RGB planar format.
 * The resulting RGB image will have separate planes for red, green, and blue channels.
 *
 * @param stream The HIP stream to be used for the kernel execution.
 * @param dst_width The width of the destination RGB image.
 * @param dst_height The height of the destination RGB image.
 * @param dst_image_r Pointer to the destination red channel plane.
 * @param dst_image_g Pointer to the destination green channel plane.
 * @param dst_image_b Pointer to the destination blue channel plane.
 * @param dst_image_stride_in_bytes The stride (in bytes) of the destination image planes.
 * @param src_luma_image Pointer to the source YUV400 luma image.
 * @param src_luma_image_stride_in_bytes The stride (in bytes) of the source luma image.
 */
void ColorConvertYUV400ToRGBPlanar(hipStream_t stream, uint32_t dst_width, uint32_t dst_height,
    uint8_t *dst_image_r, uint8_t *dst_image_g, uint8_t *dst_image_b, uint32_t dst_image_stride_in_bytes,
    const uint8_t *src_luma_image, uint32_t src_luma_image_stride_in_bytes);

/**
 * @brief Converts interleaved UV data to planar UV data.
 *
 * This function takes interleaved UV data from the source image and converts it to planar UV data,
 * which is then stored in the destination images. The conversion is performed on the GPU using
 * the specified stream.
 *
 * @param stream The HIP stream to be used for the conversion.
 * @param dst_width The width of the destination images.
 * @param dst_height The height of the destination images.
 * @param dst_image1 Pointer to the first destination image buffer.
 * @param dst_image2 Pointer to the second destination image buffer.
 * @param dst_image_stride_in_bytes The stride (in bytes) of the destination image buffers.
 * @param src_image1 Pointer to the source image buffer containing interleaved UV data.
 * @param src_image1_stride_in_bytes The stride (in bytes) of the source image buffer.
 */
void ConvertInterleavedUVToPlanarUV(hipStream_t stream, uint32_t dst_width, uint32_t dst_height,
    uint8_t *dst_image1, uint8_t *dst_image2, uint32_t dst_image_stride_in_bytes,
    const uint8_t *src_image1, uint32_t src_image1_stride_in_bytes);

/**
 * @brief Extracts the Y component from a packed YUYV image and stores it in a separate buffer.
 *
 * This function takes a packed YUYV image and extracts the Y component (luma) from it. The extracted Y component
 * is then stored in a separate buffer. The dimensions of the destination buffer, as well as the stride of the
 * source and destination buffers, should be provided.
 *
 * @param stream The HIP stream to be used for the operation.
 * @param dst_width The width of the destination buffer.
 * @param dst_height The height of the destination buffer.
 * @param destination_y Pointer to the destination buffer where the extracted Y component will be stored.
 * @param dst_luma_stride_in_bytes The stride (in bytes) of the destination buffer.
 * @param src_image Pointer to the source YUYV image.
 * @param src_image_stride_in_bytes The stride (in bytes) of the source YUYV image.
 */
void ExtractYFromPackedYUYV(hipStream_t stream, uint32_t dst_width, uint32_t dst_height,
    uint8_t *destination_y, uint32_t dst_luma_stride_in_bytes, const uint8_t *src_image, uint32_t src_image_stride_in_bytes);

/**
 * @brief Converts a packed YUYV image to planar YUV format.
 *
 * This function takes a packed YUYV image and converts it to planar YUV format.
 * The resulting planar YUV image will have separate planes for Y, U, and V components.
 *
 * @param stream The HIP stream to associate the kernel execution with.
 * @param dst_width The width of the destination image in pixels.
 * @param dst_height The height of the destination image in pixels.
 * @param destination_y Pointer to the destination Y plane.
 * @param destination_u Pointer to the destination U plane.
 * @param destination_v Pointer to the destination V plane.
 * @param dst_luma_stride_in_bytes The stride (in bytes) between consecutive rows of the destination Y plane.
 * @param dst_chroma_stride_in_bytes The stride (in bytes) between consecutive rows of the destination U and V planes.
 * @param src_image Pointer to the source packed YUYV image.
 * @param src_image_stride_in_bytes The stride (in bytes) between consecutive rows of the source image.
 */
void ConvertPackedYUYVToPlanarYUV(hipStream_t stream, uint32_t dst_width, uint32_t dst_height,
    uint8_t *destination_y, uint8_t *destination_u, uint8_t *destination_v, uint32_t dst_luma_stride_in_bytes,
    uint32_t dst_chroma_stride_in_bytes, const uint8_t *src_image, uint32_t src_image_stride_in_bytes);

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