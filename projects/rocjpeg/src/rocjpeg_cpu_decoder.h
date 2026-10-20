/*
Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

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

#ifndef ROC_JPEG_CPU_DECODER_H_
#define ROC_JPEG_CPU_DECODER_H_

#include <stdint.h>
#include <vector>
#include "../api/rocjpeg/rocjpeg.h"
#include "rocjpeg_parser.h"

/**
 * @brief The decoded components of a JPEG image, each at its own sampled resolution.
 *
 * Component c covers widths[c] x heights[c] samples, which is the image size scaled by the component's sampling
 * factors relative to the largest ones and rounded up (ISO/IEC 10918-1 A.1.1). A plane holds whole 8x8 blocks, so
 * strides[c] and the number of rows can exceed the component size; only the first widths[c] samples of the first
 * heights[c] rows are image data.
 */
struct CpuDecodedImage {
    uint8_t num_components = 0;
    uint32_t widths[NUM_COMPONENTS] = {};
    uint32_t heights[NUM_COMPONENTS] = {};
    uint32_t strides[NUM_COMPONENTS] = {};
    std::vector<uint8_t> planes[NUM_COMPONENTS];
};

/**
 * @brief One plane of output, as rows of row_bytes bytes, stride bytes apart.
 */
struct CpuOutputPlane {
    const uint8_t *data = nullptr;
    size_t stride = 0;
    uint32_t row_bytes = 0;
    uint32_t rows = 0;
};

/**
 * @brief The planes of one output format, ready to copy to the destination channels in order.
 *
 * The planes point into the CpuDecodedImage they were laid out from, or into packed, so neither may change or be
 * destroyed while the planes are used, and a CpuOutput must not be copied.
 */
struct CpuOutput {
    int num_planes = 0;
    CpuOutputPlane planes[NUM_COMPONENTS - 1];
    std::vector<uint8_t> packed;  // NV12 and YUYV, which interleave chroma.
};

/**
 * @brief Decodes a parsed baseline JPEG stream on the CPU.
 *
 * The decoder follows the processing libjpeg-turbo uses by default, so that its output can serve as a reference for
 * the hardware backend: baseline Huffman decoding with restart intervals, and the accurate integer inverse DCT
 * (the "islow" method, with 13-bit constants and two extra bits of precision between the passes). It is written from
 * ISO/IEC 10918-1 and does not use libjpeg-turbo code.
 *
 * Corrupt entropy-coded data is handled the way libjpeg-turbo handles it rather than rejected: a bad Huffman code
 * decodes as 0; once the data runs out at a marker or at the end, the MCUs up to the next restart marker are left at
 * the mid-level value; and a restart marker other than the expected one is resynchronized as jpeg_resync_to_restart
 * does.
 */
class RocJpegCpuDecoder {
    public:
        /**
         * @brief Decodes every component of a parsed JPEG stream.
         * @param jpeg_stream_params The parameters RocJpegStreamParser produced for the stream.
         * @param image Receives the decoded components.
         * @return ROCJPEG_STATUS_SUCCESS on success, ROCJPEG_STATUS_BAD_JPEG if the stream's tables or scan header
         *         cannot be decoded, or ROCJPEG_STATUS_OUTOF_MEMORY if the planes cannot be allocated.
         */
        static RocJpegStatus Decode(const JpegStreamParameters &jpeg_stream_params, CpuDecodedImage &image);

        /**
         * @brief Checks whether an output format can be produced for a chroma subsampling.
         * @return ROCJPEG_STATUS_SUCCESS, ROCJPEG_STATUS_NOT_IMPLEMENTED for an output format other than
         *         ROCJPEG_OUTPUT_NATIVE, ROCJPEG_OUTPUT_YUV_PLANAR, and ROCJPEG_OUTPUT_Y, or
         *         ROCJPEG_STATUS_JPEG_NOT_SUPPORTED for a chroma subsampling other than 4:4:4, 4:4:0, 4:2:2, 4:2:0,
         *         and 4:0:0.
         */
        static RocJpegStatus CheckOutput(RocJpegOutputFormat output_format, ChromaSubsampling subsampling);

        /**
         * @brief Lays out a decoded image in an output format, as the hardware backend writes it with the native
         *        surface formats YUV444P, YUV440P, YUYV, NV12, and Y8.
         *
         * Luma is width x height, and chroma has the size rocJpegGetImageInfo reports (half the width for 4:2:2 and
         * 4:2:0, half the height for 4:4:0 and 4:2:0, rounded down).
         * @param jpeg_stream_params The parameters the image was decoded from.
         * @param image The decoded image.
         * @param output_format The output format.
         * @param output Receives the planes.
         * @return ROCJPEG_STATUS_SUCCESS, the status of CheckOutput, ROCJPEG_STATUS_JPEG_NOT_SUPPORTED if the
         *         sampling factors do not have the geometry of the chroma subsampling they are classified as, or
         *         ROCJPEG_STATUS_OUTOF_MEMORY.
         */
        static RocJpegStatus LayOutOutput(const JpegStreamParameters &jpeg_stream_params, const CpuDecodedImage &image,
                                          RocJpegOutputFormat output_format, CpuOutput &output);
};

#endif // ROC_JPEG_CPU_DECODER_H_
