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

#include "colorspace_kernels.h"
#include "resize_kernels.h"
#include "rocvideodecode/roc_video_dec.h"       //for OutputSurfaceInfo

enum OutputFormatEnum {
    native = 0, bgr, bgr48, rgb, rgb48, bgra, bgra64, rgba, rgba64
};

class VideoPostProcess {
    public:
        VideoPostProcess(){};
        ~VideoPostProcess(){};
        
        void ColorConvertYUV2RGB(uint8_t *p_src, OutputSurfaceInfo *surf_info, uint8_t *rgb_dev_mem_ptr, OutputFormatEnum e_output_format, hipStream_t hip_stream) {
            int  rgb_width = (surf_info->output_width + 1) & ~1;    // has to be a multiple of 2 for hip colorconvert kernels
            // todo:: get color standard from the decoder
            if (surf_info->surface_format == rocDecVideoSurfaceFormat_YUV444) {
                if (e_output_format == bgr)
                YUV444ToColor24<BGR24>(p_src, surf_info->output_pitch, static_cast<uint8_t *>(rgb_dev_mem_ptr), 3 * rgb_width, surf_info->output_width, 
                                        surf_info->output_height, surf_info->output_vstride, 0, hip_stream);
                else if (e_output_format == bgra)
                YUV444ToColor32<BGRA32>(p_src, surf_info->output_pitch, static_cast<uint8_t *>(rgb_dev_mem_ptr), 4 * rgb_width, surf_info->output_width, 
                                        surf_info->output_height, surf_info->output_vstride, 0, hip_stream);
                else if (e_output_format == rgb)
                YUV444ToColor24<RGB24>(p_src, surf_info->output_pitch, static_cast<uint8_t *>(rgb_dev_mem_ptr), 3 * rgb_width, surf_info->output_width, 
                                        surf_info->output_height, surf_info->output_vstride, 0, hip_stream);
                else if (e_output_format == rgba)
                YUV444ToColor32<RGBA32>(p_src, surf_info->output_pitch, static_cast<uint8_t *>(rgb_dev_mem_ptr), 4 * rgb_width, surf_info->output_width, 
                                        surf_info->output_height, surf_info->output_vstride, 0, hip_stream);
            } else if (surf_info->surface_format == rocDecVideoSurfaceFormat_NV12) {
                if (e_output_format == bgr)
                Nv12ToColor24<BGR24>(p_src, surf_info->output_pitch, static_cast<uint8_t *>(rgb_dev_mem_ptr), 3 * rgb_width, surf_info->output_width, 
                                    surf_info->output_height, surf_info->output_vstride, 0, hip_stream);
                else if (e_output_format == bgra)
                Nv12ToColor32<BGRA32>(p_src, surf_info->output_pitch, static_cast<uint8_t *>(rgb_dev_mem_ptr), 4 * rgb_width, surf_info->output_width, 
                                    surf_info->output_height, surf_info->output_vstride, 0, hip_stream);
                else if (e_output_format == rgb)
                Nv12ToColor24<RGB24>(p_src, surf_info->output_pitch, static_cast<uint8_t *>(rgb_dev_mem_ptr), 3 * rgb_width, surf_info->output_width, 
                                    surf_info->output_height, surf_info->output_vstride, 0, hip_stream);
                else if (e_output_format == rgba)
                Nv12ToColor32<RGBA32>(p_src, surf_info->output_pitch, static_cast<uint8_t *>(rgb_dev_mem_ptr), 4 * rgb_width, surf_info->output_width, 
                                    surf_info->output_height, surf_info->output_vstride, 0, hip_stream);
            }
            if (surf_info->surface_format == rocDecVideoSurfaceFormat_YUV444_16Bit) {
                if (e_output_format == bgr)
                YUV444P16ToColor24<BGR24>(p_src, surf_info->output_pitch, static_cast<uint8_t *>(rgb_dev_mem_ptr), 3 * rgb_width, surf_info->output_width, 
                                        surf_info->output_height, surf_info->output_vstride, 0, hip_stream);
                else if (e_output_format == rgb)
                YUV444P16ToColor24<RGB24>(p_src, surf_info->output_pitch, static_cast<uint8_t *>(rgb_dev_mem_ptr), 3 * rgb_width, surf_info->output_width, 
                                        surf_info->output_height, surf_info->output_vstride, 0, hip_stream);
                else if (e_output_format == bgr48)
                YUV444P16ToColor48<BGR48>(p_src, surf_info->output_pitch, static_cast<uint8_t *>(rgb_dev_mem_ptr), 6 * rgb_width, surf_info->output_width, 
                                        surf_info->output_height, surf_info->output_vstride, 0, hip_stream);
                else if (e_output_format == rgb48)
                YUV444P16ToColor48<RGB48>(p_src, surf_info->output_pitch, static_cast<uint8_t *>(rgb_dev_mem_ptr), 6 * rgb_width, surf_info->output_width, 
                                        surf_info->output_height, surf_info->output_vstride, 0, hip_stream);
                else if (e_output_format == bgra64)
                YUV444P16ToColor64<BGRA64>(p_src, surf_info->output_pitch, static_cast<uint8_t *>(rgb_dev_mem_ptr), 8 * rgb_width, surf_info->output_width, 
                                        surf_info->output_height, surf_info->output_vstride, 0, hip_stream);
                else if (e_output_format == rgba64)
                YUV444P16ToColor64<RGBA64>(p_src, surf_info->output_pitch, static_cast<uint8_t *>(rgb_dev_mem_ptr), 8 * rgb_width, surf_info->output_width, 
                                        surf_info->output_height, surf_info->output_vstride, 0, hip_stream);
            } else if (surf_info->surface_format == rocDecVideoSurfaceFormat_P016) {
                if (e_output_format == bgr)
                P016ToColor24<BGR24>(p_src, surf_info->output_pitch, static_cast<uint8_t *>(rgb_dev_mem_ptr), 3 * rgb_width, surf_info->output_width, 
                                    surf_info->output_height, surf_info->output_vstride, 0, hip_stream);
                else if (e_output_format == rgb)
                P016ToColor24<RGB24>(p_src, surf_info->output_pitch, static_cast<uint8_t *>(rgb_dev_mem_ptr), 3 * rgb_width, surf_info->output_width, 
                                    surf_info->output_height, surf_info->output_vstride, 0, hip_stream);
                else if (e_output_format == bgr48)
                P016ToColor48<BGR48>(p_src, surf_info->output_pitch, static_cast<uint8_t *>(rgb_dev_mem_ptr), 6 * rgb_width, surf_info->output_width, 
                                    surf_info->output_height, surf_info->output_vstride, 0, hip_stream);
                else if (e_output_format == rgb48)
                P016ToColor48<RGB48>(p_src, surf_info->output_pitch, static_cast<uint8_t *>(rgb_dev_mem_ptr), 6 * rgb_width, surf_info->output_width, 
                                    surf_info->output_height, surf_info->output_vstride, 0, hip_stream);
                else if (e_output_format == bgra64)
                P016ToColor64<BGRA64>(p_src, surf_info->output_pitch, static_cast<uint8_t *>(rgb_dev_mem_ptr), 8 * rgb_width, surf_info->output_width, 
                                    surf_info->output_height, surf_info->output_vstride, 0, hip_stream);
                else if (e_output_format == rgba64)
                P016ToColor64<RGBA64>(p_src, surf_info->output_pitch, static_cast<uint8_t *>(rgb_dev_mem_ptr), 8 * rgb_width, surf_info->output_width, 
                                    surf_info->output_height, surf_info->output_vstride, 0, hip_stream);
            }   
        };
        uint32_t GetRgbStride(OutputFormatEnum e_output_format, OutputSurfaceInfo *surf_info) {
            uint32_t rgb_stride;
            uint32_t  rgb_width = (surf_info->output_width + 1) & ~1; // has to be a multiple of 2 for hip colorconvert kernels
            if (surf_info->bit_depth == 8) {
                rgb_stride = ((e_output_format == bgr) || (e_output_format == rgb)) ? rgb_width * 3 : rgb_width * 4;        // bgr/bgra/rgb/rgba
            } else {
                rgb_stride = ((e_output_format == bgr) || (e_output_format == rgb)) ? rgb_width * 3 : 
                        ((e_output_format == bgr48) || (e_output_format == rgb48)) ? rgb_width * 6 : rgb_width * 8;
            }
            return rgb_stride;
        };
};