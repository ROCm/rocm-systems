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

#pragma once

extern "C" {
#include "libavutil/md5.h"
#include "libavutil/mem.h"
}
#include "roc_video_dec.h"

/*!
 * \file
 * \brief The MD5 message digest generation utility.
 */

class MD5Generator {
public:
    MD5Generator() {};
    ~MD5Generator() {};

    /*! \brief Function to start MD5 calculation
     */
    void InitMd5() {
        md5_ctx_ = av_md5_alloc();
        av_md5_init(md5_ctx_);
    }

    /*! \brief Function to update MD5 digest for a device data buffer
     *  \param [in] data_buf Pointer to the data buffer
     *  \param [in] buf_size Buffer info
     */
    void UpdateMd5ForDataBuffer(void *data_buf, int buf_size) {
        uint8_t *hstPtr = nullptr;
        hstPtr = new uint8_t[buf_size];
        hipError_t hip_status = hipSuccess;
        hip_status = hipMemcpyDtoH((void *)hstPtr, data_buf, buf_size);
        if (hip_status != hipSuccess) {
            std::cerr << "ERROR: hipMemcpyDtoH failed! (" << hip_status << ")" << std::endl;
            delete [] hstPtr;
            return;
        }
        av_md5_update(md5_ctx_, hstPtr, buf_size);
        if (hstPtr) {
            delete [] hstPtr;
        }
    }

    /*! \brief Function to update MD5 digest for a decoded frame
    *  \param [in] surf_mem Pointer to surface memory
    *  \param [in] surf_info Surface info
    */
    void UpdateMd5ForFrame(void *surf_mem, OutputSurfaceInfo *surf_info) {
        int i;
        uint8_t *hst_ptr = nullptr;
        uint64_t output_image_size = surf_info->output_surface_size_in_bytes;
        if (surf_info->mem_type == OUT_SURFACE_MEM_DEV_INTERNAL || surf_info->mem_type == OUT_SURFACE_MEM_DEV_COPIED) {
            if (hst_ptr == nullptr) {
                hst_ptr = new uint8_t [output_image_size];
            }
            hipError_t hip_status = hipSuccess;
            hip_status = hipMemcpyDtoH((void *)hst_ptr, surf_mem, output_image_size);
            if (hip_status != hipSuccess) {
                std::cerr << "ERROR: hipMemcpyDtoH failed! (" << hip_status << ")" << std::endl;
                delete [] hst_ptr;
                return;
            }
        } else
            hst_ptr = static_cast<uint8_t *> (surf_mem);

        if (hst_ptr == nullptr) {
            ROCDEC_ERR("Null surface pointer.");
            return;
        }

        // Need to covert interleaved planar to stacked planar, assuming 4:2:0 chroma sampling.
        uint8_t *stacked_ptr = new uint8_t [output_image_size];
        uint8_t *tmp_hst_ptr = hst_ptr;
        int output_stride =  surf_info->output_pitch;
        tmp_hst_ptr += (surf_info->disp_rect.top * output_stride) + surf_info->disp_rect.left * surf_info->bytes_per_pixel;
        uint8_t *tmp_stacked_ptr = stacked_ptr;
        int img_width = surf_info->output_width;
        int img_height = surf_info->output_height;
        // Luma
        if (img_width * surf_info->bytes_per_pixel == output_stride && img_height == surf_info->output_vstride) {
            memcpy(stacked_ptr, hst_ptr, img_width * surf_info->bytes_per_pixel * img_height);
        } else {
            for (i = 0; i < img_height; i++) {
                memcpy(tmp_stacked_ptr, tmp_hst_ptr, img_width * surf_info->bytes_per_pixel);
                tmp_hst_ptr += output_stride;
                tmp_stacked_ptr += img_width * surf_info->bytes_per_pixel;
            }
        }
        // Chroma
        int img_width_chroma = img_width >> 1;
        tmp_hst_ptr = hst_ptr + output_stride * surf_info->output_vstride;
        if (surf_info->mem_type == OUT_SURFACE_MEM_DEV_INTERNAL) {
            tmp_hst_ptr += ((surf_info->disp_rect.top >> 1) * output_stride) + (surf_info->disp_rect.left * surf_info->bytes_per_pixel);
        }
        tmp_stacked_ptr = stacked_ptr + img_width * surf_info->bytes_per_pixel * img_height; // Cb
        uint8_t *tmp_stacked_ptr_v = tmp_stacked_ptr + img_width_chroma * surf_info->bytes_per_pixel * surf_info->chroma_height; // Cr
        for (i = 0; i < surf_info->chroma_height; i++) {
            for ( int j = 0; j < img_width_chroma; j++) {
                uint8_t *src_ptr, *dst_ptr;
                // Cb
                src_ptr = &tmp_hst_ptr[j * surf_info->bytes_per_pixel * 2];
                dst_ptr = &tmp_stacked_ptr[j * surf_info->bytes_per_pixel];
                memcpy(dst_ptr, src_ptr, surf_info->bytes_per_pixel);
                // Cr
                src_ptr += surf_info->bytes_per_pixel;
                dst_ptr = &tmp_stacked_ptr_v[j * surf_info->bytes_per_pixel];
                memcpy(dst_ptr, src_ptr, surf_info->bytes_per_pixel);
            }
            tmp_hst_ptr += output_stride;
            tmp_stacked_ptr += img_width_chroma * surf_info->bytes_per_pixel;
            tmp_stacked_ptr_v += img_width_chroma * surf_info->bytes_per_pixel;
        }

        int img_size = img_width * surf_info->bytes_per_pixel * (img_height + surf_info->chroma_height);
        // For 10/12 bit, convert from P010/P012 to LSB to match reference decoder output
        if (surf_info->bytes_per_pixel == 2) {
            uint16_t *ptr = reinterpret_cast<uint16_t *> (stacked_ptr);
            for (i = 0; i < img_size / 2; i++) {
                ptr[i] = ptr[i] >> (16 - surf_info->bit_depth);
            }
        }

        av_md5_update(md5_ctx_, stacked_ptr, img_size);
        if (hst_ptr && (surf_info->mem_type != OUT_SURFACE_MEM_HOST_COPIED)) {
            delete [] hst_ptr;
        }
        delete [] stacked_ptr;
    }

    /*! \brief Function to complete MD5 calculation
     *  \param [out] digest Pointer to the 16 byte message digest
     */
    void FinalizeMd5(uint8_t **digest) {
        av_md5_final(md5_ctx_, md5_digest_);
        av_freep(&md5_ctx_);
        *digest = md5_digest_;
    }

private:
    struct AVMD5 *md5_ctx_;
    uint8_t md5_digest_[16];
};