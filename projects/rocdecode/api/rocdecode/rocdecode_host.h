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

#ifndef ROCDECAPI
#if defined(_WIN32)
#define ROCDECAPI __stdcall // for future: only linux is supported in this version
#else
#define ROCDECAPI
#endif
#endif

#pragma once
#include "hip/hip_runtime.h"
#include "rocdecode.h"
#include "rocparser.h"

/*!
 * \file
 * \brief The AMD rocDecode Library.
 *
 * \defgroup group_amd_rocdecode rocDecode: AMD ROCm Software Decode API
 * \brief  The rocDecodeHost is a part of rocDecode toolkit to decode videos and images using a avcodec based video decoder on ROCm.
 */

#if defined(__cplusplus)
extern "C" {
#endif // __cplusplus

/****************************************************************/
//! \ingroup group_rocdec_struct
//! \struct RocdecVideoFormatHost
//! Video format including raw sequence header information
//! Used in rocDecCreateVideoParser API
/****************************************************************/
typedef struct {
    RocdecVideoFormat video_format; /**< OUT: RocdecVideoFormat structure */
    rocDecVideoSurfaceFormat video_surface_format;  /**< OUT: output surface format */
    uint32_t reserved[16];      // reserved for future
} RocdecVideoFormatHost;

typedef int(ROCDECAPI *PFNVIDSEQUENCECHOSTALLBACK)(void *, RocdecVideoFormatHost *);
typedef int(ROCDECAPI *PFNVIDDISPLAYHOSTCALLBACK)(void *, void *);

/******************************************************************************************/
//! \struct _RocdecPicParamsHost
//! \ingroup group_amd_rocdecode
//! Picture parameters for decoding
//! This structure is used in rocDecDecodePictureHost API
//! IN  for rocDecDecodePictureHost
/******************************************************************************************/
typedef struct _RocdecPicParamsHost {
    // Bitstream data
    uint32_t bitstream_data_len;   /**< IN: Number of bytes in bitstream data buffer */
    const uint8_t *bitstream_data; /**< IN: Ptr to bitstream data for this picture (slice-layer) */
    uint32_t flags;         /**< IN: Combination of ROCDEC_PKT_XXX flags                             */
    RocdecTimeStamp pts;    /**< IN: Presentation time stamp (10MHz clock), only valid if ROCDEC_PKT_TIMESTAMP flag is set */
} RocdecPicParamsHost;


/**************************************************************************************************************/
//! \struct RocDecoderHostCreateInfo
//! \ingroup group_amd_rocdecode
//! This structure is used in rocDecCreateDecoderHost API
/**************************************************************************************************************/
typedef struct _RocDecoderHostCreateInfo {
    uint32_t width;                        /**< IN: Coded sequence width in pixels */
    uint32_t height;                       /**< IN: Coded sequence height in pixels */
    uint32_t num_decode_threads;           /**< IN: Maximum number of internal decode threads for multi-threading <default value 0: threading will be chosen as appropriate to get the max performace> */
    rocDecVideoCodec codec_type;           /**< IN: rocDecVideoCodec_XXX */
    rocDecVideoChromaFormat chroma_format; /**< IN: rocDecVideoChromaFormat_XXX */
    uint32_t bit_depth_minus_8;            /**< IN: The value "BitDepth minus 8" */
    uint32_t intra_decode_only;            /**< IN: not used for avcodec based decoding (default value is 0). */
    uint32_t max_width;                    /**< IN: Coded sequence max width in pixels used with reconfigure Decoder */
    uint32_t max_height;                   /**< IN: Coded sequence max height in pixels used with reconfigure Decoder */
    struct {
        int16_t left;
        int16_t top;
        int16_t right;
        int16_t bottom;
    } display_rect;                         /**< IN: area of the frame that should be displayed */
    rocDecVideoSurfaceFormat output_format; /**< IN: rocDecVideoSurfaceFormat_XXX */
    uint32_t target_width;                  /**< IN: Post-processed output width (Should be aligned to 2) */
    uint32_t target_height;                 /**< IN: Post-processed output height (Should be aligned to 2) */
    uint32_t num_output_surfaces;           /**< IN: Maximum number of output surfaces simultaneously mapped */
    struct {
        int16_t left;
        int16_t top;
        int16_t right;
        int16_t bottom;
    } target_rect;          /**< IN: (for future use) target rectangle in the output frame (for aspect ratio conversion)
                                    if a null rectangle is specified, {0,0,target_width,target_height} will be used*/
    void *user_data;                              /**< IN: User data for callbacks                                             */
    // callback functions to enable users to consume decoded data 
    PFNVIDSEQUENCECHOSTALLBACK pfn_sequence_callback; /**< IN: Called before decoding frames and/or whenever there is a fmt change */
    PFNVIDDISPLAYCALLBACK pfn_display_picture;    /**< IN: Called whenever a picture is ready to be displayed (display order)  */
    PFNVIDSEIMSGCALLBACK pfn_get_sei_msg;         /**< IN: Called when all SEI messages are parsed for particular frame        */
    uint32_t reserved[4];                         /**< Reserved for future use - set to zero */
} RocDecoderHostCreateInfo;


/*****************************************************************************************************/
//! \fn rocDecStatus ROCDECAPI rocDecCreateDecoderHost(rocDecDecoderHandle *decoder_handle, RocDecoderHostCreateInfo *decoder_create_info)
//! \ingroup group_amd_rocdecode
//! Create the decoder object based on decoder_create_info. A handle to the created decoder is returned
/*****************************************************************************************************/
extern rocDecStatus ROCDECAPI rocDecCreateDecoderHost(rocDecDecoderHandle *decoder_handle, RocDecoderHostCreateInfo *decoder_create_info);

/*****************************************************************************************************/
//! \fn rocDecStatus ROCDECAPI rocDecDestroyDecoderHost(rocDecDecoderHandle decoder_handle)
//! \ingroup group_amd_rocdecode
//! Destroy the decoder object
/*****************************************************************************************************/
extern rocDecStatus ROCDECAPI rocDecDestroyDecoderHost(rocDecDecoderHandle decoder_handle);

/**********************************************************************************************************************/
//! \fn rocDecStatus ROCDECAPI rocDecGetDecoderCapsHost(RocdecDecodeCaps *decode_caps)
//! \ingroup group_amd_rocdecode
//! Queries decode capabilities of host based decoder based on codec type, chroma_format and BitDepthMinus8 parameters.
//! 1. Application fills IN parameters codec_type, chroma_format and BitDepthMinus8 of RocdecDecodeCaps structure
//! 2. For FFMpeg avcodec based decoder, this call returns success
/**********************************************************************************************************************/
extern rocDecStatus ROCDECAPI rocDecGetDecoderCapsHost(RocdecDecodeCaps *decode_caps);

/*****************************************************************************************************/
//! \fn rocDecStatus ROCDECAPI rocDecDecodeFrameHost(rocDecDecoderHandle decoder_handle, RocdecPicParamsHost *pic_params)
//! \ingroup group_amd_rocdecode
//! Decodes a single picture
//! Submits the frame for host based decoding
/*****************************************************************************************************/
extern rocDecStatus ROCDECAPI rocDecDecodeFrameHost(rocDecDecoderHandle decoder_handle, RocdecPicParamsHost *pic_params);

/************************************************************************************************************/
//! \fn rocDecStatus ROCDECAPI rocDecGetDecodeStatusHost(rocDecDecoderHandle decoder_handle, int pic_idx, RocdecDecodeStatus* decode_status);
//! \ingroup group_amd_rocdecode
//! Get the decode status for frame corresponding to nPicIdx
//! API is currently supported for HEVC, AVC/H264 and JPEG codecs.
//! API returns ROCDEC_NOT_SUPPORTED error code for unsupported codec.
/************************************************************************************************************/
extern rocDecStatus ROCDECAPI rocDecGetDecodeStatusHost(rocDecDecoderHandle decoder_handle, int pic_idx, RocdecDecodeStatus *decode_status);

/*********************************************************************************************************/
//! \fn rocDecStatus ROCDECAPI rocDecReconfigureDecoderHost(rocDecDecoderHandle decoder_handle, RocdecReconfigureDecoderInfo *reconfig_params)
//! \ingroup group_amd_rocdecode
//! Used to reuse single decoder for multiple clips. Currently supports resolution change, resize params
//! params, target area params change for same codec. Must be called during RocdecParserParams::pfn_sequence_callback
/*********************************************************************************************************/
extern rocDecStatus ROCDECAPI rocDecReconfigureDecoderHost(rocDecDecoderHandle decoder_handle, RocdecReconfigureDecoderInfo *reconfig_params);

/************************************************************************************************************************/
//! \fn extern rocDecStatus ROCDECAPI rocDecGetVideoFrameHost(rocDecDecoderHandle decoder_handle, int pic_idx,
//!                                           uint32_t *frame_data, uint32_t *line_size,
//!                                           RocdecProcParams *vid_postproc_params);
//! \ingroup group_amd_rocdecode
//! Post-process and map video frame corresponding to pic_idx for use in HIP. Returns HIP device pointer and associated
//! line_size of the video frame. Returns host memory pointers and pitch for each plane (Y, U and V) seperately
//! line_size is a pointer to an unsigned 32-bit integer array of size 3.
/************************************************************************************************************************/
extern rocDecStatus ROCDECAPI rocDecGetVideoFrameHost(rocDecDecoderHandle decoder_handle, int pic_idx,
                                                    void **frame_data, uint32_t *line_size,
                                                    RocdecProcParams *vid_postproc_params);

/*****************************************************************************************************/
//! \fn const char* ROCDECAPI rocDecGetErrorNameHost(rocDecStatus rocdec_status)
//! \ingroup group_amd_rocdecode
//! Return name of the specified error code in text form.
/*****************************************************************************************************/
extern const char *ROCDECAPI rocDecGetErrorNameHost(rocDecStatus rocdec_status);

#ifdef  __cplusplus
}
#endif
