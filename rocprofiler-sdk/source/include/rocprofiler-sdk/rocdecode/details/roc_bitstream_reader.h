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

#pragma once

#include <rocprofiler-sdk/rocdecode/details/rocdecode_headers.h>

#if ROCPROFILER_SDK_USE_SYSTEM_ROCDECODE > 0
#    include <rocdecode/rocdecode.h>
#else
#    include <rocprofiler-sdk/rocdecode/details/rocdecode.h>
#endif

/*!
 * \file
 * \brief The AMD rocBitstreamReader Library.
 *
 * \defgroup group_roc_bitstream_reader rocDecode Parser: AMD ROCm Video Bitstream Reader API
 * \brief AMD The rocBitstreamReader is a toolkit to read picture data from bitstream files for
 * decoding on AMD’s GPUs.
 */

#if defined(__cplusplus)
extern "C" {
#endif /* __cplusplus */

/*********************************************************************************/
//! HANDLE of rocBitstreamReader
//! Used in subsequent API calls after rocDecCreateBitstreamReader
/*********************************************************************************/
typedef void* RocdecBitstreamReader;

/************************************************************************************************/
//! \ingroup group_roc_bitstream_reader
//! \fn rocDecStatus ROCDECAPI rocDecCreateBitstreamReader(RocdecBitstreamReader *bs_reader_handle,
//! const char *input_file_path) Create video bitstream reader object and initialize
/************************************************************************************************/
extern rocDecStatus ROCDECAPI
rocDecCreateBitstreamReader(RocdecBitstreamReader* bs_reader_handle, const char* input_file_path);

/************************************************************************************************/
//! \ingroup group_roc_bitstream_reader
//! \fn rocDecStatus ROCDECAPI rocDecGetBitstreamCodecType(RocdecBitstreamReader bs_reader_handle,
//! rocDecVideoCodec *codec_type) Get the codec type of the bitstream
/************************************************************************************************/
extern rocDecStatus ROCDECAPI
rocDecGetBitstreamCodecType(RocdecBitstreamReader bs_reader_handle, rocDecVideoCodec* codec_type);

/************************************************************************************************/
//! \ingroup group_roc_bitstream_reader
//! \fn rocDecStatus ROCDECAPI rocDecGetBitstreamBitDepth(RocdecBitstreamReader bs_reader_handle,
//! int *bit_depth) Get the bit depth of the bitstream
/************************************************************************************************/
extern rocDecStatus ROCDECAPI
rocDecGetBitstreamBitDepth(RocdecBitstreamReader bs_reader_handle, int* bit_depth);

/************************************************************************************************/
//! \ingroup group_roc_bitstream_reader
//! \fn rocDecStatus ROCDECAPI rocDecGetBitstreamPicData(RocdecBitstreamReader bs_reader_handle,
//! uint8_t **pic_data, int *pic_size, int64_t *pts) Read one unit of picture data from the
//! bitstream. The unit can be a frame or field for AVC/HEVC, a temporal unit for AV1, or a frame
//! (including superframe) for VP9. The picture data unit is pointed by pic_data. The size of the
//! unit is specified by pic_size. The presentation time stamp, if available, is given by pts.
/************************************************************************************************/
extern rocDecStatus ROCDECAPI
rocDecGetBitstreamPicData(RocdecBitstreamReader bs_reader_handle,
                          uint8_t**             pic_data,
                          int*                  pic_size,
                          int64_t*              pts);

/************************************************************************************************/
//! \ingroup group_roc_bitstream_reader
//! \fn rocDecStatus ROCDECAPI rocDecDestroyBitstreamReader(RocdecBitstreamReader bs_reader_handle)
//! Destroy the video parser object
/************************************************************************************************/
extern rocDecStatus ROCDECAPI
rocDecDestroyBitstreamReader(RocdecBitstreamReader bs_reader_handle);

#if defined(__cplusplus)
}
#endif /* __cplusplus */
