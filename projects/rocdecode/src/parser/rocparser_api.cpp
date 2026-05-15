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
#include "parser_handle.h"
#include "../commons.h"

namespace rocdecode {
/************************************************************************************************/
//! \ingroup FUNCTS
//! \fn rocParserStatus ROCDECAPI rocDecCreateVideoParser(RocdecVideoParser *parser_handle, RocdecParserParams *parser_params)
//! Create video parser object and initialize
/************************************************************************************************/
rocDecStatus ROCDECAPI 
rocDecCreateVideoParser(RocdecVideoParser *parser_handle, RocdecParserParams *parser_params) {
    if (parser_handle == nullptr || parser_params == nullptr) {
        return ROCDEC_INVALID_PARAMETER;
    }

    if (parser_params->codec_type != rocDecVideoCodec_HEVC &&
        parser_params->codec_type != rocDecVideoCodec_AVC &&
        parser_params->codec_type != rocDecVideoCodec_VP9 &&
        parser_params->codec_type != rocDecVideoCodec_AV1) {
        RocDecLogger::AlwaysLog("Error: The current version of rocDecode officially supports only the H.265 (HEVC), H.264 (AVC), AV1 and VP9 codecs.");
        return ROCDEC_NOT_IMPLEMENTED;
    }

    RocdecVideoParser handle = nullptr;
    try {
        handle = new RocParserHandle(parser_params);
    } 
    catch(const std::exception& e) {
        RocDecLogger::AlwaysLog(STR("Error: Failed to init the rocDecode handle, ") + STR(e.what()));
        return ROCDEC_RUNTIME_ERROR;
    }
    *parser_handle = handle;
    return rocDecStatus::ROCDEC_SUCCESS;
}

/************************************************************************************************/
//! \ingroup FUNCTS
//! \fn rocParserStatus ROCDECAPI rocDecParseVideoData(RocdecVideoParser parser_handle, RocdecSourceDataPacket *packet)
//! Parse the video data from source data packet in packet
//! Extracts parameter sets like SPS, PPS, bitstream etc. from packet and
//! calls back pfn_decode_picture with RocdecPicParams data for kicking of HW decoding
//! calls back pfn_sequence_callback with RocdecVideoFormat data for initial sequence header or when
//! the decoder encounters a video format change
//! calls back pfn_display_picture with ROCDECPARSERDISPINFO data to display a video frame
/************************************************************************************************/
rocDecStatus ROCDECAPI
rocDecParseVideoData(RocdecVideoParser parser_handle, RocdecSourceDataPacket *packet) {
    if (parser_handle == nullptr || packet == nullptr) {
        return ROCDEC_INVALID_PARAMETER;
    }
    auto roc_parser_handle = static_cast<RocParserHandle *>(parser_handle);
    rocDecStatus ret;
    try {
        ret = roc_parser_handle->ParseVideoData(packet);
    }
    catch(const std::exception& e) {
        roc_parser_handle->CaptureError(e.what());
        RocDecLogger::AlwaysLog(e.what());
        return ROCDEC_RUNTIME_ERROR;
    }
    return ret;  
}

/************************************************************************************************/
//! \ingroup FUNCTS
//! \fn rocDecStatus ROCDECAPI rocDecDestroyVideoParser(RocdecVideoParser parser_handle)
//! Destroy the video parser object
/************************************************************************************************/
extern rocDecStatus ROCDECAPI
rocDecDestroyVideoParser(RocdecVideoParser parser_handle) {
    if (parser_handle == nullptr) {
        return ROCDEC_INVALID_PARAMETER;
    }
    auto roc_parser_handle = static_cast<RocParserHandle *>(parser_handle);
    rocDecStatus ret;
    try {
        ret = roc_parser_handle->DestroyParser();
    }
    catch(const std::exception& e) {
        roc_parser_handle->CaptureError(e.what());
        delete roc_parser_handle;
        RocDecLogger::AlwaysLog(e.what());
        return ROCDEC_RUNTIME_ERROR;
    }
    delete roc_parser_handle;
    return ret;
}
} //namespace rocdecode