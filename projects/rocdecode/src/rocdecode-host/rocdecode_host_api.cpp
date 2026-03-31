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
#include "dec_handle_host.h"
#include "../api/rocdecode/rocdecode.h"
#include "../src/commons.h"

// Internal C++ symbol: only the logger lives in the namespace. Exported C API functions
// are defined in the global namespace under extern "C" below to match rocdecode_host.h
// declarations and avoid mangled symbols for C/C++ callers.
namespace rocdecodehost {
    RocDecLogger logger;
}

/*****************************************************************************************************/
//! \fn rocDecStatus ROCDECAPI rocDecCreateDecoderHost(rocDecDecoderHandle *decoder_handle, RocDecoderCreateInfoHost *decoder_create_info)
//! Create the decoder object based on decoder_create_info. A handle to the created decoder is returned
/*****************************************************************************************************/
rocDecStatus ROCDECAPI 
rocDecCreateDecoderHost(rocDecDecoderHandle *decoder_handle, RocDecoderHostCreateInfo *decoder_create_info) {
    FunctionEntryLog(rocdecodehost::logger);
    if (decoder_handle == nullptr || decoder_create_info == nullptr) {
        rocdecodehost::logger.CriticalLog(MakeMsg("Null pointer"));
        FunctionExitLog(rocdecodehost::logger);
        return ROCDEC_INVALID_PARAMETER;
    }
    rocDecDecoderHandle handle = nullptr;
    try {
        handle = new DecHandleHost(*decoder_create_info);
    }
    catch(const std::exception& e) {
        rocdecodehost::logger.CriticalLog(MakeMsg("Error: Failed to init the rocDecode handle, ") + STR(e.what()));
        FunctionExitLog(rocdecodehost::logger);
        return ROCDEC_NOT_INITIALIZED;
    }
    *decoder_handle = handle;
    rocDecStatus ret = static_cast<DecHandleHost *>(handle)->roc_decoder_host_->InitializeDecoder();
    FunctionExitLog(rocdecodehost::logger);
    return ret;
}

/*****************************************************************************************************/
//! \fn rocDecStatus ROCDECAPI rocDecDestroyDecoderHost(rocDecDecoderHandle decoder_handle)
//! Destroy the decoder object
/*****************************************************************************************************/
rocDecStatus ROCDECAPI 
rocDecDestroyDecoderHost(rocDecDecoderHandle decoder_handle) {
    FunctionEntryLog(rocdecodehost::logger);
    if (decoder_handle == nullptr) {
        rocdecodehost::logger.CriticalLog(MakeMsg("Null pointer"));
        FunctionExitLog(rocdecodehost::logger);
        return ROCDEC_INVALID_PARAMETER;
    }
    auto handle = static_cast<DecHandleHost *>(decoder_handle);
    delete handle;
    FunctionExitLog(rocdecodehost::logger);
    return ROCDEC_SUCCESS;
}

/**********************************************************************************************************************/
//! \fn rocDecStatus ROCDECAPI rocDecGetDecoderCapsHost(rocDecDecoderHandle decoder_handle, RocdecDecodeCaps *pdc)
//! Queries decode capabilities of host based decoder based on codec type, chroma_format and BitDepthMinus8 parameters.
//! 1. Application fills IN parameters codec_type, chroma_format and BitDepthMinus8 of RocdecDecodeCaps structure
//! 2. On calling rocdecGetDecoderCaps, driver fills OUT parameters if the IN parameters are supported
//!    If IN parameters passed to the driver are not supported by AMD-VCN-HW, then all OUT params are set to 0.
//!    Currently this is not supported in avcodec based decoder
/**********************************************************************************************************************/
rocDecStatus ROCDECAPI
rocDecGetDecoderCapsHost(RocdecDecodeCaps *pdc) {
    FunctionEntryLog(rocdecodehost::logger);
    if (pdc == nullptr) {
        rocdecodehost::logger.CriticalLog(MakeMsg("Null pointer"));
        FunctionExitLog(rocdecodehost::logger);
        return ROCDEC_INVALID_PARAMETER;
    }
    //todo:
    FunctionExitLog(rocdecodehost::logger);
    return ROCDEC_NOT_IMPLEMENTED;
}

/*****************************************************************************************************/
//! \fn rocDecStatus ROCDECAPI rocDecDecodeFrameHost(rocDecDecoderHandle decoder_handle, RocdecPicParams *pic_params)
//! Decodes a single picture
//! Submits the frame for HW decoding 
/*****************************************************************************************************/
rocDecStatus ROCDECAPI 
rocDecDecodeFrameHost(rocDecDecoderHandle decoder_handle, _RocdecPicParamsHost *pic_params) {
    FunctionEntryLog(rocdecodehost::logger);
    if (decoder_handle == nullptr || pic_params == nullptr) {
        rocdecodehost::logger.CriticalLog(MakeMsg("Null pointer"));
        FunctionExitLog(rocdecodehost::logger);
        return ROCDEC_INVALID_PARAMETER;
    }
    auto handle = static_cast<DecHandleHost *>(decoder_handle);
    rocDecStatus ret;
    try {
        ret = handle->roc_decoder_host_->DecodeFrame(pic_params);
    }
    catch(const std::exception& e) {
        handle->CaptureError(e.what());
        rocdecodehost::logger.CriticalLog(MakeMsg(e.what()));
        FunctionExitLog(rocdecodehost::logger);
        return ROCDEC_RUNTIME_ERROR;
    }
    FunctionExitLog(rocdecodehost::logger);
    return ret;
}

/************************************************************************************************************/
//! \fn rocDecStatus ROCDECAPI rocDecGetDecodeStatusHost(rocDecDecoderHandle decoder_handle, int pic_idx, RocdecDecodeStatus* decode_status);
//! Get the decode status for frame corresponding to pic_idx
//! API is currently supported for HEVC codec.
//! API returns ROCDEC_NOT_SUPPORTED error code for unsupported GPU or codec.
/************************************************************************************************************/
rocDecStatus ROCDECAPI 
rocDecGetDecodeStatusHost(rocDecDecoderHandle decoder_handle, int pic_idx, RocdecDecodeStatus* decode_status) {
    FunctionEntryLog(rocdecodehost::logger);
    if (decoder_handle == nullptr || decode_status == nullptr) {
        rocdecodehost::logger.CriticalLog(MakeMsg("Null pointer"));
        FunctionExitLog(rocdecodehost::logger);
        return ROCDEC_INVALID_PARAMETER;
    }
    auto handle = static_cast<DecHandleHost *>(decoder_handle);
    rocDecStatus ret;
    try {
        ret = handle->roc_decoder_host_->GetDecodeStatus(pic_idx, decode_status);
    }
    catch(const std::exception& e) {
        handle->CaptureError(e.what());
        rocdecodehost::logger.CriticalLog(MakeMsg(e.what()));
        FunctionExitLog(rocdecodehost::logger);
        return ROCDEC_RUNTIME_ERROR;
    }
    FunctionExitLog(rocdecodehost::logger);
    return ret;
}

/*********************************************************************************************************/
//! \fn rocDecStatus ROCDECAPI rocDecReconfigureDecoderHost(rocDecDecoderHandle decoder_handle, RocdecReconfigureDecoderInfo *reconfig_params)
//! Used to reuse single decoder for multiple clips. Currently supports resolution change, resize params
//! params, target area params change for same codec. Must be called during RocdecParserParams::pfn_sequence_callback
/*********************************************************************************************************/
rocDecStatus ROCDECAPI 
rocDecReconfigureDecoderHost(rocDecDecoderHandle decoder_handle, RocdecReconfigureDecoderInfo *reconfig_params) {
    FunctionEntryLog(rocdecodehost::logger);
    if (decoder_handle == nullptr || reconfig_params == nullptr) {
        rocdecodehost::logger.CriticalLog(MakeMsg("Null pointer"));
        FunctionExitLog(rocdecodehost::logger);
        return ROCDEC_INVALID_PARAMETER;
    }
    auto handle = static_cast<DecHandleHost *>(decoder_handle);
    rocDecStatus ret;
    try {
        ret = handle->roc_decoder_host_->ReconfigureDecoder(reconfig_params);
    }
    catch(const std::exception& e) {
        handle->CaptureError(e.what());
        rocdecodehost::logger.CriticalLog(MakeMsg(e.what()));
        FunctionExitLog(rocdecodehost::logger);
        return ROCDEC_RUNTIME_ERROR;
    }
    FunctionExitLog(rocdecodehost::logger);
    return ret;
}

/************************************************************************************************************************/
//! \fn rocDecStatus ROCDECAPI rocDecGetVideoFrameHost(rocDecDecoderHandle decoder_handle, int pic_idx, unsigned int *dev_mem_ptr,
//!         unsigned int *horizontal_pitch, RocdecProcParams *vid_postproc_params);
//! Post-process and map video frame corresponding to pic_idx for use in HIP. Returns HIP device pointer and associated
//! pitch(horizontal stride) of the video frame. Returns device memory pointers for each plane (Y, U and V) separately
/************************************************************************************************************************/
rocDecStatus ROCDECAPI 
rocDecGetVideoFrameHost(rocDecDecoderHandle decoder_handle, int pic_idx,
                                                    void **frame_data, uint32_t *line_size,
                                                    RocdecProcParams *vid_postproc_params) {
    FunctionEntryLog(rocdecodehost::logger);
    if (decoder_handle == nullptr || frame_data == nullptr || line_size == nullptr || vid_postproc_params == nullptr) {
        rocdecodehost::logger.CriticalLog(MakeMsg("Null pointer"));
        FunctionExitLog(rocdecodehost::logger);
        return ROCDEC_INVALID_PARAMETER;
    }
    auto handle = static_cast<DecHandleHost *>(decoder_handle);
    rocDecStatus ret;
    try {
        ret = handle->roc_decoder_host_->GetVideoFrame(pic_idx, frame_data, line_size, vid_postproc_params);
    }
    catch(const std::exception& e) {
        handle->CaptureError(e.what());
        rocdecodehost::logger.CriticalLog(MakeMsg(e.what()));
        FunctionExitLog(rocdecodehost::logger);
        return ROCDEC_RUNTIME_ERROR;
    }
    FunctionExitLog(rocdecodehost::logger);
    return ret;
}

/*****************************************************************************************************/
//! \fn const char* ROCDECAPI rocDecGetErrorNameHost(rocDecStatus rocdec_status)
//! \ingroup group_amd_rocdecode
//! Return name of the specified error code in text form.
/*****************************************************************************************************/
const char* ROCDECAPI rocDecGetErrorNameHost(rocDecStatus rocdec_status) {
    FunctionEntryLog(rocdecodehost::logger);
    const char* name;
    switch (rocdec_status) {
        case ROCDEC_DEVICE_INVALID:
            name = "ROCDEC_DEVICE_INVALID";
            break;
        case ROCDEC_CONTEXT_INVALID:
            name = "ROCDEC_CONTEXT_INVALID";
            break;
        case ROCDEC_RUNTIME_ERROR:
            name = "ROCDEC_RUNTIME_ERROR";
            break;
        case ROCDEC_OUTOF_MEMORY:
            name = "ROCDEC_OUTOF_MEMORY";
            break;
        case ROCDEC_INVALID_PARAMETER:
            name = "ROCDEC_INVALID_PARAMETER";
            break;
        case ROCDEC_NOT_IMPLEMENTED:
            name = "ROCDEC_NOT_IMPLEMENTED";
            break;
        case ROCDEC_NOT_INITIALIZED:
            name = "ROCDEC_NOT_INITIALIZED";
            break;
        case ROCDEC_NOT_SUPPORTED:
            name = "ROCDEC_NOT_SUPPORTED";
            break;
        default:
            name = "UNKNOWN_ERROR";
            break;
    }
    FunctionExitLog(rocdecodehost::logger);
    return name;
}