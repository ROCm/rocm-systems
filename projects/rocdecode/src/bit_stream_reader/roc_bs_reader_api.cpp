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

#include "../commons.h"
#include "bs_reader_handle.h"

namespace rocdecode {
rocDecStatus ROCDECAPI rocDecCreateBitstreamReader(RocdecBitstreamReader *bs_reader_handle, const char *input_file_path) {
    if (bs_reader_handle == nullptr || input_file_path == nullptr) {
        return ROCDEC_INVALID_PARAMETER;
    }
    RocdecBitstreamReader handle = nullptr;
    try {
        handle = new RocBitstreamReaderHandle(input_file_path);
    } 
    catch (const std::exception& e) {
        RocDecLogger::AlwaysLog(STR("Failed to create RocBitstreamReader handle, ") + STR(e.what()));
        return ROCDEC_RUNTIME_ERROR;
    }
    *bs_reader_handle = handle;
    return ROCDEC_SUCCESS;
}

rocDecStatus ROCDECAPI rocDecGetBitstreamCodecType(RocdecBitstreamReader bs_reader_handle, rocDecVideoCodec *codec_type) {
    if (bs_reader_handle == nullptr || codec_type == nullptr) {
        return ROCDEC_INVALID_PARAMETER;
    }
    auto roc_bs_reader_handle = static_cast<RocBitstreamReaderHandle*>(bs_reader_handle);
    rocDecStatus ret;
    try {
        ret = roc_bs_reader_handle->GetBitstreamCodecType(codec_type);
    }
    catch (const std::exception& e) {
        roc_bs_reader_handle->CaptureError(e.what());
        RocDecLogger::AlwaysLog(e.what());
        return ROCDEC_RUNTIME_ERROR;
    }
    return ret;
}

rocDecStatus ROCDECAPI rocDecGetBitstreamBitDepth(RocdecBitstreamReader bs_reader_handle, int *bit_depth) {
    if (bs_reader_handle == nullptr || bit_depth == nullptr) {
        return ROCDEC_INVALID_PARAMETER;
    }
    auto roc_bs_reader_handle = static_cast<RocBitstreamReaderHandle*>(bs_reader_handle);
    rocDecStatus ret;
    try {
        ret = roc_bs_reader_handle->GetBitstreamBitDepth(bit_depth);
    }
    catch (const std::exception& e) {
        roc_bs_reader_handle->CaptureError(e.what());
        RocDecLogger::AlwaysLog(e.what());
        return ROCDEC_RUNTIME_ERROR;
    }
    return ret;
}

rocDecStatus ROCDECAPI rocDecGetBitstreamPicData(RocdecBitstreamReader bs_reader_handle, uint8_t **pic_data, int *pic_size, int64_t *pts) {
    if (bs_reader_handle == nullptr || pic_data == nullptr || pic_size == nullptr || pts == nullptr) {
        return ROCDEC_INVALID_PARAMETER;
    }
    auto roc_bs_reader_handle = static_cast<RocBitstreamReaderHandle*>(bs_reader_handle);
    rocDecStatus ret;
    try {
        ret = roc_bs_reader_handle->GetBitstreamPicData(pic_data, pic_size, pts);
    }
    catch (const std::exception& e) {
        roc_bs_reader_handle->CaptureError(e.what());
        RocDecLogger::AlwaysLog(e.what());
        return ROCDEC_RUNTIME_ERROR;
    }
    return ret;
}

rocDecStatus ROCDECAPI rocDecDestroyBitstreamReader(RocdecBitstreamReader bs_reader_handle) {
    if (bs_reader_handle == nullptr) {
        return ROCDEC_INVALID_PARAMETER;
    }
    auto roc_bs_reader_handle = static_cast<RocBitstreamReaderHandle*>(bs_reader_handle);
    delete roc_bs_reader_handle;
    return ROCDEC_SUCCESS;
}
} // namespace rocdecode