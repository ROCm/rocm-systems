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

#include <memory>
#include <string>
#include "rocdecode/rocparser.h"
#include "roc_video_parser.h"
#include "avc_parser.h"
#include "av1_parser.h"
#include "hevc_parser.h"
#include "vp9_parser.h"

class RocParserHandle {
public:
    explicit RocParserHandle(RocdecParserParams *params) { CreateParser(params); };
    ~RocParserHandle() { ClearErrors(); }
    bool NoError() { return error_.empty(); }
    const char* ErrorMsg() { return error_.c_str(); }
    void CaptureError(const std::string& err_msg) { error_ = err_msg; }
    rocDecStatus ParseVideoData(RocdecSourceDataPacket *packet) { return roc_parser_->ParseVideoData(packet); }
    rocDecStatus DestroyParser() { return DestroyParserInternal(); };

private:
    std::shared_ptr<RocVideoParser> roc_parser_ = nullptr;
    void ClearErrors() { error_ = ""; }
    void CreateParser(RocdecParserParams *params) {
        switch(params->codec_type) {
            case rocDecVideoCodec_AVC:
                roc_parser_ = std::make_shared<AvcVideoParser>();
                break;
            case rocDecVideoCodec_HEVC:
                roc_parser_ = std::make_shared<HevcVideoParser>();
                break;
            case rocDecVideoCodec_VP9:
                roc_parser_ = std::make_shared<Vp9VideoParser>();
                break;
            case rocDecVideoCodec_AV1:
                roc_parser_ = std::make_shared<Av1VideoParser>();
                break;
            default:
                THROW("Unsupported parser type "+ TOSTR(params->codec_type));
                break;
        }

        if (roc_parser_ ) {
            rocDecStatus ret = roc_parser_->Initialize(params);
            if (ret != ROCDEC_SUCCESS)
                THROW("rocParser Initialization failed with error: "+ TOSTR(ret));
        }
    }
    rocDecStatus DestroyParserInternal() {
      rocDecStatus ret = ROCDEC_NOT_INITIALIZED;
        if (roc_parser_) {
            ret = roc_parser_->UnInitialize();
            if (ret != ROCDEC_SUCCESS)
                THROW("rocParser UnInitialization failed with error: "+ TOSTR(ret));
        }
        return ret;
    }

    std::string error_;
};