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

#include "roc_decoder.h"

/**
 * @brief DecHandle structure
 * 
 */
struct DecHandle {

    explicit DecHandle(RocDecoderCreateInfo& decoder_create_info) : roc_decoder_(std::make_shared<RocDecoder>(decoder_create_info)) {};   //constructor
    ~DecHandle() { ClearErrors(); }
    std::shared_ptr<RocDecoder> roc_decoder_;
    bool NoError() { return error_.empty(); }
    const char* ErrorMsg() { return error_.c_str(); }
    void CaptureError(const std::string& err_msg) { error_ = err_msg; }

private:
    void ClearErrors() { error_ = "";}
    std::string error_;
};