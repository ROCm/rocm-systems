// MIT License
//
// Copyright (c) 2024-2026 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#pragma once
#include "trace_parser.hpp"

#include <cstdint>
#include <memory>
#include <mutex>
#include <unordered_map>

namespace Handle
{
#ifndef ROCPROF_TRACE_DECODER_COMGR_DISABLED
using AddressTable = rocprof_trace_decoder::codeobj::CodeobjAddressTranslate;
using Instruction = rocprof_trace_decoder::codeobj::Instruction;

struct DecoderInstance
{
    std::mutex mtx{};
    AddressTable table{};
};
#endif

struct HandleData
{
    std::mutex mtx;
#ifndef ROCPROF_TRACE_DECODER_COMGR_DISABLED
    std::shared_ptr<DecoderInstance> instance;
#endif
    rocprof_trace_decoder_isa_callback_t isa_cb{nullptr};
    void* isa_userdata{nullptr};

    rocprof_trace_decoder_se_data_callback_t se_data_cb{nullptr};
    void* se_data_userdata{nullptr};

    // quick_scan state data
    rocprof_trace_decoder_gfx9_header_t header{};
    std::unordered_map<uint64_t, std::unique_ptr<class CSRegisterHandler>> pipestate{};
};

using HandleMap = std::unordered_map<uint64_t, std::shared_ptr<HandleData>>;

std::shared_ptr<HandleData> get_handle_data(rocprof_trace_decoder_handle_t handle);
}