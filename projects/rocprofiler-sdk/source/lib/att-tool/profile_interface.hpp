// MIT License
//
// Copyright (c) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
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
//
// undefine NDEBUG so asserts are implemented
#ifdef NDEBUG
#    undef NDEBUG
#endif
#pragma once

#include <rocprofiler-sdk/experimental/thread-trace/trace_decoder.h>

#include <cxxabi.h>
#include <array>
#include <cstdint>
#include <iostream>
#include <map>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>
#include "code.hpp"
#include "wave.hpp"

#define C_API_BEGIN                                                                                \
    try                                                                                            \
    {
#define C_API_END                                                                                  \
    }                                                                                              \
    catch(std::exception & e)                                                                      \
    {                                                                                              \
        std::cerr << "Error in " << __FILE__ << ':' << __LINE__ << ' ' << e.what() << '\n';        \
    }                                                                                              \
    catch(...) { std::cerr << "Error in " << __FILE__ << ':' << __LINE__ << '\n'; }

namespace rocprofiler
{
namespace att_wrapper
{
using Instruction = rocprofiler::sdk::codeobj::disassembly::Instruction;
using SymbolInfo  = rocprofiler::sdk::codeobj::disassembly::SymbolInfo;

struct ToolData
{
    ToolData(std::vector<char>&                    data,
             WaveConfig&                           config,
             rocprofiler_thread_trace_decoder_id_t decoder);
    ~ToolData();

    CodeLine& get(pcinfo_t pc);

    std::shared_ptr<CodeFile> cfile{};
    WaveConfig&               config;

    std::vector<char> shader_data{};
    size_t            num_waves = 0;

    rocprofiler_thread_trace_decoder_id_t decoder{};
};

}  // namespace att_wrapper
}  // namespace rocprofiler
