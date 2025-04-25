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

#pragma once

#include "att_decoder.h"
#include "code.hpp"
#include "filenames.hpp"
#include "waitcnt/analysis.hpp"
#include "wstates.hpp"

#include "att_lib_wrapper.hpp"

#include <map>
#include <unordered_map>
#include <vector>

namespace rocprofiler
{
namespace att_wrapper
{
constexpr size_t SIMD_NUM  = 4;
constexpr size_t SIMD_SIZE = 32;

class WaveConfig
{
    using WavestateArray = std::array<std::shared_ptr<WstatesFile>, ATT_WAVE_STATE_LAST>;
    using SIMD           = std::array<size_t, SIMD_SIZE>;

public:
    WaveConfig(int                           se_id,
               std::shared_ptr<FilenameMgr>& _mgr,
               std::shared_ptr<CodeFile>&    _code,
               WavestateArray&               _wstates)
    : shader_engine(se_id)
    , wstates(_wstates)
    , code(_code)
    , filemgr(_mgr)
    {}

    const int      shader_engine;
    WavestateArray wstates;

    std::array<SIMD, SIMD_NUM>   id_count{};
    std::shared_ptr<CodeFile>    code;
    std::shared_ptr<FilenameMgr> filemgr;

    std::map<pcinfo_t, KernelName>       kernel_names{};
    std::vector<att_occupancy_info_v2_t> occupancy{};
};

class WaveFile
{
public:
    WaveFile(WaveConfig& config, const att_wave_data_t& wave);
    Fspath filename{};
};

}  // namespace att_wrapper
}  // namespace rocprofiler
