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

#include "wave.hpp"
#include <nlohmann/json.hpp>
#include "outputfile.hpp"

#include <fstream>
#include <iostream>
#include <sstream>

namespace rocprofiler
{
namespace att_wrapper
{
WaveFile::WaveFile(WaveConfig& config, const wave_t& wave)
{
    ROCP_WARNING_IF(wave.contexts != 0u)
        << "Wave had " << static_cast<int>(wave.contexts) << " context save-restores";

    if(!GlobalDefs::get().has_format("json")) return;
    if(wave.instructions_size == 0 && wave.timeline_size < 3) return;

    assert(config.filemgr);

    int assigned_id = config.id_count.at(wave.simd).at(wave.wave_id)++;
    {
        std::stringstream namess;
        namess << "se" << config.shader_engine << "_sm" << (int) wave.simd << "_sl"
               << (int) wave.wave_id << "_wv" << assigned_id << ".json";

        filename = config.filemgr->dir / namess.str();
        config.filemgr->addwave(
            filename,
            FilenameMgr::Coord{
                config.shader_engine, (int) wave.simd, (int) wave.wave_id, assigned_id},
            wave.begin_time,
            wave.end_time);
    }

    nlohmann::json instructions;

    for(size_t i = 0; i < wave.instructions_size; i++)
    {
        auto& inst = wave.instructions_array[i];
        instructions.push_back({inst.time,
                                static_cast<int>(inst.category),
                                static_cast<int>(inst.stall),
                                static_cast<int64_t>(inst.duration),
                                config.code->line_numbers[inst.pc]});
    }

    nlohmann::json timeline;
    int64_t        acc_time = wave.begin_time;

    for(size_t i = 0; i < wave.timeline_size; i++)
    {
        int type     = wave.timeline_array[i].type;
        int duration = wave.timeline_array[i].duration;

        config.wstates.at(type)->add(acc_time, duration);
        timeline.push_back({type, duration});

        acc_time += duration;
    }

    nlohmann::json waitcnt;

    try
    {
        auto wait_list = WaitcntList(config.filemgr->gfxip, wave, config.code->isa_map);

        for(const auto& line : wait_list.mem_unroll)
            if(!line.dependencies.empty())
            {
                nlohmann::json json_line;
                for(int dep : line.dependencies)
                    json_line.push_back({dep, 0});
                waitcnt.push_back({line.line_number, json_line});
            }
    } catch(std::exception& e)
    {
        ROCP_ERROR << e.what();
    } catch(...)
    {
        ROCP_ERROR << "Generic error";
    }

    nlohmann::json wave_entry = {
        {"cu", wave.cu},
        {"id", assigned_id},
        {"simd", wave.simd},
        {"slot", wave.wave_id},
        {"begin", wave.begin_time},
        {"end", wave.end_time},

        {"instructions", instructions},
        {"timeline", timeline},
        {"waitcnt", waitcnt},
    };

    nlohmann::json metadata = {
        {"name", "SE" + std::to_string(config.shader_engine)},
        {"duration", wave.end_time - wave.begin_time},
        {"wave", wave_entry},
        {"num_stitched", wave.instructions_size},
        {"num_insts", wave.instructions_size},
    };

    OutputFile(filename) << metadata;
}

}  // namespace att_wrapper
}  // namespace rocprofiler
