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

#include "profile_interface.hpp"
#include "perfcounter.hpp"

#include <rocprofiler-sdk/experimental/thread-trace/trace_decoder.h>

#include <cxxabi.h>
#include <cstring>
#include <fstream>

namespace rocprofiler
{
namespace att_wrapper
{
void
get_trace_data(rocprofiler_thread_trace_decoder_record_type_t trace_id,
               void*                                          trace_events,
               size_t                                         trace_size,
               void*                                          userdata)
{
    C_API_BEGIN

    CHECK_NOTNULL(userdata);
    ToolData& tool = *static_cast<ToolData*>(userdata);

    if(trace_id == ROCPROFILER_THREAD_TRACE_DECODER_RECORD_INFO)
    {
        auto* infos = (rocprofiler_thread_trace_decoder_info_t*) trace_events;
        for(size_t i = 0; i < trace_size; i++)
            ROCP_WARNING << rocprofiler_thread_trace_decoder_info_string(tool.decoder, infos[i]);
    }
    else if(trace_id == ROCPROFILER_THREAD_TRACE_DECODER_RECORD_GFXIP)
    {
        tool.config.filemgr->gfxip = reinterpret_cast<size_t>(trace_events);
    }
    else if(trace_id == ROCPROFILER_THREAD_TRACE_DECODER_RECORD_OCCUPANCY)
    {
        for(size_t i = 0; i < trace_size; i++)
            tool.config.occupancy.push_back(reinterpret_cast<const occupancy_t*>(trace_events)[i]);
    }
    else if(trace_id == ROCPROFILER_THREAD_TRACE_DECODER_RECORD_PERFEVENT)
    {
        PerfcounterFile(tool.config, reinterpret_cast<perfevent_t*>(trace_events), trace_size);
    }

    if(trace_id != ROCPROFILER_THREAD_TRACE_DECODER_RECORD_WAVE) return;

    bool bInvalid = false;
    for(size_t wave_n = 0; wave_n < trace_size; wave_n++)
    {
        auto&   wave           = reinterpret_cast<wave_t*>(trace_events)[wave_n];
        int64_t prev_inst_time = wave.begin_time;

        for(size_t j = 0; j < wave.instructions_size; j++)
        {
            auto& inst = wave.instructions_array[j];
            if(inst.pc.marker_id == 0 && inst.pc.addr == 0) continue;

            try
            {
                auto& line = tool.get(inst.pc);
                line.hitcount += 1;
                line.latency += inst.duration;
                line.stall += inst.stall;
                line.idle += std::max<int64_t>(inst.time - prev_inst_time, 0);
            } catch(...)
            {
                bInvalid = true;
            }
            prev_inst_time = std::max(prev_inst_time, inst.time + inst.duration);
        }

        WaveFile(tool.config, wave);
    }
    if(bInvalid) ROCP_WARNING << "Could not fetch some instructions!";

    C_API_END
}

ToolData::ToolData(std::vector<char>&                        _data,
                   WaveConfig&                               _config,
                   rocprofiler_thread_trace_decoder_handle_t _decoder)
: cfile(_config.code)
, config(_config)
, decoder(_decoder)
{
    auto status =
        rocprofiler_trace_decode(decoder, get_trace_data, _data.data(), _data.size(), this);
    ROCP_ERROR_IF(status != ROCPROFILER_STATUS_SUCCESS) << ": " << status;
}

ToolData::~ToolData() = default;

std::string
demangle(std::string_view line)
{
    int   status{0};
    char* c_name = abi::__cxa_demangle(line.data(), nullptr, nullptr, &status);

    if(c_name == nullptr) return "";

    std::string str = c_name;
    free(c_name);
    return str;
}

CodeLine&
ToolData::get(pcinfo_t _pc)
{
    auto& isa_map = cfile->isa_map;
    if(isa_map.find(_pc) != isa_map.end()) return *isa_map.at(_pc);

    // Attempt to disassemble full kernel
    if(_pc.marker_id != 0u) try
        {
            rocprofiler::sdk::codeobj::segment::CodeobjTableTranslator symbol_table;
            for(auto& [vaddr, symbol] : cfile->table->getSymbolMap(_pc.marker_id))
                symbol_table.insert({symbol.vaddr, symbol.mem_size, _pc.marker_id});

            auto addr_range = symbol_table.find_codeobj_in_range(_pc.addr);
            try
            {
                auto symbol = cfile->table->getSymbolMap(_pc.marker_id).at(addr_range.addr);
                auto pair   = KernelName{symbol.name, demangle(symbol.name)};
                cfile->kernel_names.emplace(pcinfo_t{addr_range.addr, _pc.marker_id}, pair);
            } catch(...)
            {
                ROCP_INFO << "Missing kernelSymbol at " << _pc.marker_id << ':' << addr_range.addr;
            }

            for(auto addr = addr_range.addr; addr < addr_range.addr + addr_range.size;)
            {
                pcinfo_t info{.addr = addr, .marker_id = addr_range.id};
                auto& cline = *(isa_map.emplace(info, std::make_unique<CodeLine>()).first->second);

                cline.line_number         = isa_map.size() + cfile->kernel_names.size() - 1;
                cfile->line_numbers[info] = cline.line_number;

                cline.code_line = cfile->table->get(addr_range.id, addr);
                addr += cline.code_line->size;
                if(cline.code_line->size == 0u) throw std::invalid_argument("Line has 0 bytes!");
            }

            if(isa_map.find(_pc) != isa_map.end()) return *isa_map.at(_pc);
        } catch(std::exception& e)
        {}

    auto& cline = *(isa_map.emplace(_pc, std::make_unique<CodeLine>()).first->second);

    cline.line_number        = isa_map.size();
    cfile->line_numbers[_pc] = cline.line_number;

    cline.code_line = cfile->table->get(_pc.marker_id, _pc.addr);

    return cline;
}

}  // namespace att_wrapper
}  // namespace rocprofiler
