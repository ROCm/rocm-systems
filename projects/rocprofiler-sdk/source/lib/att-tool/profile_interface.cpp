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
#include "att_decoder.h"
#include "dl.hpp"
#include "perfcounter.hpp"

#include <cxxabi.h>
#include <cstring>
#include <fstream>

namespace rocprofiler
{
namespace att_wrapper
{
struct trace_data_t
{
    int64_t   id{0};
    uint8_t*  data{nullptr};
    uint64_t  size{0};
    ToolData* tool{nullptr};
};

rocprofiler_att_decoder_status_t
get_trace_data(rocprofiler_att_decoder_record_type_t trace_id,
               int /* shader_id */,
               void*  trace_events,
               size_t trace_size,
               void*  userdata)
{
    C_API_BEGIN

    CHECK_NOTNULL(userdata);
    trace_data_t& trace_data = *reinterpret_cast<trace_data_t*>(userdata);
    CHECK_NOTNULL(trace_data.tool);
    ToolData& tool = *trace_data.tool;

    if(trace_id == ROCPROFILER_ATT_DECODER_TYPE_INFO)
    {
        auto* infos = (rocprofiler_att_decoder_info_t*) trace_events;
        for(size_t i = 0; i < trace_size; i++)
            ROCP_WARNING << tool.dl->att_info_fn(infos[i]);
    }
    else if(trace_id == ROCPROFILER_ATT_DECODER_TYPE_GFXIP)
    {
        tool.config.filemgr->gfxip = reinterpret_cast<size_t>(trace_events);
    }
    else if(trace_id == ROCPROFILER_ATT_DECODER_TYPE_OCCUPANCY)
    {
        for(size_t i = 0; i < trace_size; i++)
            tool.config.occupancy.push_back(
                reinterpret_cast<const att_occupancy_info_v2_t*>(trace_events)[i]);
    }
    else if(trace_id == ROCPROFILER_ATT_DECODER_TYPE_PERFEVENT)
    {
        PerfcounterFile(tool.config, reinterpret_cast<att_perfevent_t*>(trace_events), trace_size);
    }

    if(trace_id != ROCPROFILER_ATT_DECODER_TYPE_WAVE) return ROCPROFILER_ATT_DECODER_STATUS_SUCCESS;

    bool bInvalid = false;
    for(size_t wave_n = 0; wave_n < trace_size; wave_n++)
    {
        auto&   wave           = reinterpret_cast<att_wave_data_t*>(trace_events)[wave_n];
        int64_t prev_inst_time = wave.begin_time;

        WaveFile(tool.config, wave);

        for(size_t j = 0; j < wave.instructions_size; j++)
        {
            auto& inst = wave.instructions_array[j];
            if(inst.pc.marker_id == 0 && inst.pc.addr == 0)
                continue;
            else if(inst.category >= att_wave_inst_category_t::ATT_INST_LAST)
                continue;

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
    }
    if(bInvalid) ROCP_WARNING << "Could not fetch some instructions!";

    return ROCPROFILER_ATT_DECODER_STATUS_SUCCESS;

    C_API_END

    return ROCPROFILER_ATT_DECODER_STATUS_ERROR;
}

uint64_t
copy_trace_data(int* seid, uint8_t** buffer, uint64_t* buffer_size, void* userdata)
{
    trace_data_t& data = *reinterpret_cast<trace_data_t*>(userdata);
    *seid              = data.id;
    *buffer_size       = data.size;
    *buffer            = data.data;
    data.size          = 0;
    return *buffer_size;
}

rocprofiler_att_decoder_status_t
isa_callback(char*     isa_instruction,
             uint64_t* isa_memory_size,
             uint64_t* isa_size,
             pcinfo_t  pc,
             void*     userdata)
{
    C_API_BEGIN
    CHECK_NOTNULL(userdata);
    trace_data_t& trace_data = *reinterpret_cast<trace_data_t*>(userdata);
    CHECK_NOTNULL(trace_data.tool);
    ToolData& tool = *trace_data.tool;

    std::shared_ptr<Instruction> instruction{nullptr};

    try
    {
        CodeLine& line = tool.get(pc);
        instruction    = line.code_line;
    } catch(std::exception& e)
    {
        ROCP_WARNING << pc.marker_id << ":" << pc.addr << ' ' << e.what();
        return ROCPROFILER_ATT_DECODER_STATUS_ERROR;
    }

    if(!instruction.get()) return ROCPROFILER_ATT_DECODER_STATUS_ERROR_INVALID_ARGUMENT;

    {
        size_t tmp_isa_size = *isa_size;
        *isa_size           = instruction->inst.size();

        if(*isa_size > tmp_isa_size) return ROCPROFILER_ATT_DECODER_STATUS_ERROR_OUT_OF_RESOURCES;
    }

    memcpy(isa_instruction, instruction->inst.data(), *isa_size);
    *isa_memory_size = instruction->size;

    C_API_END
    return ROCPROFILER_ATT_DECODER_STATUS_SUCCESS;
}

ToolData::ToolData(const std::vector<char>& _data, WaveConfig& _config, std::shared_ptr<DL> _dl)
: cfile(_config.code)
, config(_config)
, dl(std::move(_dl))
{
    trace_data_t data{.id   = config.shader_engine,
                      .data = (uint8_t*) _data.data(),
                      .size = _data.size(),
                      .tool = this};

    auto status = dl->att_parse_data_fn(copy_trace_data, get_trace_data, isa_callback, &data);
    if(status != ROCPROFILER_ATT_DECODER_STATUS_SUCCESS)
        ROCP_ERROR << "Callback failed with status " << dl->att_status_fn(status);
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
