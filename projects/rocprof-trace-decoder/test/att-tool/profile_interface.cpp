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
#include "util.hpp"

#include <cxxabi.h>
#include <cstring>
#include <fstream>

namespace rocprofiler
{
namespace att_wrapper
{
struct trace_data_t
{
    uint8_t*  data{nullptr};
    uint64_t  size{0};
    ToolData* tool{nullptr};

    int gfxip = 0;
    size_t num_occupancy_events = 0;
    size_t num_shaderdata_events = 0;
    size_t num_realtime_events = 0;

    uint64_t min_rt_clock = UINT64_MAX;
    uint64_t max_rt_clock = 0;
    int64_t min_sh_clock = INT64_MAX;
    int64_t max_sh_clock = 0;

    double rt_clock_frequency = 1E8;
};

static bool supress = std::getenv("ATT_SUPPRESS_WARNING") != nullptr;

rocprofiler_thread_trace_decoder_status_t
get_trace_data(rocprofiler_thread_trace_decoder_record_type_t trace_id,
               void*  trace_events,
               size_t trace_size,
               void*  userdata)
{
    C_API_BEGIN

    CHECK_TRUE(userdata);
    trace_data_t& trace_data = *reinterpret_cast<trace_data_t*>(userdata);
    CHECK_TRUE(trace_data.tool);
    ToolData& tool = *trace_data.tool;

    if(trace_id == ROCPROFILER_THREAD_TRACE_DECODER_RECORD_INFO && !supress)
    {
        auto* infos = (rocprofiler_thread_trace_decoder_info_t*) trace_events;
        for(size_t i = 0; i < trace_size; i++)
        {
            if (infos[i] == ROCPROFILER_THREAD_TRACE_DECODER_INFO_DATA_LOST)
                std::cout << "INFO: " << rocprof_trace_decoder_get_info_string(infos[i]) << std::endl;
            else
                WARNING(rocprof_trace_decoder_get_info_string(infos[i]));
        }
    }
    else if(trace_id == ROCPROFILER_THREAD_TRACE_DECODER_RECORD_GFXIP)
    {
        trace_data.gfxip = reinterpret_cast<size_t>(trace_events);
        tool.config.filemgr->gfxip = reinterpret_cast<size_t>(trace_events);
    }
    else if(trace_id == ROCPROFILER_THREAD_TRACE_DECODER_RECORD_OCCUPANCY)
    {
        trace_data.num_occupancy_events += trace_size;
        for(size_t i = 0; i < trace_size; i++)
            tool.config.occupancy.push_back(
                reinterpret_cast<const occupancy_t*>(trace_events)[i]);
    }
    else if(trace_id == ROCPROFILER_THREAD_TRACE_DECODER_RECORD_PERFEVENT)
    {
        PerfcounterFile(tool.config, reinterpret_cast<perfevent_t*>(trace_events), trace_size);
    }
    else if(trace_id == ROCPROFILER_THREAD_TRACE_DECODER_RECORD_REALTIME)
    {
        CHECK_TRUE(trace_size >= 2);
        trace_data.num_realtime_events += trace_size;

        auto* events = static_cast<rocprofiler_thread_trace_decoder_realtime_t*>(trace_events);
        for (size_t i=1; i<trace_size; i++)
        {
            CHECK_TRUE(events[i-1].shader_clock < events[i].shader_clock);
            CHECK_TRUE(events[i-1].realtime_clock < events[i].realtime_clock);
        }

        trace_data.min_rt_clock = std::min(trace_data.min_rt_clock, events[0].realtime_clock);
        trace_data.min_sh_clock = std::min(trace_data.min_sh_clock, events[0].shader_clock);

        trace_data.max_rt_clock = std::max(trace_data.max_rt_clock, events[trace_size-1].realtime_clock);
        trace_data.max_sh_clock = std::max(trace_data.max_sh_clock, events[trace_size-1].shader_clock);
    }
    else if(trace_id == ROCPROFILER_THREAD_TRACE_DECODER_RECORD_RT_FREQUENCY)
    {
        CHECK_TRUE(trace_size >= 1);
        trace_data.rt_clock_frequency = (double) static_cast<uint64_t*>(trace_events)[0];
    }
    else if(trace_id == ROCPROFILER_THREAD_TRACE_DECODER_RECORD_SHADERDATA)
    {
        CHECK_TRUE(trace_size >= 1);
        trace_data.num_shaderdata_events += trace_size;

        auto imm_flag = 1<<ROCPROFILER_THREAD_TRACE_DECODER_SHADERDATA_FLAGS_IMM;
        auto* events = static_cast<rocprofiler_thread_trace_decoder_shaderdata_t*>(trace_events);
        for (size_t i=0; i<trace_size; i++)
        {
            if (events[i].flags & imm_flag)
                CHECK_TRUE(events[i].value == 0x7); // This is the value on s_ttracedata_imm for navi4_sdata
            CHECK_TRUE((events[i].flags & ~imm_flag) == 0) // No other trace with priv sdata here
        }
    }

    if(trace_id != ROCPROFILER_THREAD_TRACE_DECODER_RECORD_WAVE)
        return ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS;

    for(size_t wave_n = 0; wave_n < trace_size; wave_n++)
    {
        auto&   wave           = reinterpret_cast<wave_t*>(trace_events)[wave_n];
        int64_t prev_inst_time = wave.begin_time;

        WaveFile(tool.config, wave);

        for(size_t j = 0; j < wave.instructions_size; j++)
        {
            auto& inst = wave.instructions_array[j];
            if(inst.pc.code_object_id == 0 && inst.pc.address == 0)
                continue;

            auto& line = tool.get(inst.pc);
            line.hitcount += 1;
            line.latency += inst.duration;
            line.stall += inst.stall;
            line.idle += std::max<int64_t>(inst.time - prev_inst_time, 0);
            prev_inst_time = std::max(prev_inst_time, inst.time + inst.duration);
        }
    }

    return ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS;

    C_API_END

    return ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR;
}

uint64_t
copy_trace_data(uint8_t** buffer, uint64_t* buffer_size, void* userdata)
{
    trace_data_t& data = *reinterpret_cast<trace_data_t*>(userdata);
    *buffer_size       = data.size;
    *buffer            = data.data;
    data.size          = 0;
    return *buffer_size;
}

rocprofiler_thread_trace_decoder_status_t
isa_callback(char*     isa_instruction,
             uint64_t* isa_memory_size,
             uint64_t* isa_size,
             pcinfo_t  pc,
             void*     userdata)
{
    C_API_BEGIN
    CHECK_TRUE(userdata);
    trace_data_t& trace_data = *reinterpret_cast<trace_data_t*>(userdata);
    CHECK_TRUE(trace_data.tool);
    ToolData& tool = *trace_data.tool;

    std::shared_ptr<Instruction> instruction{nullptr};

    try
    {
        CodeLine& line = tool.get(pc);
        instruction    = line.code_line;
    } catch(std::exception& e)
    {
        if (!supress) WARNING(pc.code_object_id << ":" << pc.address << ' ' << e.what());
        return ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR;
    }

    if(!instruction.get()) return ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR_INVALID_ARGUMENT;

    {
        size_t tmp_isa_size = *isa_size;
        *isa_size           = instruction->inst.size();

        if(*isa_size > tmp_isa_size) return ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR_OUT_OF_RESOURCES;
    }

    memcpy(isa_instruction, instruction->inst.data(), *isa_size);
    *isa_memory_size = instruction->size;

    C_API_END
    return ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS;
}

ToolData::ToolData(const std::vector<char>& _data, WaveConfig& _config)
: cfile(_config.code)
, config(_config)
{
    trace_data_t data{.data = (uint8_t*) _data.data(),
                      .size = _data.size(),
                      .tool = this};

    auto status = rocprof_trace_decoder_parse_data(copy_trace_data, get_trace_data, isa_callback, &data);
    CHECK_DECODER(status);
    if (std::getenv("AM_DONT_CHECK") != nullptr) return;

    CHECK_TRUE(data.gfxip >= 9 && data.gfxip <= 12);
    CHECK_TRUE(data.num_occupancy_events > 0 || getenv("SKIP_OCCUPANCY") != nullptr);

    if (std::getenv("CHECK_SDATA") != nullptr) CHECK_TRUE(data.num_shaderdata_events > 0);

    // Gfx9 is not guaranteed to have RT
    if (data.gfxip == 9 && data.num_realtime_events <= 1) return;
    CHECK_TRUE(data.num_realtime_events > 1);

    double delta_sh = data.max_sh_clock - data.min_sh_clock;
    double delta_rt = data.max_rt_clock - data.min_rt_clock;

    double frequency = delta_sh / delta_rt * data.rt_clock_frequency;
    CHECK_TRUE(frequency > 0.4E9);
    CHECK_TRUE(frequency < 3.5E9);
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
    try {
        rocprofiler::sdk::codeobj::segment::CodeobjTableTranslator symbol_table;
        for(auto& [vaddr, symbol] : cfile->table->getSymbolMap(_pc.code_object_id))
            symbol_table.insert({symbol.vaddr, symbol.mem_size, _pc.code_object_id});

        auto addr_range = symbol_table.find_codeobj_in_range(_pc.address);
        try
        {
            auto symbol = cfile->table->getSymbolMap(_pc.code_object_id).at(addr_range.address);
            auto pair   = KernelName{symbol.name, demangle(symbol.name)};
            cfile->kernel_names.emplace(pcinfo_t{addr_range.address, _pc.code_object_id}, pair);
        } catch(...)
        {
            WARNING("Could not find kernel symbol for " << _pc.code_object_id << ':' << addr_range.address);
        }

        for(auto addr = addr_range.address; addr < addr_range.address + addr_range.size;)
        {
            pcinfo_t info{.address = addr, .code_object_id = addr_range.id};
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

    cline.code_line = cfile->table->get(_pc.code_object_id, _pc.address);

    return cline;
}

}  // namespace att_wrapper
}  // namespace rocprofiler
