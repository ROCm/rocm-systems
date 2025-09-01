// MIT License
//
// Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
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

#include "common.hpp"

#include <unistd.h>
#include <cassert>
#include <fstream>
#include <map>
#include <memory>
#include <set>
#include <shared_mutex>

namespace common
{
using map_value_t      = std::unordered_map<std::string, uint64_t>;
using agent_counters_t = std::unordered_map<uint64_t, std::vector<uint64_t>>;

static std::shared_mutex mut{};

auto&
agent_counters()
{
    static auto* ids = new agent_counters_t();
    return *ids;
}

auto&
id_values()
{
    static auto* ids = new std::unordered_map<uint64_t, map_value_t>();
    return *ids;
}

void
spm_data_callback(rocprofiler_spm_counter_record_t* records,
                  size_t                            record_count,
                  rocprofiler_spm_record_flags_t /* flags*/,
                  rocprofiler_user_data_t /*userdata*/)
{
    std::unique_lock<std::shared_mutex> lk(mut);
    // if(!flags&ROCPROFILER_SPM_RECORD_FLAG_DATA) return;

    for(size_t count = 0; count < record_count; count++)
    {
        auto     _counter_id = rocprofiler_counter_id_t{};
        auto     _info       = rocprofiler_counter_info_v0_t{};
        auto     agent_id    = records[count].agent_id;
        uint64_t value       = records[count].value;
        ROCPROFILER_CALL(rocprofiler_query_record_counter_id(records[count].id, &_counter_id),
                         "query record counter id");
        ROCPROFILER_CALL(
            rocprofiler_query_counter_info(
                _counter_id, ROCPROFILER_COUNTER_INFO_VERSION_0, static_cast<void*>(&_info)),
            "Could not query counter_id");
        if(id_values().find(agent_id.handle) == id_values().end())
            id_values()[agent_id.handle] = {{_info.name, value}};
        auto& val = id_values()[agent_id.handle];
        if(val.find(_info.name) == val.end()) val[_info.name] = 0;
        val[_info.name] += value;
    }
}

void
finalize()
{
    for(auto& [agent, value_by_name] : id_values())
    {
        float tcp_access = value_by_name["TCP_TOTAL_READ"] + value_by_name["TCP_TOTAL_WRITE"];
        float tcp_miss   = value_by_name["TCP_TCC_READ_REQ"] + value_by_name["TCP_TCC_WRITE_REQ"];

        // mi200 values
        constexpr float se_to_cu   = 100.0f / 13;
        constexpr float se_per_xcd = 800.0f;
        auto info = std::stringstream{};

        info << "\n\n";
        info << "SALU Insts:    "
                  << 1.0f * value_by_name["SQ_INSTS_SALU"] / value_by_name["SQ_WAVES"] << std::endl;
        info << "VALU Insts:    "
                  << 1.0f * value_by_name["SQ_INSTS_VALU"] / value_by_name["SQ_WAVES"] << std::endl;
        info << "Block Dim:     "
                  << 1.0f * value_by_name["SQ_WAVES"] / value_by_name["SPI_CSN_NUM_THREADGROUPS"]
                  << std::endl;
        info << "L2 HIT+MISS:   "
                  << 100.0f * (value_by_name["TCC_HIT"] + value_by_name["TCC_MISS"]) /
                         value_by_name["TCC_REQ"]
                  << " %" << std::endl;
        info << "SQC HIT+MISS:  "
                  << 100.0f *
                         (value_by_name["SQC_ICACHE_HITS"] + value_by_name["SQC_ICACHE_MISSES"]) /
                         value_by_name["SQC_ICACHE_REQ"]
                  << " %" << std::endl;
        info << "CPC BUSY+IDLE: "
                  << se_per_xcd *
                         (value_by_name["CPC_CPC_STAT_BUSY"] + value_by_name["CPC_CPC_STAT_IDLE"]) /
                         value_by_name["SQ_CYCLES"]
                  << " %" << std::endl;
        info << "\n";
        info << "Active SPI:    "
                  << 100.0f * value_by_name["SPI_CSN_WINDOW_VALID"] / value_by_name["SQ_CYCLES"]
                  << " %" << std::endl;
        info << "SPI Busy:      "
                  << 100.0f * value_by_name["SPI_CSN_BUSY"] / value_by_name["SQ_CYCLES"] << " %"
                  << std::endl;
        info << "SQ CU Busy:    "
                  << se_to_cu * value_by_name["SQ_BUSY_CU_CYCLES"] / value_by_name["SQ_CYCLES"]
                  << " %" << std::endl;
        info << "CPC Busy:      "
                  << se_per_xcd * value_by_name["CPC_CPC_STAT_BUSY"] / value_by_name["SQ_CYCLES"]
                  << " %" << std::endl;
        info << "CPC Idle:      "
                  << se_per_xcd * value_by_name["CPC_CPC_STAT_IDLE"] / value_by_name["SQ_CYCLES"]
                  << " %" << std::endl;
        info << "SQC HIT:       "
                  << 100.0f * value_by_name["SQC_ICACHE_HITS"] / value_by_name["SQC_ICACHE_REQ"]
                  << " %" << std::endl;
        info << "SQC MISS:      "
                  << 100.0f * value_by_name["SQC_ICACHE_MISSES"] / value_by_name["SQC_ICACHE_REQ"]
                  << " %" << std::endl;
        info << "L2 Hit:        "
                  << 100.0f * value_by_name["TCC_HIT"] / value_by_name["TCC_REQ"] << " %"
                  << std::endl;
        info << "L2 Miss:       "
                  << 100.0f * value_by_name["TCC_MISS"] / value_by_name["TCC_REQ"] << " %"
                  << std::endl;
        info << "L1 Read/wave:  "
                  << 1.0f * value_by_name["TCP_TOTAL_READ"] / value_by_name["SQ_WAVES"]
                  << std::endl;
        info << "L1 Write/wave: "
                  << 1.0f * value_by_name["TCP_TOTAL_WRITE"] / value_by_name["SQ_WAVES"]
                  << std::endl;
        info << "L1->L2 Forward:" << 100.0f * tcp_miss / tcp_access << " %" << std::endl;
        info << "TA BUSY/WAVE:  "
                  << 1.0f * value_by_name["TA_TA_BUSY"] / value_by_name["TA_TOTAL_WAVEFRONTS"]
                  << std::endl;
        std::clog << info.str() << std::endl;
    }
}

rocprofiler_status_t
iterate_agent_counters(rocprofiler_agent_id_t    agent_id,
                       rocprofiler_counter_id_t* counters,
                       size_t                    num_counters,
                       void* /* userdata */)
{
    std::set<std::string> desired = {"TA_TA_BUSY",        "TA_TOTAL_WAVEFRONTS",
                                     "TCC_HIT",           "TCC_MISS",
                                     "TCC_REQ",           "TCP_TOTAL_READ",
                                     "TCP_TOTAL_WRITE",   "TCP_TCC_READ_REQ",
                                     "TCP_TCC_WRITE_REQ", "SQ_CYCLES",
                                     "SQ_BUSY_CU_CYCLES", "SQ_WAVES",
                                     "SQ_INSTS_VALU",     "SQ_INSTS_SALU",
                                     "SQC_ICACHE_REQ",    "SQC_ICACHE_HITS",
                                     "SQC_ICACHE_MISSES", "CPC_CPC_STAT_BUSY",
                                     "CPC_CPC_STAT_IDLE", "SPI_CSN_WINDOW_VALID",
                                     "SPI_CSN_BUSY",      "SPI_CSN_NUM_THREADGROUPS"};

    auto counter_list = std::vector<uint64_t>{};
    for(size_t i = 0; i < num_counters; i++)
    {
        rocprofiler_counter_info_v0_t _info{};
        ROCPROFILER_CALL(
            rocprofiler_query_counter_info(
                counters[i], ROCPROFILER_COUNTER_INFO_VERSION_0, static_cast<void*>(&_info)),
            "Could not query counter_id");

        if(desired.find(std::string(_info.name)) != desired.end())
            counter_list.emplace_back(counters[i].handle);
    }

    agent_counters()[agent_id.handle] = std::move(counter_list);
    return ROCPROFILER_STATUS_SUCCESS;
}

std::vector<rocprofiler_counter_id_t>
init_counters(rocprofiler_agent_id_t id)
{
    ROCPROFILER_CALL(
        rocprofiler_iterate_spm_supported_counters(id, common::iterate_agent_counters, nullptr),
        "Iterate counters");

    auto counters = std::vector<rocprofiler_counter_id_t>{};
    for(auto& id_ : agent_counters().at(id.handle))
        counters.push_back(rocprofiler_counter_id_t{.handle = id_});

    return counters;
}

}  // namespace common