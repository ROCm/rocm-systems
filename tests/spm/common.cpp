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
#include <rocprofiler-sdk/experimental/spm/decode.h>

#include <unistd.h>
#include <cassert>
#include <fstream>
#include <map>
#include <memory>
#include <set>
#include <shared_mutex>

namespace common
{
using map_type_t = std::map<uint64_t, std::string>;

auto&
id_names()
{
    static auto* ids = new std::map<uint64_t, map_type_t>();
    return *ids;
}

class XccBuff
{
    struct CustomPtr
    {
        CustomPtr(char* _ptr)
        : ptr(_ptr)
        {}
        char* const         ptr;
        std::atomic<size_t> offset{0};
    };

public:
    XccBuff(size_t _size, size_t _xcc_num)
    : size(_size)
    , xcc_num(_xcc_num)
    {
        vec.resize(xcc_num * size);
        for(size_t i = 0; i < xcc_num; i++)
            data.push_back(std::make_unique<CustomPtr>(vec.data() + i * size));
    }

    const size_t size;
    const size_t xcc_num;

    std::vector<std::unique_ptr<CustomPtr>> data{};

    std::vector<char> desc{};

private:
    std::vector<char> vec{};
};
auto&
get_buffer()
{
    static auto* buffer = new std::unordered_map<uint64_t, std::unique_ptr<XccBuff>>{};
    return *buffer;
}

static std::shared_mutex mut{};

void
spm_data_callback(rocprofiler_agent_id_t        agent,
                  rocprofiler_spm_record_type_t type,
                  void*                         payload,
                  rocprofiler_user_data_t /* userdata */)
{
    C_API_BEGIN

    if(type == ROCPROFILER_SPM_RECORD_TYPE_SPM_DESC)
    {
        auto& desc = *reinterpret_cast<rocprofiler_spm_descriptor_t*>(payload);
        auto  buf  = std::make_unique<XccBuff>(32 << 20, desc.buffer_num);  // 32MB per XCC

        buf->desc = std::vector<char>((char*) desc.data, (char*) desc.data + desc.size);

        std::unique_lock<std::shared_mutex> lk(mut);
        get_buffer()[agent.handle] = std::move(buf);
    }

    if(type != ROCPROFILER_SPM_RECORD_TYPE_DATA) return;

    auto& xcc_data = *reinterpret_cast<rocprofiler_spm_data_record_t*>(payload);

    std::shared_lock<std::shared_mutex> lk(mut);
    auto*                               buf = get_buffer().at(agent.handle).get();

    char*  ptr    = buf->data.at(xcc_data.buffer_id)->ptr;
    size_t offset = buf->data.at(xcc_data.buffer_id)->offset.fetch_add(xcc_data.data_size);

    size_t xcc_data_size = xcc_data.data_size;

    if(offset + xcc_data_size >= buf->size) throw std::runtime_error("SPM Buffer overflow!");

    memcpy(ptr + offset, xcc_data.data, xcc_data_size);

    C_API_END
}

void
spm_decode_callback(rocprofiler_counter_id_t id,
                    rocprofiler_spm_coord_t* /* dimensions */,
                    uint64_t /* num_dimensions*/,
                    const uint64_t* /* timestamps */,
                    const uint64_t* values,
                    uint64_t        count,
                    void*           userdata)
{
    size_t total = 0;
    for(size_t i = 0; i < count; i++)
        total += values[i];

    auto& map = *reinterpret_cast<std::unordered_map<uint64_t, uint64_t>*>(userdata);
    if(map.find(id.handle) == map.end()) map[id.handle] = 0;
    map.at(id.handle) += total;
};

void
finalize()
{
    std::unique_lock<std::shared_mutex> lk(mut);

    std::unordered_map<std::string, uint64_t> value_by_name{};

    for(auto& [agent, buf] : get_buffer())
    {
        std::unordered_map<uint64_t, uint64_t> counter_total{};

        for(size_t i = 0; i < buf->data.size(); i++)
            if(buf->data[i]->offset.load())
            {
                char*  ptr  = buf->data[i]->ptr;
                size_t size = buf->data[i]->offset.load();

                rocprofiler_spm_descriptor_t desc{};
                desc.data = buf->desc.data();
                desc.size = buf->desc.size();

                std::cout << "Decoding " << i << " with size " << size / 1012000.0f << " MB"
                          << std::endl;

                ROCPROFILER_CALL(
                    rocprofiler_spm_decode(desc, i, spm_decode_callback, ptr, size, &counter_total),
                    "spm decode");
            }

        for(auto& [id, cnt] : counter_total)
        {
            auto& name = id_names().at(agent).at(id);
            if(value_by_name.find(name) == value_by_name.end()) value_by_name[name] = 1E-5f;
            value_by_name.at(name) += cnt;
        }
    }

    delete &get_buffer();

    for(auto& [name, cnt] : value_by_name)
        std::cout << name << ": " << cnt << std::endl;

    float tcp_access = value_by_name["TCP_TOTAL_READ"] + value_by_name["TCP_TOTAL_WRITE"];
    float tcp_miss   = value_by_name["TCP_TCC_READ_REQ"] + value_by_name["TCP_TCC_WRITE_REQ"];

    // mi200 values
    constexpr float se_to_cu   = 100.0f / 13;
    constexpr float se_per_xcd = 800.0f;

    std::cout << "\n\n";
    std::cout << "SALU Insts:    "
              << 1.0f * value_by_name["SQ_INSTS_SALU"] / value_by_name["SQ_WAVES"] << std::endl;
    std::cout << "VALU Insts:    "
              << 1.0f * value_by_name["SQ_INSTS_VALU"] / value_by_name["SQ_WAVES"] << std::endl;
    std::cout << "Block Dim:     "
              << 1.0f * value_by_name["SQ_WAVES"] / value_by_name["SPI_CSN_NUM_THREADGROUPS"]
              << std::endl;
    std::cout << "L2 HIT+MISS:   "
              << 100.0f * (value_by_name["TCC_HIT"] + value_by_name["TCC_MISS"]) /
                     value_by_name["TCC_REQ"]
              << " %" << std::endl;
    std::cout << "SQC HIT+MISS:  "
              << 100.0f * (value_by_name["SQC_ICACHE_HITS"] + value_by_name["SQC_ICACHE_MISSES"]) /
                     value_by_name["SQC_ICACHE_REQ"]
              << " %" << std::endl;
    std::cout << "CPC BUSY+IDLE: "
              << se_per_xcd *
                     (value_by_name["CPC_CPC_STAT_BUSY"] + value_by_name["CPC_CPC_STAT_IDLE"]) /
                     value_by_name["SQ_CYCLES"]
              << " %" << std::endl;
    std::cout << "\n";
    std::cout << "Active SPI:    "
              << 100.0f * value_by_name["SPI_CSN_WINDOW_VALID"] / value_by_name["SQ_CYCLES"] << " %"
              << std::endl;
    std::cout << "SPI Busy:      "
              << 100.0f * value_by_name["SPI_CSN_BUSY"] / value_by_name["SQ_CYCLES"] << " %"
              << std::endl;
    std::cout << "SQ CU Busy:    "
              << se_to_cu * value_by_name["SQ_BUSY_CU_CYCLES"] / value_by_name["SQ_CYCLES"] << " %"
              << std::endl;
    std::cout << "CPC Busy:      "
              << se_per_xcd * value_by_name["CPC_CPC_STAT_BUSY"] / value_by_name["SQ_CYCLES"]
              << " %" << std::endl;
    std::cout << "CPC Idle:      "
              << se_per_xcd * value_by_name["CPC_CPC_STAT_IDLE"] / value_by_name["SQ_CYCLES"]
              << " %" << std::endl;
    std::cout << "SQC HIT:       "
              << 100.0f * value_by_name["SQC_ICACHE_HITS"] / value_by_name["SQC_ICACHE_REQ"] << " %"
              << std::endl;
    std::cout << "SQC MISS:      "
              << 100.0f * value_by_name["SQC_ICACHE_MISSES"] / value_by_name["SQC_ICACHE_REQ"]
              << " %" << std::endl;
    std::cout << "L2 Hit:        " << 100.0f * value_by_name["TCC_HIT"] / value_by_name["TCC_REQ"]
              << " %" << std::endl;
    std::cout << "L2 Miss:       " << 100.0f * value_by_name["TCC_MISS"] / value_by_name["TCC_REQ"]
              << " %" << std::endl;
    std::cout << "L1 Read/wave:  "
              << 1.0f * value_by_name["TCP_TOTAL_READ"] / value_by_name["SQ_WAVES"] << std::endl;
    std::cout << "L1 Write/wave: "
              << 1.0f * value_by_name["TCP_TOTAL_WRITE"] / value_by_name["SQ_WAVES"] << std::endl;
    std::cout << "L1->L2 Forward:" << 100.0f * tcp_miss / tcp_access << " %" << std::endl;
    std::cout << "TA BUSY/WAVE:  "
              << 1.0f * value_by_name["TA_TA_BUSY"] / value_by_name["TA_TOTAL_WAVEFRONTS"]
              << std::endl;
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

    map_type_t counter_list{};
    for(size_t i = 0; i < num_counters; i++)
    {
        rocprofiler_counter_info_v0_t _info{};
        ROCPROFILER_CALL(
            rocprofiler_query_counter_info(
                counters[i], ROCPROFILER_COUNTER_INFO_VERSION_0, static_cast<void*>(&_info)),
            "Could not query counter_id");

        if(desired.find(std::string(_info.name)) != desired.end())
            counter_list.emplace(counters[i].handle, _info.name);
    }

    id_names()[agent_id.handle] = std::move(counter_list);
    return ROCPROFILER_STATUS_SUCCESS;
}

std::vector<rocprofiler_counter_id_t>
init_counters(rocprofiler_agent_id_t id)
{
    ROCPROFILER_CALL(
        rocprofiler_iterate_spm_supported_counters(id, common::iterate_agent_counters, nullptr),
        "Iterate counters");

    std::vector<rocprofiler_counter_id_t> counters{};
    for(auto& [id_, _] : id_names().at(id.handle))
        counters.push_back(rocprofiler_counter_id_t{.handle = id_});

    return counters;
}

}  // namespace common
