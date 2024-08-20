// MIT License
//
// Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.
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

#include <rocprofiler-sdk/amd_detail/thread_trace.h>
#include <rocprofiler-sdk/rocprofiler.h>
#include <atomic>
#include <string_view>
#include <unordered_map>
#include <vector>
#include "lib/rocprofiler-sdk/aql/aql_profile_v2.h"

#define AQLPROFILE_OCCUPANCY_RESOLUTION 8

namespace rocprofiler
{
namespace att_parser
{
hsa_status_t
forward_hsa_error(rocprofiler_status_t error_code)
{
    static thread_local std::unordered_map<int, hsa_status_t> error_fwd = {
        {ROCPROFILER_STATUS_SUCCESS, HSA_STATUS_SUCCESS},
        {ROCPROFILER_STATUS_ERROR, HSA_STATUS_ERROR},
        {ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT, HSA_STATUS_ERROR_INVALID_ARGUMENT},
        {ROCPROFILER_STATUS_ERROR_OUT_OF_RESOURCES, HSA_STATUS_ERROR_OUT_OF_RESOURCES},
    };

    try
    {
        return error_fwd.at(error_code);
    } catch(std::exception& e)
    {}

    return HSA_STATUS_ERROR;
}

rocprofiler_status_t
forward_hsa_error(hsa_status_t error_code)
{
    static thread_local std::unordered_map<int, rocprofiler_status_t> error_fwd = {
        {HSA_STATUS_SUCCESS, ROCPROFILER_STATUS_SUCCESS},
        {HSA_STATUS_ERROR, ROCPROFILER_STATUS_ERROR},
        {HSA_STATUS_ERROR_INVALID_ARGUMENT, ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT},
        {HSA_STATUS_ERROR_OUT_OF_RESOURCES, ROCPROFILER_STATUS_ERROR_OUT_OF_RESOURCES},
    };

    try
    {
        return error_fwd.at(error_code);
    } catch(std::exception& e)
    {}

    return ROCPROFILER_STATUS_ERROR;
}

struct userdata_callback_table_t
{
    rocprofiler_att_parser_trace_callback_t   trace;
    rocprofiler_att_parser_isa_callback_t     isa;
    rocprofiler_att_parser_se_data_callback_t se_data;
    void*                                     user;

    std::vector<pcinfo_t> kernel_id_map;
};

thread_local int TRACE_DATA_ID{-1};
thread_local int KERNEL_ADDR_ID{-1};
thread_local int OCCUPANCY_ID{-1};

void
iterate_trace_type(int id, const char* metadata, void*)
{
    if(std::string_view(metadata).find("occupancy") == 0)
        OCCUPANCY_ID = id;
    else if(std::string_view(metadata).find("kernel_ids_addr") == 0)
        KERNEL_ADDR_ID = id;
    else if(std::string_view(metadata).find("tracedata") == 0)
        TRACE_DATA_ID = id;
}

hsa_status_t
trace_callback(int trace_type_id,
               int /* correlation_id */,
               void*    trace_events,
               uint64_t trace_size,
               void*    userdata)
{
    assert(userdata);
    auto& table = *reinterpret_cast<userdata_callback_table_t*>(userdata);

    if(trace_type_id == KERNEL_ADDR_ID)
    {
        table.kernel_id_map.resize(trace_size);
        const auto* events = reinterpret_cast<const pcinfo_t*>(trace_events);

        for(size_t i = 0; i < trace_size; i++)
            table.kernel_id_map.at(i) = events[i];
    }
    else if(trace_type_id == OCCUPANCY_ID)
    {
        const auto* events = reinterpret_cast<const att_occupancy_info_t*>(trace_events);
        for(size_t i = 0; i < trace_size; i++)
        {
            rocprofiler_att_data_type_occupancy_t occ{};
            occ.timestamp = events[i].time * AQLPROFILE_OCCUPANCY_RESOLUTION;
            occ.enabled   = events[i].enable;
            try
            {
                pcinfo_t kernel_id_addr = table.kernel_id_map.at(events[i].kernel_id);
                occ.marker_id           = kernel_id_addr.marker_id;
                occ.offset              = kernel_id_addr.addr;
            } catch(...)
            {}  // Not having a kernel_id_map entry is unexpected, but valid
            table.trace(ROCPROFILER_ATT_PARSER_DATA_TYPE_OCCUPANCY, (void*) &occ, table.user);
        }
    }
    else if(trace_type_id == TRACE_DATA_ID)
    {
        const auto* events = reinterpret_cast<const att_trace_event_t*>(trace_events);
        for(size_t i = 0; i < trace_size; i++)
        {
            rocprofiler_att_data_type_isa_t isa{};
            isa.marker_id = events[i].pc.marker_id;
            isa.offset    = events[i].pc.addr;
            isa.hitcount  = events[i].hitcount;
            isa.latency   = events[i].latency;
            table.trace(ROCPROFILER_ATT_PARSER_DATA_TYPE_ISA, (void*) &isa, table.user);
        }
    }

    return HSA_STATUS_SUCCESS;
}

hsa_status_t
isa_callback(char* isa,
             char* /*  source_reference  */,
             uint64_t* memory_size,
             uint64_t* isa_size,
             uint64_t* source_size,
             uint64_t  marker,
             uint64_t  offset,
             void*     userdata)
{
    assert(userdata);
    assert(source_size);
    *source_size                = 0;
    const auto&          table  = *reinterpret_cast<const userdata_callback_table_t*>(userdata);
    rocprofiler_status_t status = table.isa(isa, memory_size, isa_size, marker, offset, table.user);

    if(status != ROCPROFILER_STATUS_SUCCESS)
        return rocprofiler::att_parser::forward_hsa_error(status);
    return HSA_STATUS_SUCCESS;
}

uint64_t
se_data_callback(int* seid, uint8_t** buffer, uint64_t* buffer_size, void* userdata)
{
    assert(userdata);
    auto& table = *reinterpret_cast<userdata_callback_table_t*>(userdata);
    return table.se_data(seid, buffer, buffer_size, table.user);
}

}  // namespace att_parser
}  // namespace rocprofiler

extern "C" {
rocprofiler_status_t
rocprofiler_att_parse_data(rocprofiler_att_parser_se_data_callback_t user_se_data_callback,
                           rocprofiler_att_parser_trace_callback_t   user_trace_callback,
                           rocprofiler_att_parser_isa_callback_t     user_isa_callback,
                           void*                                     userdata)
{
    static thread_local bool bInit = []() {
        aqlprofile_att_parser_iterate_event_list(rocprofiler::att_parser::iterate_trace_type,
                                                 nullptr);
        return true;
    }();
    (void) bInit;

    rocprofiler::att_parser::userdata_callback_table_t table;
    table.trace   = user_trace_callback;
    table.isa     = user_isa_callback;
    table.se_data = user_se_data_callback;
    table.user    = userdata;

    hsa_status_t status = aqlprofile_att_parse_data(rocprofiler::att_parser::se_data_callback,
                                                    rocprofiler::att_parser::trace_callback,
                                                    rocprofiler::att_parser::isa_callback,
                                                    (void*) &table);

    if(status != HSA_STATUS_SUCCESS) return rocprofiler::att_parser::forward_hsa_error(status);
    return ROCPROFILER_STATUS_SUCCESS;
}
}
