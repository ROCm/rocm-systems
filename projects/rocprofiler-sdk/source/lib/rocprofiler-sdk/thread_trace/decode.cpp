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

#include "lib/common/static_object.hpp"
#include "lib/rocprofiler-sdk/aql/helpers.hpp"
#include "lib/rocprofiler-sdk/context/context.hpp"
#include "lib/rocprofiler-sdk/hsa/agent_cache.hpp"
#include "lib/rocprofiler-sdk/registration.hpp"
#include "lib/rocprofiler-sdk/thread_trace/dl.hpp"

#include <rocprofiler-sdk/experimental/thread-trace/trace_decoder.h>
#include <rocprofiler-sdk/experimental/thread_trace.h>

#include <glog/logging.h>

#include <cstdint>

namespace
{
using DL           = rocprofiler::thread_trace::DL;
using AddressTable = rocprofiler::sdk::codeobj::disassembly::CodeobjAddressTranslate;

class DecoderInstance
{
public:
    DecoderInstance(std::unique_ptr<DL> _dl)
    : dl(std::move(_dl))
    {}

    std::unique_ptr<DL> dl{nullptr};
    AddressTable        table{};
};

std::mutex map_mut;

auto&
get_dlopens()
{
    static auto*& _v = rocprofiler::common::static_object<
        std::unordered_map<uint64_t, std::shared_ptr<DecoderInstance>>>::construct();
    return *CHECK_NOTNULL(_v);
}

std::shared_ptr<DecoderInstance>
get_dl(rocprofiler_thread_trace_decoder_handle_t handle)
{
    auto lk = std::unique_lock{map_mut};
    auto it = get_dlopens().find(handle.handle);
    if(it == get_dlopens().end()) return nullptr;

    return it->second;
}
}  // namespace

extern "C" {
rocprofiler_status_t
rocprofiler_thread_trace_decoder_create(rocprofiler_thread_trace_decoder_handle_t* handle,
                                        const char*                                path)
{
    auto dl = std::make_unique<DL>(path);
    if(dl->handle == nullptr) return ROCPROFILER_STATUS_ERROR_NOT_AVAILABLE;
    if(!dl->valid()) return ROCPROFILER_STATUS_ERROR_INCOMPATIBLE_ABI;

    auto            lk    = std::unique_lock{map_mut};
    static uint64_t count = 1;

    auto instance = std::make_shared<DecoderInstance>(std::move(dl));

    handle->handle                = count++;
    get_dlopens()[handle->handle] = std::move(instance);

    return ROCPROFILER_STATUS_SUCCESS;
}

void
rocprofiler_thread_trace_decoder_destroy(rocprofiler_thread_trace_decoder_handle_t handle)
{
    auto lk = std::unique_lock{map_mut};
    get_dlopens().erase(handle.handle);
}

rocprofiler_status_t
rocprofiler_thread_trace_decoder_codeobj_load(rocprofiler_thread_trace_decoder_handle_t handle,
                                              uint64_t                                  load_id,
                                              uint64_t                                  load_addr,
                                              uint64_t                                  load_size,
                                              const void*                               data,
                                              uint64_t                                  size)
{
    auto decoder = get_dl(handle);
    if(decoder == nullptr) return ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT;

    try
    {
        decoder->table.addDecoder(data, size, load_id, load_addr, load_size);
    } catch(...)
    {
        return ROCPROFILER_STATUS_ERROR;
    }
    return ROCPROFILER_STATUS_SUCCESS;
}

rocprofiler_status_t
rocprofiler_thread_trace_decoder_codeobj_unload(rocprofiler_thread_trace_decoder_handle_t handle,
                                                uint64_t                                  load_id)
{
    auto decoder = get_dl(handle);
    if(decoder == nullptr) return ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT;

    try
    {
        if(decoder->table.removeDecoder(load_id)) return ROCPROFILER_STATUS_SUCCESS;
    } catch(std::exception&)
    {}

    return ROCPROFILER_STATUS_ERROR;
}
}

namespace
{
using Instruction = rocprofiler::sdk::codeobj::disassembly::Instruction;
using SymbolInfo  = rocprofiler::sdk::codeobj::disassembly::SymbolInfo;

struct trace_data_t
{
    uint8_t*                         data{nullptr};
    uint64_t                         size{0};
    std::shared_ptr<DecoderInstance> decoder{nullptr};

    rocprofiler_thread_trace_decoder_callback_t cb{nullptr};
    void*                                       userdata{nullptr};
};

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
isa_callback(char*                                 isa_instruction,
             uint64_t*                             isa_memory_size,
             uint64_t*                             isa_size,
             rocprofiler_thread_trace_decoder_pc_t pc,
             void*                                 userdata)
{
    ROCP_FATAL_IF(userdata == nullptr) << "Userdata is null!";
    auto& table = static_cast<trace_data_t*>(userdata)->decoder->table;

    std::unique_ptr<Instruction> instruction{nullptr};

    try
    {
        instruction = table.get(pc.marker_id, pc.addr);
    } catch(std::exception& e)
    {
        ROCP_WARNING << pc.marker_id << ":" << pc.addr << ' ' << e.what();
        return ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR;
    }

    if(!instruction) return ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR_INVALID_ARGUMENT;

    {
        size_t tmp_isa_size = *isa_size;
        *isa_size           = instruction->inst.size();

        if(*isa_size > tmp_isa_size)
            return ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR_OUT_OF_RESOURCES;
    }

    memcpy(isa_instruction, instruction->inst.data(), *isa_size);
    *isa_memory_size = instruction->size;

    return ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS;
}

rocprofiler_thread_trace_decoder_status_t
trace_callback(rocprofiler_thread_trace_decoder_record_type_t record_type_id,
               void*                                          trace_events,
               uint64_t                                       trace_size,
               void*                                          userdata)
{
    ROCP_FATAL_IF(userdata == nullptr) << "Userdata is null!";
    auto* trace_data = static_cast<trace_data_t*>(userdata);
    trace_data->cb(record_type_id, trace_events, trace_size, trace_data->userdata);
    return ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS;
}

}  // namespace

extern "C" {
rocprofiler_status_t
rocprofiler_trace_decode(rocprofiler_thread_trace_decoder_handle_t   handle,
                         rocprofiler_thread_trace_decoder_callback_t user_callback,
                         void*                                       data,
                         uint64_t                                    size,
                         void*                                       userdata)
{
    auto decoder = get_dl(handle);
    if(decoder == nullptr) return ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT;

    trace_data_t cbdata{.data     = static_cast<uint8_t*>(data),
                        .size     = size,
                        .decoder  = decoder,
                        .cb       = user_callback,
                        .userdata = userdata};

    auto status =
        decoder->dl->att_parse_data_fn(copy_trace_data, trace_callback, isa_callback, &cbdata);
    if(status != ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS)
    {
        const char* statustr = decoder->dl->att_status_fn(status);
        if(statustr == nullptr) statustr = "Unknown error";
        ROCP_ERROR << "Callback failed with status " << status << ": " << statustr;
    }

    return ROCPROFILER_STATUS_SUCCESS;
}

const char*
rocprofiler_thread_trace_decoder_info_string(rocprofiler_thread_trace_decoder_handle_t handle,
                                             rocprofiler_thread_trace_decoder_info_t   info)
{
    auto decoder = get_dl(handle);
    if(decoder == nullptr) return nullptr;

    return decoder->dl->att_info_fn(info);
}
}
