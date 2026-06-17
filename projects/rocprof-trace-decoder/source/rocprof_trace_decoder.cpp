// MIT License
//
// Copyright (c) 2024-2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include "rocprof_trace_decoder/rocprof_trace_decoder.h"

#include <atomic>
#include <cstdint>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "handle.hpp"
#include "stitch/stitch.hpp"
#include "trace_parser.hpp"

#ifndef ROCPROF_TRACE_DECODER_COMGR_DISABLED
#    include "rocprof_trace_decoder/cxx/code_printing.hpp"
#endif

#define RADT(x) ROCPROFILER_THREAD_TRACE_DECODER_RECORD_##x

// ============================================================================
// ISA service adapter for the internal parser
// ============================================================================

class CodeService : public ICodeServicer
{
public:
    CodeService() = delete;
    CodeService(rocprof_trace_decoder_isa_callback_t isa_str, void* _userdata) : isa_cb(isa_str), userdata(_userdata)
    {
        memory_copy.resize(64);
    };

    virtual assemblyLine GetInstruction(pcinfo_t pc, int gfxip) override
    {
        uint64_t isa_len = memory_copy.size();
        uint64_t memsize = 0;

        auto status = isa_cb(memory_copy.data(), &memsize, &isa_len, pc, userdata);
        if (status == ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR_OUT_OF_RESOURCES)
        {
            if (memory_copy.size() < isa_len) memory_copy.resize(isa_len);

            status = isa_cb(memory_copy.data(), &memsize, &isa_len, pc, userdata);
        }

        if (status != ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS)
            throw std::invalid_argument("ISA Callback returned error " + std::to_string(status));

        // A callback that can't determine the instruction length may report 0,
        // which would make isa.next == isa.addr and spin the stitcher's PC walk
        // forever. Clamp to the minimum encodable instruction size (4 bytes).
        if (memsize < 4) memsize = 4;

        assemblyLine isa;
        isa.addr = pc;
        isa.line = memory_copy.substr(0, isa_len);
        isa.next = {isa.addr.address + memsize, isa.addr.code_object_id};
        isa.cat = Trie::inst_type(isa.line, gfxip);
        return isa;
    };

private:
    rocprof_trace_decoder_isa_callback_t const isa_cb;
    void* const userdata;
    std::string memory_copy;
};

// ============================================================================
// Internal parse implementation
// ============================================================================

static rocprofiler_thread_trace_decoder_status_t parse_data_impl(
    rocprof_trace_decoder_se_data_callback_t se_data_callback,
    rocprof_trace_decoder_trace_callback_t trace_callback,
    rocprof_trace_decoder_isa_callback_t isa_callback,
    void* cbdata
)
{
    uint8_t* buffer = nullptr;
    uint64_t buffer_size = 0;
    size_t remaining = se_data_callback(&buffer, &buffer_size, cbdata);

    std::once_flag rt_freq_flag{};

    auto EmitWarning = [&](rocprofiler_thread_trace_decoder_info_t info)
    { trace_callback(RADT(INFO), (void*) &info, 1, cbdata); };

    auto _isa = std::make_shared<CodeService>(isa_callback, cbdata);
    Stitcher stitcher{_isa, trace_callback, cbdata};

    while (remaining && buffer_size && buffer)
    {
        CppReturnInfo ret{};

        auto parser = AnalyseBinary_internal(ret, buffer, buffer_size, -1, stitcher);
        if (!parser) return ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR_INVALID_SHADER_DATA;

        if (ret.bPacketLost) EmitWarning(ROCPROFILER_THREAD_TRACE_DECODER_INFO_DATA_LOST);

        auto& perf = ret.perfevents;

        if (ret.realtime_frequency != 0)
            std::call_once(rt_freq_flag, trace_callback, RADT(RT_FREQUENCY), &ret.realtime_frequency, 1, cbdata);

        if (!perf.empty()) trace_callback(RADT(PERFEVENT), (void*) perf.data(), perf.size(), cbdata);
        perf.clear();

        remaining = se_data_callback(&buffer, &buffer_size, cbdata);
    }

    return ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS;
}

// ============================================================================
// Handle management (always available, no COMGR needed)
// ============================================================================

namespace
{
std::atomic<uint64_t> g_handle_counter{1};

// The "comgr_isa_callback" name is historical — this callback is the bridge
// from the wave-trace parser to whichever disassembly backend is compiled in
// (amd_comgr, LLVM-C, or none). It just calls into DecoderInstance::table
// which delegates to DisassemblyInstance::ReadInstruction; the backend choice
// is opaque from here. The COMGR_DISABLED gate now means "no disasm backend
// at all" — when LLVM is selected, this is enabled.
#ifndef ROCPROF_TRACE_DECODER_COMGR_DISABLED
rocprofiler_thread_trace_decoder_status_t comgr_isa_callback(
    char* isa_instruction,
    uint64_t* isa_memory_size,
    uint64_t* isa_size,
    rocprofiler_thread_trace_decoder_pc_t pc,
    void* userdata
)
{
    if (!userdata) return ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR;

    const auto& hd = *static_cast<ReadLock<HandleData>*>(userdata);
    if (!hd.valid()) return ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR;

    auto decoder = hd->decoder();
    if (!decoder.valid()) return ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR;

    try
    {
        auto instruction = decoder->get(pc.code_object_id, pc.address);

        if (!instruction) return ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR_INVALID_ARGUMENT;

        {
            size_t tmp_isa_size = *isa_size;
            *isa_size = instruction->inst.size();

            if (*isa_size > tmp_isa_size) return ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR_OUT_OF_RESOURCES;
        }

        std::memcpy(isa_instruction, instruction->inst.data(), *isa_size);
        *isa_memory_size = instruction->size;
    }
    catch (...)
    {
        return ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR;
    }
    return ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS;
}
#endif

// Adapter context for routing callbacks through parse_data_impl
struct parse_ctx_t
{
    const uint8_t* data{nullptr};
    uint64_t size{0};
    bool consumed{false};

    rocprof_trace_decoder_isa_callback_t isa_cb{nullptr};
    void* isa_ud{nullptr};

    rocprof_trace_decoder_trace_callback_t trace_cb{nullptr};
    void* trace_ud{nullptr};

    rocprof_trace_decoder_se_data_callback_t se_data_cb{nullptr};
    void* se_data_ud{nullptr};
};

uint64_t parse_se_adapter(uint8_t** buffer, uint64_t* buffer_size, void* userdata)
{
    auto* ctx = static_cast<parse_ctx_t*>(userdata);
    if (ctx->consumed || ctx->size == 0)
    {
        *buffer = nullptr;
        *buffer_size = 0;
        return 0;
    }
    *buffer = const_cast<uint8_t*>(ctx->data);
    *buffer_size = ctx->size;
    ctx->consumed = true;
    return ctx->size;
}

uint64_t se_data_cb_adapter(uint8_t** buffer, uint64_t* buffer_size, void* userdata)
{
    auto* ctx = static_cast<parse_ctx_t*>(userdata);
    return ctx->se_data_cb(buffer, buffer_size, ctx->se_data_ud);
}

rocprofiler_thread_trace_decoder_status_t parse_trace_adapter(
    rocprofiler_thread_trace_decoder_record_type_t record_type_id,
    void* trace_events,
    uint64_t trace_size,
    void* userdata
)
{
    auto* ctx = static_cast<parse_ctx_t*>(userdata);
    return ctx->trace_cb(record_type_id, trace_events, trace_size, ctx->trace_ud);
}

rocprofiler_thread_trace_decoder_status_t parse_isa_adapter(
    char* instruction,
    uint64_t* memory_size,
    uint64_t* size,
    rocprofiler_thread_trace_decoder_pc_t address,
    void* userdata
)
{
    auto* ctx = static_cast<parse_ctx_t*>(userdata);
    return ctx->isa_cb(instruction, memory_size, size, address, ctx->isa_ud);
}

} // namespace

// ============================================================================
// Public C API
// ============================================================================

extern "C"
{
ROCPROF_TRACE_DECODER_API rocprofiler_thread_trace_decoder_status_t
rocprof_trace_decoder_create_handle(rocprof_trace_decoder_handle_t* handle)
{
    if (!handle) return ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR_INVALID_ARGUMENT;

    // HandleData's constructor initializes `instance` (the AddressTable) when
    // a disasm backend is compiled in; nothing to do here.
    auto hd = std::make_shared<HandleData>();

    handle->handle = g_handle_counter.fetch_add(1);

    // Warmup scan
    rocprof_trace_decoder_quick_scan(*handle, 0, nullptr, 0, nullptr, nullptr);

    std::lock_guard<std::mutex> lock(HandleData::get_map_mutex());
    HandleData::get_map()[handle->handle] = std::move(hd);

    return ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS;
}

ROCPROF_TRACE_DECODER_API rocprofiler_thread_trace_decoder_status_t
rocprof_trace_decoder_destroy_handle(rocprof_trace_decoder_handle_t handle)
{
    std::lock_guard<std::mutex> lock(HandleData::get_map_mutex());
    auto& map = HandleData::get_map();
    if (map.erase(handle.handle) == 0) return ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR_INVALID_ARGUMENT;
    return ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS;
}

ROCPROF_TRACE_DECODER_API rocprofiler_thread_trace_decoder_status_t rocprof_trace_decoder_set_isa_callback(
    rocprof_trace_decoder_handle_t handle, rocprof_trace_decoder_isa_callback_t callback, void* userdata
)
{
    auto hd = HandleData::get_write_handle(handle);
    if (!hd.valid()) return ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR_INVALID_ARGUMENT;

    hd->isa_cb = callback;
    hd->isa_userdata = userdata;

    return ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS;
}

ROCPROF_TRACE_DECODER_API rocprofiler_thread_trace_decoder_status_t rocprof_trace_decoder_set_se_data_callback(
    rocprof_trace_decoder_handle_t handle, rocprof_trace_decoder_se_data_callback_t callback, void* userdata
)
{
    auto hd = HandleData::get_write_handle(handle);
    if (!hd.valid()) return ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR_INVALID_ARGUMENT;

    hd->se_data_cb = callback;
    hd->se_data_userdata = userdata;

    return ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS;
}

// COMGR-dependent functions

// V1 API: stateless 4-arg parse, no handle management
ROCPROF_TRACE_DECODER_API rocprofiler_thread_trace_decoder_status_t rocprof_trace_decoder_parse_data(
    rocprof_trace_decoder_se_data_callback_t se_data_callback,
    rocprof_trace_decoder_trace_callback_t trace_callback,
    rocprof_trace_decoder_isa_callback_t isa_callback,
    void* userdata
)
{
    if (!se_data_callback || !trace_callback || !isa_callback)
        return ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR_INVALID_ARGUMENT;

    return parse_data_impl(se_data_callback, trace_callback, isa_callback, userdata);
}

// V2 API: handle-based with built-in code object management

#ifndef ROCPROF_TRACE_DECODER_COMGR_DISABLED

ROCPROF_TRACE_DECODER_API rocprofiler_thread_trace_decoder_status_t rocprof_trace_decoder_codeobj_load(
    rocprof_trace_decoder_handle_t handle,
    uint64_t load_id,
    uint64_t load_addr,
    uint64_t load_size,
    const void* data,
    uint64_t data_size
)
{
    auto hd = HandleData::get_write_handle(handle);
    if (!hd.valid()) return ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR_INVALID_ARGUMENT;

    auto decoder = hd->decoder();
    if (!decoder.valid()) return ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR_INVALID_ARGUMENT;

    try
    {
        decoder->addDecoder(data, data_size, load_id, load_addr, load_size);
    }
    catch (...)
    {
        return ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR;
    }
    return ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS;
}

ROCPROF_TRACE_DECODER_API rocprofiler_thread_trace_decoder_status_t
rocprof_trace_decoder_codeobj_unload(rocprof_trace_decoder_handle_t handle, uint64_t load_id)
{
    auto hd = HandleData::get_write_handle(handle);
    if (!hd.valid()) return ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR_INVALID_ARGUMENT;

    auto decoder = hd->decoder();
    if (!decoder.valid()) return ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR_INVALID_ARGUMENT;

    try
    {
        bool result = decoder->removeDecoder(load_id);
        if (result) return ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS;
    }
    catch (...)
    {}

    return ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR;
}

ROCPROF_TRACE_DECODER_API rocprofiler_thread_trace_decoder_status_t rocprof_trace_decoder_parse(
    rocprof_trace_decoder_handle_t handle,
    const void* data,
    uint64_t data_size,
    rocprof_trace_decoder_trace_callback_t trace_callback,
    void* userdata
)
{
    auto hd = HandleData::get_read_handle(handle);
    if (!hd.valid()) return ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR_INVALID_ARGUMENT;

    if (!trace_callback) return ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR_INVALID_ARGUMENT;

    parse_ctx_t ctx{};
    ctx.trace_cb = trace_callback;
    ctx.trace_ud = userdata;

    auto se_adapter = parse_se_adapter;
    {
        if (hd->isa_cb)
        {
            ctx.isa_cb = hd->isa_cb;
            ctx.isa_ud = hd->isa_userdata;
        }
        else
        {
            ctx.isa_cb = comgr_isa_callback;
            ctx.isa_ud = &hd;
        }

        if (hd->se_data_cb)
        {
            ctx.se_data_cb = hd->se_data_cb;
            ctx.se_data_ud = hd->se_data_userdata;
            se_adapter = se_data_cb_adapter;
        }
    }

    if (se_adapter == parse_se_adapter)
    {
        ctx.data = static_cast<const uint8_t*>(data);
        ctx.size = data_size;
    }

    try
    {
        return parse_data_impl(se_adapter, parse_trace_adapter, parse_isa_adapter, &ctx);
    }
    catch (...)
    {
        return ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR_INVALID_SHADER_DATA;
    }
}

#else // ROCPROF_TRACE_DECODER_COMGR_DISABLED

ROCPROF_TRACE_DECODER_API rocprofiler_thread_trace_decoder_status_t
rocprof_trace_decoder_codeobj_load(rocprof_trace_decoder_handle_t, uint64_t, uint64_t, uint64_t, const void*, uint64_t)
{
    return ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR_NOT_IMPLEMENTED;
}

ROCPROF_TRACE_DECODER_API rocprofiler_thread_trace_decoder_status_t
rocprof_trace_decoder_codeobj_unload(rocprof_trace_decoder_handle_t, uint64_t)
{
    return ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR_NOT_IMPLEMENTED;
}

ROCPROF_TRACE_DECODER_API rocprofiler_thread_trace_decoder_status_t rocprof_trace_decoder_parse(
    rocprof_trace_decoder_handle_t handle,
    const void* data,
    uint64_t data_size,
    rocprof_trace_decoder_trace_callback_t trace_callback,
    void* userdata
)
{
    auto hd = HandleData::get_read_handle(handle);
    if (!hd.valid()) return ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR_INVALID_ARGUMENT;

    if (!trace_callback) return ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR_INVALID_ARGUMENT;

    parse_ctx_t ctx{};
    ctx.trace_cb = trace_callback;
    ctx.trace_ud = userdata;

    auto se_adapter = parse_se_adapter;
    {
        if (!hd->isa_cb) return ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR_NOT_IMPLEMENTED;

        ctx.isa_cb = hd->isa_cb;
        ctx.isa_ud = hd->isa_userdata;

        if (hd->se_data_cb)
        {
            ctx.se_data_cb = hd->se_data_cb;
            ctx.se_data_ud = hd->se_data_userdata;
            se_adapter = se_data_cb_adapter;
        }
    }

    if (se_adapter == parse_se_adapter)
    {
        ctx.data = static_cast<const uint8_t*>(data);
        ctx.size = data_size;
    }

    try
    {
        return parse_data_impl(se_adapter, parse_trace_adapter, parse_isa_adapter, &ctx);
    }
    catch (...)
    {
        return ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR_INVALID_SHADER_DATA;
    }
}

#endif // ROCPROF_TRACE_DECODER_COMGR_DISABLED

ROCPROF_TRACE_DECODER_API const char* rocprof_trace_decoder_get_info_string(rocprofiler_thread_trace_decoder_info_t info
)
{
    static const char* stitch =
        "Stitch Incomplete: The parser could not fully match a trace token to the underlying disassembly.";
    static const char* datalost = "Data Lost: The profiler dropped part of the trace due to bandwidth limitations.";
    static const char* incomplete = "Wave incomplete: The trace was cutoff before all waves ended.";

    const std::map<int, const char*> map = {
        {ROCPROFILER_THREAD_TRACE_DECODER_INFO_NONE,              "NONE"     },
        {ROCPROFILER_THREAD_TRACE_DECODER_INFO_DATA_LOST,         datalost   },
        {ROCPROFILER_THREAD_TRACE_DECODER_INFO_STITCH_INCOMPLETE, stitch     },
        {ROCPROFILER_THREAD_TRACE_DECODER_INFO_WAVE_INCOMPLETE,   incomplete },
        {ROCPROFILER_THREAD_TRACE_DECODER_INFO_LAST,              "INFO_LAST"},
    };

    try
    {
        return map.at((int) info);
    }
    catch (...)
    {
        return "Unknown info parameter.";
    }
}

ROCPROF_TRACE_DECODER_API const char* rocprof_trace_decoder_get_status_string(
    rocprofiler_thread_trace_decoder_status_t status
)
{
    const std::map<int, const char*> map = {
        {ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS,                   "ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS"},
        {ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR,                     "ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR"  },
        {ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR_OUT_OF_RESOURCES,
         "ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR_OUT_OF_RESOURCES"                                                    },
        {ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR_INVALID_ARGUMENT,
         "ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR_INVALID_ARGUMENT"                                                    },
        {ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR_INVALID_SHADER_DATA,
         "ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR_INVALID_SHADER_DATA"                                                 },
        {ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR_NOT_IMPLEMENTED,
         "ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR_NOT_IMPLEMENTED"                                                     },
        {ROCPROFILER_THREAD_TRACE_DECODER_STATUS_LAST,                      "ROCPROFILER_THREAD_TRACE_DECODER_STATUS_LAST"   },
    };

    try
    {
        return map.at((int) status);
    }
    catch (...)
    {
        return "STATUS_UNKNOWN";
    }
}

} // extern "C"
