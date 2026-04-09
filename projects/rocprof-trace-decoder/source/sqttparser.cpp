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

#include <cstdint>
#include <map>
#include <shared_mutex>
#include <string>
#include <vector>
#include "stitch/stitch.hpp"
#include "trace_decoder_api.h"
#include "trace_parser.hpp"

#define PUBLIC_API __attribute__((visibility("default")))

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

#define RADT(x) ROCPROFILER_THREAD_TRACE_DECODER_RECORD_##x

extern "C"
{
rocprofiler_thread_trace_decoder_status_t _internal_rocprof_trace_decoder_parse_data(
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
};

PUBLIC_API rocprofiler_thread_trace_decoder_status_t rocprof_trace_decoder_parse_data(
    rocprof_trace_decoder_se_data_callback_t se_data_callback,
    rocprof_trace_decoder_trace_callback_t trace_callback,
    rocprof_trace_decoder_isa_callback_t isa_callback,
    void* cbdata
)
{
    try
    {
        return _internal_rocprof_trace_decoder_parse_data(se_data_callback, trace_callback, isa_callback, cbdata);
    }
    catch (...)
    {
        return ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR_INVALID_SHADER_DATA;
    }
};

PUBLIC_API const char* rocprof_trace_decoder_get_info_string(rocprofiler_thread_trace_decoder_info_t info)
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

PUBLIC_API const char* rocprof_trace_decoder_get_status_string(rocprofiler_thread_trace_decoder_status_t status)
{
    static const char* stitch =
        "Stitch Incomplete: The parser could not fully match a trace token to the underlying disassembly.";
    static const char* datalost = "Data Lost: The profiler dropped part of the trace due to bandwidth limitations.";

    const std::map<int, const char*> map = {
        {ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS,                   "ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS"},
        {ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR,                     "ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR"  },
        {ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR_OUT_OF_RESOURCES,
         "ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR_OUT_OF_RESOURCES"                                                    },
        {ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR_INVALID_ARGUMENT,
         "ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR_INVALID_ARGUMENT"                                                    },
        {ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR_INVALID_SHADER_DATA,
         "ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR_INVALID_SHADER_DATA"                                                 },
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
};