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

// Shared hotspot tracking + output for the thread_trace samples.
// Both agent flavours feed RECORD_WAVE / RECORD_OCCUPANCY events into the
// same global latency table via `accumulate`, register loaded code objects
// for disassembly via `register_codeobj_disasm`, and emit the same
// disassembled top-N hotspot report via `write_top_hotspots`.

#pragma once

#include <rocprofiler-sdk/cxx/codeobj/code_printing.hpp>
#include <rocprofiler-sdk/cxx/operators.hpp>

#include <rocprofiler-sdk/callback_tracing.h>
#include <rocprofiler-sdk/fwd.h>

#include <rocprof_trace_decoder/rocprof_trace_decoder.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <tuple>
#include <vector>

// Free function in the global namespace so std::less<> finds it via ADL when
// instantiating std::map keyed on rocprofiler_thread_trace_decoder_pc_t.
inline bool
operator<(rocprofiler_thread_trace_decoder_pc_t lhs, rocprofiler_thread_trace_decoder_pc_t rhs)
{
    return std::tie(lhs.code_object_id, lhs.address) < std::tie(rhs.code_object_id, rhs.address);
}

namespace hotspots
{
struct Latency
{
    uint64_t latency{0};
    uint64_t hitcount{0};
};

using pcinfo_t     = rocprofiler_thread_trace_decoder_pc_t;
using LatencyTable = std::map<pcinfo_t, Latency>;
using AddressTable = rocprofiler::sdk::codeobj::disassembly::CodeobjAddressTranslate;

inline std::mutex&
lock()
{
    static std::mutex m;
    return m;
}

// Heap-allocated and intentionally leaked: rocprofiler-sdk invokes
// tool_fini during its own shutdown, which runs AFTER namespace-scope
// (and function-local static) destructors would otherwise have fired.
// Atomics below are trivially destructible so they survive; the maps
// are not, so we leak them deliberately.
inline LatencyTable&
latencies()
{
    static auto* t = new LatencyTable{};
    return *t;
}
inline AddressTable&
table()
{
    static auto* t = new AddressTable{};
    return *t;
}
inline std::atomic<int64_t>&
wave_lifetime()
{
    static std::atomic<int64_t> v{0};
    return v;
}
inline std::atomic<int64_t>&
waves_started()
{
    static std::atomic<int64_t> v{0};
    return v;
}
inline std::atomic<int64_t>&
waves_ended()
{
    static std::atomic<int64_t> v{0};
    return v;
}

// Aggregate WAVE / OCCUPANCY records into the shared latency table.
// Suitable as the body of a rocprof_trace_decoder_parse callback.
inline rocprofiler_thread_trace_decoder_status_t
accumulate(rocprofiler_thread_trace_decoder_record_type_t record_type_id,
           void*                                          events,
           uint64_t                                       num_events,
           void* /*userdata*/)
{
    if(record_type_id == ROCPROFILER_THREAD_TRACE_DECODER_RECORD_OCCUPANCY)
    {
        for(uint64_t i = 0; i < num_events; i++)
        {
            auto& e = static_cast<rocprofiler_thread_trace_decoder_occupancy_t*>(events)[i];
            if(e.start)
            {
                wave_lifetime().fetch_sub(static_cast<int64_t>(e.time));
                waves_started().fetch_add(1);
            }
            else
            {
                wave_lifetime().fetch_add(static_cast<int64_t>(e.time));
                waves_ended().fetch_add(1);
            }
        }
    }
    else if(record_type_id == ROCPROFILER_THREAD_TRACE_DECODER_RECORD_WAVE)
    {
        std::lock_guard<std::mutex> guard(lock());
        auto* waves = static_cast<rocprofiler_thread_trace_decoder_wave_t*>(events);
        for(uint64_t w = 0; w < num_events; w++)
        {
            auto& wave = waves[w];
            for(uint64_t i = 0; i < wave.instructions_size; i++)
            {
                auto& inst = wave.instructions_array[i];
                auto& l    = latencies()[inst.pc];
                l.latency += inst.duration;
                l.hitcount += 1;
            }
        }
    }
    return ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS;
}

// Register a loaded code object with the shared AddressTable so the
// post-decode hotspot output can disassemble instructions at PC.
inline void
register_codeobj_disasm(const rocprofiler_callback_tracing_code_object_load_data_t& data)
{
    std::lock_guard<std::mutex> guard(lock());
    if(data.storage_type == ROCPROFILER_CODE_OBJECT_STORAGE_TYPE_FILE)
    {
        table().addDecoder(data.uri, data.code_object_id, data.load_delta, data.load_size);
        return;
    }
    auto* memorybase = reinterpret_cast<const void*>(data.memory_base);
    if(!memorybase) std::abort();
    table().addDecoder(
        memorybase, data.memory_size, data.code_object_id, data.load_delta, data.load_size);
}

inline void
write_top_hotspots(const std::string& path, size_t topn = 50)
{
    std::ofstream file(path);
    if(!file.is_open())
        std::cout << "Could not open log file: " << path << ", writing to stdout\n";
    else
        std::cout << "Writing log to: " << path << std::endl;

    std::ostream& output = file.is_open() ? file : std::cout;

    using Element = std::pair<pcinfo_t, Latency>;
    std::lock_guard<std::mutex> guard(lock());

    std::vector<Element> sorted(latencies().begin(), latencies().end());
    std::stable_sort(sorted.begin(), sorted.end(), [](const Element& a, const Element& b) {
        return a.second.latency > b.second.latency;
    });

    output << "Top " << topn << " hotspots for trace (cycles):\n";
    for(size_t i = 0; i < sorted.size() && i < topn; i++)
    {
        auto& addr = sorted.at(i).first;
        auto& lat  = sorted.at(i).second;
        auto  inst = std::unique_ptr<rocprofiler::sdk::codeobj::disassembly::Instruction>{};
        if(addr.code_object_id != 0 || addr.address != 0)
        {
            try
            {
                inst = table().get(addr.code_object_id, addr.address);
            } catch(const std::exception&)
            {}
        }

        if(!inst)
        {
            output << "Latency:" << lat.latency << "\tHit:" << lat.hitcount << " \t<no disasm for "
                   << "co=" << addr.code_object_id << " addr=0x" << std::hex << addr.address
                   << std::dec << ">\n";
            continue;
        }

        auto   comment = inst->comment;
        size_t pos     = comment.rfind('/');
        if(pos != std::string::npos && pos + 1 < comment.size()) comment = comment.substr(pos + 1);

        output << "Latency:" << lat.latency << "\tHit:" << lat.hitcount << " \t" << inst->inst
               << " [" << comment << "]\n";
    }

    int64_t ws = waves_started().load();
    int64_t we = waves_ended().load();
    int64_t wl = wave_lifetime().load();
    if(ws != we)
        std::cerr << "Error: Some waves have not ended!" << std::endl;
    else if(ws == 0)
        std::cerr << "Error: No waves started!" << std::endl;
    else
        output << "\nMean wave lifetime: " << wl / ws << " cycles";

    output << "\nWaves started: " << ws << "\nWaves ended: " << we << "\n";
}

}  // namespace hotspots
