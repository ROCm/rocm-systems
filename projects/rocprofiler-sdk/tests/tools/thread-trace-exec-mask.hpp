// MIT License
//
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

#pragma once

#include <rocprofiler-sdk/experimental/thread-trace/trace_decoder_types.h>

#include <atomic>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>

namespace Callbacks
{
// Optional EXEC validation for the single-dispatch test. Aggregate results are shared across
// callbacks, while each decode call keeps its architecture and expected wave size in Trace.
class ExecMaskCheck
{
public:
    struct Trace
    {
        uint64_t gfxip;
        uint64_t wave_size;
    };

    void set_enabled(bool value) { enabled = value; }

    void on_record(rocprofiler_thread_trace_decoder_record_type_t type, void* events, Trace& trace)
    {
        if(type != ROCPROFILER_THREAD_TRACE_DECODER_RECORD_GFXIP) return;
        trace.gfxip = reinterpret_cast<uintptr_t>(events);
        if(trace.gfxip >= 12) has_supported_gfxip = true;
    }

    void check_wave(const rocprofiler_thread_trace_decoder_wave_t& wave, const Trace& trace)
    {
        if(!enabled || trace.gfxip < 12) return;

        // The loaded decoder may predate these fields, regardless of our header version.
        constexpr size_t exec_size =
            offsetof(rocprofiler_thread_trace_decoder_wave_t, execmask_size) +
            sizeof(wave.execmask_size);
        if(wave.size < exec_size)
        {
            old_decoder = true;
            return;
        }

        assert(wave.execmask_size == 0 || wave.execmask != nullptr);
        for(size_t i = 0; i < wave.execmask_size; ++i)
        {
            // branching_kernel's block-uniform condition leaves either the whole wave active
            // or none. Require a full-wave record at finalization.
            assert(trace.wave_size != 0);
            assert(wave.execmask[i].count == 0 || wave.execmask[i].count == trace.wave_size);
            if(wave.execmask[i].count == trace.wave_size) has_exec_mask = true;
        }
    }

    void finalize(bool decoder_available) const
    {
        if(!enabled || !decoder_available || !has_supported_gfxip) return;

        if(old_decoder)
            std::cout << "EXEC mask check skipped: decoder wave.size predates execmask fields\n";
        else if(!has_exec_mask)
            throw std::runtime_error("Missing EXEC mask record with a full wave of active lanes");
        else
            std::cout << "EXEC mask check passed: found records with a full wave of active lanes\n";
    }

private:
    bool              enabled{false};
    std::atomic<bool> has_exec_mask{false};
    std::atomic<bool> has_supported_gfxip{false};
    std::atomic<bool> old_decoder{false};
};
}  // namespace Callbacks
