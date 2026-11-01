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

// Tests for RDNA dispatch_id tracking and ROCPROF_TRACE_DECODER_EVENT_DISPATCH_END.
// REG_INIT starts a dispatch, COMPUTE_DISPATCH_END marks the end of wave launches, and the
// dispatch ends once all of its waves on the pipe have ended.
// Assumes no waves of a new dispatch are launched while waves of the previous one are running.
// CWSR and data losses are not covered.

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <vector>

#include "gfx10/gfx10token.h"
#include "gfx12/gfx12token.h"
#include "rocprof_trace_decoder/rocprof_trace_decoder.h"
#include "trace_parser.hpp"

namespace
{
constexpr uint8_t kSimd = 1; // Not the target SIMD, so no wave stitching is involved

class BitStreamBuilder
{
public:
    void writeBits(uint64_t value, size_t count)
    {
        for (size_t i = 0; i < count; ++i)
        {
            const size_t byte_idx = bit_pos_ / 8;
            const size_t bit_idx = bit_pos_ % 8;
            if (byte_idx >= data_.size()) data_.resize(byte_idx + 1, 0);
            if ((value >> i) & 1u) data_[byte_idx] |= static_cast<uint8_t>(1u << bit_idx);
            ++bit_pos_;
        }
    }

    std::vector<uint8_t> finish(size_t trailing_zero_bytes)
    {
        while ((bit_pos_ % 8) != 0) writeBits(0, 1);
        data_.resize(data_.size() + trailing_zero_bytes, 0);
        return data_;
    }

private:
    std::vector<uint8_t> data_{};
    size_t bit_pos_ = 0;
};

// Builds a gfx12 token stream. All tokens have a zero time delta and time only advances
// through TIME tokens, so now() is the exact timestamp of the next token.
class TraceBuilder
{
public:
    TraceBuilder()
    {
        header_type header{};
        header.header = 0b0010001;
        header.version = 4;
        bits.writeBits(header.raw, 64);
    }

    int64_t now() const { return time; }

    void advance(int64_t cycles)
    {
        // TIME token: 4-bit header, 4-bit delta encoded as (cycles - 4)
        while (cycles > 0)
        {
            int64_t delta = cycles > 19 ? 16 : cycles;
            EXPECT_GE(delta, 4);
            bits.writeBits(0b1000 | ((delta - 4) << 4), 8);
            time += delta;
            cycles -= delta;
        }
    }

    void reg_init(uint8_t me, uint8_t pipe)
    {
        gfx10::reg_init_type reg{};
        reg.header = 0b1110001;
        reg.me = me;
        reg.pipe = pipe;
        reg.type = 2;
        reg.data = 1;
        bits.writeBits(reg.raw, 64);
    }

    void dispatch_end(uint8_t me, uint8_t pipe)
    {
        gfx10::reg_write_type reg{};
        reg.header = 0b1001;
        reg.me = me;
        reg.pipe = pipe;
        reg.CS = 1;
        reg.regaddr = COMPUTE_DISPATCH_END;
        bits.writeBits(reg.raw, 64);
    }

    void wave_start(uint8_t me, uint8_t pipe, uint8_t wid)
    {
        gfx12::wstart_type start{};
        start.header = 0b1100;
        start.simd = kSimd;
        start.wid = wid;
        start.me = me;
        start.pipe = pipe;
        bits.writeBits(start.raw, 32);
    }

    void wave_end(uint8_t wid)
    {
        gfx12::wend_type end{};
        end.header = 0b10101;
        end.simd = kSimd;
        end.wid = wid;
        bits.writeBits(end.raw, 20);
    }

    std::vector<uint8_t> finish() { return bits.finish(32); }

private:
    BitStreamBuilder bits{};
    int64_t time = 0;
};

struct HandleGuard
{
    rocprof_trace_decoder_handle_t value{};
    ~HandleGuard()
    {
        if (value.handle != 0) rocprof_trace_decoder_destroy_handle(value);
    }
};

struct Records
{
    enum Kind
    {
        OCCUPANCY,
        DISPATCH_END
    };

    std::vector<rocprofiler_thread_trace_decoder_dispatch_t> dispatches{};
    std::vector<rocprofiler_thread_trace_decoder_occupancy_t> occupancy{};
    std::vector<rocprofiler_thread_trace_decoder_event_t> ends{};
    std::vector<std::pair<Kind, size_t>> order{}; // Delivery order of occupancy and DISPATCH_END records

    std::vector<uint64_t> end_ids() const
    {
        std::vector<uint64_t> ids{};
        for (auto& end : ends) ids.push_back(end.payload.raw);
        return ids;
    }

    // Occupancy wave-end record of the matched wave in slot wid ending at time
    const rocprofiler_thread_trace_decoder_occupancy_t* wave_end(uint8_t wid, int64_t time) const
    {
        for (auto& occ : occupancy)
            if (!occ.start && occ.wave_id == wid && occ.time == static_cast<uint64_t>(time)) return &occ;
        return nullptr;
    }

    const rocprofiler_thread_trace_decoder_occupancy_t* wave_start(uint8_t wid, int64_t time) const
    {
        for (auto& occ : occupancy)
            if (occ.start && occ.wave_id == wid && occ.time == static_cast<uint64_t>(time)) return &occ;
        return nullptr;
    }
};

rocprofiler_thread_trace_decoder_status_t
collect(rocprofiler_thread_trace_decoder_record_type_t type, void* data, uint64_t size, void* userdata)
{
    auto* records = static_cast<Records*>(userdata);
    switch (type)
    {
        case ROCPROFILER_THREAD_TRACE_DECODER_RECORD_DISPATCH:
        {
            auto* dispatches = static_cast<rocprofiler_thread_trace_decoder_dispatch_t*>(data);
            records->dispatches.insert(records->dispatches.end(), dispatches, dispatches + size);
            break;
        }
        case ROCPROFILER_THREAD_TRACE_DECODER_RECORD_OCCUPANCY:
        {
            auto* occupancy = static_cast<rocprofiler_thread_trace_decoder_occupancy_t*>(data);
            for (uint64_t i = 0; i < size; i++)
            {
                records->order.emplace_back(Records::OCCUPANCY, records->occupancy.size());
                records->occupancy.push_back(occupancy[i]);
            }
            break;
        }
        case ROCPROFILER_THREAD_TRACE_DECODER_RECORD_EVENT:
        {
            auto* events = static_cast<rocprofiler_thread_trace_decoder_event_t*>(data);
            for (uint64_t i = 0; i < size; i++)
            {
                if (events[i].type != ROCPROF_TRACE_DECODER_EVENT_DISPATCH_END) continue;
                records->order.emplace_back(Records::DISPATCH_END, records->ends.size());
                records->ends.push_back(events[i]);
            }
            break;
        }
        default: break;
    }
    return ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS;
}

rocprofiler_thread_trace_decoder_status_t
nop_isa(char*, uint64_t* memory_size, uint64_t* size, rocprofiler_thread_trace_decoder_pc_t, void*)
{
    *memory_size = 4;
    *size = 0;
    return ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS;
}

Records decode(TraceBuilder& builder)
{
    auto trace = builder.finish();
    Records records{};

    HandleGuard handle;
    EXPECT_EQ(rocprof_trace_decoder_create_handle(&handle.value), ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS);
    EXPECT_EQ(
        rocprof_trace_decoder_set_isa_callback(handle.value, nop_isa, nullptr),
        ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS
    );
    EXPECT_EQ(
        rocprof_trace_decoder_parse(handle.value, trace.data(), trace.size(), collect, &records),
        ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS
    );
    return records;
}

void expect_end(
    const rocprofiler_thread_trace_decoder_event_t& end, uint64_t dispatch_id, int64_t time, uint8_t me, uint8_t pipe
)
{
    EXPECT_EQ(end.payload.raw, dispatch_id);
    EXPECT_EQ(end.time, time);
    EXPECT_EQ(end.me_id, me);
    EXPECT_EQ(end.pipe_id, pipe);
    EXPECT_TRUE(end.flags & ROCPROF_TRACE_DECODER_EVENT_FLAGS_PER_PIPE);
}
} // namespace

TEST(DispatchEndTest, EndsAtLastWaveEndAfterDispatchEnd)
{
    TraceBuilder trace;
    trace.advance(10);
    trace.reg_init(1, 2);
    trace.advance(10);
    trace.wave_start(1, 2, 0);
    trace.wave_start(1, 2, 1);
    trace.advance(10);
    trace.dispatch_end(1, 2);
    trace.advance(10);
    trace.wave_end(0);
    trace.advance(10);
    int64_t last_end = trace.now();
    trace.wave_end(1);
    trace.advance(10);

    auto records = decode(trace);

    ASSERT_EQ(records.dispatches.size(), 1u);
    EXPECT_EQ(records.dispatches[0].dispatch_id, 1u);
    ASSERT_EQ(records.ends.size(), 1u);
    expect_end(records.ends[0], 1, last_end, 1, 2);
}

TEST(DispatchEndTest, OccupancyRecordsCarryDispatchId)
{
    TraceBuilder trace;
    trace.advance(10);
    trace.reg_init(0, 0);
    trace.advance(10);
    int64_t start1 = trace.now();
    trace.wave_start(0, 0, 0);
    trace.dispatch_end(0, 0);
    trace.advance(10);
    int64_t end1 = trace.now();
    trace.wave_end(0);
    trace.advance(10);
    trace.reg_init(0, 0);
    trace.advance(10);
    int64_t start2 = trace.now();
    trace.wave_start(0, 0, 0);
    trace.dispatch_end(0, 0);
    trace.advance(10);
    int64_t end2 = trace.now();
    trace.wave_end(0);
    trace.advance(10);

    auto records = decode(trace);

    ASSERT_EQ(records.dispatches.size(), 2u);
    EXPECT_EQ(records.dispatches[0].dispatch_id, 1u);
    EXPECT_EQ(records.dispatches[1].dispatch_id, 2u);

    auto* s1 = records.wave_start(0, start1);
    auto* e1 = records.wave_end(0, end1);
    auto* s2 = records.wave_start(0, start2);
    auto* e2 = records.wave_end(0, end2);
    ASSERT_TRUE(s1 && e1 && s2 && e2);
    EXPECT_EQ(s1->dispatch_id, 1u);
    EXPECT_EQ(e1->dispatch_id, 1u);
    EXPECT_EQ(s2->dispatch_id, 2u);
    EXPECT_EQ(e2->dispatch_id, 2u);

    EXPECT_EQ(records.end_ids(), (std::vector<uint64_t>{1, 2}));
}

TEST(DispatchEndTest, EndDeliveredAfterLastWaveEndRecord)
{
    TraceBuilder trace;
    trace.advance(10);
    trace.reg_init(0, 0);
    trace.wave_start(0, 0, 0);
    trace.dispatch_end(0, 0);
    trace.advance(10);
    int64_t last_end = trace.now();
    trace.wave_end(0);
    trace.advance(10);

    auto records = decode(trace);

    ASSERT_EQ(records.ends.size(), 1u);
    auto* wave_end = records.wave_end(0, last_end);
    ASSERT_NE(wave_end, nullptr);
    size_t wave_end_idx = wave_end - records.occupancy.data();

    auto pos = [&](Records::Kind kind, size_t idx)
    { return std::find(records.order.begin(), records.order.end(), std::make_pair(kind, idx)) - records.order.begin(); };
    EXPECT_LT(pos(Records::OCCUPANCY, wave_end_idx), pos(Records::DISPATCH_END, 0));
}

TEST(DispatchEndTest, AllWavesEndBeforeDispatchEnd)
{
    TraceBuilder trace;
    trace.advance(10);
    trace.reg_init(0, 1);
    trace.wave_start(0, 1, 0);
    trace.wave_start(0, 1, 1);
    trace.advance(10);
    trace.wave_end(0);
    trace.wave_end(1);
    trace.advance(10);
    int64_t dispatch_end = trace.now();
    trace.dispatch_end(0, 1);
    trace.advance(10);

    auto records = decode(trace);

    ASSERT_EQ(records.ends.size(), 1u);
    expect_end(records.ends[0], 1, dispatch_end, 0, 1);
}

TEST(DispatchEndTest, ZeroWaveDispatchEndsAtDispatchEnd)
{
    TraceBuilder trace;
    trace.advance(10);
    trace.reg_init(1, 3);
    trace.advance(10);
    int64_t dispatch_end = trace.now();
    trace.dispatch_end(1, 3);
    trace.advance(10);

    auto records = decode(trace);

    ASSERT_EQ(records.ends.size(), 1u);
    expect_end(records.ends[0], 1, dispatch_end, 1, 3);
}

TEST(DispatchEndTest, WavesDrainingBeforeDispatchEndDoNotDuplicateEnd)
{
    TraceBuilder trace;
    trace.advance(10);
    trace.reg_init(0, 0);
    trace.wave_start(0, 0, 0);
    trace.dispatch_end(0, 0);
    trace.advance(10);
    trace.wave_end(0); // END(1)
    trace.advance(10);
    trace.reg_init(0, 0);
    trace.wave_start(0, 0, 0);
    trace.advance(10);
    trace.wave_end(0); // Count drops to 0 before DISPATCH_END(2): must not resend END(1)
    trace.advance(10);
    trace.wave_start(0, 0, 1);
    trace.advance(10);
    trace.dispatch_end(0, 0);
    trace.advance(10);
    int64_t last_end = trace.now();
    trace.wave_end(1);
    trace.advance(10);

    auto records = decode(trace);

    ASSERT_EQ(records.end_ids(), (std::vector<uint64_t>{1, 2}));
    expect_end(records.ends[1], 2, last_end, 0, 0);
}

TEST(DispatchEndTest, RegInitWhileWavesInFlight)
{
    TraceBuilder trace;
    trace.advance(10);
    trace.reg_init(0, 0);
    trace.wave_start(0, 0, 0);
    trace.dispatch_end(0, 0);
    trace.advance(10);
    trace.reg_init(0, 0); // Next dispatch is initiated before the waves of dispatch 1 ended
    trace.advance(10);
    int64_t end1 = trace.now();
    trace.wave_end(0);
    trace.advance(10);
    trace.wave_start(0, 0, 1);
    trace.dispatch_end(0, 0);
    trace.advance(10);
    int64_t end2 = trace.now();
    trace.wave_end(1);
    trace.advance(10);

    auto records = decode(trace);

    ASSERT_EQ(records.ends.size(), 2u);
    expect_end(records.ends[0], 1, end1, 0, 0);
    expect_end(records.ends[1], 2, end2, 0, 0);
}

TEST(DispatchEndTest, EmptyDispatchWhileWavesInFlight)
{
    TraceBuilder trace;
    trace.advance(10);
    trace.reg_init(0, 0);
    trace.wave_start(0, 0, 0);
    trace.dispatch_end(0, 0);
    trace.advance(10);
    trace.reg_init(0, 0); // Dispatch 2 launches no waves
    trace.dispatch_end(0, 0);
    trace.advance(10);
    int64_t end1 = trace.now();
    trace.wave_end(0);
    trace.advance(10);

    auto records = decode(trace);

    // END(1) keeps its own ID and time. Known limitation: the empty dispatch 2 gets no END.
    ASSERT_EQ(records.ends.size(), 1u);
    expect_end(records.ends[0], 1, end1, 0, 0);
}

TEST(DispatchEndTest, PipesAndMesAreIndependent)
{
    TraceBuilder trace;
    trace.advance(10);
    trace.reg_init(0, 1);
    trace.reg_init(1, 1);
    trace.wave_start(0, 1, 0);
    trace.wave_start(1, 1, 1);
    trace.dispatch_end(0, 1);
    trace.dispatch_end(1, 1);
    trace.advance(10);
    int64_t end_pipe2 = trace.now();
    trace.reg_init(0, 2); // Zero-wave dispatch on ME0/PIPE2 while ME0/PIPE1 has a running wave
    trace.dispatch_end(0, 2);
    trace.advance(10);
    int64_t end_me1 = trace.now();
    trace.wave_end(1); // ME0/PIPE1 still has a running wave
    trace.advance(10);
    int64_t end_me0 = trace.now();
    trace.wave_end(0);
    trace.advance(10);

    auto records = decode(trace);

    ASSERT_EQ(records.dispatches.size(), 3u);
    for (auto& dispatch : records.dispatches) EXPECT_EQ(dispatch.dispatch_id, 1u);

    ASSERT_EQ(records.ends.size(), 3u);
    expect_end(records.ends[0], 1, end_pipe2, 0, 2);
    expect_end(records.ends[1], 1, end_me1, 1, 1);
    expect_end(records.ends[2], 1, end_me0, 0, 1);
}

TEST(DispatchEndTest, TraceStartsMidDispatch)
{
    TraceBuilder trace;
    trace.advance(10);
    int64_t unknown_start = trace.now();
    trace.wave_start(0, 0, 0); // Launched before the first REG_INIT: dispatch_id unknown
    trace.dispatch_end(0, 0);  // No matching REG_INIT, ignored
    trace.advance(10);
    int64_t unknown_end = trace.now();
    trace.wave_end(0);
    trace.advance(10);
    trace.reg_init(0, 0);
    trace.wave_start(0, 0, 1);
    trace.dispatch_end(0, 0);
    trace.advance(10);
    int64_t end1 = trace.now();
    trace.wave_end(1);
    trace.advance(10);

    auto records = decode(trace);

    auto* s0 = records.wave_start(0, unknown_start);
    auto* e0 = records.wave_end(0, unknown_end);
    ASSERT_TRUE(s0 && e0);
    EXPECT_EQ(s0->dispatch_id, 0u);
    EXPECT_EQ(e0->dispatch_id, 0u);

    ASSERT_EQ(records.dispatches.size(), 1u);
    EXPECT_EQ(records.dispatches[0].dispatch_id, 1u);
    ASSERT_EQ(records.ends.size(), 1u);
    expect_end(records.ends[0], 1, end1, 0, 0);
}

TEST(DispatchEndTest, UnknownWavesDoNotBlockLaterEnds)
{
    TraceBuilder trace;
    trace.advance(10);
    trace.wave_start(0, 0, 0); // Unknown dispatch, still running at the first REG_INIT
    trace.advance(10);
    trace.reg_init(0, 0);
    trace.advance(10);
    trace.wave_end(0);
    trace.advance(10);
    trace.wave_start(0, 0, 1);
    trace.dispatch_end(0, 0);
    trace.advance(10);
    int64_t end1 = trace.now();
    trace.wave_end(1);
    trace.advance(10);

    auto records = decode(trace);

    ASSERT_EQ(records.ends.size(), 1u);
    expect_end(records.ends[0], 1, end1, 0, 0);
}

TEST(DispatchEndTest, UnmatchedWaveEndDoesNotEndDispatchEarly)
{
    TraceBuilder trace;
    trace.advance(10);
    trace.reg_init(0, 0);
    trace.wave_start(0, 0, 0);
    trace.wave_start(0, 0, 1);
    trace.dispatch_end(0, 0);
    trace.advance(10);
    trace.wave_end(5); // Wave started before the trace
    trace.advance(10);
    trace.wave_end(0);
    trace.advance(10);
    int64_t last_end = trace.now();
    trace.wave_end(1);
    trace.advance(10);

    auto records = decode(trace);

    ASSERT_EQ(records.ends.size(), 1u);
    expect_end(records.ends[0], 1, last_end, 0, 0);
}

TEST(DispatchEndTest, DispatchEndWithoutRegInitIsIgnored)
{
    TraceBuilder trace;
    trace.advance(10);
    trace.dispatch_end(0, 2);
    trace.dispatch_end(0, 2);
    trace.advance(10);
    trace.reg_init(0, 2);
    trace.advance(10);
    int64_t dispatch_end = trace.now();
    trace.dispatch_end(0, 2);
    trace.advance(10);

    auto records = decode(trace);

    ASSERT_EQ(records.dispatches.size(), 1u);
    EXPECT_EQ(records.dispatches[0].dispatch_id, 1u);
    ASSERT_EQ(records.ends.size(), 1u);
    expect_end(records.ends[0], 1, dispatch_end, 0, 2);
}

TEST(DispatchEndTest, RegTokenMeAboveOneIsMasked)
{
    // REG and REG_INIT carry a 2-bit ME while WAVE_START carries 1 bit; ME 3 aliases ME 1
    TraceBuilder trace;
    trace.advance(10);
    trace.reg_init(3, 1);
    trace.wave_start(1, 1, 0);
    trace.advance(10);
    trace.wave_end(0);
    trace.advance(10);
    int64_t dispatch_end = trace.now();
    trace.dispatch_end(3, 1);
    trace.advance(10);

    auto records = decode(trace);

    ASSERT_EQ(records.ends.size(), 1u);
    expect_end(records.ends[0], 1, dispatch_end, 1, 1);
}

TEST(DispatchEndTest, DispatchIdSkipsMultiplesOf1024)
{
    // Occupancy records keep the low 10 bits of the dispatch ID and 0 means unknown,
    // so the 1024th dispatch on a pipe gets ID 1025.
    TraceBuilder trace;
    trace.advance(10);
    for (int i = 0; i < 1022; i++)
    {
        trace.reg_init(0, 0);
        trace.dispatch_end(0, 0);
    }

    std::vector<std::pair<int64_t, int64_t>> waves{}; // {start, end} of dispatches 1023, 1025 and 1026
    for (uint8_t wid = 0; wid < 3; wid++)
    {
        trace.reg_init(0, 0);
        int64_t start = trace.now();
        trace.wave_start(0, 0, wid);
        trace.dispatch_end(0, 0);
        trace.advance(10);
        waves.emplace_back(start, trace.now());
        trace.wave_end(wid);
        trace.advance(10);
    }

    auto records = decode(trace);

    ASSERT_EQ(records.dispatches.size(), 1025u);
    EXPECT_EQ(records.dispatches[1021].dispatch_id, 1022u);
    EXPECT_EQ(records.dispatches[1022].dispatch_id, 1023u);
    EXPECT_EQ(records.dispatches[1023].dispatch_id, 1025u);
    EXPECT_EQ(records.dispatches[1024].dispatch_id, 1026u);
    for (auto& dispatch : records.dispatches) EXPECT_NE(dispatch.dispatch_id & 0x3FF, 0u);

    // The dispatch-end counter skips the same IDs, so ENDs still match their dispatches
    ASSERT_EQ(records.ends.size(), 1025u);
    expect_end(records.ends[1022], 1023, waves[0].second, 0, 0);
    expect_end(records.ends[1023], 1025, waves[1].second, 0, 0);
    expect_end(records.ends[1024], 1026, waves[2].second, 0, 0);

    const uint64_t expected[] = {1023, 1, 2};
    for (uint8_t wid = 0; wid < 3; wid++)
    {
        auto* s = records.wave_start(wid, waves[wid].first);
        auto* e = records.wave_end(wid, waves[wid].second);
        ASSERT_TRUE(s && e);
        EXPECT_EQ(s->dispatch_id, expected[wid]);
        EXPECT_EQ(e->dispatch_id, expected[wid]);
    }
}
