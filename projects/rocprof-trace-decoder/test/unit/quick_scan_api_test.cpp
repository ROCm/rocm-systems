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

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <vector>

#include "gfx10/gfx10token.h"
#include "gfx12/gfx12token.h"
#include "gfx9/gfx9token.h"
#include "mi400/mi400token.h"
#include "quick_scan_export.hpp"
#include "rocprof_trace_decoder/rocprof_trace_decoder.h"
#include "rocprof_trace_decoder/trace_decoder_instrument.h"
#include "trace_parser.hpp"

namespace
{
constexpr uint8_t kMe = 1;
constexpr uint8_t kPipe = 2;
constexpr uint8_t kSimd = 3;
constexpr uint8_t kWave = 5;
constexpr uint8_t kWorkgroup = 17;
constexpr uint8_t kCluster = 7;
constexpr uint8_t kTargetSa = 1;
constexpr uint8_t kTargetWgp = 2;
constexpr uint8_t kTargetCu = (kTargetSa << ROCPROFILER_TRACE_DECODER_CU_SA_SHIFT) + kTargetWgp;
constexpr uint64_t kPgmRegValue = 0x12345678ull;
constexpr uint64_t kPcBitmask = (uint64_t{1} << 48) - 1;
constexpr uint64_t kCodeObjectBase = ((kPgmRegValue << 8) & kPcBitmask) - 0x80;
constexpr uint64_t kCodeObjectSize = 0x400;
constexpr uint64_t kCodeObjectId = 0x102030405060708ull;
constexpr uint64_t kDispatchPkt = 0xabcddcba12344321ull;
constexpr uint32_t kRsrc1 =
    7u | (1u << 10) | (1u << 11) | (1u << 14);           // vgprs=64 + scalar/vector invalidate + ctx restore
constexpr uint32_t kRsrc2 = 1u | (9u << 1) | (3u << 15); // scratch + user_sgprs=9 + lds=1536B

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

    size_t byteSize() const { return (bit_pos_ + 7) / 8; }

    std::vector<uint8_t> finish(size_t trailing_zero_bytes = 0)
    {
        while ((bit_pos_ % 8) != 0) writeBits(0, 1);
        data_.resize(data_.size() + trailing_zero_bytes, 0);
        return data_;
    }

private:
    std::vector<uint8_t> data_{};
    size_t bit_pos_ = 0;
};

struct HandleGuard
{
    rocprof_trace_decoder_handle_t value{};
    ~HandleGuard()
    {
        if (value.handle != 0) rocprof_trace_decoder_destroy_handle(value);
    }
};

struct RecordSink
{
    std::vector<rocprofiler_thread_trace_decoder_event_t> events{};
    std::vector<rocprofiler_thread_trace_decoder_dispatch_t> dispatches{};
    std::vector<rocprofiler_thread_trace_decoder_occupancy_t> occupancy{};
    std::vector<rocprofiler_thread_trace_decoder_wave_t> waves{};
};

rocprofiler_thread_trace_decoder_status_t collect_records(
    rocprofiler_thread_trace_decoder_record_type_t record_type_id,
    void* trace_events,
    uint64_t trace_size,
    void* userdata
)
{
    auto* sink = static_cast<RecordSink*>(userdata);

    switch (record_type_id)
    {
        case ROCPROFILER_THREAD_TRACE_DECODER_RECORD_EVENT:
        {
            auto* events = static_cast<rocprofiler_thread_trace_decoder_event_t*>(trace_events);
            sink->events.insert(sink->events.end(), events, events + trace_size);
            break;
        }
        case ROCPROFILER_THREAD_TRACE_DECODER_RECORD_DISPATCH:
        {
            auto* dispatches = static_cast<rocprofiler_thread_trace_decoder_dispatch_t*>(trace_events);
            sink->dispatches.insert(sink->dispatches.end(), dispatches, dispatches + trace_size);
            break;
        }
        case ROCPROFILER_THREAD_TRACE_DECODER_RECORD_OCCUPANCY:
        {
            auto* occupancy = static_cast<rocprofiler_thread_trace_decoder_occupancy_t*>(trace_events);
            sink->occupancy.insert(sink->occupancy.end(), occupancy, occupancy + trace_size);
            break;
        }
        case ROCPROFILER_THREAD_TRACE_DECODER_RECORD_WAVE:
        {
            auto* waves = static_cast<rocprofiler_thread_trace_decoder_wave_t*>(trace_events);
            sink->waves.insert(sink->waves.end(), waves, waves + trace_size);
            break;
        }
        default: break;
    }

    return ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS;
}

rocprofiler_thread_trace_decoder_status_t
reject_records(rocprofiler_thread_trace_decoder_record_type_t, void*, uint64_t, void*)
{
    return ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR;
}

rocprofiler_thread_trace_decoder_status_t
nop_isa(char*, uint64_t* memory_size, uint64_t* size, rocprofiler_thread_trace_decoder_pc_t, void*)
{
    *memory_size = 4;
    *size = 0;
    return ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS;
}

void append_gfx12_header(BitStreamBuilder& builder)
{
    header_type header{};
    header.header = 0b0010001;
    header.version = 4;
    builder.writeBits(header.raw, 64);
}

void append_mi400_header(BitStreamBuilder& builder)
{
    mi400::header_type header{};
    header.header = 0b0010001;
    header.version = 5;
    header.DSA = kTargetSa;
    header.DWGP = kTargetWgp;
    header.DSIMD = kSimd;
    builder.writeBits(header.raw, 64);
}

void append_little_endian(std::vector<uint8_t>& data, uint64_t bits, size_t bytes)
{
    for (size_t i = 0; i < bytes; ++i) data.push_back(static_cast<uint8_t>((bits >> (i * 8)) & 0xFFu));
}

void append_gfx9_header(std::vector<uint8_t>& data)
{
    rocprof_trace_decoder_gfx9_header_t header{};
    header.legacy_version = 0;
    header.gfx9_version2 = 5;
    header.DCU = 0;
    append_little_endian(data, header.raw, sizeof(header.raw));
}

void append_reg(BitStreamBuilder& builder, uint16_t regaddr, uint32_t regdata)
{
    gfx10::reg_write_type reg{};
    reg.header = 0b1001;
    reg.pipe = kPipe;
    reg.me = kMe;
    reg.CS = 1;
    reg.regaddr = regaddr;
    reg.regdata = regdata;
    builder.writeBits(reg.raw, 64);
}

void append_rdna_userdata2(BitStreamBuilder& builder, uint32_t regdata)
{
    gfx10::reg_write_type reg{};
    reg.header = 0b1001;
    reg.regaddr = 0xC342;
    reg.regdata = regdata;
    builder.writeBits(reg.raw, 64);
}

void append_rdna_codeobj_field(
    BitStreamBuilder& builder, rocprof_trace_decoder_codeobj_marker_type_t marker_type, uint32_t payload
)
{
    rocprof_trace_decoder_packet_header_t header{};
    header.opcode = ROCPROF_TRACE_DECODER_PACKET_OPCODE_CODEOBJ;
    header.type = marker_type;
    append_rdna_userdata2(builder, header.u32All);
    append_rdna_userdata2(builder, payload);
}

void append_rdna_codeobj_load(BitStreamBuilder& builder)
{
    rocprof_trace_decoder_instrument_enable_t enable{};
    enable.char1 = '\0';
    enable.char2 = 'R';
    enable.char3 = 'O';
    enable.char4 = 'C';
    append_rdna_userdata2(builder, enable.u32All);

    append_rdna_codeobj_field(
        builder, ROCPROF_TRACE_DECODER_CODEOBJ_MARKER_TYPE_ID_LO, static_cast<uint32_t>(kCodeObjectId)
    );
    append_rdna_codeobj_field(builder, ROCPROF_TRACE_DECODER_CODEOBJ_MARKER_TYPE_ID_HI, kCodeObjectId >> 32);
    append_rdna_codeobj_field(builder, ROCPROF_TRACE_DECODER_CODEOBJ_MARKER_TYPE_SIZE_LO, kCodeObjectSize);
    append_rdna_codeobj_field(builder, ROCPROF_TRACE_DECODER_CODEOBJ_MARKER_TYPE_SIZE_HI, kCodeObjectSize >> 32);
    append_rdna_codeobj_field(
        builder, ROCPROF_TRACE_DECODER_CODEOBJ_MARKER_TYPE_ADDR_LO, static_cast<uint32_t>(kCodeObjectBase)
    );
    append_rdna_codeobj_field(builder, ROCPROF_TRACE_DECODER_CODEOBJ_MARKER_TYPE_ADDR_HI, kCodeObjectBase >> 32);

    rocprof_trace_decoder_codeobj_marker_tail_t tail{};
    tail.bFromStart = 1;
    append_rdna_codeobj_field(builder, ROCPROF_TRACE_DECODER_CODEOBJ_MARKER_TYPE_TAIL, tail.raw);
}

uint64_t encode_gfx9_reg_cs(uint8_t me, uint8_t pipe, uint8_t regaddr, uint32_t regdata)
{
    const uint64_t enc_me = (1u - (me & 1u)) & 1u;
    return static_cast<uint64_t>(gfx9::TOKEN_REG_CS) | (static_cast<uint64_t>(pipe & 0x3) << 5) | (enc_me << 7) |
           (static_cast<uint64_t>(regaddr & 0x7F) << 9) | (static_cast<uint64_t>(regdata) << 16);
}

uint64_t encode_gfx9_reg(uint8_t me, uint8_t pipe, uint16_t regaddr, uint32_t regdata)
{
    const uint64_t enc_me = (1u - (me & 1u)) & 1u;
    return static_cast<uint64_t>(gfx9::TOKEN_REG) | (static_cast<uint64_t>(pipe & 0x3) << 5) | (enc_me << 7) |
           (uint64_t{1} << 15) | (uint64_t{gfx9::Reg::REG_TYPE_USERDATA} << 10) |
           (static_cast<uint64_t>(regaddr) << 16) | (static_cast<uint64_t>(regdata) << 32);
}

void append_gfx9_reg_cs(std::vector<uint8_t>& data, uint8_t regaddr, uint32_t regdata)
{
    append_little_endian(data, encode_gfx9_reg_cs(kMe, kPipe, regaddr, regdata), 6);
}

void append_gfx9_userdata2(std::vector<uint8_t>& data, uint32_t regdata)
{
    append_little_endian(data, encode_gfx9_reg(0, 0, 0xC342, regdata), 8);
}

void append_gfx9_codeobj_field(
    std::vector<uint8_t>& data, rocprof_trace_decoder_codeobj_marker_type_t marker_type, uint32_t payload
)
{
    rocprof_trace_decoder_packet_header_t header{};
    header.opcode = ROCPROF_TRACE_DECODER_PACKET_OPCODE_CODEOBJ;
    header.type = marker_type;
    header.data20 = 0;
    append_gfx9_userdata2(data, header.u32All);
    append_gfx9_userdata2(data, payload);
}

void append_gfx9_codeobj_load(std::vector<uint8_t>& data)
{
    rocprof_trace_decoder_instrument_enable_t enable{};
    enable.char1 = '\0';
    enable.char2 = 'R';
    enable.char3 = 'O';
    enable.char4 = 'C';
    append_gfx9_userdata2(data, enable.u32All);

    append_gfx9_codeobj_field(
        data, ROCPROF_TRACE_DECODER_CODEOBJ_MARKER_TYPE_ID_LO, static_cast<uint32_t>(kCodeObjectId)
    );
    append_gfx9_codeobj_field(data, ROCPROF_TRACE_DECODER_CODEOBJ_MARKER_TYPE_ID_HI, kCodeObjectId >> 32);
    append_gfx9_codeobj_field(data, ROCPROF_TRACE_DECODER_CODEOBJ_MARKER_TYPE_SIZE_LO, kCodeObjectSize);
    append_gfx9_codeobj_field(data, ROCPROF_TRACE_DECODER_CODEOBJ_MARKER_TYPE_SIZE_HI, kCodeObjectSize >> 32);
    append_gfx9_codeobj_field(
        data, ROCPROF_TRACE_DECODER_CODEOBJ_MARKER_TYPE_ADDR_LO, static_cast<uint32_t>(kCodeObjectBase)
    );
    append_gfx9_codeobj_field(data, ROCPROF_TRACE_DECODER_CODEOBJ_MARKER_TYPE_ADDR_HI, kCodeObjectBase >> 32);

    rocprof_trace_decoder_codeobj_marker_tail_t tail{};
    tail.isUnload = 0;
    tail.bFromStart = 1;
    tail.legacy_id = 0;
    append_gfx9_codeobj_field(data, ROCPROF_TRACE_DECODER_CODEOBJ_MARKER_TYPE_TAIL, tail.raw);
}

void append_dispatch_registers(BitStreamBuilder& builder)
{
    append_reg(builder, COMPUTE_PGM_LO, static_cast<uint32_t>(kPgmRegValue));
    append_reg(builder, COMPUTE_PGM_HI, static_cast<uint32_t>(kPgmRegValue >> 32));
    append_reg(builder, COMPUTE_NUM_THREAD_X, 64);
    append_reg(builder, COMPUTE_NUM_THREAD_Y, 8);
    append_reg(builder, COMPUTE_NUM_THREAD_Z, 2);
    append_reg(builder, COMPUTE_PGM_RSRC1, kRsrc1);
    append_reg(builder, COMPUTE_PGM_RSRC2, kRsrc2);
    append_reg(builder, COMPUTE_PGM_RSRC3, 0x55);
    append_reg(builder, COMPUTE_DISPATCH_PKT_LO, static_cast<uint32_t>(kDispatchPkt));
    append_reg(builder, COMPUTE_DISPATCH_PKT_HI, static_cast<uint32_t>(kDispatchPkt >> 32));
}

void append_gfx9_dispatch_registers(std::vector<uint8_t>& data)
{
    append_gfx9_reg_cs(data, COMPUTE_PGM_LO, static_cast<uint32_t>(kPgmRegValue));
    append_gfx9_reg_cs(data, COMPUTE_PGM_HI, static_cast<uint32_t>(kPgmRegValue >> 32));
    append_gfx9_reg_cs(data, COMPUTE_NUM_THREAD_X, 64);
    append_gfx9_reg_cs(data, COMPUTE_NUM_THREAD_Y, 8);
    append_gfx9_reg_cs(data, COMPUTE_NUM_THREAD_Z, 2);
    append_gfx9_reg_cs(data, COMPUTE_PGM_RSRC1, kRsrc1);
    append_gfx9_reg_cs(data, COMPUTE_PGM_RSRC2, kRsrc2);
    append_gfx9_reg_cs(data, COMPUTE_PGM_RSRC3, 0x55);
    append_gfx9_reg_cs(data, COMPUTE_DISPATCH_PKT_LO, static_cast<uint32_t>(kDispatchPkt));
    append_gfx9_reg_cs(data, COMPUTE_DISPATCH_PKT_HI, static_cast<uint32_t>(kDispatchPkt >> 32));
}

void append_event(BitStreamBuilder& builder, uint32_t event_id, bool bop = false)
{
    gfx10::event_type event{};
    event.header = 0b01100001;
    event.bop = bop ? 1 : 0;
    event.pipe = kPipe;
    event.me = kMe;
    event.id = event_id;
    builder.writeBits(event.raw, 24);
}

void append_bop_event(BitStreamBuilder& builder) { append_event(builder, EVENT_BOTTOM_OF_PIPE_WR, true); }

void append_dispatch(BitStreamBuilder& builder)
{
    gfx10::reg_init_type reg{};
    reg.header = 0b1110001;
    reg.pipe = kPipe;
    reg.me = kMe;
    reg.type = 2;
    reg.data = 1;
    builder.writeBits(reg.raw, 64);
}

void append_gfx9_event(std::vector<uint8_t>& data)
{
    append_gfx9_reg_cs(data, COMPUTE_NOWHERE, EVENT_CS_PARTIAL_FLUSH);
}

void append_gfx9_dispatch(std::vector<uint8_t>& data) { append_gfx9_reg_cs(data, COMPUTE_DISPATCH_INITIATOR, 1); }

struct DispatchTrace
{
    std::vector<uint8_t> data{};
    uint64_t cut_offset = 0;
    uint64_t event_offset = 0;
    uint64_t dispatch_offset = 0;
};

DispatchTrace make_dispatch_trace()
{
    BitStreamBuilder builder;
    append_gfx12_header(builder);
    append_dispatch_registers(builder);

    DispatchTrace trace{};
    trace.cut_offset = builder.byteSize();

    append_bop_event(builder);
    trace.event_offset = trace.cut_offset;

    append_dispatch(builder);
    trace.dispatch_offset = trace.event_offset + 3;

    trace.data = builder.finish();
    return trace;
}

DispatchTrace make_rdna_codeobj_dispatch_trace()
{
    BitStreamBuilder builder;
    append_gfx12_header(builder);
    append_rdna_codeobj_load(builder);
    append_dispatch_registers(builder);

    DispatchTrace trace{};
    trace.cut_offset = builder.byteSize();

    append_bop_event(builder);
    trace.event_offset = trace.cut_offset;

    append_dispatch(builder);
    trace.dispatch_offset = trace.event_offset + 3;

    trace.data = builder.finish();
    return trace;
}

DispatchTrace make_mi400_dispatch_trace()
{
    BitStreamBuilder builder;
    append_mi400_header(builder);
    append_dispatch_registers(builder);

    DispatchTrace trace{};
    trace.cut_offset = builder.byteSize();

    append_bop_event(builder);
    trace.event_offset = trace.cut_offset;

    append_dispatch(builder);
    trace.dispatch_offset = trace.event_offset + 3;

    trace.data = builder.finish();
    return trace;
}

DispatchTrace make_gfx9_dispatch_trace()
{
    DispatchTrace trace{};
    append_gfx9_header(trace.data);
    trace.data.resize(trace.data.size() + 64, 0);
    append_gfx9_codeobj_load(trace.data);
    append_gfx9_dispatch_registers(trace.data);
    trace.cut_offset = trace.data.size();

    append_gfx9_event(trace.data);
    trace.event_offset = trace.cut_offset;

    append_gfx9_dispatch(trace.data);
    trace.dispatch_offset = trace.event_offset + 6;

    trace.data.resize(trace.data.size() + 32, 0);
    return trace;
}

std::vector<uint8_t> make_gfx12_event_mapping_trace()
{
    BitStreamBuilder builder;
    append_gfx12_header(builder);
    append_event(builder, EVENT_CS_PARTIAL_FLUSH);
    append_event(builder, EVENT_CACHE_FLUSH);
    append_event(builder, EVENT_CACHE_FLUSH_WR);
    append_event(builder, EVENT_CACHE_FLUSH_INV);
    append_event(builder, EVENT_CACHE_FLUSH_INV_WR);
    append_event(builder, EVENT_TT_FLUSH);
    append_event(builder, EVENT_BOTTOM_OF_PIPE_WR, true);
    return builder.finish();
}

std::vector<uint8_t> make_unknown_header_trace()
{
    std::vector<uint8_t> data(sizeof(uint64_t), 0);
    header_type header{};
    header.header = 0b0010001;
    header.version = 6;
    std::memcpy(data.data(), &header.raw, sizeof(header.raw));
    return data;
}

std::vector<uint8_t> make_mi400_scan_stress_stream()
{
    std::vector<uint8_t> data(64, 0);

    gfx12::wstart_type ext_start{};
    ext_start.header = 0b1100;
    ext_start.isExt = 1;
    ext_start.tm = 1;
    ext_start.sa = kTargetSa;
    ext_start.simd = kSimd;
    ext_start.wgp = kTargetWgp;
    ext_start.wid = kWave;
    ext_start.pipe = kPipe;
    ext_start.me = kMe;
    ext_start.count = 10;
    ext_start.extlds = 0;
    ext_start.wgid = kWorkgroup;
    ext_start.last = 1;
    append_little_endian(data, ext_start.raw, 5);

    data.push_back(0x01);
    data.push_back(0x03);
    data.push_back(0x00);

    gfx10::event_type event{};
    event.header = 0b01100001;
    event.pipe = kPipe;
    event.me = kMe;
    event.id = EVENT_CS_PARTIAL_FLUSH;
    append_little_endian(data, event.raw, 3);

    gfx10::reg_write_type reg{};
    reg.header = 0b1001;
    reg.pipe = kPipe;
    reg.me = kMe;
    reg.CS = 1;
    reg.regaddr = COMPUTE_NUM_THREAD_X;
    reg.regdata = 256;
    append_little_endian(data, reg.raw, 8);

    gfx10::reg_init_type dispatch{};
    dispatch.header = 0b1110001;
    dispatch.pipe = kPipe;
    dispatch.me = kMe;
    dispatch.type = 2;
    dispatch.data = 1;
    append_little_endian(data, dispatch.raw, 8);

    data.resize(144, 0);

    event.id = EVENT_TT_FLUSH;
    append_little_endian(data, event.raw, 3);
    return data;
}

void expect_dispatch_payload(
    const rocprofiler_thread_trace_decoder_dispatch_t& dispatch,
    uint32_t expected_vgprs,
    uint32_t expected_sgprs,
    uint32_t expected_lds_size,
    rocprofiler_thread_trace_decoder_pc_t expected_entry =
        {.address = (kPgmRegValue << 8) & kPcBitmask, .code_object_id = 0}
)
{
    constexpr uint64_t bitmask = (uint64_t{1} << 48) - 1;

    EXPECT_EQ(dispatch.size, sizeof(rocprofiler_thread_trace_decoder_dispatch_t));
    EXPECT_EQ(dispatch.time, 0);
    EXPECT_EQ(dispatch.me_id, kMe);
    EXPECT_EQ(dispatch.pipe_id, kPipe);
    EXPECT_EQ(dispatch.thread_dim_x, 64u);
    EXPECT_EQ(dispatch.thread_dim_y, 8u);
    EXPECT_EQ(dispatch.thread_dim_z, 2u);
    EXPECT_EQ(dispatch.user_sgprs, 9u);
    EXPECT_EQ(dispatch.sgprs, expected_sgprs);
    EXPECT_EQ(dispatch.vgprs, expected_vgprs);
    EXPECT_EQ(dispatch.lds_size, expected_lds_size);
    EXPECT_EQ(dispatch.dispatch_pkt_addr, kDispatchPkt);
    EXPECT_EQ(dispatch.entry_point.code_object_id, expected_entry.code_object_id);
    EXPECT_EQ(dispatch.entry_point.address, expected_entry.address & bitmask);
    EXPECT_EQ(
        dispatch.flags,
        ROCPROFILER_THREAD_TRACE_DECODER_DISPATCH_FLAGS_SCALAR_CACHE_INVALIDATE |
            ROCPROFILER_THREAD_TRACE_DECODER_DISPATCH_FLAGS_VECTOR_CACHE_INVALIDATE |
            ROCPROFILER_THREAD_TRACE_DECODER_DISPATCH_FLAGS_IS_CTX_RESTORE |
            ROCPROFILER_THREAD_TRACE_DECODER_DISPATCH_FLAGS_SCRATCH_ENABLED
    );
}

void expect_gfx12_dispatch_payload(const rocprofiler_thread_trace_decoder_dispatch_t& dispatch)
{
    expect_dispatch_payload(dispatch, 64, 128, 1536);
}

void expect_mi400_dispatch_payload(const rocprofiler_thread_trace_decoder_dispatch_t& dispatch)
{
    expect_dispatch_payload(dispatch, 128, 128, 3072);
}

void expect_gfx9_dispatch_payload(const rocprofiler_thread_trace_decoder_dispatch_t& dispatch)
{
    rocprofiler_thread_trace_decoder_pc_t expected_entry{};
    expected_entry.address = 0x80;
    expected_entry.code_object_id = kCodeObjectId;
    expect_dispatch_payload(dispatch, 64, 16, 1536, expected_entry);
}

std::vector<uint8_t> make_mi400_wave_metadata_trace()
{
    BitStreamBuilder builder;
    append_mi400_header(builder);

    mi400::misc_type misc{};
    misc.header = 0b1010001;
    misc.tm = 1;
    misc.CLF = 1;
    misc.CLID = kCluster;
    builder.writeBits(misc.raw, 24);

    gfx12::wstart_type start{};
    start.header = 0b1100;
    start.isExt = 1;
    start.tm = 1;
    start.sa = kTargetSa;
    start.simd = kSimd;
    start.wgp = kTargetWgp;
    start.wid = kWave;
    start.pipe = kPipe;
    start.me = kMe;
    start.count = 10;
    start.extlds = 0;
    start.wgid = kWorkgroup;
    start.last = 1;
    builder.writeBits(start.raw, 40);

    mi400::inst_type ignored_inst{};
    ignored_inst.header = 0b010;
    ignored_inst.wid = kWave;
    ignored_inst.inst = 100;
    builder.writeBits(ignored_inst.raw, 24);

    mi400::wend_type end{};
    end.header = 0b1000001;
    end.tm = 4;
    end.sa = kTargetSa;
    end.simd = kSimd;
    end.wgp = kTargetWgp;
    end.wid = kWave;
    builder.writeBits(end.raw, 24);

    return builder.finish(16);
}

#if ROCPROF_TRACE_DECODER_QUICK_SCAN_HAS_SIMD
bool avx2_available_for_test()
{
    __builtin_cpu_init();
    return __builtin_cpu_supports("avx2");
}

using Gfx9ScannerForTest = size_t (*)(const uint8_t*, size_t, quick_scan::QuickToken* __restrict__, size_t);

void expect_gfx9_scanner_captures_dispatch_context(Gfx9ScannerForTest scanner)
{
    const auto trace = make_gfx9_dispatch_trace();
    const uint8_t* stream = trace.data.data() + sizeof(uint64_t);
    const size_t stream_size = trace.data.size() - sizeof(uint64_t);

    std::array<quick_scan::QuickToken, 1> scratch{};
    EXPECT_EQ(scanner(nullptr, stream_size, scratch.data(), scratch.size()), 0u);
    EXPECT_EQ(scanner(stream, 0, scratch.data(), scratch.size()), 0u);
    EXPECT_EQ(scanner(stream, stream_size, scratch.data(), 0), 0u);

    std::array<quick_scan::QuickToken, 64> tokens{};
    const size_t n = scanner(stream, stream_size, tokens.data(), tokens.size());
    ASSERT_GT(n, 0u);
    ASSERT_LT(n, tokens.size());

    bool saw_event = false;
    bool saw_dispatch = false;
    bool saw_dispatch_pkt_lo = false;
    bool saw_dispatch_pkt_hi = false;
    uint32_t dispatch_pkt_lo = 0;
    uint32_t dispatch_pkt_hi = 0;

    for (size_t i = 0; i < n; ++i)
    {
        const auto& tok = tokens[i];
        if (tok.type != gfx9::TOKEN_REG_CS && tok.type != gfx9::TOKEN_REG_CS_PRIV) continue;

        gfx9::RegCs reg{tok.contents};
        if (reg.regaddr == COMPUTE_NOWHERE && reg.regdata == EVENT_CS_PARTIAL_FLUSH)
        {
            saw_event = true;
            EXPECT_EQ(tok.offset, trace.event_offset - sizeof(uint64_t));
            EXPECT_EQ(reg.me, kMe);
            EXPECT_EQ(reg.pipe, kPipe);
        }
        else if (reg.regaddr == COMPUTE_DISPATCH_INITIATOR && (reg.regdata & 1) != 0)
        {
            saw_dispatch = true;
            EXPECT_EQ(tok.offset, trace.dispatch_offset - sizeof(uint64_t));
            EXPECT_EQ(reg.me, kMe);
            EXPECT_EQ(reg.pipe, kPipe);
        }
        else if (reg.regaddr == COMPUTE_DISPATCH_PKT_LO)
        {
            saw_dispatch_pkt_lo = true;
            dispatch_pkt_lo = reg.regdata;
        }
        else if (reg.regaddr == COMPUTE_DISPATCH_PKT_HI)
        {
            saw_dispatch_pkt_hi = true;
            dispatch_pkt_hi = reg.regdata;
        }
    }

    EXPECT_TRUE(saw_event);
    EXPECT_TRUE(saw_dispatch);
    ASSERT_TRUE(saw_dispatch_pkt_lo);
    ASSERT_TRUE(saw_dispatch_pkt_hi);
    EXPECT_EQ((uint64_t{dispatch_pkt_hi} << 32) | dispatch_pkt_lo, kDispatchPkt);
}
#endif
} // namespace

TEST(QuickScanApiTest, ReportsDispatchAndEventRecordsFromGfx12Trace)
{
    if (!quick_scan::avx512_available()) GTEST_SKIP() << "quick_scan requires AVX-512";

    const auto trace = make_dispatch_trace();

    HandleGuard handle;
    ASSERT_EQ(rocprof_trace_decoder_create_handle(&handle.value), ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS);

    RecordSink sink;
    ASSERT_EQ(
        rocprof_trace_decoder_quick_scan(handle.value, 0, trace.data.data(), trace.data.size(), collect_records, &sink),
        ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS
    );

    ASSERT_EQ(sink.events.size(), 1u);
    const auto& event = sink.events.front();
    EXPECT_EQ(event.size, sizeof(rocprofiler_thread_trace_decoder_event_t));
    EXPECT_EQ(event.time, 0);
    EXPECT_EQ(event.type, ROCPROF_TRACE_DECODER_EVENT_BOTTOM_OF_PIPE_TS);
    EXPECT_EQ(event.me_id, kMe);
    EXPECT_EQ(event.pipe_id, kPipe);
    EXPECT_EQ(event.flags, ROCPROF_TRACE_DECODER_EVENT_FLAGS_PER_PIPE | ROCPROF_TRACE_DECODER_EVENT_FLAGS_BOP);
    EXPECT_EQ(event.payload.raw, 0u);
    EXPECT_EQ(event.byte_offset, trace.event_offset);

    ASSERT_EQ(sink.dispatches.size(), 1u);
    expect_gfx12_dispatch_payload(sink.dispatches.front());
    EXPECT_EQ(sink.dispatches.front().byte_offset, trace.dispatch_offset);
}

TEST(QuickScanApiTest, ReportsDispatchAndEventRecordsFromMi400Trace)
{
    if (!quick_scan::avx512_available()) GTEST_SKIP() << "quick_scan requires AVX-512";

    const auto trace = make_mi400_dispatch_trace();

    HandleGuard handle;
    ASSERT_EQ(rocprof_trace_decoder_create_handle(&handle.value), ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS);

    RecordSink sink;
    ASSERT_EQ(
        rocprof_trace_decoder_quick_scan(handle.value, 0, trace.data.data(), trace.data.size(), collect_records, &sink),
        ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS
    );

    ASSERT_EQ(sink.events.size(), 1u);
    EXPECT_EQ(sink.events.front().type, ROCPROF_TRACE_DECODER_EVENT_BOTTOM_OF_PIPE_TS);
    EXPECT_EQ(sink.events.front().me_id, kMe);
    EXPECT_EQ(sink.events.front().pipe_id, kPipe);
    EXPECT_EQ(sink.events.front().byte_offset, trace.event_offset);

    ASSERT_EQ(sink.dispatches.size(), 1u);
    expect_mi400_dispatch_payload(sink.dispatches.front());
    EXPECT_EQ(sink.dispatches.front().byte_offset, trace.dispatch_offset);
}

TEST(QuickScanApiTest, ReportsCodeObjectAttributedDispatchAndEventFromGfx9Trace)
{
    if (!quick_scan::avx512_available()) GTEST_SKIP() << "quick_scan requires AVX-512";

    const auto trace = make_gfx9_dispatch_trace();

    HandleGuard handle;
    ASSERT_EQ(rocprof_trace_decoder_create_handle(&handle.value), ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS);

    RecordSink sink;
    ASSERT_EQ(
        rocprof_trace_decoder_quick_scan(handle.value, 0, trace.data.data(), trace.data.size(), collect_records, &sink),
        ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS
    );

    ASSERT_EQ(sink.events.size(), 1u);
    const auto& event = sink.events.front();
    EXPECT_EQ(event.size, sizeof(rocprofiler_thread_trace_decoder_event_t));
    EXPECT_EQ(event.time, 0);
    EXPECT_EQ(event.type, ROCPROF_TRACE_DECODER_EVENT_CS_PARTIAL_FLUSH);
    EXPECT_EQ(event.me_id, kMe);
    EXPECT_EQ(event.pipe_id, kPipe);
    EXPECT_EQ(event.flags, ROCPROF_TRACE_DECODER_EVENT_FLAGS_NONE);
    EXPECT_EQ(event.payload.raw, 0u);
    EXPECT_EQ(event.byte_offset, trace.event_offset);

    ASSERT_EQ(sink.dispatches.size(), 1u);
    expect_gfx9_dispatch_payload(sink.dispatches.front());
    EXPECT_EQ(sink.dispatches.front().byte_offset, trace.dispatch_offset);
}

TEST(QuickScanApiTest, MapsGfx12EventIds)
{
    if (!quick_scan::avx512_available()) GTEST_SKIP() << "quick_scan requires AVX-512";

    const auto trace = make_gfx12_event_mapping_trace();

    HandleGuard handle;
    ASSERT_EQ(rocprof_trace_decoder_create_handle(&handle.value), ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS);

    RecordSink sink;
    ASSERT_EQ(
        rocprof_trace_decoder_quick_scan(handle.value, 0, trace.data(), trace.size(), collect_records, &sink),
        ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS
    );

    ASSERT_EQ(sink.events.size(), 7u);
    EXPECT_EQ(sink.events[0].type, ROCPROF_TRACE_DECODER_EVENT_CS_PARTIAL_FLUSH);
    EXPECT_EQ(sink.events[1].type, ROCPROF_TRACE_DECODER_EVENT_CACHE_FLUSH);
    EXPECT_EQ(sink.events[2].type, ROCPROF_TRACE_DECODER_EVENT_CACHE_FLUSH);
    EXPECT_EQ(sink.events[3].type, ROCPROF_TRACE_DECODER_EVENT_CACHE_FLUSH);
    EXPECT_EQ(sink.events[4].type, ROCPROF_TRACE_DECODER_EVENT_CACHE_FLUSH);
    EXPECT_EQ(sink.events[5].type, ROCPROF_TRACE_DECODER_EVENT_TT_FLUSH);
    EXPECT_EQ(sink.events[6].type, ROCPROF_TRACE_DECODER_EVENT_BOTTOM_OF_PIPE_TS);
    EXPECT_EQ(sink.events[6].flags, ROCPROF_TRACE_DECODER_EVENT_FLAGS_PER_PIPE | ROCPROF_TRACE_DECODER_EVENT_FLAGS_BOP);
}

TEST(QuickScanApiTest, HandlesChunkStateAndApiErrors)
{
    if (!quick_scan::avx512_available()) GTEST_SKIP() << "quick_scan requires AVX-512";

    const auto trace = make_dispatch_trace();

    HandleGuard handle;
    ASSERT_EQ(rocprof_trace_decoder_create_handle(&handle.value), ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS);

    ASSERT_EQ(
        rocprof_trace_decoder_quick_scan(
            handle.value, 0, trace.data.data(), sizeof(uint64_t), collect_records, nullptr
        ),
        ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS
    );

    RecordSink sink;
    ASSERT_EQ(
        rocprof_trace_decoder_quick_scan(
            handle.value,
            1,
            trace.data.data() + sizeof(uint64_t),
            trace.data.size() - sizeof(uint64_t),
            collect_records,
            &sink
        ),
        ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS
    );
    ASSERT_EQ(sink.dispatches.size(), 1u);
    expect_gfx12_dispatch_payload(sink.dispatches.front());

    EXPECT_EQ(
        rocprof_trace_decoder_quick_scan(handle.value, 0, trace.data.data(), trace.data.size(), nullptr, nullptr),
        ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR_INVALID_ARGUMENT
    );
    EXPECT_EQ(
        rocprof_trace_decoder_quick_scan(
            handle.value, 0, trace.data.data(), trace.data.size(), reject_records, nullptr
        ),
        ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR
    );

    rocprof_trace_decoder_handle_t invalid{0};
    EXPECT_EQ(
        rocprof_trace_decoder_quick_scan(invalid, 0, trace.data.data(), trace.data.size(), collect_records, nullptr),
        ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR_INVALID_ARGUMENT
    );
}

TEST(QuickScanApiTest, AcceptsEmptyAndGfx9HeaderOnlyChunks)
{
    if (!quick_scan::avx512_available()) GTEST_SKIP() << "quick_scan requires AVX-512";

    HandleGuard handle;
    ASSERT_EQ(rocprof_trace_decoder_create_handle(&handle.value), ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS);

    EXPECT_EQ(
        rocprof_trace_decoder_quick_scan(handle.value, 0, nullptr, 0, collect_records, nullptr),
        ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS
    );

    std::vector<uint8_t> gfx9_header;
    append_gfx9_header(gfx9_header);
    EXPECT_EQ(
        rocprof_trace_decoder_quick_scan(
            handle.value, 0, gfx9_header.data(), gfx9_header.size(), collect_records, nullptr
        ),
        ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS
    );
}

#if ROCPROF_TRACE_DECODER_QUICK_SCAN_HAS_SIMD
TEST(QuickScanInternalTest, CapturesMi400RareTokensAcrossFastPathExtAndTail)
{
    if (!quick_scan::avx512_available()) GTEST_SKIP() << "quick_scan requires AVX-512";

    const auto stream = make_mi400_scan_stress_stream();

    std::array<quick_scan::QuickToken, 1> scratch{};
    EXPECT_EQ(quick_scan::scan_mi400(nullptr, stream.size(), scratch.data(), scratch.size()), 0u);
    EXPECT_EQ(quick_scan::scan_mi400(stream.data(), 0, scratch.data(), scratch.size()), 0u);
    EXPECT_EQ(quick_scan::scan_mi400(stream.data(), stream.size(), scratch.data(), 0), 0u);

    std::array<quick_scan::QuickToken, 2> limited{};
    ASSERT_EQ(quick_scan::scan_mi400(stream.data(), stream.size(), limited.data(), limited.size()), limited.size());
    EXPECT_EQ(limited[0].type, RdnaType::EVENT);
    EXPECT_EQ(limited[0].offset, 72u);
    EXPECT_EQ(limited[1].type, RdnaType::REG);
    EXPECT_EQ(limited[1].offset, 75u);

    std::array<quick_scan::QuickToken, 8> tokens{};
    ASSERT_EQ(quick_scan::scan_mi400(stream.data(), stream.size(), tokens.data(), tokens.size()), 4u);

    EXPECT_EQ(tokens[0].type, RdnaType::EVENT);
    EXPECT_EQ(tokens[0].offset, 72u);
    gfx10::event_type first_event{.raw = tokens[0].contents};
    EXPECT_EQ(first_event.id, EVENT_CS_PARTIAL_FLUSH);
    EXPECT_EQ(first_event.me, kMe);
    EXPECT_EQ(first_event.pipe, kPipe);

    EXPECT_EQ(tokens[1].type, RdnaType::REG);
    EXPECT_EQ(tokens[1].offset, 75u);
    gfx10::reg_write_type reg{.raw = tokens[1].contents};
    EXPECT_TRUE(reg.CS);
    EXPECT_EQ(reg.regaddr, COMPUTE_NUM_THREAD_X);
    EXPECT_EQ(reg.regdata, 256u);

    EXPECT_EQ(tokens[2].type, RdnaType::REG_INIT);
    EXPECT_EQ(tokens[2].offset, 83u);
    gfx10::reg_init_type dispatch{.raw = tokens[2].contents};
    EXPECT_EQ(dispatch.type, 2u);
    EXPECT_EQ(dispatch.data, 1u);

    EXPECT_EQ(tokens[3].type, RdnaType::EVENT);
    EXPECT_EQ(tokens[3].offset, 144u);
    gfx10::event_type tail_event{.raw = tokens[3].contents};
    EXPECT_EQ(tail_event.id, EVENT_TT_FLUSH);
}

TEST(QuickScanInternalTest, CapturesGfx9RareTokensThroughAvx2TestingEntry)
{
    if (!avx2_available_for_test()) GTEST_SKIP() << "AVX2 scanner requires AVX2";

    expect_gfx9_scanner_captures_dispatch_context(gfx9::quick_scan::scan_gfx9_avx2_for_testing);
}
#endif

TEST(BuildStandaloneApiTest, ReplaysDispatchContextFromCutGfx12Trace)
{
    if (!quick_scan::avx512_available()) GTEST_SKIP() << "quick_scan requires AVX-512";

    const auto trace = make_dispatch_trace();

    HandleGuard builder_handle;
    ASSERT_EQ(
        rocprof_trace_decoder_create_handle(&builder_handle.value), ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS
    );

    std::array<uint8_t, 1> tiny{};
    uint64_t standalone_size = tiny.size();
    ASSERT_EQ(
        rocprof_trace_decoder_build_standalone(
            builder_handle.value,
            0,
            trace.data.data(),
            trace.data.size(),
            trace.cut_offset,
            trace.data.size(),
            tiny.data(),
            &standalone_size
        ),
        ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR_OUT_OF_RESOURCES
    );
    ASSERT_GT(standalone_size, trace.data.size() - trace.cut_offset + sizeof(uint64_t));

    std::vector<uint8_t> standalone(standalone_size);
    ASSERT_EQ(
        rocprof_trace_decoder_build_standalone(
            builder_handle.value,
            0,
            trace.data.data(),
            trace.data.size(),
            trace.cut_offset,
            trace.data.size(),
            standalone.data(),
            &standalone_size
        ),
        ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS
    );
    standalone.resize(standalone_size);

    ASSERT_GE(standalone.size(), sizeof(uint64_t));
    EXPECT_EQ(std::memcmp(standalone.data(), trace.data.data(), sizeof(uint64_t)), 0);

    const size_t slice_size = trace.data.size() - trace.cut_offset;
    ASSERT_GE(standalone.size(), sizeof(uint64_t) + slice_size);
    EXPECT_EQ(
        std::memcmp(
            standalone.data() + standalone.size() - slice_size, trace.data.data() + trace.cut_offset, slice_size
        ),
        0
    );

    HandleGuard scan_handle;
    ASSERT_EQ(rocprof_trace_decoder_create_handle(&scan_handle.value), ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS);

    RecordSink sink;
    ASSERT_EQ(
        rocprof_trace_decoder_quick_scan(
            scan_handle.value, 0, standalone.data(), standalone.size(), collect_records, &sink
        ),
        ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS
    );

    ASSERT_EQ(sink.dispatches.size(), 1u);
    expect_gfx12_dispatch_payload(sink.dispatches.front());
}

TEST(BuildStandaloneApiTest, ReplaysRdnaCodeObjectContextFromCutTrace)
{
    if (!quick_scan::avx512_available()) GTEST_SKIP() << "quick_scan requires AVX-512";

    const auto trace = make_rdna_codeobj_dispatch_trace();

    HandleGuard builder_handle;
    ASSERT_EQ(
        rocprof_trace_decoder_create_handle(&builder_handle.value), ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS
    );

    uint64_t standalone_size = trace.data.size() + 512;
    std::vector<uint8_t> standalone(standalone_size);
    ASSERT_EQ(
        rocprof_trace_decoder_build_standalone(
            builder_handle.value,
            0,
            trace.data.data(),
            trace.data.size(),
            trace.cut_offset,
            trace.data.size(),
            standalone.data(),
            &standalone_size
        ),
        ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS
    );
    standalone.resize(standalone_size);

    HandleGuard scan_handle;
    ASSERT_EQ(rocprof_trace_decoder_create_handle(&scan_handle.value), ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS);

    RecordSink sink;
    ASSERT_EQ(
        rocprof_trace_decoder_quick_scan(
            scan_handle.value, 0, standalone.data(), standalone.size(), collect_records, &sink
        ),
        ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS
    );

    ASSERT_EQ(sink.dispatches.size(), 1u);
    rocprofiler_thread_trace_decoder_pc_t expected_entry{0x80, kCodeObjectId};
    expect_dispatch_payload(sink.dispatches.front(), 64, 128, 1536, expected_entry);
}

TEST(BuildStandaloneApiTest, ReplaysDispatchContextFromCutMi400Trace)
{
    if (!quick_scan::avx512_available()) GTEST_SKIP() << "quick_scan requires AVX-512";

    const auto trace = make_mi400_dispatch_trace();

    HandleGuard builder_handle;
    ASSERT_EQ(
        rocprof_trace_decoder_create_handle(&builder_handle.value), ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS
    );

    uint64_t standalone_size = 0;
    std::array<uint8_t, 1> tiny{};
    ASSERT_EQ(
        rocprof_trace_decoder_build_standalone(
            builder_handle.value,
            0,
            trace.data.data(),
            trace.data.size(),
            trace.cut_offset,
            trace.data.size(),
            tiny.data(),
            &standalone_size
        ),
        ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR_OUT_OF_RESOURCES
    );

    std::vector<uint8_t> standalone(standalone_size);
    ASSERT_EQ(
        rocprof_trace_decoder_build_standalone(
            builder_handle.value,
            0,
            trace.data.data(),
            trace.data.size(),
            trace.cut_offset,
            trace.data.size(),
            standalone.data(),
            &standalone_size
        ),
        ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS
    );
    standalone.resize(standalone_size);

    HandleGuard scan_handle;
    ASSERT_EQ(rocprof_trace_decoder_create_handle(&scan_handle.value), ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS);

    RecordSink sink;
    ASSERT_EQ(
        rocprof_trace_decoder_quick_scan(
            scan_handle.value, 0, standalone.data(), standalone.size(), collect_records, &sink
        ),
        ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS
    );

    ASSERT_EQ(sink.dispatches.size(), 1u);
    expect_mi400_dispatch_payload(sink.dispatches.front());
}

TEST(BuildStandaloneApiTest, ReplaysGfx9CodeObjectAndDispatchContextFromCutTrace)
{
    if (!quick_scan::avx512_available()) GTEST_SKIP() << "quick_scan requires AVX-512";

    const auto trace = make_gfx9_dispatch_trace();

    HandleGuard builder_handle;
    ASSERT_EQ(
        rocprof_trace_decoder_create_handle(&builder_handle.value), ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS
    );

    std::array<uint8_t, 16> tiny{};
    uint64_t standalone_size = tiny.size();
    ASSERT_EQ(
        rocprof_trace_decoder_build_standalone(
            builder_handle.value,
            0,
            trace.data.data(),
            trace.data.size(),
            trace.cut_offset,
            trace.data.size(),
            tiny.data(),
            &standalone_size
        ),
        ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR_OUT_OF_RESOURCES
    );
    ASSERT_GT(standalone_size, trace.data.size() - trace.cut_offset + sizeof(uint64_t));

    std::vector<uint8_t> standalone(standalone_size);
    ASSERT_EQ(
        rocprof_trace_decoder_build_standalone(
            builder_handle.value,
            0,
            trace.data.data(),
            trace.data.size(),
            trace.cut_offset,
            trace.data.size(),
            standalone.data(),
            &standalone_size
        ),
        ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS
    );
    standalone.resize(standalone_size);

    ASSERT_GE(standalone.size(), sizeof(uint64_t));
    EXPECT_EQ(std::memcmp(standalone.data(), trace.data.data(), sizeof(uint64_t)), 0);

    const size_t slice_size = trace.data.size() - trace.cut_offset;
    ASSERT_GE(standalone.size(), sizeof(uint64_t) + slice_size);
    EXPECT_EQ(
        std::memcmp(
            standalone.data() + standalone.size() - slice_size, trace.data.data() + trace.cut_offset, slice_size
        ),
        0
    );

    HandleGuard scan_handle;
    ASSERT_EQ(rocprof_trace_decoder_create_handle(&scan_handle.value), ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS);

    RecordSink sink;
    ASSERT_EQ(
        rocprof_trace_decoder_quick_scan(
            scan_handle.value, 0, standalone.data(), standalone.size(), collect_records, &sink
        ),
        ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS
    );

    ASSERT_EQ(sink.events.size(), 1u);
    EXPECT_EQ(sink.events.front().type, ROCPROF_TRACE_DECODER_EVENT_CS_PARTIAL_FLUSH);

    ASSERT_EQ(sink.dispatches.size(), 1u);
    expect_gfx9_dispatch_payload(sink.dispatches.front());
}

TEST(BuildStandaloneApiTest, DoesNotCopyGfx9ChunkHeaderIntoStandalonePayload)
{
    if (!quick_scan::avx512_available()) GTEST_SKIP() << "quick_scan requires AVX-512";

    const auto trace = make_gfx9_dispatch_trace();

    HandleGuard handle;
    ASSERT_EQ(rocprof_trace_decoder_create_handle(&handle.value), ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS);

    auto build_from = [&](uint64_t offset_begin)
    {
        uint64_t standalone_size = trace.data.size() + 512;
        std::vector<uint8_t> standalone(standalone_size);
        EXPECT_EQ(
            rocprof_trace_decoder_build_standalone(
                handle.value,
                0,
                trace.data.data(),
                trace.data.size(),
                offset_begin,
                trace.data.size(),
                standalone.data(),
                &standalone_size
            ),
            ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS
        );
        standalone.resize(standalone_size);
        return standalone;
    };

    auto from_chunk_start = build_from(0);
    auto from_payload_start = build_from(sizeof(uint64_t));
    EXPECT_EQ(from_chunk_start, from_payload_start);

    RecordSink sink;
    ASSERT_EQ(
        rocprof_trace_decoder_quick_scan(
            handle.value, 0, from_chunk_start.data(), from_chunk_start.size(), collect_records, &sink
        ),
        ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS
    );
    ASSERT_EQ(sink.events.size(), 1u);
    ASSERT_EQ(sink.dispatches.size(), 1u);
    expect_gfx9_dispatch_payload(sink.dispatches.front());
}

TEST(BuildStandaloneApiTest, ReportsValidationAndUnsupportedStatuses)
{
    if (!quick_scan::avx512_available()) GTEST_SKIP() << "quick_scan requires AVX-512";

    const auto trace = make_dispatch_trace();

    HandleGuard handle;
    ASSERT_EQ(rocprof_trace_decoder_create_handle(&handle.value), ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS);

    std::array<uint8_t, 128> out{};
    uint64_t size_out = out.size();
    EXPECT_EQ(
        rocprof_trace_decoder_build_standalone(
            handle.value,
            1,
            trace.data.data(),
            trace.data.size(),
            trace.cut_offset,
            trace.data.size(),
            out.data(),
            &size_out
        ),
        ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR_INVALID_ARGUMENT
    );

    EXPECT_EQ(
        rocprof_trace_decoder_build_standalone(
            handle.value,
            0,
            trace.data.data(),
            trace.data.size(),
            trace.cut_offset,
            trace.data.size(),
            nullptr,
            &size_out
        ),
        ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR_INVALID_ARGUMENT
    );

    auto unknown = make_unknown_header_trace();
    unknown.resize(32, 0);
    size_out = out.size();
    EXPECT_EQ(
        rocprof_trace_decoder_build_standalone(
            handle.value, 0, unknown.data(), unknown.size(), 8, unknown.size(), out.data(), &size_out
        ),
        ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR_NOT_IMPLEMENTED
    );
}

TEST(ParseApiTest, ReportsWaveWorkgroupDispatcherAndClusterMetadata)
{
    const auto trace = make_mi400_wave_metadata_trace();

    HandleGuard handle;
    ASSERT_EQ(rocprof_trace_decoder_create_handle(&handle.value), ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS);
    ASSERT_EQ(
        rocprof_trace_decoder_set_isa_callback(handle.value, nop_isa, nullptr),
        ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS
    );

    RecordSink sink;
    ASSERT_EQ(
        rocprof_trace_decoder_parse(handle.value, trace.data(), trace.size(), collect_records, &sink),
        ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS
    );

    auto start =
        std::find_if(sink.occupancy.begin(), sink.occupancy.end(), [](const auto& occ) { return occ.start != 0; });
    ASSERT_NE(start, sink.occupancy.end());
    EXPECT_EQ(start->cu, kTargetCu);
    EXPECT_EQ(start->simd, kSimd);
    EXPECT_EQ(start->wave_id, kWave);
    EXPECT_EQ(start->me_id, kMe);
    EXPECT_EQ(start->pipe_id, kPipe);
    EXPECT_EQ(start->is_ext, 1u);
    EXPECT_EQ(start->workgroup_id, kWorkgroup);
    EXPECT_EQ(start->cluster_id, kCluster);

    ASSERT_EQ(sink.waves.size(), 1u);
    const auto& wave = sink.waves.front();
    EXPECT_EQ(wave.size, sizeof(rocprofiler_thread_trace_decoder_wave_t));
    EXPECT_EQ(wave.cu, kTargetCu);
    EXPECT_EQ(wave.simd, kSimd);
    EXPECT_EQ(wave.wave_id, kWave);
    EXPECT_EQ(wave.dispatcher, ((kMe & 0x7) << 4) | (kPipe & 0xF));
    EXPECT_EQ(wave.workgroup_id, kWorkgroup);
    EXPECT_EQ(wave.cluster_id, kCluster);
    EXPECT_LT(wave.begin_time, wave.end_time);
}
