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
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <utility>
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

struct CallbackFailure
{
    uint64_t calls = 0;
    uint64_t records = 0;
};

rocprofiler_thread_trace_decoder_status_t reject_and_count_records(
    rocprofiler_thread_trace_decoder_record_type_t, void*, uint64_t count, void* userdata
)
{
    auto& failure = *static_cast<CallbackFailure*>(userdata);
    ++failure.calls;
    failure.records += count;
    return ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR_OUT_OF_RESOURCES;
}

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

TEST(RecordFilterApiTest, ValidatesRequestsAndSupportsReset)
{
    HandleGuard handle;
    ASSERT_EQ(rocprof_trace_decoder_create_handle(&handle.value), ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS);

    EXPECT_EQ(
        rocprof_trace_decoder_set_record_filter(handle.value, nullptr), ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS
    );

    rocprofiler_thread_trace_decoder_record_filter_t empty{
        .size = sizeof(empty), .records = nullptr, .record_count = 0};
    EXPECT_EQ(
        rocprof_trace_decoder_set_record_filter(handle.value, &empty), ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS
    );

    rocprofiler_thread_trace_decoder_record_request_t request{
        .size = sizeof(request),
        .type = ROCPROFILER_THREAD_TRACE_DECODER_RECORD_MARKER,
        .flags = ROCPROFILER_THREAD_TRACE_DECODER_RECORD_REQUEST_FLAGS_IMMEDIATE};
    rocprofiler_thread_trace_decoder_record_filter_t marker_only{
        .size = sizeof(marker_only), .records = &request, .record_count = 1};
    EXPECT_EQ(
        rocprof_trace_decoder_set_record_filter(handle.value, &marker_only),
        ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS
    );

    request.type = ROCPROFILER_THREAD_TRACE_DECODER_RECORD_WAVE;
    EXPECT_EQ(
        rocprof_trace_decoder_set_record_filter(handle.value, &marker_only),
        ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR_INVALID_ARGUMENT
    );

    request.type = ROCPROFILER_THREAD_TRACE_DECODER_RECORD_MARKER;
    rocprofiler_thread_trace_decoder_record_request_t duplicate[] = {request, request};
    rocprofiler_thread_trace_decoder_record_filter_t duplicate_filter{
        .size = sizeof(duplicate_filter), .records = duplicate, .record_count = 2};
    EXPECT_EQ(
        rocprof_trace_decoder_set_record_filter(handle.value, &duplicate_filter),
        ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR_INVALID_ARGUMENT
    );
}

TEST(RecordFilterApiTest, RejectsTruncatedAndOverflowingRequests)
{
    HandleGuard handle;
    ASSERT_EQ(rocprof_trace_decoder_create_handle(&handle.value), ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS);

    using Request = rocprofiler_thread_trace_decoder_record_request_t;
    alignas(Request) uint64_t truncated_size = sizeof(truncated_size);
    rocprofiler_thread_trace_decoder_record_filter_t filter{
        .size = sizeof(filter), .records = reinterpret_cast<const Request*>(&truncated_size), .record_count = 1
    };
    EXPECT_EQ(
        rocprof_trace_decoder_set_record_filter(handle.value, &filter),
        ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR_INVALID_ARGUMENT
    );

    Request request{
        .size = std::numeric_limits<uint64_t>::max(), .type = ROCPROFILER_THREAD_TRACE_DECODER_RECORD_EVENT, .flags = 0
    };
    filter.records = &request;
    EXPECT_EQ(
        rocprof_trace_decoder_set_record_filter(handle.value, &filter),
        ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR_INVALID_ARGUMENT
    );

    request.size = sizeof(request);
    request.type = ROCPROFILER_THREAD_TRACE_DECODER_RECORD_LAST;
    EXPECT_EQ(
        rocprof_trace_decoder_set_record_filter(handle.value, &filter),
        ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR_INVALID_ARGUMENT
    );

    filter.records = nullptr;
    EXPECT_EQ(
        rocprof_trace_decoder_set_record_filter(handle.value, &filter),
        ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR_INVALID_ARGUMENT
    );
    filter.record_count = 0;
    filter.size = sizeof(filter) - 1;
    EXPECT_EQ(
        rocprof_trace_decoder_set_record_filter(handle.value, &filter),
        ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR_INVALID_ARGUMENT
    );
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

std::vector<uint8_t> make_marker_code_object(const std::string& name, const std::string& custom_funcmap = {})
{
    namespace elf = rocprof_trace_decoder::codeobj::elf_inline;
    const char section_names[] = "\0.sqtt_funcmap\0.shstrtab";
    std::string funcmap =
        custom_funcmap.empty() ? "W:64\nP:7:" + name + "@api.cpp:42\nR:7:extra_payload_count=1\n" : custom_funcmap;
    funcmap.push_back('\0');

    elf::Elf64Ehdr header{};
    std::memcpy(header.e_ident, "\177ELF", 4);
    header.e_ident[4] = 2;  // ELF64
    header.e_ident[5] = 1;  // Little endian
    header.e_ident[6] = 1;  // ELF version
    header.e_ident[7] = 64; // AMDGPU HSA
    header.e_ident[8] = 2;  // HSA code object v4
    header.e_type = 3;      // ET_DYN
    header.e_machine = elf::EM_AMDGPU;
    header.e_version = 1;
    header.e_flags = 0x3f; // gfx90a
    header.e_ehsize = sizeof(header);
    header.e_phentsize = sizeof(elf::Elf64Phdr);
    header.e_shentsize = sizeof(elf::Elf64Shdr);
    header.e_shnum = 3;
    header.e_shstrndx = 2;
    header.e_shoff = (sizeof(header) + funcmap.size() + sizeof(section_names) + 7) & ~uint64_t{7};

    elf::Elf64Shdr sections[3]{};
    sections[1].sh_name = 1;
    sections[1].sh_type = 1; // SHT_PROGBITS
    sections[1].sh_offset = sizeof(header);
    sections[1].sh_size = funcmap.size();
    sections[1].sh_addralign = 1;
    sections[2].sh_name = 1 + sizeof(".sqtt_funcmap");
    sections[2].sh_type = elf::SHT_STRTAB;
    sections[2].sh_offset = sizeof(header) + funcmap.size();
    sections[2].sh_size = sizeof(section_names);
    sections[2].sh_addralign = 1;

    std::vector<uint8_t> bytes(header.e_shoff + sizeof(sections), 0);
    std::memcpy(bytes.data(), &header, sizeof(header));
    std::memcpy(bytes.data() + sections[1].sh_offset, funcmap.data(), funcmap.size());
    std::memcpy(bytes.data() + sections[2].sh_offset, section_names, sizeof(section_names));
    std::memcpy(bytes.data() + header.e_shoff, sections, sizeof(sections));
    return bytes;
}

std::vector<uint8_t> make_gfx9_marker_trace()
{
    std::vector<uint8_t> trace;
    append_gfx9_header(trace);
    append_gfx9_codeobj_load(trace);
    append_gfx9_dispatch_registers(trace);
    const uint64_t location = (uint64_t{kWave} << 10) | (uint64_t{kSimd} << 14);
    append_little_endian(
        trace,
        gfx9::TOKEN_WAVE_START | (uint64_t{1} << 4) | location | (uint64_t{kPipe} << 16) | (uint64_t{kMe} << 18) |
            (uint64_t{8} << 22),
        4
    );
    for (uint32_t value : {7u << 2, 0xCAFEBABEu})
        append_little_endian(
            trace, gfx9::TOKEN_SHADERDATA | (uint64_t{1} << 4) | location | (uint64_t{value} << 16), 6
        );
    append_little_endian(trace, gfx9::TOKEN_WAVE_END | (uint64_t{1} << 4) | location, 2);
    return trace;
}

std::vector<uint8_t> make_packed_marker_trace(uint32_t initial_sample)
{
    std::vector<uint8_t> trace;
    append_gfx9_header(trace);
    append_gfx9_codeobj_load(trace);
    append_gfx9_dispatch_registers(trace);
    const uint64_t location = (uint64_t{kWave} << 10) | (uint64_t{kSimd} << 14);
    append_little_endian(
        trace,
        gfx9::TOKEN_WAVE_START | (uint64_t{1} << 4) | location | (uint64_t{kPipe} << 16) | (uint64_t{kMe} << 18) |
            (uint64_t{8} << 22),
        4
    );
    for (uint32_t offset : {0u, 11u, 26u})
    {
        // MISC and SHADERDATA deltas are multiples of four cycles: 49 + 1
        // makes adjacent headers arrive 200 cycles apart.
        if (offset) append_little_endian(trace, gfx9::TOKEN_MISC | (uint64_t{49} << 4), 2);
        const uint32_t value = ((initial_sample + offset) << 24) | (7u << 2);
        append_little_endian(
            trace, gfx9::TOKEN_SHADERDATA | (uint64_t{1} << 4) | location | (uint64_t{value} << 16), 6
        );
    }
    append_little_endian(trace, gfx9::TOKEN_WAVE_END | (uint64_t{1} << 4) | location, 2);
    return trace;
}

struct MarkerApiSink
{
    struct Marker
    {
        uint64_t code_object_id;
        uint32_t id;
        uint32_t kind;
        uint32_t declared_payload_count;
        std::string name;
        std::string source;
        rocprofiler_thread_trace_decoder_pc_t kernel_entry;
        std::vector<uint32_t> payload;
        rocprofiler_thread_trace_decoder_shaderdata_t shaderdata;
        uint32_t flags;
    };
    std::vector<Marker> markers;
    std::vector<rocprofiler_thread_trace_decoder_shaderdata_t> raw_shaderdata;
    uint64_t other_callbacks = 0;
};

rocprofiler_thread_trace_decoder_status_t collect_api_markers(
    rocprofiler_thread_trace_decoder_record_type_t type, void* data, uint64_t count, void* userdata
)
{
    auto& sink = *static_cast<MarkerApiSink*>(userdata);
    if (type == ROCPROFILER_THREAD_TRACE_DECODER_RECORD_SHADERDATA)
    {
        auto* records = static_cast<const rocprofiler_thread_trace_decoder_shaderdata_t*>(data);
        sink.raw_shaderdata.insert(sink.raw_shaderdata.end(), records, records + count);
        return ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS;
    }
    if (type != ROCPROFILER_THREAD_TRACE_DECODER_RECORD_MARKER)
    {
        ++sink.other_callbacks;
        return ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS;
    }
    auto* cursor = static_cast<const uint8_t*>(data);
    for (uint64_t i = 0; i < count; ++i)
    {
        const auto& marker = *reinterpret_cast<const rocprofiler_thread_trace_decoder_marker_t*>(cursor);
        EXPECT_GE(marker.size, sizeof(marker));
        if (!marker.metadata) return ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR;
        const auto& metadata = *marker.metadata;
        MarkerApiSink::Marker copy{
            metadata.code_object_id,
            metadata.marker_id,
            metadata.marker_kind,
            metadata.payload_count,
            metadata.name ? metadata.name : "",
            metadata.source_location ? metadata.source_location : "",
            marker.kernel_entry,
            {},
            marker.shaderdata,
            marker.marker_flags
        };
        if (marker.payload_count != 0) copy.payload.assign(marker.payload, marker.payload + marker.payload_count);
        sink.markers.push_back(std::move(copy));
        cursor += marker.size;
    }
    return ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS;
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

TEST(MarkerApiTest, LoadsOwnedMetadataWithAndWithoutCustomIsa)
{
    const auto trace = make_gfx9_marker_trace();
    for (bool custom_isa : {false, true})
    {
        SCOPED_TRACE(custom_isa);
        HandleGuard handle;
        ASSERT_EQ(rocprof_trace_decoder_create_handle(&handle.value), ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS);
        uint64_t isa_calls = 0;
        if (custom_isa)
        {
            auto unused_isa = [](char*, uint64_t*, uint64_t*, rocprofiler_thread_trace_decoder_pc_t, void* userdata)
            {
                ++*static_cast<uint64_t*>(userdata);
                return ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR;
            };
            ASSERT_EQ(
                rocprof_trace_decoder_set_isa_callback(handle.value, unused_isa, &isa_calls),
                ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS
            );
        }
        rocprofiler_thread_trace_decoder_record_request_t request{
            .size = sizeof(request), .type = ROCPROFILER_THREAD_TRACE_DECODER_RECORD_MARKER, .flags = 0
        };
        rocprofiler_thread_trace_decoder_record_filter_t filter{
            .size = sizeof(filter), .records = &request, .record_count = 1
        };
        ASSERT_EQ(
            rocprof_trace_decoder_set_record_filter(handle.value, &filter),
            ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS
        );

        for (const std::string name : {"api-marker", "reloaded-marker"})
        {
            SCOPED_TRACE(name);
            auto elf = make_marker_code_object(name);
            ASSERT_EQ(
                rocprof_trace_decoder_codeobj_load(
                    handle.value, kCodeObjectId, kCodeObjectBase, kCodeObjectSize, elf.data(), elf.size()
                ),
                ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS
            );
            std::fill(elf.begin(), elf.end(), 0); // The decoder must own its metadata.

            MarkerApiSink sink;
            ASSERT_EQ(
                rocprof_trace_decoder_parse(handle.value, trace.data(), trace.size(), collect_api_markers, &sink),
                ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS
            );
            EXPECT_EQ(isa_calls, 0u);
            EXPECT_EQ(sink.other_callbacks, 0u);
            ASSERT_EQ(sink.markers.size(), 1u);
            const auto& marker = sink.markers.front();
            EXPECT_EQ(marker.code_object_id, kCodeObjectId);
            EXPECT_EQ(marker.id, 7u);
            EXPECT_EQ(marker.kind, ROCPROFILER_THREAD_TRACE_DECODER_MARKER_KIND_POINT);
            EXPECT_EQ(marker.name, name);
            EXPECT_EQ(marker.source, "api.cpp:42");
            EXPECT_EQ(marker.kernel_entry.code_object_id, kCodeObjectId);
            EXPECT_EQ(marker.kernel_entry.address, 0x80u);
            EXPECT_EQ(marker.declared_payload_count, 1u);
            ASSERT_EQ(marker.payload.size(), 1u);
            EXPECT_EQ(marker.payload.front(), 0xCAFEBABEu);

            EXPECT_EQ(
                rocprof_trace_decoder_codeobj_unload(handle.value, kCodeObjectId),
                ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS
            );
        }
        EXPECT_EQ(
            rocprof_trace_decoder_codeobj_unload(handle.value, kCodeObjectId),
            ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR
        );
    }
}

TEST(MarkerApiTest, CalibratesPackedClocksForPooledImmediateAndStreamedInput)
{
    for (bool immediate : {false, true})
        for (bool streamed : {false, true})
        {
            SCOPED_TRACE(testing::Message() << "immediate=" << immediate << ", streamed=" << streamed);
            HandleGuard handle;
            ASSERT_EQ(
                rocprof_trace_decoder_create_handle(&handle.value), ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS
            );
            auto elf = make_marker_code_object("clock", "M:shader_clock_bits=8;shader_clock_shift=4\nP:7:clock\n");
            ASSERT_EQ(
                rocprof_trace_decoder_codeobj_load(
                    handle.value, kCodeObjectId, kCodeObjectBase, kCodeObjectSize, elf.data(), elf.size()
                ),
                ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS
            );
            rocprofiler_thread_trace_decoder_record_request_t requests[] = {
                {sizeof(requests[0]), ROCPROFILER_THREAD_TRACE_DECODER_RECORD_MARKER,
                 immediate ? ROCPROFILER_THREAD_TRACE_DECODER_RECORD_REQUEST_FLAGS_IMMEDIATE : 0u},
                {sizeof(requests[0]), ROCPROFILER_THREAD_TRACE_DECODER_RECORD_SHADERDATA, 0}
            };
            rocprofiler_thread_trace_decoder_record_filter_t filter{
                .size = sizeof(filter), .records = requests, .record_count = 2
            };
            ASSERT_EQ(
                rocprof_trace_decoder_set_record_filter(handle.value, &filter),
                ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS
            );

            struct Source
            {
                std::vector<std::vector<uint8_t>> buffers{make_packed_marker_trace(50), make_packed_marker_trace(5)};
                size_t next = 0;
                size_t calls = 0;
            } source;
            if (streamed)
            {
                auto supply = [](uint8_t** data, uint64_t* size, void* userdata) -> uint64_t
                {
                    auto& source = *static_cast<Source*>(userdata);
                    ++source.calls;
                    if (source.next == source.buffers.size())
                    {
                        *data = nullptr;
                        *size = 0;
                        return 0;
                    }
                    auto& buffer = source.buffers[source.next++];
                    *data = buffer.data();
                    *size = buffer.size();
                    return buffer.size();
                };
                ASSERT_EQ(
                    rocprof_trace_decoder_set_se_data_callback(handle.value, supply, &source),
                    ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS
                );
            }

            MarkerApiSink sink;
            ASSERT_EQ(
                rocprof_trace_decoder_parse(
                    handle.value, source.buffers[0].data(), source.buffers[0].size(), collect_api_markers, &sink
                ),
                ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS
            );
            EXPECT_EQ(sink.other_callbacks, 0u);
            EXPECT_EQ(source.calls, streamed ? 3u : 0u); // Calibration must not fetch buffers a second time.
            ASSERT_EQ(sink.markers.size(), streamed ? 6u : 3u);
            ASSERT_EQ(sink.raw_shaderdata.size(), sink.markers.size());
            // The experimental correction is opt-in through the environment.
            const char* correction = std::getenv("ROCPROF_TRACE_DECODER_MARKER_CLOCK_CORRECTION");
            const bool corrected = correction && std::strcmp(correction, "1") == 0;
            const int64_t expected_corrections[] = {corrected ? 1 : 0, corrected ? 25 : 0, 0};
            const uint32_t sample_offsets[] = {0, 11, 26};
            for (size_t i = 0; i < sink.markers.size(); ++i)
            {
                const size_t position = i % 3;
                const auto& marker = sink.markers[i];
                const auto& raw = sink.raw_shaderdata[i];
                const uint32_t origin_sample = i < 3 ? 50 : 5;
                EXPECT_EQ(marker.id, 7u);
                EXPECT_EQ(marker.code_object_id, kCodeObjectId);
                EXPECT_TRUE(marker.payload.empty());
                EXPECT_EQ(raw.value, ((origin_sample + sample_offsets[position]) << 24) | (7u << 2));
                EXPECT_EQ(marker.shaderdata.value, raw.value);
                EXPECT_EQ(
                    raw.time - sink.raw_shaderdata[i - position].time,
                    static_cast<int64_t>(position * 200)
                );
                EXPECT_EQ(marker.shaderdata.time, raw.time - expected_corrections[position]);
            }
        }
}

TEST(ParseApiTest, Mi400PacketLossKeepsItsTokenTimestampWithMarkersEnabled)
{
    // The MI400 tail reader returns TIME tokens only while its loss flag is
    // set, so marker handling must not clear that flag early.
    BitStreamBuilder builder;
    append_mi400_header(builder);
    gfx12::timestamp_type stamp{};
    stamp.header = 0b0000001;
    stamp.pl = 1;
    stamp.time = 100;
    builder.writeBits(stamp.raw, 64);
    gfx12::shader_data_type shader{};
    shader.header = 0b0000110 | (1u << 7); // One cycle after the timestamp.
    shader.data = 7u << 2;
    builder.writeBits(shader.raw, 56);
    builder.writeBits(0b1110, 8); // TIME
    const auto trace = builder.finish();

    for (bool markers : {false, true})
    {
        SCOPED_TRACE(testing::Message() << "markers=" << markers);
        HandleGuard handle;
        ASSERT_EQ(rocprof_trace_decoder_create_handle(&handle.value), ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS);
        rocprofiler_thread_trace_decoder_record_request_t requests[] = {
            {sizeof(requests[0]), ROCPROFILER_THREAD_TRACE_DECODER_RECORD_EVENT, 0},
            {sizeof(requests[0]), ROCPROFILER_THREAD_TRACE_DECODER_RECORD_MARKER, 0}
        };
        rocprofiler_thread_trace_decoder_record_filter_t filter{
            .size = sizeof(filter), .records = requests, .record_count = markers ? 2u : 1u
        };
        ASSERT_EQ(
            rocprof_trace_decoder_set_record_filter(handle.value, &filter), ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS
        );
        RecordSink sink;
        ASSERT_EQ(
            rocprof_trace_decoder_parse(handle.value, trace.data(), trace.size(), collect_records, &sink),
            ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS
        );
        ASSERT_EQ(sink.events.size(), 1u);
        EXPECT_EQ(sink.events[0].type, ROCPROF_TRACE_DECODER_EVENT_PACKET_LOSS);
        EXPECT_EQ(sink.events[0].time, 102);
    }
}

TEST(MarkerApiTest, EveryPacketLossStopsAPayloadEvenWhileALossIsPending)
{
    // gfx12 keeps its loss flag set until an unhandled token, so a second loss
    // can arrive while the first is still pending. A wave started in between
    // must not join payload words across the second gap.
    BitStreamBuilder builder;
    append_gfx12_header(builder);
    append_rdna_codeobj_load(builder);
    append_dispatch_registers(builder);
    auto loss = [&]()
    {
        gfx12::timestamp_type stamp{};
        stamp.header = 0b0000001;
        stamp.pl = 1;
        builder.writeBits(stamp.raw, 64);
    };
    auto shaderdata = [&](uint32_t value)
    {
        gfx12::shader_data_type shader{};
        shader.header = 0b0000110;
        shader.data = value;
        builder.writeBits(shader.raw, 56);
    };
    loss();
    gfx12::wstart_type start{};
    start.header = 0b1100;
    start.pipe = kPipe;
    start.me = kMe;
    start.count = 8;
    builder.writeBits(start.raw, 32);
    shaderdata(2u << 2);
    shaderdata(0);
    loss();
    for (uint32_t word = 1; word < 130; ++word) shaderdata(word);
    const auto trace = builder.finish(16);

    HandleGuard handle;
    ASSERT_EQ(rocprof_trace_decoder_create_handle(&handle.value), ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS);
    auto elf = make_marker_code_object("payload", "P:2:payload\nR:2:extra_payload_count=130\n");
    ASSERT_EQ(
        rocprof_trace_decoder_codeobj_load(
            handle.value, kCodeObjectId, kCodeObjectBase, kCodeObjectSize, elf.data(), elf.size()
        ),
        ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS
    );
    rocprofiler_thread_trace_decoder_record_request_t request{
        sizeof(request), ROCPROFILER_THREAD_TRACE_DECODER_RECORD_MARKER, 0
    };
    rocprofiler_thread_trace_decoder_record_filter_t filter{.size = sizeof(filter), .records = &request, .record_count = 1};
    ASSERT_EQ(
        rocprof_trace_decoder_set_record_filter(handle.value, &filter), ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS
    );
    MarkerApiSink sink;
    ASSERT_EQ(
        rocprof_trace_decoder_parse(handle.value, trace.data(), trace.size(), collect_api_markers, &sink),
        ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS
    );

    ASSERT_EQ(sink.markers.size(), 1u);
    EXPECT_EQ(sink.markers[0].id, 2u);
    EXPECT_NE(sink.markers[0].flags & ROCPROFILER_THREAD_TRACE_DECODER_MARKER_FLAGS_INCOMPLETE, 0u);
    EXPECT_EQ(sink.markers[0].payload, std::vector<uint32_t>{0});
}

TEST(MarkerApiTest, RawShaderdataDoesNotDependOnMarkerDecoding)
{
    struct Case
    {
        std::string funcmap;
        std::vector<uint8_t> trace;
    };
    const Case cases[] = {
        {"", make_gfx9_marker_trace()},
        {"M:shader_clock_bits=8;shader_clock_shift=4\nP:7:clock\n", make_packed_marker_trace(50)},
    };
    for (const auto& test : cases)
    {
        std::vector<std::vector<rocprofiler_thread_trace_decoder_shaderdata_t>> results;
        for (uint32_t markers : {0u, 1u, 2u}) // disabled, pooled, immediate
        {
            HandleGuard handle;
            ASSERT_EQ(
                rocprof_trace_decoder_create_handle(&handle.value), ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS
            );
            auto elf = make_marker_code_object("raw", test.funcmap);
            ASSERT_EQ(
                rocprof_trace_decoder_codeobj_load(
                    handle.value, kCodeObjectId, kCodeObjectBase, kCodeObjectSize, elf.data(), elf.size()
                ),
                ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS
            );
            rocprofiler_thread_trace_decoder_record_request_t requests[] = {
                {sizeof(requests[0]), ROCPROFILER_THREAD_TRACE_DECODER_RECORD_SHADERDATA, 0},
                {sizeof(requests[0]), ROCPROFILER_THREAD_TRACE_DECODER_RECORD_MARKER,
                 markers == 2 ? ROCPROFILER_THREAD_TRACE_DECODER_RECORD_REQUEST_FLAGS_IMMEDIATE : 0u}
            };
            rocprofiler_thread_trace_decoder_record_filter_t filter{
                .size = sizeof(filter), .records = requests, .record_count = markers ? 2u : 1u
            };
            ASSERT_EQ(
                rocprof_trace_decoder_set_record_filter(handle.value, &filter),
                ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS
            );
            MarkerApiSink sink;
            ASSERT_EQ(
                rocprof_trace_decoder_parse(
                    handle.value, test.trace.data(), test.trace.size(), collect_api_markers, &sink
                ),
                ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS
            );
            EXPECT_EQ(sink.markers.empty(), markers == 0);
            results.push_back(sink.raw_shaderdata);
        }
        for (size_t i = 1; i < results.size(); ++i)
        {
            ASSERT_EQ(results[i].size(), results[0].size());
            for (size_t j = 0; j < results[0].size(); ++j)
                EXPECT_EQ(std::memcmp(&results[i][j], &results[0][j], sizeof(results[0][j])), 0);
        }
    }
}

TEST(CodeObjectApiTest, ValidatesMetadataLoadsWithoutRequiringDisassembly)
{
    HandleGuard handle;
    ASSERT_EQ(rocprof_trace_decoder_create_handle(&handle.value), ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS);
    auto elf = make_marker_code_object("api-marker");
    EXPECT_EQ(
        rocprof_trace_decoder_codeobj_load(handle.value, kCodeObjectId, kCodeObjectBase, kCodeObjectSize, nullptr, 0),
        ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR_INVALID_ARGUMENT
    );
    EXPECT_EQ(
        rocprof_trace_decoder_codeobj_load(
            handle.value, kCodeObjectId, std::numeric_limits<uint64_t>::max(), 2, elf.data(), elf.size()
        ),
        ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR_INVALID_ARGUMENT
    );
    EXPECT_EQ(
        rocprof_trace_decoder_codeobj_load(
            {0}, kCodeObjectId, kCodeObjectBase, kCodeObjectSize, elf.data(), elf.size()
        ),
        ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR_INVALID_ARGUMENT
    );
    std::fill(elf.begin(), elf.end(), 0);
    EXPECT_EQ(
        rocprof_trace_decoder_codeobj_load(
            handle.value, kCodeObjectId, kCodeObjectBase, kCodeObjectSize, elf.data(), elf.size()
        ),
        ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR
    );
}

#if !ROCPROF_TRACE_DECODER_HAS_DISASM
TEST(ParseApiTest, DefaultWaveRecordsStillRequireIsaWhenDisassemblyIsDisabled)
{
    HandleGuard handle;
    ASSERT_EQ(rocprof_trace_decoder_create_handle(&handle.value), ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS);
    const auto trace = make_gfx9_marker_trace();
    RecordSink sink;
    EXPECT_EQ(
        rocprof_trace_decoder_parse(handle.value, trace.data(), trace.size(), collect_records, &sink),
        ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR_NOT_IMPLEMENTED
    );
    ASSERT_EQ(
        rocprof_trace_decoder_set_isa_callback(handle.value, nop_isa, nullptr),
        ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS
    );
    EXPECT_EQ(
        rocprof_trace_decoder_parse(handle.value, trace.data(), trace.size(), collect_records, &sink),
        ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS
    );
}
#endif

TEST(RecordFilterApiTest, AcceptsExtendedStructuresAndUsesRequestSizeAsStride)
{
    struct ExtendedRequest
    {
        rocprofiler_thread_trace_decoder_record_request_t request;
        uint64_t extension;
    } requests[] = {
        {{sizeof(ExtendedRequest), ROCPROFILER_THREAD_TRACE_DECODER_RECORD_EVENT, 0},    0xAABBCCDDu},
        {{sizeof(ExtendedRequest), ROCPROFILER_THREAD_TRACE_DECODER_RECORD_DISPATCH, 0}, 0x11223344u}
    };
    struct ExtendedFilter
    {
        rocprofiler_thread_trace_decoder_record_filter_t filter;
        uint64_t extension;
    } filter{
        {sizeof(ExtendedFilter), &requests[0].request, 2},
        0xFFFFu
    };

    HandleGuard handle;
    ASSERT_EQ(rocprof_trace_decoder_create_handle(&handle.value), ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS);
    ASSERT_EQ(
        rocprof_trace_decoder_set_isa_callback(handle.value, nop_isa, nullptr),
        ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS
    );
    ASSERT_EQ(
        rocprof_trace_decoder_set_record_filter(handle.value, &filter.filter),
        ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS
    );

    const auto trace = make_dispatch_trace();
    RecordSink sink;
    ASSERT_EQ(
        rocprof_trace_decoder_parse(handle.value, trace.data.data(), trace.data.size(), collect_records, &sink),
        ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS
    );
    EXPECT_EQ(sink.events.size(), 1u);
    EXPECT_EQ(sink.dispatches.size(), 1u);
}

TEST(RecordFilterApiTest, CopiesRequestsAndPreservesFilterAfterInvalidUpdate)
{
    const auto trace = make_dispatch_trace();
    HandleGuard handle;
    ASSERT_EQ(rocprof_trace_decoder_create_handle(&handle.value), ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS);
    ASSERT_EQ(
        rocprof_trace_decoder_set_isa_callback(handle.value, nop_isa, nullptr),
        ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS
    );

    rocprofiler_thread_trace_decoder_record_request_t request{
        .size = sizeof(request), .type = ROCPROFILER_THREAD_TRACE_DECODER_RECORD_EVENT, .flags = 0
    };
    rocprofiler_thread_trace_decoder_record_filter_t filter{
        .size = sizeof(filter), .records = &request, .record_count = 1
    };
    ASSERT_EQ(
        rocprof_trace_decoder_set_record_filter(handle.value, &filter), ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS
    );

    request.type = ROCPROFILER_THREAD_TRACE_DECODER_RECORD_DISPATCH;
    request.flags = 0x80000000u;
    EXPECT_EQ(
        rocprof_trace_decoder_set_record_filter(handle.value, &filter),
        ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR_INVALID_ARGUMENT
    );

    RecordSink sink;
    ASSERT_EQ(
        rocprof_trace_decoder_parse(handle.value, trace.data.data(), trace.data.size(), collect_records, &sink),
        ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS
    );
    EXPECT_EQ(sink.events.size(), 1u);
    EXPECT_TRUE(sink.dispatches.empty());

    filter.record_count = 0;
    filter.records = nullptr;
    ASSERT_EQ(
        rocprof_trace_decoder_set_record_filter(handle.value, &filter), ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS
    );
    CallbackFailure failure;
    EXPECT_EQ(
        rocprof_trace_decoder_parse(
            handle.value, trace.data.data(), trace.data.size(), reject_and_count_records, &failure
        ),
        ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS
    );
    EXPECT_EQ(failure.calls, 0u);

    ASSERT_EQ(
        rocprof_trace_decoder_set_record_filter(handle.value, nullptr), ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS
    );
    sink = RecordSink{};
    ASSERT_EQ(
        rocprof_trace_decoder_parse(handle.value, trace.data.data(), trace.data.size(), collect_records, &sink),
        ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS
    );
    EXPECT_EQ(sink.events.size(), 1u);
    EXPECT_EQ(sink.dispatches.size(), 1u);
}

TEST(RecordFilterApiTest, PropagatesPooledAndImmediateCallbackFailures)
{
    const auto trace = make_mi400_wave_metadata_trace();
    for (uint32_t flags : {0u, uint32_t{ROCPROFILER_THREAD_TRACE_DECODER_RECORD_REQUEST_FLAGS_IMMEDIATE}})
    {
        SCOPED_TRACE(flags);
        HandleGuard handle;
        ASSERT_EQ(rocprof_trace_decoder_create_handle(&handle.value), ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS);
        ASSERT_EQ(
            rocprof_trace_decoder_set_isa_callback(handle.value, nop_isa, nullptr),
            ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS
        );
        rocprofiler_thread_trace_decoder_record_request_t request{
            .size = sizeof(request), .type = ROCPROFILER_THREAD_TRACE_DECODER_RECORD_OCCUPANCY, .flags = flags
        };
        rocprofiler_thread_trace_decoder_record_filter_t filter{
            .size = sizeof(filter), .records = &request, .record_count = 1
        };
        ASSERT_EQ(
            rocprof_trace_decoder_set_record_filter(handle.value, &filter),
            ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS
        );

        CallbackFailure failure;
        EXPECT_EQ(
            rocprof_trace_decoder_parse(handle.value, trace.data(), trace.size(), reject_and_count_records, &failure),
            ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR_OUT_OF_RESOURCES
        );
        EXPECT_EQ(failure.calls, 1u);
        EXPECT_EQ(failure.records, flags == 0 ? 2u : 1u);
    }
}

TEST(ParseApiTest, StopsRequestingInputAfterCallbackFailure)
{
    struct Input
    {
        std::vector<uint8_t> trace;
        uint64_t reads = 0;
        CallbackFailure failure;
    } input{make_mi400_wave_metadata_trace()};
    auto read_input = [](uint8_t** data, uint64_t* size, void* userdata) -> uint64_t
    {
        auto& input = *static_cast<Input*>(userdata);
        ++input.reads;
        *data = input.trace.data();
        *size = input.trace.size();
        return input.reads == 1 ? input.trace.size() : 0;
    };
    auto reject = [](rocprofiler_thread_trace_decoder_record_type_t type, void* data, uint64_t count, void* userdata)
    { return reject_and_count_records(type, data, count, &static_cast<Input*>(userdata)->failure); };
    EXPECT_EQ(
        rocprof_trace_decoder_parse_data(read_input, reject, nop_isa, &input),
        ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR_OUT_OF_RESOURCES
    );
    EXPECT_EQ(input.reads, 1u);
    EXPECT_EQ(input.failure.calls, 1u);
}

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
