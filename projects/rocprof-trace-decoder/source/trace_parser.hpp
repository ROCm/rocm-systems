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

#pragma once
#include <stdint.h>
#include <array>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iostream>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include "cow_ptr.hpp"
#include "rocprof_trace_decoder/cxx/common.hpp"
#include "rocprof_trace_decoder/trace_decoder_instrument.h"
#include "segment.hpp"

inline bool bValid(pcinfo_t pc) { return pc.code_object_id != 0 || pc.address != 0; }

enum sqtt_token_reg_t
{
    COMPUTE_DISPATCH_INITIATOR = 0x0,
    COMPUTE_NUM_THREAD_X = 0x7,
    COMPUTE_NUM_THREAD_Y,
    COMPUTE_NUM_THREAD_Z,
    COMPUTE_PGM_LO = 0xC,
    COMPUTE_PGM_HI,
    COMPUTE_DISPATCH_PKT_LO,
    COMPUTE_DISPATCH_PKT_HI,
    COMPUTE_PGM_RSRC1 = 0x12,
    COMPUTE_PGM_RSRC2,
    COMPUTE_PGM_RSRC3 = 0x2D,
    COMPUTE_NOWHERE = 0x7F
};

enum sqtt_event_type_t
{
    EVENT_CACHE_FLUSH_WR = 0x4,
    EVENT_CACHE_FLUSH = 0x6,
    EVENT_CS_PARTIAL_FLUSH = 0x7,
    EVENT_CACHE_FLUSH_INV_WR = 0x14,
    EVENT_CACHE_FLUSH_INV = 0x16,
    EVENT_BOTTOM_OF_PIPE_WR = 0x28,
    EVENT_TT_FLUSH = 0x36,
};

struct occupancy_info_t : public rocprofiler_thread_trace_decoder_occupancy_t
{
    occupancy_info_t() = default;
    occupancy_info_t(
        pcinfo_t pc,
        int64_t time,
        uint64_t cu,
        uint64_t simd,
        uint64_t slot,
        uint64_t start,
        uint64_t me,
        uint64_t pipe,
        uint64_t is_ext,
        uint64_t wg
    )
    {
        this->pc = pc;
        this->time = time;
        this->reserved = (uint8_t) 0;
        this->cu = (uint8_t) cu;
        this->simd = (uint8_t) simd;
        this->wave_id = (uint8_t) slot;
        this->start = start;
        this->me_id = me & 0x7;
        this->pipe_id = pipe & 0xF;
        this->is_ext = is_ext;
        this->workgroup_id = wg & 0x7F;
        this->_rsvd = 0;
    }
    occupancy_info_t(
        pcinfo_t pc,
        int64_t time,
        int8_t cu,
        int8_t simd,
        int8_t slot,
        uint64_t start,
        uint64_t me,
        uint64_t pipe,
        uint64_t is_ext,
        uint64_t wg
    )
    {
        this->pc = pc;
        this->time = time;
        this->reserved = (uint8_t) 0;
        this->cu = (uint8_t) cu;
        this->simd = (uint8_t) simd;
        this->wave_id = (uint8_t) slot;
        this->start = start;
        this->me_id = me & 0x7;
        this->pipe_id = pipe & 0xF;
        this->is_ext = is_ext;
        this->workgroup_id = wg & 0x7F;
        this->_rsvd = 0;
    }
};

static_assert(
    sizeof(occupancy_info_t) == sizeof(rocprofiler_thread_trace_decoder_occupancy_t), "Occ cannot share layout!"
);

struct Instruction : public rocprofiler_thread_trace_decoder_inst_t
{
    Instruction() = default;
    Instruction(int64_t _time, WaveInstCategory _category, int64_t _cycles, int64_t _stall)
    {
        this->time = _time;
        this->category = uint32_t(_category);
        this->duration = uint32_t(_cycles);
        this->stall = uint32_t(_stall);
        this->pc.code_object_id = 0;
        this->pc.address = 0;
    }

    inline bool operator==(const Instruction& other) const { return !(*this != other); };
    inline bool operator!=(const Instruction& other) const
    {
        if (this->category != other.category)
            return true;
        else
            return pc != other.pc;
    };
};
static_assert(sizeof(rocprofiler_thread_trace_decoder_inst_t) == sizeof(Instruction), "Structs cannot share layout!");

struct WaveDataInternal : public rocprofiler_thread_trace_decoder_wave_t
{
    WaveDataInternal(
        int cu,
        int simd,
        int slot,
        int64_t start,
        pcinfo_t addr,
        bool exbarw,
        uint8_t me = 0,
        uint8_t pipe = 0,
        uint8_t wg = 0
    )
    {
        this->cu = (uint8_t) cu;
        this->simd = (uint8_t) simd;
        this->wave_id = (uint8_t) slot;
        this->contexts = (uint8_t) 0;

        this->dispatcher = (uint8_t) (((me & 0x7) << 4) | (pipe & 0xF));
        this->workgroup_id = wg;
        this->reserved = 0;
        this->size = sizeof(rocprofiler_thread_trace_decoder_wave_t);

        this->begin_time = start;
        this->end_time = 0;

        this->instructions_size = 0;
        this->timeline_size = 0;
        this->instructions_array = nullptr;
        this->timeline_array = nullptr;

        if (bValid(addr))
        {
            pc_infos.push_back({0, addr});
            if (addr.code_object_id == 0 && addr.address != 0) unattrib_pcs.push_back(0);
        }

        this->exclude_barrier_wait = exbarw;
    }
    WaveTrapStatus trap_status = WaveTrapStatus::TRAP_RESTORED;

    bool exclude_barrier_wait = false;
    bool bIsComplete = false;
    bool isTrapping = false;
    bool callbackComplete = false;

    std::vector<Instruction> instructions{};
    std::vector<att_wave_state_t> timeline{}; // wave state in each cycle

    std::vector<size_t> unattrib_pcs{};
    std::vector<std::pair<size_t, pcinfo_t>> pc_infos{};

    void lookbackpcs(class CSRegisterHandler& reg);
};

struct CppReturnInfo
{
    std::vector<att_perfevent_t> perfevents{};
    uint64_t realtime_frequency{0};
    uint64_t counter_frequency{0};
    bool bPacketLost{false};

    int64_t globaltime{0};
    int64_t basetime{0};
};

class TokenGenerator
{
public:
    TokenGenerator(const uint8_t* _buffer, size_t size, int64_t _globaltime, int64_t _base_time) :
    BUFFER_SIZE(size), buffer(_buffer), globaltime(_globaltime), base_time(_base_time)
    {}
    virtual ~TokenGenerator() = default;

    int64_t get_time() const { return globaltime; };
    int64_t get_base_time() const { return base_time; };

    const size_t BUFFER_SIZE;

protected:
    const uint8_t* buffer;
    int64_t globaltime = 0;
    int64_t base_time = 0;
};

class SQTTParser
{
public:
    virtual ~SQTTParser() = default;
    virtual void sqtt_simd_analysis(CppReturnInfo& info, class TokenGenerator& generator, class Stitcher& stitch) = 0;
};

std::unique_ptr<SQTTParser> AnalyseBinary_internal(
    CppReturnInfo& info, const uint8_t* buffer, uint64_t BUFFER_SIZE, int gfx9_target_cu, class Stitcher& stitch
);

// Token iteration architectures recognised by IterateTokens_internal.
enum class TraceArch
{
    UNKNOWN = 0,
    GFX9,
    GFX10,
    GFX11,
    GFX12,
    MI400
};

// Token-walking helper IterateTokens_internal lives in iterate_tokens.hpp
// (header-only template) so the visitor inlines and we don't drag the
// per-arch token generator headers into every TU that includes this file.

// Sniffs only the buffer header to identify the trace architecture, without
// constructing any token generator. Mirrors the dispatch in
// AnalyseBinary_internal / IterateTokens_internal.
TraceArch DetectArch_internal(const uint8_t* buffer, uint64_t buffer_size);

/*
void applyGenerator(
    CppReturnInfo& info,
    SQTTParser& parser,
    class Stitcher& stitch,
    const uint8_t* buffer,
    uint64_t BUFFER_SIZE,
    int64_t globaltime,
    int64_t basetime
);
*/

template <typename Type> class PipeArray : public std::array<std::array<Type, 4>, 2>
{
public:
    template <typename T2> Type& at_reg(const T2& token) { return this->at(token.me & 0x1).at(token.pipe); }
};

class PipeArray64 : public PipeArray<uint64_t>
{
public:
    template <typename T2> void setlo(const T2& token, uint64_t lo)
    {
        uint64_t& elem = at_reg(token);
        elem = (elem & ~((1ul << 32) - 1)) | lo;
    }
    template <typename T2> void sethi(const T2& token, uint64_t hi)
    {
        uint64_t& elem = at_reg(token);
        elem = (elem & ((1ul << 32) - 1)) | (hi << 32);
    }
    template <typename T2> void setlo(const T2& token) { setlo(token, token.regdata); }
    template <typename T2> void sethi(const T2& token) { sethi(token, token.regdata); }
};

// Register-state tracker shared by quick_scan, build_standalone, and the
// full wave decoder. The heavy members (active codeobj log, address-indexed
// translator tables) are wrapped in CowPtr so per-chunk pipestate copies just
// bump shared_ptr refcounts in the steady state — the underlying vector and
// CodeobjTableTranslators only fork on the rare chunk that actually loads or
// unloads a codeobj. The single class is used by every code path (no
// base/derived split): quick_scan and build_standalone never touch the
// translator tables, so their snapshots leave those CowPtrs null and pay
// zero alloc cost.
class CSRegisterHandler
{
public:
    PipeArray64 wave_start_addr{};
    PipeArray64 current_codeobj_size{};
    PipeArray64 current_codeobj_addr{};
    PipeArray64 current_codeobj_id{};
    PipeArray64 dispatch_pkt_addr{};

    uint32_t num_thread_x{0};
    uint32_t num_thread_y{0};
    uint32_t num_thread_z{0};
    uint32_t rsrc1{0};
    uint32_t rsrc2{0};
    uint32_t rsrc3{0};
    uint32_t dispatch_initiator{0};

    uint64_t realtime_frequency{0};
    uint64_t counter_frequency{0};

    bool bIsROCMFormat = false;
    int userdata_state{};

    CowPtr<std::vector<address_range_t>> active_codeobjs{};
    CowPtr<CodeobjTableTranslator> table{};
    CowPtr<CodeobjTableTranslator> table_from_start{};

    std::vector<att_decoder_realtime_t> realtime{};

    struct RegUpdateEvent
    {
        enum Kind
        {
            NONE,
            CODEOBJ_LOAD,
            CODEOBJ_UNLOAD,
            COUNTER_FREQUENCY_CHANGED,
        };
        Kind kind = NONE;
        uint64_t id = 0; // code object id for CODEOBJ_LOAD / CODEOBJ_UNLOAD
    };

    template <typename TokenType> uint32_t get_regaddr(const TokenType& token) { return token.regaddr; }
    template <typename TokenType> uint32_t get_regdata(const TokenType& token) { return token.regdata; }

    template <typename TokenType> rocprof_trace_decoder_packet_header_t UserdataF(const TokenType& token)
    {
        return rocprof_trace_decoder_packet_header_t{.u32All = static_cast<uint32_t>(token.regdata)};
    }

    template <typename TokenType> std::pair<bool, bool> isUserdataState(const TokenType& token)
    {
        auto data = UserdataF(token);

        if (data.opcode == ROCPROF_TRACE_DECODER_PACKET_OPCODE_AGENT_INFO)
        {
            if (data.type == ROCPROF_TRACE_DECODER_AGENT_INFO_TYPE_RT_FREQUENCY_KHZ)
                realtime_frequency = data.data20 * 1000;
            else if (data.type == ROCPROF_TRACE_DECODER_AGENT_INFO_TYPE_COUNTER_INTERVAL)
                counter_frequency = data.data20;
            return {false, data.type == ROCPROF_TRACE_DECODER_AGENT_INFO_TYPE_COUNTER_INTERVAL};
        }

        return {data.opcode == ROCPROF_TRACE_DECODER_PACKET_OPCODE_CODEOBJ && data.data20 == 0, false};
    }

    template <typename TokenType>
    std::pair<rocprof_trace_decoder_agent_info_type_t, uint32_t> isAgentInfo(const TokenType& token)
    {
        auto data = UserdataF(token);
        if (data.opcode != ROCPROF_TRACE_DECODER_PACKET_OPCODE_AGENT_INFO)
            return {ROCPROF_TRACE_DECODER_AGENT_INFO_TYPE_LAST, 0};
        return {static_cast<rocprof_trace_decoder_agent_info_type_t>(data.type), data.data20};
    }

    template <typename TokenType> bool isUserdataHeader(const TokenType& token)
    {
        uint32_t regdata = static_cast<uint32_t>(token.regdata);
        rocprof_trace_decoder_instrument_enable_t data{.u32All = regdata};
        return data.char1 == '\0' && data.char2 == 'R' && data.char3 == 'O' && data.char4 == 'C';
    }

    static constexpr size_t USERDATA_ADDR_0 = 0xC340;
    static constexpr size_t USERDATA_ADDR_1 = 0xC341;
    static constexpr size_t USERDATA_ADDR_2 = 0xC342;
    static constexpr size_t USERDATA_ADDR_3 = 0xC343;

    bool IsUserdata(size_t addr) { return addr >= USERDATA_ADDR_0 && addr <= USERDATA_ADDR_3; }
    bool IsUserdata0(size_t addr) { return addr == USERDATA_ADDR_0; }
    bool IsUserdata1(size_t addr) { return addr == USERDATA_ADDR_1; }
    bool IsUserdata2(size_t addr) { return addr == USERDATA_ADDR_2; }
    bool IsUserdata3(size_t addr) { return addr == USERDATA_ADDR_3; }

    ~CSRegisterHandler() = default;

    template <typename TokenType> void UpdateRegCS(const TokenType& token)
    {
        switch (token.regaddr)
        {
            case COMPUTE_PGM_LO: wave_start_addr.setlo(token); break;
            case COMPUTE_PGM_HI: wave_start_addr.sethi(token); break;
            case COMPUTE_NUM_THREAD_X: num_thread_x = token.regdata; break;
            case COMPUTE_NUM_THREAD_Y: num_thread_y = token.regdata; break;
            case COMPUTE_NUM_THREAD_Z: num_thread_z = token.regdata; break;
            case COMPUTE_PGM_RSRC1: rsrc1 = token.regdata; break;
            case COMPUTE_PGM_RSRC2: rsrc2 = token.regdata; break;
            case COMPUTE_PGM_RSRC3: rsrc3 = token.regdata; break;
            case COMPUTE_DISPATCH_PKT_LO: dispatch_pkt_addr.setlo(token); break;
            case COMPUTE_DISPATCH_PKT_HI: dispatch_pkt_addr.sethi(token); break;
            case COMPUTE_DISPATCH_INITIATOR: dispatch_initiator = token.regdata; break;
            default: break;
        };
    }

    template <typename TokenType> RegUpdateEvent UpdateRegNoCS(const TokenType& token)
    {
        auto WAIT_FOR_HEADER = ROCPROF_TRACE_DECODER_CODEOBJ_MARKER_TYPE_LAST;

        if (!IsUserdata2(token.regaddr)) return {};

        if (!bIsROCMFormat && isUserdataHeader(token))
        {
            userdata_state = ROCPROF_TRACE_DECODER_CODEOBJ_MARKER_TYPE_LAST;
            bIsROCMFormat = true;
            return {};
        }

        if (!bIsROCMFormat) return {};

        if (userdata_state == WAIT_FOR_HEADER)
        {
            auto statechange = isUserdataState(token);
            if (statechange.first) userdata_state = UserdataF(token).type;
            if (statechange.second) return {RegUpdateEvent::COUNTER_FREQUENCY_CHANGED, 0};
            return {};
        }

        RegUpdateEvent event{};

        switch (userdata_state)
        {
            case ROCPROF_TRACE_DECODER_CODEOBJ_MARKER_TYPE_TAIL:
            {
                rocprof_trace_decoder_codeobj_marker_tail_t header{.raw = uint32_t(token.regdata)};
                uint64_t id = header.legacy_id;
                if (id == 0) // If not using legacy code ID
                    id = current_codeobj_id.at_reg(token);

                bool active = false;
                for (const auto& co : active_codeobjs.read())
                    if (co.id == id)
                    {
                        active = true;
                        break;
                    }

                if (!header.isUnload && !active)
                {
                    uint64_t base_addr = current_codeobj_addr.at_reg(token);
                    uint64_t size = current_codeobj_size.at_reg(token);
                    address_range_t arange{base_addr, size, id};
                    active_codeobjs.write().push_back(arange);
                    table.write().insert(arange);
                    if (header.bFromStart) table_from_start.write().insert(arange);
                    event = {RegUpdateEvent::CODEOBJ_LOAD, id};
                }
                else if (header.isUnload && active)
                {
                    auto& v = active_codeobjs.write();
                    for (auto it = v.begin(); it != v.end(); ++it)
                    {
                        if (it->id == id)
                        {
                            uint64_t addr = it->addr;
                            v.erase(it);
                            try
                            {
                                table.write().remove(addr);
                            }
                            catch (...)
                            {}
                            break;
                        }
                    }
                    event = {RegUpdateEvent::CODEOBJ_UNLOAD, id};
                }
                break;
            }
            case ROCPROF_TRACE_DECODER_CODEOBJ_MARKER_TYPE_ID_LO: current_codeobj_id.setlo(token); break;
            case ROCPROF_TRACE_DECODER_CODEOBJ_MARKER_TYPE_ID_HI: current_codeobj_id.sethi(token); break;
            case ROCPROF_TRACE_DECODER_CODEOBJ_MARKER_TYPE_SIZE_LO: current_codeobj_size.setlo(token); break;
            case ROCPROF_TRACE_DECODER_CODEOBJ_MARKER_TYPE_SIZE_HI: current_codeobj_size.sethi(token); break;
            case ROCPROF_TRACE_DECODER_CODEOBJ_MARKER_TYPE_ADDR_LO: current_codeobj_addr.setlo(token); break;
            case ROCPROF_TRACE_DECODER_CODEOBJ_MARKER_TYPE_ADDR_HI: current_codeobj_addr.sethi(token); break;
            default: break;
        }

        userdata_state = WAIT_FOR_HEADER;

        return event;
    }

    rocprofiler_thread_trace_decoder_dispatch_t PopulateDispatch(int64_t time, int me, int pipe, int tt_version = 0)
    {
        rocprofiler_thread_trace_decoder_dispatch_t event{};
        event.size = sizeof(rocprofiler_thread_trace_decoder_dispatch_t);
        event.time = time;
        event.me_id = me;
        event.pipe_id = pipe;

        uint64_t pc = wave_start_addr.at(me).at(pipe) << 8;
        event.entry_point = pcinfo_t{.address = pc, .code_object_id = 0};
        for (const auto& co : active_codeobjs.read())
            if (co.inrange(pc))
            {
                event.entry_point = {pc - co.addr, co.id};
                break;
            }

        event.thread_dim_x = num_thread_x;
        event.thread_dim_y = num_thread_y;
        event.thread_dim_z = num_thread_z;
        event.lds_size = ((rsrc2 >> 15) & 0x1FF) * 512;

        event.sgprs = 128;
        event.vgprs = (rsrc1 & 0x3F) * 8 + 8;
        event.user_sgprs = (rsrc2 >> 1) & 0x1F;

        if (tt_version == 0) event.sgprs = ((rsrc1 >> 7) & 0x7) * 16 + 16;

        if (tt_version >= 5)
        {
            event.vgprs *= 2;
            event.lds_size *= 2;
        }

        event.flags = ROCPROFILER_THREAD_TRACE_DECODER_DISPATCH_FLAGS_NONE;
        if ((rsrc1 >> 10) & 1) event.flags |= ROCPROFILER_THREAD_TRACE_DECODER_DISPATCH_FLAGS_SCALAR_CACHE_INVALIDATE;
        if ((rsrc1 >> 11) & 1) event.flags |= ROCPROFILER_THREAD_TRACE_DECODER_DISPATCH_FLAGS_VECTOR_CACHE_INVALIDATE;
        if ((rsrc1 >> 14) & 1) event.flags |= ROCPROFILER_THREAD_TRACE_DECODER_DISPATCH_FLAGS_IS_CTX_RESTORE;
        if (rsrc2 & 1) event.flags |= ROCPROFILER_THREAD_TRACE_DECODER_DISPATCH_FLAGS_SCRATCH_ENABLED;

        return event;
    }

    template <typename TokenType> pcinfo_t get_wave_start(const TokenType& token)
    {
        constexpr uint64_t BITMASK = (1ul << 48) - 1;
        return ToPcV2(table.write(), (wave_start_addr.at_reg(token) << 8) & BITMASK);
    }

    pcinfo_t get_wave_start_delayed(uint64_t addr) { return ToPcV2(table_from_start.write(), addr); }
};

template <typename WaveArray> struct AnalysisReturnData
{
    WaveArray waves;
    std::vector<att_perfevent_t> perfevents;
    bool packetLost;
};
