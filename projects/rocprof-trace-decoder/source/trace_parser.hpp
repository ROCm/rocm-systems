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
#include <iostream>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include "common.hpp"
#include "segment.hpp"
#include "trace_decoder_instrument.h"

inline bool bValid(pcinfo_t pc) { return pc.code_object_id != 0 || pc.address != 0; }

struct occupancy_info_t : public rocprofiler_thread_trace_decoder_occupancy_t
{
    occupancy_info_t() = default;
    occupancy_info_t(pcinfo_t pc, int64_t time, uint64_t cu, uint64_t simd, uint64_t slot, uint64_t start)
    {
        this->pc = pc;
        this->time = time;
        this->reserved = (uint8_t) 0;
        this->cu = (uint8_t) cu;
        this->simd = (uint8_t) simd;
        this->wave_id = (uint8_t) slot;
        this->start = start;
        this->_rsvd = 0;
    }
    occupancy_info_t(pcinfo_t pc, int64_t time, int8_t cu, int8_t simd, int8_t slot, uint64_t start)
    {
        this->pc = pc;
        this->time = time;
        this->reserved = (uint8_t) 0;
        this->cu = (uint8_t) cu;
        this->simd = (uint8_t) simd;
        this->wave_id = (uint8_t) slot;
        this->start = start;
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
    WaveDataInternal(int cu, int simd, int slot, int64_t start, pcinfo_t addr, bool exbarw)
    {
        this->cu = (uint8_t) cu;
        this->simd = (uint8_t) simd;
        this->wave_id = (uint8_t) slot;
        this->contexts = (uint8_t) 0;

        this->_rsvd1 = 0;
        this->_rsvd2 = 0;
        this->_rsvd3 = 0;

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

class CSRegisterHandler
{
public:
    CodeobjTableTranslator table{};
    CodeobjTableTranslator table_from_start{};

    std::unordered_map<size_t, uint64_t> active_codeobj_id{};

    PipeArray64 wave_start_addr{};
    PipeArray64 current_codeobj_size{};
    PipeArray64 current_codeobj_addr{};
    PipeArray64 current_codeobj_id{};

    std::vector<att_decoder_realtime_t> realtime{};

    uint64_t realtime_frequency{0};
    uint64_t counter_frequency{0};

    bool bIsROCMFormat = false;
    int userdata_state{};

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

    static constexpr size_t COMPUTE_PGM_LO = 0xC;
    static constexpr size_t COMPUTE_PGM_HI = 0xD;
    static constexpr size_t USERDATA_ADDR_0 = 0xC340;
    static constexpr size_t USERDATA_ADDR_1 = 0xC341;
    static constexpr size_t USERDATA_ADDR_2 = 0xC342;
    static constexpr size_t USERDATA_ADDR_3 = 0xC343;

    bool IsPgmLo(size_t addr) { return addr == COMPUTE_PGM_LO; }
    bool IsPgmHi(size_t addr) { return addr == COMPUTE_PGM_HI; }
    bool IsUserdata(size_t addr) { return addr >= USERDATA_ADDR_0 && addr <= USERDATA_ADDR_3; }
    bool IsUserdata0(size_t addr) { return addr == USERDATA_ADDR_0; }
    bool IsUserdata1(size_t addr) { return addr == USERDATA_ADDR_1; }
    bool IsUserdata2(size_t addr) { return addr == USERDATA_ADDR_2; }
    bool IsUserdata3(size_t addr) { return addr == USERDATA_ADDR_3; }
    virtual ~CSRegisterHandler() = default;

    template <typename TokenType> void UpdateRegCS(const TokenType& token)
    {
        if (IsPgmLo(token.regaddr))
            wave_start_addr.setlo(token);
        else if (IsPgmHi(token.regaddr))
            wave_start_addr.sethi(token);
    }

    template <typename TokenType> bool UpdateRegNoCS(const TokenType& token)
    {
        auto WAIT_FOR_HEADER = ROCPROF_TRACE_DECODER_CODEOBJ_MARKER_TYPE_LAST;

        if (!IsUserdata2(token.regaddr)) return false;

        if (!bIsROCMFormat && isUserdataHeader(token))
        {
            userdata_state = ROCPROF_TRACE_DECODER_CODEOBJ_MARKER_TYPE_LAST;
            bIsROCMFormat = true;
            return false;
        }

        if (!bIsROCMFormat) return false;

        if (userdata_state == WAIT_FOR_HEADER)
        {
            auto statechange = isUserdataState(token);
            if (statechange.first) userdata_state = UserdataF(token).type;
            return statechange.second;
        }

        switch (userdata_state)
        {
            case ROCPROF_TRACE_DECODER_CODEOBJ_MARKER_TYPE_TAIL:
            {
                rocprof_trace_decoder_codeobj_marker_tail_t header{.raw = uint32_t(token.regdata)};
                uint64_t id = header.legacy_id;
                if (id == 0) // If not using legacy code ID
                    id = current_codeobj_id.at_reg(token);

                auto it = active_codeobj_id.find(id);
                if (!header.isUnload && it == active_codeobj_id.end())
                {
                    uint64_t base_addr = current_codeobj_addr.at_reg(token);
                    active_codeobj_id.emplace(id, base_addr);
                    address_range_t arange = {base_addr, current_codeobj_size.at_reg(token), id};
                    table.insert(arange);
                    if (header.bFromStart) table_from_start.insert(arange);
                }
                else if (header.isUnload && it != active_codeobj_id.end())
                {
                    try
                    {
                        table.remove(active_codeobj_id.at(id));
                        active_codeobj_id.erase(id);
                    }
                    catch (...)
                    {}
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

        return false;
    }

    template <typename TokenType> pcinfo_t get_wave_start(const TokenType& token)
    {
        constexpr uint64_t BITMASK = (1ul << 48) - 1;
        return table.ToPcV2((wave_start_addr.at_reg(token) << 8) & BITMASK);
    }

    pcinfo_t get_wave_start_delayed(uint64_t addr) { return table_from_start.ToPcV2(addr); }
};

template <typename WaveArray> struct AnalysisReturnData
{
    WaveArray waves;
    std::vector<att_perfevent_t> perfevents;
    bool packetLost;
};
