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
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include "gfx9token.h"
#include "trace_parser.hpp"

namespace gfx9
{

struct wave_t : public WaveDataInternal
{
    wave_t(int cu, pcinfo_t addr, Wave&, int64_t token_time);

    int cur_state = 0;
    int64_t state_start_cycle = 0;

    int64_t last_message_time = 0;
    std::vector<int64_t> stall_start_times{};
    std::vector<int64_t> issued_instructions{};

    std::array<size_t, 17> last_inst_for_xnack{};

    void complete_wave(int64_t token_time);
    void apply_inst(Token& token, int64_t& start_phase);
    void apply_pc(Token& token, CodeobjTableTranslator& table);
    int64_t apply_issue(uint64_t wave_status, int64_t token_time);
};

class CSRegisterHandlerGFX9 : public CSRegisterHandler
{
    static constexpr size_t SQTT_TOKEN_REG = 2;
    static constexpr size_t SQTT_TOKEN_REG_CS = 5;
    static constexpr size_t SQTT_TOKEN_REG_CS_PRIV = 15;

public:
    bool IsRegCS(size_t type) { return type == SQTT_TOKEN_REG_CS || type == SQTT_TOKEN_REG_CS_PRIV; };
    bool IsRegNoCS(size_t type) { return type == SQTT_TOKEN_REG; };

    void HandleRealtimeClock(size_t time, size_t data);

    enum RealtimeEntry
    {
        RT_DELTA = 0,
        RT_HI,
        RT_LOW,
        RT_LAST
    };

    int userdata3_count = 0;
    std::array<size_t, RT_LAST> userdata3_values{};
};

class MISQTTParser : public SQTTParser
{
public:
    MISQTTParser(int tg_cu, bool _double_buffer) : target_cu(tg_cu), double_buffer(_double_buffer){};
    ~MISQTTParser() override{};

    void sqtt_simd_analysis(CppReturnInfo& info, class TokenGenerator& generator, class Stitcher& stitch) override;

    static constexpr uint64_t SQTT_CFG_SIMDS = 4;
    static constexpr uint64_t SQTT_CFG_WAVES = 10;
    static constexpr int NUM_CU = 16;
    static constexpr int NUM_BANK = 4;

    typedef std::array<std::array<std::vector<wave_t>, SQTT_CFG_WAVES>, SQTT_CFG_SIMDS> WaveArray;

protected:
    static int64_t array_apply_issue(Token& token, WaveArray& SIMD);
    const int target_cu;
    const bool double_buffer;

    WaveArray SIMD{};

    std::array<std::array<int64_t, NUM_BANK>, NUM_CU> last_event{};
    std::array<std::array<bool, NUM_BANK>, NUM_CU> last_value{};

    int64_t CYCLE_START_PHASE{15};

    std::unordered_map<uint64_t, pcinfo_t> running_waves{};

    CSRegisterHandlerGFX9 csregister;
};

}; // namespace gfx9
