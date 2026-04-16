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
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include "gfx10parser.h"
#include "token_types.h"
#include "trace_parser.hpp"

namespace gfx10
{
struct wave_t : public WaveDataInternal
{
    wave_t(int tg_cu, int tg_simd, int slot, pcinfo_t addr, class Token&, bool exbarw);

    int64_t last_state_cycle = 0;
    WaveslotState cur_state = WaveslotState::WS_IDLE;
    bool extend_barrier_gfx11 = false;

    void update_barrier_gfx11(int64_t token_time);
    void apply_wave_rdy(int64_t time);
    void update_state(WaveslotState new_state, int64_t time);
    void complete_wave(Token& token);
    void apply_inst(int64_t token_time, int enum_inst, mapped_inst_t mapped, int tt_version);
    void apply_valu_inst(int64_t token_time);
    void apply_immediate(int64_t token_time);
    void new_pc(int64_t time, int64_t pc_value, class CodeobjTableTranslator& table);

    // static std::unordered_map<int, const char*> INST_NAMES;
    static mapped_inst_t map_to_common_type(int einst, int dprate, int derate);
};

class CSRegisterHandler : public ::CSRegisterHandler
{
public:
};
} // namespace gfx10

class RDNASQTParser : public SQTTParser
{
public:
    ~RDNASQTParser() override = default;
    void sqtt_simd_analysis(CppReturnInfo& info, class TokenGenerator& generator, class Stitcher& stitch) override;

protected:

    static constexpr uint64_t SQTT_CFG_WAVES = 32;
    std::array<std::vector<gfx10::wave_t>, SQTT_CFG_WAVES> SIMD{};

    int num_waves_started = 0;
    int num_waves_completed = 0;
    bool bInitBeginTime = false;
    bool double_buffer = false;

    int target_sa_wgp = 0;
    int target_simd = 0;
    int tt_version = 0;
    bool exclude_barrier_wait = false;

    // data from all waves
    std::unordered_map<uint64_t, pcinfo_t> running_waves{};
    std::unordered_map<uint64_t, pcinfo_t> saved_waves{};

    int dprate = 1;
    int derate = 1;

    gfx10::CSRegisterHandler csregister;
    PipeArray64 wave_start_addr{};

    std::unordered_map<uint32_t, uint64_t> active_codeobj_id{};
};
