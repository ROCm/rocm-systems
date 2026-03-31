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

#include "gfx9wave.h"
#include <algorithm>
#include <array>
#include <cassert>
#include <sstream>
#include <string_view>
#include <utility>
#include <vector>
#include "stitch/stitch.hpp"

#define MAX_ACCUM_RECORDS 65536

#define empty_wave_check(waveslot_size)                                                                                \
    if (waveslot_size == 0) { continue; }

namespace gfx9
{

constexpr std::array<std::string_view, 29> inst_type_dict = {
    {
     "INST_SMEM_RD", "INST_SALU_32",
     "INST_VMEM_RD", "INST_VMEM_WR",
     "INST_FLAT_WR", "INST_VALU_32",
     "INST_LDS", "INST_PC",
     "INST_EXPREQ_GDS", "INST_EXPREQ_GFX",
     "INST_EXPGNT_PAR_COL", "INST_EXPGNT_POS_GDS",
     "INST_JUMP", "INST_NEXT",
     "INST_FLAT_RD", "INST_OTHER_MSG",
     "INST_SMEM_WR", "INST_SALU_64",
     "INST_VALU_64", "INST_SMEM_RD_REPLAY",
     "INST_SMEM_WR_REPLAY", "INST_VMEM_RD_REPLAY",
     "INST_VMEM_WR_REPLAY", "INST_FLAT_WR_REPLAY",
     "INST_FLAT_RD_REPLAY", "INST_FATAL_HALT",
     "RESERVED0", "RESERVED1", "INST_VALU_MAI", }
};

constexpr std::array<std::string_view, 16> token_name_dict = {
    {
     "TOKEN_MISC", "TOKEN_TIMESTAMP",
     "TOKEN_REG", "TOKEN_WAVE_START",
     "TOKEN_WAVE_ALLOC", "TOKEN_REG_CSPRIV",
     "TOKEN_WAVE_END", "TOKEN_EVENT",
     "TOKEN_EVENT_CS", "TOKEN_EVENT_GFX1",
     "TOKEN_INST", "TOKEN_INST_PC",
     "TOKEN_INST_USERDATA", "TOKEN_ISSUE",
     "TOKEN_PERF", "TOKEN_REG_CS",
     }
};

constexpr std::array<std::string_view, 6> misc_token_type_dict = {
    {
     "TIME", "TIME_RESET",
     "PACKET_LOST", "SURF_SYNC",
     "TTRACE_STALL_BEGIN", "TTRACE_STALL_END",
     }
};

std::string print_token(Token& token)
{
    std::stringstream ss;

    ss << token_name_dict.at(token.type);
    if (token.type == TOKEN_MISC)
        ss << " misc:" << misc_token_type_dict[token.fields.misc.misc_type];
    else if (token.type == TOKEN_WAVE_START)
        ss << " start: " << token.fields.wave.cu << " " << token.fields.wave.simd << " " << token.fields.wave.wave;

    return ss.str();
}

int64_t lmax(int64_t a, int64_t b) { return std::max(a, b); }

void convertPerfEventsTs(std::vector<att_perfevent_t>& vec, int64_t interval);

enum InstType
{
    SMEM_RD = 0,
    SALU_32,
    VMEM_RD,
    VMEM_WR,
    FLAT_WR,
    VALU_32,
    LDS,
    PC,
    EXPREQ_GDS,
    EXPREQ_GFX,
    EXPGNT_PAR_COL,
    EXPGNT_POS_GDS,
    JUMP,
    NEXT,
    FLAT_RD,
    OTHER_MSG,
    SMEM_WR,
    SALU_64,
    VALU_64,
    SMEM_RD_REPLAY,
    SMEM_WR_REPLAY,
    VMEM_RD_REPLAY,
    VMEM_WR_REPLAY,
    FLAT_WR_REPLAY,
    FLAT_RD_REPLAY,
    FATAL_HALT,
    RESERVED0,
    RESERVED1,
    VALU_MAI,
    GO_TO_TRAP
};

const uint64_t SQTT_ISSUE_NULL = 0;
const uint64_t SQTT_ISSUE_STALL = 1;
const uint64_t SQTT_ISSUE_INST = 2;
const uint64_t SQTT_ISSUE_IMMED = 3;

wave_t::wave_t(int cu, pcinfo_t addr, Wave& token, int64_t token_time) :
WaveDataInternal(cu, token.simd, token.wave, token_time, addr, false)
{
    this->cur_state = WaveslotState::WS_IDLE;
    this->state_start_cycle = token_time;
}

void wave_t::complete_wave(int64_t token_time)
{
    token_time += 4; // This state change eneds 4 cycle later
    if (instructions.size() && timeline.size() && token_time > state_start_cycle)
    {
        att_wave_state_t state{};
        state.type = this->cur_state;
        state.duration = token_time - state_start_cycle;

        timeline.push_back(state);
    }

    this->cur_state = WaveslotState::WS_EMPTY;
    this->state_start_cycle = token_time;
    this->end_time = token_time;
    bIsComplete = true;
}

inline void mend_inst_time(Instruction& prev, int64_t time)
{
    if (time < prev.time + prev.duration)
    {
        prev.duration = lmax(time - prev.time, 4); // Remove token overlap
        prev.stall = std::min<int>(prev.stall, prev.duration);
    }
}

void wave_t::apply_inst(Token& token, int64_t& CYCLE_START_PHASE)
{
    int inst_type = token.fields.inst.inst_type;

    if (this->trap_status == WaveTrapStatus::TRAP_WAIT_FOR_NEW_PC && inst_type == PC)
    {
        pc_infos.push_back({
            instructions.size(), pcinfo_t{0, 0}
        });
        this->trap_status = WaveTrapStatus::TRAP_RESTORED;

        if (!instructions.empty() && instructions.back().category == WaveInstCategory::TRAP)
            instructions.back().duration = token.time + 4 - instructions.back().time;

        return;
    }

    if (this->trap_status != WaveTrapStatus::TRAP_RESTORED || issued_instructions.empty()) return;

    size_t issued_inst_index = issued_instructions.front();
    issued_instructions.erase(issued_instructions.begin());

    Instruction& the_inst = instructions.at(issued_inst_index);

    if (instructions.size() >= 2)
    {
        auto& prev = instructions.at(instructions.size() - 2);
        if (the_inst.time < prev.time + prev.duration) CYCLE_START_PHASE++;
    }

    if (issued_inst_index > 0)
    {
        auto& prev = instructions.at(issued_inst_index - 1);
        mend_inst_time(prev, the_inst.time);
    }

    int64_t phase = (CYCLE_START_PHASE + simd) % 4;

    const int SALU_PHASE = 5;
    const int VALU_PHASE = 2;
    const int MISC_PHASE = 1;

    if (inst_type < last_inst_for_xnack.size()) last_inst_for_xnack.at(inst_type) = issued_inst_index + 1;

    switch (inst_type)
    {
        case SMEM_RD:
        case SMEM_WR:
            the_inst.category = WaveInstCategory::SMEM;
            phase += SALU_PHASE;
            break;
        case SALU_32:
        case SALU_64:
            the_inst.category = WaveInstCategory::SALU;
            phase += SALU_PHASE;
            break;
        case VMEM_RD:
        case VMEM_WR: the_inst.category = WaveInstCategory::VMEM; break;
        case FLAT_WR:
        case FLAT_RD: the_inst.category = WaveInstCategory::FLAT; break;
        case LDS:
            the_inst.category = WaveInstCategory::LDS;
            phase += MISC_PHASE;
            break;
        case VALU_32:
        case VALU_64:
        case VALU_MAI:
            the_inst.category = WaveInstCategory::VALU;
            phase += VALU_PHASE;
            break;
        case JUMP:
            the_inst.category = WaveInstCategory::JUMP;
            phase += MISC_PHASE;
            break;
        case NEXT:
            the_inst.category = WaveInstCategory::NEXT;
            phase += MISC_PHASE;
            break;
        case PC:
            the_inst.category = WaveInstCategory::SALU;
            pc_infos.push_back({
                issued_inst_index + 1, pcinfo_t{0, 0}
            });
            break;
        case GO_TO_TRAP:
            trap_status = WaveTrapStatus::TRAP_WAIT_FOR_NEW_PC;
            the_inst.category = WaveInstCategory::TRAP;
            isTrapping = true;
            // Move the pcinfos forward if they are pointing to a trap
            for (auto& info : pc_infos)
                if (info.first == instructions.size() - 1) info.first++;
            break;
        case OTHER_MSG:
            the_inst.category = WaveInstCategory::MSG;
            phase += MISC_PHASE;
            break;
        default:
        {
            int entry = -1;
            switch (inst_type)
            {
                case SMEM_RD_REPLAY: entry = SMEM_RD; break;
                case SMEM_WR_REPLAY: entry = SMEM_WR; break;
                case VMEM_RD_REPLAY: entry = VMEM_RD; break;
                case VMEM_WR_REPLAY: entry = VMEM_WR; break;
                case FLAT_RD_REPLAY: entry = FLAT_RD; break;
                case FLAT_WR_REPLAY: entry = FLAT_WR; break;
                default: break;
            }

            if (entry >= 0 && entry < (int) last_inst_for_xnack.size())
            {
                size_t index = last_inst_for_xnack.at(entry);
                last_inst_for_xnack.at(entry) = 0;

                if (index > 0 && index < instructions.size() && index < issued_inst_index)
                {
                    auto& xnack = instructions.at(index - 1);
                    auto duration = xnack.duration;
                    xnack.duration = lmax(xnack.duration, token.time + 4 - xnack.time);
                    xnack.stall += xnack.duration - duration;
                }
            }
            instructions.erase(instructions.begin() + issued_inst_index);
            return;
        }
    }

    the_inst.duration = 4 + lmax(token.time - the_inst.time - 4 * (phase / 4), 0);

    if (instructions.size() && stall_start_times.size() && the_inst.time > *stall_start_times.begin())
    {
        int64_t prev_time = 0;
        if (issued_inst_index > 0)
            prev_time = instructions.at(issued_inst_index - 1).time + instructions.at(issued_inst_index - 1).duration;

        auto stall_start = the_inst.time;
        while (stall_start_times.size() && the_inst.time > *stall_start_times.begin())
        {
            stall_start = lmax(stall_start_times.at(0), prev_time);
            stall_start_times.erase(stall_start_times.begin());
        }

        the_inst.stall = lmax(the_inst.time - stall_start, 0);
        the_inst.duration = lmax(the_inst.time + the_inst.duration - stall_start, 4);
        the_inst.time = stall_start;
    }
}

int64_t MISQTTParser::array_apply_issue(Token& token, WaveArray& SIMD)
{
    int64_t active_issue_cycle = 0;
    auto& issue = token.fields.issue;

    for (uint64_t wave_id = 0; wave_id < 10; wave_id++)
    {
        uint64_t wave_status = issue.inst[wave_id];
        if (wave_status == SQTT_ISSUE_NULL) continue;

        empty_wave_check(SIMD[issue.simd][wave_id].size());
        wave_t& waveid_token = SIMD[issue.simd][wave_id].back();

        active_issue_cycle += waveid_token.apply_issue(wave_status, token.time);
    }
    // number of cycles with instructions issued
    return active_issue_cycle;
}

static uint64_t getGPULocation(const Wave& token) { return token.wave | (token.simd << 5) | (token.cu << 7); }

void MISQTTParser::sqtt_simd_analysis(CppReturnInfo& info, TokenGenerator& _gen, Stitcher& stitch)
{
    auto& generator = static_cast<MITokenGenerator&>(_gen);
    auto& perfEvents = info.perfevents;

    std::vector<att_shader_data_t> shaderdata{};
    std::vector<occupancy_info_t> occupancy{};

    auto send_occupancy = [&]()
    {
        for (auto& occ : occupancy)
            if (occ.pc.code_object_id == 0)
            {
                auto new_pc = csregister.get_wave_start_delayed(occ.pc.address);
                if (new_pc.code_object_id != 0) occ.pc = new_pc;
            }

        stitch.sendOccupancy(occupancy);
    };

    while (generator.valid())
    {
        Token token = generator.next();

        switch (token.type)
        {
            case TOKEN_MISC:
                if (token.fields.misc.misc_type == 2) { info.bPacketLost = true; }
                else if (token.fields.misc.misc_type == 6)
                {
                    for (auto& simd : SIMD)
                        for (auto& slot : simd)
                            if (slot.size() && slot.back().end_time == 0)
                                slot.back().trap_status = WaveTrapStatus::TRAP_REQUEST;
                }
                break;
            case TOKEN_WAVE_START:
            {
                auto& wstart = token.fields.wave;
                pcinfo_t wave_addr = csregister.get_wave_start(wstart);

                if ((int) wstart.cu == target_cu && wstart.sh == 0)
                {
                    if (wstart.count > 64)
                    {
                        if (!SIMD[wstart.simd][wstart.wave].size()) continue;
                        auto& wave = SIMD[wstart.simd][wstart.wave].back();

                        if (wave.trap_status == WaveTrapStatus::TRAP_REQUEST)
                        {
                            wave.contexts = std::min<int>(wave.contexts + 1, 255);
                            wave.trap_status = WaveTrapStatus::TRAP_STANDBY;
                            wave.instructions.push_back({token.time, WaveInstCategory::TRAP, 4, 0});
                            wave.isTrapping = true;
                        }
                        else if (wave.trap_status == WaveTrapStatus::TRAP_STANDBY)
                        {
                            wave.trap_status = WaveTrapStatus::TRAP_WAIT_FOR_NEW_PC;
                        }
                        continue;
                    }

                    auto& wslot = SIMD[wstart.simd][wstart.wave];
                    if (!wslot.size() || wslot.back().end_time != 0)
                        wslot.push_back(wave_t(target_cu, wave_addr, wstart, token.time));
                }

                auto it = running_waves.emplace(getGPULocation(wstart), wave_addr);
                if (it.second)
                    occupancy.push_back({wave_addr, token.time, wstart.cu, wstart.simd, wstart.wave, 1});
                else
                    it.first->second = wave_addr;
                break;
            }
            case TOKEN_WAVE_END:
            {
                auto& wend = token.fields.wave;

                if ((int) wend.cu == target_cu && wend.sh == 0)
                {
                    auto& wslot = SIMD[wend.simd][wend.wave];
                    empty_wave_check(wslot.size());

                    auto& wave = wslot.back();
                    if (wave.trap_status != WaveTrapStatus::TRAP_RESTORED) continue;

                    wave.complete_wave(token.time);
                    wave.lookbackpcs(csregister);
                    stitch.stitch(wave);
                    wslot.pop_back();
                }

                pcinfo_t startpc{};
                auto occ_it = running_waves.find(getGPULocation(wend));
                if (occ_it != running_waves.end())
                {
                    startpc = occ_it->second;
                    running_waves.erase(occ_it);
                    occupancy.push_back({startpc, token.time, wend.cu, wend.simd, wend.wave, 0});
                }
                else if (!info.bPacketLost)
                {
                    occupancy.insert(occupancy.begin(), {startpc, 0, wend.cu, wend.simd, wend.wave, 1});
                    occupancy.push_back({startpc, 0, wend.cu, wend.simd, wend.wave, 0});
                }

                if (double_buffer && occupancy.size() >= MAX_ACCUM_RECORDS) send_occupancy();
                break;
            }
            case TOKEN_INST:
            {
                auto& wslot = SIMD[token.fields.inst.simd][token.fields.inst.wave];
                // Handle lost packet for wave start
                if (wslot.empty() || wslot.back().end_time != 0)
                {
                    Wave wave(0);
                    wave.cu = target_cu;
                    wave.simd = token.fields.inst.simd;
                    wave.wave = token.fields.inst.wave;
                    wslot.push_back(wave_t(target_cu, pcinfo_t{0, 0}, wave, token.time));
                }
                wslot.back().apply_inst(token, CYCLE_START_PHASE);
                break;
            }
            case TOKEN_ISSUE: array_apply_issue(token, SIMD); break;
            case TOKEN_PERF:
                if (token.fields.perf.sh == 0)
                {
                    auto event = att_perfevent_t{};
                    event.time = token.time - 4 * token.fields.perf.cu;
                    event.bank = (uint8_t) token.fields.perf.cntr_bank;
                    event.CU = (uint8_t) token.fields.perf.cu;

                    if (auto interval = csregister.counter_frequency) try
                        {
                            auto& last_time = last_event.at(event.CU).at(event.bank);
                            if (last_time > 0 && event.time > last_time + interval) perfEvents.emplace_back(event);

                            last_time = event.time;
                            last_value.at(event.CU).at(event.bank) = true;
                        }
                        catch (std::exception&)
                        {}

                    event.events0 = (uint16_t) token.fields.perf.cntr[0];
                    event.events1 = (uint16_t) token.fields.perf.cntr[1];
                    event.events2 = (uint16_t) token.fields.perf.cntr[2];
                    event.events3 = (uint16_t) token.fields.perf.cntr[3];
                    perfEvents.emplace_back(event);
                }
                break;
            case TOKEN_INST_PC:
            {
                auto& sm = SIMD[token.fields.inst_pc.simd][token.fields.inst_pc.wave];
                empty_wave_check(sm.size());
                sm.back().apply_pc(token, csregister.table);
                break;
            }
            case TOKEN_REG_CS:
            case TOKEN_REG_CS_PRIV: csregister.UpdateRegCS(token.fields.regcs); break;
            case TOKEN_REG:
                if (csregister.IsUserdata3(token.fields.reg.regaddr))
                    csregister.HandleRealtimeClock(token.time, token.fields.reg.regdata);
                else if (csregister.UpdateRegNoCS(token.fields.reg))
                    convertPerfEventsTs(perfEvents, csregister.counter_frequency);
                break;
            case TOKEN_SHADERDATA:
                shaderdata.push_back(att_shader_data_t{
                    token.time,
                    token.fields.userdata.data,
                    (uint8_t) token.fields.userdata.cu,
                    (uint8_t) token.fields.userdata.simd,
                    (uint8_t) token.fields.userdata.wave,
                    0,
                });

                if (shaderdata.size() >= MAX_ACCUM_RECORDS) stitch.sendShaderdata(shaderdata);
                break;
            default: break;
        }
    }

    for (auto& waveslot : SIMD)
        for (auto& slot : waveslot)
            for (auto& wave : slot) wave.lookbackpcs(csregister);

    if (!occupancy.empty()) send_occupancy();
    if (csregister.realtime.size()) stitch.sendRealtime(csregister.realtime);
    if (shaderdata.size()) stitch.sendShaderdata(shaderdata);

    info.realtime_frequency = csregister.realtime_frequency;
    info.counter_frequency = csregister.counter_frequency;

    if (auto interval = csregister.counter_frequency)
    {
        att_perfevent_t empty_ev{};

        for (int cu = 0; cu < NUM_CU; cu++)
            for (int bank = 0; bank < NUM_BANK; bank++)
            {
                if (last_event.at(cu).at(bank) && last_value.at(cu).at(bank))
                {
                    empty_ev.bank = bank;
                    empty_ev.CU = cu;
                    empty_ev.time = last_event.at(cu).at(bank) + interval;

                    perfEvents.emplace_back(empty_ev);
                }
            }
    }

    for (auto& simd : SIMD)
        for (auto& slot : simd)
            for (auto& wave : slot) stitch.stitch(wave);
}

void wave_t::apply_pc(Token& token, CodeobjTableTranslator& table)
{
    if (!pc_infos.size() || trap_status != WaveTrapStatus::TRAP_RESTORED) return;

    int info_idx = pc_infos.size();
    while (info_idx > 0 && !bValid(pc_infos.at(info_idx - 1).second)) info_idx--;

    if (info_idx == pc_infos.size())
        pc_infos.push_back({
            instructions.size(), pcinfo_t{0, 0}
        });

    auto& back = pc_infos.at(info_idx).second;

    back = table.ToPcV2(token.fields.inst_pc.pc << 2);
    if (back.code_object_id == 0) unattrib_pcs.push_back(info_idx);
}

int64_t wave_t::apply_issue(uint64_t wave_status, int64_t token_time)
{
    if (this->trap_status != WaveTrapStatus::TRAP_RESTORED) return 0;

    auto previous_state = this->cur_state;
    int64_t active_issue_cycle = 0;

    if (wave_status == SQTT_ISSUE_IMMED)
    {
        int64_t immed_time = token_time;

        if (instructions.size() && instructions.back().category != WaveInstCategory::WAVE_NOT_FINISHED)
        {
            auto& inst = instructions.back();
            mend_inst_time(inst, token_time);
            immed_time = std::min<int64_t>(inst.time + inst.duration, token_time);
        }

        int64_t cycles_time = 4 + lmax(0, token_time - immed_time);
        instructions.push_back({immed_time, WaveInstCategory::IMMED, (int) cycles_time, (int) cycles_time});

        if (state_start_cycle < immed_time)
        {
            if (timeline.size() && this->cur_state == timeline.back().type)
                timeline.back().duration += immed_time - state_start_cycle;
            else
                timeline.push_back({this->cur_state, int(immed_time - state_start_cycle)});
        }

        this->cur_state = WaveslotState::WS_EXEC;

        timeline.push_back({WaveslotState::WS_WAIT, (int) cycles_time});
        state_start_cycle = immed_time + cycles_time;
    }
    else if (wave_status == SQTT_ISSUE_STALL)
    {
        this->cur_state = WaveslotState::WS_STALL;
        this->stall_start_times.push_back(token_time);
    }
    else if (wave_status == SQTT_ISSUE_INST)
    {
        active_issue_cycle = 1;
        this->cur_state = WaveslotState::WS_EXEC;
        issued_instructions.push_back(instructions.size());
        instructions.push_back({token_time, WaveInstCategory::WAVE_NOT_FINISHED, 0, 0});
    }

    int64_t state_duration = token_time - state_start_cycle;
    if (state_duration > 0)
    {
        if (timeline.size() && timeline.back().type == previous_state)
            timeline.back().duration += state_duration;
        else
            this->timeline.push_back({previous_state, (int) state_duration});
    }

    this->state_start_cycle = lmax(token_time, state_start_cycle);
    return active_issue_cycle;
}

void CSRegisterHandlerGFX9::HandleRealtimeClock(size_t time, size_t data)
{
    if (!bIsROCMFormat) return;

    if (userdata3_count == 0)
    {
        if (data == 0x5) userdata3_count = 3;
        return;
    }

    userdata3_count--;
    userdata3_values[userdata3_count] = data;

    if (userdata3_count > 0) return;

    att_decoder_realtime_t rt{};
    rt.shader_clock = time;
    rt.realtime_clock = userdata3_values[RT_DELTA] | (userdata3_values[RT_HI] << 32);

    // handle wrapping of lowest 32 bits
    if (userdata3_values[RT_DELTA] < userdata3_values[RT_LOW]) rt.realtime_clock += 1ul << 32;

    realtime.push_back(rt);
}

void convertPerfEventsTs(std::vector<att_perfevent_t>& vec, int64_t interval)
{
    try
    {
        if (interval == 0 || vec.empty()) return;

        std::vector<att_perfevent_t> additional{};
        std::array<std::array<int64_t, MISQTTParser::NUM_BANK>, MISQTTParser::NUM_CU> last_event{};

        att_perfevent_t empty_ev{};

        for (const auto& event : vec)
        {
            auto& last_time = last_event.at(event.CU).at(event.bank);
            if (last_time > 0 && event.time > last_time + interval)
            {
                empty_ev.bank = event.bank;
                empty_ev.CU = event.CU;
                empty_ev.time = last_time + interval;
                additional.emplace_back(empty_ev);
            }
            last_time = event.time;
        }

        if (!additional.empty())
        {
            vec.insert(vec.end(), additional.begin(), additional.end());
            std::stable_sort(
                vec.begin(),
                vec.end(),
                [](const att_perfevent_t& a, const att_perfevent_t& b) { return a.time < b.time; }
            );
        }
    }
    catch (std::exception& e)
    {}
}

} // namespace gfx9