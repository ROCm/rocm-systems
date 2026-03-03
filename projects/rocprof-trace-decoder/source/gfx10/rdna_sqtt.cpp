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

#include <algorithm>
#include <cassert>
#include <utility>
#include <vector>
#ifdef SQTT_LOGGING
#    include <iostream>
#endif
#include "gfx10wave.h"
#include "gfx11/gfx11wave.h"
#include "gfx12/gfx12wave.h"
#include "segment.hpp"
#include "stitch/stitch.hpp"

typedef gfx10::wave_t wave_t;
typedef gfx10::Token Token;

#define MAX_ACCUM_RECORDS 65536

#ifdef SQTT_LOGGING
#    define DEBUGPRINT(_decoded)                                                                                       \
        do {                                                                                                           \
            auto _d = (_decoded);                                                                                      \
            std::cout << token.time << " " << _d.typestr() << " - " << _d.print().str() << std::endl;                  \
        }                                                                                                              \
        while (0)
#else
#    define DEBUGPRINT(_decoded)
#endif

#define empty_wave_check(waveslot_size)                                                                                \
    if (waveslot_size == 0) { continue; }

std::pair<WaveInstCategory, uint16_t> get_other_simd(int einst, int tt_version)
{
    if (tt_version == 3)
        return gfx11::wave_t::get_other_simd(einst);
    else if (tt_version == 4)
        return gfx12::wave_t::get_other_simd(einst);
    else
        return {};
}

void RDNASQTParser::sqtt_simd_analysis(CppReturnInfo& info, TokenGenerator& _gen, Stitcher& stitch)
{
    auto& generator = static_cast<NaviTokenGenerator&>(_gen);

    std::vector<att_shader_data_t> shaderdata{};
    std::vector<rocprofiler_thread_trace_decoder_inst_other_simd_t> other_simd{};
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

    while (generator.nextValid())
    {
        Token token = generator.next();

        switch (token.type)
        {
            case RdnaType::MISC_GFX10:
            {
                bool bCtxSave = false;
                {
                    gfx10::misc_type misc{.raw = token.contents};
                    DEBUGPRINT(misc);
                    bCtxSave = misc.save_context;
                    info.bPacketLost |= misc.spm_or_pl == 1 && tt_version <= 2;
                }

                if (bCtxSave)
                {
                    for (auto& slot : SIMD)
                        if (slot.size())
                        {
                            auto& back = slot.back();
                            if (back.trap_status == WaveTrapStatus::TRAP_RESTORED)
                                back.trap_status = WaveTrapStatus::TRAP_STANDBY;
                        }
                }
                break;
            }
            case RdnaType::HEADER:
            {
                tt_version = header_type{.raw = token.contents}.version;

                if (tt_version >= 4) double_buffer = (token.contents >> 43) & 1;

                {
                    header_type header{.raw = token.contents};
                    DEBUGPRINT(header);
                    target_sa_wgp = get_sa_wgp(header.DSA, header.DWGP);
                    target_simd = header.DSIMD;

                    int mask = tt_version < 3 ? 0xF : 7;
                    dprate = 1 << ((header.DPRate & mask) - 1);
                    derate = 1 << (header.dp_derate);
                    if (tt_version == 4) exclude_barrier_wait = (token.contents >> 59) & 1;
                }
                break;
            }
            case RdnaType::WAVE_START_EXT:
            case RdnaType::WAVE_START:
            {
                wstart_type_common start;
                if (tt_version >= 4)
                    start = gfx12::wstart_type{.raw = token.contents}.get();
                else
                    start = gfx10::wstart_type{.raw = token.contents}.get();
                DEBUGPRINT(start);

                pcinfo_t wave_addr = csregister.get_wave_start(start);
                bool bIsTarget = start.SACU() == target_sa_wgp && start.simd == target_simd;

                constexpr uint64_t CTX_BEGIN = 66;
                constexpr uint64_t CTX_END = 90;

                if (start.count >= CTX_BEGIN && start.count < CTX_END)
                {
                    // We are in context save or restore
                    constexpr uint64_t ctx_save_array = 0b0000110011;
                    constexpr uint64_t ctx_resr_array = 0b1111001100;

                    bool isCtxSave = (ctx_save_array >> (start.count - CTX_BEGIN)) & 1;
                    bool isCtxResr = (ctx_resr_array >> (start.count - CTX_BEGIN)) & 1;

                    if (bIsTarget)
                    {
                        auto& slot = SIMD[start.wid];

                        // Only save started waves
                        if (isCtxSave && !slot.empty())
                        {
                            auto& wave = slot.back();
                            wave.trap_status = WaveTrapStatus::TRAP_STANDBY;
                            wave.contexts++;
                            wave.instructions.push_back({token.time, WaveInstCategory::TRAP, 1, 0});
                        }
                        // Context restore can 'start new wave' if not existent
                        if (isCtxResr)
                        {
                            if (slot.empty() || slot.back().bIsComplete)
                                slot.push_back(wave_t(
                                    target_sa_wgp, start.simd, start.wid, pcinfo_t{0, 0}, token, exclude_barrier_wait
                                ));

                            auto& wave = slot.back();

                            if (!wave.instructions.empty())
                            {
                                auto& back = wave.instructions.back();
                                if (back.category == WaveInstCategory::TRAP)
                                    back.duration = std::max<int64_t>(back.duration, token.time + 1 - back.time);
                            }

                            wave.trap_status = WaveTrapStatus::TRAP_STANDBY;
                            wave.contexts++;
                        }
                    }

                    if (isCtxSave)
                    {
                        auto it = running_waves.find(start.getGPULocation());
                        if (it != running_waves.end())
                        {
                            occupancy.push_back({it->second, token.time, start.SACU(), start.simd, start.wid, 0});
                            saved_waves[start.getGPULocation()] = it->second;
                        }
                    }
                    else if (isCtxResr)
                    {
                        pcinfo_t addr{0, 0};
                        if (auto it = saved_waves.find(start.getGPULocation()); it != saved_waves.end())
                            addr = it->second;

                        saved_waves.erase(start.getGPULocation());
                        running_waves[start.getGPULocation()] = addr;
                        occupancy.push_back({addr, token.time, start.SACU(), start.simd, start.wid, 1});
                    }

                    break;
                }

                running_waves[start.getGPULocation()] = wave_addr;

                occupancy.push_back({wave_addr, token.time, start.SACU(), start.simd, start.wid, 1});
                if (double_buffer && occupancy.size() >= MAX_ACCUM_RECORDS) send_occupancy();

                if (bIsTarget)
                {
                    while (SIMD[start.wid].size() && SIMD[start.wid].back().bIsComplete)
                    {
                        auto& wave = SIMD[start.wid].back();
                        wave.lookbackpcs(csregister);
                        stitch.stitch(wave);
                        SIMD[start.wid].pop_back();
                    }
                    SIMD[start.wid].push_back(
                        wave_t(target_sa_wgp, start.simd, start.wid, wave_addr, token, exclude_barrier_wait)
                    );
                }
                break;
            }
            case RdnaType::WAVE_END:
            {
                wend_type_common end;
                if (tt_version == 4)
                    end = gfx12::wend_type{.raw = token.contents}.get();
                else
                    end = gfx10::wend_type{.raw = token.contents}.get();
                DEBUGPRINT(end);

                // We dont wanna stitch here because NEW_PC can arrive after WAVE_END
                if (end.SACU() == target_sa_wgp && end.simd == target_simd && SIMD[end.wid].size())
                    SIMD[end.wid].back().complete_wave(token);

                pcinfo_t startpc{};
                if (running_waves.find(end.getGPULocation()) != running_waves.end())
                {
                    num_waves_completed += 1;
                    startpc = running_waves[end.getGPULocation()];
                    running_waves.erase(end.getGPULocation());
                }
                else
                    occupancy.insert(occupancy.begin(), {startpc, 0, end.SACU(), end.simd, end.wid, 1});

                occupancy.push_back({startpc, token.time, end.SACU(), end.simd, end.wid, 0});
                break;
            }
            case RdnaType::INST:
            {
                inst_type_common inst;
                if (tt_version == 4)
                    inst = gfx12::inst_type{.raw = token.contents}.get();
                else
                    inst = gfx10::inst_type{.raw = token.contents}.get();
                DEBUGPRINT(inst);

                if (auto other = get_other_simd(inst.inst, tt_version); other.first != WaveInstCategory::NONE)
                {
                    other_simd.push_back(
                        {sizeof(rocprofiler_thread_trace_decoder_inst_other_simd_t),
                         token.time,
                         (uint16_t) other.second,
                         (uint8_t) target_sa_wgp,
                         (uint8_t) other.first}
                    );
                    if (other_simd.size() >= MAX_ACCUM_RECORDS) stitch.sendOtherSimd(other_simd);
                }
                else
                {
                    auto& simd = SIMD[inst.wid];
                    empty_wave_check(simd.size());
                    simd.back().apply_inst(token, inst, tt_version, dprate, derate);
                }
                break;
            }
            case RdnaType::VALU_INST:
            {
                valu_inst_type vinst{.raw = token.contents};
                DEBUGPRINT(vinst);
                auto& simd = SIMD[vinst.wid];
                empty_wave_check(simd.size());
                simd.back().apply_valu_inst(token, vinst.w64h);
                break;
            }
            case RdnaType::IMM_ONE:
            {
                int wid;
                {
                    immed_one_type immed_one{.raw = token.contents};
                    DEBUGPRINT(immed_one);
                    wid = immed_one.wid;
                }
                empty_wave_check(SIMD[wid].size());
                SIMD[wid].back().apply_immediate(token.time);
                break;
            }
            case RdnaType::IMMEDIATE:
            {
                immediate_type immed{.raw = token.contents};
                DEBUGPRINT(immed);
                for (int i = 0; i < 16; i++)
                    if (SIMD[i].size() && (immed.waves & (1 << i))) SIMD[i].back().apply_immediate(token.time);
                break;
            }
            case RdnaType::NEW_PC_GFX10:
            {
                gfx10::new_pc_type pc{.raw = token.contents};
                DEBUGPRINT(pc);
                if (pc.wave < SIMD.size() && SIMD[pc.wave].size())
                    SIMD[pc.wave].back().new_pc((uint64_t) token.time, pc.pc, csregister.table);
                break;
            }
            case RdnaType::NEW_PC_GFX12:
            {
                gfx12::new_pc_type pc{.raw = token.contents};
                DEBUGPRINT(pc);
                if (pc.wave < SIMD.size() && SIMD[pc.wave].size())
                    SIMD[pc.wave].back().new_pc((uint64_t) token.time, pc.pc, csregister.table);
                break;
            }
            case RdnaType::REG:
            {
                reg_write_type reg{.raw = token.contents};
                DEBUGPRINT(reg);

                if (reg.CS) csregister.UpdateRegCS(reg);
                // gfx10 gets userdata in reg.cs
                if (!reg.CS || tt_version <= 2) csregister.UpdateRegNoCS(reg);
                break;
            }
            case RdnaType::WAVE_READY:
            {
                wave_ready_type rdy{.raw = token.contents};
                DEBUGPRINT(rdy);
                for (int i = 0; i < 16; i++)
                    if (((rdy.waves >> i) & 1) && SIMD[i].size()) SIMD[i].back().apply_wave_rdy(token.time);
                break;
            }
            case RdnaType::SHADER_DATA:
            case RdnaType::SHADER_DATA_SHORT:
            {
                att_shader_data_t shader{};
                shader.time = token.time;

                shader_data_common_type common{};
                if (tt_version >= 4)
                {
                    if (token.type == RdnaType::SHADER_DATA_SHORT)
                        common = gfx12::shader_data_short_type{.raw = token.contents}.get();
                    else
                        common = gfx12::shader_data_type{.raw = token.contents}.get();
                }
                else
                {
                    if (token.type == RdnaType::SHADER_DATA_SHORT)
                        common = gfx11::shader_data_short_type{.raw = token.contents}.get();
                    else
                        common = gfx11::shader_data_type{.raw = token.contents}.get();
                }
                DEBUGPRINT(common);

                if (common.invalid && tt_version > 2) continue;
                shader.cu = common.cu;
                shader.simd = common.simd;
                shader.wave_id = common.wave;
                shader.flags = (common.priv != 0) << ROCPROFILER_THREAD_TRACE_DECODER_SHADERDATA_FLAGS_PRIV;
                shader.flags |= (common.isshort != 0) << ROCPROFILER_THREAD_TRACE_DECODER_SHADERDATA_FLAGS_IMM;
                shader.value = common.data;

                shaderdata.emplace_back(shader);
                if (shaderdata.size() >= MAX_ACCUM_RECORDS) stitch.sendShaderdata(shaderdata);

                break;
            }
            default:
                if (generator.realtime.size() >= MAX_ACCUM_RECORDS) stitch.sendRealtime(generator.realtime);
                break;
        }
    }

    for (auto& slot : SIMD)
        for (auto& wave : slot) wave.lookbackpcs(csregister);

    if (!occupancy.empty()) send_occupancy();

    info.realtime_frequency = csregister.realtime_frequency;
    info.counter_frequency = csregister.counter_frequency;
    info.bPacketLost |= generator.packetlost;

    if (generator.realtime.size()) stitch.sendRealtime(generator.realtime);
    if (shaderdata.size()) stitch.sendShaderdata(shaderdata);
    if (other_simd.size()) stitch.sendOtherSimd(other_simd);

    info.globaltime = generator.get_time();
    info.basetime = generator.get_base_time();

    for (auto& slot : SIMD)
        for (auto& wave : slot) stitch.stitch(wave);
}
