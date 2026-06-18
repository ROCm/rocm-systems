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
#include "mi400/mi400token.h"
#include "mi400/mi400wave.h"
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

    auto consume_launch_cluster = [&]() -> uint8_t
    {
        auto cluster_id = active_cluster_id;
        if (cluster_end_pending)
        {
            active_cluster_id = 0;
            cluster_end_pending = false;
        }
        return cluster_id;
    };

    while (generator.nextValid())
    {
        Token token = generator.next();

        switch (token.type)
        {
            case RdnaType::MISC_GFX10:
            {
                gfx10::misc_fields fields{};

                auto generate_event = [&](rocprofiler_thread_trace_decoder_event_type_t type, uint64_t payload = 0)
                { stitch.sendEvent(type, token.time, 0, 0, payload, false, false); };

                if (tt_version >= 5)
                {
                    mi400::misc_type misc{.raw = token.contents};
                    DEBUGPRINT(misc);
                    fields.raw = (uint8_t) misc.fields;
                    fields.tt5_shift();

                    if (misc.CLF)
                    {
                        active_cluster_id = static_cast<uint8_t>(misc.CLID);
                        cluster_end_pending = false;
                    }
                    if (misc.CLL)
                    {
                        active_cluster_id = static_cast<uint8_t>(misc.CLID);
                        cluster_end_pending = true;
                    }
                }
                else
                {
                    gfx10::misc_type misc{.raw = token.contents};
                    DEBUGPRINT(misc);
                    fields.raw = (uint8_t) misc.fields;
                }

                if (tt_version <= 2 && fields.spm_or_pl == 1)
                {
                    fields.spm_or_pl = 0;
                    generate_event(ROCPROF_TRACE_DECODER_EVENT_PACKET_LOSS);
                    info.bPacketLost = true;
                }

                if (fields.save_context)
                {
                    generate_event(ROCPROF_TRACE_DECODER_EVENT_SAVE_CONTEXT);

                    for (auto& slot : SIMD)
                        if (slot.size())
                        {
                            auto& back = slot.back();
                            if (back.trap_status == WaveTrapStatus::TRAP_RESTORED)
                                back.trap_status = WaveTrapStatus::TRAP_STANDBY;
                        }
                }

                if (fields.tt_stall_start) generate_event(ROCPROF_TRACE_DECODER_EVENT_TT_STALL_BEGIN);
                if (fields.tt_stall_end) generate_event(ROCPROF_TRACE_DECODER_EVENT_TT_STALL_END);
                if (fields.DIDT_stall_start) generate_event(ROCPROF_TRACE_DECODER_EVENT_DIDT_STALL_BEGIN);
                if (fields.DIDT_stall_end) generate_event(ROCPROF_TRACE_DECODER_EVENT_DIDT_STALL_END);
                if (fields.gc_rinse) generate_event(ROCPROF_TRACE_DECODER_EVENT_GC_RINSE);
                if (fields.spm_or_pl) generate_event(ROCPROF_TRACE_DECODER_EVENT_SPM_SAMPLE);

                break;
            }
            case RdnaType::HEADER:
            {
                tt_version = header_type{.raw = token.contents}.version;

                if (tt_version >= 4) double_buffer = (token.contents >> 43) & 1;

                if (tt_version >= 5)
                {
                    mi400::header_type header{.raw = token.contents};
                    DEBUGPRINT(header);
                    target_sa_wgp = get_sa_wgp(header.DSA, header.DWGP);
                    target_simd = header.DSIMD;
                    derate = header.trans2;
                }
                else
                {
                    header_type header{.raw = token.contents};
                    DEBUGPRINT(header);
                    target_sa_wgp = get_sa_wgp(header.DSA, header.DWGP);
                    target_simd = header.DSIMD;

                    int mask = tt_version < 3 ? 0xF : 7;
                    int dpr = header.DPRate & mask;
                    dprate = dpr ? (1 << (dpr - 1)) : 1;
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
                                    target_sa_wgp,
                                    start.simd,
                                    start.wid,
                                    pcinfo_t{0, 0},
                                    token,
                                    exclude_barrier_wait,
                                    (uint8_t) start.me,
                                    (uint8_t) start.pipe,
                                    (uint8_t) start.wgid
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
                            occupancy.push_back(
                                {it->second,
                                 token.time,
                                 start.SACU(),
                                 start.simd,
                                 start.wid,
                                 0,
                                 (uint64_t) start.me,
                                 (uint64_t) start.pipe,
                                 start.isExt,
                                 (uint64_t) start.wgid}
                            );
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
                        occupancy.push_back(
                            {addr,
                             token.time,
                             start.SACU(),
                             start.simd,
                             start.wid,
                             1,
                             (uint64_t) start.me,
                             (uint64_t) start.pipe,
                             start.isExt,
                             (uint64_t) start.wgid}
                        );
                    }

                    break;
                }

                if (generator.packetlost && running_waves.find(start.getGPULocation()) != running_waves.end())
                {
                    occupancy.push_back({
                        {0, 0},
                        token.time - 1, start.SACU(), start.simd, start.wid, 0, 0, 0, 0, 0
                    });
                }
                auto cluster_id = consume_launch_cluster();
                running_waves[start.getGPULocation()] = wave_addr;

                occupancy.push_back(
                    {wave_addr,
                     token.time,
                     start.SACU(),
                     start.simd,
                     start.wid,
                     1,
                     (uint64_t) start.me,
                     (uint64_t) start.pipe,
                     start.isExt,
                     (uint64_t) start.wgid,
                     cluster_id}
                );
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
                    SIMD[start.wid].push_back(wave_t(
                        target_sa_wgp,
                        start.simd,
                        start.wid,
                        wave_addr,
                        token,
                        exclude_barrier_wait,
                        (uint8_t) start.me,
                        (uint8_t) start.pipe,
                        (uint8_t) start.wgid,
                        cluster_id
                    ));
                }
                break;
            }
            case RdnaType::WAVE_END:
            {
                wend_type_common end;
                if (tt_version >= 5)
                    end = mi400::wend_type{.raw = token.contents}.get();
                else if (tt_version == 4)
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
                    occupancy.insert(occupancy.begin(), {startpc, 0, end.SACU(), end.simd, end.wid, 1, 0, 0, 0, 0});

                occupancy.push_back({startpc, token.time, end.SACU(), end.simd, end.wid, 0, 0, 0, 0, 0});
                break;
            }
            case RdnaType::INST:
            {
                inst_type_common inst;
                auto mapped = mapped_inst_t{WaveInstCategory::NONE, 0};

                if (tt_version >= 5)
                {
                    inst = mi400::inst_type{.raw = token.contents}.get();
                    try
                    {
                        mapped = mi400::map_to_common_type(inst.inst, derate);
                    }
                    catch (std::exception&)
                    {
                        auto& simd = SIMD[inst.wid];
                        if (simd.size()) mi400::handle_xnack_rewind(simd.back());
                        break;
                    }
                }
                else if (tt_version == 4)
                {
                    inst = gfx12::inst_type{.raw = token.contents}.get();
                    mapped = gfx12::map_to_common_type(inst.inst, dprate, derate);
                }
                else
                {
                    inst = gfx10::inst_type{.raw = token.contents}.get();
                    if (tt_version == 3)
                        mapped = gfx11::map_to_common_type(inst.inst, dprate, derate);
                    else
                        mapped = gfx10::map_to_common_type(inst.inst, dprate, derate);
                }
                DEBUGPRINT(inst);

                if (mapped.category & OTHER_SIMD_BIT)
                {
                    other_simd.push_back(
                        {sizeof(rocprofiler_thread_trace_decoder_inst_other_simd_t),
                         token.time,
                         (uint16_t) mapped.cycles,
                         (uint8_t) target_sa_wgp,
                         (uint8_t) (mapped.category ^ OTHER_SIMD_BIT)}
                    );
                    if (other_simd.size() >= MAX_ACCUM_RECORDS) stitch.sendOtherSimd(other_simd);
                }
                else
                {
                    auto& simd = SIMD[inst.wid];
                    empty_wave_check(simd.size());
                    auto& wave = simd.back();

                    if (mapped.category == WaveInstCategory::LD_SCALE)
                    {
                        wave.next_ld_scale = true;
                        break;
                    }

                    int64_t inst_time = token.time;
                    if (wave.next_ld_scale)
                    {
                        wave.next_ld_scale = false;
                        inst_time -= 1;
                        mapped.cycles++;
                    }

                    wave.apply_inst(inst_time, inst.inst, mapped, tt_version);
                }
                break;
            }
            case RdnaType::VALU_INST:
            {
                valu_inst_type vinst{.raw = token.contents};
                DEBUGPRINT(vinst);
                auto& simd = SIMD[vinst.wid];
                empty_wave_check(simd.size());
                simd.back().apply_valu_inst(token.time);
                break;
            }
            case RdnaType::IMM_ONE:
            {
                int wid;
                if (tt_version >= 5)
                {
                    mi400::immed_one_type immed_one{.raw = token.contents};
                    DEBUGPRINT(immed_one);
                    wid = immed_one.wid;
                }
                else
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
                gfx10::reg_write_type reg{.raw = token.contents};
                DEBUGPRINT(reg);

                if (reg.CS) csregister.UpdateRegCS(reg);

                // gfx10 gets userdata in reg.cs
                if (!reg.CS || tt_version <= 2)
                {
                    auto ev = csregister.UpdateRegNoCS(reg);
                    switch (ev.kind)
                    {
                        case CSRegisterHandler::RegUpdateEvent::COUNTER_FREQUENCY_CHANGED: break;
                        case CSRegisterHandler::RegUpdateEvent::CODEOBJ_LOAD:
                            stitch.sendEvent(
                                ROCPROF_TRACE_DECODER_EVENT_CODE_OBJECT_LOAD,
                                token.time,
                                reg.me,
                                reg.pipe,
                                static_cast<uint32_t>(ev.id),
                                false,
                                false
                            );
                            break;
                        case CSRegisterHandler::RegUpdateEvent::CODEOBJ_UNLOAD:
                            stitch.sendEvent(
                                ROCPROF_TRACE_DECODER_EVENT_CODE_OBJECT_UNLOAD,
                                token.time,
                                reg.me,
                                reg.pipe,
                                static_cast<uint32_t>(ev.id),
                                false,
                                false
                            );
                            break;
                        case CSRegisterHandler::RegUpdateEvent::NONE: break;
                    }
                }

                break;
            }
            case RdnaType::REG_INIT:
            {
                gfx10::reg_init_type reg{.raw = token.contents};
                if (reg.type == 2 && (reg.data & 1) == 1) stitch.sendDispatch(csregister, token.time, reg.me, reg.pipe);
                break;
            }
            case RdnaType::EVENT:
            {
                gfx10::event_type event{.raw = token.contents};

                rocprofiler_thread_trace_decoder_event_type_t type{};

                switch (event.id)
                {
                    case EVENT_CS_PARTIAL_FLUSH: type = ROCPROF_TRACE_DECODER_EVENT_CS_PARTIAL_FLUSH; break;
                    case EVENT_CACHE_FLUSH:
                    case EVENT_CACHE_FLUSH_WR:
                    case EVENT_CACHE_FLUSH_INV:
                    case EVENT_CACHE_FLUSH_INV_WR: type = ROCPROF_TRACE_DECODER_EVENT_CACHE_FLUSH; break;
                    case EVENT_BOTTOM_OF_PIPE_WR: type = ROCPROF_TRACE_DECODER_EVENT_BOTTOM_OF_PIPE_TS; break;
                    case EVENT_TT_FLUSH: type = ROCPROF_TRACE_DECODER_EVENT_TT_FLUSH; break;
                    default: break;
                }

                if (type != ROCPROF_TRACE_DECODER_EVENT_NONE)
                    stitch.sendEvent(type, token.time, event.me, event.pipe, 0, event.bop, true);

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
                if (generator.packetlost)
                {
                    info.bPacketLost = true;
                    stitch.sendEvent(ROCPROF_TRACE_DECODER_EVENT_PACKET_LOSS, token.time, 0, 0, 0, false, false);
                    generator.packetlost = false;
                }
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
