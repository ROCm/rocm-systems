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

#include <string>
#include <string_view>
#include <vector>

#include "stitch/stitch.hpp"
#include "trace_parser.hpp"
#include "trie.h"

#define MAX_FAILED_STTICHES 1000

inline bool skippable(InstCategory line)
{
    return line == InstCategory::VALU || line == InstCategory::VMEM || line == InstCategory::FLAT ||
           line == InstCategory::IMMED || line == InstCategory::LDS;
};

inline bool is_trivial_match(int wave, InstCategory line)
{
    if (line == wave)
        return true;
    else if (line == InstCategory::BRANCH)
        return wave == WaveInstCategory::NEXT;
    else if (wave == WaveInstCategory::MSG)
        return line == InstCategory::IMMED;
    else if (wave == WaveInstCategory::FLAT)
        return line == InstCategory::VMEM;

    return false;
}

inline bool is_elaborate_match(int wave, InstCategory line)
{
    if (is_trivial_match(wave, line)) return true;

    if (line == InstCategory::BRANCH)
        return wave == WaveInstCategory::JUMP;
    else if (wave == WaveInstCategory::IMMED)
        return skippable(line);

    return false;
}

std::pair<size_t, barrier_list_t> Stitcher::stitchWave(class WaveDataInternal& wave)
{
    auto& insts = wave.instructions;

    int num_failed_stitches = 0;
    int inst_index = 0;
    int skipped_stitchs = 0;
    int skipped_immed = 0;
    int pc_info_index = 0;

    assemblyLinePtr line = nullptr;
    assemblyLinePtr next = nullptr;

    barrier_list_t barrier_gap{};

    while (inst_index < insts.size() && num_failed_stitches < MAX_FAILED_STTICHES)
    {
        while (pc_info_index < wave.pc_infos.size() && wave.pc_infos.at(pc_info_index).first <= inst_index)
        {
            if (wave.pc_infos.at(pc_info_index).first == inst_index)
            {
                auto pc = wave.pc_infos.at(pc_info_index).second;
                try
                {
                    auto result = pctranslator->getcode(pc);
                    if (!result) throw std::exception();

                    insts.at(inst_index).pc = wave.pc_infos.at(pc_info_index).second;
                }
                catch (const std::exception&)
                {
                    if (pc_info_index + 1 < wave.pc_infos.size())
                    {
                        auto next_inst = wave.pc_infos.at(pc_info_index + 1).first;
                        skipped_stitchs += next_inst - inst_index;
                        inst_index = next_inst;
                        if (inst_index >= insts.size()) return {inst_index, barrier_gap};
                    }
                }
            }
            pc_info_index++;
        }

        Instruction& inst = insts.at(inst_index);

        if (inst.category == WaveInstCategory::TRAP)
        {
            inst_index++;
            continue;
        }
#ifndef ARCH_MODEL
        else if (bValid(inst.pc))
#else
        else if (bValid(inst.pc) || inst_index == 0)
#endif
        {
            inst_index++;
            next = nullptr;
            try
            {
                line = pctranslator->getcode(inst.pc);
                if (line) try
                    {
                        while (line && bValid(line->next) && line->cat == InstCategory::SKIP)
                        {
                            inst.pc = line->next;
                            line = pctranslator->getcode(line->next);
                        }

                        // Edge case: First instruction is s_barrier
                        if (line->cat == InstCategory::GFX9_BARRIER && inst_index < insts.size())
                        {
                            insts.at(inst_index).pc = inst.pc;
                            inst_index++;
                        }

                        if (line->cat == InstCategory::BRANCH && inst.category == WaveInstCategory::JUMP)
                            next = pctranslator->jump(*line);
                        else
                            next = pctranslator->getcode(line->next);
                    }
                    catch (...)
                    {}
            }
            catch (...)
            {
                return {0, barrier_gap};
            }
            continue;
        }

        line = std::move(next);
        next = nullptr;

        if (!line || inst.category == WaveInstCategory::WAVE_NOT_FINISHED) break;

        try
        {
            next = pctranslator->getcode(line->next);
        }
        catch (...)
        {}

        if (line->cat == InstCategory::DONT_KNOW || line->cat == InstCategory::SKIP) continue;

        bool bMatched = true;

        if (wave.exclude_barrier_wait &&
            (line->cat == InstCategory::BARRIER_WAIT_EXCLUDED ||
             (!line->parsed && line->cat == InstCategory::IMMED && line->line.find("s_barrier_wait") == 0)))
        {
            // If barrier wait was excluded, it can send an IMMED or send nothing. Look ahead to find which.
            bool addEntry = true;
            if (inst.category == WaveInstCategory::IMMED) try
                {
                    auto type = 0;
                    auto lookahead = next;
                    // Apply ahead and if we dont match, then we should not exclude this
                    for (size_t idx = inst_index; idx < wave.instructions.size() && addEntry && lookahead; idx++)
                    {
                        auto line2cat = lookahead->cat;
                        auto inst2cat = wave.instructions.at(idx).category;

                        if (line2cat == InstCategory::BRANCH && inst2cat == WaveInstCategory::JUMP)
                            lookahead = pctranslator->jump(*lookahead);
                        else
                            lookahead = pctranslator->getcode(lookahead->next);

                        if (line2cat == InstCategory::DONT_KNOW || line2cat == InstCategory::SKIP) continue;

                        addEntry = is_elaborate_match(inst2cat, line2cat);
                        if (inst2cat != WaveInstCategory::IMMED && line2cat != InstCategory::VALU)
                        {
                            if (type != 0 && type != (int) line2cat) break;
                            type = (int) line2cat;
                        }
                        idx++;
                    }
                }
                catch (std::exception& e)
                {}

            if (addEntry)
            {
                barrier_gap.push_back({inst_index, line->addr});
                line->cat = InstCategory::BARRIER_WAIT_EXCLUDED;
                continue;
            }
        }

        if (is_trivial_match(inst.category, line->cat))
        {
            // line->print();
        }
        else if (line->cat == InstCategory::BRANCH)
        {
            if (inst.category != WaveInstCategory::JUMP) break;
            next = pctranslator->jump(*line);
        }
        else if (gfxip == 9 && line->cat == InstCategory::MFMA_SCALE && inst.category == WaveInstCategory::VALU)
        {
            // MFMA_SCALE requires manual intervation for the scale part
            int64_t prev_min_time = wave.begin_time;
            int64_t next_min_time = wave.end_time;
            if (inst_index > 0) prev_min_time = insts.at(inst_index - 1).time + insts.at(inst_index - 1).duration;
            if (inst_index < insts.size() - 1) next_min_time = insts.at(inst_index + 1).time;

            // LD_SCALE part should start 4 cycles earlier and end at same point
            inst.duration += 4;
            if (inst.time > prev_min_time)
            {
                inst.time -= 4;
                // LD_SCALE tends to have a 4-cycle gap with LDS, so we add another 4 cycles
                if (inst.time + inst.duration < next_min_time) inst.duration += 4;
            }
            else if (inst.time + inst.duration > next_min_time)
            {
                // If we cant fit in the trace, then dont increase duration
                inst.duration -= 4;
            }
        }
        else if (gfxip == 9 && line->cat == InstCategory::GFX9_BARRIER)
        {
            // GFX9 Barriers can emit 2 tokens: MSG for entry and IMMED for exit.
            if (inst.category == WaveInstCategory::MSG)
            {
                inst_index++;
                if (inst_index < insts.size() && insts.at(inst_index).category == WaveInstCategory::IMMED)
                    insts.at(inst_index).pc = line->addr;
            }
            else
                bMatched = inst.category == WaveInstCategory::IMMED;
        }
        else
        {
            bMatched = false;
            if (gfxip == 9 && insts.size() > inst_index + 1 &&
                pctranslator->try_match_swapped(inst, insts.at(inst_index + 1), *line))
            {
                Instruction temp = insts.at(inst_index);
                insts.at(inst_index) = insts.at(inst_index + 1);
                insts.at(inst_index + 1) = temp;
                next = line;
            }
            else if (gfxip != 9 && inst.category == WaveInstCategory::IMMED && skippable(line->cat))
            {
                bMatched = true;
                skipped_immed++;
            }
            else if (line->line.find("s_waitcnt") == 0 || line->line.find("_load_") != std::string::npos)
            {
                if (skipped_immed > 0 && line->line.find("s_waitcnt") == 0)
                {
                    bMatched = true;
                    skipped_immed -= 1;
                }
                else if (line->line.find("scratch_") == std::string::npos) { break; }
            }
        }

        // std::cout << bMatched << " " << inst.category << " " << int(line->cat) << " " << line->line << std::endl;

        if (bMatched)
        {
            inst.pc = line->addr;
            inst_index++;
            num_failed_stitches = 0;
            line->parsed = true;
        }
        else { num_failed_stitches++; }
    }

    return {inst_index, barrier_gap};
}

void insert_gfx12_barrier_wait(WaveDataInternal& wave, const barrier_list_t& barriers)
{
    if (wave.instructions.empty() || barriers.empty()) return;

    std::vector<Instruction> insts{};
    std::vector<att_wave_state_t> wstates{};

    insts.reserve(barriers.size() + wave.instructions.size());
    wstates.reserve(barriers.size() + wave.timeline.size());

    size_t prev_index = 0;
    size_t timeline_index = 0;
    int64_t current_time = wave.begin_time;

    for (auto& [index, pc] : barriers)
    {
        if (index <= prev_index) continue;

        insts.insert(insts.end(), wave.instructions.begin() + prev_index, wave.instructions.begin() + index);

        auto& previous = wave.instructions.at(index - 1);
        auto& current = wave.instructions.at(index);

        Instruction inst{};
        inst.category = WaveInstCategory::IMMED;
        inst.pc = pc;
        inst.time = previous.time + previous.duration;
        if (current.time == inst.time && current.duration > 1)
        {
            current.time++;
            current.duration--;
        }
        inst.duration = current.time - inst.time;
        inst.stall = 0;

        while (current_time <= inst.time && timeline_index < wave.timeline.size())
        {
            current_time += wave.timeline[timeline_index].duration;
            wstates.push_back(wave.timeline[timeline_index]);
            timeline_index++;
        }

        if (wave.timeline.size() && wstates.back().duration >= current_time - inst.time && current_time >= current.time)
        {
            wstates.back().duration -= current_time - inst.time;
            int type = wstates.back().type;
            wstates.push_back(att_wave_state_t{.type = WaveslotState::WS_WAIT, .duration = int(inst.duration)});
            wstates.push_back(att_wave_state_t{.type = type, .duration = int(current_time - inst.time - inst.duration)}
            );
        }

        insts.push_back(std::move(inst));
        prev_index = index;
    }

    insts.insert(insts.end(), wave.instructions.begin() + prev_index, wave.instructions.end());
    while (timeline_index < wave.timeline.size())
    {
        wstates.push_back(wave.timeline[timeline_index]);
        timeline_index++;
    }

    wave.instructions = std::move(insts);
    wave.timeline = std::move(wstates);
}

#define RADT(x) ROCPROFILER_THREAD_TRACE_DECODER_RECORD_##x

void Stitcher::setgfxip(uint64_t _gfxip)
{
    this->gfxip = _gfxip;
    std::call_once(gfx_flag, callback, RADT(GFXIP), reinterpret_cast<void*>(_gfxip), 0, cbdata);
}

void Stitcher::stitch(WaveDataInternal& wave)
{
    assert(gfxip != 0);
    auto EmitWarning = [&](rocprofiler_thread_trace_decoder_info_t info)
    { callback(RADT(INFO), (void*) &info, 1, cbdata); };

    if (wave.callbackComplete) return;
    wave.callbackComplete = true;

    if (!pctranslator) pctranslator = std::make_unique<PCTranslator>(raw_code, codeobj_service, gfxip);

    auto stitch_rate = stitchWave(wave);
    if (!wave.bIsComplete && !wave.isTrapping)
        std::call_once(incomp_flag, EmitWarning, ROCPROFILER_THREAD_TRACE_DECODER_INFO_WAVE_INCOMPLETE);
    else if (stitch_rate.first != wave.instructions.size() && !wave.pc_infos.empty() && wave.pc_infos.front().second.code_object_id != 0)
        std::call_once(stitch_flag, EmitWarning, ROCPROFILER_THREAD_TRACE_DECODER_INFO_STITCH_INCOMPLETE);
#ifdef ARCH_MODEL
    else if (stitch_rate.first != wave.instructions.size())
        std::call_once(stitch_flag, EmitWarning, ROCPROFILER_THREAD_TRACE_DECODER_INFO_STITCH_INCOMPLETE);
#endif

    if (gfxip == 12 && !stitch_rate.second.empty() && wave.exclude_barrier_wait)
        insert_gfx12_barrier_wait(wave, stitch_rate.second);

    wave.instructions_size = wave.instructions.size();
    wave.instructions_array = wave.instructions.data();
    wave.timeline_size = wave.timeline.size();
    wave.timeline_array = wave.timeline.data();

    if (!wave.bIsComplete)
        for (auto& inst : wave.instructions) wave.end_time = std::max(wave.end_time, inst.time + inst.duration + 4);

    callback(RADT(WAVE), (void*) &wave, 1, cbdata);
}

Stitcher::Stitcher(
    std::shared_ptr<ICodeServicer> service, rocprof_trace_decoder_trace_callback_t _callback, void* _cbdata
) :
codeobj_service(service), callback(_callback), cbdata(_cbdata)
{
#ifndef ARCH_MODEL
    raw_code.push_back(std::make_shared<assemblyLine>());
    raw_code.at(0)->line = "; Begin ASM";
#endif
}
