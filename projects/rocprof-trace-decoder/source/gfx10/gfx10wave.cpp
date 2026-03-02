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

#include "gfx10wave.h"
#include <algorithm>
#include <cassert>
#include <map>
#include <utility>
#include <vector>
#include "gfx11/gfx11wave.h"
#include "gfx12/gfx12wave.h"
#include "segment.hpp"

#define INST_JUMP_TYPE 5
#define INST_TRAP_TYPE 6
#define INST_RFE_TYPE  117

namespace gfx10
{

typedef gfx10::Token Token;

enum EINST
{
    salu = 0,
    smem_rd,
    smem_wr,
    branch_taken,
    branch_not_taken,
    jump,
    trap,
    salu_no_exec,
    fatal_halt,
    message,
    valu_1,
    valut_4,
    valub_1,
    valub_2,
    valub_4,
    valub_16,
    valub_dfdp,
    valub_dfdp_derate,
    vinterp = 18,
    barrier,
    expreq_gds,
    expreq_gfx,
    flat_rd = 30,
    flat_wr_2,
    flat_wr_3,
    flat_wr_4,
    flat_wr_5,
    sgmem_rd,
    sgmem_wr_1,
    sgmem_wr_2,
    sgmem_wr_3,
    sgmem_wr_4,
    sgmem_wr_5,
    lds_rd,
    lds_wr_1,
    lds_wr_2,
    lds_wr_3,
    lds_wr_4,
    lds_wr_5,
    buf_rd_1,
    buf_rd_2,
    buf_wr_1,
    buf_wr_2,
    buf_wr_3,
    buf_wr_4,
    buf_wr_5,
    buf_wr_6,
    img_sample_1,
    img_sample_2,
    img_sample_3,
    img_sample_4,
    img_sample_5,
    img_sample_6,
    img_sample_7,
    img_sample_8,
    img_sample_9,
    img_sample_10,
    img_sample_11,
    img_sample_12,
    img_sample_reserved = 67,
    img_rd_1,
    img_rd_2,
    img_rd_3,
    img_rd_4,
    img_wr_2,
    img_wr_3,
    img_wr_4,
    img_wr_5,
    img_wr_6,
    img_wr_7,
    img_wr_8,

    other_simd_start = 79,
    other_simd_end = 102,
    raytrace8,
    raytrace9,
    raytrace11,
    raytrace12,

    lds_dir_load = 110,
    lds_param_load,
    subv_loop_begin,
    subv_loop_end,

    einst_final
};

static std::unordered_map<int, std::pair<WaveInstCategory, uint16_t>> table_map_to_common_type{
    {(int) EINST::salu,              {WaveInstCategory::SALU, 1} },
    {(int) EINST::smem_rd,           {WaveInstCategory::SMEM, 1} },
    {(int) EINST::smem_wr,           {WaveInstCategory::SMEM, 1} },
    {(int) EINST::branch_taken,      {WaveInstCategory::JUMP, 1} },
    {(int) EINST::branch_not_taken,  {WaveInstCategory::NEXT, 1} },
    {(int) EINST::jump,              {WaveInstCategory::SALU, 1} },
    {(int) EINST::trap,              {WaveInstCategory::TRAP, 1} },
    {(int) EINST::salu_no_exec,      {WaveInstCategory::SALU, 1} },
    {(int) EINST::fatal_halt,        {WaveInstCategory::TRAP, 1} },
    {(int) EINST::message,           {WaveInstCategory::MSG, 1}  },
    {(int) EINST::valu_1,            {WaveInstCategory::VALU, 1} },
    {(int) EINST::valut_4,           {WaveInstCategory::VALU, 4} },
    {(int) EINST::valub_1,           {WaveInstCategory::VALU, 1} },
    {(int) EINST::valub_2,           {WaveInstCategory::VALU, 2} },
    {(int) EINST::valub_4,           {WaveInstCategory::VALU, 4} },
    {(int) EINST::valub_16,          {WaveInstCategory::VALU, 16}},
    {(int) EINST::valub_dfdp,        {WaveInstCategory::VALU, 1} },
    {(int) EINST::valub_dfdp_derate, {WaveInstCategory::VALU, 1} },
    {(int) EINST::vinterp,           {WaveInstCategory::VALU, 1} },
    {(int) EINST::barrier,           {WaveInstCategory::IMMED, 1}},
    {(int) EINST::flat_rd,           {WaveInstCategory::FLAT, 1} },
    {(int) EINST::flat_wr_2,         {WaveInstCategory::FLAT, 2} },
    {(int) EINST::flat_wr_3,         {WaveInstCategory::FLAT, 3} },
    {(int) EINST::flat_wr_4,         {WaveInstCategory::FLAT, 4} },
    {(int) EINST::flat_wr_5,         {WaveInstCategory::FLAT, 5} },
    {(int) EINST::sgmem_rd,          {WaveInstCategory::VMEM, 1} },
    {(int) EINST::sgmem_wr_1,        {WaveInstCategory::VMEM, 1} },
    {(int) EINST::sgmem_wr_2,        {WaveInstCategory::VMEM, 2} },
    {(int) EINST::sgmem_wr_3,        {WaveInstCategory::VMEM, 3} },
    {(int) EINST::sgmem_wr_4,        {WaveInstCategory::VMEM, 4} },
    {(int) EINST::sgmem_wr_5,        {WaveInstCategory::VMEM, 5} },
    {(int) EINST::lds_rd,            {WaveInstCategory::LDS, 1}  },
    {(int) EINST::lds_wr_1,          {WaveInstCategory::LDS, 1}  },
    {(int) EINST::lds_wr_2,          {WaveInstCategory::LDS, 2}  },
    {(int) EINST::lds_wr_3,          {WaveInstCategory::LDS, 3}  },
    {(int) EINST::lds_wr_4,          {WaveInstCategory::LDS, 4}  },
    {(int) EINST::lds_wr_5,          {WaveInstCategory::LDS, 5}  },
    {(int) EINST::buf_rd_1,          {WaveInstCategory::VMEM, 1} },
    {(int) EINST::buf_rd_2,          {WaveInstCategory::VMEM, 2} },
    {(int) EINST::buf_wr_1,          {WaveInstCategory::VMEM, 1} },
    {(int) EINST::buf_wr_2,          {WaveInstCategory::VMEM, 2} },
    {(int) EINST::buf_wr_3,          {WaveInstCategory::VMEM, 3} },
    {(int) EINST::buf_wr_4,          {WaveInstCategory::VMEM, 4} },
    {(int) EINST::buf_wr_5,          {WaveInstCategory::VMEM, 5} },
    {(int) EINST::buf_wr_6,          {WaveInstCategory::VMEM, 6} },
    {(int) EINST::img_rd_1,          {WaveInstCategory::VMEM, 1} },
    {(int) EINST::img_rd_2,          {WaveInstCategory::VMEM, 2} },
    {(int) EINST::img_rd_3,          {WaveInstCategory::VMEM, 3} },
    {(int) EINST::img_rd_4,          {WaveInstCategory::VMEM, 4} },
    {(int) EINST::img_wr_2,          {WaveInstCategory::VMEM, 2} },
    {(int) EINST::img_wr_3,          {WaveInstCategory::VMEM, 3} },
    {(int) EINST::img_wr_4,          {WaveInstCategory::VMEM, 4} },
    {(int) EINST::img_wr_5,          {WaveInstCategory::VMEM, 5} },
    {(int) EINST::img_wr_6,          {WaveInstCategory::VMEM, 6} },
    {(int) EINST::img_wr_7,          {WaveInstCategory::VMEM, 7} },
    {(int) EINST::img_wr_8,          {WaveInstCategory::VMEM, 8} },
    {(int) EINST::img_sample_1,      {WaveInstCategory::VMEM, 1} },
    {(int) EINST::img_sample_2,      {WaveInstCategory::VMEM, 2} },
    {(int) EINST::img_sample_3,      {WaveInstCategory::VMEM, 3} },
    {(int) EINST::img_sample_4,      {WaveInstCategory::VMEM, 4} },
    {(int) EINST::img_sample_5,      {WaveInstCategory::VMEM, 5} },
    {(int) EINST::img_sample_6,      {WaveInstCategory::VMEM, 6} },
    {(int) EINST::img_sample_7,      {WaveInstCategory::VMEM, 7} },
    {(int) EINST::img_sample_8,      {WaveInstCategory::VMEM, 8} },
    {(int) EINST::img_sample_9,      {WaveInstCategory::VMEM, 9} },
    {(int) EINST::img_sample_10,     {WaveInstCategory::VMEM, 10}},
    {(int) EINST::img_sample_11,     {WaveInstCategory::VMEM, 11}},
    {(int) EINST::img_sample_12,     {WaveInstCategory::VMEM, 12}},

    {(int) EINST::raytrace8,         {WaveInstCategory::BVH, 8}  },
    {(int) EINST::raytrace9,         {WaveInstCategory::BVH, 9}  },
    {(int) EINST::raytrace11,        {WaveInstCategory::BVH, 11} },
    {(int) EINST::raytrace12,        {WaveInstCategory::BVH, 12} },

    {(int) EINST::lds_dir_load,      {WaveInstCategory::LDS, 1}  },
    {(int) EINST::lds_param_load,    {WaveInstCategory::LDS, 1}  },
    {(int) EINST::subv_loop_begin,   {WaveInstCategory::SALU, 1} },
    {(int) EINST::subv_loop_end,     {WaveInstCategory::SALU, 1} }
};

std::pair<WaveInstCategory, uint16_t> wave_t::map_to_common_type(int einst, int dprate, int derate)
{
    static thread_local auto empty = std::pair<WaveInstCategory, uint16_t>{WaveInstCategory::NONE, 0};
    if (einst >= (int) EINST::other_simd_start && einst <= (int) EINST::other_simd_end) return empty;

    try
    {
        auto inst = table_map_to_common_type.at(einst);
        if (einst >= (int) EINST::valub_1 && einst <= (int) EINST::valub_dfdp_derate) inst.second *= dprate;

        return inst;
    }
    catch (...)
    {
        return empty;
    }
}

wave_t::wave_t(int target_wgp, int tg_simd, int slot, pcinfo_t addr, Token& token, bool exbarw) :
WaveDataInternal(target_wgp, tg_simd, slot, token.time, addr, exbarw)
{
    this->last_state_cycle = token.time;
    this->cur_state = WaveslotState::WS_IDLE;
}

void wave_t::complete_wave(Token& token)
{
    update_state(WaveslotState::WS_EMPTY, token.time);
    this->end_time = token.time;
    bIsComplete = true;

    if (instructions.empty()) return;

    int64_t time = instructions.back().time + instructions.back().duration;
    // TT not generating s_endpgm message
    if (time < token.time && trap_status == WaveTrapStatus::TRAP_RESTORED)
        instructions.push_back({time, WaveInstCategory::MSG, token.time - time, 0});
}

void wave_t::apply_wave_rdy(int64_t time)
{
    update_barrier_gfx11(time - 1);
    update_state(WaveslotState::WS_STALL, time);
}

void wave_t::update_state(WaveslotState new_state, int64_t time)
{
    if (time > last_state_cycle)
    {
        if (!timeline.size() || this->timeline.back().type != cur_state)
            this->timeline.push_back(att_wave_state_t{.type = cur_state, .duration = 0});

        this->timeline.back().duration += time - last_state_cycle;
        this->last_state_cycle = time;
    }
    this->cur_state = new_state;
}

void wave_t::new_pc(int64_t time, int64_t pc, CodeobjTableTranslator& table)
{
    update_barrier_gfx11(time);
    if (pc_infos.empty() || trap_status != WaveTrapStatus::TRAP_RESTORED) return;

    int info_idx = pc_infos.size();
    while (info_idx > 0 && !bValid(pc_infos.at(info_idx - 1).second)) info_idx--;

    if (info_idx == pc_infos.size())
        pc_infos.push_back({
            instructions.size(), pcinfo_t{0, 0}
        });

    auto& back = pc_infos.at(info_idx).second;

    back = table.ToPcV2(pc << 2);
    if (back.code_object_id == 0) unattrib_pcs.push_back(pc_infos.size() - 1);
}

void wave_t::apply_valu_inst(Token token, bool wave64)
{
    update_barrier_gfx11(token.time);

    if (trap_status != WaveTrapStatus::TRAP_RESTORED) return;

    int64_t time = token.time;
    int64_t duration = wave64 ? 2 : 1;
    int64_t stall = 0;
    if (cur_state == WaveslotState::WS_STALL && last_state_cycle < time)
    {
        stall = time - last_state_cycle;
        duration += stall;
        time = last_state_cycle;
    }

    this->instructions.push_back({time, WaveInstCategory::VALU, duration, stall});
    update_state(WaveslotState::WS_EXEC, token.time);
}

void wave_t::apply_immediate(int64_t token_time)
{
    update_barrier_gfx11(token_time);
    if (trap_status != WaveTrapStatus::TRAP_RESTORED) return;

    int64_t time = token_time;
    if (instructions.size()) time = std::min(time, instructions.back().time + instructions.back().duration);

    int stall = token_time - time;
    this->instructions.push_back({time, WaveInstCategory::IMMED, stall + 1, stall});
    update_state(WaveslotState::WS_WAIT, time);
    update_state(WaveslotState::WS_EXEC, token_time + 1);
}

#ifdef SQTT_LOGGING
static std::vector<const char*> INST_CATEGORIES = {
    "NONE", "SMEM", "SALU", "VMEM", "FLAT", "LDS", "VALU", "JUMP", "NEXT", "IMMED", "MESSAGE", "CONTEXT", "BVH"};
#endif

void wave_t::update_barrier_gfx11(int64_t token_time)
{
    if (!extend_barrier_gfx11) return;
    extend_barrier_gfx11 = false;
    if (instructions.empty()) return;

    auto& back = instructions.back();
    if (back.category == WaveInstCategory::IMMED)
    {
        update_state(WaveslotState::WS_WAIT, back.time + back.duration);
        update_state(WaveslotState::WS_EXEC, token_time + 1);

        back.duration = std::max<int64_t>(0, token_time - back.time);
        back.category = WaveInstCategory::MSG;
    }
}

void wave_t::apply_inst(Token token, inst_type_common inst, int tt_version, int dprate, int derate)
{
    update_barrier_gfx11(token.time);

    if (inst.inst == INST_RFE_TYPE)
    {
        pc_infos.push_back({
            instructions.size(), pcinfo_t{0, 0}
        });
        trap_status = WaveTrapStatus::TRAP_RESTORED;

        if (instructions.size() && instructions.back().category == WaveInstCategory::TRAP)
        {
            auto& back = instructions.back();
            back.duration = std::max<int64_t>(back.duration, token.time + 1 - back.time);
        }
        return;
    }
    else if (inst.inst == INST_TRAP_TYPE)
    {
        instructions.push_back({token.time, WaveInstCategory::TRAP, 1, 0});
        trap_status = WaveTrapStatus::TRAP_STANDBY;
    }

    if (trap_status != WaveTrapStatus::TRAP_RESTORED) return;

    this->end_time = token.time;

    auto mapped = std::pair<WaveInstCategory, uint16_t>{WaveInstCategory::NONE, 0};
    if (tt_version == 4)
        mapped = gfx12::wave_t::map_to_common_type(inst.inst, dprate, derate);
    else if (tt_version == 3)
        mapped = gfx11::wave_t::map_to_common_type(inst.inst, dprate, derate);
    else
        mapped = wave_t::map_to_common_type(inst.inst, dprate, derate);

    if (mapped.first == WaveInstCategory::IMMED)
    {
        apply_immediate(token.time);
        extend_barrier_gfx11 = tt_version <= 3;
        return;
    }
    if (mapped.first == WaveInstCategory::NONE) return;

    int64_t time = token.time;
    int64_t duration = mapped.second * (inst.w64h ? 2 : 1);
    int64_t stall = 0;
    if (cur_state == WaveslotState::WS_STALL && last_state_cycle < time)
    {
        stall = time - last_state_cycle;
        duration += stall;
        time = last_state_cycle;
    }

    this->instructions.push_back({time, mapped.first, duration, stall});

    if (inst.inst == INST_JUMP_TYPE)
        pc_infos.push_back({
            instructions.size(), pcinfo_t{0, 0}
        });

    update_state(WaveslotState::WS_EXEC, token.time);
}

} // namespace gfx10
