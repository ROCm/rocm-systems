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

#include "mi400wave.h"
#include <algorithm>
#include <cassert>
#include <utility>
#include <vector>

typedef gfx10::Token Token;

namespace mi400
{

void handle_xnack_rewind(gfx10::wave_t& wave)
{
    // Rewind PC. TODO: This may accidentally skip s_nops
    while (wave.instructions.size() && wave.instructions.back().category == WaveInstCategory::IMMED)
        wave.instructions.pop_back();

    // Rewind wave states
    if (!wave.instructions.empty())
    {
        int64_t rewindtime = wave.instructions.back().time - 1;
        while (!wave.timeline.empty())
        {
            wave.last_state_cycle -= wave.timeline.back().duration;
            if (wave.last_state_cycle <= rewindtime)
            {
                wave.timeline.back().duration = rewindtime - wave.last_state_cycle;
                wave.last_state_cycle = rewindtime;
                break;
            }
            wave.timeline.pop_back();
        }
        uint32_t cat = wave.instructions.back().category;
        if (cat == WaveInstCategory::VMEM || cat == WaveInstCategory::LDS || cat == WaveInstCategory::SMEM)
            wave.instructions.pop_back();
    }
    wave.cur_state = WaveslotState::WS_STALL;
    wave.bIsXnack = true;
}

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
    valut,
    valub_1,
    valub_2,
    valub_4,
    valub_16,
    valub_32,
    valub_8,
    vinterp,
    barrier_wait,
    expreq_gds,
    expreq_gfx,
    flat_rd_1 = 26,
    flat_rd_2,
    flat_wr_2,
    flat_wr_3,
    flat_wr_4,
    flat_wr_5,
    flat_wr_6,
    sgmem_rd_1,
    sgmem_rd_2,
    sgmem_wr_1,
    sgmem_wr_2,
    sgmem_wr_3,
    sgmem_wr_4,
    sgmem_wr_5,
    sgmem_wr_6,
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
    replay = 79,
    lds_other_simd_1 = 80,
    lds_other_simd_2,
    lds_other_simd_3,
    lds_other_simd_4,
    lds_other_simd_5,
    flat_other_simd_2,
    flat_other_simd_3,
    flat_other_simd_4,
    flat_other_simd_5,
    flat_other_simd_6,
    raytrace8 = 103,
    raytrace9,
    raytrace11,
    raytrace12,
    flat_other_simd_1,
    lds_dir_load = 110,
    lds_param_load,
    salu_wr_exec = 114,
    valu1_wr_exec,
    valu_b2_wr_exec,
    rfe,
    lds_bvh_6,
    lds_other_simd_6,
    lds_other_simd_10,
    barrier_signal = 122,
    dyn_vgpr = 135,
    try_lock,
    unlock,
    barrier_join,
    icpref = 150,
    kcpref,
    salu_float3,
    valu_scl_trans,
    salu2 = 155,
    salu5,
    multicast_load_vgpr,
    multicast_load_lds,
    async_load_1,
    async_load_2,
    async_store_2,
    async_store_3,
    async_store_5,
    valut_1 = 165,
    valu_selftest,
    valu_no_exec = 169,
    vmpref,
    tdm,
    tdm_other_simd,
    lds_async_barrier,
    cluster_barrier_signal,
    reserved_valu_16,
    reserved_valu_32,
    wmma_spfp_16,
    wmma_xdl_8,
    wmma_xdl_16,
    reserved_valu_1,
    reserved_valu_4,
    reserved_valu_2,
    wmma_ld_scale,
    valu_dpmacc_32 = 187,
    vmem_other_simd_start,
    block_store = 222,
    block_store_end = 255,
    einst_final
};

static std::unordered_map<int, mapped_inst_t> table_map_to_common_type{
    {(int) EINST::salu,                   {WaveInstCategory::SALU, 1}           },
    {(int) EINST::smem_rd,                {WaveInstCategory::SMEM, 1}           },
    {(int) EINST::smem_wr,                {WaveInstCategory::SMEM, 1}           },
    {(int) EINST::branch_taken,           {WaveInstCategory::JUMP, 1}           },
    {(int) EINST::branch_not_taken,       {WaveInstCategory::NEXT, 1}           },
    {(int) EINST::jump,                   {WaveInstCategory::SALU, 1}           },
    {(int) EINST::trap,                   {WaveInstCategory::TRAP, 1}           },
    {(int) EINST::salu_no_exec,           {WaveInstCategory::SALU, 1}           },
    {(int) EINST::fatal_halt,             {WaveInstCategory::TRAP, 1}           },
    {(int) EINST::message,                {WaveInstCategory::MSG, 1}            },
    {(int) EINST::valu_1,                 {WaveInstCategory::VALU, 1}           },
    {(int) EINST::valub_1,                {WaveInstCategory::VALU, 1}           },
    {(int) EINST::valub_2,                {WaveInstCategory::VALU, 2}           },
    {(int) EINST::valub_4,                {WaveInstCategory::VALU, 4}           },
    {(int) EINST::valub_8,                {WaveInstCategory::VALU, 8}           },
    {(int) EINST::valub_16,               {WaveInstCategory::VALU, 16}          },
    {(int) EINST::valub_32,               {WaveInstCategory::VALU, 32}          },
    {(int) EINST::vinterp,                {WaveInstCategory::VALU, 1}           },
    {(int) EINST::barrier_wait,           {WaveInstCategory::IMMED, 1}          },
    {(int) EINST::flat_rd_1,              {WaveInstCategory::FLAT, 1}           },
    {(int) EINST::flat_rd_2,              {WaveInstCategory::FLAT, 2}           },
    {(int) EINST::flat_wr_2,              {WaveInstCategory::FLAT, 2}           },
    {(int) EINST::flat_wr_3,              {WaveInstCategory::FLAT, 3}           },
    {(int) EINST::flat_wr_4,              {WaveInstCategory::FLAT, 4}           },
    {(int) EINST::flat_wr_5,              {WaveInstCategory::FLAT, 5}           },
    {(int) EINST::flat_wr_6,              {WaveInstCategory::FLAT, 6}           },
    {(int) EINST::sgmem_rd_1,             {WaveInstCategory::VMEM, 1}           },
    {(int) EINST::sgmem_rd_2,             {WaveInstCategory::VMEM, 2}           },
    {(int) EINST::sgmem_wr_1,             {WaveInstCategory::VMEM, 1}           },
    {(int) EINST::sgmem_wr_2,             {WaveInstCategory::VMEM, 2}           },
    {(int) EINST::sgmem_wr_3,             {WaveInstCategory::VMEM, 3}           },
    {(int) EINST::sgmem_wr_4,             {WaveInstCategory::VMEM, 4}           },
    {(int) EINST::sgmem_wr_5,             {WaveInstCategory::VMEM, 5}           },
    {(int) EINST::sgmem_wr_6,             {WaveInstCategory::VMEM, 6}           },
    {(int) EINST::lds_rd,                 {WaveInstCategory::LDS, 1}            },
    {(int) EINST::lds_wr_1,               {WaveInstCategory::LDS, 1}            },
    {(int) EINST::lds_wr_2,               {WaveInstCategory::LDS, 2}            },
    {(int) EINST::lds_wr_3,               {WaveInstCategory::LDS, 3}            },
    {(int) EINST::lds_wr_4,               {WaveInstCategory::LDS, 4}            },
    {(int) EINST::lds_wr_5,               {WaveInstCategory::LDS, 5}            },
    {(int) EINST::buf_rd_1,               {WaveInstCategory::VMEM, 1}           },
    {(int) EINST::buf_rd_2,               {WaveInstCategory::VMEM, 2}           },
    {(int) EINST::buf_wr_1,               {WaveInstCategory::VMEM, 1}           },
    {(int) EINST::buf_wr_2,               {WaveInstCategory::VMEM, 2}           },
    {(int) EINST::buf_wr_3,               {WaveInstCategory::VMEM, 3}           },
    {(int) EINST::buf_wr_4,               {WaveInstCategory::VMEM, 4}           },
    {(int) EINST::buf_wr_5,               {WaveInstCategory::VMEM, 5}           },
    {(int) EINST::buf_wr_6,               {WaveInstCategory::VMEM, 6}           },
    {(int) EINST::img_sample_1,           {WaveInstCategory::VMEM, 1}           },
    {(int) EINST::img_sample_2,           {WaveInstCategory::VMEM, 2}           },
    {(int) EINST::img_sample_3,           {WaveInstCategory::VMEM, 3}           },
    {(int) EINST::img_sample_4,           {WaveInstCategory::VMEM, 4}           },
    {(int) EINST::img_sample_5,           {WaveInstCategory::VMEM, 5}           },
    {(int) EINST::img_sample_6,           {WaveInstCategory::VMEM, 6}           },
    {(int) EINST::img_sample_7,           {WaveInstCategory::VMEM, 7}           },
    {(int) EINST::img_sample_8,           {WaveInstCategory::VMEM, 8}           },
    {(int) EINST::img_sample_9,           {WaveInstCategory::VMEM, 9}           },
    {(int) EINST::img_sample_10,          {WaveInstCategory::VMEM, 10}          },
    {(int) EINST::img_sample_11,          {WaveInstCategory::VMEM, 11}          },
    {(int) EINST::img_sample_12,          {WaveInstCategory::VMEM, 12}          },

    {(int) EINST::img_rd_1,               {WaveInstCategory::VMEM, 1}           },
    {(int) EINST::img_rd_2,               {WaveInstCategory::VMEM, 2}           },
    {(int) EINST::img_rd_3,               {WaveInstCategory::VMEM, 3}           },
    {(int) EINST::img_rd_4,               {WaveInstCategory::VMEM, 4}           },
    {(int) EINST::img_wr_2,               {WaveInstCategory::VMEM, 2}           },
    {(int) EINST::img_wr_3,               {WaveInstCategory::VMEM, 3}           },
    {(int) EINST::img_wr_4,               {WaveInstCategory::VMEM, 4}           },
    {(int) EINST::img_wr_5,               {WaveInstCategory::VMEM, 5}           },
    {(int) EINST::img_wr_6,               {WaveInstCategory::VMEM, 6}           },
    {(int) EINST::img_wr_7,               {WaveInstCategory::VMEM, 7}           },
    {(int) EINST::img_wr_8,               {WaveInstCategory::VMEM, 8}           },

    {(int) EINST::raytrace8,              {WaveInstCategory::BVH, 8}            },
    {(int) EINST::raytrace9,              {WaveInstCategory::BVH, 9}            },
    {(int) EINST::raytrace11,             {WaveInstCategory::BVH, 11}           },
    {(int) EINST::raytrace12,             {WaveInstCategory::BVH, 12}           },
    {(int) EINST::flat_other_simd_1,      {WaveInstCategory::NONE, 0}           },

    {(int) EINST::lds_dir_load,           {WaveInstCategory::LDS, 1}            },
    {(int) EINST::lds_param_load,         {WaveInstCategory::LDS, 1}            },
    {(int) EINST::salu_wr_exec,           {WaveInstCategory::SALU, 1}           },
    {(int) EINST::valu1_wr_exec,          {WaveInstCategory::VALU, 1}           },
    {(int) EINST::valu_b2_wr_exec,        {WaveInstCategory::VALU, 2}           },

    {(int) EINST::rfe,                    {WaveInstCategory::SALU, 1}           },
    {(int) EINST::barrier_signal,         {WaveInstCategory::MSG, 1}            },
    {(int) EINST::dyn_vgpr,               {WaveInstCategory::SALU, 1}           },
    {(int) EINST::try_lock,               {WaveInstCategory::SALU, 1}           },
    {(int) EINST::unlock,                 {WaveInstCategory::SALU, 1}           },
    {(int) EINST::barrier_join,           {WaveInstCategory::MSG, 1}            },

    {(int) EINST::icpref,                 {WaveInstCategory::SMEM, 1}           },
    {(int) EINST::kcpref,                 {WaveInstCategory::SMEM, 1}           },
    {(int) EINST::salu_float3,            {WaveInstCategory::SALU, 3}           },

    {(int) EINST::valu_scl_trans,         {WaveInstCategory::VALU, 1}           },
    {(int) EINST::salu2,                  {WaveInstCategory::SALU, 2}           },
    {(int) EINST::salu5,                  {WaveInstCategory::SALU, 5}           },
    {(int) EINST::multicast_load_vgpr,    {WaveInstCategory::VMEM, 1}           },
    {(int) EINST::multicast_load_lds,     {WaveInstCategory::VMEM, 1}           },
    {(int) EINST::async_load_1,           {WaveInstCategory::VMEM, 1}           },
    {(int) EINST::async_load_2,           {WaveInstCategory::VMEM, 2}           },
    {(int) EINST::async_store_2,          {WaveInstCategory::VMEM, 2}           },
    {(int) EINST::async_store_3,          {WaveInstCategory::VMEM, 3}           },
    {(int) EINST::async_store_5,          {WaveInstCategory::VMEM, 5}           },
    {(int) EINST::valu_no_exec,           {WaveInstCategory::VALU, 1}           },
    {(int) EINST::tdm,                    {WaveInstCategory::VMEM, 1}           },
    {(int) EINST::tdm_other_simd,         {WaveInstCategory::NONE, 0}           },
    {(int) EINST::lds_async_barrier,      {WaveInstCategory::LDS, 1}            },
    {(int) EINST::vmpref,                 {WaveInstCategory::VMEM, 1}           },
    {(int) EINST::cluster_barrier_signal, {WaveInstCategory::MSG, 1}            },

    {(int) EINST::wmma_spfp_16,           {WaveInstCategory::VALU, 16}          },
    {(int) EINST::wmma_xdl_8,             {WaveInstCategory::VALU, 8}           },
    {(int) EINST::wmma_xdl_16,            {WaveInstCategory::VALU, 16}          },
    {(int) EINST::wmma_ld_scale,          {WaveInstCategory::LD_SCALE, 1}       },

    {(int) EINST::lds_other_simd_1,       {WaveInstCategory::LDS_OTHER_SIMD, 1} },
    {(int) EINST::lds_other_simd_2,       {WaveInstCategory::LDS_OTHER_SIMD, 2} },
    {(int) EINST::lds_other_simd_3,       {WaveInstCategory::LDS_OTHER_SIMD, 3} },
    {(int) EINST::lds_other_simd_4,       {WaveInstCategory::LDS_OTHER_SIMD, 4} },
    {(int) EINST::lds_other_simd_5,       {WaveInstCategory::LDS_OTHER_SIMD, 5} },
    {(int) EINST::lds_other_simd_6,       {WaveInstCategory::LDS_OTHER_SIMD, 6} },
    {(int) EINST::lds_other_simd_10,      {WaveInstCategory::LDS_OTHER_SIMD, 10}},

    {(int) EINST::tdm_other_simd,         {WaveInstCategory::VMEM_OTHER_SIMD, 1}},
    {(int) EINST::flat_other_simd_2,      {WaveInstCategory::FLAT_OTHER_SIMD, 2}},
    {(int) EINST::flat_other_simd_3,      {WaveInstCategory::FLAT_OTHER_SIMD, 3}},
    {(int) EINST::flat_other_simd_4,      {WaveInstCategory::FLAT_OTHER_SIMD, 4}},
    {(int) EINST::flat_other_simd_5,      {WaveInstCategory::FLAT_OTHER_SIMD, 5}},
    {(int) EINST::flat_other_simd_6,      {WaveInstCategory::FLAT_OTHER_SIMD, 6}},

    {(int) EINST::reserved_valu_1,        {WaveInstCategory::VALU, 1}           },
    {(int) EINST::reserved_valu_2,        {WaveInstCategory::VALU, 2}           },
    {(int) EINST::reserved_valu_4,        {WaveInstCategory::VALU, 4}           },
    {(int) EINST::reserved_valu_16,       {WaveInstCategory::VALU, 16}          },
    {(int) EINST::reserved_valu_32,       {WaveInstCategory::VALU, 32}          }
};

mapped_inst_t map_to_common_type(int einst, int trans2)
{
    {
        auto it = table_map_to_common_type.find(einst);
        if (it != table_map_to_common_type.end()) return it->second;
    }

    if (einst == (int) EINST::valut)
        return {WaveInstCategory::VALU, 4 >> trans2};
    else if (einst == (int) EINST::replay)
        throw std::exception();

    if (einst >= (int) EINST::vmem_other_simd_start)
    {
        if (einst < (int) EINST::block_store)
            return {WaveInstCategory::VMEM_OTHER_SIMD, einst + 1 - (int) EINST::vmem_other_simd_start};
        else
            return {WaveInstCategory::VMEM, einst + 1 - (int) EINST::block_store};
    }

    return mapped_inst_t{WaveInstCategory::NONE, 0};
}

} // namespace mi400
