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

#include "gfx12wave.h"
#include <algorithm>
#include <cassert>
#include <utility>
#include <vector>

typedef gfx10::Token Token;

namespace gfx12
{

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
    vinterp = 18,
    barrier_wait,
    expreq_gds,
    expreq_gfx,
    flat_rd_2 = 28,
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
    lds_dir_load = 110,
    lds_param_load,
    salu_wr_exec = 114,
    valu1_wr_exec,
    valu_b2_wr_exec,
    rfe,
    lds_bvh_6,
    lds_other_simd_6,
    lds_other_simd_10,
    lds_bvh_10,
    barrier_signal,
    dyn_vgpr = 135,
    try_lock,
    unlock,
    barrier_join,
    wmma8 = 140,
    wmma16,
    wmma32,
    wmma64,
    valu_dpfp = 146,
    valu_derate,
    icpref = 150,
    kcpref,
    salu_float3,
    valu_scl_trans,
    salu2 = 155,
    salu5,
    vmem_other_simd_start = 188,
    block_store = 222,
    einst_final
};

static const std::unordered_map<int, mapped_inst_t> table_map_to_common_type{
    {(int) EINST::salu,              {WaveInstCategory::SALU, 1}           },
    {(int) EINST::smem_rd,           {WaveInstCategory::SMEM, 1}           },
    {(int) EINST::smem_wr,           {WaveInstCategory::SMEM, 1}           },
    {(int) EINST::branch_taken,      {WaveInstCategory::JUMP, 1}           },
    {(int) EINST::branch_not_taken,  {WaveInstCategory::NEXT, 1}           },
    {(int) EINST::jump,              {WaveInstCategory::SALU, 1}           },
    {(int) EINST::trap,              {WaveInstCategory::TRAP, 1}           },
    {(int) EINST::salu_no_exec,      {WaveInstCategory::SALU, 1}           },
    {(int) EINST::fatal_halt,        {WaveInstCategory::TRAP, 1}           },
    {(int) EINST::message,           {WaveInstCategory::MSG, 1}            },
    {(int) EINST::valu_1,            {WaveInstCategory::VALU, 1}           },
    {(int) EINST::valut_4,           {WaveInstCategory::VALU, 4}           },
    {(int) EINST::valub_1,           {WaveInstCategory::VALU, 1}           },
    {(int) EINST::valub_2,           {WaveInstCategory::VALU, 2}           },
    {(int) EINST::valub_4,           {WaveInstCategory::VALU, 4}           },
    {(int) EINST::valub_16,          {WaveInstCategory::VALU, 16}          },
    {(int) EINST::vinterp,           {WaveInstCategory::VALU, 1}           },
    {(int) EINST::barrier_wait,      {WaveInstCategory::IMMED, 1}          },
    {(int) EINST::flat_rd_2,         {WaveInstCategory::FLAT, 2}           },
    {(int) EINST::flat_wr_3,         {WaveInstCategory::FLAT, 3}           },
    {(int) EINST::flat_wr_4,         {WaveInstCategory::FLAT, 4}           },
    {(int) EINST::flat_wr_5,         {WaveInstCategory::FLAT, 5}           },
    {(int) EINST::flat_wr_6,         {WaveInstCategory::FLAT, 6}           },
    {(int) EINST::sgmem_rd_1,        {WaveInstCategory::VMEM, 1}           },
    {(int) EINST::sgmem_rd_2,        {WaveInstCategory::VMEM, 2}           },
    {(int) EINST::sgmem_wr_1,        {WaveInstCategory::VMEM, 1}           },
    {(int) EINST::sgmem_wr_2,        {WaveInstCategory::VMEM, 2}           },
    {(int) EINST::sgmem_wr_3,        {WaveInstCategory::VMEM, 3}           },
    {(int) EINST::sgmem_wr_4,        {WaveInstCategory::VMEM, 4}           },
    {(int) EINST::sgmem_wr_5,        {WaveInstCategory::VMEM, 5}           },
    {(int) EINST::sgmem_wr_6,        {WaveInstCategory::VMEM, 6}           },
    {(int) EINST::lds_rd,            {WaveInstCategory::LDS, 1}            },
    {(int) EINST::lds_wr_1,          {WaveInstCategory::LDS, 1}            },
    {(int) EINST::lds_wr_2,          {WaveInstCategory::LDS, 2}            },
    {(int) EINST::lds_wr_3,          {WaveInstCategory::LDS, 3}            },
    {(int) EINST::lds_wr_4,          {WaveInstCategory::LDS, 4}            },
    {(int) EINST::lds_wr_5,          {WaveInstCategory::LDS, 5}            },
    {(int) EINST::buf_rd_1,          {WaveInstCategory::VMEM, 1}           },
    {(int) EINST::buf_rd_2,          {WaveInstCategory::VMEM, 2}           },
    {(int) EINST::buf_wr_1,          {WaveInstCategory::VMEM, 1}           },
    {(int) EINST::buf_wr_2,          {WaveInstCategory::VMEM, 2}           },
    {(int) EINST::buf_wr_3,          {WaveInstCategory::VMEM, 3}           },
    {(int) EINST::buf_wr_4,          {WaveInstCategory::VMEM, 4}           },
    {(int) EINST::buf_wr_5,          {WaveInstCategory::VMEM, 5}           },
    {(int) EINST::buf_wr_6,          {WaveInstCategory::VMEM, 6}           },
    {(int) EINST::img_sample_1,      {WaveInstCategory::VMEM, 1}           },
    {(int) EINST::img_sample_2,      {WaveInstCategory::VMEM, 2}           },
    {(int) EINST::img_sample_3,      {WaveInstCategory::VMEM, 3}           },
    {(int) EINST::img_sample_4,      {WaveInstCategory::VMEM, 4}           },
    {(int) EINST::img_sample_5,      {WaveInstCategory::VMEM, 5}           },
    {(int) EINST::img_sample_6,      {WaveInstCategory::VMEM, 6}           },
    {(int) EINST::img_sample_7,      {WaveInstCategory::VMEM, 7}           },
    {(int) EINST::img_sample_8,      {WaveInstCategory::VMEM, 8}           },
    {(int) EINST::img_sample_9,      {WaveInstCategory::VMEM, 9}           },
    {(int) EINST::img_sample_10,     {WaveInstCategory::VMEM, 10}          },
    {(int) EINST::img_sample_11,     {WaveInstCategory::VMEM, 11}          },
    {(int) EINST::img_sample_12,     {WaveInstCategory::VMEM, 12}          },

    {(int) EINST::img_rd_1,          {WaveInstCategory::VMEM, 1}           },
    {(int) EINST::img_rd_2,          {WaveInstCategory::VMEM, 2}           },
    {(int) EINST::img_rd_3,          {WaveInstCategory::VMEM, 3}           },
    {(int) EINST::img_rd_4,          {WaveInstCategory::VMEM, 4}           },
    {(int) EINST::img_wr_2,          {WaveInstCategory::VMEM, 2}           },
    {(int) EINST::img_wr_3,          {WaveInstCategory::VMEM, 3}           },
    {(int) EINST::img_wr_4,          {WaveInstCategory::VMEM, 4}           },
    {(int) EINST::img_wr_5,          {WaveInstCategory::VMEM, 5}           },
    {(int) EINST::img_wr_6,          {WaveInstCategory::VMEM, 6}           },
    {(int) EINST::img_wr_7,          {WaveInstCategory::VMEM, 7}           },
    {(int) EINST::img_wr_8,          {WaveInstCategory::VMEM, 8}           },

    {(int) EINST::raytrace8,         {WaveInstCategory::BVH, 8}            },
    {(int) EINST::raytrace9,         {WaveInstCategory::BVH, 9}            },
    {(int) EINST::raytrace11,        {WaveInstCategory::BVH, 11}           },
    {(int) EINST::raytrace12,        {WaveInstCategory::BVH, 12}           },

    {(int) EINST::lds_dir_load,      {WaveInstCategory::LDS, 1}            },
    {(int) EINST::lds_param_load,    {WaveInstCategory::LDS, 1}            },
    {(int) EINST::salu_wr_exec,      {WaveInstCategory::SALU, 1}           },
    {(int) EINST::valu1_wr_exec,     {WaveInstCategory::VALU, 1}           },
    {(int) EINST::valu_b2_wr_exec,   {WaveInstCategory::VALU, 2}           },

    {(int) EINST::rfe,               {WaveInstCategory::SALU, 1}           },
    {(int) EINST::lds_bvh_6,         {WaveInstCategory::BVH, 6}            },
    {(int) EINST::lds_bvh_10,        {WaveInstCategory::BVH, 10}           },
    {(int) EINST::barrier_signal,    {WaveInstCategory::MSG, 1}            },
    {(int) EINST::dyn_vgpr,          {WaveInstCategory::SALU, 1}           },
    {(int) EINST::try_lock,          {WaveInstCategory::SALU, 1}           },
    {(int) EINST::unlock,            {WaveInstCategory::SALU, 1}           },
    {(int) EINST::barrier_join,      {WaveInstCategory::MSG, 1}            },

    {(int) EINST::wmma8,             {WaveInstCategory::VALU, 8}           },
    {(int) EINST::wmma16,            {WaveInstCategory::VALU, 16}          },
    {(int) EINST::wmma32,            {WaveInstCategory::VALU, 32}          },
    {(int) EINST::wmma64,            {WaveInstCategory::VALU, 64}          },

    {(int) EINST::icpref,            {WaveInstCategory::SMEM, 1}           },
    {(int) EINST::kcpref,            {WaveInstCategory::SMEM, 1}           },
    {(int) EINST::salu_float3,       {WaveInstCategory::SALU, 3}           },

    {(int) EINST::valu_scl_trans,    {WaveInstCategory::VALU, 1}           },
    {(int) EINST::salu2,             {WaveInstCategory::SALU, 2}           },
    {(int) EINST::salu5,             {WaveInstCategory::SALU, 5}           },
    {(int) EINST::valu_scl_trans,    {WaveInstCategory::VALU, 1}           },

    {(int) EINST::lds_other_simd_1,  {WaveInstCategory::LDS_OTHER_SIMD, 1} },
    {(int) EINST::lds_other_simd_2,  {WaveInstCategory::LDS_OTHER_SIMD, 2} },
    {(int) EINST::lds_other_simd_3,  {WaveInstCategory::LDS_OTHER_SIMD, 3} },
    {(int) EINST::lds_other_simd_4,  {WaveInstCategory::LDS_OTHER_SIMD, 4} },
    {(int) EINST::lds_other_simd_5,  {WaveInstCategory::LDS_OTHER_SIMD, 5} },
    {(int) EINST::lds_other_simd_6,  {WaveInstCategory::LDS_OTHER_SIMD, 6} },
    {(int) EINST::lds_other_simd_10, {WaveInstCategory::LDS_OTHER_SIMD, 10}},

    {(int) EINST::flat_other_simd_2, {WaveInstCategory::FLAT_OTHER_SIMD, 2}},
    {(int) EINST::flat_other_simd_3, {WaveInstCategory::FLAT_OTHER_SIMD, 3}},
    {(int) EINST::flat_other_simd_4, {WaveInstCategory::FLAT_OTHER_SIMD, 4}},
    {(int) EINST::flat_other_simd_5, {WaveInstCategory::FLAT_OTHER_SIMD, 5}},
    {(int) EINST::flat_other_simd_6, {WaveInstCategory::FLAT_OTHER_SIMD, 6}}
};

mapped_inst_t wave_t::map_to_common_type(int einst, int dprate, int derate)
{
    {
        auto it = table_map_to_common_type.find(einst);
        if (it != table_map_to_common_type.end()) return it->second;
    }

    if (einst >= (int) EINST::vmem_other_simd_start)
    {
        if (einst < (int) EINST::block_store)
            return {WaveInstCategory::VMEM_OTHER_SIMD, einst + 1 - (int) EINST::vmem_other_simd_start};
        else
            return {WaveInstCategory::VMEM, einst + 1 - (int) EINST::block_store};
    }
    else if (einst == (int) EINST::valu_dpfp || einst == (int) EINST::valu_derate)
    {
        auto inst = mapped_inst_t{WaveInstCategory::VALU, dprate};
        if (einst == (int) EINST::valu_derate) inst.cycles *= derate;
        return inst;
    }

    return mapped_inst_t{WaveInstCategory::NONE, 0};
}

} // namespace gfx12
