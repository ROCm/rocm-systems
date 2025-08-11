// MIT License
//
// Copyright (c) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include "lib/att-tool/waitcnt/analysis.hpp"
#include "lib/common/logging.hpp"
#include "lib/rocprofiler-sdk/registration.hpp"

#include <gtest/gtest.h>
#include <iostream>

namespace rocprofiler
{
namespace att_wrapper
{
// This is used so the first line number dont get skipped because their vaddr==0
constexpr uint64_t LINE_OFFSET = 1;

TEST(att_decoder_waitcnt_test, gfx9)
{
    registration::init_logging();

    constexpr size_t LOOP_CNT = 4;

    WaitcntList::isa_map_t isa_map{};

    auto append_isa = [&](size_t line_number, const char* line) {
        pcinfo_t pc{};
        pc.addr      = line_number + LINE_OFFSET;
        pc.marker_id = 0;

        auto code             = std::make_unique<CodeLine>();
        code->code_line       = std::make_shared<CodeLine::Instruction>();
        code->code_line->inst = line;
        code->line_number     = line_number;

        isa_map.emplace(pc, std::move(code));
    };

    append_isa(0, "s_nop 0");
    append_isa(1, "v_add_ 0");
    append_isa(2, "s_waitcnt vmcnt(0) lgkmcnt(0)");
    append_isa(3, "s_add_ 0");

    append_isa(4, "global_load_");
    append_isa(5, "buffer_store_");
    append_isa(6, "scratch_load_");
    append_isa(7, "s_waitcnt lkgmcnt(0)");
    append_isa(8, "s_waitcnt vmcnt(2)");
    append_isa(9, "s_waitcnt vmcnt(1)");
    append_isa(10, "s_load_");
    append_isa(11, "s_store_");
    append_isa(12, "s_sendmsg ");
    append_isa(13, "s_waitcnt vmcnt(0) lgkmcnt(0)");
    append_isa(14, "ds_load_");
    append_isa(15, "ds_store_");
    append_isa(16, "ds_load_");
    append_isa(17, "s_waitcnt lgkmcnt(2)");
    append_isa(18, "s_waitcnt lgkmcnt(1)");
    append_isa(19, "flat_load_");
    append_isa(20, "s_waitcnt vmcnt(  0) lgkmcnt(0x0)");  // some weird strings
    append_isa(21, "invalid");

    std::vector<wave_instruction_t> insts{};

    for(size_t j = 0; j < LOOP_CNT; j++)
    {
        for(size_t i = 0; i < isa_map.size(); i++)
        {
            wave_instruction_t inst{};
            inst.pc.addr = i + LINE_OFFSET;
            insts.push_back(inst);
        }
    }

    WaitcntList::wave_t wave{};
    wave.instructions_array = insts.data();
    wave.instructions_size  = insts.size();

    auto data = WaitcntList(9, wave, isa_map);

    std::map<int, std::set<int>> dependencies{};

    for(const auto& [dst, src] : data.mem_unroll)
    {
        auto& dep = dependencies[dst];
        for(const auto& p : src)
            dep.insert(p);
    }

    auto set_equal = [&](int dep, const std::set<int>& set) {
        for(int s : set)
            ASSERT_NE(dependencies.at(dep).find(s), dependencies.at(dep).end());
        ASSERT_EQ(dependencies.at(dep).size(), set.size());
    };

    ASSERT_EQ(dependencies.size(), 6);
    set_equal(8, {4});
    set_equal(9, {5});
    set_equal(13, {6, 10, 11, 12});
    set_equal(17, {14});
    set_equal(18, {15});
    set_equal(20, {16, 19});
}

TEST(att_decoder_waitcnt_test, gfx10)
{
    registration::init_logging();

    WaitcntList::isa_map_t isa_map{};

    auto append_isa = [&](size_t line_number, const char* line) {
        pcinfo_t pc{};
        pc.addr      = line_number + LINE_OFFSET;
        pc.marker_id = 0;

        auto code             = std::make_unique<CodeLine>();
        code->code_line       = std::make_shared<CodeLine::Instruction>();
        code->code_line->inst = line;
        code->line_number     = line_number;

        isa_map.emplace(pc, std::move(code));
    };

    append_isa(0, "buffer_load_");
    append_isa(1, "global_load_");
    append_isa(2, "v_add_ 0");
    append_isa(3, "s_add_ 0");
    append_isa(4, "buffer_store_");
    append_isa(5, "s_waitcnt vmcnt(1)");

    append_isa(6, "scratch_load_");
    append_isa(7, "scratch_store_");
    append_isa(8, "s_wait_alu ");
    append_isa(9, "s_waitcnt vmcnt 0x2");
    append_isa(10, "s_waitcnt vmcnt(1)");
    append_isa(11, "s_waitcnt vscnt(1)");
    append_isa(12, "s_waitcnt vmcnt(0)");
    append_isa(13, "s_waitcnt vscnt(0)");

    append_isa(14, "s_load");
    append_isa(15, "s_store");
    append_isa(16, "s_waitcnt lgkmcnt 0");
    append_isa(17, "s_sendmsg");
    append_isa(18, "s_sendmsg_rtn");
    append_isa(19, "s_waitcnt lgkmcnt 0x2");  // waits on sendmsg_rtn
    append_isa(20, "flat_load_");
    append_isa(21, "flat_store_");
    append_isa(22, "s_waitcnt vmcnt(0) lgkmcnt(0) vscnt(0)");

    append_isa(23, "ds_load");
    append_isa(24, "ds_store");
    append_isa(25, "s_waitcnt lgkmcnt 0x1");
    append_isa(26, "s_waitcnt lgkmcnt 0");
    append_isa(27, "invalid");

    std::vector<wave_instruction_t> insts{};
    for(size_t i = 0; i < isa_map.size(); i++)
    {
        wave_instruction_t inst{};
        inst.pc.addr = i + LINE_OFFSET;
        insts.push_back(inst);
    }

    WaitcntList::wave_t wave{};
    wave.instructions_array = insts.data();
    wave.instructions_size  = insts.size();

    auto data = WaitcntList(10, wave, isa_map);

    std::map<int, std::set<int>> dependencies{};

    for(const auto& [dst, src] : data.mem_unroll)
    {
        auto& dep = dependencies[dst];
        for(const auto& p : src)
            dep.insert(p);
    }

    auto set_equal = [&](int dep, const std::set<int>& set) {
        for(int s : set)
            ASSERT_NE(dependencies.at(dep).find(s), dependencies.at(dep).end());
        ASSERT_EQ(dependencies.at(dep).size(), set.size());
    };

    ASSERT_EQ(dependencies.size(), 10);
    set_equal(5, {0});
    set_equal(10, {1});
    set_equal(11, {4});
    set_equal(12, {6});
    set_equal(13, {7});
    set_equal(16, {14, 15});
    set_equal(19, {17});
    set_equal(22, {18, 20, 21});
    set_equal(25, {23});
    set_equal(26, {24});
}

TEST(att_decoder_waitcnt_test, gfx12)
{
    registration::init_logging();

    WaitcntList::isa_map_t isa_map{};

    auto append_isa = [&](size_t line_number, const char* line) {
        pcinfo_t pc{};
        pc.addr      = line_number + LINE_OFFSET;
        pc.marker_id = 0;

        auto code             = std::make_unique<CodeLine>();
        code->code_line       = std::make_shared<CodeLine::Instruction>();
        code->code_line->inst = line;
        code->line_number     = line_number;

        isa_map.emplace(pc, std::move(code));
    };

    // messages
    append_isa(0, "s_wait_alu ");
    append_isa(1, "s_waitcnt samplecnt(0)");
    append_isa(2, "s_sendmsg ");
    append_isa(3, "s_sendmsg ");
    append_isa(4, "s_sendmsg_rtn");
    append_isa(5, "s_sendmsg_rtn");
    append_isa(6, "s_waitcnt kmcnt(4)");
    append_isa(7, "s_waitcnt kmcnt(2)");
    append_isa(8, "s_waitcnt kmcnt(0)");

    // scalar
    append_isa(9, "s_load_");
    append_isa(10, "s_store_");
    append_isa(11, "s_sendmsg ");
    append_isa(12, "s_waitcnt kmcnt(0)");

    // flat
    append_isa(13, "flat_load_");
    append_isa(14, "flat_store_");
    append_isa(15, "global_load_");
    append_isa(16, "ds_load");
    append_isa(17,
               "s_waitcnt bvhcnt(0) expcnt(0) kmcnt(0) kmcnt(0) loadcnt(0) storecnt(0) "
               "samplecnt(0) dscnt(0)");

    // load/store
    append_isa(18, "global_load");
    append_isa(19, "buffer_load");
    append_isa(20, "global_store");
    append_isa(21, "global_wb");
    append_isa(22, "buffer_store");
    append_isa(23, "scratch_load");
    append_isa(24, "scratch_store");
    append_isa(25, "s_waitcnt loadcnt(2)");
    append_isa(26, "s_waitcnt storecnt(2)");
    append_isa(27, "s_waitcnt storecnt(0) loadcnt(0)");

    // skipped
    append_isa(28, "s_wait_alu");
    append_isa(29, "s_mul ");
    append_isa(30, "v_mul ");

    // ds vs exp vs bvh
    append_isa(31, "ds_store");
    append_isa(32, "_bvh_");
    append_isa(33, "_bvh_");
    append_isa(34, "ds_param_load");
    append_isa(35, "ds_direct");
    append_isa(36, "ds_load");
    append_isa(37, "s_waitcnt dscnt(1)");
    append_isa(38, "s_waitcnt expcnt(0) bvhcnt(0)");
    append_isa(39, "s_waitcnt dscnt(0)");

    append_isa(40, "ds_store");
    append_isa(41, "global_load");
    append_isa(42, "s_wait_idle");
    append_isa(43, "invalid");

    std::vector<wave_instruction_t> insts{};
    for(size_t i = 0; i < isa_map.size(); i++)
    {
        wave_instruction_t inst{};
        inst.pc.addr = i + LINE_OFFSET;
        insts.push_back(inst);
    }

    WaitcntList::wave_t wave{};
    wave.instructions_array = insts.data();
    wave.instructions_size  = insts.size();

    auto data = WaitcntList(12, wave, isa_map);

    std::map<int, std::set<int>> dependencies{};

    for(const auto& [dst, src] : data.mem_unroll)
    {
        auto& dep = dependencies[dst];
        for(const auto& p : src)
            dep.insert(p);
    }

    auto set_equal = [&](int dep, const std::set<int>& set) {
        for(int s : set)
            ASSERT_NE(dependencies.at(dep).find(s), dependencies.at(dep).end());
        ASSERT_EQ(dependencies.at(dep).size(), set.size());
    };

    ASSERT_EQ(dependencies.size(), 12);
    set_equal(6, {2, 3});
    set_equal(7, {4});
    set_equal(8, {5});
    set_equal(12, {9, 10, 11});
    set_equal(17, {13, 14, 15, 16});
    set_equal(25, {18});
    set_equal(26, {20, 21});
    set_equal(27, {19, 22, 23, 24});
    set_equal(37, {31});
    set_equal(38, {32, 33, 34, 35});
    set_equal(39, {36});
    set_equal(42, {40, 41});
}

TEST(att_decoder_waitcnt_test, fail_conditions)
{
    registration::init_logging();

    WaitcntList::isa_map_t isa_map{};

    std::vector<wave_instruction_t> insts{};

    for(size_t i = 0; i < 10; i++)
    {
        wave_instruction_t inst{};
        inst.pc.addr = i + LINE_OFFSET;
        insts.push_back(inst);
    }

    WaitcntList::wave_t wave{};
    wave.instructions_array = insts.data();
    wave.instructions_size  = insts.size();

    // It should give warning and return
    ASSERT_TRUE(WaitcntList(9, wave, isa_map).mem_unroll.empty());
    ASSERT_TRUE(WaitcntList(10, wave, isa_map).mem_unroll.empty());
    ASSERT_TRUE(WaitcntList(12, wave, isa_map).mem_unroll.empty());

    // it cant operate on invalid gfxip
    try
    {
        WaitcntList(-1, wave, isa_map);
        // fail
        ASSERT_TRUE(false);
    } catch(std::runtime_error& e)
    {
        // pass
    }
}

};  // namespace att_wrapper
};  // namespace rocprofiler
