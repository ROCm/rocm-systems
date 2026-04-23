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
#include <deque>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

#include "trace_parser.hpp"

#define get_bits(low, high) ((val >> (low)) & ((uint64_t(1) << ((high) - (low) + 1)) - 1))
#define CHECK_WAVE()                                                                                                   \
    if (wave > 9) throw std::exception();

namespace gfx9
{

struct group_id
{
    int8_t cu, wave, simd, sh;
    group_id(uint64_t val)
    {
        cu = get_bits(6, 9);
        sh = get_bits(5, 5);
        wave = get_bits(10, 13);
        simd = get_bits(14, 15);

        CHECK_WAVE();
    }
};

struct Wave : public group_id
{
    Wave(uint64_t val) : group_id(val)
    {
        pipe = get_bits(16, 17);
        me = get_bits(18, 19);
        count = get_bits(22, 28);
    };

    int16_t pipe, me, count;
};

struct Regfields
{
    uint32_t regdata;
    uint16_t regaddr;
    int8_t pipe, me;
};

struct Reg : public Regfields
{
    Reg(uint64_t val)
    {
        pipe = get_bits(5, 6);
        me = (get_bits(7, 8) + 1) & 0x1;
        regaddr = get_bits(16, 31);
        regdata = get_bits(32, 63);
    }
};

struct RegCs : public Regfields
{
    RegCs(uint64_t val)
    {
        pipe = get_bits(5, 6);
        me = (get_bits(7, 8) + 1) & 0x1;
        regaddr = get_bits(9, 15);
        regdata = get_bits(16, 47);
    }
};

struct Misc
{
    Misc(uint64_t val)
    {
        sh = get_bits(12, 12);
        misc_type = get_bits(13, 15);
    }

    int16_t sh, misc_type;
};

struct MsgInst
{
    MsgInst(uint64_t val)
    {
        wave = get_bits(5, 8);
        simd = get_bits(9, 10);
        inst_type = get_bits(11, 15);

        CHECK_WAVE();
    }
    int16_t inst_type, wave, simd;
};

struct MsgInstPc
{
    MsgInstPc(uint64_t val)
    {
        wave = get_bits(5, 8);
        simd = get_bits(9, 10);
        err = get_bits(15, 15);
        pc = get_bits(16, 63);

        CHECK_WAVE();
    }
    uint64_t pc;
    int16_t wave, simd, err;
};

struct UserData : public group_id
{
    UserData(uint64_t val) : group_id(val) { data = get_bits(16, 47); }
    uint32_t data;
};

struct Issue
{
    Issue(uint64_t val)
    {
        simd = get_bits(5, 6);
        for (int i = 0; i < 10; i++) inst[i] = get_bits(2 * i + 8, 2 * i + 9);
    }
    int8_t simd;
    int8_t inst[10];
};

struct MsgPerf
{
    MsgPerf(uint64_t val)
    {
        sh = get_bits(5, 5);
        cu = get_bits(6, 9);
        cntr_bank = (uint8_t) get_bits(10, 11);
        cntr[0] = (uint16_t) get_bits(12, 24);
        cntr[1] = (uint16_t) get_bits(25, 37);
        cntr[2] = (uint16_t) get_bits(38, 50);
        cntr[3] = (uint16_t) get_bits(51, 63);
    }
    int8_t sh, cu, cntr_bank;
    uint16_t cntr[4];
};

union TokenFields
{
    TokenFields(){};
    Wave wave;
    Misc misc;
    Reg reg;
    RegCs regcs;
    Issue issue;
    MsgPerf perf;
    MsgInst inst;
    MsgInstPc inst_pc;
    UserData userdata;
};

enum sqtt_token_type_t
{
    TOKEN_MISC = 0,
    TOKEN_TIME,
    TOKEN_REG,
    TOKEN_WAVE_START,
    TOKEN_WAVE_ALLOC,
    TOKEN_REG_CS,
    TOKEN_WAVE_END,
    TOKEN_EVENT,
    TOKEN_EVENT_CS,
    TOKEN_EVENT_GFX1,
    TOKEN_INST,
    TOKEN_INST_PC,
    TOKEN_SHADERDATA,
    TOKEN_ISSUE,
    TOKEN_PERF,
    TOKEN_REG_CS_PRIV
};

class Token
{
public:
    Token(uint64_t _type, uint64_t val) : type(_type)
    {
        if (type == TOKEN_MISC)
            fields.misc = Misc(val);
        else if (type == TOKEN_TIME)
            time = get_bits(16, 63);
        else if (type == TOKEN_REG)
            fields.reg = Reg(val);
        else if (type == TOKEN_WAVE_START || type == TOKEN_WAVE_ALLOC || type == TOKEN_WAVE_END)
            fields.wave = Wave(val);
        else if (type == TOKEN_REG_CS || type == TOKEN_REG_CS_PRIV)
            fields.regcs = RegCs(val);
        else if (type == TOKEN_INST)
            fields.inst = MsgInst(val);
        else if (type == TOKEN_INST_PC)
            fields.inst_pc = MsgInstPc(val);
        else if (type == TOKEN_SHADERDATA)
            fields.userdata = UserData(val);
        else if (type == TOKEN_ISSUE)
            fields.issue = Issue(val);
        else if (type == TOKEN_PERF)
            fields.perf = MsgPerf(val);

        if (type == 0)
            delta = get_bits(4, 11);
        else
            delta = get_bits(4, 4);
    }

    int64_t time  : 46;
    int64_t delta : 12;
    int64_t type  : 6;

    TokenFields fields;
};

class MITokenGenerator : public TokenGenerator
{
public:
    MITokenGenerator(const uint8_t* _buffer, size_t size, int64_t _global_time, int64_t _base_time) :
    TokenGenerator(_buffer, size, _global_time, _base_time), cur_len(0)
    {
        if (!_buffer || !size) throw std::exception();
    }

    Token next();
    bool valid() { return !lookahead.empty() || cur_len < BUFFER_SIZE; }

private:
    void patch_time();

    std::deque<Token> lookahead{};
    size_t cur_len = 0;
};

}; // namespace gfx9
