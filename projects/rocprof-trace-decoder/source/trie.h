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
#include <stdint.h>
#include <algorithm>
#include <cassert>
#include <cctype>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <regex>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>
#include "common.hpp"

enum class InstCategory
{
    SKIP = WaveInstCategory::NONE,
    SMEM = WaveInstCategory::SMEM,
    SALU = WaveInstCategory::SALU,
    VMEM = WaveInstCategory::VMEM,
    FLAT = WaveInstCategory::FLAT,
    LDS = WaveInstCategory::LDS,
    VALU = WaveInstCategory::VALU,
    IMMED = WaveInstCategory::IMMED,
    BVH = WaveInstCategory::BVH,
    DONT_KNOW = WaveInstCategory::WAVE_NOT_FINISHED + 1,
    BRANCH,
    BARRIER_WAIT_EXCLUDED,
    MFMA_SCALE,
    GFX9_BARRIER,
    LAST
};

inline bool operator==(const InstCategory& cat, uint64_t value) { return static_cast<uint64_t>(cat) == value; }
inline bool operator==(uint64_t value, const InstCategory& cat) { return static_cast<uint64_t>(cat) == value; }

class Trie
{
public:
    Trie() = default;
    ~Trie();

    InstCategory type_from_trie(const std::string_view inst);

    static Trie root_trie;

    static InstCategory inst_type(const std::string_view line, int gfxip)
    {
        if (line.find("branch") != std::string::npos) return InstCategory::BRANCH;

        InstCategory type = root_trie.type_from_trie(line);

        if (gfxip == 9 && type == InstCategory::VALU)
        {
            // Check for scale MFMA
            if (line.find("v_mfma_scale_") != std::string::npos) type = InstCategory::MFMA_SCALE;
        }
        else if (gfxip == 9 && type == InstCategory::IMMED)
        {
            if (line.find("s_barrier") == 0) type = InstCategory::GFX9_BARRIER;
        }
        else if (gfxip == 12 && type == InstCategory::LDS)
        {
            if (line.find("ds_bvh") == 0) return InstCategory::BVH;
        }

        return type;
    }

private:
    void add_type(const std::string& inst_header, InstCategory type);

    bool bInit = false;
    InstCategory type = InstCategory::LAST;
    std::unordered_map<char, Trie*> paths; //  Change to smart pointer
};
