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

#include "trace_parser.hpp"
#include <cassert>
#include <cstring>
#include <mutex>
#include <sstream>
#include "gfx10/gfx10token.h"
#include "gfx10/gfx10wave.h"
#include "gfx11/gfx11token.h"
#include "gfx11/gfx11wave.h"
#include "gfx12/gfx12token.h"
#include "gfx12/gfx12wave.h"
#include "gfx9/gfx9token.h"
#include "gfx9/gfx9wave.h"
#include "stitch/stitch.hpp"

void WaveDataInternal::lookbackpcs(class CSRegisterHandler& reg)
{
    for (size_t index : unattrib_pcs)
        if (index < instructions.size())
        {
            auto& inst = instructions.at(index);
            if (inst.pc.code_object_id == 0) inst.pc = reg.get_wave_start_delayed(inst.pc.address);
        }

    for (auto& [_, info] : pc_infos)
    {
        if (info.code_object_id == 0) info = reg.get_wave_start_delayed(info.address);
    }
}

std::unique_ptr<SQTTParser> AnalyseBinary_GFX9_internal(
    CppReturnInfo& info,
    const uint8_t* tokendata,
    uint64_t buffersize,
    int target_cu,
    class Stitcher& stitch,
    bool double_buffer
)
{
    stitch.setgfxip(9);

    auto generator = gfx9::MITokenGenerator(tokendata, buffersize, 0, 0);
    auto parser = std::make_unique<gfx9::MISQTTParser>(target_cu, double_buffer);
    parser->sqtt_simd_analysis(info, generator, stitch);

    return parser;
}

std::unique_ptr<SQTTParser> AnalyseBinary_GFX10_internal(
    CppReturnInfo& info, const uint8_t* tokendata, uint64_t buffersize, class Stitcher& stitch
)
{
    stitch.setgfxip(10);

    auto generator = gfx10::TokenGenerator(tokendata, buffersize, 0, 0);
    auto parser = std::make_unique<RDNASQTParser>();
    parser->sqtt_simd_analysis(info, generator, stitch);

    return parser;
}

std::unique_ptr<SQTTParser> AnalyseBinary_GFX11_internal(
    CppReturnInfo& info, const uint8_t* tokendata, uint64_t buffersize, class Stitcher& stitch
)
{
    stitch.setgfxip(11);

    auto generator = gfx11::TokenGenerator(tokendata, buffersize, 0, 0);
    auto parser = std::make_unique<RDNASQTParser>();
    parser->sqtt_simd_analysis(info, generator, stitch);

    return parser;
}

std::unique_ptr<SQTTParser> AnalyseBinary_GFX12_internal(
    CppReturnInfo& info, const uint8_t* tokendata, uint64_t buffersize, class Stitcher& stitch
)
{
    stitch.setgfxip(12);

    auto generator = gfx12::TokenGenerator(tokendata, buffersize, 0, 0);
    auto parser = std::make_unique<RDNASQTParser>();
    parser->sqtt_simd_analysis(info, generator, stitch);

    return parser;
}

/*
void applyGenerator(
    CppReturnInfo& info,
    SQTTParser& parser,
    Stitcher& stitch,
    const uint8_t* buffer,
    uint64_t BUFFER_SIZE,
    int64_t globaltime,
    int64_t basetime
)
{
    if(stitch.getgfxip() == 9)
    {
        auto generator = gfx9::MITokenGenerator(buffer, BUFFER_SIZE, globaltime, basetime);
        parser.sqtt_simd_analysis(info, generator, stitch);
        return;
    }

    std::unique_ptr<NaviTokenGenerator> generator{nullptr};
    if(stitch.getgfxip() == 10)
        generator = std::make_unique<gfx10::TokenGenerator>(buffer, BUFFER_SIZE, globaltime, basetime);
    else if(stitch.getgfxip() == 11)
        generator = std::make_unique<gfx11::TokenGenerator>(buffer, BUFFER_SIZE, globaltime, basetime);
    else if(stitch.getgfxip() == 12)
        generator = std::make_unique<gfx12::TokenGenerator>(buffer, BUFFER_SIZE, globaltime, basetime);

    if(!generator) return;

    parser.sqtt_simd_analysis(info, *generator, stitch);
}
*/

// If target_cu < 0, find target_cu from software header
std::unique_ptr<SQTTParser> AnalyseBinary_internal(
    CppReturnInfo& info, const uint8_t* buffer, uint64_t BUFFER_SIZE, int gfx9_target_cu, class Stitcher& stitch
)
{
    if (gfx9_target_cu < 0)
    {
        auto gfx9_header = *reinterpret_cast<const rocprof_trace_decoder_gfx9_header_t*>(buffer);
        if ((gfx9_header.legacy_version == 0 || gfx9_header.legacy_version == 0x11) &&
            (gfx9_header.gfx9_version2 >= 4 && gfx9_header.gfx9_version2 <= 6))
        {
            buffer += sizeof(rocprof_trace_decoder_gfx9_header_t);
            if (BUFFER_SIZE <= sizeof(rocprof_trace_decoder_gfx9_header_t)) return nullptr;

            BUFFER_SIZE -= sizeof(rocprof_trace_decoder_gfx9_header_t);
            return AnalyseBinary_GFX9_internal(
                info, buffer, BUFFER_SIZE, gfx9_header.DCU, stitch, gfx9_header.double_buffer
            );
        }
        else if (gfx9_header.legacy_version != 0)
        {
            auto hw_header = *reinterpret_cast<const header_type*>(buffer);

            if (hw_header.version == 4)
                return AnalyseBinary_GFX12_internal(info, buffer, BUFFER_SIZE, stitch);
            else if (hw_header.version == 3)
                return AnalyseBinary_GFX11_internal(info, buffer, BUFFER_SIZE, stitch);
            else if (hw_header.version == 2 || hw_header.version == 1)
                return AnalyseBinary_GFX10_internal(info, buffer, BUFFER_SIZE, stitch);
        }
    }
    else { return AnalyseBinary_GFX9_internal(info, buffer, BUFFER_SIZE, gfx9_target_cu, stitch, false); }

    return nullptr;
}

pcinfo_t CodeobjTableTranslator::ToPcV2(uint64_t pc)
{
    pcinfo_t pcinfo{.address = pc, .code_object_id = 0};
    try
    {
        address_range_t codeobj;
        if (this->find_codeobj_in_range(pc, codeobj))
        {
            pcinfo.code_object_id = codeobj.id;
            pcinfo.address = pc - codeobj.vbegin;
        }
    }
    catch (const std::exception&)
    {}

    return pcinfo;
}
