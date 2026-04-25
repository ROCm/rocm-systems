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

#include <memory>
#include <optional>
#include <shared_mutex>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>
#include "trace_decoder_api.h"
#include "trace_parser.hpp"
#include "trie.h"

typedef std::string String;
typedef std::string_view StringView;
typedef std::shared_mutex SharedMutex;

struct assemblyLine
{
    String line{};
    String loc{};
    InstCategory cat{};
    pcinfo_t addr{};
    pcinfo_t next{};
    pcinfo_t to_line{};
    bool parsed{};
};
typedef std::shared_ptr<assemblyLine> assemblyLinePtr;

class ICodeServicer
{
public:
    virtual assemblyLine GetInstruction(pcinfo_t addr, int gfxip) = 0;
    virtual ~ICodeServicer(){};
};

inline std::pair<StringView, StringView> split2(StringView view, StringView s)
{
    static std::string empty{""};

    size_t pos = view.find(s);
    if (pos == std::string::npos || pos + s.size() >= view.size()) return {view, empty};
    return {view.substr(0, pos), view.substr(pos + s.size())};
}

inline std::string_view strip(std::string_view input)
{
    if (input.size() == 0) return input;
    const char* start_it = input.data();
    const char* end_it = input.data() + input.size() - 1;

    while (start_it != end_it && std::isspace(*start_it)) ++start_it;
    while (end_it != start_it && std::isspace(*end_it)) --end_it;
    return std::string_view(start_it, static_cast<uint64_t>(end_it + 1 - start_it));
}

inline std::vector<StringView> splitv(StringView view, char s)
{
    if (!view.size()) return {view};

    std::vector<StringView> ret;

    size_t first = 0;
    bool bIsQuotes = false;

    for (size_t idx = 0; idx < view.size(); idx++)
    {
        if (view[idx] == '\"')
            bIsQuotes = !bIsQuotes;
        else if (!bIsQuotes && view[idx] == s)
        {
            ret.push_back(StringView(view.data() + first, idx - first));
            first = idx + 1;
        }
    }
    ret.push_back(StringView(view.data() + first, view.size() - first));

    for (size_t i = 0; i < ret.size(); i++) ret.at(i) = strip(ret.at(i));

    return ret;
}

// Translates PC values to instructions, for auto captured ISA
class PCTranslator
{
public:
    PCTranslator(std::vector<assemblyLinePtr>& _code, std::shared_ptr<ICodeServicer>& _service, int _gfxip);
    ~PCTranslator() {}

    assemblyLinePtr jump(const assemblyLine& source);
    assemblyLinePtr getcode(pcinfo_t addr);

    pcinfo_t getjump_loc(const assemblyLine& line);

    bool try_match_swapped(const Instruction& first, const Instruction& second, const assemblyLine& line)
    {
        return second.category == line.cat && first.category == getcode(line.next)->cat;
    }

    std::vector<assemblyLinePtr>& code;
    std::shared_ptr<ICodeServicer> service;
    std::unordered_map<pcinfo_t, assemblyLinePtr> jump_map;
    std::unordered_map<pcinfo_t, assemblyLinePtr> addrmap;

    SharedMutex code_mut;
    SharedMutex jump_mut;

    const int gfxip;
};

typedef std::vector<std::pair<int, pcinfo_t>> barrier_list_t;

// Exposed for unit testing
void insert_gfx12_barrier_wait(WaveDataInternal& wave, const barrier_list_t& barriers);

class Stitcher
{
public:
    Stitcher(std::shared_ptr<ICodeServicer> service, rocprof_trace_decoder_trace_callback_t _callback, void* cbdata);

    void stitch(class WaveDataInternal& wave);
    std::vector<assemblyLinePtr> raw_code;
    void setgfxip(uint64_t gfxip);
    uint64_t getgfxip() const
    {
        assert(gfxip != 0);
        return gfxip;
    }

    void sendOtherSimd(std::vector<att_other_simd_t>& vec)
    {
        sendVec(ROCPROFILER_THREAD_TRACE_DECODER_RECORD_INST_OTHER_SIMD, vec);
    }
    void sendShaderdata(std::vector<att_shader_data_t>& vec)
    {
        sendVec(ROCPROFILER_THREAD_TRACE_DECODER_RECORD_SHADERDATA, vec);
    }
    void sendRealtime(std::vector<att_decoder_realtime_t>& vec)
    {
        sendVec(ROCPROFILER_THREAD_TRACE_DECODER_RECORD_REALTIME, vec);
    };
    void sendOccupancy(std::vector<occupancy_info_t>& vec)
    {
        sendVec(ROCPROFILER_THREAD_TRACE_DECODER_RECORD_OCCUPANCY, vec);
    };

private:
    std::pair<size_t, barrier_list_t> stitchWave(class WaveDataInternal& wave);

    template <typename Type>
    void sendVec(rocprofiler_thread_trace_decoder_record_type_t type, std::vector<Type>& record)
    {
        callback(type, record.data(), record.size(), cbdata);
        record.clear();
    };

    std::shared_ptr<ICodeServicer> codeobj_service{};
    std::unordered_map<int, int> jumps{};
    std::unique_ptr<PCTranslator> pctranslator{nullptr};

    std::once_flag stitch_flag{}, incomp_flag{}, gfx_flag{};

    rocprof_trace_decoder_trace_callback_t callback{};

    void* const cbdata;
    uint64_t gfxip = 0;
};
