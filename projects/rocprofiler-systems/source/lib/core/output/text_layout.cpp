// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "output/text_layout.hpp"

#include <spdlog/fmt/fmt.h>

#include <algorithm>

namespace rocprofsys::output
{

namespace
{
// UTF-8 continuation bytes match the bit pattern 10xxxxxx; the mask
// extracts the top two bits, the bits value is what they equal when
// the byte is a continuation byte (not the start of a code point).
inline constexpr unsigned char UTF8_CONTINUATION_MASK = 0xC0;
inline constexpr unsigned char UTF8_CONTINUATION_BITS = 0x80;

// Singletons and pairs render verbatim; runs of three or more
// contiguous ids collapse to `min-max` to keep the marker short.
inline constexpr std::size_t MIN_RUN_FOR_RANGE_COMPRESSION = 3;
}  // namespace

std::vector<std::string>
wrap_to_width(std::string_view content, std::size_t width)
{
    std::vector<std::string> out;
    if(content.empty())
    {
        out.emplace_back();
        return out;
    }
    if(width == 0) width = 1;
    while(!content.empty())
    {
        if(content.size() <= width)
        {
            out.emplace_back(content);
            break;
        }
        std::size_t cut = width;
        // UTF-8 backoff: never split inside a multi-byte code point.
        while(cut > 0 &&
              (static_cast<unsigned char>(content[cut]) & UTF8_CONTINUATION_MASK) ==
                  UTF8_CONTINUATION_BITS)
            --cut;
        // ws==0 would yield an empty leading chunk and never advance;
        // treat as "no break" and fall through to byte chunking.
        const auto ws             = content.rfind(' ', cut);
        bool       broke_on_space = false;
        if(ws != std::string_view::npos && ws > 0)
        {
            cut            = ws;
            broke_on_space = true;
        }
        if(cut == 0) cut = 1;  // pathological: force progress
        out.emplace_back(content.substr(0, cut));
        const std::size_t next_start = cut + (broke_on_space ? 1 : 0);
        content.remove_prefix(std::min(next_start, content.size()));
    }
    return out;
}

std::string
format_gpu_ids(const std::vector<int>& gpu_ids)
{
    if(gpu_ids.empty()) return "";

    std::vector<int> sorted = gpu_ids;
    std::sort(sorted.begin(), sorted.end());
    sorted.erase(std::unique(sorted.begin(), sorted.end()), sorted.end());

    std::vector<std::string> tokens;
    for(std::size_t i = 0; i < sorted.size();)
    {
        std::size_t j = i;
        while(j + 1 < sorted.size() && sorted[j + 1] == sorted[j] + 1)
            ++j;
        const std::size_t run_len = j - i + 1;
        if(run_len >= MIN_RUN_FOR_RANGE_COMPRESSION)
            tokens.push_back(fmt::format("{}-{}", sorted[i], sorted[j]));
        else
            for(std::size_t k = i; k <= j; ++k)
                tokens.push_back(fmt::format("{}", sorted[k]));
        i = j + 1;
    }

    std::string out = ":";
    if(tokens.size() <= MAX_RENDERED_GPU_IDS)
    {
        for(std::size_t k = 0; k < tokens.size(); ++k)
        {
            if(k > 0) out += ",";
            out += tokens[k];
        }
    }
    else
    {
        const std::size_t extra = tokens.size() - MAX_RENDERED_GPU_IDS;
        for(std::size_t k = 0; k < MAX_RENDERED_GPU_IDS; ++k)
        {
            if(k > 0) out += ",";
            out += tokens[k];
        }
        out += fmt::format(",...(+{} more)", extra);
    }
    return out;
}

std::string
format_duration(std::chrono::nanoseconds dur)
{
    if(dur.count() <= 0) return "?";
    const double seconds = std::chrono::duration<double>(dur).count();
    return fmt::format("{:.2f}s", seconds);
}

}  // namespace rocprofsys::output
