// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "output/text_layout.hpp"

#include <spdlog/fmt/fmt.h>

#include <algorithm>
#include <vector>

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

std::size_t
display_width(std::string_view text)
{
    std::size_t columns = 0;
    for(char byte : text)
        if((static_cast<unsigned char>(byte) & UTF8_CONTINUATION_MASK) !=
           UTF8_CONTINUATION_BITS)
            ++columns;
    return columns;
}

std::string
repeat_glyph(std::string_view glyph, std::size_t count)
{
    std::string out;
    out.reserve(glyph.size() * count);
    for(std::size_t index = 0; index < count; ++index)
        out.append(glyph);
    return out;
}

std::string
summarize_command(std::string_view command)
{
    const std::string cleaned = strip_terminal_control_chars(command);
    if(cleaned.empty()) return {};

    const auto  token_end = cleaned.find_first_of(" \t");
    std::string program =
        (token_end == std::string::npos) ? cleaned : cleaned.substr(0, token_end);

    const auto slash = program.find_last_of('/');
    if(slash != std::string::npos) program = program.substr(slash + 1);
    return program;
}

std::string
format_gpu_ids(const std::vector<int>& gpu_ids)
{
    if(gpu_ids.empty()) return "";

    std::vector<int> sorted = gpu_ids;

    std::ranges::sort(sorted);
    const auto duplicates = std::ranges::unique(sorted);
    sorted.erase(duplicates.begin(), duplicates.end());

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

std::string
strip_terminal_control_chars(std::string_view s)
{
    std::string out;
    out.reserve(s.size());
    for(std::size_t i = 0; i < s.size();)
    {
        const auto byte = static_cast<unsigned char>(s[i]);
        // CSI sequence: ESC [ ... <final byte in 0x40..0x7E>
        if(byte == 0x1B && i + 1 < s.size() && s[i + 1] == '[')
        {
            std::size_t j = i + 2;
            while(j < s.size())
            {
                const auto fb = static_cast<unsigned char>(s[j]);
                if(fb >= 0x40 && fb <= 0x7E)
                {
                    ++j;
                    break;
                }
                ++j;
            }
            i = j;
            continue;
        }
        // Drop other C0 controls + DEL; keep tab (0x09) and newline (0x0A)
        // so downstream layout still sees structure.
        if((byte < 0x20 && byte != 0x09 && byte != 0x0A) || byte == 0x7F)
        {
            ++i;
            continue;
        }
        out.push_back(static_cast<char>(byte));
        ++i;
    }
    return out;
}

}  // namespace rocprofsys::output
