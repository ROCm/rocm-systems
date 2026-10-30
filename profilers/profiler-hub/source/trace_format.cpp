// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "trace_format.hpp"
#include "pytorch/kineto_format.hpp"
#include "utility.hpp"

#include <zlib.h>

#include <cstdio>
#include <cstring>
#include <memory>
#include <vector>

namespace profiler_hub
{
namespace
{
constexpr std::size_t k_text_probe_bytes = 64 * 1024;
constexpr std::size_t k_sqlite_magic_len = 16;

using file_ptr_t = std::unique_ptr<std::FILE, common::file_closer>;

bool
read_plain_prefix(std::FILE* fp, std::string& text)
{
    text.resize(k_text_probe_bytes);
    text.resize(std::fread(text.data(), 1, text.size(), fp));
    return !text.empty();
}

bool
read_gzip_prefix(std::FILE* fp, std::string& text)
{
    z_stream zs{};
    if(inflateInit2(&zs, 15 + 16) != Z_OK) return false;

    std::vector<unsigned char> in(16 * 1024);
    text.resize(k_text_probe_bytes);
    std::size_t produced = 0;
    bool        ok       = true;

    while(produced < text.size())
    {
        if(zs.avail_in == 0)
        {
            zs.avail_in = static_cast<uInt>(std::fread(in.data(), 1, in.size(), fp));
            zs.next_in  = in.data();
            if(zs.avail_in == 0) break;
        }
        zs.next_out  = reinterpret_cast<Bytef*>(text.data()) + produced;
        zs.avail_out = static_cast<uInt>(text.size() - produced);
        const int r  = inflate(&zs, Z_NO_FLUSH);
        produced     = text.size() - zs.avail_out;
        if(r == Z_STREAM_END) break;
        if(r != Z_OK)
        {
            ok = false;
            break;
        }
    }

    inflateEnd(&zs);
    text.resize(produced);
    return ok && produced > 0;
}
}  // namespace

ph_trace_format_t
detect_trace_format(const std::string& path)
{
    try
    {
        file_ptr_t fp{ std::fopen(path.c_str(), "rb") };
        if(!fp) return PH_TRACE_FORMAT_UNKNOWN;

        unsigned char head[k_sqlite_magic_len] = {};
        const auto    n = std::fread(head, 1, sizeof(head), fp.get());
        if(n == sizeof(head) &&
           std::memcmp(head, "SQLite format 3\0", k_sqlite_magic_len) == 0)
        {
            return PH_TRACE_FORMAT_SQLITE;
        }

        std::rewind(fp.get());
        const bool  gzip = n >= 2 && head[0] == 0x1f && head[1] == 0x8b;
        std::string text;
        if(!(gzip ? read_gzip_prefix(fp.get(), text) : read_plain_prefix(fp.get(), text)))
        {
            return PH_TRACE_FORMAT_UNKNOWN;
        }
        return pytorch::looks_like_kineto(text) ? PH_TRACE_FORMAT_PYTORCH_KINETO
                                                : PH_TRACE_FORMAT_UNKNOWN;
    } catch(...)
    {
        return PH_TRACE_FORMAT_UNKNOWN;
    }
}
}  // namespace profiler_hub
