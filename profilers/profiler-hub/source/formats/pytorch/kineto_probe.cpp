// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "kineto_probe.hpp"
#include "kineto_format.hpp"
#include "utility.hpp"

#include <zlib.h>

#include <cstdio>
#include <memory>
#include <vector>

namespace profiler_hub::pytorch
{
namespace
{
constexpr std::size_t k_text_probe_bytes = std::size_t{ 64 } * 1024;

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

    std::vector<unsigned char> in(std::size_t{ 16 } * 1024);
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

bool
read_prefix(const std::string& path, std::string& text)
{
    const file_ptr_t fp{ std::fopen(path.c_str(), "rb") };
    if(!fp) return false;

    unsigned char head[2] = {};
    const auto    n       = std::fread(head, 1, sizeof(head), fp.get());
    if(std::fseek(fp.get(), 0, SEEK_SET) != 0) return false;
    const bool gzip = n == sizeof(head) && head[0] == 0x1f && head[1] == 0x8b;
    return gzip ? read_gzip_prefix(fp.get(), text) : read_plain_prefix(fp.get(), text);
}
}  // namespace

bool
kineto_prefix_matches(const std::string& path)
{
    try
    {
        std::string text;
        return read_prefix(path, text) && looks_like_kineto(text);
    } catch(...)
    {
        return false;
    }
}

reader_types::file_probe_result_t
probe_kineto(const std::string& path)
{
    reader_types::file_probe_result_t result;
    result.status       = PH_RESULT_SUCCESS;
    result.trace_schema = reader_types::trace_schema_t::pytorch_kineto;
    try
    {
        std::string text;
        if(read_prefix(path, text))
        {
            result.schema_version.major = schema_version_in_prefix(text);
        }
    } catch(...)
    {
        result.schema_version.major = 0;
    }
    return result;
}
}  // namespace profiler_hub::pytorch
