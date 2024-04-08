// MIT License
//
// Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.
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

#include <sys/mman.h>
#include <algorithm>
#include <atomic>
#include <cassert>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <iostream>
#include <map>
#include <mutex>
#include <optional>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include "code_object_track.hpp"

void
CodeobjRecorder::Load(uint64_t           addr,
                      uint64_t           load_size,
                      const std::string& URI,
                      uint64_t           mem_addr,
                      uint64_t           mem_size,
                      uint64_t           id)
{
    Load(std::make_shared<CodeobjCaptureInstance>(
        addr, load_size, URI, mem_addr, mem_size, id, capture_mode));
}

void
CodeobjCaptureInstance::copyCodeobjFromFile(uint64_t offset, uint64_t size, const std::string& path)
{
    std::ifstream file(path, std::ios::in | std::ios::binary);
    if(!file)
    {
        printf("could not open `%s'\n", path.c_str());
        return;
    }

    if(!size)
    {
        file.ignore(std::numeric_limits<std::streamsize>::max());
        size_t bytes = file.gcount();
        file.clear();

        if(bytes < offset)
        {
            printf("invalid uri `%s' (file size < offset)\n", path.c_str());
            return;
        }
        size = bytes - offset;
    }

    file.seekg(offset, std::ios_base::beg);
    buffer.resize(size);
    file.read(&buffer[0], size);
}

void CodeobjCaptureInstance::copyCodeobjFromMemory(uint64_t, uint64_t)
{
    // buffer.resize(mem_size);
    // std::memcpy(buffer.data(), (uint64_t*)mem_addr, mem_size);
}

std::pair<size_t, size_t>
CodeobjCaptureInstance::parse_uri()
{
    const std::string protocol_delim{"://"};

    size_t protocol_end = URI.find(protocol_delim);
    protocol            = URI.substr(0, protocol_end);
    protocol_end += protocol_delim.length();

    std::transform(protocol.begin(), protocol.end(), protocol.begin(), [](unsigned char c) {
        return std::tolower(c);
    });

    std::string path;
    size_t      path_end = URI.find_first_of("#?", protocol_end);
    if(path_end != std::string::npos)
    {
        path = URI.substr(protocol_end, path_end++ - protocol_end);
    }
    else
    {
        path = URI.substr(protocol_end);
    }

    /* %-decode the string.  */
    decoded_path = std::string{};
    decoded_path.reserve(path.length());
    for(size_t i = 0; i < path.length(); ++i)
    {
        if(path[i] == '%' && std::isxdigit(path[i + 1]) && std::isxdigit(path[i + 2]))
        {
            decoded_path += std::stoi(path.substr(i + 1, 2), 0, 16);
            i += 2;
        }
        else
        {
            decoded_path += path[i];
        }
    }

    /* Tokenize the query/fragment.  */
    std::vector<std::string> tokens;
    size_t                   pos, last = path_end;
    while((pos = URI.find('&', last)) != std::string::npos)
    {
        tokens.emplace_back(URI.substr(last, pos - last));
        last = pos + 1;
    }
    if(last != std::string::npos) tokens.emplace_back(URI.substr(last));

    /* Create a tag-value map from the tokenized query/fragment.  */
    std::unordered_map<std::string, std::string> params;
    std::for_each(tokens.begin(), tokens.end(), [&](std::string& token) {
        size_t delim = token.find('=');
        if(delim != std::string::npos)
        {
            params.emplace(token.substr(0, delim), token.substr(delim + 1));
        }
    });

    size_t offset = 0;
    size_t size   = 0;

    if(auto offset_it = params.find("offset"); offset_it != params.end())
        offset = std::stoul(offset_it->second, nullptr, 0);

    if(auto size_it = params.find("size"); size_it != params.end())
    {
        if(!(size = std::stoul(size_it->second, nullptr, 0))) throw std::exception();
    }

    return {offset, size};
}

void
CodeobjCaptureInstance::reset(codeobj_capture_mode_t mode)
{
    if(static_cast<int>(mode) <= static_cast<int>(capture_mode)) return;

    capture_mode = mode;
    if(!buffer.empty()) return;

    size_t offset, size;
    try
    {
        std::tie(offset, size) = parse_uri();
    } catch(...)
    {
        std::cerr << "Error parsing URI " << URI << std::endl;
        return;
    }

    if(protocol == "file")
    {
        if(mode == ROCPROFILER_CODEOBJ_CAPTURE_COPY_FILE_AND_MEMORY)
            copyCodeobjFromFile(offset, size, decoded_path);
    }
    else if(protocol == "memory")
    {
        if(mode != ROCPROFILER_CODEOBJ_CAPTURE_SYMBOLS_ONLY)
            copyCodeobjFromMemory(mem_addr, mem_size);
    }
    else
    {
        printf("\"%s\" protocol not supported\n", protocol.c_str());
    }
}
