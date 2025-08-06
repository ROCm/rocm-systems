// MIT License
//
// Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
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

#include <rocprofiler-sdk/cxx/serialization.hpp>

#include <unordered_map>
#include <vector>

namespace rocprofiler
{
namespace tool
{
struct node_info
{
    node_info()                     = default;
    ~node_info()                    = default;
    node_info(const node_info&)     = default;
    node_info(node_info&&) noexcept = default;
    node_info& operator=(const node_info&) = default;
    node_info& operator=(node_info&&) noexcept = default;

    uint64_t    id            = 0;
    uint64_t    hash          = 0;
    std::string machine_id    = {};
    std::string system_name   = {};
    std::string hostname      = {};
    std::string release       = {};
    std::string version       = {};
    std::string hardware_name = {};
    std::string domain_name   = {};
};

node_info&
read_node_info(node_info& _info);

node_info
read_node_info();

using node_info_vec_t = std::vector<node_info>;
}  // namespace tool
}  // namespace rocprofiler

namespace cereal
{
#define SAVE_DATA_FIELD(FIELD) ar(make_nvp(#FIELD, data.FIELD))
#define LOAD_DATA_FIELD(FIELD) ar(make_nvp(#FIELD, data.FIELD))

template <typename ArchiveT>
void
save(ArchiveT& ar, const ::rocprofiler::tool::node_info& data)
{
    SAVE_DATA_FIELD(id);
    SAVE_DATA_FIELD(hash);
    SAVE_DATA_FIELD(machine_id);
    SAVE_DATA_FIELD(system_name);
    SAVE_DATA_FIELD(hostname);
    SAVE_DATA_FIELD(release);
    SAVE_DATA_FIELD(version);
    SAVE_DATA_FIELD(hardware_name);
    SAVE_DATA_FIELD(domain_name);
}

template <typename ArchiveT>
void
load(ArchiveT& ar, ::rocprofiler::tool::node_info& data)
{
    LOAD_DATA_FIELD(id);
    LOAD_DATA_FIELD(hash);
    LOAD_DATA_FIELD(machine_id);
    LOAD_DATA_FIELD(system_name);
    LOAD_DATA_FIELD(hostname);
    LOAD_DATA_FIELD(release);
    LOAD_DATA_FIELD(version);
    LOAD_DATA_FIELD(hardware_name);
    LOAD_DATA_FIELD(domain_name);
}

#undef SAVE_DATA_FIELD
#undef LOAD_DATA_FIELD
}  // namespace cereal
