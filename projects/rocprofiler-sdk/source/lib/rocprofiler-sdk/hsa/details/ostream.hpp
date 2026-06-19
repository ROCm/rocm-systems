// MIT License
//
// Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#pragma once

#include <rocprofiler-sdk/hsa.h>

#include <iostream>
#include <string>
#include <string_view>

namespace rocprofiler
{
namespace hsa
{
namespace detail
{
static int              HSA_depth_max     = 1;
static thread_local int HSA_depth_max_cnt = 0;
static std::string_view HSA_structs_regex = {};

template <typename T>
inline static std::ostream&
operator<<(std::ostream& out, const T& v)
{
    using std::              operator<<;
    static thread_local bool recursion = false;
    if(recursion == false)
    {
        recursion = true;
        out << v;
        recursion = false;
    }
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const unsigned char& v)
{
    out << (unsigned int) v;
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const char& v)
{
    out << (unsigned char) v;
    return out;
}
// End of basic ostream ops

inline static std::ostream&
operator<<(std::ostream& out, const hsa_dim3_t& v)
{
    std::operator<<(out, '{');
    HSA_depth_max_cnt++;
    if(HSA_depth_max == -1 || HSA_depth_max_cnt <= HSA_depth_max)
    {
        if(std::string_view{"hsa_dim3_t::z"}.find(HSA_structs_regex) != std::string_view::npos)
        {
            rocprofiler::hsa::detail::operator<<(out, "z=");
            rocprofiler::hsa::detail::operator<<(out, v.z);
            rocprofiler::hsa::detail::operator<<(out, ", ");
        }
        if(std::string_view{"hsa_dim3_t::y"}.find(HSA_structs_regex) != std::string_view::npos)
        {
            rocprofiler::hsa::detail::operator<<(out, "y=");
            rocprofiler::hsa::detail::operator<<(out, v.y);
            rocprofiler::hsa::detail::operator<<(out, ", ");
        }
        if(std::string_view{"hsa_dim3_t::x"}.find(HSA_structs_regex) != std::string_view::npos)
        {
            rocprofiler::hsa::detail::operator<<(out, "x=");
            rocprofiler::hsa::detail::operator<<(out, v.x);
        }
    };
    HSA_depth_max_cnt--;
    std::operator<<(out, '}');
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const hsa_agent_t& v)
{
    std::operator<<(out, '{');
    HSA_depth_max_cnt++;
    if(HSA_depth_max == -1 || HSA_depth_max_cnt <= HSA_depth_max)
    {
        if(std::string_view{"hsa_agent_t::handle"}.find(HSA_structs_regex) !=
           std::string_view::npos)
        {
            rocprofiler::hsa::detail::operator<<(out, "handle=");
            rocprofiler::hsa::detail::operator<<(out, v.handle);
        }
    };
    HSA_depth_max_cnt--;
    std::operator<<(out, '}');
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const hsa_cache_t& v)
{
    std::operator<<(out, '{');
    HSA_depth_max_cnt++;
    if(HSA_depth_max == -1 || HSA_depth_max_cnt <= HSA_depth_max)
    {
        if(std::string_view{"hsa_cache_t::handle"}.find(HSA_structs_regex) !=
           std::string_view::npos)
        {
            rocprofiler::hsa::detail::operator<<(out, "handle=");
            rocprofiler::hsa::detail::operator<<(out, v.handle);
        }
    };
    HSA_depth_max_cnt--;
    std::operator<<(out, '}');
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const hsa_signal_t& v)
{
    std::operator<<(out, '{');
    HSA_depth_max_cnt++;
    if(HSA_depth_max == -1 || HSA_depth_max_cnt <= HSA_depth_max)
    {
        if(std::string_view{"hsa_signal_t::handle"}.find(HSA_structs_regex) !=
           std::string_view::npos)
        {
            rocprofiler::hsa::detail::operator<<(out, "handle=");
            rocprofiler::hsa::detail::operator<<(out, v.handle);
        }
    };
    HSA_depth_max_cnt--;
    std::operator<<(out, '}');
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const hsa_signal_group_t& v)
{
    std::operator<<(out, '{');
    HSA_depth_max_cnt++;
    if(HSA_depth_max == -1 || HSA_depth_max_cnt <= HSA_depth_max)
    {
        if(std::string_view{"hsa_signal_group_t::handle"}.find(HSA_structs_regex) !=
           std::string_view::npos)
        {
            rocprofiler::hsa::detail::operator<<(out, "handle=");
            rocprofiler::hsa::detail::operator<<(out, v.handle);
        }
    };
    HSA_depth_max_cnt--;
    std::operator<<(out, '}');
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const hsa_region_t& v)
{
    std::operator<<(out, '{');
    HSA_depth_max_cnt++;
    if(HSA_depth_max == -1 || HSA_depth_max_cnt <= HSA_depth_max)
    {
        if(std::string_view{"hsa_region_t::handle"}.find(HSA_structs_regex) !=
           std::string_view::npos)
        {
            rocprofiler::hsa::detail::operator<<(out, "handle=");
            rocprofiler::hsa::detail::operator<<(out, v.handle);
        }
    };
    HSA_depth_max_cnt--;
    std::operator<<(out, '}');
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const hsa_queue_t& v)
{
    std::operator<<(out, '{');
    HSA_depth_max_cnt++;
    if(HSA_depth_max == -1 || HSA_depth_max_cnt <= HSA_depth_max)
    {
        if(std::string_view{"hsa_queue_t::id"}.find(HSA_structs_regex) != std::string_view::npos)
        {
            rocprofiler::hsa::detail::operator<<(out, "id=");
            rocprofiler::hsa::detail::operator<<(out, v.id);
            rocprofiler::hsa::detail::operator<<(out, ", ");
        }
        if(std::string_view{"hsa_queue_t::reserved1"}.find(HSA_structs_regex) !=
           std::string_view::npos)
        {
            rocprofiler::hsa::detail::operator<<(out, "reserved1=");
            rocprofiler::hsa::detail::operator<<(out, v.reserved1);
            rocprofiler::hsa::detail::operator<<(out, ", ");
        }
        if(std::string_view{"hsa_queue_t::size"}.find(HSA_structs_regex) != std::string_view::npos)
        {
            rocprofiler::hsa::detail::operator<<(out, "size=");
            rocprofiler::hsa::detail::operator<<(out, v.size);
            rocprofiler::hsa::detail::operator<<(out, ", ");
        }
        if(std::string_view{"hsa_queue_t::doorbell_signal"}.find(HSA_structs_regex) !=
           std::string_view::npos)
        {
            rocprofiler::hsa::detail::operator<<(out, "doorbell_signal=");
            rocprofiler::hsa::detail::operator<<(out, v.doorbell_signal);
            rocprofiler::hsa::detail::operator<<(out, ", ");
        }
        if(std::string_view{"hsa_queue_t::features"}.find(HSA_structs_regex) !=
           std::string_view::npos)
        {
            rocprofiler::hsa::detail::operator<<(out, "features=");
            rocprofiler::hsa::detail::operator<<(out, v.features);
            rocprofiler::hsa::detail::operator<<(out, ", ");
        }
        if(std::string_view{"hsa_queue_t::type"}.find(HSA_structs_regex) != std::string_view::npos)
        {
            rocprofiler::hsa::detail::operator<<(out, "type=");
            rocprofiler::hsa::detail::operator<<(out, v.type);
        }
    };
    HSA_depth_max_cnt--;
    std::operator<<(out, '}');
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const hsa_kernel_dispatch_packet_t& v)
{
    std::operator<<(out, '{');
    HSA_depth_max_cnt++;
    if(HSA_depth_max == -1 || HSA_depth_max_cnt <= HSA_depth_max)
    {
        if(std::string_view{"hsa_kernel_dispatch_packet_t::completion_signal"}.find(
               HSA_structs_regex) != std::string_view::npos)
        {
            rocprofiler::hsa::detail::operator<<(out, "completion_signal=");
            rocprofiler::hsa::detail::operator<<(out, v.completion_signal);
            rocprofiler::hsa::detail::operator<<(out, ", ");
        }
        if(std::string_view{"hsa_kernel_dispatch_packet_t::reserved2"}.find(HSA_structs_regex) !=
           std::string_view::npos)
        {
            rocprofiler::hsa::detail::operator<<(out, "reserved2=");
            rocprofiler::hsa::detail::operator<<(out, v.reserved2);
            rocprofiler::hsa::detail::operator<<(out, ", ");
        }
        if(std::string_view{"hsa_kernel_dispatch_packet_t::kernel_object"}.find(
               HSA_structs_regex) != std::string_view::npos)
        {
            rocprofiler::hsa::detail::operator<<(out, "kernel_object=");
            rocprofiler::hsa::detail::operator<<(out, v.kernel_object);
            rocprofiler::hsa::detail::operator<<(out, ", ");
        }
        if(std::string("hsa_kernel_dispatch_packet_t::group_segment_size")
               .find(HSA_structs_regex) != std::string_view::npos)
        {
            rocprofiler::hsa::detail::operator<<(out, "group_segment_size=");
            rocprofiler::hsa::detail::operator<<(out, v.group_segment_size);
            rocprofiler::hsa::detail::operator<<(out, ", ");
        }
        if(std::string("hsa_kernel_dispatch_packet_t::private_segment_size")
               .find(HSA_structs_regex) != std::string_view::npos)
        {
            rocprofiler::hsa::detail::operator<<(out, "private_segment_size=");
            rocprofiler::hsa::detail::operator<<(out, v.private_segment_size);
            rocprofiler::hsa::detail::operator<<(out, ", ");
        }
        if(std::string_view{"hsa_kernel_dispatch_packet_t::grid_size_z"}.find(HSA_structs_regex) !=
           std::string_view::npos)
        {
            rocprofiler::hsa::detail::operator<<(out, "grid_size_z=");
            rocprofiler::hsa::detail::operator<<(out, v.grid_size_z);
            rocprofiler::hsa::detail::operator<<(out, ", ");
        }
        if(std::string_view{"hsa_kernel_dispatch_packet_t::grid_size_y"}.find(HSA_structs_regex) !=
           std::string_view::npos)
        {
            rocprofiler::hsa::detail::operator<<(out, "grid_size_y=");
            rocprofiler::hsa::detail::operator<<(out, v.grid_size_y);
            rocprofiler::hsa::detail::operator<<(out, ", ");
        }
        if(std::string_view{"hsa_kernel_dispatch_packet_t::grid_size_x"}.find(HSA_structs_regex) !=
           std::string_view::npos)
        {
            rocprofiler::hsa::detail::operator<<(out, "grid_size_x=");
            rocprofiler::hsa::detail::operator<<(out, v.grid_size_x);
            rocprofiler::hsa::detail::operator<<(out, ", ");
        }
        if(std::string_view{"hsa_kernel_dispatch_packet_t::reserved0"}.find(HSA_structs_regex) !=
           std::string_view::npos)
        {
            rocprofiler::hsa::detail::operator<<(out, "reserved0=");
            rocprofiler::hsa::detail::operator<<(out, v.reserved0);
            rocprofiler::hsa::detail::operator<<(out, ", ");
        }
        if(std::string_view{"hsa_kernel_dispatch_packet_t::workgroup_size_z"}.find(
               HSA_structs_regex) != std::string_view::npos)
        {
            rocprofiler::hsa::detail::operator<<(out, "workgroup_size_z=");
            rocprofiler::hsa::detail::operator<<(out, v.workgroup_size_z);
            rocprofiler::hsa::detail::operator<<(out, ", ");
        }
        if(std::string_view{"hsa_kernel_dispatch_packet_t::workgroup_size_y"}.find(
               HSA_structs_regex) != std::string_view::npos)
        {
            rocprofiler::hsa::detail::operator<<(out, "workgroup_size_y=");
            rocprofiler::hsa::detail::operator<<(out, v.workgroup_size_y);
            rocprofiler::hsa::detail::operator<<(out, ", ");
        }
        if(std::string_view{"hsa_kernel_dispatch_packet_t::workgroup_size_x"}.find(
               HSA_structs_regex) != std::string_view::npos)
        {
            rocprofiler::hsa::detail::operator<<(out, "workgroup_size_x=");
            rocprofiler::hsa::detail::operator<<(out, v.workgroup_size_x);
            rocprofiler::hsa::detail::operator<<(out, ", ");
        }
        if(std::string_view{"hsa_kernel_dispatch_packet_t::setup"}.find(HSA_structs_regex) !=
           std::string_view::npos)
        {
            rocprofiler::hsa::detail::operator<<(out, "setup=");
            rocprofiler::hsa::detail::operator<<(out, v.setup);
            rocprofiler::hsa::detail::operator<<(out, ", ");
        }
        if(std::string_view{"hsa_kernel_dispatch_packet_t::header"}.find(HSA_structs_regex) !=
           std::string_view::npos)
        {
            rocprofiler::hsa::detail::operator<<(out, "header=");
            rocprofiler::hsa::detail::operator<<(out, v.header);
        }
    };
    HSA_depth_max_cnt--;
    std::operator<<(out, '}');
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const hsa_agent_dispatch_packet_t& v)
{
    std::operator<<(out, '{');
    HSA_depth_max_cnt++;
    if(HSA_depth_max == -1 || HSA_depth_max_cnt <= HSA_depth_max)
    {
        if(std::string_view{"hsa_agent_dispatch_packet_t::completion_signal"}.find(
               HSA_structs_regex) != std::string_view::npos)
        {
            rocprofiler::hsa::detail::operator<<(out, "completion_signal=");
            rocprofiler::hsa::detail::operator<<(out, v.completion_signal);
            rocprofiler::hsa::detail::operator<<(out, ", ");
        }
        if(std::string_view{"hsa_agent_dispatch_packet_t::reserved2"}.find(HSA_structs_regex) !=
           std::string_view::npos)
        {
            rocprofiler::hsa::detail::operator<<(out, "reserved2=");
            rocprofiler::hsa::detail::operator<<(out, v.reserved2);
            rocprofiler::hsa::detail::operator<<(out, ", ");
        }
        if(std::string_view{"hsa_agent_dispatch_packet_t::arg"}.find(HSA_structs_regex) !=
           std::string_view::npos)
        {
            rocprofiler::hsa::detail::operator<<(out, "arg=");
            rocprofiler::hsa::detail::operator<<(out, v.arg);
            rocprofiler::hsa::detail::operator<<(out, ", ");
        }
        if(std::string_view{"hsa_agent_dispatch_packet_t::reserved0"}.find(HSA_structs_regex) !=
           std::string_view::npos)
        {
            rocprofiler::hsa::detail::operator<<(out, "reserved0=");
            rocprofiler::hsa::detail::operator<<(out, v.reserved0);
            rocprofiler::hsa::detail::operator<<(out, ", ");
        }
        if(std::string_view{"hsa_agent_dispatch_packet_t::type"}.find(HSA_structs_regex) !=
           std::string_view::npos)
        {
            rocprofiler::hsa::detail::operator<<(out, "type=");
            rocprofiler::hsa::detail::operator<<(out, v.type);
            rocprofiler::hsa::detail::operator<<(out, ", ");
        }
        if(std::string_view{"hsa_agent_dispatch_packet_t::header"}.find(HSA_structs_regex) !=
           std::string_view::npos)
        {
            rocprofiler::hsa::detail::operator<<(out, "header=");
            rocprofiler::hsa::detail::operator<<(out, v.header);
        }
    };
    HSA_depth_max_cnt--;
    std::operator<<(out, '}');
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const hsa_barrier_and_packet_t& v)
{
    std::operator<<(out, '{');
    HSA_depth_max_cnt++;
    if(HSA_depth_max == -1 || HSA_depth_max_cnt <= HSA_depth_max)
    {
        if(std::string_view{"hsa_barrier_and_packet_t::completion_signal"}.find(
               HSA_structs_regex) != std::string_view::npos)
        {
            rocprofiler::hsa::detail::operator<<(out, "completion_signal=");
            rocprofiler::hsa::detail::operator<<(out, v.completion_signal);
            rocprofiler::hsa::detail::operator<<(out, ", ");
        }
        if(std::string_view{"hsa_barrier_and_packet_t::reserved2"}.find(HSA_structs_regex) !=
           std::string_view::npos)
        {
            rocprofiler::hsa::detail::operator<<(out, "reserved2=");
            rocprofiler::hsa::detail::operator<<(out, v.reserved2);
            rocprofiler::hsa::detail::operator<<(out, ", ");
        }
        if(std::string_view{"hsa_barrier_and_packet_t::dep_signal"}.find(HSA_structs_regex) !=
           std::string_view::npos)
        {
            rocprofiler::hsa::detail::operator<<(out, "dep_signal=");
            rocprofiler::hsa::detail::operator<<(out, v.dep_signal);
            rocprofiler::hsa::detail::operator<<(out, ", ");
        }
        if(std::string_view{"hsa_barrier_and_packet_t::reserved1"}.find(HSA_structs_regex) !=
           std::string_view::npos)
        {
            rocprofiler::hsa::detail::operator<<(out, "reserved1=");
            rocprofiler::hsa::detail::operator<<(out, v.reserved1);
            rocprofiler::hsa::detail::operator<<(out, ", ");
        }
        if(std::string_view{"hsa_barrier_and_packet_t::reserved0"}.find(HSA_structs_regex) !=
           std::string_view::npos)
        {
            rocprofiler::hsa::detail::operator<<(out, "reserved0=");
            rocprofiler::hsa::detail::operator<<(out, v.reserved0);
            rocprofiler::hsa::detail::operator<<(out, ", ");
        }
        if(std::string_view{"hsa_barrier_and_packet_t::header"}.find(HSA_structs_regex) !=
           std::string_view::npos)
        {
            rocprofiler::hsa::detail::operator<<(out, "header=");
            rocprofiler::hsa::detail::operator<<(out, v.header);
        }
    };
    HSA_depth_max_cnt--;
    std::operator<<(out, '}');
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const hsa_barrier_or_packet_t& v)
{
    std::operator<<(out, '{');
    HSA_depth_max_cnt++;
    if(HSA_depth_max == -1 || HSA_depth_max_cnt <= HSA_depth_max)
    {
        if(std::string_view{"hsa_barrier_or_packet_t::completion_signal"}.find(HSA_structs_regex) !=
           std::string_view::npos)
        {
            rocprofiler::hsa::detail::operator<<(out, "completion_signal=");
            rocprofiler::hsa::detail::operator<<(out, v.completion_signal);
            rocprofiler::hsa::detail::operator<<(out, ", ");
        }
        if(std::string_view{"hsa_barrier_or_packet_t::reserved2"}.find(HSA_structs_regex) !=
           std::string_view::npos)
        {
            rocprofiler::hsa::detail::operator<<(out, "reserved2=");
            rocprofiler::hsa::detail::operator<<(out, v.reserved2);
            rocprofiler::hsa::detail::operator<<(out, ", ");
        }
        if(std::string_view{"hsa_barrier_or_packet_t::dep_signal"}.find(HSA_structs_regex) !=
           std::string_view::npos)
        {
            rocprofiler::hsa::detail::operator<<(out, "dep_signal=");
            rocprofiler::hsa::detail::operator<<(out, v.dep_signal);
            rocprofiler::hsa::detail::operator<<(out, ", ");
        }
        if(std::string_view{"hsa_barrier_or_packet_t::reserved1"}.find(HSA_structs_regex) !=
           std::string_view::npos)
        {
            rocprofiler::hsa::detail::operator<<(out, "reserved1=");
            rocprofiler::hsa::detail::operator<<(out, v.reserved1);
            rocprofiler::hsa::detail::operator<<(out, ", ");
        }
        if(std::string_view{"hsa_barrier_or_packet_t::reserved0"}.find(HSA_structs_regex) !=
           std::string_view::npos)
        {
            rocprofiler::hsa::detail::operator<<(out, "reserved0=");
            rocprofiler::hsa::detail::operator<<(out, v.reserved0);
            rocprofiler::hsa::detail::operator<<(out, ", ");
        }
        if(std::string_view{"hsa_barrier_or_packet_t::header"}.find(HSA_structs_regex) !=
           std::string_view::npos)
        {
            rocprofiler::hsa::detail::operator<<(out, "header=");
            rocprofiler::hsa::detail::operator<<(out, v.header);
        }
    };
    HSA_depth_max_cnt--;
    std::operator<<(out, '}');
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const hsa_isa_t& v)
{
    std::operator<<(out, '{');
    HSA_depth_max_cnt++;
    if(HSA_depth_max == -1 || HSA_depth_max_cnt <= HSA_depth_max)
    {
        if(std::string_view{"hsa_isa_t::handle"}.find(HSA_structs_regex) != std::string_view::npos)
        {
            rocprofiler::hsa::detail::operator<<(out, "handle=");
            rocprofiler::hsa::detail::operator<<(out, v.handle);
        }
    };
    HSA_depth_max_cnt--;
    std::operator<<(out, '}');
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const hsa_wavefront_t& v)
{
    std::operator<<(out, '{');
    HSA_depth_max_cnt++;
    if(HSA_depth_max == -1 || HSA_depth_max_cnt <= HSA_depth_max)
    {
        if(std::string_view{"hsa_wavefront_t::handle"}.find(HSA_structs_regex) !=
           std::string_view::npos)
        {
            rocprofiler::hsa::detail::operator<<(out, "handle=");
            rocprofiler::hsa::detail::operator<<(out, v.handle);
        }
    };
    HSA_depth_max_cnt--;
    std::operator<<(out, '}');
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const hsa_code_object_reader_t& v)
{
    std::operator<<(out, '{');
    HSA_depth_max_cnt++;
    if(HSA_depth_max == -1 || HSA_depth_max_cnt <= HSA_depth_max)
    {
        if(std::string_view{"hsa_code_object_reader_t::handle"}.find(HSA_structs_regex) !=
           std::string_view::npos)
        {
            rocprofiler::hsa::detail::operator<<(out, "handle=");
            rocprofiler::hsa::detail::operator<<(out, v.handle);
        }
    };
    HSA_depth_max_cnt--;
    std::operator<<(out, '}');
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const hsa_executable_t& v)
{
    std::operator<<(out, '{');
    HSA_depth_max_cnt++;
    if(HSA_depth_max == -1 || HSA_depth_max_cnt <= HSA_depth_max)
    {
        if(std::string_view{"hsa_executable_t::handle"}.find(HSA_structs_regex) !=
           std::string_view::npos)
        {
            rocprofiler::hsa::detail::operator<<(out, "handle=");
            rocprofiler::hsa::detail::operator<<(out, v.handle);
        }
    };
    HSA_depth_max_cnt--;
    std::operator<<(out, '}');
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const hsa_loaded_code_object_t& v)
{
    std::operator<<(out, '{');
    HSA_depth_max_cnt++;
    if(HSA_depth_max == -1 || HSA_depth_max_cnt <= HSA_depth_max)
    {
        if(std::string_view{"hsa_loaded_code_object_t::handle"}.find(HSA_structs_regex) !=
           std::string_view::npos)
        {
            rocprofiler::hsa::detail::operator<<(out, "handle=");
            rocprofiler::hsa::detail::operator<<(out, v.handle);
        }
    };
    HSA_depth_max_cnt--;
    std::operator<<(out, '}');
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const hsa_executable_symbol_t& v)
{
    std::operator<<(out, '{');
    HSA_depth_max_cnt++;
    if(HSA_depth_max == -1 || HSA_depth_max_cnt <= HSA_depth_max)
    {
        if(std::string_view{"hsa_executable_symbol_t::handle"}.find(HSA_structs_regex) !=
           std::string_view::npos)
        {
            rocprofiler::hsa::detail::operator<<(out, "handle=");
            rocprofiler::hsa::detail::operator<<(out, v.handle);
        }
    };
    HSA_depth_max_cnt--;
    std::operator<<(out, '}');
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const hsa_code_object_t& v)
{
    std::operator<<(out, '{');
    HSA_depth_max_cnt++;
    if(HSA_depth_max == -1 || HSA_depth_max_cnt <= HSA_depth_max)
    {
        if(std::string_view{"hsa_code_object_t::handle"}.find(HSA_structs_regex) !=
           std::string_view::npos)
        {
            rocprofiler::hsa::detail::operator<<(out, "handle=");
            rocprofiler::hsa::detail::operator<<(out, v.handle);
        }
    };
    HSA_depth_max_cnt--;
    std::operator<<(out, '}');
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const hsa_callback_data_t& v)
{
    std::operator<<(out, '{');
    HSA_depth_max_cnt++;
    if(HSA_depth_max == -1 || HSA_depth_max_cnt <= HSA_depth_max)
    {
        if(std::string_view{"hsa_callback_data_t::handle"}.find(HSA_structs_regex) !=
           std::string_view::npos)
        {
            rocprofiler::hsa::detail::operator<<(out, "handle=");
            rocprofiler::hsa::detail::operator<<(out, v.handle);
        }
    };
    HSA_depth_max_cnt--;
    std::operator<<(out, '}');
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const hsa_code_symbol_t& v)
{
    std::operator<<(out, '{');
    HSA_depth_max_cnt++;
    if(HSA_depth_max == -1 || HSA_depth_max_cnt <= HSA_depth_max)
    {
        if(std::string_view{"hsa_code_symbol_t::handle"}.find(HSA_structs_regex) !=
           std::string_view::npos)
        {
            rocprofiler::hsa::detail::operator<<(out, "handle=");
            rocprofiler::hsa::detail::operator<<(out, v.handle);
        }
    };
    HSA_depth_max_cnt--;
    std::operator<<(out, '}');
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const hsa_ext_image_t& v)
{
    std::operator<<(out, '{');
    HSA_depth_max_cnt++;
    if(HSA_depth_max == -1 || HSA_depth_max_cnt <= HSA_depth_max)
    {
        if(std::string_view{"hsa_ext_image_t::handle"}.find(HSA_structs_regex) !=
           std::string_view::npos)
        {
            rocprofiler::hsa::detail::operator<<(out, "handle=");
            rocprofiler::hsa::detail::operator<<(out, v.handle);
        }
    };
    HSA_depth_max_cnt--;
    std::operator<<(out, '}');
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const hsa_ext_image_format_t& v)
{
    std::operator<<(out, '{');
    HSA_depth_max_cnt++;
    if(HSA_depth_max == -1 || HSA_depth_max_cnt <= HSA_depth_max)
    {
        if(std::string_view{"hsa_ext_image_format_t::channel_order"}.find(HSA_structs_regex) !=
           std::string_view::npos)
        {
            rocprofiler::hsa::detail::operator<<(out, "channel_order=");
            rocprofiler::hsa::detail::operator<<(out, v.channel_order);
            rocprofiler::hsa::detail::operator<<(out, ", ");
        }
        if(std::string_view{"hsa_ext_image_format_t::channel_type"}.find(HSA_structs_regex) !=
           std::string_view::npos)
        {
            rocprofiler::hsa::detail::operator<<(out, "channel_type=");
            rocprofiler::hsa::detail::operator<<(out, v.channel_type);
        }
    };
    HSA_depth_max_cnt--;
    std::operator<<(out, '}');
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const hsa_ext_image_descriptor_t& v)
{
    std::operator<<(out, '{');
    HSA_depth_max_cnt++;
    if(HSA_depth_max == -1 || HSA_depth_max_cnt <= HSA_depth_max)
    {
        if(std::string_view{"hsa_ext_image_descriptor_t::format"}.find(HSA_structs_regex) !=
           std::string_view::npos)
        {
            rocprofiler::hsa::detail::operator<<(out, "format=");
            rocprofiler::hsa::detail::operator<<(out, v.format);
            rocprofiler::hsa::detail::operator<<(out, ", ");
        }
        if(std::string_view{"hsa_ext_image_descriptor_t::array_size"}.find(HSA_structs_regex) !=
           std::string_view::npos)
        {
            rocprofiler::hsa::detail::operator<<(out, "array_size=");
            rocprofiler::hsa::detail::operator<<(out, v.array_size);
            rocprofiler::hsa::detail::operator<<(out, ", ");
        }
        if(std::string_view{"hsa_ext_image_descriptor_t::depth"}.find(HSA_structs_regex) !=
           std::string_view::npos)
        {
            rocprofiler::hsa::detail::operator<<(out, "depth=");
            rocprofiler::hsa::detail::operator<<(out, v.depth);
            rocprofiler::hsa::detail::operator<<(out, ", ");
        }
        if(std::string_view{"hsa_ext_image_descriptor_t::height"}.find(HSA_structs_regex) !=
           std::string_view::npos)
        {
            rocprofiler::hsa::detail::operator<<(out, "height=");
            rocprofiler::hsa::detail::operator<<(out, v.height);
            rocprofiler::hsa::detail::operator<<(out, ", ");
        }
        if(std::string_view{"hsa_ext_image_descriptor_t::width"}.find(HSA_structs_regex) !=
           std::string_view::npos)
        {
            rocprofiler::hsa::detail::operator<<(out, "width=");
            rocprofiler::hsa::detail::operator<<(out, v.width);
            rocprofiler::hsa::detail::operator<<(out, ", ");
        }
        if(std::string_view{"hsa_ext_image_descriptor_t::geometry"}.find(HSA_structs_regex) !=
           std::string_view::npos)
        {
            rocprofiler::hsa::detail::operator<<(out, "geometry=");
            rocprofiler::hsa::detail::operator<<(out, v.geometry);
        }
    };
    HSA_depth_max_cnt--;
    std::operator<<(out, '}');
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const hsa_ext_image_data_info_t& v)
{
    std::operator<<(out, '{');
    HSA_depth_max_cnt++;
    if(HSA_depth_max == -1 || HSA_depth_max_cnt <= HSA_depth_max)
    {
        if(std::string_view{"hsa_ext_image_data_info_t::alignment"}.find(HSA_structs_regex) !=
           std::string_view::npos)
        {
            rocprofiler::hsa::detail::operator<<(out, "alignment=");
            rocprofiler::hsa::detail::operator<<(out, v.alignment);
            rocprofiler::hsa::detail::operator<<(out, ", ");
        }
        if(std::string_view{"hsa_ext_image_data_info_t::size"}.find(HSA_structs_regex) !=
           std::string_view::npos)
        {
            rocprofiler::hsa::detail::operator<<(out, "size=");
            rocprofiler::hsa::detail::operator<<(out, v.size);
        }
    };
    HSA_depth_max_cnt--;
    std::operator<<(out, '}');
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const hsa_ext_image_region_t& v)
{
    std::operator<<(out, '{');
    HSA_depth_max_cnt++;
    if(HSA_depth_max == -1 || HSA_depth_max_cnt <= HSA_depth_max)
    {
        if(std::string_view{"hsa_ext_image_region_t::range"}.find(HSA_structs_regex) !=
           std::string_view::npos)
        {
            rocprofiler::hsa::detail::operator<<(out, "range=");
            rocprofiler::hsa::detail::operator<<(out, v.range);
            rocprofiler::hsa::detail::operator<<(out, ", ");
        }
        if(std::string_view{"hsa_ext_image_region_t::offset"}.find(HSA_structs_regex) !=
           std::string_view::npos)
        {
            rocprofiler::hsa::detail::operator<<(out, "offset=");
            rocprofiler::hsa::detail::operator<<(out, v.offset);
        }
    };
    HSA_depth_max_cnt--;
    std::operator<<(out, '}');
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const hsa_ext_sampler_t& v)
{
    std::operator<<(out, '{');
    HSA_depth_max_cnt++;
    if(HSA_depth_max == -1 || HSA_depth_max_cnt <= HSA_depth_max)
    {
        if(std::string_view{"hsa_ext_sampler_t::handle"}.find(HSA_structs_regex) !=
           std::string_view::npos)
        {
            rocprofiler::hsa::detail::operator<<(out, "handle=");
            rocprofiler::hsa::detail::operator<<(out, v.handle);
        }
    };
    HSA_depth_max_cnt--;
    std::operator<<(out, '}');
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const hsa_ext_sampler_descriptor_t& v)
{
    std::operator<<(out, '{');
    HSA_depth_max_cnt++;
    if(HSA_depth_max == -1 || HSA_depth_max_cnt <= HSA_depth_max)
    {
        if(std::string_view{"hsa_ext_sampler_descriptor_t::address_mode"}.find(HSA_structs_regex) !=
           std::string_view::npos)
        {
            rocprofiler::hsa::detail::operator<<(out, "address_mode=");
            rocprofiler::hsa::detail::operator<<(out, v.address_mode);
            rocprofiler::hsa::detail::operator<<(out, ", ");
        }
        if(std::string_view{"hsa_ext_sampler_descriptor_t::filter_mode"}.find(HSA_structs_regex) !=
           std::string_view::npos)
        {
            rocprofiler::hsa::detail::operator<<(out, "filter_mode=");
            rocprofiler::hsa::detail::operator<<(out, v.filter_mode);
            rocprofiler::hsa::detail::operator<<(out, ", ");
        }
        if(std::string_view{"hsa_ext_sampler_descriptor_t::coordinate_mode"}.find(
               HSA_structs_regex) != std::string_view::npos)
        {
            rocprofiler::hsa::detail::operator<<(out, "coordinate_mode=");
            rocprofiler::hsa::detail::operator<<(out, v.coordinate_mode);
        }
    };
    HSA_depth_max_cnt--;
    std::operator<<(out, '}');
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const hsa_ext_images_1_00_pfn_t& v)
{
    std::operator<<(out, '{');
    HSA_depth_max_cnt++;
    if(HSA_depth_max == -1 || HSA_depth_max_cnt <= HSA_depth_max)
    {
        if(std::string("hsa_ext_images_1_00_pfn_t::hsa_ext_sampler_destroy")
               .find(HSA_structs_regex) != std::string_view::npos)
        {
            rocprofiler::hsa::detail::operator<<(out, "hsa_ext_sampler_destroy=");
            rocprofiler::hsa::detail::operator<<(out, v.hsa_ext_sampler_destroy);
            rocprofiler::hsa::detail::operator<<(out, ", ");
        }
        if(std::string("hsa_ext_images_1_00_pfn_t::hsa_ext_sampler_create")
               .find(HSA_structs_regex) != std::string_view::npos)
        {
            rocprofiler::hsa::detail::operator<<(out, "hsa_ext_sampler_create=");
            rocprofiler::hsa::detail::operator<<(out, v.hsa_ext_sampler_create);
            rocprofiler::hsa::detail::operator<<(out, ", ");
        }
        if(std::string_view{"hsa_ext_images_1_00_pfn_t::hsa_ext_image_copy"}.find(
               HSA_structs_regex) != std::string_view::npos)
        {
            rocprofiler::hsa::detail::operator<<(out, "hsa_ext_image_copy=");
            rocprofiler::hsa::detail::operator<<(out, v.hsa_ext_image_copy);
            rocprofiler::hsa::detail::operator<<(out, ", ");
        }
        if(std::string("hsa_ext_images_1_00_pfn_t::hsa_ext_image_destroy")
               .find(HSA_structs_regex) != std::string_view::npos)
        {
            rocprofiler::hsa::detail::operator<<(out, "hsa_ext_image_destroy=");
            rocprofiler::hsa::detail::operator<<(out, v.hsa_ext_image_destroy);
            rocprofiler::hsa::detail::operator<<(out, ", ");
        }
        if(std::string("hsa_ext_images_1_00_pfn_t::hsa_ext_image_data_get_info")
               .find(HSA_structs_regex) != std::string_view::npos)
        {
            rocprofiler::hsa::detail::operator<<(out, "hsa_ext_image_data_get_info=");
            rocprofiler::hsa::detail::operator<<(out, v.hsa_ext_image_data_get_info);
            rocprofiler::hsa::detail::operator<<(out, ", ");
        }
        if(std::string("hsa_ext_images_1_00_pfn_t::hsa_ext_image_get_capability")
               .find(HSA_structs_regex) != std::string_view::npos)
        {
            rocprofiler::hsa::detail::operator<<(out, "hsa_ext_image_get_capability=");
            rocprofiler::hsa::detail::operator<<(out, v.hsa_ext_image_get_capability);
        }
    };
    HSA_depth_max_cnt--;
    std::operator<<(out, '}');
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const hsa_ext_images_1_pfn_t& v)
{
    std::operator<<(out, '{');
    HSA_depth_max_cnt++;
    if(HSA_depth_max == -1 || HSA_depth_max_cnt <= HSA_depth_max)
    {
        if(std::string("hsa_ext_images_1_pfn_t::hsa_ext_image_data_get_info_with_layout")
               .find(HSA_structs_regex) != std::string_view::npos)
        {
            rocprofiler::hsa::detail::operator<<(out, "hsa_ext_image_data_get_info_with_layout=");
            rocprofiler::hsa::detail::operator<<(out, v.hsa_ext_image_data_get_info_with_layout);
            rocprofiler::hsa::detail::operator<<(out, ", ");
        }
        if(std::string("hsa_ext_images_1_pfn_t::hsa_ext_image_get_capability_with_layout")
               .find(HSA_structs_regex) != std::string_view::npos)
        {
            rocprofiler::hsa::detail::operator<<(out, "hsa_ext_image_get_capability_with_layout=");
            rocprofiler::hsa::detail::operator<<(out, v.hsa_ext_image_get_capability_with_layout);
            rocprofiler::hsa::detail::operator<<(out, ", ");
        }
        if(std::string_view{"hsa_ext_images_1_pfn_t::hsa_ext_sampler_destroy"}.find(
               HSA_structs_regex) != std::string_view::npos)
        {
            rocprofiler::hsa::detail::operator<<(out, "hsa_ext_sampler_destroy=");
            rocprofiler::hsa::detail::operator<<(out, v.hsa_ext_sampler_destroy);
            rocprofiler::hsa::detail::operator<<(out, ", ");
        }
        if(std::string_view{"hsa_ext_images_1_pfn_t::hsa_ext_sampler_create"}.find(
               HSA_structs_regex) != std::string_view::npos)
        {
            rocprofiler::hsa::detail::operator<<(out, "hsa_ext_sampler_create=");
            rocprofiler::hsa::detail::operator<<(out, v.hsa_ext_sampler_create);
            rocprofiler::hsa::detail::operator<<(out, ", ");
        }
        if(std::string_view{"hsa_ext_images_1_pfn_t::hsa_ext_image_copy"}.find(HSA_structs_regex) !=
           std::string_view::npos)
        {
            rocprofiler::hsa::detail::operator<<(out, "hsa_ext_image_copy=");
            rocprofiler::hsa::detail::operator<<(out, v.hsa_ext_image_copy);
            rocprofiler::hsa::detail::operator<<(out, ", ");
        }
        if(std::string_view{"hsa_ext_images_1_pfn_t::hsa_ext_image_destroy"}.find(
               HSA_structs_regex) != std::string_view::npos)
        {
            rocprofiler::hsa::detail::operator<<(out, "hsa_ext_image_destroy=");
            rocprofiler::hsa::detail::operator<<(out, v.hsa_ext_image_destroy);
            rocprofiler::hsa::detail::operator<<(out, ", ");
        }
        if(std::string("hsa_ext_images_1_pfn_t::hsa_ext_image_data_get_info")
               .find(HSA_structs_regex) != std::string_view::npos)
        {
            rocprofiler::hsa::detail::operator<<(out, "hsa_ext_image_data_get_info=");
            rocprofiler::hsa::detail::operator<<(out, v.hsa_ext_image_data_get_info);
            rocprofiler::hsa::detail::operator<<(out, ", ");
        }
        if(std::string("hsa_ext_images_1_pfn_t::hsa_ext_image_get_capability")
               .find(HSA_structs_regex) != std::string_view::npos)
        {
            rocprofiler::hsa::detail::operator<<(out, "hsa_ext_image_get_capability=");
            rocprofiler::hsa::detail::operator<<(out, v.hsa_ext_image_get_capability);
        }
    };
    HSA_depth_max_cnt--;
    std::operator<<(out, '}');
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const hsa_amd_vendor_packet_header_t& v)
{
    std::operator<<(out, '{');
    HSA_depth_max_cnt++;
    if(HSA_depth_max == -1 || HSA_depth_max_cnt <= HSA_depth_max)
    {
        if(std::string_view{"hsa_amd_vendor_packet_header_t::reserved"}.find(HSA_structs_regex) !=
           std::string_view::npos)
        {
            rocprofiler::hsa::detail::operator<<(out, "reserved=");
            rocprofiler::hsa::detail::operator<<(out, v.reserved);
            rocprofiler::hsa::detail::operator<<(out, ", ");
        }
        if(std::string_view{"hsa_amd_vendor_packet_header_t::AmdFormat"}.find(HSA_structs_regex) !=
           std::string_view::npos)
        {
            rocprofiler::hsa::detail::operator<<(out, "AmdFormat=");
            rocprofiler::hsa::detail::operator<<(out, v.AmdFormat);
            rocprofiler::hsa::detail::operator<<(out, ", ");
        }
        if(std::string_view{"hsa_amd_vendor_packet_header_t::header"}.find(HSA_structs_regex) !=
           std::string_view::npos)
        {
            rocprofiler::hsa::detail::operator<<(out, "header=");
            rocprofiler::hsa::detail::operator<<(out, v.header);
        }
    };
    HSA_depth_max_cnt--;
    std::operator<<(out, '}');
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const hsa_amd_barrier_value_packet_t& v)
{
    std::operator<<(out, '{');
    HSA_depth_max_cnt++;
    if(HSA_depth_max == -1 || HSA_depth_max_cnt <= HSA_depth_max)
    {
        if(std::string("hsa_amd_barrier_value_packet_t::completion_signal")
               .find(HSA_structs_regex) != std::string_view::npos)
        {
            rocprofiler::hsa::detail::operator<<(out, "completion_signal=");
            rocprofiler::hsa::detail::operator<<(out, v.completion_signal);
            rocprofiler::hsa::detail::operator<<(out, ", ");
        }
        if(std::string_view{"hsa_amd_barrier_value_packet_t::reserved3"}.find(HSA_structs_regex) !=
           std::string_view::npos)
        {
            rocprofiler::hsa::detail::operator<<(out, "reserved3=");
            rocprofiler::hsa::detail::operator<<(out, v.reserved3);
            rocprofiler::hsa::detail::operator<<(out, ", ");
        }
        if(std::string_view{"hsa_amd_barrier_value_packet_t::reserved2"}.find(HSA_structs_regex) !=
           std::string_view::npos)
        {
            rocprofiler::hsa::detail::operator<<(out, "reserved2=");
            rocprofiler::hsa::detail::operator<<(out, v.reserved2);
            rocprofiler::hsa::detail::operator<<(out, ", ");
        }
        if(std::string_view{"hsa_amd_barrier_value_packet_t::reserved1"}.find(HSA_structs_regex) !=
           std::string_view::npos)
        {
            rocprofiler::hsa::detail::operator<<(out, "reserved1=");
            rocprofiler::hsa::detail::operator<<(out, v.reserved1);
            rocprofiler::hsa::detail::operator<<(out, ", ");
        }
        if(std::string_view{"hsa_amd_barrier_value_packet_t::cond"}.find(HSA_structs_regex) !=
           std::string_view::npos)
        {
            rocprofiler::hsa::detail::operator<<(out, "cond=");
            rocprofiler::hsa::detail::operator<<(out, v.cond);
            rocprofiler::hsa::detail::operator<<(out, ", ");
        }
        if(std::string_view{"hsa_amd_barrier_value_packet_t::mask"}.find(HSA_structs_regex) !=
           std::string_view::npos)
        {
            rocprofiler::hsa::detail::operator<<(out, "mask=");
            rocprofiler::hsa::detail::operator<<(out, v.mask);
            rocprofiler::hsa::detail::operator<<(out, ", ");
        }
        if(std::string_view{"hsa_amd_barrier_value_packet_t::value"}.find(HSA_structs_regex) !=
           std::string_view::npos)
        {
            rocprofiler::hsa::detail::operator<<(out, "value=");
            rocprofiler::hsa::detail::operator<<(out, v.value);
            rocprofiler::hsa::detail::operator<<(out, ", ");
        }
        if(std::string_view{"hsa_amd_barrier_value_packet_t::signal"}.find(HSA_structs_regex) !=
           std::string_view::npos)
        {
            rocprofiler::hsa::detail::operator<<(out, "signal=");
            rocprofiler::hsa::detail::operator<<(out, v.signal);
            rocprofiler::hsa::detail::operator<<(out, ", ");
        }
        if(std::string_view{"hsa_amd_barrier_value_packet_t::reserved0"}.find(HSA_structs_regex) !=
           std::string_view::npos)
        {
            rocprofiler::hsa::detail::operator<<(out, "reserved0=");
            rocprofiler::hsa::detail::operator<<(out, v.reserved0);
            rocprofiler::hsa::detail::operator<<(out, ", ");
        }
        if(std::string_view{"hsa_amd_barrier_value_packet_t::header"}.find(HSA_structs_regex) !=
           std::string_view::npos)
        {
            rocprofiler::hsa::detail::operator<<(out, "header=");
            rocprofiler::hsa::detail::operator<<(out, v.header);
        }
    };
    HSA_depth_max_cnt--;
    std::operator<<(out, '}');
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const hsa_amd_hdp_flush_t& v)
{
    std::operator<<(out, '{');
    HSA_depth_max_cnt++;
    if(HSA_depth_max == -1 || HSA_depth_max_cnt <= HSA_depth_max)
    {
        if(std::string_view{"hsa_amd_hdp_flush_t::HDP_REG_FLUSH_CNTL"}.find(HSA_structs_regex) !=
           std::string_view::npos)
        {
            rocprofiler::hsa::detail::operator<<(out, "HDP_REG_FLUSH_CNTL=");
            rocprofiler::hsa::detail::operator<<(out, v.HDP_REG_FLUSH_CNTL);
            rocprofiler::hsa::detail::operator<<(out, ", ");
        }
        if(std::string_view{"hsa_amd_hdp_flush_t::HDP_MEM_FLUSH_CNTL"}.find(HSA_structs_regex) !=
           std::string_view::npos)
        {
            rocprofiler::hsa::detail::operator<<(out, "HDP_MEM_FLUSH_CNTL=");
            rocprofiler::hsa::detail::operator<<(out, v.HDP_MEM_FLUSH_CNTL);
        }
    };
    HSA_depth_max_cnt--;
    std::operator<<(out, '}');
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const hsa_amd_profiling_dispatch_time_t& v)
{
    std::operator<<(out, '{');
    HSA_depth_max_cnt++;
    if(HSA_depth_max == -1 || HSA_depth_max_cnt <= HSA_depth_max)
    {
        if(std::string_view{"hsa_amd_profiling_dispatch_time_t::end"}.find(HSA_structs_regex) !=
           std::string_view::npos)
        {
            rocprofiler::hsa::detail::operator<<(out, "end=");
            rocprofiler::hsa::detail::operator<<(out, v.end);
            rocprofiler::hsa::detail::operator<<(out, ", ");
        }
        if(std::string_view{"hsa_amd_profiling_dispatch_time_t::start"}.find(HSA_structs_regex) !=
           std::string_view::npos)
        {
            rocprofiler::hsa::detail::operator<<(out, "start=");
            rocprofiler::hsa::detail::operator<<(out, v.start);
        }
    };
    HSA_depth_max_cnt--;
    std::operator<<(out, '}');
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const hsa_amd_profiling_async_copy_time_t& v)
{
    std::operator<<(out, '{');
    HSA_depth_max_cnt++;
    if(HSA_depth_max == -1 || HSA_depth_max_cnt <= HSA_depth_max)
    {
        if(std::string_view{"hsa_amd_profiling_async_copy_time_t::end"}.find(HSA_structs_regex) !=
           std::string_view::npos)
        {
            rocprofiler::hsa::detail::operator<<(out, "end=");
            rocprofiler::hsa::detail::operator<<(out, v.end);
            rocprofiler::hsa::detail::operator<<(out, ", ");
        }
        if(std::string_view{"hsa_amd_profiling_async_copy_time_t::start"}.find(HSA_structs_regex) !=
           std::string_view::npos)
        {
            rocprofiler::hsa::detail::operator<<(out, "start=");
            rocprofiler::hsa::detail::operator<<(out, v.start);
        }
    };
    HSA_depth_max_cnt--;
    std::operator<<(out, '}');
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const hsa_amd_memory_pool_t& v)
{
    std::operator<<(out, '{');
    HSA_depth_max_cnt++;
    if(HSA_depth_max == -1 || HSA_depth_max_cnt <= HSA_depth_max)
    {
        if(std::string_view{"hsa_amd_memory_pool_t::handle"}.find(HSA_structs_regex) !=
           std::string_view::npos)
        {
            rocprofiler::hsa::detail::operator<<(out, "handle=");
            rocprofiler::hsa::detail::operator<<(out, v.handle);
        }
    };
    HSA_depth_max_cnt--;
    std::operator<<(out, '}');
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const hsa_pitched_ptr_t& v)
{
    std::operator<<(out, '{');
    HSA_depth_max_cnt++;
    if(HSA_depth_max == -1 || HSA_depth_max_cnt <= HSA_depth_max)
    {
        if(std::string_view{"hsa_pitched_ptr_t::slice"}.find(HSA_structs_regex) !=
           std::string_view::npos)
        {
            rocprofiler::hsa::detail::operator<<(out, "slice=");
            rocprofiler::hsa::detail::operator<<(out, v.slice);
            rocprofiler::hsa::detail::operator<<(out, ", ");
        }
        if(std::string_view{"hsa_pitched_ptr_t::pitch"}.find(HSA_structs_regex) !=
           std::string_view::npos)
        {
            rocprofiler::hsa::detail::operator<<(out, "pitch=");
            rocprofiler::hsa::detail::operator<<(out, v.pitch);
        }
    };
    HSA_depth_max_cnt--;
    std::operator<<(out, '}');
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const hsa_amd_memory_pool_link_info_t& v)
{
    std::operator<<(out, '{');
    HSA_depth_max_cnt++;
    if(HSA_depth_max == -1 || HSA_depth_max_cnt <= HSA_depth_max)
    {
        if(std::string_view{"hsa_amd_memory_pool_link_info_t::numa_distance"}.find(
               HSA_structs_regex) != std::string_view::npos)
        {
            rocprofiler::hsa::detail::operator<<(out, "numa_distance=");
            rocprofiler::hsa::detail::operator<<(out, v.numa_distance);
            rocprofiler::hsa::detail::operator<<(out, ", ");
        }
        if(std::string_view{"hsa_amd_memory_pool_link_info_t::link_type"}.find(HSA_structs_regex) !=
           std::string_view::npos)
        {
            rocprofiler::hsa::detail::operator<<(out, "link_type=");
            rocprofiler::hsa::detail::operator<<(out, v.link_type);
            rocprofiler::hsa::detail::operator<<(out, ", ");
        }
        if(std::string_view{"hsa_amd_memory_pool_link_info_t::max_bandwidth"}.find(
               HSA_structs_regex) != std::string_view::npos)
        {
            rocprofiler::hsa::detail::operator<<(out, "max_bandwidth=");
            rocprofiler::hsa::detail::operator<<(out, v.max_bandwidth);
            rocprofiler::hsa::detail::operator<<(out, ", ");
        }
        if(std::string_view{"hsa_amd_memory_pool_link_info_t::min_bandwidth"}.find(
               HSA_structs_regex) != std::string_view::npos)
        {
            rocprofiler::hsa::detail::operator<<(out, "min_bandwidth=");
            rocprofiler::hsa::detail::operator<<(out, v.min_bandwidth);
            rocprofiler::hsa::detail::operator<<(out, ", ");
        }
        if(std::string_view{"hsa_amd_memory_pool_link_info_t::max_latency"}.find(
               HSA_structs_regex) != std::string_view::npos)
        {
            rocprofiler::hsa::detail::operator<<(out, "max_latency=");
            rocprofiler::hsa::detail::operator<<(out, v.max_latency);
            rocprofiler::hsa::detail::operator<<(out, ", ");
        }
        if(std::string_view{"hsa_amd_memory_pool_link_info_t::min_latency"}.find(
               HSA_structs_regex) != std::string_view::npos)
        {
            rocprofiler::hsa::detail::operator<<(out, "min_latency=");
            rocprofiler::hsa::detail::operator<<(out, v.min_latency);
        }
    };
    HSA_depth_max_cnt--;
    std::operator<<(out, '}');
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const hsa_amd_image_descriptor_t& v)
{
    std::operator<<(out, '{');
    HSA_depth_max_cnt++;
    if(HSA_depth_max == -1 || HSA_depth_max_cnt <= HSA_depth_max)
    {
        if(std::string_view{"hsa_amd_image_descriptor_t::data"}.find(HSA_structs_regex) !=
           std::string_view::npos)
        {
            rocprofiler::hsa::detail::operator<<(out, "data=");
            rocprofiler::hsa::detail::operator<<(out, v.data);
            rocprofiler::hsa::detail::operator<<(out, ", ");
        }
        if(std::string_view{"hsa_amd_image_descriptor_t::deviceID"}.find(HSA_structs_regex) !=
           std::string_view::npos)
        {
            rocprofiler::hsa::detail::operator<<(out, "deviceID=");
            rocprofiler::hsa::detail::operator<<(out, v.deviceID);
            rocprofiler::hsa::detail::operator<<(out, ", ");
        }
        if(std::string_view{"hsa_amd_image_descriptor_t::version"}.find(HSA_structs_regex) !=
           std::string_view::npos)
        {
            rocprofiler::hsa::detail::operator<<(out, "version=");
            rocprofiler::hsa::detail::operator<<(out, v.version);
        }
    };
    HSA_depth_max_cnt--;
    std::operator<<(out, '}');
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const hsa_amd_pointer_info_t& v)
{
    std::operator<<(out, '{');
    HSA_depth_max_cnt++;
    if(HSA_depth_max == -1 || HSA_depth_max_cnt <= HSA_depth_max)
    {
        if(std::string_view{"hsa_amd_pointer_info_t::global_flags"}.find(HSA_structs_regex) !=
           std::string_view::npos)
        {
            rocprofiler::hsa::detail::operator<<(out, "global_flags=");
            rocprofiler::hsa::detail::operator<<(out, v.global_flags);
            rocprofiler::hsa::detail::operator<<(out, ", ");
        }
        if(std::string_view{"hsa_amd_pointer_info_t::agentOwner"}.find(HSA_structs_regex) !=
           std::string_view::npos)
        {
            rocprofiler::hsa::detail::operator<<(out, "agentOwner=");
            rocprofiler::hsa::detail::operator<<(out, v.agentOwner);
            rocprofiler::hsa::detail::operator<<(out, ", ");
        }
        if(std::string_view{"hsa_amd_pointer_info_t::sizeInBytes"}.find(HSA_structs_regex) !=
           std::string_view::npos)
        {
            rocprofiler::hsa::detail::operator<<(out, "sizeInBytes=");
            rocprofiler::hsa::detail::operator<<(out, v.sizeInBytes);
            rocprofiler::hsa::detail::operator<<(out, ", ");
        }
        if(std::string_view{"hsa_amd_pointer_info_t::type"}.find(HSA_structs_regex) !=
           std::string_view::npos)
        {
            rocprofiler::hsa::detail::operator<<(out, "type=");
            rocprofiler::hsa::detail::operator<<(out, v.type);
            rocprofiler::hsa::detail::operator<<(out, ", ");
        }
        if(std::string_view{"hsa_amd_pointer_info_t::size"}.find(HSA_structs_regex) !=
           std::string_view::npos)
        {
            rocprofiler::hsa::detail::operator<<(out, "size=");
            rocprofiler::hsa::detail::operator<<(out, v.size);
        }
    };
    HSA_depth_max_cnt--;
    std::operator<<(out, '}');
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const hsa_amd_ipc_memory_t& v)
{
    std::operator<<(out, '{');
    HSA_depth_max_cnt++;
    if(HSA_depth_max == -1 || HSA_depth_max_cnt <= HSA_depth_max)
    {
        if(std::string_view{"hsa_amd_ipc_memory_t::handle"}.find(HSA_structs_regex) !=
           std::string_view::npos)
        {
            rocprofiler::hsa::detail::operator<<(out, "handle=");
            rocprofiler::hsa::detail::operator<<(out, v.handle);
        }
    };
    HSA_depth_max_cnt--;
    std::operator<<(out, '}');
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const hsa_amd_gpu_memory_fault_info_t& v)
{
    std::operator<<(out, '{');
    HSA_depth_max_cnt++;
    if(HSA_depth_max == -1 || HSA_depth_max_cnt <= HSA_depth_max)
    {
        if(std::string("hsa_amd_gpu_memory_fault_info_t::fault_reason_mask")
               .find(HSA_structs_regex) != std::string_view::npos)
        {
            rocprofiler::hsa::detail::operator<<(out, "fault_reason_mask=");
            rocprofiler::hsa::detail::operator<<(out, v.fault_reason_mask);
            rocprofiler::hsa::detail::operator<<(out, ", ");
        }
        if(std::string("hsa_amd_gpu_memory_fault_info_t::virtual_address")
               .find(HSA_structs_regex) != std::string_view::npos)
        {
            rocprofiler::hsa::detail::operator<<(out, "virtual_address=");
            rocprofiler::hsa::detail::operator<<(out, v.virtual_address);
            rocprofiler::hsa::detail::operator<<(out, ", ");
        }
        if(std::string_view{"hsa_amd_gpu_memory_fault_info_t::agent"}.find(HSA_structs_regex) !=
           std::string_view::npos)
        {
            rocprofiler::hsa::detail::operator<<(out, "agent=");
            rocprofiler::hsa::detail::operator<<(out, v.agent);
        }
    };
    HSA_depth_max_cnt--;
    std::operator<<(out, '}');
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const hsa_amd_event_t& v)
{
    std::operator<<(out, '{');
    HSA_depth_max_cnt++;
    if(HSA_depth_max == -1 || HSA_depth_max_cnt <= HSA_depth_max)
    {
        if(std::string_view{"hsa_amd_event_t::event_type"}.find(HSA_structs_regex) !=
           std::string_view::npos)
        {
            rocprofiler::hsa::detail::operator<<(out, "event_type=");
            rocprofiler::hsa::detail::operator<<(out, v.event_type);
        }
    };
    HSA_depth_max_cnt--;
    std::operator<<(out, '}');
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const hsa_amd_svm_attribute_pair_t& v)
{
    std::operator<<(out, '{');
    HSA_depth_max_cnt++;
    if(HSA_depth_max == -1 || HSA_depth_max_cnt <= HSA_depth_max)
    {
        if(std::string_view{"hsa_amd_svm_attribute_pair_t::value"}.find(HSA_structs_regex) !=
           std::string_view::npos)
        {
            rocprofiler::hsa::detail::operator<<(out, "value=");
            rocprofiler::hsa::detail::operator<<(out, v.value);
            rocprofiler::hsa::detail::operator<<(out, ", ");
        }
        if(std::string_view{"hsa_amd_svm_attribute_pair_t::attribute"}.find(HSA_structs_regex) !=
           std::string_view::npos)
        {
            rocprofiler::hsa::detail::operator<<(out, "attribute=");
            rocprofiler::hsa::detail::operator<<(out, v.attribute);
        }
    };
    HSA_depth_max_cnt--;
    std::operator<<(out, '}');
    return out;
}
// end ostream ops for HSA
}  // namespace detail
}  // namespace hsa
}  // namespace rocprofiler

inline static std::ostream&
operator<<(std::ostream& out, const hsa_dim3_t& v)
{
    rocprofiler::hsa::detail::operator<<(out, v);
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const hsa_agent_t& v)
{
    rocprofiler::hsa::detail::operator<<(out, v);
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const hsa_cache_t& v)
{
    rocprofiler::hsa::detail::operator<<(out, v);
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const hsa_signal_t& v)
{
    rocprofiler::hsa::detail::operator<<(out, v);
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const hsa_signal_group_t& v)
{
    rocprofiler::hsa::detail::operator<<(out, v);
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const hsa_region_t& v)
{
    rocprofiler::hsa::detail::operator<<(out, v);
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const hsa_queue_t& v)
{
    rocprofiler::hsa::detail::operator<<(out, v);
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const hsa_kernel_dispatch_packet_t& v)
{
    rocprofiler::hsa::detail::operator<<(out, v);
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const hsa_agent_dispatch_packet_t& v)
{
    rocprofiler::hsa::detail::operator<<(out, v);
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const hsa_barrier_and_packet_t& v)
{
    rocprofiler::hsa::detail::operator<<(out, v);
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const hsa_barrier_or_packet_t& v)
{
    rocprofiler::hsa::detail::operator<<(out, v);
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const hsa_isa_t& v)
{
    rocprofiler::hsa::detail::operator<<(out, v);
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const hsa_wavefront_t& v)
{
    rocprofiler::hsa::detail::operator<<(out, v);
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const hsa_code_object_reader_t& v)
{
    rocprofiler::hsa::detail::operator<<(out, v);
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const hsa_executable_t& v)
{
    rocprofiler::hsa::detail::operator<<(out, v);
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const hsa_loaded_code_object_t& v)
{
    rocprofiler::hsa::detail::operator<<(out, v);
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const hsa_executable_symbol_t& v)
{
    rocprofiler::hsa::detail::operator<<(out, v);
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const hsa_code_object_t& v)
{
    rocprofiler::hsa::detail::operator<<(out, v);
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const hsa_callback_data_t& v)
{
    rocprofiler::hsa::detail::operator<<(out, v);
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const hsa_code_symbol_t& v)
{
    rocprofiler::hsa::detail::operator<<(out, v);
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const hsa_ext_image_t& v)
{
    rocprofiler::hsa::detail::operator<<(out, v);
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const hsa_ext_image_format_t& v)
{
    rocprofiler::hsa::detail::operator<<(out, v);
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const hsa_ext_image_descriptor_t& v)
{
    rocprofiler::hsa::detail::operator<<(out, v);
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const hsa_ext_image_data_info_t& v)
{
    rocprofiler::hsa::detail::operator<<(out, v);
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const hsa_ext_image_region_t& v)
{
    rocprofiler::hsa::detail::operator<<(out, v);
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const hsa_ext_sampler_t& v)
{
    rocprofiler::hsa::detail::operator<<(out, v);
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const hsa_ext_sampler_descriptor_t& v)
{
    rocprofiler::hsa::detail::operator<<(out, v);
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const hsa_ext_images_1_00_pfn_t& v)
{
    rocprofiler::hsa::detail::operator<<(out, v);
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const hsa_ext_images_1_pfn_t& v)
{
    rocprofiler::hsa::detail::operator<<(out, v);
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const hsa_amd_vendor_packet_header_t& v)
{
    rocprofiler::hsa::detail::operator<<(out, v);
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const hsa_amd_barrier_value_packet_t& v)
{
    rocprofiler::hsa::detail::operator<<(out, v);
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const hsa_amd_hdp_flush_t& v)
{
    rocprofiler::hsa::detail::operator<<(out, v);
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const hsa_amd_profiling_dispatch_time_t& v)
{
    rocprofiler::hsa::detail::operator<<(out, v);
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const hsa_amd_profiling_async_copy_time_t& v)
{
    rocprofiler::hsa::detail::operator<<(out, v);
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const hsa_amd_memory_pool_t& v)
{
    rocprofiler::hsa::detail::operator<<(out, v);
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const hsa_pitched_ptr_t& v)
{
    rocprofiler::hsa::detail::operator<<(out, v);
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const hsa_amd_memory_pool_link_info_t& v)
{
    rocprofiler::hsa::detail::operator<<(out, v);
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const hsa_amd_image_descriptor_t& v)
{
    rocprofiler::hsa::detail::operator<<(out, v);
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const hsa_amd_pointer_info_t& v)
{
    rocprofiler::hsa::detail::operator<<(out, v);
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const hsa_amd_ipc_memory_t& v)
{
    rocprofiler::hsa::detail::operator<<(out, v);
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const hsa_amd_gpu_memory_fault_info_t& v)
{
    rocprofiler::hsa::detail::operator<<(out, v);
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const hsa_amd_event_t& v)
{
    rocprofiler::hsa::detail::operator<<(out, v);
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const hsa_amd_svm_attribute_pair_t& v)
{
    rocprofiler::hsa::detail::operator<<(out, v);
    return out;
}
