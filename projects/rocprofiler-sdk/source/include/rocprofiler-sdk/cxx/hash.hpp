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
//

#pragma once

#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/hsa.h>
#include <rocprofiler-sdk/internal_threading.h>

namespace rocprofiler
{
namespace sdk
{
namespace hash
{
template <typename Tp>
struct handle_hasher
{
    static_assert(sizeof(Tp) == sizeof(uint64_t), "error! only for opaque handle types");
    size_t operator()(Tp val) const { return val.handle; }
};
}  // namespace hash
}  // namespace sdk
}  // namespace rocprofiler

namespace std
{
template <typename Tp>
struct hash;

#define ROCPROFILER_CXX_SPECIALIZE_HANDLE_HASHER(TYPE)                                             \
    template <>                                                                                    \
    struct hash<TYPE> : public rocprofiler::sdk::hash::handle_hasher<TYPE>                         \
    {                                                                                              \
        using parent_type = ::rocprofiler::sdk::hash::handle_hasher<TYPE>;                         \
        using parent_type::operator();                                                             \
    };

ROCPROFILER_CXX_SPECIALIZE_HANDLE_HASHER(rocprofiler_context_id_t)
ROCPROFILER_CXX_SPECIALIZE_HANDLE_HASHER(rocprofiler_agent_id_t)
ROCPROFILER_CXX_SPECIALIZE_HANDLE_HASHER(rocprofiler_address_t)
ROCPROFILER_CXX_SPECIALIZE_HANDLE_HASHER(rocprofiler_queue_id_t)
ROCPROFILER_CXX_SPECIALIZE_HANDLE_HASHER(rocprofiler_stream_id_t)
ROCPROFILER_CXX_SPECIALIZE_HANDLE_HASHER(rocprofiler_buffer_id_t)
ROCPROFILER_CXX_SPECIALIZE_HANDLE_HASHER(rocprofiler_counter_id_t)
ROCPROFILER_CXX_SPECIALIZE_HANDLE_HASHER(rocprofiler_counter_config_id_t)
ROCPROFILER_CXX_SPECIALIZE_HANDLE_HASHER(rocprofiler_callback_thread_t)
ROCPROFILER_CXX_SPECIALIZE_HANDLE_HASHER(hsa_agent_t)
ROCPROFILER_CXX_SPECIALIZE_HANDLE_HASHER(hsa_signal_t)
ROCPROFILER_CXX_SPECIALIZE_HANDLE_HASHER(hsa_executable_t)
ROCPROFILER_CXX_SPECIALIZE_HANDLE_HASHER(hsa_region_t)
ROCPROFILER_CXX_SPECIALIZE_HANDLE_HASHER(hsa_amd_memory_pool_t)

#undef ROCPROFILER_CXX_SPECIALIZE_HANDLE_HASHER
}  // namespace std
