// MIT License
//
// Copyright (c) 2023-2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include "lib/rocprofiler-sdk/spm/interface.hpp"
#include "lib/common/logging.hpp"

#include <fmt/format.h>

#include <dlfcn.h>
#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>

namespace rocprofiler
{
namespace spm
{
const spm_interface*
construct_spm_interface()
{
    static std::once_flag                 flag;
    static std::unique_ptr<spm_interface> cached;

    std::call_once(flag, []() {
        auto iface    = std::make_unique<spm_interface>();
        iface->handle = dlopen("libhsa-amd-aqlprofile64.so", RTLD_LAZY);

        if(!iface->handle)
        {
            ROCP_CI_LOG(WARNING) << fmt::format("aqlprofile cannot be opened");
            return;
        }

        iface->spm_create_packets = (spm_interface::spm_create_packets_fn_t*) dlsym(
            iface->handle, "aqlprofile_spm_create_packets");
        iface->spm_delete_packets = (spm_interface::spm_delete_packets_fn_t*) dlsym(
            iface->handle, "aqlprofile_spm_delete_packets");
        iface->spm_start =
            (spm_interface::spm_start_fn_t*) dlsym(iface->handle, "aqlprofile_spm_start");
        iface->spm_stop =
            (spm_interface::spm_stop_fn_t*) dlsym(iface->handle, "aqlprofile_spm_stop");
        iface->spm_decode_stream_v1 = (spm_interface::spm_decode_stream_v1_fn_t*) dlsym(
            iface->handle, "aqlprofile_spm_decode_stream_v1");
        iface->spm_decode_query = (spm_interface::spm_decode_query_fn_t*) dlsym(
            iface->handle, "aqlprofile_spm_decode_query");
        iface->spm_is_event_supported = (spm_interface::spm_is_event_supported_fn_t*) dlsym(
            iface->handle, "aqlprofile_spm_is_event_supported");
        iface->spm_query_agent_capabilities =
            (spm_interface::spm_query_agent_capabilities_fn_t*) dlsym(
                iface->handle, "aqlprofile_spm_query_agent_capabilities");
        cached = std::move(iface);
    });

    return cached.get();
}
spm_interface::~spm_interface()
{
    // Only the cached singleton owns the dlopen handle; copies/moves clear their handle
    if(handle) dlclose(handle);
}
}  // namespace spm
}  // namespace rocprofiler
