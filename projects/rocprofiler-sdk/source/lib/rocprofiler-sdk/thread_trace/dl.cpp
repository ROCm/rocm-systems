// MIT License
//
// Copyright (c) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include "lib/rocprofiler-sdk/thread_trace/dl.hpp"
#include "lib/common/filesystem.hpp"
#include "lib/common/logging.hpp"
#include "lib/common/static_object.hpp"

#include <dlfcn.h>
#include <cassert>
#include <cstdlib>

namespace rocprofiler
{
namespace thread_trace
{
DL::DL(const char* libpath)
{
    if(libpath == nullptr) return;

    auto path = common::filesystem::path(libpath) / "librocprof-trace-decoder.so";

    handle = dlopen(path.c_str(), RTLD_LAZY | RTLD_LOCAL);
    if(!handle) return;

    att_parse_data_fn =
        reinterpret_cast<ParseFn*>(dlsym(handle, "rocprof_trace_decoder_parse_data"));
    att_info_fn = reinterpret_cast<InfoFn*>(dlsym(handle, "rocprof_trace_decoder_get_info_string"));
    att_status_fn =
        reinterpret_cast<StatusFn*>(dlsym(handle, "rocprof_trace_decoder_get_status_string"));
};

DL::~DL()
{
    if(handle) dlclose(handle);
}

AQLProfileDL::AQLProfileDL()
{
#if defined(ROCPROFILER_BUILD_AQLPROFILE) && ROCPROFILER_BUILD_AQLPROFILE
    // Vendored aqlprofile: bind directly to the in-binary copy. Going through
    // dlsym(RTLD_DEFAULT) can resolve to a different copy in the system
    // libhsa-amd-aqlprofile64.so, which has its own MemoryManager static map
    // and would not see handles registered by the vendored aqlprofile_att_create_packets.
    get_buffer_packets_fn   = &aqlprofile_att_get_buffer_packets;
    update_buffer_status_fn = &aqlprofile_att_update_buffer_status;
#else
    // External aqlprofile: load symbols from libhsa-amd-aqlprofile64.so
    get_buffer_packets_fn = reinterpret_cast<GetBufferPacketsFn*>(
        dlsym(RTLD_DEFAULT, "aqlprofile_att_get_buffer_packets"));
    update_buffer_status_fn = reinterpret_cast<UpdateBufferStatusFn*>(
        dlsym(RTLD_DEFAULT, "aqlprofile_att_update_buffer_status"));

    if(valid()) return;

    const char* lib_names[] = {
        "libhsa-amd-aqlprofile64.so",
        "libhsa-amd-aqlprofile64.so.1",
    };

    for(const char* lib_name : lib_names)
    {
        handle = dlopen(lib_name, RTLD_LAZY | RTLD_LOCAL);
        if(handle) break;
    }

    if(!handle)
    {
        ROCP_WARNING << "Failed to load libhsa-amd-aqlprofile64.so.1: " << dlerror();
        return;
    }

    get_buffer_packets_fn =
        reinterpret_cast<GetBufferPacketsFn*>(dlsym(handle, "aqlprofile_att_get_buffer_packets"));
    update_buffer_status_fn = reinterpret_cast<UpdateBufferStatusFn*>(
        dlsym(handle, "aqlprofile_att_update_buffer_status"));
#endif
}

AQLProfileDL::~AQLProfileDL()
{
    if(handle) dlclose(handle);
}

AQLProfileDL*
get_aqlprofile_dl()
{
    static auto& instance = common::static_object<AQLProfileDL>::construct();
    return instance;
}

}  // namespace thread_trace
}  // namespace rocprofiler
