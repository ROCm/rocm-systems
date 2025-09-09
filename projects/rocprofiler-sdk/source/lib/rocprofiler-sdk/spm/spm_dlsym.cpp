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

#include "lib/rocprofiler-sdk/spm/spm_dlsym.hpp"

#include <dlfcn.h>
#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>

namespace rocprofiler
{
namespace SPM
{
Dlsym::Dlsym()
{
    handle = dlopen("libhsa-amd-aqlprofile64.so", RTLD_NOW | RTLD_LOCAL | RTLD_NOLOAD);
    if(!handle)
        handle = dlopen("libhsa-amd-aqlprofile64.so.1", RTLD_NOW | RTLD_LOCAL | RTLD_NOLOAD);

    if(!handle) return;

    create_packets_fn = (CreateFn*) dlsym(handle, "aqlprofile_spm_create_packets");
    delete_packets_fn = (DeleteFn*) dlsym(handle, "aqlprofile_spm_delete_packets");
    spm_start_fn      = (StartFn*) dlsym(handle, "aqlprofile_spm_start");
    spm_stop_fn       = (StopFn*) dlsym(handle, "aqlprofile_spm_stop");
    spm_decode_fn     = (DecodeFn*) dlsym(handle, "aqlprofile_spm_decode_stream_v1");
    spm_query_fn      = (QueryFn*) dlsym(handle, "aqlprofile_spm_decode_query");
    is_supported_fn   = (SupportFn*) dlsym(handle, "aqlprofile_spm_is_event_supported");
}

Dlsym::~Dlsym()
{
    if(handle) dlclose(handle);
}
}  // namespace SPM
}  // namespace rocprofiler
