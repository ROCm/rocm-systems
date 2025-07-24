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

#include <dlfcn.h>
#include <link.h>
#include <cassert>
#include <cstdlib>

namespace rocprofiler
{
namespace thread_trace
{
DL::DL(const char* libpath)
{
    auto path = common::filesystem::path();

    if(libpath != nullptr)
    {
        path = common::filesystem::path(libpath);
    }
    else if(auto* aqlprofile_handle = dlopen("libhsa-amd-aqlprofile64.so", RTLD_NOLOAD | RTLD_LAZY))
    {
        link_map* linkmap{nullptr};
        if(dlinfo(aqlprofile_handle, RTLD_DI_LINKMAP, &linkmap) == 0 && linkmap != nullptr)
            path = common::filesystem::path(linkmap->l_name).parent_path();

        dlclose(aqlprofile_handle);
    }

    path   = path / "librocprof-trace-decoder.so";
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

}  // namespace thread_trace
}  // namespace rocprofiler
