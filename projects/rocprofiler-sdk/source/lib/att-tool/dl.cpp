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

#include "lib/att-tool/dl.hpp"
#include "lib/common/environment.hpp"
#include "lib/common/filesystem.hpp"
#include "lib/common/logging.hpp"

#include <rocprofiler-sdk/cxx/details/tokenize.hpp>

#include <dlfcn.h>
#include <cassert>
#include <cstdlib>
#include <mutex>
#include <set>
#include <stdexcept>
#include <string>

namespace rocprofiler
{
namespace att_wrapper
{
namespace fs = ::rocprofiler::common::filesystem;

fs::path
get_search_path(std::string_view path_name)
{
    if(fs::exists(path_name)) return fs::path(path_name);
    return "";
}

DL::DL(const char* libname)
{
    auto paths = rocprofiler::common::get_env("ROCPROF_ATT_LIBRARY_PATH", "");
    if(paths.empty()) return;
    auto path_set = rocprofiler::sdk::parse::tokenize(paths, ":");

    for(auto&& name : path_set)
    {
        handle = dlopen((get_search_path(name) / libname).string().c_str(), RTLD_LAZY | RTLD_LOCAL);
        if(handle) break;
    }
    if(!handle) return;

    att_parse_data_fn =
        reinterpret_cast<ParseFn*>(dlsym(handle, "rocprofiler_att_decoder_parse_data"));
    att_info_fn =
        reinterpret_cast<InfoFn*>(dlsym(handle, "rocprofiler_att_decoder_get_info_string"));
    att_status_fn =
        reinterpret_cast<StatusFn*>(dlsym(handle, "rocprofiler_att_decoder_get_status_string"));
};

DL::~DL()
{
    if(handle) dlclose(handle);
}

}  // namespace att_wrapper
}  // namespace rocprofiler
