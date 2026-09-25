// MIT License
//
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include <rocprofiler-sdk/experimental/registration.h>
#include <rocprofiler-sdk/registration.h>

#include <dlfcn.h>

namespace rocprofiler
{
namespace registration
{
inline rocprofiler_configure_attach_func_t
resolve_attach_for_configure(rocprofiler_configure_func_t        configure,
                             rocprofiler_configure_attach_func_t candidate)
{
    if(!candidate) return nullptr;

    auto owner = Dl_info{};
    if(!configure || dladdr(reinterpret_cast<const void*>(configure), &owner) == 0 ||
       owner.dli_fbase == nullptr || owner.dli_fname == nullptr)
        return nullptr;

    auto belongs_to_owner = [&](const void* symbol) {
        auto info = Dl_info{};
        return symbol && dladdr(symbol, &info) != 0 && info.dli_fbase == owner.dli_fbase;
    };

    if(belongs_to_owner(reinterpret_cast<const void*>(candidate))) return candidate;

    auto* handle = dlopen(owner.dli_fname, RTLD_LAZY | RTLD_NOLOAD);
    if(!handle) return nullptr;

    auto owner_attach         = rocprofiler_configure_attach_func_t{};
    *(void**) (&owner_attach) = dlsym(handle, "rocprofiler_configure_attach");
    auto* result =
        belongs_to_owner(reinterpret_cast<const void*>(owner_attach)) ? owner_attach : nullptr;
    dlclose(handle);
    return result;
}
}  // namespace registration
}  // namespace rocprofiler
