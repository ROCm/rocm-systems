// Copyright (c) 2018 Advanced Micro Devices, Inc. All Rights Reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// with the Software without restriction, including without limitation the
// rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
// sell copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// * Redistributions of source code must retain the above copyright notice,
// this list of conditions and the following disclaimers.
//
// * Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimers in the
// documentation and/or other materials provided with the distribution.
//
// * Neither the names of Advanced Micro Devices, Inc. nor the names of its
// contributors may be used to endorse or promote products derived from
// this Software without specific prior written permission.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS WITH
// THE SOFTWARE.

#pragma once

#include "library/debug.hpp"

#include <dlfcn.h>
#include <string>
#include <timemory/environment.hpp>

struct dynamic_library
{
    dynamic_library()                           = delete;
    dynamic_library(const dynamic_library&)     = delete;
    dynamic_library(dynamic_library&&) noexcept = default;
    dynamic_library& operator=(const dynamic_library&) = delete;
    dynamic_library& operator=(dynamic_library&&) noexcept = default;

    dynamic_library(const char* _env, const char* _fname,
                    int _flags = (RTLD_NOW | RTLD_GLOBAL), bool _store = false)
    : envname{ _env }
    , filename{ tim::get_env<std::string>(_env, _fname, _store) }
    , flags{ _flags }
    {
        if(!filename.empty())
        {
            handle = dlopen(filename.c_str(), flags);
            if(!handle)
            {
                OMNITRACE_DEBUG("%s\n", dlerror());
            }
            dlerror();  // Clear any existing error
        }
    }

    ~dynamic_library()
    {
        if(handle) dlclose(handle);
    }

    std::string envname  = {};
    std::string filename = {};
    int         flags    = 0;
    void*       handle   = nullptr;
};
