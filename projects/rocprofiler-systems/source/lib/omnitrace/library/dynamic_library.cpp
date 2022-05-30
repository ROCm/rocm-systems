// MIT License
//
// Copyright (c) 2022 Advanced Micro Devices, Inc. All Rights Reserved.
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

#include "library/dynamic_library.hpp"
#include "library/debug.hpp"
#include "library/defines.hpp"

#include <timemory/environment.hpp>

namespace omnitrace
{
dynamic_library::dynamic_library(const char* _env, const char* _fname, int _flags,
                                 bool _store)
: envname{ _env }
, filename{ tim::get_env<std::string>(_env, _fname, _store) }
, flags{ _flags }
{
    open();
}

dynamic_library::~dynamic_library() { close(); }

bool
dynamic_library::open()
{
    if(!filename.empty())
    {
        handle = dlopen(filename.c_str(), flags);
        if(!handle)
        {
            OMNITRACE_VERBOSE(2, "[dynamic_library][%s][%s] %s\n", envname.c_str(),
                              filename.c_str(), dlerror());
        }
        dlerror();  // Clear any existing error
    }
    return (handle != nullptr);
}

int
dynamic_library::close() const
{
    if(handle) return dlclose(handle);
    return -1;
}
}  // namespace omnitrace
