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

#include "lib/common/environment.hpp"
#include "lib/common/logging.hpp"
#include "lib/common/static_object.hpp"
#include "lib/rocprofiler-sdk/registration.hpp"

#include <iostream>

namespace rocprofiler
{
namespace shared_library
{
namespace
{
struct lifetime
{
    lifetime();
    ~lifetime();
};

lifetime::lifetime()
{
    registration::init_logging();

    if(common::get_env("ROCPROFILER_LIBRARY_CTOR", false))
    {
        ROCP_INFO << "Initializing rocprofiler-sdk library...";
        registration::initialize();
        ROCP_INFO << "rocprofiler-sdk library initialized";
    }
}

lifetime::~lifetime()
{
    if(common::get_env("ROCPROFILER_LIBRARY_DTOR", false))
    {
        ROCP_INFO << "Finalizing rocprofiler-sdk library...";
        registration::finalize();
        ROCP_INFO << "rocprofiler-sdk library finalized";
    }
}

auto*&
get_lifetime()
{
    static auto* _v = common::static_object<lifetime>::construct();
    return _v;
}
}  // namespace
}  // namespace shared_library

auto rocprofiler_sdk_shlib_lifetime = shared_library::get_lifetime();

void
rocprofiler_sdk_shlib_ctor() ROCPROFILER_ATTRIBUTE(constructor(101));

void
rocprofiler_sdk_shlib_ctor()
{
    (void) shared_library::get_lifetime();
}
}  // namespace rocprofiler
