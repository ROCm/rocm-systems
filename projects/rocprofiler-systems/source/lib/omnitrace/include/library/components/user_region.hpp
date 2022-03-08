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

#pragma once

#include "library/defines.hpp"
#include "library/timemory.hpp"

namespace omnitrace
{
namespace component
{
// timemory component which calls omnitrace functions
// (used in gotcha wrappers)
struct user_region : comp::base<user_region, void>
{
    static std::string label() { return "user_region"; }
    void               start();
    void               stop();
    void               set_prefix(const char*);

private:
    const char* m_prefix = nullptr;
};
}  // namespace component
}  // namespace omnitrace

TIMEMORY_COMPONENT_ALIAS(omnitrace_user_region, omnitrace::component::user_region)

#if !defined(OMNITRACE_EXTERN_COMPONENTS) ||                                             \
    (defined(OMNITRACE_EXTERN_COMPONENTS) && OMNITRACE_EXTERN_COMPONENTS > 0)

#    include <timemory/operations.hpp>

TIMEMORY_DECLARE_EXTERN_COMPONENT(omnitrace_user_region, false, void)

#endif
