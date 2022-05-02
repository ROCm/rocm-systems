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

#include "library/components/user_region.hpp"
#include "library/api.hpp"
#include "library/components/fwd.hpp"

namespace omnitrace
{
namespace component
{
void
user_region::start()
{
    if(m_prefix) omnitrace_push_region_hidden(m_prefix);
}

void
user_region::stop()
{
    if(m_prefix) omnitrace_pop_region_hidden(m_prefix);
}

void
user_region::set_prefix(const char* _prefix)
{
    m_prefix = _prefix;
}
}  // namespace component
}  // namespace omnitrace

TIMEMORY_INITIALIZE_STORAGE(omnitrace::component::user_region)
TIMEMORY_INSTANTIATE_EXTERN_COMPONENT(omnitrace_user_region, false, void)
