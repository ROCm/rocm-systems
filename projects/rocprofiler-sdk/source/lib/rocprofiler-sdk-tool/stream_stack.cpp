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

#include "stream_stack.hpp"

#include "lib/common/container/small_vector.hpp"
#include "lib/common/static_tl_object.hpp"
#include "lib/rocprofiler-sdk/hip/stream.hpp"

namespace rocprofiler
{
namespace tool
{
namespace stream
{
namespace
{
auto*
get_stream_stack()
{
    static thread_local auto*& _v =
        common::static_tl_object<common::container::small_vector<rocprofiler_stream_id_t>>::
            construct(rocprofiler_stream_id_t{.handle = 0});
    return _v;
}
}  // namespace

bool
stream_stack_not_null()
{
    return get_stream_stack() != nullptr;
}

void
push_stream_id(rocprofiler_stream_id_t id)
{
    CHECK_NOTNULL(get_stream_stack())->emplace_back(id);
}

void
pop_stream_id()
{
    CHECK_NOTNULL(get_stream_stack())->pop_back();
}

rocprofiler_stream_id_t
get_stream_id()
{
    return (CHECK_NOTNULL(get_stream_stack())->empty()) ? rocprofiler_stream_id_t{.handle = 0}
                                                        : get_stream_stack()->back();
}

bool
stream_stack_empty()
{
    return CHECK_NOTNULL(get_stream_stack())->empty();
}
}  // namespace stream
}  // namespace tool
}  // namespace rocprofiler
