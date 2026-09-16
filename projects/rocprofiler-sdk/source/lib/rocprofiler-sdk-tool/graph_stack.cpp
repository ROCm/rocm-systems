// MIT License
//
// Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include "graph_stack.hpp"

#include "lib/common/container/small_vector.hpp"
#include "lib/common/static_tl_object.hpp"

namespace rocprofiler
{
namespace tool
{
namespace graph
{
namespace
{
auto*
get_graph_stack()
{
    static thread_local auto*& _v =
        common::static_tl_object<common::container::small_vector<attribution>>::construct(
            size_t{0});
    return _v;
}
}  // namespace

bool
graph_stack_not_null()
{
    return get_graph_stack() != nullptr;
}

void
push(rocprofiler_graph_exec_id_t graph_exec_id)
{
    CHECK_NOTNULL(get_graph_stack())
        ->emplace_back(attribution{graph_exec_id, rocprofiler_graph_node_id_t{.handle = 0}});
}

void
pop()
{
    auto* _stack = CHECK_NOTNULL(get_graph_stack());
    if(!_stack->empty()) _stack->pop_back();
}

attribution*
current()
{
    auto* _stack = CHECK_NOTNULL(get_graph_stack());
    return _stack->empty() ? nullptr : &_stack->back();
}

bool
graph_stack_empty()
{
    return CHECK_NOTNULL(get_graph_stack())->empty();
}
}  // namespace graph
}  // namespace tool
}  // namespace rocprofiler
