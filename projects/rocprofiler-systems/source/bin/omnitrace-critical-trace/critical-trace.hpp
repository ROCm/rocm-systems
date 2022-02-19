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

#include "library/config.hpp"
#include "library/critical_trace.hpp"
#include "library/debug.hpp"
#include "library/defines.hpp"
#include "library/perfetto.hpp"
#include "library/ptl.hpp"

#include <PTL/ThreadPool.hh>
#include <timemory/backends/dmp.hpp>
#include <timemory/backends/threading.hpp>
#include <timemory/hash/types.hpp>
#include <timemory/tpls/cereal/cereal/archives/json.hpp>
#include <timemory/tpls/cereal/cereal/cereal.hpp>
#include <timemory/utility/macros.hpp>
#include <timemory/utility/types.hpp>
#include <timemory/utility/utility.hpp>

#include <cctype>
#include <cstdint>
#include <exception>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace omnitrace
{
namespace critical_trace
{
namespace
{
using call_graph_t              = tim::graph<entry>;
using call_graph_itr_t          = typename call_graph_t::iterator;
using call_graph_sibling_itr_t  = typename call_graph_t::sibling_iterator;
using call_graph_preorder_itr_t = typename call_graph_t::pre_order_iterator;

hash_ids   complete_hash_ids{};
call_chain complete_call_chain{};
std::mutex complete_call_mutex{};

void
update_critical_path(call_chain _chain, int64_t _tid);

void
load_call_chain(const std::string& _fname, const std::string& _label,
                call_chain& _call_chain);

void
compute_critical_trace();

void
find_children(PTL::ThreadPool& _tp, call_graph_t& _graph, const call_chain& _chain);

void
find_sequences(PTL::ThreadPool& _tp, call_graph_t& _graph,
               std::vector<call_chain>& _chain);

void
find_sequences(PTL::ThreadPool& _tp, call_graph_t& _graph, call_graph_itr_t _root,
               std::vector<call_chain>& _chain);

template <typename ArchiveT, typename T, typename AllocatorT>
void
serialize_graph(ArchiveT& ar, const tim::graph<T, AllocatorT>& _graph);

template <typename ArchiveT, typename T, typename AllocatorT>
void
serialize_subgraph(ArchiveT& ar, const tim::graph<T, AllocatorT>& _graph,
                   typename tim::graph<T, AllocatorT>::iterator _root);

void
compute_critical_trace();

template <Device DevT>
void
generate_perfetto(const std::vector<call_chain>& _data);

inline void
copy_hash_ids()
{
    // make copy to avoid parallel iteration issues
    auto _hash_ids = complete_hash_ids;
    // ensure all hash ids exist
    for(const auto& itr : _hash_ids)
        tim::hash::add_hash_id(itr);
}
}  // namespace
}  // namespace critical_trace
}  // namespace omnitrace
