// MIT License
//
// Copyright (c) 2022-2025 Advanced Micro Devices, Inc. All Rights Reserved.
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

#include "core/common.hpp"
#include "core/components/fwd.hpp"
#include "core/defines.hpp"

#include <timemory/components/base.hpp>
#include <timemory/hash/types.hpp>
#include <timemory/mpl/concepts.hpp>
#include <timemory/mpl/type_traits.hpp>
#include <timemory/tpls/cereal/cereal.hpp>
#include <timemory/utility/types.hpp>

#include <cstdint>
#include <unordered_map>

namespace rocprofsys
{
namespace causal
{
namespace component
{
struct progress_point : comp::base<progress_point, void>
{
    using base_type     = comp::base<progress_point, void>;
    using value_type    = int64_t;
    using hash_type     = tim::hash_value_t;
    using iterator_type = progress_point*;

    static std::string label();
    static std::string description();

    ROCPROFSYS_DEFAULT_OBJECT(progress_point)

    void            start();
    void            stop();
    void            mark();
    void            set_value(int64_t);
    progress_point& operator+=(const progress_point&);
    progress_point& operator-=(const progress_point&);

    bool is_throughput_point() const;
    bool is_latency_point() const;
    void print(std::ostream& os) const;

    void    set_hash(hash_type _v) { m_hash = _v; }
    void    set_iterator(iterator_type _v) { m_iterator = _v; }
    auto    get_iterator() const { return m_iterator; }
    auto    get_hash() const { return m_hash; }
    int64_t get_delta() const;
    int64_t get_arrival() const;
    int64_t get_departure() const;
    int64_t get_latency_delta() const;
    int64_t get_laps() const;

    template <typename ArchiveT>
    void load(ArchiveT& ar, const unsigned)
    {
        namespace cereal = ::tim::cereal;
        auto _name       = std::string{};

        ar(cereal::make_nvp("name", _name));
        ar(cereal::make_nvp("delta", m_delta));
        ar(cereal::make_nvp("arrival", m_arrival));
        ar(cereal::make_nvp("departure", m_departure));
        m_hash = tim::hash::add_hash_id(_name);
    }

    template <typename ArchiveT>
    void save(ArchiveT& ar, const unsigned) const
    {
        namespace cereal = ::tim::cereal;
        ar(cereal::make_nvp("hash", m_hash));
        ar(cereal::make_nvp("name", std::string{ tim::get_hash_identifier(m_hash) }));
        ar(cereal::make_nvp("delta", m_delta));
        ar(cereal::make_nvp("arrival", m_arrival));
        ar(cereal::make_nvp("departure", m_departure));
    }

    static std::unordered_map<tim::hash_value_t, progress_point> get_progress_points();

private:
    hash_type       m_hash      = 0;
    int64_t         m_delta     = 0;
    int64_t         m_arrival   = 0;
    int64_t         m_departure = 0;
    progress_point* m_iterator  = nullptr;
};
}  // namespace component
}  // namespace causal
}  // namespace rocprofsys

ROCPROFSYS_DEFINE_CONCRETE_TRAIT(uses_storage, causal::component::progress_point,
                                 false_type)
ROCPROFSYS_DEFINE_CONCRETE_TRAIT(flat_storage, causal::component::progress_point,
                                 true_type)
ROCPROFSYS_DEFINE_CONCRETE_TRAIT(uses_timing_units, causal::component::progress_point,
                                 true_type)
ROCPROFSYS_DEFINE_CONCRETE_TRAIT(is_timing_category, causal::component::progress_point,
                                 true_type)

namespace tim
{
namespace operation
{
template <>
struct push_node<rocprofsys::causal::component::progress_point>
{
    using type = rocprofsys::causal::component::progress_point;

    ROCPROFSYS_DEFAULT_OBJECT(push_node)

    push_node(type& _obj, scope::config _scope, hash_value_t _hash,
              int64_t _tid = threading::get_id())
    {
        (*this)(_obj, _scope, _hash, _tid);
    }

    void operator()(type& _obj, scope::config, hash_value_t _hash,
                    int64_t _tid = threading::get_id()) const;
};

template <>
struct pop_node<rocprofsys::causal::component::progress_point>
{
    using type = rocprofsys::causal::component::progress_point;

    ROCPROFSYS_DEFAULT_OBJECT(pop_node)

    pop_node(type& _obj, int64_t _tid = threading::get_id()) { (*this)(_obj, _tid); }

    void operator()(type& _obj, int64_t _tid = threading::get_id()) const;
};
}  // namespace operation
}  // namespace tim
