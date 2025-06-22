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
#include "core/concepts.hpp"
#include "core/config.hpp"
#include "core/containers/stable_vector.hpp"
#include "core/debug.hpp"
#include "core/defines.hpp"
#include "core/state.hpp"
#include "core/timemory.hpp"
#include "core/utility.hpp"
#include "library/thread_deleter.hpp"

#include <timemory/utility/macros.hpp>
#include <timemory/utility/types.hpp>

#include <array>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <optional>
#include <type_traits>
#include <vector>

namespace rocprofsys
{
// bundle of components used in instrumentation
using instrumentation_bundle_t =
    tim::component_bundle<project::rocprofsys, comp::wall_clock*,
                          comp::user_global_bundle*>;

// allocator for instrumentation_bundle_t
using bundle_allocator_t = tim::data::ring_buffer_allocator<instrumentation_bundle_t>;

using grow_functor_t = int64_t (*)(int64_t);

inline auto&
grow_functors()
{
    static auto _v = container::stable_vector<grow_functor_t>{};
    return _v;
}

template <typename Tp>
struct base_thread_data
{
    base_thread_data()
    {
        auto _func = [](int64_t _sz) -> int64_t {
            decltype(auto) _v = Tp::private_instance();
            if(_v && _v->capacity() < static_cast<size_t>(_sz + 1))
            {
                _v->reserve(_v->capacity() + 1);
                _v->resize(_v->capacity());
            }
            return (_v) ? _v->capacity() : 0;
        };
        grow_functors().emplace_back(std::move(_func));
    }
};

template <typename Tp, typename Tag = void, size_t MaxThreads = max_supported_threads>
struct thread_data;

template <typename Tp, typename Tag, size_t MaxThreads>
struct use_placement_new_when_generating_unique_ptr<thread_data<Tp, Tag, MaxThreads>>
: std::true_type
{};

template <typename Tp, typename Tag, size_t MaxThreads>
struct use_placement_new_when_generating_unique_ptr<
    thread_data<std::optional<Tp>, Tag, MaxThreads>> : std::true_type
{};

template <typename Tp, typename Tag, size_t MaxThreads>
struct use_placement_new_when_generating_unique_ptr<
    thread_data<identity<Tp>, Tag, MaxThreads>> : std::true_type
{};

template <typename Tp, typename Tag, size_t MaxThreads>
struct thread_data : base_thread_data<thread_data<Tp, Tag, MaxThreads>>
{
    using this_type  = thread_data<Tp, Tag, MaxThreads>;
    using value_type = unique_ptr_t<Tp>;
    using array_type =
        container::stable_vector<value_type, MaxThreads, container::cacheline_align_v>;
    using functor_type = std::function<value_type()>;

    template <typename... Args>
    static void        construct(construct_on_thread&&, Args&&...);
    static value_type& instance();
    template <typename... Args>
    static value_type& instance(construct_on_thread&&, Args&&...);

    template <typename... Args>
    static void construct(Args&&... args)
    {
        construct(construct_on_thread{}, std::forward<Args>(args)...);
    }

    template <typename... Args>
    static value_type& instance(Args&&... args)
    {
        return instance(construct_on_thread{}, std::forward<Args>(args)...);
    }

    static size_t size() { return private_instance()->m_data.size(); }

    decltype(auto) data() { return m_data; }
    decltype(auto) data() const { return m_data; }

    decltype(auto) begin() { return m_data.begin(); }
    decltype(auto) end() { return m_data.end(); }

    decltype(auto) begin() const { return m_data.begin(); }
    decltype(auto) end() const { return m_data.end(); }

    decltype(auto) at(size_t _idx) { return m_data.at(_idx); }
    decltype(auto) at(size_t _idx) const { return m_data.at(_idx); }

    decltype(auto) operator[](size_t _idx) { return m_data[_idx]; }
    decltype(auto) operator[](size_t _idx) const { return m_data[_idx]; }

    decltype(auto) reserve(size_t _n) { return m_data.reserve(_n); }
    decltype(auto) capacity() const { return m_data.capacity(); }
    decltype(auto) empty() const { return m_data.empty(); }

    void resize(size_t _n) { container::resize(m_data, _n, m_init()); }

    template <typename Up>
    void resize(size_t _n, Up&& _v)
    {
        static_assert(std::is_assignable<value_type, Up>::value,
                      "value is not assignable to optional<Tp>");
        container::resize(m_data, _n, std::forward<Up>(_v));
    }

    static array_type* get()
    {
        return (private_instance()) ? &private_instance()->m_data : nullptr;
    }

private:
    friend struct base_thread_data<this_type>;
    static unique_ptr_t<this_type>& private_instance();
    static array_type&              instances();

    template <typename... Args>
    static array_type& instances(construct_on_init, Args&&...);

    array_type   m_data = array_type(MaxThreads);
    functor_type m_init = []() { return value_type{}; };
};

template <typename Tp, typename Tag, size_t MaxThreads>
unique_ptr_t<thread_data<Tp, Tag, MaxThreads>>&
thread_data<Tp, Tag, MaxThreads>::private_instance()
{
    static auto _v = unique_ptr_t<this_type>{ new this_type{} };
    return _v;
}

template <typename Tp, typename Tag, size_t MaxThreads>
template <typename... Args>
void
thread_data<Tp, Tag, MaxThreads>::construct(construct_on_thread&& _t, Args&&... _args)
{
    // construct outside of lambda to prevent data-race
    static auto& _instances = instances();
    if(!_instances.at(_t.index))
        _instances.at(_t.index) =
            utility::generate<value_type>{}(std::forward<Args>(_args)...);
}

template <typename Tp, typename Tag, size_t MaxThreads>
unique_ptr_t<Tp>&
thread_data<Tp, Tag, MaxThreads>::instance()
{
    return instances().at(threading::get_id());
}

template <typename Tp, typename Tag, size_t MaxThreads>
typename thread_data<Tp, Tag, MaxThreads>::array_type&
thread_data<Tp, Tag, MaxThreads>::instances()
{
    return private_instance()->m_data;
}

template <typename Tp, typename Tag, size_t MaxThreads>
template <typename... Args>
unique_ptr_t<Tp>&
thread_data<Tp, Tag, MaxThreads>::instance(construct_on_thread&& _t, Args&&... _args)
{
    construct(construct_on_thread{ _t }, std::forward<Args>(_args)...);
    return instances().at(_t.index);
}

template <typename Tp, typename Tag, size_t MaxThreads>
template <typename... Args>
typename thread_data<Tp, Tag, MaxThreads>::array_type&
thread_data<Tp, Tag, MaxThreads>::instances(construct_on_init, Args&&... _args)
{
    static auto& _v = [&]() -> array_type& {
        auto& _internal = instances();
        for(size_t i = 0; i < MaxThreads; ++i)
            _internal.at(i) =
                utility::generate<value_type>{}(std::forward<Args>(_args)...);
        private_instance()->m_init = [_args...]() {
            return utility::generate<value_type>{}(_args...);
        };
        return _internal;
    }();
    return _v;
}

//--------------------------------------------------------------------------------------//
//
//          thread_data with std::optional
//
//--------------------------------------------------------------------------------------//

template <typename Tp, typename Tag, size_t MaxThreads>
struct thread_data<std::optional<Tp>, Tag, MaxThreads>
: base_thread_data<thread_data<std::optional<Tp>, Tag, MaxThreads>>
{
    using this_type    = thread_data<std::optional<Tp>, Tag, MaxThreads>;
    using value_type   = std::optional<Tp>;
    using functor_type = std::function<value_type()>;
    using array_type =
        container::stable_vector<value_type, MaxThreads, container::cacheline_align_v>;

    thread_data()  = default;
    ~thread_data() = default;

    explicit thread_data(functor_type&& _init)
    : m_init{ std::move(_init) }
    {}

    thread_data(const thread_data&)     = default;
    thread_data(thread_data&&) noexcept = default;

    thread_data& operator=(const thread_data&)     = default;
    thread_data& operator=(thread_data&&) noexcept = default;

    static unique_ptr_t<this_type>& instance();

    template <typename... Args>
    static unique_ptr_t<this_type>& instance(construct_on_init, Args&&...);

    template <typename... Args>
    static value_type& instance(construct_on_thread&&, Args&&...);

    template <typename... Args>
    static unique_ptr_t<this_type>& construct(construct_on_init, Args&&...);

    template <typename... Args>
    static value_type& construct(construct_on_thread&&, Args&&...);

    size_t size() { return m_data.size(); }

    decltype(auto) data() { return m_data; }
    decltype(auto) data() const { return m_data; }

    decltype(auto) begin() { return m_data.begin(); }
    decltype(auto) end() { return m_data.end(); }

    decltype(auto) begin() const { return m_data.begin(); }
    decltype(auto) end() const { return m_data.end(); }

    decltype(auto) at(size_t _idx) { return m_data.at(_idx); }
    decltype(auto) at(size_t _idx) const { return m_data.at(_idx); }

    decltype(auto) operator[](size_t _idx) { return m_data[_idx]; }
    decltype(auto) operator[](size_t _idx) const { return m_data[_idx]; }

    decltype(auto) reserve(size_t _n) { return m_data.reserve(_n); }
    decltype(auto) capacity() const { return m_data.capacity(); }
    decltype(auto) empty() const { return m_data.empty(); }

    void resize(size_t _n) { container::resize(m_data, _n, m_init()); }

    template <typename Up>
    void resize(size_t _n, Up&& _v)
    {
        static_assert(std::is_assignable<value_type, Up>::value,
                      "value is not assignable to optional<Tp>");
        container::resize(m_data, _n, std::forward<Up>(_v));
    }

private:
    friend struct base_thread_data<this_type>;
    static decltype(auto) private_instance() { return instance(); }

    array_type   m_data = {};
    functor_type m_init = []() { return value_type{}; };
};

template <typename Tp, typename Tag, size_t MaxThreads>
unique_ptr_t<thread_data<std::optional<Tp>, Tag, MaxThreads>>&
thread_data<std::optional<Tp>, Tag, MaxThreads>::instance()
{
    static auto _v = unique_ptr_t<this_type>{};
    return _v;
}

template <typename Tp, typename Tag, size_t MaxThreads>
template <typename... Args>
unique_ptr_t<thread_data<std::optional<Tp>, Tag, MaxThreads>>&
thread_data<std::optional<Tp>, Tag, MaxThreads>::instance(construct_on_init,
                                                          Args&&... _args)
{
    static auto& _v = [&]() -> unique_ptr_t<this_type>& {
        auto& _ref = instance();
        if(!_ref)
            _ref = utility::generate<unique_ptr_t<this_type>>{}(
                std::forward<Args>(_args)...);
        if(_ref->size() < MaxThreads) _ref->resize(MaxThreads);
        return _ref;
    }();
    return _v;
}

template <typename Tp, typename Tag, size_t MaxThreads>
template <typename... Args>
unique_ptr_t<thread_data<std::optional<Tp>, Tag, MaxThreads>>&
thread_data<std::optional<Tp>, Tag, MaxThreads>::construct(construct_on_init,
                                                           Args&&... _args)
{
    // construct outside of lambda to prevent data-race
    static auto& _ref = instance(construct_on_init{});
    static auto  _v   = [&]() {
        if(_ref)
        {
            for(auto& itr : *_ref)
                itr = utility::generate<value_type>{}(std::forward<Args>(_args)...);
        }
        return (_ref != nullptr);
    }();
    return _ref;
    (void) _v;
}

template <typename Tp, typename Tag, size_t MaxThreads>
template <typename... Args>
std::optional<Tp>&
thread_data<std::optional<Tp>, Tag, MaxThreads>::construct(construct_on_thread&& _t,
                                                           Args&&... _args)
{
    // construct outside of lambda to prevent data-race
    static auto& _instance = instance(construct_on_init{});
    static auto  _constructed =
        container::stable_vector<bool, MaxThreads, container::cacheline_align_v>{};
    static auto _grow = []() {
        container::resize(_constructed, MaxThreads, false);
        grow_functors().emplace_back([](int64_t _n) -> int64_t {
            if(static_cast<size_t>(_n) >= _constructed.size())
            {
                _constructed.reserve(_constructed.capacity() + 1);
                container::resize(_constructed, _constructed.capacity(), false);
            }
            return _constructed.size();
        });
        return true;
    }();

    if(!_constructed.at(_t.index))
        _constructed.at(_t.index) =
            (_instance->at(_t.index) =
                 utility::generate<value_type>{}(std::forward<Args>(_args)...),
             true);

    return _instance->at(_t.index);

    (void) _grow;
}

template <typename Tp, typename Tag, size_t MaxThreads>
template <typename... Args>
std::optional<Tp>&
thread_data<std::optional<Tp>, Tag, MaxThreads>::instance(construct_on_thread&& _t,
                                                          Args&&... _args)
{
    construct(construct_on_thread{ _t }, std::forward<Args>(_args)...);
    return instance()->at(_t.index);
}

//--------------------------------------------------------------------------------------//
//
//          thread_data with raw data (no pointer)
//
//--------------------------------------------------------------------------------------//

template <typename Tp, typename Tag, size_t MaxThreads>
struct thread_data<identity<Tp>, Tag, MaxThreads>
: base_thread_data<thread_data<identity<Tp>, Tag, MaxThreads>>
{
    using this_type  = thread_data<identity<Tp>, Tag, MaxThreads>;
    using value_type = Tp;
    using array_type =
        container::stable_vector<value_type, MaxThreads, container::cacheline_align_v>;
    using functor_type = std::function<value_type()>;

    thread_data()  = default;
    ~thread_data() = default;

    explicit thread_data(functor_type&& _init)
    : m_init{ std::move(_init) }
    {}

    thread_data(const thread_data&)     = default;
    thread_data(thread_data&&) noexcept = default;

    thread_data& operator=(const thread_data&)     = default;
    thread_data& operator=(thread_data&&) noexcept = default;

    static unique_ptr_t<this_type>& instance();

    template <typename... Args>
    static unique_ptr_t<this_type>& instance(construct_on_init, Args&&...);

    template <typename... Args>
    static value_type& instance(construct_on_thread&&, Args&&...);

    template <typename... Args>
    static unique_ptr_t<this_type>& construct(construct_on_init, Args&&...);

    template <typename... Args>
    static value_type& construct(construct_on_thread&&, Args&&...);

    size_t size() { return m_data.size(); }

    decltype(auto) data() { return m_data; }
    decltype(auto) data() const { return m_data; }

    decltype(auto) begin() { return m_data.begin(); }
    decltype(auto) end() { return m_data.end(); }

    decltype(auto) begin() const { return m_data.begin(); }
    decltype(auto) end() const { return m_data.end(); }

    decltype(auto) at(size_t _idx) { return m_data.at(_idx); }
    decltype(auto) at(size_t _idx) const { return m_data.at(_idx); }

    decltype(auto) operator[](size_t _idx) { return m_data[_idx]; }
    decltype(auto) operator[](size_t _idx) const { return m_data[_idx]; }

    decltype(auto) reserve(size_t _n) { return m_data.reserve(_n); }
    decltype(auto) capacity() const { return m_data.capacity(); }
    decltype(auto) empty() const { return m_data.empty(); }

    void resize(size_t _n) { container::resize(m_data, _n, m_init()); }
    void resize(size_t _n, value_type&& _v) { container::resize(m_data, _n, _v); }

    void fill(value_type _v)
    {
        for(auto& itr : m_data)
            itr = _v;
    }

private:
    friend struct base_thread_data<this_type>;
    static decltype(auto) private_instance() { return instance(); }

    array_type   m_data = {};
    functor_type m_init = []() { return value_type{}; };
};

template <typename Tp, typename Tag, size_t MaxThreads>
unique_ptr_t<thread_data<identity<Tp>, Tag, MaxThreads>>&
thread_data<identity<Tp>, Tag, MaxThreads>::instance()
{
    static auto _v = unique_ptr_t<this_type>{};
    return _v;
}

template <typename Tp, typename Tag, size_t MaxThreads>
template <typename... Args>
unique_ptr_t<thread_data<identity<Tp>, Tag, MaxThreads>>&
thread_data<identity<Tp>, Tag, MaxThreads>::instance(construct_on_init, Args&&... _args)
{
    static auto& _v = [&]() -> unique_ptr_t<this_type>& {
        auto& _ref = instance();
        if(!_ref)
            _ref = utility::generate<unique_ptr_t<this_type>>{}(
                std::forward<Args>(_args)...);
        if(_ref->size() < MaxThreads) _ref->resize(MaxThreads);
        return _ref;
    }();
    return _v;
}

template <typename Tp, typename Tag, size_t MaxThreads>
template <typename... Args>
unique_ptr_t<thread_data<identity<Tp>, Tag, MaxThreads>>&
thread_data<identity<Tp>, Tag, MaxThreads>::construct(construct_on_init, Args&&... _args)
{
    // construct outside of lambda to prevent data-race
    static auto& _ref = instance(construct_on_init{});
    static auto  _v   = [&]() {
        if(_ref)
        {
            for(auto& itr : *_ref)
                itr = utility::generate<value_type>{}(std::forward<Args>(_args)...);
        }
        return (_ref != nullptr);
    }();
    return _ref;
    (void) _v;
}

template <typename Tp, typename Tag, size_t MaxThreads>
template <typename... Args>
Tp&
thread_data<identity<Tp>, Tag, MaxThreads>::construct(construct_on_thread&& _t,
                                                      Args&&... _args)
{
    // construct outside of lambda to prevent data-race
    static auto& _instance = instance(construct_on_init{});
    static auto  _constructed =
        container::stable_vector<bool, MaxThreads, container::cacheline_align_v>{};
    static auto _grow = []() {
        container::resize(_constructed, MaxThreads, false);
        grow_functors().emplace_back([](int64_t _n) -> int64_t {
            if(static_cast<size_t>(_n) >= _constructed.size())
            {
                _constructed.reserve(_constructed.capacity() + 1);
                container::resize(_constructed, _constructed.capacity(), false);
            }
            return _constructed.size();
        });
        return true;
    }();

    if(!_constructed.at(_t.index))
        _constructed.at(_t.index) =
            (_instance->at(_t.index) =
                 utility::generate<value_type>{}(std::forward<Args>(_args)...),
             true);

    return _instance->at(_t.index);
    (void) _grow;
}

template <typename Tp, typename Tag, size_t MaxThreads>
template <typename... Args>
Tp&
thread_data<identity<Tp>, Tag, MaxThreads>::instance(construct_on_thread&& _t,
                                                     Args&&... _args)
{
    construct(construct_on_thread{ _t }, std::forward<Args>(_args)...);
    return instance()->at(_t.index);
}

//--------------------------------------------------------------------------------------//

// there are currently some strange things that happen with
// vector<instrumentation_bundle_t> so using vector<instrumentation_bundle_t*> and
// timemory's ring_buffer_allocator to create contiguous memory-page aligned instances of
// the bundle
template <typename... Tp>
struct component_bundle_cache_impl
{
    using this_type      = component_bundle_cache_impl<Tp...>;
    using bundle_type    = tim::component_bundle<project::rocprofsys, Tp...>;
    using allocator_type = tim::data::ring_buffer_allocator<bundle_type>;
    using array_type     = std::vector<bundle_type*>;

    using iterator         = typename array_type::iterator;
    using const_iterator   = typename array_type::const_iterator;
    using reverse_iterator = typename array_type::reverse_iterator;

    component_bundle_cache_impl()  = default;
    ~component_bundle_cache_impl() = default;

    component_bundle_cache_impl(const component_bundle_cache_impl&)     = delete;
    component_bundle_cache_impl(component_bundle_cache_impl&&) noexcept = delete;

    component_bundle_cache_impl& operator=(const component_bundle_cache_impl&) = delete;
    component_bundle_cache_impl& operator=(component_bundle_cache_impl&&) noexcept =
        delete;

    bool empty() const { return m_bundles.empty(); }

    auto& front() { return m_bundles.front(); }
    auto& front() const { return m_bundles.front(); }

    auto& back() { return m_bundles.back(); }
    auto& back() const { return m_bundles.back(); }

    auto begin() { return m_bundles.begin(); }
    auto end() { return m_bundles.end(); }

    auto rbegin() { return m_bundles.rbegin(); }
    auto rend() { return m_bundles.rend(); }

    auto begin() const { return m_bundles.begin(); }
    auto end() const { return m_bundles.end(); }

    auto size() const { return m_bundles.size(); }

    auto&       at(size_t _idx) { return m_bundles.at(_idx); }
    const auto& at(size_t _idx) const { return m_bundles.at(_idx); }

    template <typename... Args>
    bundle_type* construct(Args&&... args)
    {
        bundle_type* _v = m_allocator.allocate(1);
        m_allocator.construct(_v, std::forward<Args>(args)...);
        return m_bundles.emplace_back(_v);
    }

    void destroy(bundle_type* _v, size_t _idx)
    {
        m_allocator.destroy(_v);
        m_allocator.deallocate(_v, 1);
        m_bundles.erase(m_bundles.begin() + _idx);
    }

    void pop_back()
    {
        bundle_type* _v = m_bundles.back();
        m_allocator.destroy(_v);
        m_allocator.deallocate(_v, 1);
        m_bundles.pop_back();
    }

    template <typename IterT>
    void destroy(IterT _v)
    {
        iterator itr = begin();
        if constexpr(std::is_same<IterT, reverse_iterator>::value)
        {
            if(_v == rend()) return;
            std::advance(itr, std::distance(rbegin(), _v));
        }
        else
        {
            if(_v == end()) return;
            itr = _v;
        }
        m_allocator.destroy(*itr);
        m_allocator.deallocate(*itr, 1);
        m_bundles.erase(itr);
    }

private:
    allocator_type m_allocator = {};
    array_type     m_bundles   = {};
};

template <typename... Tp>
struct component_bundle_cache_impl<tim::component_bundle<project::rocprofsys, Tp...>>
: component_bundle_cache_impl<Tp...>
{
    using base_type = component_bundle_cache_impl<Tp...>;
};

//--------------------------------------------------------------------------------------//

template <typename... Tp>
using component_bundle_cache  = thread_data<component_bundle_cache_impl<Tp...>>;
using instrumentation_bundles = component_bundle_cache<instrumentation_bundle_t>;

extern template struct component_bundle_cache_impl<instrumentation_bundle_t>;
}  // namespace rocprofsys
