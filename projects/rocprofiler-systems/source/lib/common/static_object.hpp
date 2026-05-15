// MIT License
//
// Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
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

#include <array>
#include <cstddef>
#include <deque>
#include <functional>
#include <iostream>
#include <mutex>
#include <stack>

namespace rocprofsys
{
inline namespace common
{
using static_dtor_func_t = void (*)();

void
destroy_static_objects();

void
register_static_dtor(static_dtor_func_t&&);

namespace
{
struct anonymous
{};
}  // namespace

struct do_not_destroy
{};

template <typename Tp>
constexpr size_t
static_buffer_size()
{
    return sizeof(Tp);
}

/**
 * @brief This struct is used to create static singleton objects which have the properties
 * of a heap-allocated static object without a memory leak.
 *
 * @tparam Tp Data type of singleton
 * @tparam ContextT Use to differentiate singletons in different translation units (if
 * using default parameter) or ensure the singleton can be accessed in different
 * translation units (not recommended) as long as this type is not in an anonymous
 * namespace
 *
 * This template works by creating a buffer of at least `sizeof(Tp)` bytes in the binary
 * and does a placement new into that buffer. The object created is NOT heap allocated,
 * the address of the object is an address in between the library load address and the
 * load address + size of library.
 */
template <typename Tp, typename ContextT = anonymous>
struct static_object
{
    static_object()                                    = delete;
    ~static_object()                                   = delete;
    static_object(const static_object&)                = delete;
    static_object(static_object&&) noexcept            = delete;
    static_object& operator=(const static_object&)     = delete;
    static_object& operator=(static_object&&) noexcept = delete;

    template <typename... Args>
    static Tp*& construct(Args&&... args);

    template <typename... Args>
    static Tp*& construct(do_not_destroy&&, Args&&... args);

    static Tp* get() { return m_object; }

    static constexpr bool is_trivial_standard_layout();

private:
    static Tp*                                             m_object;
    static std::array<std::byte, static_buffer_size<Tp>()> m_buffer;
};

template <typename Tp, typename ContextT>
Tp* static_object<Tp, ContextT>::m_object = nullptr;

template <typename Tp, typename ContextT>
std::array<std::byte, static_buffer_size<Tp>()>
    static_object<Tp, ContextT>::m_buffer = {};

template <typename Tp, typename ContextT>
constexpr bool
static_object<Tp, ContextT>::is_trivial_standard_layout()
{
    return (std::is_standard_layout<Tp>::value && std::is_trivial<Tp>::value);
}

template <typename Tp, typename ContextT>
template <typename... Args>
Tp*&
static_object<Tp, ContextT>::construct(Args&&... args)
{
    if constexpr(!is_trivial_standard_layout())
    {
        static auto _once = std::once_flag{};
        std::call_once(_once, []() {
            register_static_dtor([]() {
                if(static_object<Tp, ContextT>::m_object)
                {
                    static_object<Tp, ContextT>::m_object->~Tp();
                    static_object<Tp, ContextT>::m_object = nullptr;
                }
            });
        });
    }

    if(m_object)
    {
        std::cerr
            << "reconstructing static object. Use get() function to retrieve pointer"
            << std::endl;
        abort();
    }

    m_object = new(m_buffer.data()) Tp{ std::forward<Args>(args)... };
    return m_object;
}

template <typename Tp, typename ContextT>
template <typename... Args>
Tp*&
static_object<Tp, ContextT>::construct(do_not_destroy&&, Args&&... args)
{
    if(m_object)
    {
        std::cerr
            << "reconstructing static object. Use get() function to retrieve pointer"
            << std::endl;
        abort();
    }

    m_object = new(m_buffer.data()) Tp{ std::forward<Args>(args)... };
    return m_object;
}

namespace
{
inline auto*&
get_static_object_stack()
{
    static auto* _v = new std::stack<static_dtor_func_t>{};
    return _v;
}
}  // namespace

inline void
destroy_static_objects()
{
    static auto _sync = std::mutex{};
    auto        _lk   = std::unique_lock<std::mutex>{ _sync };

    auto*& _stack = get_static_object_stack();
    if(_stack)
    {
        while(!_stack->empty())
        {
            auto& itr = _stack->top();
            if(itr) itr();
            _stack->pop();
        }

        delete _stack;
        _stack = nullptr;
    }
}

inline void
register_static_dtor(static_dtor_func_t&& _func)
{
    static auto _sync = std::mutex{};
    auto        _lk   = std::unique_lock<std::mutex>{ _sync };

    auto*& _stack = get_static_object_stack();
    if(_stack)
    {
        _stack->push(_func);
    }
}
}  // namespace common
}  // namespace rocprofsys
