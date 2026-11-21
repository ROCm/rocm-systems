// MIT License
//
// Copyright (c) 2023-2025 Advanced Micro Devices, Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#pragma once

#include "lib/common/defines.hpp"
#include "lib/common/logging.hpp"
#include "lib/common/static_object.hpp"

#include <array>
#include <cstddef>
#include <functional>
#include <mutex>
#include <type_traits>

namespace rocprofiler
{
namespace common
{
void
destroy_static_tl_objects();

void
register_static_tl_dtor(static_dtor_func_t&&);

/**
 * @brief This struct is used to create static singleton objects which have the properties of a
 * heap-allocated static object without a memory leak.
 *
 * @tparam Tp Data type of singleton
 * @tparam ContextT Use to differentiate singletons in different translation units (if using default
 * parameter) or ensure the singleton can be accessed in different translation units (not
 * recommended) as long as this type is not in an anonymous namespace
 *
 * This template works by creating a buffer of at least `sizeof(Tp)` bytes in the binary and does a
 * placement new into that buffer. The object created is NOT heap allocated, the address of the
 * object is an address in between the library load address and the load address + size of library.
 */
template <typename Tp, typename ContextT = anonymous>
struct static_tl_object
{
    static_tl_object()                            = delete;
    ~static_tl_object()                           = delete;
    static_tl_object(const static_tl_object&)     = delete;
    static_tl_object(static_tl_object&&) noexcept = delete;
    static_tl_object& operator=(const static_tl_object&) = delete;
    static_tl_object& operator=(static_tl_object&&) noexcept = delete;

    template <typename... Args>
    static Tp*& construct(Args&&... args);

    static Tp* get() { return m_object; }

    /// This thread's object, default-constructed on first use. Unlike a `thread_local` of a type
    /// with a non-trivial destructor, first use does not take glibc's dl_load_lock, so it is safe
    /// on paths that run inside HIP/HSA calls and runtime callbacks (see register_static_tl_dtor).
    static Tp& instance();

    static constexpr bool is_trivial_standard_layout();

private:
    static thread_local Tp* m_object;
    alignas(Tp) static thread_local std::array<std::byte, static_buffer_size<Tp>()> m_buffer;
};

template <typename Tp, typename ContextT>
thread_local Tp* static_tl_object<Tp, ContextT>::m_object = nullptr;

template <typename Tp, typename ContextT>
alignas(Tp) thread_local std::
    array<std::byte, static_buffer_size<Tp>()> static_tl_object<Tp, ContextT>::m_buffer = {};

template <typename Tp, typename ContextT>
constexpr bool
static_tl_object<Tp, ContextT>::is_trivial_standard_layout()
{
    return (std::is_standard_layout<Tp>::value && std::is_trivially_destructible<Tp>::value);
}

template <typename Tp, typename ContextT>
template <typename... Args>
Tp*&
static_tl_object<Tp, ContextT>::construct(Args&&... args)
{
    ROCP_FATAL_IF(m_object)
        << "reconstructing static object. Use get() function to retrieve pointer";

    // registered per construction: an object rebuilt on this thread after its destructor ran
    // (e.g. by a later thread-exit destructor) is destroyed again
    if constexpr(!is_trivial_standard_layout())
    {
        register_static_tl_dtor([]() {
            if(static_tl_object<Tp, ContextT>::m_object)
            {
                static_tl_object<Tp, ContextT>::m_object->~Tp();
                static_tl_object<Tp, ContextT>::m_object = nullptr;
            }
        });
    }

    m_object = new(m_buffer.data()) Tp{std::forward<Args>(args)...};
    return m_object;
}

template <typename Tp, typename ContextT>
Tp&
static_tl_object<Tp, ContextT>::instance()
{
    if(ROCPROFILER_UNLIKELY(!m_object)) construct();
    return *m_object;
}
}  // namespace common
}  // namespace rocprofiler
