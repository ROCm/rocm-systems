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
#include "lib/common/memory/deleter.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <new>
#include <stdexcept>

namespace rocprofiler
{
namespace common
{
namespace memory
{
template <typename Tp, size_t Alignment = 64, typename DeleterT = deleter<void>>
class stateless_allocator
{
public:
    using value_type                             = Tp;
    using pointer                                = Tp*;
    using const_pointer                          = const Tp*;
    using reference                              = Tp&;
    using const_reference                        = const Tp&;
    using size_type                              = size_t;
    using difference_type                        = ptrdiff_t;
    using propagate_on_container_move_assignment = std::true_type;

    template <typename Up>
    struct rebind
    {
        using other = stateless_allocator<Up, Alignment, DeleterT>;
    };

    stateless_allocator()                                   = default;
    stateless_allocator(const stateless_allocator& rhs)     = default;
    stateless_allocator(stateless_allocator&& rhs) noexcept = default;
    stateless_allocator& operator=(const stateless_allocator& rhs) = default;
    stateless_allocator& operator=(stateless_allocator&& rhs) noexcept = default;

    template <typename Up>
    stateless_allocator(const stateless_allocator<Up, Alignment, DeleterT>& rhs);

    static Tp*  allocate(size_t n);
    static void deallocate(Tp* ptr, size_t n);
    static void construct(value_type* const _p, const value_type& _v);
    static void construct(value_type* const _p, value_type&& _v);
    static void construct_at(value_type* const _p, const value_type& _v);
    static void construct_at(value_type* const _p, value_type&& _v);
    static void destroy(value_type* const _p);
    static void destroy_at(value_type* const _p);
};

template <typename Tp, size_t Alignment, typename DeleterT>
template <typename Up>
stateless_allocator<Tp, Alignment, DeleterT>::stateless_allocator(
    const stateless_allocator<Up, Alignment, DeleterT>& rhs)
{
    (void) rhs;
}

template <typename Tp, size_t Alignment, typename DeleterT>
Tp*
stateless_allocator<Tp, Alignment, DeleterT>::allocate(size_t n)
{
    constexpr auto alignment_v = Alignment / sizeof(void*);
    Tp*            ptr         = nullptr;

    if constexpr(sizeof(Tp) >= alignment_v && sizeof(Tp) % alignment_v == 0)
        ptr = static_cast<Tp*>(::aligned_alloc(Alignment / sizeof(void*), sizeof(Tp) * n));
    else
        ptr = static_cast<Tp*>(::malloc(sizeof(Tp) * n));

    if(ptr) return ptr;

    throw std::bad_alloc{};
}

template <typename Tp, size_t Alignment, typename DeleterT>
void
stateless_allocator<Tp, Alignment, DeleterT>::deallocate(Tp* ptr, size_t n)
{
    (void) n;
    ::free(ptr);
}

template <typename Tp, size_t Alignment, typename DeleterT>
void
stateless_allocator<Tp, Alignment, DeleterT>::construct(value_type* const _p, const value_type& _v)
{
    ::new((void*) _p) value_type{_v};
}

template <typename Tp, size_t Alignment, typename DeleterT>
void
stateless_allocator<Tp, Alignment, DeleterT>::construct(value_type* const _p, value_type&& _v)
{
    ::new((void*) _p) value_type{std::move(_v)};
}

template <typename Tp, size_t Alignment, typename DeleterT>
void
stateless_allocator<Tp, Alignment, DeleterT>::construct_at(value_type* const _p,
                                                           const value_type& _v)
{
    ::new((void*) _p) value_type{_v};
}

template <typename Tp, size_t Alignment, typename DeleterT>
void
stateless_allocator<Tp, Alignment, DeleterT>::construct_at(value_type* const _p, value_type&& _v)
{
    ::new((void*) _p) value_type{std::move(_v)};
}

template <typename Tp, size_t Alignment, typename DeleterT>
void
stateless_allocator<Tp, Alignment, DeleterT>::destroy(value_type* const _p)
{
    DeleterT{}();
    _p->~value_type();
}

template <typename Tp, size_t Alignment, typename DeleterT>
void
stateless_allocator<Tp, Alignment, DeleterT>::destroy_at(value_type* const _p)
{
    DeleterT{}();
    _p->~value_type();
}

template <typename LhsTp,
          size_t LhsAlignment,
          typename LhsDeleterT,
          typename RhsTp,
          size_t RhsAlignment,
          typename RhsDeleterT>
constexpr bool
operator==(const stateless_allocator<LhsTp, LhsAlignment, LhsDeleterT>&,
           const stateless_allocator<RhsTp, RhsAlignment, RhsDeleterT>&)
{
    return true;
}

template <typename LhsTp,
          size_t LhsAlignment,
          typename LhsDeleterT,
          typename RhsTp,
          size_t RhsAlignment,
          typename RhsDeleterT>
constexpr bool
operator!=(const stateless_allocator<LhsTp, LhsAlignment, LhsDeleterT>&,
           const stateless_allocator<RhsTp, RhsAlignment, RhsDeleterT>&)
{
    return false;
}
}  // namespace memory
}  // namespace common
}  // namespace rocprofiler
