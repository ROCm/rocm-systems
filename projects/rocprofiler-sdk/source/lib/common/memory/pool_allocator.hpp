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
#include "lib/common/memory/pool.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <memory>

namespace rocprofiler
{
namespace common
{
namespace memory
{
// template <typename Tp, size_t Alignment, size_t BlockSize, size_t ReservedBlocks, typename
// DeleterT> pool_allocator<Tp, Alignment, BlockSize, ReservedBlocks, DeleterT>::
template <typename Tp,
          size_t Alignment      = 64,
          size_t BlockSize      = 4096,
          size_t ReservedBlocks = 0,
          typename DeleterT     = deleter<void>>
class pool_allocator
{
public:
    using value_type                             = Tp;
    using pointer                                = Tp*;
    using const_pointer                          = const Tp*;
    using reference                              = Tp&;
    using const_reference                        = const Tp&;
    using size_type                              = size_t;
    using difference_type                        = ptrdiff_t;
    using propagate_on_container_move_assignment = std::false_type;
    using is_always_equal                        = value_type;

    pool_allocator() = default;

    // Rebind copy constructor
    template <typename Up>
    pool_allocator(const pool_allocator<Up>& rhs);

    pool_allocator(const pool_allocator& rhs)     = default;
    pool_allocator(pool_allocator&& rhs) noexcept = default;
    pool_allocator& operator=(const pool_allocator& rhs) = default;
    pool_allocator& operator=(pool_allocator&& rhs) noexcept = default;

    value_type* allocate(size_t n);
    void        deallocate(value_type* ptr, size_t n);
    void        construct(value_type* const _p, const value_type& _v) const;
    void        construct(value_type* const _p, value_type&& _v) const;
    void        construct_at(value_type* const _p, const value_type& _v) const;
    void        construct_at(value_type* const _p, value_type&& _v) const;
    void        destroy(value_type* const _p) const;
    void        destroy_at(value_type* const _p) const;

    template <typename Up>
    struct rebind
    {
        using other = pool_allocator<Up, Alignment, BlockSize, ReservedBlocks>;
    };

private:
    using pool_type = pool<BlockSize, ReservedBlocks>;

    std::shared_ptr<pool_type> m_pool = std::make_shared<pool_type>(sizeof(value_type));
};

template <typename Tp, size_t AlignV, size_t BlockSz, size_t ReservedBlocks, typename DeleterT>
template <typename Up>
pool_allocator<Tp, AlignV, BlockSz, ReservedBlocks, DeleterT>::pool_allocator(
    const pool_allocator<Up>& rhs)
: m_pool{rhs.m_pool}
{
    m_pool->rebind(sizeof(value_type));
}

template <typename Tp, size_t AlignV, size_t BlockSz, size_t ReservedBlocks, typename DeleterT>
typename pool_allocator<Tp, AlignV, BlockSz, ReservedBlocks, DeleterT>::value_type*
pool_allocator<Tp, AlignV, BlockSz, ReservedBlocks, DeleterT>::allocate(size_t n)
{
    if(n > 1)
    {
        return static_cast<value_type*>(::aligned_alloc(AlignV, sizeof(value_type) * n));
    }

    return static_cast<value_type*>(m_pool->allocate());
}

template <typename Tp, size_t AlignV, size_t BlockSz, size_t ReservedBlocks, typename DeleterT>
void
pool_allocator<Tp, AlignV, BlockSz, ReservedBlocks, DeleterT>::deallocate(value_type* ptr, size_t n)
{
    DeleterT{}();
    if(n > 1)
    {
        ::free(ptr);
        return;
    }

    m_pool->deallocate(ptr);
}

template <typename Tp, size_t AlignV, size_t BlockSz, size_t ReservedV, typename DeleterT>
void
pool_allocator<Tp, AlignV, BlockSz, ReservedV, DeleterT>::construct(value_type* const _p,
                                                                    const value_type& _v) const
{
    ::new((void*) _p) value_type{_v};
}

template <typename Tp, size_t AlignV, size_t BlockSz, size_t ReservedV, typename DeleterT>
void
pool_allocator<Tp, AlignV, BlockSz, ReservedV, DeleterT>::construct(value_type* const _p,
                                                                    value_type&&      _v) const
{
    ::new((void*) _p) value_type{std::move(_v)};
}

template <typename Tp, size_t AlignV, size_t BlockSz, size_t ReservedV, typename DeleterT>
void
pool_allocator<Tp, AlignV, BlockSz, ReservedV, DeleterT>::construct_at(value_type* const _p,
                                                                       const value_type& _v) const
{
    ::new((void*) _p) value_type{_v};
}

template <typename Tp, size_t AlignV, size_t BlockSz, size_t ReservedV, typename DeleterT>
void
pool_allocator<Tp, AlignV, BlockSz, ReservedV, DeleterT>::construct_at(value_type* const _p,
                                                                       value_type&&      _v) const
{
    ::new((void*) _p) value_type{std::move(_v)};
}

template <typename Tp, size_t AlignV, size_t BlockSz, size_t ReservedV, typename DeleterT>
void
pool_allocator<Tp, AlignV, BlockSz, ReservedV, DeleterT>::destroy(value_type* const _p) const
{
    DeleterT{}();
    _p->~value_type();
}

template <typename Tp, size_t AlignV, size_t BlockSz, size_t ReservedV, typename DeleterT>
void
pool_allocator<Tp, AlignV, BlockSz, ReservedV, DeleterT>::destroy_at(value_type* const _p) const
{
    DeleterT{}();
    _p->~value_type();
}
}  // namespace memory
}  // namespace common
}  // namespace rocprofiler
