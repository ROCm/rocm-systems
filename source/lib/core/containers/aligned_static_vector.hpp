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
#include "core/containers/operators.hpp"
#include "core/debug.hpp"
#include "core/exception.hpp"

#include <array>
#include <atomic>
#include <cstdlib>
#include <new>

namespace rocprofsys
{
namespace container
{
#if !defined(ROCPROFSYS_CACHELINE_SIZE)
#    ifdef __cpp_lib_hardware_interference_size
#        define ROCPROFSYS_CACHELINE_SIZE std::hardware_destructive_interference_size
#    else
// 64 bytes on x86-64 │ L1_CACHE_BYTES │ L1_CACHE_SHIFT │ __cacheline_aligned │ ...
#        define ROCPROFSYS_CACHELINE_SIZE 64
#    endif
#endif

constexpr std::size_t cacheline_align_v =
    std::max<size_t>(ROCPROFSYS_CACHELINE_SIZE, ROCPROFSYS_CACHELINE_SIZE_MIN);

template <typename Tp, size_t N, size_t AlignN = cacheline_align_v,
          bool AtomicSizeV = false>
struct aligned_static_vector
{
    struct aligned_value_type
    {
        alignas(AlignN) Tp value = {};
    };

    using count_type      = std::conditional_t<AtomicSizeV, std::atomic<size_t>, size_t>;
    using this_type       = aligned_static_vector<Tp, N>;
    using const_this_type = const aligned_static_vector<Tp, N>;
    using value_type      = Tp;
    using array_type      = std::array<aligned_value_type, N>;
    using reference       = value_type&;
    using const_reference = const value_type&;
    using pointer         = value_type*;
    using const_pointer   = const value_type*;
    using size_type       = size_t;
    using difference_type = std::ptrdiff_t;

    aligned_static_vector()                                            = default;
    aligned_static_vector(const aligned_static_vector&)                = default;
    aligned_static_vector(aligned_static_vector&&) noexcept            = default;
    aligned_static_vector& operator=(const aligned_static_vector&)     = default;
    aligned_static_vector& operator=(aligned_static_vector&&) noexcept = default;

    aligned_static_vector(size_t _n, Tp _v = {});

    aligned_static_vector& operator=(std::initializer_list<Tp>&& _v);
    aligned_static_vector& operator=(std::pair<std::array<Tp, N>, size_t>&&);

    template <typename... Args>
    value_type& emplace_back(Args&&... _v);

    template <typename Up>
    decltype(auto) push_back(Up&& _v)
    {
        return emplace_back(Tp{ std::forward<Up>(_v) });
    }

    void pop_back() { --m_size; }

    void clear();
    void reserve(size_t) noexcept {}
    void shrink_to_fit() noexcept {}
    auto capacity() noexcept { return N; }

    size_t size() const { return m_size; }
    bool   empty() const { return (size() == 0); }

    reference       operator[](size_t _idx) { return m_data[_idx].value; }
    const_reference operator[](size_t _idx) const { return m_data[_idx].value; }

    reference       at(size_t _idx) { return m_data.at(_idx).value; }
    const_reference at(size_t _idx) const { return m_data.at(_idx).value; }

    reference       front() { return m_data.front().value; }
    const_reference front() const { return m_data.front().value; }
    reference       back() { return *(m_data.begin() + size() - 1).value; }
    const_reference back() const { return *(m_data.begin() + size() - 1).value; }

    void swap(this_type& _v);

    friend void swap(this_type& _lhs, this_type& _rhs) { _lhs.swap(_rhs); }

    template <typename ContainerT>
    struct iterator_base
    {
        iterator_base(ContainerT* c = nullptr, size_type i = 0)
        : m_container(c)
        , m_index(i)
        {}

        iterator_base& operator+=(size_type i)
        {
            m_index += i;
            return *this;
        }
        iterator_base& operator-=(size_type i)
        {
            m_index -= i;
            return *this;
        }
        iterator_base& operator++()
        {
            ++m_index;
            return *this;
        }
        iterator_base& operator--()
        {
            --m_index;
            return *this;
        }

        difference_type operator-(const iterator_base& itr)
        {
            assert(m_container == itr.m_container);
            return m_index - itr.m_index;
        }

        bool operator<(const iterator_base& itr) const
        {
            assert(m_container == itr.m_container);
            return m_index < itr.m_index;
        }
        bool operator==(const iterator_base& itr) const
        {
            return m_container == itr.m_container && m_index == itr.m_index;
        }

    protected:
        ContainerT* m_container;
        size_type   m_index;
    };

public:
    struct const_iterator;

    struct iterator
    : public iterator_base<this_type>
    , public random_access_iterator_helper<iterator, value_type>
    {
        using iterator_base<this_type>::iterator_base;
        friend struct const_iterator;

        reference operator*() { return (*this->m_container)[this->m_index]; }
    };

    struct const_iterator
    : public iterator_base<const_this_type>
    , public random_access_iterator_helper<const_iterator, const value_type>
    {
        using iterator_base<const_this_type>::iterator_base;

        const_iterator(const iterator& itr)
        : iterator_base<const_this_type>(itr.m_container, itr.m_index)
        {}

        const_reference operator*() const { return (*this->m_container)[this->m_index]; }

        bool operator==(const const_iterator& itr) const
        {
            return iterator_base<const_this_type>::operator==(itr);
        }

        friend bool operator==(const iterator& l, const const_iterator& r)
        {
            return r == l;
        }
    };

    iterator       begin() noexcept { return { this, 0 }; }
    const_iterator begin() const noexcept { return { this, 0 }; }
    const_iterator cbegin() const noexcept { return begin(); }

    iterator       end() noexcept { return { this, size() }; }
    const_iterator end() const noexcept { return { this, size() }; }
    const_iterator cend() const noexcept { return end(); }

private:
    count_type m_size = count_type{ 0 };
    array_type m_data = {};
};

template <typename Tp, size_t N, size_t AlignN, bool AtomicSizeV>
aligned_static_vector<Tp, N, AlignN, AtomicSizeV>::aligned_static_vector(size_t _n, Tp _v)
{
    m_data.fill(_v);
    if constexpr(AtomicSizeV)
        m_size.store(_n);
    else
        m_size = _n;
}

template <typename Tp, size_t N, size_t AlignN, bool AtomicSizeV>
aligned_static_vector<Tp, N, AlignN, AtomicSizeV>&
aligned_static_vector<Tp, N, AlignN, AtomicSizeV>::operator=(
    std::initializer_list<Tp>&& _v)
{
    if(ROCPROFSYS_UNLIKELY(_v.size() > N))
    {
        throw exception<std::out_of_range>(
            std::string{ "aligned_static_vector::operator=(initializer_list) size > " } +
            std::to_string(N));
    }

    clear();
    for(auto&& itr : _v)
        m_data[m_size++] = itr;
    return *this;
}

template <typename Tp, size_t N, size_t AlignN, bool AtomicSizeV>
aligned_static_vector<Tp, N, AlignN, AtomicSizeV>&
aligned_static_vector<Tp, N, AlignN, AtomicSizeV>::operator=(
    std::pair<std::array<Tp, N>, size_t>&& _v)
{
    if constexpr(AtomicSizeV) m_size.store(0);

    for(size_t i = 0; i < N; ++i)
        m_data[i].value = std::move(_v.first[i]);

    if constexpr(AtomicSizeV)
        m_size.store(_v.second);
    else
        m_size = _v.second;

    return *this;
}

template <typename Tp, size_t N, size_t AlignN, bool AtomicSizeV>
void
aligned_static_vector<Tp, N, AlignN, AtomicSizeV>::clear()
{
    if constexpr(AtomicSizeV)
        m_size.store(0);
    else
        m_size = 0;
}

template <typename Tp, size_t N, size_t AlignN, bool AtomicSizeV>
void
aligned_static_vector<Tp, N, AlignN, AtomicSizeV>::swap(this_type& _v)
{
    if constexpr(AtomicSizeV)
    {
        auto _t_size = m_size;
        auto _v_size = _v.m_size;
        std::swap(m_data, _v.m_data);
        m_size.store(_v_size);
        _v.m_size.store(_t_size);
    }
    else
    {
        std::swap(m_size, _v.m_size);
        std::swap(m_data, _v.m_data);
    }
}

template <typename Tp, size_t N, size_t AlignN, bool AtomicSizeV>
template <typename... Args>
Tp&
aligned_static_vector<Tp, N, AlignN, AtomicSizeV>::emplace_back(Args&&... _v)
{
    auto _idx = m_size++;
    if(_idx >= N)
    {
        throw exception<std::out_of_range>(
            std::string{ "aligned_static_vector::emplace_back - reached capacity " } +
            std::to_string(N));
    }

    if constexpr(sizeof...(Args) > 0)
    {
        if constexpr(std::is_assignable<Tp, decltype(std::forward<Args>(_v))...>::value)
            m_data[_idx].value = { std::forward<Args>(_v)... };
        else if constexpr(std::is_constructible<Tp, decltype(std::forward<Args>(
                                                        _v))...>::value)
            m_data[_idx].value = Tp{ std::forward<Args>(_v)... };
        else
            static_assert(
                sizeof...(Args) == 0,
                "Error! Tp is not assignable or constructible with provided args");
    }
    else
    {
        // _v... expands to nothing but is used to suppress unused variable warnings
        m_data[_idx].value = { _v... };
    }

    return m_data[_idx].value;
}

}  // namespace container
}  // namespace rocprofsys
