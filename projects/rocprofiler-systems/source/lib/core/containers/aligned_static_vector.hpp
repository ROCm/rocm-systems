// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "common/defines.h"
#include "core/containers/operators.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <concepts>
#include <cstddef>
#include <cstdlib>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

namespace rocprofsys::container
{
#ifndef ROCPROFSYS_CACHELINE_SIZE
// Fixed default (not std::hardware_destructive_interference_size): stable constexpr, no
// GCC -Winterference-size. Override with ROCPROFSYS_CACHELINE_SIZE if needed.
#    define ROCPROFSYS_CACHELINE_SIZE 64
#endif

constexpr std::size_t k_cacheline_align =
    std::max<size_t>(ROCPROFSYS_CACHELINE_SIZE, ROCPROFSYS_CACHELINE_SIZE_MIN);

template <typename Tp, size_t N, size_t AlignN = k_cacheline_align,
          bool AtomicSizeV = false>
struct aligned_static_vector
{
    struct aligned_value_type
    {
        alignas(AlignN) Tp value = {};
    };

    using count_type      = std::conditional_t<AtomicSizeV, std::atomic<size_t>, size_t>;
    using this_type       = aligned_static_vector<Tp, N, AlignN, AtomicSizeV>;
    using const_this_type = const aligned_static_vector<Tp, N, AlignN, AtomicSizeV>;
    using value_type      = Tp;
    using array_type      = std::array<aligned_value_type, N>;
    using reference       = value_type&;
    using const_reference = const value_type&;
    using pointer         = value_type*;
    using const_pointer   = const value_type*;
    using size_type       = size_t;
    using difference_type = std::ptrdiff_t;

    constexpr aligned_static_vector() = default;

    // std::atomic is neither copyable nor movable, so the AtomicSizeV=true instantiation
    // cannot use defaulted copy/move special members; read/write it through
    // load()/store() instead while the non-atomic instantiation copies/moves the plain
    // size_t directly. Each member is split into a `requires`-constrained overload with
    // the real body and a `= deleted` overload for the complementary case, so this class
    // stays SFINAE-friendly (e.g. std::is_copy_constructible_v) for non-copyable/movable
    // Tp, matching what a defaulted special member would give.
    constexpr aligned_static_vector(const aligned_static_vector& other)
        requires std::copyable<Tp>
    : m_data(other.m_data)
    {
        if constexpr(AtomicSizeV)
        {
            m_size.store(other.m_size.load());
        }
        else
        {
            m_size = other.m_size;
        }
    }

    constexpr aligned_static_vector(const aligned_static_vector&)
        requires(!std::copyable<Tp>)
    = delete;

    constexpr aligned_static_vector(aligned_static_vector&& other) noexcept
        requires std::movable<Tp>
    : m_data(std::move(other.m_data))
    {
        if constexpr(AtomicSizeV)
        {
            m_size.store(other.m_size.load());
        }
        else
        {
            m_size = other.m_size;
        }
    }

    constexpr aligned_static_vector(aligned_static_vector&&) noexcept
        requires(!std::movable<Tp>)
    = delete;

    constexpr aligned_static_vector& operator=(const aligned_static_vector& other)
        requires std::copyable<Tp>
    {
        if(this == &other)
        {
            return *this;
        }
        m_data = other.m_data;
        if constexpr(AtomicSizeV)
        {
            m_size.store(other.m_size.load());
        }
        else
        {
            m_size = other.m_size;
        }
        return *this;
    }

    constexpr aligned_static_vector& operator=(const aligned_static_vector&)
        requires(!std::copyable<Tp>)
    = delete;

    constexpr aligned_static_vector& operator=(aligned_static_vector&& other) noexcept
        requires std::movable<Tp>
    {
        if(this == &other)
        {
            return *this;
        }
        m_data = std::move(other.m_data);
        if constexpr(AtomicSizeV)
        {
            m_size.store(other.m_size.load());
        }
        else
        {
            m_size = other.m_size;
        }
        return *this;
    }

    constexpr aligned_static_vector& operator=(aligned_static_vector&&) noexcept
        requires(!std::movable<Tp>)
    = delete;

    explicit constexpr aligned_static_vector(size_t count, Tp value = {});

    constexpr aligned_static_vector& operator=(std::initializer_list<Tp>&& values);
    constexpr aligned_static_vector& operator=(std::pair<std::array<Tp, N>, size_t>&&);

    template <typename... Args>
    constexpr value_type& emplace_back(Args&&... args);

    template <typename Up>
    constexpr decltype(auto) push_back(Up&& value)
    {
        return emplace_back(Tp{ std::forward<Up>(value) });
    }

    constexpr void pop_back() { --m_size; }

    constexpr void clear();
    constexpr void reserve(size_t) noexcept {}
    constexpr void shrink_to_fit() noexcept {}
    constexpr auto capacity() noexcept { return N; }

    [[nodiscard]] constexpr size_t size() const { return m_size; }
    [[nodiscard]] constexpr bool   empty() const { return (size() == 0); }

    constexpr reference       operator[](size_t idx) { return m_data[idx].value; }
    constexpr const_reference operator[](size_t idx) const { return m_data[idx].value; }

    constexpr reference       at(size_t idx) { return m_data.at(idx).value; }
    constexpr const_reference at(size_t idx) const { return m_data.at(idx).value; }

    constexpr reference       front() { return m_data.front().value; }
    constexpr const_reference front() const { return m_data.front().value; }
    constexpr reference       back() { return (*(m_data.begin() + size() - 1)).value; }
    constexpr const_reference back() const
    {
        return (*(m_data.begin() + size() - 1)).value;
    }

    constexpr void swap(this_type& other);

    friend constexpr void swap(this_type& lhs, this_type& rhs) noexcept { lhs.swap(rhs); }

    template <typename ContainerT>
    struct iterator_base
    {
        explicit constexpr iterator_base(ContainerT* container = nullptr,
                                         size_type   index     = 0)
        : m_container(container)
        , m_index(index)
        {}

        constexpr iterator_base& operator+=(size_type count)
        {
            m_index += count;
            return *this;
        }
        constexpr iterator_base& operator-=(size_type count)
        {
            m_index -= count;
            return *this;
        }
        constexpr iterator_base& operator++()
        {
            ++m_index;
            return *this;
        }
        constexpr iterator_base& operator--()
        {
            --m_index;
            return *this;
        }

        constexpr difference_type operator-(const iterator_base& itr)
        {
            assert(m_container == itr.m_container);
            return m_index - itr.m_index;
        }

        constexpr bool operator<(const iterator_base& itr) const
        {
            assert(m_container == itr.m_container);
            return m_index < itr.m_index;
        }
        constexpr bool operator==(const iterator_base& itr) const
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

        constexpr reference operator*() { return (*this->m_container)[this->m_index]; }
    };

    struct const_iterator
    : public iterator_base<const_this_type>
    , public random_access_iterator_helper<const_iterator, const value_type>
    {
        using iterator_base<const_this_type>::iterator_base;

        explicit constexpr const_iterator(const iterator& itr)
        : iterator_base<const_this_type>(itr.m_container, itr.m_index)
        {}

        constexpr const_reference operator*() const
        {
            return (*this->m_container)[this->m_index];
        }

        constexpr bool operator==(const const_iterator& itr) const
        {
            return iterator_base<const_this_type>::operator==(itr);
        }

        friend constexpr bool operator==(const iterator& lhs, const const_iterator& rhs)
        {
            return rhs == lhs;
        }
    };

    constexpr iterator       begin() noexcept { return iterator{ this, 0 }; }
    constexpr const_iterator begin() const noexcept { return const_iterator{ this, 0 }; }
    constexpr const_iterator cbegin() const noexcept { return begin(); }

    constexpr iterator       end() noexcept { return iterator{ this, size() }; }
    constexpr const_iterator end() const noexcept
    {
        return const_iterator{ this, size() };
    }
    constexpr const_iterator cend() const noexcept { return end(); }

private:
    constexpr void update_size(size_t);

private:
    count_type m_size = count_type{ 0 };
    array_type m_data = {};
};

template <typename Tp, size_t N, size_t AlignN, bool AtomicSizeV>
constexpr aligned_static_vector<Tp, N, AlignN, AtomicSizeV>::aligned_static_vector(
    size_t count, Tp value)
{
    m_data.fill(aligned_value_type{ value });
    if constexpr(AtomicSizeV)
    {
        m_size.store(count);
    }
    else
    {
        m_size = count;
    }
}

template <typename Tp, size_t N, size_t AlignN, bool AtomicSizeV>
constexpr aligned_static_vector<Tp, N, AlignN, AtomicSizeV>&
aligned_static_vector<Tp, N, AlignN, AtomicSizeV>::operator=(
    std::initializer_list<Tp>&& values)
{
    if(values.size() > N) [[unlikely]]
    {
        throw std::out_of_range{
            std::string{ "aligned_static_vector::operator=(initializer_list) size > " } +
            std::to_string(N)
        };
    }

    clear();
    for(auto&& itr : values)
    {
        m_data[m_size++].value = itr;
    }
    return *this;
}

template <typename Tp, size_t N, size_t AlignN, bool AtomicSizeV>
constexpr aligned_static_vector<Tp, N, AlignN, AtomicSizeV>&
aligned_static_vector<Tp, N, AlignN, AtomicSizeV>::operator=(
    std::pair<std::array<Tp, N>, size_t>&& src_pair)
{
    if constexpr(AtomicSizeV)
    {
        m_size.store(0);
    }

    for(size_t i = 0; i < N; ++i)
    {
        m_data[i].value = std::move(src_pair.first[i]);
    }

    if constexpr(AtomicSizeV)
    {
        m_size.store(src_pair.second);
    }
    else
    {
        m_size = src_pair.second;
    }

    return *this;
}

template <typename Tp, size_t N, size_t AlignN, bool AtomicSizeV>
constexpr void
aligned_static_vector<Tp, N, AlignN, AtomicSizeV>::clear()
{
    if constexpr(AtomicSizeV)
    {
        m_size.store(0);
    }
    else
    {
        m_size = 0;
    }
}

template <typename Tp, size_t N, size_t AlignN, bool AtomicSizeV>
constexpr void
aligned_static_vector<Tp, N, AlignN, AtomicSizeV>::swap(this_type& other)
{
    if constexpr(AtomicSizeV)
    {
        auto self_size  = m_size;
        auto other_size = other.m_size;
        std::swap(m_data, other.m_data);
        m_size.store(other_size);
        other.m_size.store(self_size);
    }
    else
    {
        std::swap(m_size, other.m_size);
        std::swap(m_data, other.m_data);
    }
}

template <typename Tp, size_t N, size_t AlignN, bool AtomicSizeV>
template <typename... Args>
constexpr Tp&
aligned_static_vector<Tp, N, AlignN, AtomicSizeV>::emplace_back(Args&&... args)
{
    const auto idx = static_cast<size_t>(m_size);
    if(idx >= N) [[unlikely]]
    {
        throw std::out_of_range{
            std::string{ "aligned_static_vector::emplace_back - reached capacity " } +
            std::to_string(N)
        };
    }
    update_size(idx + 1);

    if constexpr(sizeof...(Args) > 0)
    {
        if constexpr(std::is_assignable_v<Tp, decltype(std::forward<Args>(args))...>)
        {
            m_data[idx].value = { std::forward<Args>(args)... };
        }
        else if constexpr(std::is_constructible_v<Tp,
                                                  decltype(std::forward<Args>(args))...>)
        {
            m_data[idx].value = Tp{ std::forward<Args>(args)... };
        }
        else
        {
            static_assert(
                sizeof...(Args) == 0,
                "Error! Tp is not assignable or constructible with provided args");
        }
    }
    else
    {
        // args... expands to nothing but is used to suppress unused variable warnings
        m_data[idx].value = { args... };
    }

    return m_data[idx].value;
}

template <typename Tp, size_t N, size_t AlignN, bool AtomicSizeV>
constexpr void
aligned_static_vector<Tp, N, AlignN, AtomicSizeV>::update_size(size_t count)
{
    if constexpr(AtomicSizeV)
    {
        m_size.store(count);
    }
    else
    {
        m_size = count;
    }
}

}  // namespace rocprofsys::container
