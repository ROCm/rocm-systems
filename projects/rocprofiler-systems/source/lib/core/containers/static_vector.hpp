// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "core/containers/c_array.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdlib>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

namespace rocprofsys::container
{
template <typename Tp, size_t N, bool AtomicSizeV = false>
struct static_vector
{
    using count_type      = std::conditional_t<AtomicSizeV, std::atomic<size_t>, size_t>;
    using this_type       = static_vector<Tp, N, AtomicSizeV>;
    using value_type      = Tp;
    using array_type      = std::array<Tp, N>;
    using reference       = value_type&;
    using const_reference = const value_type&;
    using pointer         = value_type*;
    using const_pointer   = const value_type*;
    using size_type       = size_t;
    using difference_type = std::ptrdiff_t;

    constexpr static_vector()                                    = default;
    constexpr static_vector(const static_vector&)                = default;
    constexpr static_vector(static_vector&&) noexcept            = default;
    constexpr static_vector& operator=(const static_vector&)     = default;
    constexpr static_vector& operator=(static_vector&&) noexcept = default;

    constexpr static_vector(size_t count, Tp value = {});
    explicit constexpr static_vector(c_array<Tp>&&);

    template <size_t M>
    explicit constexpr static_vector(std::array<Tp, M>&&);

    constexpr static_vector& operator=(std::initializer_list<Tp>&& values);
    constexpr static_vector& operator=(std::pair<std::array<Tp, N>, size_t>&&);

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

    constexpr auto begin() { return m_data.begin(); }
    constexpr auto begin() const { return m_data.begin(); }
    constexpr auto cbegin() const { return m_data.cbegin(); }

    constexpr auto end() { return m_data.begin() + size(); }
    constexpr auto end() const { return m_data.begin() + size(); }
    constexpr auto cend() const { return m_data.cbegin() + size(); }

    constexpr decltype(auto) operator[](size_t idx) { return m_data[idx]; }
    constexpr decltype(auto) operator[](size_t idx) const { return m_data[idx]; }

    constexpr decltype(auto) at(size_t idx) { return m_data.at(idx); }
    constexpr decltype(auto) at(size_t idx) const { return m_data.at(idx); }

    constexpr decltype(auto) front() { return m_data.front(); }
    constexpr decltype(auto) front() const { return m_data.front(); }
    constexpr decltype(auto) back() { return *(m_data.begin() + size() - 1); }
    constexpr decltype(auto) back() const { return *(m_data.begin() + size() - 1); }

    constexpr auto*       data() { return m_data.data(); }
    constexpr const auto* data() const { return m_data.data(); }

    constexpr void swap(this_type& other);

    friend constexpr void swap(this_type& lhs, this_type& rhs) { lhs.swap(rhs); }

private:
    constexpr void update_size(size_t);

private:
    count_type        m_size = count_type{ 0 };
    std::array<Tp, N> m_data = {};
};

template <typename Tp, size_t N, bool AtomicSizeV>
constexpr static_vector<Tp, N, AtomicSizeV>::static_vector(size_t count, Tp value)
{
    m_data.fill(value);
    update_size(count);
}

template <typename Tp, size_t N, bool AtomicSizeV>
constexpr static_vector<Tp, N, AtomicSizeV>::static_vector(c_array<Tp>&& array)
{
    auto count = std::min<size_t>(N, array.size());
    for(size_t i = 0; i < count; ++i, ++m_size)
    {
        m_data[i] = array[i];
    }
}

template <typename Tp, size_t N, bool AtomicSizeV>
template <size_t M>
constexpr static_vector<Tp, N, AtomicSizeV>::static_vector(std::array<Tp, M>&& arr)
{
    auto count = std::min<size_t>(N, M);
    for(size_t i = 0; i < count; ++i, ++m_size)
    {
        m_data[i] = arr[i];
    }
}

template <typename Tp, size_t N, bool AtomicSizeV>
constexpr static_vector<Tp, N, AtomicSizeV>&
static_vector<Tp, N, AtomicSizeV>::operator=(std::initializer_list<Tp>&& values)
{
    if(values.size() > N) [[unlikely]]
    {
        throw std::out_of_range{
            std::string{ "static_vector::operator=(initializer_list) size > " } +
            std::to_string(N)
        };
    }

    clear();
    for(auto&& itr : values)
    {
        m_data[m_size++] = itr;
    }
    return *this;
}

template <typename Tp, size_t N, bool AtomicSizeV>
constexpr static_vector<Tp, N, AtomicSizeV>&
static_vector<Tp, N, AtomicSizeV>::operator=(
    std::pair<std::array<Tp, N>, size_t>&& src_pair)
{
    update_size(0);
    m_data = std::move(src_pair.first);
    update_size(src_pair.second);

    return *this;
}

template <typename Tp, size_t N, bool AtomicSizeV>
constexpr void
static_vector<Tp, N, AtomicSizeV>::clear()
{
    update_size(0);
}

template <typename Tp, size_t N, bool AtomicSizeV>
constexpr void
static_vector<Tp, N, AtomicSizeV>::swap(this_type& other)
{
    if constexpr(AtomicSizeV)
    {
        auto self_size  = m_size;
        auto other_size = other.m_size;
        std::swap(m_data, other.m_data);
        update_size(other_size);
        other.update_size(self_size);
    }
    else
    {
        std::swap(m_size, other.m_size);
        std::swap(m_data, other.m_data);
    }
}

template <typename Tp, size_t N, bool AtomicSizeV>
template <typename... Args>
constexpr Tp&
static_vector<Tp, N, AtomicSizeV>::emplace_back(Args&&... args)
{
    const auto idx = static_cast<size_t>(m_size);
    if(idx >= N) [[unlikely]]
    {
        throw std::out_of_range{ std::string{
                                     "static_vector::emplace_back - reached capacity " } +
                                 std::to_string(N) };
    }
    update_size(idx + 1);

    if constexpr(std::is_assignable<Tp, decltype(std::forward<Args>(args))...>::value)
    {
        m_data[idx] = { std::forward<Args>(args)... };
    }
    else
    {
        m_data[idx] = Tp{ std::forward<Args>(args)... };
    }
    return m_data[idx];
}

template <typename Tp, size_t N, bool AtomicSizeV>
constexpr void
static_vector<Tp, N, AtomicSizeV>::update_size(size_t count)
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
