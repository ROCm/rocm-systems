// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <stdexcept>
#include <string>
#include <typeinfo>

namespace rocprofsys::container
{
template <typename Tp>
struct c_array
{
    // Construct an array wrapper from a base pointer and array size
    constexpr c_array(Tp* base, size_t size)
    : m_base{ base }
    , m_size{ size }
    {}

    ~c_array()                             = default;
    c_array(const c_array&)                = default;
    c_array& operator=(const c_array&)     = default;
    c_array& operator=(c_array&&) noexcept = default;

    // Get the size of the wrapped array
    [[nodiscard]] constexpr size_t size() const { return m_size; }

    // Access an element by index
    constexpr Tp& operator[](size_t idx) { return m_base[idx]; }

    // Access an element by index
    constexpr const Tp& operator[](size_t idx) const { return m_base[idx]; }

    // Access an element by index with bounds check
    constexpr Tp& at(size_t idx)
    {
        if(idx < m_size)
        {
            return m_base[idx];
        }
        throw std::out_of_range{ std::string{ typeid(*this).name() } +
                                 std::to_string(idx) + " exceeds size " +
                                 std::to_string(m_size) };
    }

    // Access an element by index with bounds check
    constexpr const Tp& at(size_t idx) const
    {
        if(idx < m_size)
        {
            return m_base[idx];
        }
        throw std::out_of_range{ std::string{ typeid(*this).name() } +
                                 std::to_string(idx) + " exceeds size " +
                                 std::to_string(m_size) };
    }

    // Get a slice of this array, from a start index (inclusive) to end index (exclusive)
    constexpr c_array<Tp> slice(size_t start, size_t end)
    {
        return c_array<Tp>(&m_base[start], end - start);
    }

    constexpr void pop_front()
    {
        ++m_base;
        --m_size;
    }

    constexpr void pop_back() { --m_size; }

    constexpr operator Tp*() const { return m_base; }

    // Iterator class for convenient range-based for loop support
    template <typename Up>
    struct iterator
    {
        // Start the iterator at a given pointer
        constexpr iterator(Tp* ptr)
        : m_ptr{ ptr }
        {}

        // Advance to the next element
        constexpr void operator++() { ++m_ptr; }
        constexpr void operator++(int) { m_ptr++; }

        // Get the current element
        constexpr Up& operator*() const { return *m_ptr; }

        // Compare iterators
        constexpr bool operator==(const iterator& rhs) const
        {
            return m_ptr == rhs.m_ptr;
        }
        constexpr bool operator!=(const iterator& rhs) const
        {
            return m_ptr != rhs.m_ptr;
        }

    private:
        Tp* m_ptr = nullptr;
    };

    // Get an iterator positioned at the beginning of the wrapped array
    constexpr iterator<Tp>       begin() { return iterator<Tp>{ m_base }; }
    constexpr iterator<const Tp> begin() const { return iterator<const Tp>{ m_base }; }

    // Get an iterator positioned at the end of the wrapped array
    constexpr iterator<Tp>       end() { return iterator<Tp>{ &m_base[m_size] }; }
    constexpr iterator<const Tp> end() const
    {
        return iterator<const Tp>{ &m_base[m_size] };
    }

private:
    Tp*    m_base = nullptr;
    size_t m_size = 0;
};

// Function for automatic template argument deduction
template <typename Tp>
constexpr c_array<Tp>
wrap_c_array(Tp* base, size_t size)
{
    return c_array<Tp>(base, size);
}
}  // namespace rocprofsys::container
