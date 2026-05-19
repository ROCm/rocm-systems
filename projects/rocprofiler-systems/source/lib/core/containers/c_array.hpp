// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "core/exception.hpp"

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace rocprofsys
{
namespace container
{
template <typename Tp>
struct c_array
{
    // Construct an array wrapper from a base pointer and array size
    c_array(Tp* _base, size_t _size)
    : m_base{ _base }
    , m_size{ _size }
    {}

    ~c_array()                             = default;
    c_array(const c_array&)                = default;
    c_array& operator=(const c_array&)     = default;
    c_array& operator=(c_array&&) noexcept = default;

    // Get the size of the wrapped array
    size_t size() const { return m_size; }

    // Access an element by index
    Tp& operator[](size_t i) { return m_base[i]; }

    // Access an element by index
    const Tp& operator[](size_t i) const { return m_base[i]; }

    // Access an element by index with bounds check
    Tp& at(size_t i)
    {
        if(i < m_size) return m_base[i];
        throw ::rocprofsys::exception<std::out_of_range>(
            std::string{ typeid(*this).name() } + std::to_string(i) + " exceeds size " +
            std::to_string(m_size));
    }

    // Access an element by index with bounds check
    const Tp& at(size_t i) const
    {
        if(i < m_size) return m_base[i];
        throw ::rocprofsys::exception<std::out_of_range>(
            std::string{ typeid(*this).name() } + std::to_string(i) + " exceeds size " +
            std::to_string(m_size));
    }

    // Get a slice of this array, from a start index (inclusive) to end index (exclusive)
    c_array<Tp> slice(size_t start, size_t end)
    {
        return c_array<Tp>(&m_base[start], end - start);
    }

    void pop_front()
    {
        ++m_base;
        --m_size;
    }

    void pop_back() { --m_size; }

    operator Tp*() const { return m_base; }

    // Iterator class for convenient range-based for loop support
    template <typename Up>
    struct iterator
    {
        // Start the iterator at a given pointer
        iterator(Tp* p)
        : m_ptr{ p }
        {}

        // Advance to the next element
        void operator++() { ++m_ptr; }
        void operator++(int) { m_ptr++; }

        // Get the current element
        Up& operator*() const { return *m_ptr; }

        // Compare iterators
        bool operator==(const iterator& rhs) const { return m_ptr == rhs.m_ptr; }
        bool operator!=(const iterator& rhs) const { return m_ptr != rhs.m_ptr; }

    private:
        Tp* m_ptr = nullptr;
    };

    // Get an iterator positioned at the beginning of the wrapped array
    iterator<Tp>       begin() { return iterator<Tp>{ m_base }; }
    iterator<const Tp> begin() const { return iterator<const Tp>{ m_base }; }

    // Get an iterator positioned at the end of the wrapped array
    iterator<Tp>       end() { return iterator<Tp>{ &m_base[m_size] }; }
    iterator<const Tp> end() const { return iterator<const Tp>{ &m_base[m_size] }; }

private:
    Tp*    m_base = nullptr;
    size_t m_size = 0;
};

// Function for automatic template argument deduction
template <typename Tp>
c_array<Tp>
wrap_c_array(Tp* base, size_t size)
{
    return c_array<Tp>(base, size);
}
}  // namespace container
}  // namespace rocprofsys
