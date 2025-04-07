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

#include "lib/common/container/operators.hpp"
#include "lib/common/container/static_vector.hpp"
#include "lib/common/defines.hpp"

#include <algorithm>
#include <initializer_list>
#include <iterator>
#include <limits>
#include <memory>
#include <numeric>
#include <type_traits>
#include <vector>

namespace rocprofiler
{
namespace common
{
namespace container
{
struct reserve_size
{
    explicit reserve_size(size_t _v)
    : value{_v}
    {}

    size_t value;
};

template <typename Tp, size_t ChunkSizeV = 64>
class stable_vector
{
public:
    using value_type      = Tp;
    using reference       = value_type&;
    using const_reference = const value_type&;
    using pointer         = value_type*;
    using const_pointer   = const value_type*;
    using size_type       = size_t;
    using difference_type = std::ptrdiff_t;

    static constexpr const size_t chunk_size = ChunkSizeV;

private:
    template <size_t N>
    struct is_pow2
    {
        static constexpr bool value = (N & (N - 1)) == 0;
    };

    static_assert(ChunkSizeV > 0, "ChunkSize needs to be greater than zero");
    static_assert(is_pow2<ChunkSizeV>::value, "ChunkSize needs to be a power of 2");

    using this_type       = stable_vector<Tp, ChunkSizeV>;
    using const_this_type = const stable_vector<Tp, ChunkSizeV>;

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

        difference_type operator-(const iterator_base& it)
        {
            assert(m_container == it.m_container);
            return m_index - it.m_index;
        }

        bool operator<(const iterator_base& it) const
        {
            assert(m_container == it.m_container);
            return m_index < it.m_index;
        }
        bool operator==(const iterator_base& it) const
        {
            return m_container == it.m_container && m_index == it.m_index;
        }

    protected:
        ContainerT* m_container;
        size_type   m_index;
    };

public:
    struct const_iterator;

    struct iterator
    : public iterator_base<this_type>
    //, std::iterator<std::random_access_iterator_tag, value_type>
    , public random_access_iterator_helper<iterator, value_type>
    {
        using iterator_base<this_type>::iterator_base;
        friend struct const_iterator;

        reference operator*() { return (*this->m_container)[this->m_index]; }
    };

    struct const_iterator
    : public iterator_base<const_this_type>
    //, std::iterator<std::random_access_iterator_tag, const value_type>
    , public random_access_iterator_helper<const_iterator, const value_type>
    {
        using iterator_base<const_this_type>::iterator_base;

        explicit const_iterator(const iterator& it)
        : iterator_base<const_this_type>(it.m_container, it.m_index)
        {}

        const_reference operator*() const { return (*this->m_container)[this->m_index]; }

        bool operator==(const const_iterator& it) const
        {
            return iterator_base<const_this_type>::operator==(it);
        }

        friend bool operator==(const iterator& l, const const_iterator& r) { return r == l; }
    };

    stable_vector() = default;
    explicit stable_vector(size_type count, const Tp& value);
    explicit stable_vector(size_type count);
    explicit stable_vector(reserve_size&& reserve_count);

    template <typename InputItrT,
              typename = std::enable_if_t<
                  std::is_convertible<typename std::iterator_traits<InputItrT>::iterator_category,
                                      std::input_iterator_tag>::value>>
    stable_vector(InputItrT first, InputItrT last);

    explicit stable_vector(std::initializer_list<Tp>);

    stable_vector(const stable_vector& other);
    stable_vector(stable_vector&& other) noexcept;

    stable_vector& operator=(stable_vector v);

    iterator       begin() noexcept { return {this, 0}; }
    const_iterator begin() const noexcept { return {this, 0}; }
    const_iterator cbegin() const noexcept { return begin(); }

    iterator       end() noexcept { return {this, size()}; }
    const_iterator end() const noexcept { return {this, size()}; }
    const_iterator cend() const noexcept { return end(); }

    size_type size() const noexcept
    {
        return empty() ? 0 : (m_chunks.size() - 1) * ChunkSizeV + m_chunks.back()->size();
    }
    size_type max_size() const noexcept { return std::numeric_limits<size_type>::max(); }
    size_type capacity() const noexcept { return m_chunks.size() * ChunkSizeV; }

    bool empty() const noexcept { return m_chunks.empty(); }

    void reserve(size_type new_capacity);
    void shrink_to_fit() noexcept {}

    bool operator==(const this_type& c) const
    {
        return size() == c.size() && std::equal(cbegin(), cend(), c.cbegin());
    }
    bool operator!=(const this_type& c) const { return !operator==(c); }

    void swap(this_type& v) noexcept { std::swap(m_chunks, v.m_chunks); }

    friend void swap(this_type& l, this_type& r) noexcept { l.swap(r); }

    reference       front() { return m_chunks.front()->front(); }
    const_reference front() const { return front(); }

    reference       back() { return m_chunks.back()->back(); }
    const_reference back() const { return back(); }

    void push_back(const Tp& t);
    void push_back(Tp&& t);

    template <typename... Args>
    reference emplace_back(Args&&... args);

    reference operator[](size_type i);

    const_reference operator[](size_type i) const;

    reference at(size_type i);

    const_reference at(size_type i) const;

private:
    using chunk_type   = container::static_vector<Tp, ChunkSizeV, true>;
    using storage_type = std::vector<std::unique_ptr<chunk_type>>;

    void        add_chunk();
    chunk_type& last_chunk();

    storage_type m_chunks;
};

template <typename Tp, size_t ChunkSizeV>
stable_vector<Tp, ChunkSizeV>::stable_vector(size_type count, const Tp& value)
{
    for(size_type i = 0; i < count; ++i)
    {
        push_back(value);
    }
}

template <typename Tp, size_t ChunkSizeV>
stable_vector<Tp, ChunkSizeV>::stable_vector(size_type count)
{
    for(size_type i = 0; i < count; ++i)
    {
        emplace_back();
    }
}

template <typename Tp, size_t ChunkSizeV>
stable_vector<Tp, ChunkSizeV>::stable_vector(reserve_size&& reserve_count)
{
    reserve(reserve_count.value);
}

template <typename Tp, size_t ChunkSizeV>
template <typename InputItrT, typename>
stable_vector<Tp, ChunkSizeV>::stable_vector(InputItrT first, InputItrT last)
{
    for(; first != last; ++first)
    {
        push_back(*first);
    }
}

template <typename Tp, size_t ChunkSizeV>
stable_vector<Tp, ChunkSizeV>::stable_vector(const stable_vector& other)
{
    for(const auto& chunk : other.m_chunks)
    {
        m_chunks.emplace_back(std::make_unique<chunk_type>(*chunk));
    }
}

template <typename Tp, size_t ChunkSizeV>
stable_vector<Tp, ChunkSizeV>::stable_vector(stable_vector&& other) noexcept
: m_chunks(std::move(other.m_chunks))
{}

template <typename Tp, size_t ChunkSizeV>
stable_vector<Tp, ChunkSizeV>::stable_vector(std::initializer_list<Tp> ilist)
{
    for(const auto& t : ilist)
    {
        push_back(t);
    }
}

template <typename Tp, size_t ChunkSizeV>
stable_vector<Tp, ChunkSizeV>&
stable_vector<Tp, ChunkSizeV>::operator=(stable_vector v)
{
    swap(v);
    return *this;
}

template <typename Tp, size_t ChunkSizeV>
void
stable_vector<Tp, ChunkSizeV>::add_chunk()
{
    m_chunks.emplace_back(std::make_unique<chunk_type>());
}

template <typename Tp, size_t ChunkSizeV>
typename stable_vector<Tp, ChunkSizeV>::chunk_type&
stable_vector<Tp, ChunkSizeV>::last_chunk()
{
    if(ROCPROFILER_UNLIKELY(m_chunks.empty() || m_chunks.back()->size() == ChunkSizeV))
    {
        add_chunk();
    }

    return *m_chunks.back();
}

template <typename Tp, size_t ChunkSizeV>
void
stable_vector<Tp, ChunkSizeV>::reserve(size_type new_capacity)
{
    const size_t initial_capacity = capacity();
    for(difference_type i = new_capacity - initial_capacity; i > 0; i -= ChunkSizeV)
    {
        add_chunk();
    }
}

template <typename Tp, size_t ChunkSizeV>
void
stable_vector<Tp, ChunkSizeV>::push_back(const Tp& t)
{
    last_chunk().push_back(t);
}

template <typename Tp, size_t ChunkSizeV>
void
stable_vector<Tp, ChunkSizeV>::push_back(Tp&& t)
{
    last_chunk().push_back(std::move(t));
}

template <typename Tp, size_t ChunkSizeV>
template <typename... Args>
typename stable_vector<Tp, ChunkSizeV>::reference
stable_vector<Tp, ChunkSizeV>::emplace_back(Args&&... args)
{
    return last_chunk().emplace_back(std::forward<Args>(args)...);
}

template <typename Tp, size_t ChunkSizeV>
typename stable_vector<Tp, ChunkSizeV>::reference
stable_vector<Tp, ChunkSizeV>::operator[](size_type i)
{
    return (*m_chunks[i / ChunkSizeV])[i % ChunkSizeV];
}

template <typename Tp, size_t ChunkSizeV>
typename stable_vector<Tp, ChunkSizeV>::const_reference
stable_vector<Tp, ChunkSizeV>::operator[](size_type i) const
{
    return const_cast<this_type&>(*this)[i];
}

template <typename Tp, size_t ChunkSizeV>
typename stable_vector<Tp, ChunkSizeV>::reference
stable_vector<Tp, ChunkSizeV>::at(size_type i)
{
    if(ROCPROFILER_UNLIKELY(i >= size()))
    {
        throw std::out_of_range("stable_vector::at(" + std::to_string(i) + "). size is " +
                                std::to_string(size()));
    }

    return operator[](i);
}

template <typename Tp, size_t ChunkSizeV>
typename stable_vector<Tp, ChunkSizeV>::const_reference
stable_vector<Tp, ChunkSizeV>::at(size_type i) const
{
    return const_cast<this_type&>(*this).at(i);
}

template <typename Tp, size_t ChunkSizeV, typename... Args>
auto
resize(stable_vector<Tp, ChunkSizeV>& _v, size_t _n, Args&&... args)
{
    if(_n > _v.capacity()) _v.reserve(_n);

    while(_v.size() < _n)
        _v.emplace_back(std::forward<Args>(args)...);

    return _v.size();
}
}  // namespace container
}  // namespace common
}  // namespace rocprofiler
