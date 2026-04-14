// MIT License
//
// Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
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

/**
 * @file small_vector.hpp
 * @brief This is inspired and largely derived from llvm/ADT/SmallVector.h. It provides a STL-like
 * vector class which uses a small allocation on the stack when the number of elements is small.
 *
 * This container is ideal for vectors which are allocated frequently, will more than likely only
 * contain a few elements, and are allocated in places where performance is a concern. When the
 * number of elements is small, storing these elements will not require a heap allocation but it can
 * also grow to accommodate larger allocation needs. In other words, it effectively has memory
 * allocation like std::array<T, N> until the number of elements exceeds N. Once the number of
 * elements exceeds N, it turns into std::vector<T>.
 *
 * Reference: https://github.com/llvm/llvm-project/blob/main/llvm/include/llvm/ADT/SmallVector.h
 */

#include "lib/common/defines.hpp"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <initializer_list>
#include <iterator>
#include <limits>
#include <memory>
#include <new>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace rocprofiler
{
namespace common
{
namespace container
{
template <typename T>
class array_ref;

template <typename iteratorT>
class iterator_range;

template <typename IteratorT>
using enable_if_convertible_to_input_iterator = std::enable_if_t<
    std::is_convertible<typename std::iterator_traits<IteratorT>::iterator_category,
                        std::input_iterator_tag>::value>;

/// this is all the stuff common to all small_vectors.
///
/// the template parameter specifies the type which should be used to hold the
/// size and capacity of the small_vector, so it can be adjusted.
/// using 32 bit size is desirable to shrink the size of the small_vector.
/// using 64 bit size is desirable for cases like small_vector<char>, where a
/// 32 bit size would limit the vector to ~4GB. small_vectors are used for
/// buffering bitcode output - which can exceed 4GB.
template <typename SizeT>
class small_vector_base
{
protected:
    void* m_begin_x  = nullptr;
    SizeT m_size     = 0;
    SizeT m_capacity = 0;

    /// the maximum value of the SizeT used.
    static constexpr size_t size_type_max() { return std::numeric_limits<SizeT>::max(); }

    small_vector_base(void* first_el, size_t total_capacity)
    : m_begin_x(first_el)
    , m_capacity(total_capacity)
    {}

    /// this is a helper for \a grow() that's out of line to reduce code
    /// duplication.  this function will report a fatal error if it can't grow at
    /// least to \p min_size.
    void* malloc_for_grow(void* first_el, size_t min_size, size_t t_size, size_t& new_capacity);

    /// this is an implementation of the grow() method which only works
    /// on POD-like data types and is out of line to reduce code duplication.
    /// this function will report a fatal error if it cannot increase capacity.
    void grow_pod(void* first_el, size_t min_size, size_t t_size);

    /// if vector was first created with capacity 0, get_first_el() points to the
    /// memory right after, an area unallocated. if a subsequent allocation,
    /// that grows the vector, happens to return the same pointer as get_first_el(),
    /// get a new allocation, otherwise is_small() will falsely return that no
    /// allocation was done (true) and the memory will not be freed in the
    /// destructor. if a v_size is given (vector size), also copy that many
    /// elements to the new allocation - used if realloca fails to increase
    /// space, and happens to allocate precisely at begin_x.
    /// this is unlikely to be called often, but resolves a memory leak when the
    /// situation does occur.
    void* replace_allocation(void* new_elts, size_t t_size, size_t new_capacity, size_t v_size = 0);

public:
    small_vector_base() = delete;

    size_t size() const { return m_size; }
    size_t capacity() const { return m_capacity; }

    [[nodiscard]] bool empty() const { return !m_size; }

protected:
    /// set the array size to \p N, which the current array must have enough
    /// capacity for.
    ///
    /// this does not construct or destroy any elements in the vector.
    void set_size(size_t N)
    {
        assert(N <= capacity());
        m_size = N;
    }
};

template <typename T>
using small_vector_size_type =
    std::conditional_t<sizeof(T) < 4 && sizeof(void*) >= 8, uint64_t, uint32_t>;

/// figure out the offset of the first element.
template <typename T, typename = void>
struct small_vector_alignment_and_size
{
    alignas(small_vector_base<small_vector_size_type<T>>) char base[sizeof(
        small_vector_base<small_vector_size_type<T>>)];
    alignas(T) char first_el[sizeof(T)];
};

/// this is the part of small_vector_template_base which does not depend on whether
/// the type T is a POD. the extra dummy template argument is used by array_ref
/// to avoid unnecessarily requiring T to be complete.
template <typename T, typename = void>
class small_vector_template_common : public small_vector_base<small_vector_size_type<T>>
{
    using base = small_vector_base<small_vector_size_type<T>>;

protected:
    /// find the address of the first element.  for this pointer math to be valid
    /// with small-size of 0 for T with lots of alignment, it's important that
    /// small_vector_storage is properly-aligned even for small-size of 0.
    void* get_first_el() const
    {
        return const_cast<void*>(
            reinterpret_cast<const void*>(reinterpret_cast<const char*>(this) +
                                          offsetof(small_vector_alignment_and_size<T>, first_el)));
    }
    // space after 'First_el' is clobbered, do not add any instance vars after it.

    small_vector_template_common(size_t _size)
    : base(get_first_el(), _size)
    {}

    void grow_pod(size_t min_size, size_t t_size)
    {
        base::grow_pod(get_first_el(), min_size, t_size);
    }

    /// return true if this is a smallvector which has not had dynamic
    /// memory allocated for it.
    bool is_small() const { return this->m_begin_x == get_first_el(); }

    /// put this vector in a state of being small.
    void reset_to_small()
    {
        this->m_begin_x = get_first_el();
        this->m_size = this->m_capacity = 0;  // FIXME: setting capacity to 0 is suspect.
    }

    /// return true if V is an internal reference to the given range.
    bool is_reference_to_range(const void* V, const void* first, const void* last) const
    {
        // use std::less to avoid UB.
        std::less<> less_than;
        return !less_than(V, first) && less_than(V, last);
    }

    /// return true if V is an internal reference to this vector.
    bool is_reference_to_storage(const void* V) const
    {
        return is_reference_to_range(V, this->begin(), this->end());
    }

    /// return true if first and last form a valid (possibly empty) range in this
    /// vector's storage.
    bool is_range_in_storage(const void* first, const void* last) const
    {
        // use std::less to avoid UB.
        std::less<> less_than;
        return !less_than(first, this->begin()) && !less_than(last, first) &&
               !less_than(this->end(), last);
    }

    /// return true unless elt will be invalidated by resizing the vector to
    /// new_size.
    bool is_safe_to_reference_after_resize(const void* elt, size_t new_size)
    {
        // past the end.
        if(ROCPROFILER_LIKELY(!is_reference_to_storage(elt))) return true;

        // return false if elt will be destroyed by shrinking.
        if(new_size <= this->size()) return elt < this->begin() + new_size;

        // return false if we need to grow.
        return new_size <= this->capacity();
    }

    /// check whether elt will be invalidated by resizing the vector to new_size.
    void assert_safe_to_reference_after_resize(const void* elt, size_t new_size)
    {
        assert(is_safe_to_reference_after_resize(elt, new_size) &&
               "Attempting to reference an element of the vector in an operation "
               "that invalidates it");
        (void) elt;
        (void) new_size;
    }

    /// check whether elt will be invalidated by increasing the size of the
    /// vector by N.
    void assert_safe_to_add(const void* elt, size_t N = 1)
    {
        this->assert_safe_to_reference_after_resize(elt, this->size() + N);
    }

    /// check whether any part of the range will be invalidated by clearing.
    void assert_safe_to_reference_after_clear(const T* from, const T* to)
    {
        if(from == to) return;
        this->assert_safe_to_reference_after_resize(from, 0);
        this->assert_safe_to_reference_after_resize(to - 1, 0);
    }
    template <typename ITp,
              std::enable_if_t<!std::is_same<std::remove_const_t<ITp>, T*>::value, bool> = false>
    void assert_safe_to_reference_after_clear(ITp, ITp)
    {}

    /// check whether any part of the range will be invalidated by growing.
    void assert_safe_to_add_range(const T* from, const T* to)
    {
        if(from == to) return;
        this->assert_safe_to_add(from, to - from);
        this->assert_safe_to_add(to - 1, to - from);
    }
    template <typename ITp,
              std::enable_if_t<!std::is_same<std::remove_const_t<ITp>, T*>::value, bool> = false>
    void assert_safe_to_add_range(ITp, ITp)
    {}

    /// reserve enough space to add one element, and return the updated element
    /// pointer in case it was a reference to the storage.
    template <typename U>
    static const T* reserve_for_param_and_get_address_impl(U* _this, const T& elt, size_t N)
    {
        size_t new_size = _this->size() + N;
        if(ROCPROFILER_LIKELY(new_size <= _this->capacity())) return &elt;

        bool    references_storage = false;
        int64_t index              = -1;
        if(!U::takes_param_by_value)
        {
            if(ROCPROFILER_UNLIKELY(_this->is_reference_to_storage(&elt)))
            {
                references_storage = true;
                index              = &elt - _this->begin();
            }
        }
        _this->grow(new_size);
        return references_storage ? _this->begin() + index : &elt;
    }

public:
    using size_type       = size_t;
    using difference_type = ptrdiff_t;
    using value_type      = T;
    using iterator        = T*;
    using const_iterator  = const T*;

    using const_reverse_iterator = std::reverse_iterator<const_iterator>;
    using reverse_iterator       = std::reverse_iterator<iterator>;

    using reference       = T&;
    using const_reference = const T&;
    using pointer         = T*;
    using const_pointer   = const T*;

    using base::capacity;
    using base::empty;
    using base::size;

    // forward iterator creation methods.
    iterator       begin() { return (iterator) this->m_begin_x; }
    const_iterator begin() const { return (const_iterator) this->m_begin_x; }
    iterator       end() { return begin() + size(); }
    const_iterator end() const { return begin() + size(); }

    // reverse iterator creation methods.
    reverse_iterator       rbegin() { return reverse_iterator(end()); }
    const_reverse_iterator rbegin() const { return const_reverse_iterator(end()); }
    reverse_iterator       rend() { return reverse_iterator(begin()); }
    const_reverse_iterator rend() const { return const_reverse_iterator(begin()); }

    size_type size_in_bytes() const { return size() * sizeof(T); }
    size_type max_size() const
    {
        return std::min(this->m_size_type_max(), size_type(-1) / sizeof(T));
    }

    size_t capacity_in_bytes() const { return capacity() * sizeof(T); }

    /// return a pointer to the vector's buffer, even if empty().
    pointer data() { return pointer(begin()); }
    /// return a pointer to the vector's buffer, even if empty().
    const_pointer data() const { return const_pointer(begin()); }

    reference operator[](size_type idx)
    {
        assert(idx < size());
        return begin()[idx];
    }
    const_reference operator[](size_type idx) const
    {
        assert(idx < size());
        return begin()[idx];
    }

    reference at(size_type idx)
    {
        if(idx >= size()) throw std::out_of_range{"small_vector::at"};
        return begin()[idx];
    }
    const_reference at(size_type idx) const
    {
        if(idx >= size()) throw std::out_of_range{"small_vector::at"};
        return begin()[idx];
    }

    reference front()
    {
        assert(!empty());
        return begin()[0];
    }
    const_reference front() const
    {
        assert(!empty());
        return begin()[0];
    }

    reference back()
    {
        assert(!empty());
        return end()[-1];
    }
    const_reference back() const
    {
        assert(!empty());
        return end()[-1];
    }
};

/// small_vector_template_base<Trivially_copyable = false> - this is where we put
/// method implementations that are designed to work with non-trivial T's.
///
/// we approximate is_trivially_copyable with trivial move/copy construction and
/// trivial destruction. while the standard doesn't specify that you're allowed
/// copy these types with memcpy, there is no way for the type to observe this.
/// this catches the important case of std::pair<POD, POD>, which is not
/// trivially assignable.
template <typename T,
          bool = (std::is_trivially_copy_constructible<T>::value) &&
                 (std::is_trivially_move_constructible<T>::value) &&
                 std::is_trivially_destructible<T>::value>
class small_vector_template_base : public small_vector_template_common<T>
{
    friend class small_vector_template_common<T>;

protected:
    static constexpr bool takes_param_by_value = false;
    using value_param_t                        = const T&;

    small_vector_template_base(size_t size)
    : small_vector_template_common<T>(size)
    {}

    static void destroy_range(T* S, T* E)
    {
        while(S != E)
        {
            --E;
            E->~T();
        }
    }

    /// move the range [I, E) into the uninitialized memory starting with "Dest",
    /// constructing elements as needed.
    template <typename It1, typename It2>
    static void uninitialized_move(It1 I, It1 E, It2 dest)
    {
        std::uninitialized_move(I, E, dest);
    }

    /// copy the range [I, E) onto the uninitialized memory starting with "Dest",
    /// constructing elements as needed.
    template <typename It1, typename It2>
    static void uninitialized_copy(It1 I, It1 E, It2 dest)
    {
        std::uninitialized_copy(I, E, dest);
    }

    /// grow the allocated memory (without initializing new elements), doubling
    /// the size of the allocated memory. guarantees space for at least one more
    /// element, or min_size more elements if specified.
    void grow(size_t min_size = 0);

    /// create a new allocation big enough for \p min_size and pass back its size
    /// in \p new_capacity. this is the first section of \a grow().
    T* malloc_for_grow(size_t min_size, size_t& new_capacity);

    /// move existing elements over to the new allocation \p new_elts, the middle
    /// section of \a grow().
    void move_elements_for_grow(T* new_elts);

    /// transfer ownership of the allocation, finishing up \a grow().
    void take_allocation_for_grow(T* new_elts, size_t new_capacity);

    /// reserve enough space to add one element, and return the updated element
    /// pointer in case it was a reference to the storage.
    const T* reserve_for_param_and_get_address(const T& elt, size_t N = 1)
    {
        return this->reserve_for_param_and_get_address_impl(this, elt, N);
    }

    /// reserve enough space to add one element, and return the updated element
    /// pointer in case it was a reference to the storage.
    T* reserve_for_param_and_get_address(T& elt, size_t N = 1)
    {
        return const_cast<T*>(this->reserve_for_param_and_get_address_impl(this, elt, N));
    }

    static T&&      forward_value_param(T&& V) { return std::move(V); }
    static const T& forward_value_param(const T& V) { return V; }

    void grow_and_assign(size_t num_elts, const T& elt)
    {
        // grow manually in case elt is an internal reference.
        size_t new_capacity;
        T*     new_elts = malloc_for_grow(num_elts, new_capacity);
        std::uninitialized_fill_n(new_elts, num_elts, elt);
        this->destroy_range(this->begin(), this->end());
        take_allocation_for_grow(new_elts, new_capacity);
        this->set_size(num_elts);
    }

    template <typename... Args>
    T& grow_and_emplace_back(Args&&... args)
    {
        // grow manually in case one of args is an internal reference.
        size_t new_capacity;
        T*     new_elts = malloc_for_grow(0, new_capacity);
        ::new((void*) (new_elts + this->size())) T(std::forward<Args>(args)...);
        move_elements_for_grow(new_elts);
        take_allocation_for_grow(new_elts, new_capacity);
        this->set_size(this->size() + 1);
        return this->back();
    }

public:
    void push_back(const T& elt)
    {
        const T* elt_ptr = reserve_for_param_and_get_address(elt);
        ::new((void*) this->end()) T(*elt_ptr);
        this->set_size(this->size() + 1);
    }

    void push_back(T&& elt)
    {
        T* elt_ptr = reserve_for_param_and_get_address(elt);
        ::new((void*) this->end()) T(::std::move(*elt_ptr));
        this->set_size(this->size() + 1);
    }

    void pop_back()
    {
        this->set_size(this->size() - 1);
        this->end()->~T();
    }
};

// define this out-of-line to dissuade the C++ compiler from inlining it.
template <typename T, bool trivially_copyable>
void
small_vector_template_base<T, trivially_copyable>::grow(size_t min_size)
{
    size_t new_capacity;
    T*     new_elts = malloc_for_grow(min_size, new_capacity);
    move_elements_for_grow(new_elts);
    take_allocation_for_grow(new_elts, new_capacity);
}

template <typename T, bool trivially_copyable>
T*
small_vector_template_base<T, trivially_copyable>::malloc_for_grow(size_t  min_size,
                                                                   size_t& new_capacity)
{
    return static_cast<T*>(small_vector_base<small_vector_size_type<T>>::malloc_for_grow(
        this->get_first_el(), min_size, sizeof(T), new_capacity));
}

// define this out-of-line to dissuade the C++ compiler from inlining it.
template <typename T, bool trivially_copyable>
void
small_vector_template_base<T, trivially_copyable>::move_elements_for_grow(T* new_elts)
{
    // move the elements over.
    this->uninitialized_move(this->begin(), this->end(), new_elts);

    // destroy the original elements.
    destroy_range(this->begin(), this->end());
}

// define this out-of-line to dissuade the C++ compiler from inlining it.
template <typename T, bool trivially_copyable>
void
small_vector_template_base<T, trivially_copyable>::take_allocation_for_grow(T*     new_elts,
                                                                            size_t new_capacity)
{
    // if this wasn't grown from the inline copy, deallocate the old space.
    if(!this->is_small()) free(this->begin());

    this->m_begin_x  = new_elts;
    this->m_capacity = new_capacity;
}

/// small_vector_template_base<Trivially_copyable = true> - this is where we put
/// method implementations that are designed to work with trivially copyable
/// T's. this allows using memcpy in place of copy/move construction and
/// skipping destruction.
template <typename T>
class small_vector_template_base<T, true> : public small_vector_template_common<T>
{
    friend class small_vector_template_common<T>;

protected:
    /// true if it's cheap enough to take parameters by value. doing so avoids
    /// overhead related to mitigations for reference invalidation.
    static constexpr bool takes_param_by_value = sizeof(T) <= 2 * sizeof(void*);

    /// either const T& or T, depending on whether it's cheap enough to take
    /// parameters by value.
    using value_param_t = std::conditional_t<takes_param_by_value, T, const T&>;

    small_vector_template_base(size_t size)
    : small_vector_template_common<T>(size)
    {}

    // no need to do a destroy loop for POD's.
    static void destroy_range(T*, T*) {}

    /// move the range [I, E) onto the uninitialized memory
    /// starting with "Dest", constructing elements into it as needed.
    template <typename It1, typename It2>
    static void uninitialized_move(It1 I, It1 E, It2 dest)
    {
        // just do a copy.
        uninitialized_copy(I, E, dest);
    }

    /// copy the range [I, E) onto the uninitialized memory
    /// starting with "Dest", constructing elements into it as needed.
    template <typename It1, typename It2>
    static void uninitialized_copy(It1 I, It1 E, It2 dest)
    {
        // arbitrary iterator types; just use the basic implementation.
        std::uninitialized_copy(I, E, dest);
    }

    /// copy the range [I, E) onto the uninitialized memory
    /// starting with "Dest", constructing elements into it as needed.
    template <typename T1, typename T2>
    static void uninitialized_copy(
        T1* I,
        T1* E,
        T2* dest,
        std::enable_if_t<std::is_same<std::remove_const_t<T1>, T2>::value>* = nullptr)
    {
        // use memcpy for PODs iterated by pointers (which includes small_vector
        // iterators): std::uninitialized_copy optimizes to memmove, but we can
        // use memcpy here. note that I and E are iterators and thus might be
        // invalid for memcpy if they are equal.
        if(I != E) memcpy(reinterpret_cast<void*>(dest), I, (E - I) * sizeof(T));
    }

    /// double the size of the allocated memory, guaranteeing space for at
    /// least one more element or min_size if specified.
    void grow(size_t min_size = 0) { this->grow_pod(min_size, sizeof(T)); }

    /// reserve enough space to add one element, and return the updated element
    /// pointer in case it was a reference to the storage.
    const T* reserve_for_param_and_get_address(const T& elt, size_t N = 1)
    {
        return this->reserve_for_param_and_get_address_impl(this, elt, N);
    }

    /// reserve enough space to add one element, and return the updated element
    /// pointer in case it was a reference to the storage.
    T* reserve_for_param_and_get_address(T& elt, size_t N = 1)
    {
        return const_cast<T*>(this->reserve_for_param_and_get_address_impl(this, elt, N));
    }

    /// copy \p V or return a reference, depending on \a value_param_t.
    static value_param_t forward_value_param(value_param_t V) { return V; }

    void grow_and_assign(size_t num_elts, T elt)
    {
        // elt has been copied in case it's an internal reference, side-stepping
        // reference invalidation problems without losing the realloc optimization.
        this->set_size(0);
        this->grow(num_elts);
        std::uninitialized_fill_n(this->begin(), num_elts, elt);
        this->set_size(num_elts);
    }

    template <typename... Args>
    T& grow_and_emplace_back(Args&&... args)
    {
        // use push_back with a copy in case args has an internal reference,
        // side-stepping reference invalidation problems without losing the realloc
        // optimization.
        push_back(T(std::forward<Args>(args)...));
        return this->back();
    }

public:
    void push_back(value_param_t elt)
    {
        const T* elt_ptr = reserve_for_param_and_get_address(elt);
        memcpy(reinterpret_cast<void*>(this->end()), elt_ptr, sizeof(T));
        this->set_size(this->size() + 1);
    }

    void pop_back() { this->set_size(this->size() - 1); }
};

/// this class consists of common code factored out of the small_vector class to
/// reduce code duplication based on the small_vector 'N' template parameter.
template <typename T>
class small_vector_impl : public small_vector_template_base<T>
{
    using super_class = small_vector_template_base<T>;

public:
    using iterator       = typename super_class::iterator;
    using const_iterator = typename super_class::const_iterator;
    using reference      = typename super_class::reference;
    using size_type      = typename super_class::size_type;

protected:
    using small_vector_template_base<T>::takes_param_by_value;
    using value_param_t = typename super_class::value_param_t;

    // default ctor - initialize to empty.
    explicit small_vector_impl(unsigned N)
    : small_vector_template_base<T>(N)
    {}

    void assign_remote(small_vector_impl&& RHS)
    {
        this->destroy_range(this->begin(), this->end());
        if(!this->is_small()) free(this->begin());
        this->m_begin_x  = RHS.m_begin_x;
        this->m_size     = RHS.m_size;
        this->m_capacity = RHS.m_capacity;
        RHS.reset_to_small();
    }

public:
    small_vector_impl(const small_vector_impl&) = delete;

    ~small_vector_impl()
    {
        // subclass has already destructed this vector's elements.
        // if this wasn't grown from the inline copy, deallocate the old space.
        if(!this->is_small()) free(this->begin());
    }

    void clear()
    {
        this->destroy_range(this->begin(), this->end());
        this->m_size = 0;
    }

private:
    // make set_size() private to avoid misuse in subclasses.
    using super_class::set_size;

    template <bool for_overwrite>
    void resize_impl(size_type N)
    {
        if(N == this->size()) return;

        if(N < this->size())
        {
            this->truncate(N);
            return;
        }

        this->reserve(N);
        for(auto I = this->end(), E = this->begin() + N; I != E; ++I)
            if(for_overwrite)
                new(&*I) T;
            else
                new(&*I) T();
        this->set_size(N);
    }

public:
    void resize(size_type N) { resize_impl<false>(N); }

    /// like resize, but \ref T is POD, the new values won't be initialized.
    void resize_for_overwrite(size_type N) { resize_impl<true>(N); }

    /// like resize, but requires that \p N is less than \a size().
    void truncate(size_type N)
    {
        assert(this->size() >= N && "Cannot increase size with truncate");
        this->destroy_range(this->begin() + N, this->end());
        this->set_size(N);
    }

    void resize(size_type N, value_param_t NV)
    {
        if(N == this->size()) return;

        if(N < this->size())
        {
            this->truncate(N);
            return;
        }

        // N > this->size(). defer to append.
        this->append(N - this->size(), NV);
    }

    void reserve(size_type N)
    {
        if(this->capacity() < N) this->grow(N);
    }

    void pop_back_n(size_type num_items)
    {
        assert(this->size() >= num_items);
        truncate(this->size() - num_items);
    }

    [[nodiscard]] T pop_back_val()
    {
        T result = ::std::move(this->back());
        this->pop_back();
        return result;
    }

    void swap(small_vector_impl& RHS) noexcept;

    /// add the specified range to the end of the small_vector.
    template <typename ITp, typename = enable_if_convertible_to_input_iterator<ITp>>
    void append(ITp in_start, ITp in_end)
    {
        this->assert_safe_to_add_range(in_start, in_end);
        size_type num_inputs = std::distance(in_start, in_end);
        this->reserve(this->size() + num_inputs);
        this->uninitialized_copy(in_start, in_end, this->end());
        this->set_size(this->size() + num_inputs);
    }

    /// append \p num_inputs copies of \p elt to the end.
    void append(size_type num_inputs, value_param_t elt)
    {
        const T* elt_ptr = this->reserve_for_param_and_get_address(elt, num_inputs);
        std::uninitialized_fill_n(this->end(), num_inputs, *elt_ptr);
        this->set_size(this->size() + num_inputs);
    }

    void append(std::initializer_list<T> IL) { append(IL.begin(), IL.end()); }

    void append(const small_vector_impl& RHS) { append(RHS.begin(), RHS.end()); }

    void assign(size_type num_elts, value_param_t elt)
    {
        // note that elt could be an internal reference.
        if(num_elts > this->capacity())
        {
            this->grow_and_assign(num_elts, elt);
            return;
        }

        // assign over existing elements.
        std::fill_n(this->begin(), std::min(num_elts, this->size()), elt);
        if(num_elts > this->size())
            std::uninitialized_fill_n(this->end(), num_elts - this->size(), elt);
        else if(num_elts < this->size())
            this->destroy_range(this->begin() + num_elts, this->end());
        this->set_size(num_elts);
    }

    // FIXME: consider assigning over existing elements, rather than clearing &
    // re-initializing them - for all assign(...) variants.

    template <typename ITp, typename = enable_if_convertible_to_input_iterator<ITp>>
    void assign(ITp in_start, ITp in_end)
    {
        this->assert_safe_to_reference_after_clear(in_start, in_end);
        clear();
        append(in_start, in_end);
    }

    void assign(std::initializer_list<T> IL)
    {
        clear();
        append(IL);
    }

    void assign(const small_vector_impl& RHS) { assign(RHS.begin(), RHS.end()); }

    iterator erase(const_iterator CI)
    {
        // just cast away constness because this is a non-const member function.
        iterator I = const_cast<iterator>(CI);

        assert(this->is_reference_to_storage(CI) && "Iterator to erase is out of bounds.");

        iterator N = I;
        // shift all elts down one.
        std::move(I + 1, this->end(), I);
        // drop the last elt.
        this->pop_back();
        return (N);
    }

    iterator erase(const_iterator CS, const_iterator CE)
    {
        // just cast away constness because this is a non-const member function.
        iterator S = const_cast<iterator>(CS);
        iterator E = const_cast<iterator>(CE);

        assert(this->is_range_in_storage(S, E) && "Range to erase is out of bounds.");

        iterator N = S;
        // shift all elts down.
        iterator I = std::move(E, this->end(), S);
        // drop the last elts.
        this->destroy_range(I, this->end());
        this->set_size(I - this->begin());
        return (N);
    }

private:
    template <typename ArgT>
    iterator insert_one_impl(iterator I, ArgT&& elt)
    {
        // callers ensure that ArgT is derived from T.
        static_assert(std::is_same<std::remove_const_t<std::remove_reference_t<ArgT>>, T>::value,
                      "ArgT must be derived from T!");

        if(I == this->end())
        {  // important special case for empty vector.
            this->push_back(::std::forward<ArgT>(elt));
            return this->end() - 1;
        }

        assert(this->is_reference_to_storage(I) && "Insertion iterator is out of bounds.");

        // grow if necessary.
        size_t                         index   = I - this->begin();
        std::remove_reference_t<ArgT>* elt_ptr = this->reserve_for_param_and_get_address(elt);
        I                                      = this->begin() + index;

        ::new((void*) this->end()) T(::std::move(this->back()));
        // push everything else over.
        std::move_backward(I, this->end() - 1, this->end());
        this->set_size(this->size() + 1);

        // if we just moved the element we're inserting, be sure to update
        // the reference (never happens if takes_param_by_value).
        static_assert(!takes_param_by_value || std::is_same<ArgT, T>::value,
                      "ArgT must be 'T' when taking by value!");
        if(!takes_param_by_value && this->is_reference_to_range(elt_ptr, I, this->end())) ++elt_ptr;

        *I = ::std::forward<ArgT>(*elt_ptr);
        return I;
    }

public:
    iterator insert(iterator I, T&& elt)
    {
        return insert_one_impl(I, this->forward_value_param(std::move(elt)));
    }

    iterator insert(iterator I, const T& elt)
    {
        return insert_one_impl(I, this->forward_value_param(elt));
    }

    iterator insert(iterator I, size_type num_to_insert, value_param_t elt)
    {
        // convert iterator to elt# to avoid invalidating iterator when we reserve()
        size_t insert_elt = I - this->begin();

        if(I == this->end())
        {  // important special case for empty vector.
            append(num_to_insert, elt);
            return this->begin() + insert_elt;
        }

        assert(this->is_reference_to_storage(I) && "Insertion iterator is out of bounds.");

        // ensure there is enough space, and get the (maybe updated) address of
        // elt.
        const T* elt_ptr = this->reserve_for_param_and_get_address(elt, num_to_insert);

        // uninvalidate the iterator.
        I = this->begin() + insert_elt;

        // if there are more elements between the insertion point and the end of the
        // range than there are being inserted, we can use a simple approach to
        // insertion.  since we already reserved space, we know that this won't
        // reallocate the vector.
        if(size_t(this->end() - I) >= num_to_insert)
        {
            T* old_end = this->end();
            append(std::move_iterator<iterator>(this->end() - num_to_insert),
                   std::move_iterator<iterator>(this->end()));

            // copy the existing elements that get replaced.
            std::move_backward(I, old_end - num_to_insert, old_end);

            // if we just moved the element we're inserting, be sure to update
            // the reference (never happens if takes_param_by_value).
            if(!takes_param_by_value && I <= elt_ptr && elt_ptr < this->end())
                elt_ptr += num_to_insert;

            std::fill_n(I, num_to_insert, *elt_ptr);
            return I;
        }

        // otherwise, we're inserting more elements than exist already, and we're
        // not inserting at the end.

        // move over the elements that we're about to overwrite.
        T* old_end = this->end();
        this->set_size(this->size() + num_to_insert);
        size_t num_overwritten = old_end - I;
        this->uninitialized_move(I, old_end, this->end() - num_overwritten);

        // if we just moved the element we're inserting, be sure to update
        // the reference (never happens if takes_param_by_value).
        if(!takes_param_by_value && I <= elt_ptr && elt_ptr < this->end()) elt_ptr += num_to_insert;

        // replace the overwritten part.
        std::fill_n(I, num_overwritten, *elt_ptr);

        // insert the non-overwritten middle part.
        std::uninitialized_fill_n(old_end, num_to_insert - num_overwritten, *elt_ptr);
        return I;
    }

    template <typename ITp, typename = enable_if_convertible_to_input_iterator<ITp>>
    iterator insert(iterator I, ITp from, ITp to)
    {
        // convert iterator to elt# to avoid invalidating iterator when we reserve()
        size_t insert_elt = I - this->begin();

        if(I == this->end())
        {  // important special case for empty vector.
            append(from, to);
            return this->begin() + insert_elt;
        }

        assert(this->is_reference_to_storage(I) && "Insertion iterator is out of bounds.");

        // check that the reserve that follows doesn't invalidate the iterators.
        this->assert_safe_to_add_range(from, to);

        size_t num_to_insert = std::distance(from, to);

        // ensure there is enough space.
        reserve(this->size() + num_to_insert);

        // uninvalidate the iterator.
        I = this->begin() + insert_elt;

        // if there are more elements between the insertion point and the end of the
        // range than there are being inserted, we can use a simple approach to
        // insertion.  since we already reserved space, we know that this won't
        // reallocate the vector.
        if(size_t(this->end() - I) >= num_to_insert)
        {
            T* old_end = this->end();
            append(std::move_iterator<iterator>(this->end() - num_to_insert),
                   std::move_iterator<iterator>(this->end()));

            // copy the existing elements that get replaced.
            std::move_backward(I, old_end - num_to_insert, old_end);

            std::copy(from, to, I);
            return I;
        }

        // otherwise, we're inserting more elements than exist already, and we're
        // not inserting at the end.

        // move over the elements that we're about to overwrite.
        T* old_end = this->end();
        this->set_size(this->size() + num_to_insert);
        size_t num_overwritten = old_end - I;
        this->uninitialized_move(I, old_end, this->end() - num_overwritten);

        // replace the overwritten part.
        for(T* J = I; num_overwritten > 0; --num_overwritten)
        {
            *J = *from;
            ++J;
            ++from;
        }

        // insert the non-overwritten middle part.
        this->uninitialized_copy(from, to, old_end);
        return I;
    }

    void insert(iterator I, std::initializer_list<T> IL) { insert(I, IL.begin(), IL.end()); }

    template <typename... Args>
    reference emplace_back(Args&&... args)
    {
        if(ROCPROFILER_UNLIKELY(this->size() >= this->capacity()))
            return this->grow_and_emplace_back(std::forward<Args>(args)...);

        ::new((void*) this->end()) T(std::forward<Args>(args)...);
        this->set_size(this->size() + 1);
        return this->back();
    }

    small_vector_impl& operator=(const small_vector_impl& RHS);

    small_vector_impl& operator=(small_vector_impl&& RHS) noexcept;

    bool operator==(const small_vector_impl& RHS) const
    {
        if(this->size() != RHS.size()) return false;
        return std::equal(this->begin(), this->end(), RHS.begin());
    }
    bool operator!=(const small_vector_impl& RHS) const { return !(*this == RHS); }

    bool operator<(const small_vector_impl& RHS) const
    {
        return std::lexicographical_compare(this->begin(), this->end(), RHS.begin(), RHS.end());
    }
    bool operator>(const small_vector_impl& RHS) const { return RHS < *this; }
    bool operator<=(const small_vector_impl& RHS) const { return !(*this > RHS); }
    bool operator>=(const small_vector_impl& RHS) const { return !(*this < RHS); }
};

template <typename T>
void
small_vector_impl<T>::swap(small_vector_impl<T>& RHS) noexcept
{
    if(this == &RHS) return;

    // we can only avoid copying elements if neither vector is small.
    if(!this->is_small() && !RHS.is_small())
    {
        std::swap(this->m_begin_x, RHS.m_begin_x);
        std::swap(this->m_size, RHS.m_size);
        std::swap(this->m_capacity, RHS.m_capacity);
        return;
    }
    this->reserve(RHS.size());
    RHS.reserve(this->size());

    // swap the shared elements.
    size_t num_shared = this->size();
    if(num_shared > RHS.size()) num_shared = RHS.size();
    for(size_type i = 0; i != num_shared; ++i)
        std::swap((*this)[i], RHS[i]);

    // copy over the extra elts.
    if(this->size() > RHS.size())
    {
        size_t elt_diff = this->size() - RHS.size();
        this->uninitialized_copy(this->begin() + num_shared, this->end(), RHS.end());
        RHS.set_size(RHS.size() + elt_diff);
        this->destroy_range(this->begin() + num_shared, this->end());
        this->set_size(num_shared);
    }
    else if(RHS.size() > this->size())
    {
        size_t elt_diff = RHS.size() - this->size();
        this->uninitialized_copy(RHS.begin() + num_shared, RHS.end(), this->end());
        this->set_size(this->size() + elt_diff);
        this->destroy_range(RHS.begin() + num_shared, RHS.end());
        RHS.set_size(num_shared);
    }
}

template <typename T>
small_vector_impl<T>&
small_vector_impl<T>::operator=(const small_vector_impl<T>& RHS)
{
    // avoid self-assignment.
    if(this == &RHS) return *this;

    // if we already have sufficient space, assign the common elements, then
    // destroy any excess.
    size_t RHSSize  = RHS.size();
    size_t cur_size = this->size();
    if(cur_size >= RHSSize)
    {
        // assign common elements.
        iterator new_end;
        if(RHSSize != 0u)
            new_end = std::copy(RHS.begin(), RHS.begin() + RHSSize, this->begin());
        else
            new_end = this->begin();

        // destroy excess elements.
        this->destroy_range(new_end, this->end());

        // trim.
        this->set_size(RHSSize);
        return *this;
    }

    // if we have to grow to have enough elements, destroy the current elements.
    // this allows us to avoid copying them during the grow.
    // FIXME: don't do this if they're efficiently moveable.
    if(this->capacity() < RHSSize)
    {
        // destroy current elements.
        this->clear();
        cur_size = 0;
        this->grow(RHSSize);
    }
    else if(cur_size != 0u)
    {
        // otherwise, use assignment for the already-constructed elements.
        std::copy(RHS.begin(), RHS.begin() + cur_size, this->begin());
    }

    // copy construct the new elements in place.
    this->uninitialized_copy(RHS.begin() + cur_size, RHS.end(), this->begin() + cur_size);

    // set end.
    this->set_size(RHSSize);
    return *this;
}

template <typename T>
small_vector_impl<T>&
small_vector_impl<T>::operator=(small_vector_impl<T>&& RHS) noexcept
{
    // avoid self-assignment.
    if(this == &RHS) return *this;

    // if the RHS isn't small, clear this vector and then steal its buffer.
    if(!RHS.is_small())
    {
        this->assign_remote(std::move(RHS));
        return *this;
    }

    // if we already have sufficient space, assign the common elements, then
    // destroy any excess.
    size_t RHSSize  = RHS.size();
    size_t cur_size = this->size();
    if(cur_size >= RHSSize)
    {
        // assign common elements.
        iterator new_end = this->begin();
        if(RHSSize != 0u) new_end = std::move(RHS.begin(), RHS.end(), new_end);

        // destroy excess elements and trim the bounds.
        this->destroy_range(new_end, this->end());
        this->set_size(RHSSize);

        // clear the RHS.
        RHS.clear();

        return *this;
    }

    // if we have to grow to have enough elements, destroy the current elements.
    // this allows us to avoid copying them during the grow.
    // FIXME: this may not actually make any sense if we can efficiently move
    // elements.
    if(this->capacity() < RHSSize)
    {
        // destroy current elements.
        this->clear();
        cur_size = 0;
        this->grow(RHSSize);
    }
    else if(cur_size != 0u)
    {
        // otherwise, use assignment for the already-constructed elements.
        std::move(RHS.begin(), RHS.begin() + cur_size, this->begin());
    }

    // move-construct the new elements in place.
    this->uninitialized_move(RHS.begin() + cur_size, RHS.end(), this->begin() + cur_size);

    // set end.
    this->set_size(RHSSize);

    RHS.clear();
    return *this;
}

/// storage for the small_vector elements.  this is specialized for the N=0 case
/// to avoid allocating unnecessary storage.
template <typename T, unsigned N>
struct small_vector_storage
{
    alignas(T) char inline_elts[N * sizeof(T)];
};

/// we need the storage to be properly aligned even for small-size of 0 so that
/// the pointer math in \a small_vector_template_common::get_first_el() is
/// well-defined.
template <typename T>
struct alignas(T) small_vector_storage<T, 0>
{};

/// forward declaration of small_vector so that
/// calculate_small_vector_default_inlined_elements can reference
/// `sizeof(small_vector<T, 0>)`.
template <typename T, unsigned N>
class small_vector;

/// helper class for calculating the default number of inline elements for
/// `small_vector<T>`.
///
/// this should be migrated to a constexpr function when our minimum
/// compiler support is enough for multi-statement constexpr functions.
template <typename T>
struct calculate_small_vector_default_inlined_elements
{
    // parameter controlling the default number of inlined elements
    // for `small_vector<T>`.
    //
    // the default number of inlined elements ensures that
    // 1. there is at least one inlined element.
    // 2. `sizeof(small_vector<T>) <= k_preferred_small_vector_sizeof` unless
    // it contradicts 1.
    static constexpr size_t k_preferred_small_vector_sizeof = 64;

    // static_assert that sizeof(T) is not "too big".
    //
    // because our policy guarantees at least one inlined element, it is possible
    // for an arbitrarily large inlined element to allocate an arbitrarily large
    // amount of inline storage. we generally consider it an antipattern for a
    // small_vector to allocate an excessive amount of inline storage, so we want
    // to call attention to these cases and make sure that users are making an
    // intentional decision if they request a lot of inline storage.
    //
    // we want this assertion to trigger in pathological cases, but otherwise
    // not be too easy to hit. to accomplish that, the cutoff is actually somewhat
    // larger than k_preferred_small_vector_sizeof (otherwise,
    // `small_vector<small_vector<T>>` would be one easy way to trip it, and that
    // pattern seems useful in practice).
    //
    // one wrinkle is that this assertion is in theory non-portable, since
    // sizeof(T) is in general platform-dependent. however, we don't expect this
    // to be much of an issue, because most LLVM development happens on 64-bit
    // hosts, and therefore sizeof(T) is expected to *decrease* when compiled for
    // 32-bit hosts, dodging the issue. the reverse situation, where development
    // happens on a 32-bit host and then fails due to sizeof(T) *increasing* on a
    // 64-bit host, is expected to be very rare.
    static_assert(sizeof(T) <= 256,
                  "You are trying to use a default number of inlined elements for "
                  "`small_vector<T>` but `sizeof(T)` is really big! please use an "
                  "explicit number of inlined elements with `small_vector<T, N>` to make "
                  "sure you really want that much inline storage.");

    // discount the size of the header itself when calculating the maximum inline
    // bytes.
    static constexpr size_t preferred_inline_bytes =
        k_preferred_small_vector_sizeof - sizeof(small_vector<T, 0>);
    static constexpr size_t num_elements_that_fit = preferred_inline_bytes / sizeof(T);
    static constexpr size_t value = num_elements_that_fit == 0 ? 1 : num_elements_that_fit;
};

/// this is a 'vector' (really, a variable-sized array), optimized
/// for the case when the array is small.  it contains some number of elements
/// in-place, which allows it to avoid heap allocation when the actual number of
/// elements is below that threshold.  this allows normal "small" cases to be
/// fast without losing generality for large inputs.
///
/// \note
/// in the absence of a well-motivated choice for the number of inlined
/// elements \p N, it is recommended to use \c small_vector<T> (that is,
/// omitting the \p N). this will choose a default number of inlined elements
/// reasonable for allocation on the stack (for example, trying to keep \c
/// sizeof(small_vector<T>) around 64 bytes).
///
/// \warning this does not attempt to be exception safe.
///
/// \see https://llvm.org/docs/Programmers_manual.html#llvm-adt-smallvector-h
template <typename T, unsigned N = calculate_small_vector_default_inlined_elements<T>::value>
class small_vector
: public small_vector_impl<T>
, small_vector_storage<T, N>
{
public:
    small_vector()
    : small_vector_impl<T>(N)
    {}

    ~small_vector()
    {
        // destroy the constructed elements in the vector.
        this->destroy_range(this->begin(), this->end());
    }

    explicit small_vector(size_t size)
    : small_vector_impl<T>(N)
    {
        this->resize(size);
    }

    small_vector(size_t size, const T& value)
    : small_vector_impl<T>(N)
    {
        this->assign(size, value);
    }

    template <typename ITp, typename = enable_if_convertible_to_input_iterator<ITp>>
    small_vector(ITp S, ITp E)
    : small_vector_impl<T>(N)
    {
        this->append(S, E);
    }

    template <typename RangeTp>
    explicit small_vector(const iterator_range<RangeTp>& R)
    : small_vector_impl<T>(N)
    {
        this->append(R.begin(), R.end());
    }

    small_vector(std::initializer_list<T> IL)
    : small_vector_impl<T>(N)
    {
        this->append(IL);
    }

    template <typename U, typename = std::enable_if_t<std::is_convertible<U, T>::value>>
    explicit small_vector(array_ref<U> A)
    : small_vector_impl<T>(N)
    {
        this->append(A.begin(), A.end());
    }

    small_vector(const small_vector& RHS)
    : small_vector_impl<T>(N)
    {
        if(!RHS.empty()) small_vector_impl<T>::operator=(RHS);
    }

    small_vector& operator=(const small_vector& RHS)
    {
        small_vector_impl<T>::operator=(RHS);
        return *this;
    }

    small_vector(small_vector&& RHS) noexcept
    : small_vector_impl<T>(N)
    {
        if(!RHS.empty()) small_vector_impl<T>::operator=(::std::move(RHS));
    }

    small_vector(small_vector_impl<T>&& RHS)
    : small_vector_impl<T>(N)
    {
        if(!RHS.empty()) small_vector_impl<T>::operator=(::std::move(RHS));
    }

    small_vector& operator=(small_vector&& RHS) noexcept
    {
        if(N != 0u)
        {
            small_vector_impl<T>::operator=(::std::move(RHS));
            return *this;
        }
        // small_vector_impl<T>::operator= does not leverage N==0. optimize the
        // case.
        if(this == &RHS) return *this;
        if(RHS.empty())
        {
            this->destroy_range(this->begin(), this->end());
            this->m_size = 0;
        }
        else
        {
            this->assign_remote(std::move(RHS));
        }
        return *this;
    }

    small_vector& operator=(small_vector_impl<T>&& RHS)
    {
        small_vector_impl<T>::operator=(::std::move(RHS));
        return *this;
    }

    small_vector& operator=(std::initializer_list<T> IL)
    {
        this->assign(IL);
        return *this;
    }
};

template <typename T, unsigned N>
inline size_t
capacity_in_bytes(const small_vector<T, N>& X)
{
    return X.capacity_in_bytes();
}

template <typename range_type>
using value_type_from_range_type = std::remove_const_t<
    std::remove_reference_t<decltype(*std::begin(std::declval<range_type&>()))>>;

/// given a range of type R, iterate the entire range and return a
/// small_vector with elements of the vector.  this is useful, for example,
/// when you want to iterate a range and then sort the results.
template <unsigned size, typename R>
small_vector<value_type_from_range_type<R>, size>
to_vector(R&& range)
{
    return {std::begin(range), std::end(range)};
}
template <typename R>
small_vector<value_type_from_range_type<R>>
to_vector(R&& range)
{
    return {std::begin(range), std::end(range)};
}

template <typename OutT, unsigned SizeT, typename R>
small_vector<OutT, SizeT>
to_vector_of(R&& range)
{
    return {std::begin(range), std::end(range)};
}

template <typename OutT, typename R>
small_vector<OutT>
to_vector_of(R&& range)
{
    return {std::begin(range), std::end(range)};
}
}  // namespace container
}  // namespace common
}  // namespace rocprofiler

// explicit instantiations
extern template class rocprofiler::common::container::small_vector_base<uint32_t>;
#if SIZE_MAX > UINT32_MAX
extern template class rocprofiler::common::container::small_vector_base<uint64_t>;
#endif

namespace std
{
/// implement std::swap in terms of small_vector swap.
template <typename T>
inline void
swap(rocprofiler::common::container::small_vector_impl<T>& LHS,
     rocprofiler::common::container::small_vector_impl<T>& RHS) noexcept
{
    LHS.swap(RHS);
}

/// implement std::swap in terms of small_vector swap.
template <typename T, unsigned N>
inline void
swap(rocprofiler::common::container::small_vector<T, N>& LHS,
     rocprofiler::common::container::small_vector<T, N>& RHS) noexcept
{
    LHS.swap(RHS);
}
}  // end namespace std
