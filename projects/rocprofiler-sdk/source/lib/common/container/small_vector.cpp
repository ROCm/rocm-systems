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

#include "lib/common/container/small_vector.hpp"

namespace rocprofiler
{
namespace common
{
namespace container
{
namespace
{
[[noreturn]] void
report_size_overflow(size_t min_size, size_t max_size);

[[noreturn]] void
report_at_maximum_capacity(size_t max_size);

/// Report that min_size doesn't fit into this vector's size type. Throws
/// std::length_error
void
report_size_overflow(size_t min_size, size_t max_size)
{
    std::string Reason =
        "small_vector unable to grow. Requested capacity (" + std::to_string(min_size) +
        ") is larger than maximum value for size type (" + std::to_string(max_size) + ")";
    throw std::length_error(Reason);
}

/// Report that this vector is already at maximum capacity. Throws
/// std::length_error
void
report_at_maximum_capacity(size_t max_size)
{
    std::string Reason =
        "small_vector capacity unable to grow. Already at maximum size " + std::to_string(max_size);
    throw std::length_error(Reason);
}

template <typename SizeT>
size_t
get_new_capacity(size_t min_size, size_t /*t_size*/, size_t old_capacity)
{
    constexpr size_t max_size = std::numeric_limits<SizeT>::max();

    // Ensure we can fit the new capacity.
    // This is only going to be applicable when the capacity is 32 bit.
    if(min_size > max_size) report_size_overflow(min_size, max_size);

    // Ensure we can meet the guarantee of space for at least one more element.
    // The above check alone will not catch the case where grow is called with a
    // default min_size of 0, but the current capacity cannot be increased.
    // This is only going to be applicable when the capacity is 32 bit.
    if(old_capacity == max_size) report_at_maximum_capacity(max_size);

    // In theory 2*capacity can overflow if the capacity is 64 bit, but the
    // original capacity would never be large enough for this to be a problem.
    size_t new_capacity = (2 * old_capacity) + 1;  // Always grow.
    return std::clamp(new_capacity, min_size, max_size);
}
}  // namespace

template <typename SizeT>
void*
small_vector_base<SizeT>::replace_allocation(void*  new_elts,
                                             size_t t_size,
                                             size_t new_capacity,
                                             size_t v_size)
{
    void* new_eltsReplace = ::malloc(new_capacity * t_size);
    if(v_size != 0u) memcpy(new_eltsReplace, new_elts, v_size * t_size);
    free(new_elts);
    return new_eltsReplace;
}

// Note: Moving this function into the header may cause performance regression.
template <typename SizeT>
void*
small_vector_base<SizeT>::malloc_for_grow(void*   first_el,
                                          size_t  min_size,
                                          size_t  t_size,
                                          size_t& new_capacity)
{
    new_capacity = get_new_capacity<SizeT>(min_size, t_size, this->capacity());
    // Even if capacity is not 0 now, if the vector was originally created with
    // capacity 0, it's possible for the malloc to return first_el.
    void* new_elts = ::malloc(new_capacity * t_size);
    if(new_elts == first_el) new_elts = replace_allocation(new_elts, t_size, new_capacity);
    return new_elts;
}

// Note: Moving this function into the header may cause performance regression.
template <typename SizeT>
void
small_vector_base<SizeT>::grow_pod(void* first_el, size_t min_size, size_t t_size)
{
    size_t new_capacity = get_new_capacity<SizeT>(min_size, t_size, this->capacity());
    void*  new_elts;
    if(m_begin_x == first_el)
    {
        new_elts = ::malloc(new_capacity * t_size);
        if(new_elts == first_el) new_elts = replace_allocation(new_elts, t_size, new_capacity);

        // Copy the elements over.  No need to run dtors on PODs.
        memcpy(new_elts, this->m_begin_x, size() * t_size);
    }
    else
    {
        // If this wasn't grown from the inline copy, grow the allocated space.
        new_elts = ::realloc(this->m_begin_x, new_capacity * t_size);
        if(new_elts == first_el)
            new_elts = replace_allocation(new_elts, t_size, new_capacity, size());
    }

    this->m_begin_x  = new_elts;
    this->m_capacity = new_capacity;
}
}  // namespace container
}  // namespace common
}  // namespace rocprofiler

// explicit instantiations
template class rocprofiler::common::container::small_vector_base<uint32_t>;
#if SIZE_MAX > UINT32_MAX
template class rocprofiler::common::container::small_vector_base<uint64_t>;
#endif
