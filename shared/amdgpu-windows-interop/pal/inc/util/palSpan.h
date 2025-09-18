/*
 ***********************************************************************************************************************
 *
 *  Copyright (c) 2022-2025 Advanced Micro Devices, Inc. All Rights Reserved.
 *
 *  Permission is hereby granted, free of charge, to any person obtaining a copy
 *  of this software and associated documentation files (the "Software"), to deal
 *  in the Software without restriction, including without limitation the rights
 *  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 *  copies of the Software, and to permit persons to whom the Software is
 *  furnished to do so, subject to the following conditions:
 *
 *  The above copyright notice and this permission notice shall be included in all
 *  copies or substantial portions of the Software.
 *
 *  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 *  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 *  SOFTWARE.
 *
 **********************************************************************************************************************/
/**
***********************************************************************************************************************
* @file  palSpan.h
* @brief PAL utility collection Span class declaration.
***********************************************************************************************************************
*/

#pragma once

#include "palUtil.h"
#include "palAssert.h"
#include "palSysMemory.h"
#include "palInlineFuncs.h"
#include <type_traits>

namespace Util
{

/**
 ***********************************************************************************************************************
 * @brief Span container
 *
 * Span is an array with a length, where the data is not owned by the Span object. It is similar to C++20 std::span,
 * but only the dynamic extent variant. It is similar to LLVM MutableArrayRef and ArrayRef. A Span is intended to
 * be passed around by value.
 *
 ***********************************************************************************************************************
 */
template<typename T>
class Span
{
public:
    /// Constructor from nothing. This allows you to use {} to mean an empty Span.
    constexpr Span() : m_pData(nullptr), m_numElements(0) {}

    /// Constructor from pointer and length
    ///
    /// @param [in] data Pointer to the start of the array
    /// @param numElements Number of elements in the array
    constexpr Span(T* pData, size_t numElements) : m_pData(pData), m_numElements(numElements) {}

    /// Copy constructor
    ///
    /// @param [in] src Other Span to copy from
    constexpr Span(const Span<T>& src) : m_pData(src.m_pData), m_numElements(src.m_numElements) {}

    /// Constructor from C++ array
    ///
    /// @param [in] src C++ array
    template<size_t NumElements> constexpr Span(T(& src)[NumElements]) : m_pData(&src[0]), m_numElements(NumElements) {}

    /// Constructor from single element
    ///
    /// @param [in] src Single element
    constexpr Span(T& src) : m_pData(&src), m_numElements(1) {}

    /// Implicitly convert a Span to its const-element equivalent.
    ///
    /// @returns The same span, but with const element type
    constexpr operator Span<const T>() const { return Span<const T>(m_pData, m_numElements); }

    /// Assignment operator
    ///
    /// @param [in] src Other Span to copy from
    constexpr Span<T>& operator=(const Span<T>& src) {
      m_pData = src.m_pData;
      m_numElements = src.m_numElements;
      return *this;
    }

    ///@{
    /// Returns the element at the location specified.
    ///
    /// @param [in] index Integer location of the element needed.
    ///
    /// @returns The element at location specified by index by reference
    constexpr T& At(size_t index) const
    {
        PAL_CONSTEXPR_ASSERT(index < m_numElements);
        return *(m_pData + index);
    }

    constexpr T& operator[](size_t index) const noexcept { return At(index); }
    ///@}

    /// Returns the data at the front of the vector.
    ///
    /// @returns The data at the front of the vector.
    constexpr T& Front() const
    {
        PAL_CONSTEXPR_ASSERT(IsEmpty() == false);
        return *m_pData;
    }

    /// Returns the data at the back of the vector.
    ///
    /// @returns The data at the back of the vector.
    constexpr T& Back() const
    {
        PAL_CONSTEXPR_ASSERT(IsEmpty() == false);
        return *(m_pData + (m_numElements - 1));
    }

    /// Returns an iterator to the first element of the vector.
    ///
    /// @returns An iterator to first element of the vector.
    constexpr T* Begin() const { return m_pData; }

    /// Returns an iterator beyond the last element of the vector. (NOT at the last element like Util::Vector::End()!)
    ///
    /// @warning Accessing an element using an iterator of an empty vector will cause an access violation!
    ///
    /// @returns VectorIterator An iterator to last element of the vector.
    constexpr T* End() const { return m_pData + m_numElements; }

    /// Returns pointer to the underlying buffer serving as data storage.
    ///
    /// @returns Pointer to the underlying data storage.
    ///          For a non-empty span, the returned pointer contains address of the first element.
    ///          For an empty span, the returned pointer may or may not be a null pointer.
    constexpr T* Data() const { return m_pData; }

    /// Returns the extent of the span.
    ///
    /// @returns An unsigned integer equal to the number of elements currently present in the span.
    constexpr size_t NumElements() const { return m_numElements; }

    /// Returns the size in bytes the Span represents.
    ///
    /// @returns An unsigned integer equal to the size in bytes the entire span represents.
    constexpr size_t SizeInBytes() const { return ElementSize() * m_numElements; }

    /// Returns true if the number of elements present in the vector is equal to zero.
    ///
    /// @returns True if the span is empty.
    constexpr bool IsEmpty() const { return (m_numElements == 0); }

    /// Returns a "subspan", a view over a subset range of the elements.
    ///
    /// @warning Behavior is undefined if either
    ///          - offset is greater than NumElements(), or
    ///          - count is not size_t(-1) and is greater than NumElements()-offset.
    ///
    /// Note that size_t(-1) is equivalent to C++20 std::dynamic_extent, which the C++20 std::span::subspan uses
    /// in the same way to mean "take the remainder of the elements from offset".
    ///
    /// @param offset Zero-based offset to start the subspan at
    /// @param count Number of elements in the subspan, or size_t(-1) for the remainder of the elements from offset
    ///
    /// @returns The subspan
    constexpr Span Subspan(
        size_t offset,
        size_t count) const
    {
        PAL_CONSTEXPR_ASSERT((offset <= NumElements())
                             && ((count == size_t(-1)) || (count <= NumElements() - offset)));
        if (count == size_t(-1))
        {
            count = NumElements() - offset;
        }
        return Span(Data() + offset, count);
    }

    /// Returns a subspan dropping the specified number (default 1) of elements from the front.
    /// Returns an empty Span if there were no more elements than that to start with.
    ///
    /// @param count Number of elements to drop from the front
    ///
    /// @returns The subspan
    constexpr Span DropFront(
        size_t count = 1) const
    {
        Span retVal;
        if (count < NumElements())
        {
            retVal = Subspan(count, size_t(-1));
        }
        return retVal;
    }

    /// Returns a subspan dropping the specified number (default 1) of elements from the back.
    /// Returns an empty Span if there were no more elements than that to start with.
    ///
    /// @param count Number of elements to drop from the back
    ///
    /// @returns The subspan
    constexpr Span DropBack(
        size_t count = 1) const
    {
        Span retVal;
        if (count < NumElements())
        {
            retVal = Subspan(0, NumElements() - count);
        }
        return retVal;
    }

    ///@{
    /// @internal Satisfies concept `range_expression`, using T* as `iterator` and 32-bit size and difference types
    ///
    /// @note - These are a convenience intended to be used by c++ language features such as `range for`.
    ///         These should not be called directly as they do not adhere to PAL coding standards.
    using value_type      = T;
    using reference       = T&;
    using iterator        = T*;
    using difference_type = size_t;
    using size_type       = size_t;

    constexpr iterator  begin()  const noexcept { return m_pData; }
    constexpr iterator  end()    const noexcept { return (m_pData + m_numElements); }
    constexpr bool      empty()  const noexcept { return IsEmpty(); }
    constexpr size_type size()   const noexcept { return m_numElements; }
    ///@}

protected:
    template<typename U = T, typename R = std::conditional_t<std::is_void_v<U>, char, U>>
    static constexpr size_t ElementSize() { return sizeof(R); }

    template<typename R = T, bool Condition = true>
    using IfConst = std::enable_if_t<std::is_const_v<R> == Condition>;

    template<typename R = T, bool Condition = true>
    using IfPtr   = std::enable_if_t<std::is_pointer_v<R> == Condition>;

    T*               m_pData;                  // Pointer to the current data.
    size_t           m_numElements;            // Number of elements present.
};

/// ====================================================================================================================
/// Span template specialization for const void byte buffers.
template<>
class Span<const void> : public Span<const char>
{
using Byte = const char;
using Base = Span<Byte>;

public:
    /// Constructor from nothing. This allows you to use {} to mean an empty Span.
    constexpr Span() : Base() {}

    /// Template constructor from any pointer and length
    ///
    /// @param [in] data Pointer to the start of the buffer or array
    /// @param numElements Number of bytes or elements in the buffer or array
    template<typename T>
    Span(const T* pData, size_t numElements) : Base(reinterpret_cast<Byte*>(pData), ElementSize<T>() * numElements) {}

    /// Template copy constructor
    ///
    /// @param [in] src Other Span<T> to copy from
    template<typename T = const void>
    Span(const Span<T>& src) : Span(src.Data(), src.NumElements()) {}

    /// Template constructor from any C++ array
    ///
    /// @param [in] src C++ array
    template<typename T, size_t NumElements>
    Span(const T(& src)[NumElements]) : Span(&src[0], NumElements) {}

    /// Constructor from any single element
    ///
    /// @param [in] src Single element
    template<typename T, typename Enabled = IfPtr<T, false>>
    Span(const T& src) : Span(&src, 1) {}

    /// Templated conversion of this typeless Span to a typed Subspan
    ///
    /// @returns A subspan with typed elemeents and NumElements truncated down to the nearest sizeof(type)
    template<typename T>
    operator Span<const T>() const
        { return Span<const T>(static_cast<const T*>(Data()), SizeInBytes() / ElementSize<T>()); }

    /// Returns the element at the location specified.
    ///
    /// @param [in] index Integer location of the element needed.
    ///
    /// @returns The element at location specified by index by reference
    template<typename T = Byte>
    const T& At(size_t index) const { return reinterpret_cast<const T&>(Base::At(index)); }

    /// Returns pointer to the underlying buffer serving as data storage.
    ///
    /// @returns Pointer to the underlying data storage.
    ///          For a non-empty span, the returned pointer contains address of the first element.
    ///          For an empty span, the returned pointer may or may not be a null pointer.
    template<typename T = void>
    constexpr const T* Data() const { return reinterpret_cast<const T*>(Base::Data()); }

    /// Returns a "subspan", a view over a subset range of the elements.
    ///
    /// @warning Behavior is undefined if either
    ///          - offset is greater than NumElements(), or
    ///          - count is not size_t(-1) and is greater than NumElements()-offset.
    ///
    /// Note that size_t(-1) is equivalent to C++20 std::dynamic_extent, which the C++20 std::span::subspan uses
    /// in the same way to mean "take the remainder of the elements from offset".
    ///
    /// @param offset Zero-based offset to start the subspan at
    /// @param count Number of elements in the subspan, or size_t(-1) for the remainder of the elements from offset
    ///
    /// @returns The subspan
    template<typename T = void>
    Span<const T> Subspan(size_t offset, size_t count) const { return Span<const void>(Base::Subspan(offset, count)); }

    /// Returns a subspan dropping the specified number (default 1) of elements from the front.
    /// Returns an empty Span if there were no more elements than that to start with.
    ///
    /// @param count Number of elements to drop from the front
    ///
    /// @returns The subspan
    template<typename T = void>
    Span<const T> DropFront(size_t count = 1) const { return Span<const void>(Base::DropFront(count)); }

    /// Returns a subspan dropping the specified number (default 1) of elements from the back.
    /// Returns an empty Span if there were no more elements than that to start with.
    ///
    /// @param count Number of elements to drop from the back
    ///
    /// @returns The subspan
    template<typename T = void>
    Span<const T> DropBack(size_t count = 1) const { return Span<const void>(Base::DropBack(count)); }
};

/// ====================================================================================================================
/// Span template specialization for mutable void byte buffers.
template<>
class Span<void> : public Span<char>
{
using Byte = char;
using Base = Span<Byte>;

public:
    /// Constructor from nothing. This allows you to use {} to mean an empty Span.
    constexpr Span() : Base() {}

    /// Template constructor from any pointer and length
    ///
    /// @param [in] data Pointer to the start of the buffer or array
    /// @param sizeInBytes,numElements Number of bytes or elements in the buffer or array
    template<typename T, typename Enabled = IfConst<T, false>>
    Span(T* pData, size_t numElements) : Base(reinterpret_cast<Byte*>(pData), ElementSize<T>() * numElements) { }

    /// Template copy constructor
    ///
    /// @param [in] src Other Span<T> to copy from
    template<typename T = void, typename Enabled = IfConst<T, false>>
    Span(const Span<T>& src) : Span(src.Data(), src.NumElements()) {}

    /// Template constructor from any C++ array
    ///
    /// @param [in] src C++ array
    template<typename T, size_t NumElements, typename Enabled = IfConst<T, false>>
    Span(T(& src)[NumElements]) : Span(&src[0], NumElements) {}

    /// Constructor from any single element
    ///
    /// @param [in] src Single element
    template<typename T, typename Enabled = IfConst<T, false>, typename = IfPtr<T, false>>
    Span(T& src) : Span(&src, 1) {}

    /// Templated conversion of this typeless Span to a typed Subspan
    ///
    /// @returns A subspan with typed elemeents and NumElements truncated down to the nearest sizeof(type)
    template<typename T>
    operator Span<T>() const { return Span<T>(static_cast<T*>(Data()), SizeInBytes() / ElementSize<T>()); }

    /// Returns the element at the location specified.
    ///
    /// @param [in] index Integer location of the element needed.
    ///
    /// @returns The element at location specified by index by reference
    template<typename T = Byte>
    T& At(size_t index) const { return reinterpret_cast<T&>(Base::At(index)); }

    /// Returns pointer to the underlying buffer serving as data storage
    ///
    /// @returns Pointer to the underlying data storage.
    ///          For a non-empty span, the returned pointer contains address of the first element.
    ///          For an empty span, the returned pointer may or may not be a null pointer.
    template<typename T = void>
    constexpr T* Data() const { return reinterpret_cast<T*>(Base::Data()); }

    /// Implicitly convert this void Span to its const void equivalent
    ///
    /// @returns The same span, but of const void type
    operator Span<const void>() const { return Span<const void>(Data(), NumElements()); }

    /// Returns a "subspan", a view over a subset range of the elements.
    ///
    /// @warning Behavior is undefined if either
    ///          - offset is greater than NumElements(), or
    ///          - count is not size_t(-1) and is greater than NumElements()-offset.
    ///
    /// Note that size_t(-1) is equivalent to C++20 std::dynamic_extent, which the C++20 std::span::subspan uses
    /// in the same way to mean "take the remainder of the elements from offset".
    ///
    /// @param offset Zero-based offset to start the subspan at
    /// @param count Number of elements in the subspan, or size_t(-1) for the remainder of the elements from offset
    ///
    /// @returns The subspan
    template<typename T = void>
    Span<T> Subspan(size_t offset, size_t count) const { return Span<void>(Base::Subspan(offset, count)); }

    /// Returns a subspan dropping the specified number (default 1) of elements from the front.
    /// Returns an empty Span if there were no more elements than that to start with.
    ///
    /// @param count Number of elements to drop from the front
    ///
    /// @returns The subspan
    template<typename T = void>
    Span<T> DropFront(size_t count = 1) const { return Span<void>(Base::DropFront(count)); }

    /// Returns a subspan dropping the specified number (default 1) of elements from the back.
    /// Returns an empty Span if there were no more elements than that to start with.
    ///
    /// @param count Number of elements to drop from the back
    ///
    /// @returns The subspan
    template<typename T = void>
    Span<T> DropBack(size_t count = 1) const { return Span<void>(Base::DropBack(count)); }
};

} // Util
