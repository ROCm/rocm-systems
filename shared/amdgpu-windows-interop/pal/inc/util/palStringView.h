/*
 ***********************************************************************************************************************
 *
 *  Copyright (c) 2021-2025 Advanced Micro Devices, Inc. All Rights Reserved.
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
* @file  palStringView.h
* @brief PAL utility collection string view declaration.
***********************************************************************************************************************
*/

#pragma once

#include "palAssert.h"
#include "palInlineFuncs.h"
#include "palStringUtil.h"
#include "palUtil.h"
#include <type_traits>

namespace Util
{

/**
***********************************************************************************************************************
* @brief String view.
*
* A StringView is a templated view over a constant contiguous sequence of characters.
*
* @warning The string view assumes that its lifetime does not extend past that of the pointed-to character sequence.
***********************************************************************************************************************
*/
template<typename CharT>
class StringView
{
    static_assert((std::is_same<CharT, char>::value || std::is_same<CharT, wchar_t>::value),
                  "StringView type T must be either char or wchar_t.");
public:
    /// Constructs an empty StringView.
    constexpr StringView()
        :
        m_pData{},
        m_length{}
    {}

    constexpr StringView(
        const CharT* s,
        uint32       count)
        :
        m_pData{s},
        m_length{count}
    {
        PAL_CONSTEXPR_ASSERT((s != nullptr) || (count == 0));
    }

    StringView(
        const CharT* s)
        :
        StringView()
    {
        if (s != nullptr)
        {
            m_length = StringLength(s);
            m_pData  = s;
        }
    }

    constexpr StringView(std::nullptr_t) = delete;

    ///@{
    /// Returns the element at the location specified.
    ///
    /// @warning Calling this function with an out-of-bounds index will cause an access violation!
    ///
    /// @param [in] index Integer location of the element needed.
    ///
    /// @returns The element at location specified by index by reference
    constexpr const CharT& At(uint32 index) const
    {
        PAL_CONSTEXPR_ASSERT(index < Length());
        return m_pData[index];
    }

    constexpr const CharT& operator[](uint32 index) const { return At(index); }
    ///@}

    /// Returns pointer to the underlying string serving as data storage.
    /// The returned pointer defines always valid range [Data(), Data() + Length()),
    /// even if the view does not point to any data storage (Data() is not dereferenceable in that case).
    ///
    /// @warning Dereferencing pointer returned by Data() from a view that does not point to a data storage will cause
    ///          an access violation!
    ///
    /// @returns Pointer to the underlying data storage for read access.
    ///          For a view to a valid data storage, the returned pointer contains address of the first element.
    ///          For a view without a valid data storage, the returned pointer will be a @c nullptr.
    constexpr const CharT* Data() const noexcept { return m_pData; }

    /// Returns the data at the front of the view.
    ///
    /// @warning Calling this function on an empty view will cause an access violation!
    ///
    /// @returns The data at the front of the view.
    constexpr const CharT& Front() const
    {
        PAL_CONSTEXPR_ASSERT(IsEmpty() == false);
        return m_pData[0];
    }

    /// Returns the data at the back of the view.
    ///
    /// @warning Calling this function on an empty view will cause an access violation!
    ///
    /// @returns The data at the back of the view.
    constexpr const CharT& Back() const
    {
        PAL_CONSTEXPR_ASSERT(IsEmpty() == false);
        return m_pData[Length() - 1];
    }

    /// Returns the length of the string.
    ///
    /// @returns An unsigned integer equal to the length of the string.
    constexpr uint32 Length() const { return m_length; }

    /// Returns true if the number of characters the view points to is equal to zero.
    ///
    /// @returns True if the view points to an empty or non-existing data storage.
    constexpr bool IsEmpty() const { return (m_length == 0); }

    ///@{
    /// @internal Satisfies concept `range_expression`, using CharT* as `iterator`.
    ///
    /// @note - These are a convenience intended to be used by C++ language features such as `range-based for`.
    ///         These should not be called directly as they do not adhere to PAL coding standards.
    using const_iterator = const CharT*;

    constexpr const_iterator begin() const noexcept { return m_pData; }
    constexpr const_iterator end()   const noexcept { return m_pData + Length(); }
    ///@}

private:
    const CharT* m_pData;
    uint32       m_length;
};

// =====================================================================================================================
template<typename CharT>
constexpr bool operator==(
    StringView<CharT> x,
    StringView<CharT> y)
{
    bool equal = (x.Length() == y.Length());
    if (equal)
    {
        if (x.Data() != y.Data())
        {
            // they are not pointing to the same storage, so we need to compare the contents
            for (uint32 index = 0; equal && (index < x.Length()); ++index)
            {
                equal = (x[index] == y[index]);
            }
        }
    }
    return equal;
}

// =====================================================================================================================
template<typename CharT>
constexpr bool operator!=(StringView<CharT> x, StringView<CharT> y) { return (x == y) == false; }

// =====================================================================================================================
template<typename CharT>
bool operator<(
    StringView<CharT> x,
    StringView<CharT> y)
{
    const uint32 minLength = Min(x.Length(), y.Length());
    int compare            = strncmp(x.Data(), y.Data(), minLength);
    if (compare == 0)
    {
        // strings are equal up to minLength, so check which is shorter
        compare = int(x.Length()) - int(y.Length());
    }

    return compare < 0;
}

// =====================================================================================================================
template<typename CharT>
bool operator<=(StringView<CharT> x, StringView<CharT> y) { return (y < x) == false; }

// =====================================================================================================================
template<typename CharT>
bool operator>(StringView<CharT> x, StringView<CharT> y) { return y < x; }

// =====================================================================================================================
template<typename CharT>
bool operator>=(StringView<CharT> x, StringView<CharT> y) { return (x < y) == false; }

/// Specialization of @ref HashString(const char*,size_t) for @ref StringView.
template<typename T>
constexpr uint32 HashString(
    StringView<T> sv)
{
    return HashString(sv.Data(), sv.Length());
}

} // Util
