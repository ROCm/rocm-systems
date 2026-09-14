/*
 ***********************************************************************************************************************
 *
 *  Copyright (c) Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
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

// pal
#include "palUtil.h"
#include "palAssert.h"
#include "palConcepts.h"
#include "palSysMemory.h"
#include "palInlineFuncs.h"

// stl
#include <type_traits>

namespace Util
{
template<typename T> class Span;
class ByteSpan;
class ConstByteSpan;

///@{
/// @internal Implementation namespace for @ref Span.
namespace _detail
{
// A simple tag type which invoke's Span's hidden "I promise my length is zero if my pointer is null" constructor.
struct NullInvariantHolds{};

// Satisfied by any Span, including the derived ones.
// This is used to prevent accidentally constructing a Span<Span<T>> when a copy-construction is intended.
template<typename T>
concept IsSpan = requires { typename T::value_type; } && std::is_base_of_v<Span<typename T::value_type>, T>;
}
///@}

/**
 ***********************************************************************************************************************
 * @brief Span container
 *
 * Span is an array with a length, where the data is not owned by the Span object. It is similar to C++20 std::span,
 * but only the dynamic extent variant. It is similar to LLVM MutableArrayRef and ArrayRef. A Span is intended to
 * be passed around by value.
 *
 * Note that it's impossible to construct a Span with a null pointer and a non-zero element count; all Spans with null
 * pointers are normalized to a count of zero. This means that "IsEmpty() == false" implies that both the pointer is
 * non-null and the count is non-zero. HasData() is provided as a convenient alias for "IsEmpty() == false".
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
    /// @note The length of the span is overridden to zero if the pointer is a nullptr. This invariant holds for all
    ///       Spans. Some constructors satisfy this invariant by definition, they should be preferred over this generic
    ///       constructor as they will produce faster code.
    ///
    /// @param [in] pData        Pointer to the start of the array
    /// @param [in] numElements  Number of elements in the array
    constexpr Span(T* pData, size_t numElements)
        : m_pData(pData), m_numElements((pData != nullptr) ? numElements : 0) {}

    /// Constructor from C++ array
#if PAL_CLIENT_INTERFACE_MAJOR_VERSION >= 1009
    ///
    /// @note Implicit for all types except Characters.
    ///       This is to discourage accidentally constructing a Span from a string literal.
    ///       You can still explicitly construct one by invoking the ctor directly i.e. Span<char>("hello").
    ///       If you're dealing with strings, you should consider StringView<char> instead.
#endif
    ///
    /// @param [in] src C++ array
    template<size_t NumElements>
#if PAL_CLIENT_INTERFACE_MAJOR_VERSION >= 1009
    explicit(Concepts::Character<T>)
#endif
    constexpr Span(T(& src)[NumElements]) : m_pData(&src[0]), m_numElements(NumElements) {}

    /// Constructor from single element
    ///
    /// @param [in] src Single element
    constexpr Span(T& src) : m_pData(&src), m_numElements(1) {}

    /// Implicitly convert a Span to its const-element equivalent.
    ///
    /// @returns The same span, but with const element type
    constexpr operator Span<const T>() const
    {
        // Any existing Span must satisfy the null data invariant so we can call the optimized constructor.
        return Span<const T>(m_pData, m_numElements, _detail::NullInvariantHolds{});
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
        PAL_CONSTEXPR_ASSERT(HasData());
        return *m_pData;
    }

    /// Returns the data at the back of the vector.
    ///
    /// @returns The data at the back of the vector.
    constexpr T& Back() const
    {
        PAL_CONSTEXPR_ASSERT(HasData());
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

    /// Returns true if the number of elements present in this Span is equal to zero.
    ///
    /// @note If this returns false, it implies that m_pData is non-null. This must be true because the Span
    ///       constructors force m_numElements to zero if m_pData is null.
    ///
    /// @returns True if the span is empty.
    constexpr bool IsEmpty() const { return (m_numElements == 0); }

    /// Returns true if the number of elements present in this Span is non-zero and if its data pointer is non-null.
    ///
    /// This works because the Span constructors force m_numElements to zero if m_pData is null. Thus if this Span is
    /// not empty (m_numElements != 0) then it must be the case that m_pData is not null.
    ///
    /// @returns True if the span contains valid data.
    constexpr bool HasData() const { return (IsEmpty() == false); }

    /// Sets this Span to an empty Span, meaning its pointer is null and its number of elements is zero.
    void Clear()
    {
        m_pData       = nullptr;
        m_numElements = 0;
    }

    /// Returns a "subspan", a view over a subset range of the elements.
    ///
    /// The "offset" and "count" are both clamped to ensure that the subspan stays in bounds. If "offset" is greater
    /// than or equal to the length of this span, then an empty span is returned.
    ///
    /// Note that count = size_t(-1) is equivalent to C++20 std::dynamic_extent, which the C++20 std::span::subspan
    /// uses in the same way to mean "take the remainder of the elements from offset". This is a natural result of
    /// the boundary clamping logic.
    ///
    /// @param offset  Zero-based offset to start the subspan at
    /// @param count   Number of elements in the subspan, or size_t(-1) for the remainder of the elements from offset
    ///
    /// @returns The subspan
    constexpr Span Subspan(
        size_t offset,
        size_t count) const
    {
        // The offset must be clamped first or we might underflow in the count clamping logic.
        if (offset > NumElements())
        {
            offset = NumElements();
        }
        if (count > NumElements() - offset)
        {
            count = NumElements() - offset;
        }

        // Any existing Span must satisfy the null data invariant so we can call the optimized constructor.
        return Span(Data() + offset, count, _detail::NullInvariantHolds{});
    }

    /// Returns a subspan dropping the specified number (default 1) of elements from the front.
    /// Returns an empty Span if there were no more elements than that to start with.
    ///
    /// @param count Number of elements to drop from the front
    ///
    /// @returns The subspan
    constexpr Span DropFront(size_t count = 1) const { return Subspan(count, size_t(-1)); }

    /// Returns a subspan dropping the specified number (default 1) of elements from the back.
    /// Returns an empty Span if there were no more elements than that to start with.
    ///
    /// @param count Number of elements to drop from the back
    ///
    /// @returns The subspan
    constexpr Span DropBack(
        size_t count = 1) const
    {
        // Unlike DropFront, this one needs a manual bounds check to avoid underflowing the count.
        return Subspan(0, (NumElements() > count) ? (NumElements() - count) : 0);
    }

    /// Assigns the elements in this Span to the values in the "src" Span. This modifies this Span's underlying data,
    /// not the Span object itself.
    ///
    /// The Spans may have different lengths; the lesser of the two lengths determines how many elements are copied.
    ///
    /// @warning The two Spans absolutely must **not** overlap!
    ///
    /// @param [in] src  Specifies the address and length of the source range.
    void AssignSpan(
        Span<const T> src) const
    {
        static_assert(std::is_const_v<T> == false, "AssignSpan is only available on non-const data!");

        // No buffer overflows! Clamp the count down to prevent any out of bounds accesses.
        const size_t  count    = Min(NumElements(), src.NumElements());
        T*const       pDstData = Data();
        const T*const pSrcData = src.Data();

        // It's illegal to use AssignSpan on overlapping Spans. We don't need an early return here as executing an
        // overlapping assignment may corrupt data but won't write out of bounds. If someone finds a use-case for
        // overlapping Span assignments we can add a new AssignSpanOverlapping function.
        PAL_ASSERT(VoidPtrsOverlap(pSrcData, pDstData, ElementSize<T>() * count) == false);

        // This memcpy is an optimization to avoid generating a slower assignment loop which handles overlapping data.
        // Note that by C++ spec definition only trivially copyable types are safe to use with memcpy.
        if constexpr (std::is_trivially_copyable_v<T>)
        {
            // Note that Span's constructor guarantees that NumElements() is zero if Data() is null. This means that
            // we only need to check the clamped count to avoid passing null to memcpy (that's undefined behavior).
            if (count != 0)
            {
                // banned_function_exemption: memcpy
                // This case is exempt from the coding standards memcpy restriction as this function clearly labels
                // the use-case (copying arrays known sizes) and has sufficient bounds checking logic to prevent all
                // possible out-of-bounds accesses.
                memcpy(pDstData, pSrcData, ElementSize<T>() * count);
            }
        }
        else
        {
            // Note that Span's constructor guarantees that NumElements() is zero if Data() is null. This means that
            // this loop's boundary check is also an implicit nullptr check on both Spans.
            for (size_t idx = 0; idx < count; ++idx)
            {
                pDstData[idx] = pSrcData[idx];
            }
        }
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
    // An internal constructor which doesn't do the "(pData != nullptr) ? numElements : 0" validation.
    // This is an optimization for cases where we're guaranteed that this invariant must already hold.
    constexpr Span(T* pData, size_t numElements, _detail::NullInvariantHolds)
        : m_pData(pData), m_numElements(numElements) {}

    // Let Spans call each other's internal constructors.
    template<typename U>
    friend class Span;
    friend class ByteSpan;
    friend class ConstByteSpan;

    template<typename U = T, typename R = std::conditional_t<std::is_void_v<U>, char, U>>
    static constexpr size_t ElementSize() { return sizeof(R); }

    template<typename R = T, bool Condition = true>
    using IfConst = std::enable_if_t<std::is_const_v<R> == Condition>;

    template<typename R = T, bool Condition = true>
    using IfPtr   = std::enable_if_t<std::is_pointer_v<R> == Condition>;

    T*     m_pData;       // Pointer to the current data.
    size_t m_numElements; // Number of elements present.
};

/**
 ***********************************************************************************************************************
 * @brief Span template specialization for const void byte buffers.
 ***********************************************************************************************************************
 */
template<>
class Span<const void> : public Span<const char>
{
public:
    using Byte = const char;
    using Base = Span<Byte>;

    /// Constructor from nothing. This allows you to use {} to mean an empty Span.
    Span() : Base() {}

    /// Template constructor from any pointer and length
    ///
    /// @param [in] pData        Pointer to the start of the array
    /// @param [in] numElements  Number of elements in the array
    template<typename T>
    Span(const T* pData, size_t numElements) : Base(reinterpret_cast<Byte*>(pData), ElementSize<T>() * numElements)
    {
        // The C++ spec only lets us manipulate a type's underlying "object representation" if it's trivially copyable.
        static_assert(std::is_trivially_copyable_v<T> || std::is_void_v<T>,
                      "Only trivially copyable types can be stored as a sequence of bytes!");
    }

    /// Default copy constructor
    ///
    /// @param [in] src Other Span<const void> to copy from
    Span(const Span<const void>& src) = default;

    /// Template copy constructor
    ///
    /// @param [in] src Other Span<T> to copy from
    template<typename T>
    Span(const Span<T>& src) : Span(src.Data(), src.NumElements(), _detail::NullInvariantHolds{}) {}

    /// Template constructor from any C++ array
    ///
    /// @param [in] src C++ array
    template<typename T, size_t NumElements>
    Span(const T(& src)[NumElements]) : Span(&src[0], NumElements, _detail::NullInvariantHolds{}) {}

    /// Constructor from any single value
    ///
    /// @param [in] src Single value
    template<typename T, typename Enabled = IfPtr<T, false>,
             typename = std::enable_if_t<std::is_same_v<T, Span<const void>> == false>>
    explicit Span(const T& src) : Span(&src, 1, _detail::NullInvariantHolds{}) {}

    /// Returns pointer to the underlying buffer serving as data storage.
    ///
    /// @returns Pointer to the underlying data storage.
    ///          For a non-empty span, the returned pointer contains address of the first byte.
    ///          For an empty span, the returned pointer may or may not be a null pointer.
    const void* Data() const { return Base::Data(); }

    /// Returns a "subspan", a view over a subset range of bytes.
    ///
    /// The "offset" and "count" are both clamped to ensure that the subspan stays in bounds. If "offset" is greater
    /// than or equal to the length of this span, then an empty span is returned.
    ///
    /// Note that count = size_t(-1) is equivalent to C++20 std::dynamic_extent, which the C++20 std::span::subspan
    /// uses in the same way to mean "take the remainder of the bytes from offset". This is a natural result of
    /// the boundary clamping logic.
    ///
    /// @param offset  Zero-based byte offset to start the subspan at
    /// @param count   Number of bytes in the subspan, or size_t(-1) for the remainder of bytes from offset
    ///
    /// @returns The subspan
    Span<const void> Subspan(size_t offset, size_t count) const
        { return Span<const void>(Base::Subspan(offset, count)); }

    /// Returns a subspan dropping the specified number (default 1) of bytes from the front.
    /// Returns an empty Span if there were no more bytes than that to start with.
    ///
    /// @param count Number of bytes to drop from the front
    ///
    /// @returns The subspan
    Span<const void> DropFront(size_t count = 1) const { return Span<const void>(Base::DropFront(count)); }

    /// Returns a subspan dropping the specified number (default 1) of bytes from the back.
    /// Returns an empty Span if there were no more bytes than that to start with.
    ///
    /// @param count Number of bytes to drop from the back
    ///
    /// @returns The subspan
    Span<const void> DropBack(size_t count = 1) const { return Span<const void>(Base::DropBack(count)); }

    /// Equivalent to a static_cast<T*> on this Span's Data() pointer with an added bounds-check. If this Span isn't
    /// large enough to fit a "T" object then nullptr is returned The caller must check for null before dereferencing.
    ///
    /// @returns A pointer interpreting this Span as the given type, or nullptr if this Span is too small.
    template<typename T>
    const T* Cast() const
    {
        // The C++ spec only lets us manipulate a type's underlying "object representation" if it's trivially copyable.
        static_assert(std::is_trivially_copyable_v<T> || std::is_void_v<T>,
                      "Only trivially copyable types can be stored as a sequence of bytes!");

        // Even if "T" is trivially copyable, it's still undefined behavior to do this cast if we don't meet alignment.
        PAL_ASSERT(VoidPtrIsPow2Aligned(Data(), alignof(T)));

        return (sizeof(T) <= SizeInBytes()) ? static_cast<const T*>(Data()) : nullptr;
    }

    /// Templated conversion of this typeless Span to a typed subspan. Equivalent to a static_cast<T*> on this Span's
    /// Data() pointer bundled with a NumElements units conversion.
    ///
    /// Note that any trailing bytes at the end of this span which cannot form a full "T" element are dropped. This can
    /// result in an empty span even if the original span was non-empty.
    ///
    /// @returns A subspan with typed elements and NumElements truncated down to the nearest sizeof(T).
    template<typename T>
    Span<const T> CastSpan() const
    {
        // The C++ spec only lets us manipulate a type's underlying "object representation" if it's trivially copyable.
        static_assert(std::is_trivially_copyable_v<T> || std::is_void_v<T>,
                      "Only trivially copyable types can be stored as a sequence of bytes!");

        // Even if "T" is trivially copyable, it's still undefined behavior to do this cast if we don't meet alignment.
        PAL_ASSERT(VoidPtrIsPow2Aligned(Data(), alignof(T)));

        // Any existing Span must satisfy the null data invariant so we can call the optimized constructor.
        return Span<const T>(static_cast<const T*>(Data()), SizeInBytes() / ElementSize<T>(),
                             _detail::NullInvariantHolds{});
    }

private:
    // An internal constructor which doesn't do the "(pData != nullptr) ? numElements : 0" validation.
    // This is an optimization for cases where we're guaranteed that this invariant must already hold.
    template<typename T>
    Span(const T* pData, size_t numElements, _detail::NullInvariantHolds)
        : Base(reinterpret_cast<Byte*>(pData), ElementSize<T>() * numElements, _detail::NullInvariantHolds{})
    {
        // The C++ spec only lets us manipulate a type's underlying "object representation" if it's trivially copyable.
        static_assert(std::is_trivially_copyable_v<T> || std::is_void_v<T>,
                      "Only trivially copyable types can be stored as a sequence of bytes!");
    }

    // Let Spans call each other's internal constructors.
    template<typename U>
    friend class Span;
};

/**
 ***********************************************************************************************************************
 * @brief Span template specialization for mutable void byte buffers.
 ***********************************************************************************************************************
 */
template<>
class Span<void> : public Span<char>
{
public:
    using Byte = char;
    using Base = Span<Byte>;

    /// Constructor from nothing. This allows you to use {} to mean an empty Span.
    Span() : Base() {}

    /// Template constructor from any pointer and length
    ///
    /// @param [in] pData        Pointer to the start of the array
    /// @param [in] numElements  Number of elements in the array
    template<typename T, typename Enabled = IfConst<T, false>>
    Span(T* pData, size_t numElements) : Base(reinterpret_cast<Byte*>(pData), ElementSize<T>() * numElements)
    {
        // The C++ spec only lets us manipulate a type's underlying "object representation" if it's trivially copyable.
        static_assert(std::is_trivially_copyable_v<T> || std::is_void_v<T>,
                      "Only trivially copyable types can be stored as a sequence of bytes!");
    }

    /// Default copy constructor
    ///
    /// @param [in] src Other Span<void> to copy from
    Span(const Span<void>& src) = default;

    /// Template copy constructor
    ///
    /// @param [in] src Other Span<T> to copy from
    template<typename T, typename Enabled = IfConst<T, false>>
    Span(const Span<T>& src) : Span(src.Data(), src.NumElements(), _detail::NullInvariantHolds{}) {}

    /// Template constructor from any C++ array
    ///
    /// @param [in] src C++ array
    template<typename T, size_t NumElements, typename Enabled = IfConst<T, false>>
    Span(T(& src)[NumElements]) : Span(&src[0], NumElements, _detail::NullInvariantHolds{}) {}

    /// Constructor from any single value
    ///
    /// @param [in] src Single value
    template<typename T, typename Enabled = IfConst<T, false>, typename = IfPtr<T, false>,
             typename = std::enable_if_t<std::is_same_v<T, Span<void>> == false>>
    explicit Span(T& src) : Span(&src, 1, _detail::NullInvariantHolds{}) {}

    /// Implicitly convert this void Span to its const void equivalent
    ///
    /// @returns The same span, but of const void type
    operator Span<const void>() const
    {
        // Any existing Span must satisfy the null data invariant so we can call the optimized constructor.
        return Span<const void>(Data(), NumElements(), _detail::NullInvariantHolds{});
    }

    /// Returns pointer to the underlying buffer serving as data storage
    ///
    /// @returns Pointer to the underlying data storage.
    ///          For a non-empty span, the returned pointer contains address of the first byte.
    ///          For an empty span, the returned pointer may or may not be a null pointer.
    void* Data() const { return Base::Data(); }

    /// Returns a "subspan", a view over a subset range of bytes.
    ///
    /// The "offset" and "count" are both clamped to ensure that the subspan stays in bounds. If "offset" is greater
    /// than or equal to the length of this span, then an empty span is returned.
    ///
    /// Note that count = size_t(-1) is equivalent to C++20 std::dynamic_extent, which the C++20 std::span::subspan
    /// uses in the same way to mean "take the remainder of the bytes from offset". This is a natural result of
    /// the boundary clamping logic.
    ///
    /// @param offset  Zero-based offset to start the subspan at
    /// @param count   Number of bytes in the subspan, or size_t(-1) for the remainder of the bytes from offset
    ///
    /// @returns The subspan
    Span<void> Subspan(size_t offset, size_t count) const { return Span<void>(Base::Subspan(offset, count)); }

    /// Returns a subspan dropping the specified number (default 1) of bytes from the front.
    /// Returns an empty Span if there were no more bytes than that to start with.
    ///
    /// @param count Number of bytes to drop from the front
    ///
    /// @returns The subspan
    Span<void> DropFront(size_t count = 1) const { return Span<void>(Base::DropFront(count)); }

    /// Returns a subspan dropping the specified number (default 1) of bytes from the back.
    /// Returns an empty Span if there were no more bytes than that to start with.
    ///
    /// @param count Number of bytes to drop from the back
    ///
    /// @returns The subspan
    Span<void> DropBack(size_t count = 1) const { return Span<void>(Base::DropBack(count)); }

    /// Assigns the bytes in this Span to the bytes in the "src" Span. This modifies this Span's underlying data,
    /// not the Span object itself.
    ///
    /// The Spans may have different lengths; the lesser of the two lengths determines how many bytes are copied.
    ///
    /// @warning The two Spans absolutely must **not** overlap!
    ///
    /// @param [in] src  Specifies the address and length of the source range.
    void AssignSpan(Span<const void> src) const { Base::AssignSpan(src); }

    /// Fills the bytes in this Span with the given 8-bit pattern. This modifies this Span's underlying data,
    /// not the Span object itself. This function does nothing if this Span is empty.
    void Memset(uint8 pattern) const
    {
        // Note that it's illegal to pass a null pointer into memset, even is the count is zero.
        if (HasData())
        {
            memset(Data(), pattern, SizeInBytes());
        }
    }

    /// Equivalent to a static_cast<T*> on this Span's Data() pointer with an added bounds-check. If this Span isn't
    /// large enough to fit a "T" object then nullptr is returned The caller must check for null before dereferencing.
    ///
    /// @returns A pointer interpreting this Span as the given type, or nullptr if this Span is too small.
    template<typename T>
    T* Cast() const
    {
        // The C++ spec only lets us manipulate a type's underlying "object representation" if it's trivially copyable.
        static_assert(std::is_trivially_copyable_v<T> || std::is_void_v<T>,
                      "Only trivially copyable types can be stored as a sequence of bytes!");

        // Even if "T" is trivially copyable, it's still undefined behavior to do this cast if we don't meet alignment.
        PAL_ASSERT(VoidPtrIsPow2Aligned(Data(), alignof(T)));

        return (sizeof(T) <= SizeInBytes()) ? static_cast<T*>(Data()) : nullptr;
    }

    /// Templated conversion of this typeless Span to a typed subspan. Equivalent to a static_cast<T*> on this Span's
    /// Data() pointer bundled with a NumElements units conversion.
    ///
    /// Note that any trailing bytes at the end of this span which cannot form a full "T" element are dropped. This can
    /// result in an empty span even if the original span was non-empty.
    ///
    /// @returns A subspan with typed elements and NumElements truncated down to the nearest sizeof(T).
    template<typename T>
    Span<T> CastSpan() const
    {
        // The C++ spec only lets us manipulate a type's underlying "object representation" if it's trivially copyable.
        static_assert(std::is_trivially_copyable_v<T> || std::is_void_v<T>,
                      "Only trivially copyable types can be stored as a sequence of bytes!");

        // Even if "T" is trivially copyable, it's still undefined behavior to do this cast if we don't meet alignment.
        PAL_ASSERT(VoidPtrIsPow2Aligned(Data(), alignof(T)));

        // Any existing Span must satisfy the null data invariant so we can call the optimized constructor.
        return Span<T>(static_cast<T*>(Data()), SizeInBytes() / ElementSize<T>(), _detail::NullInvariantHolds{});
    }

private:
    // An internal constructor which doesn't do the "(pData != nullptr) ? numElements : 0" validation.
    // This is an optimization for cases where we're guaranteed that this invariant must already hold.
    template<typename T>
    Span(T* pData, size_t numElements, _detail::NullInvariantHolds)
        : Base(reinterpret_cast<Byte*>(pData), ElementSize<T>() * numElements, _detail::NullInvariantHolds{})
    {
        // The C++ spec only lets us manipulate a type's underlying "object representation" if it's trivially copyable.
        static_assert(std::is_trivially_copyable_v<T> || std::is_void_v<T>,
                      "Only trivially copyable types can be stored as a sequence of bytes!");
    }

    // Let Spans call each other's internal constructors.
    template<typename U>
    friend class Span;
};

/**
 ***********************************************************************************************************************
 * @brief Span derived class for immutable byte buffers.
 ***********************************************************************************************************************
 */
class ConstByteSpan : public Span<const std::byte>
{
public:
    using Base = Span<const std::byte>;

    /// Factory constructor from any pointer and length.
    ///
    /// @param [in] pData        Pointer to the start of the array
    /// @param [in] numElements  Number of elements in the array
    ///
    /// @returns newly constructed span over the data
    template<typename T>
    static ConstByteSpan FromArray(const T* pData, size_t numElements)
    { return Base(reinterpret_cast<const std::byte*>(pData), numElements * sizeof(T)); }

    /// Factory constructor from any pointer and length.
    ///
    /// @param [in] pData        Pointer to the start of the array
    /// @param [in] numBytes     Number of bytes in the array
    ///
    /// @returns newly constructed span over the data
    static ConstByteSpan FromBytes(const void* pData, size_t numBytes)
    { return Base(reinterpret_cast<const std::byte*>(pData), numBytes); }

    /// Constructor from nothing. This allows you to use {} to mean an empty Span.
    constexpr ConstByteSpan() : Base() {}

    /// Copy constructor from another Span.
    ///
    /// @param [in] src Other Span<T> to view the bytes of
    template<typename T>
    constexpr ConstByteSpan(const Span<T>& src)
        : ConstByteSpan(src.Data(), src.NumElements(), _detail::NullInvariantHolds{}) {}

    /// Template constructor from any array
    ///
    /// @param [in] src array
    template<typename T, size_t NumElements>
    constexpr ConstByteSpan(const T(& src)[NumElements])
        : ConstByteSpan(src, NumElements, _detail::NullInvariantHolds{}) {}

    /// Constructor from any single value
    ///
    /// @param [in] src Single value
    template<typename T>
    requires ((std::is_pointer_v<T> == false) && (_detail::IsSpan<T> == false))
    explicit constexpr ConstByteSpan(const T& src) : ConstByteSpan(&src, 1, _detail::NullInvariantHolds{}) {}

    /// Returns a "subspan", a view over a subset range of bytes.
    ///
    /// The "offset" and "count" are both clamped to ensure that the subspan stays in bounds. If "offset" is greater
    /// than or equal to the length of this span, then an empty span is returned.
    ///
    /// Note that count = size_t(-1) is equivalent to C++20 std::dynamic_extent, which the C++20 std::span::subspan
    /// uses in the same way to mean "take the remainder of the bytes from offset". This is a natural result of
    /// the boundary clamping logic.
    ///
    /// @param offset  Zero-based offset to start the subspan at
    /// @param count   Number of bytes in the subspan, or size_t(-1) for the remainder of the bytes from offset
    ///
    /// @returns The subspan
    constexpr ConstByteSpan Subspan(size_t offset, size_t count) const { return Base::Subspan(offset, count); }

    /// Returns a subspan dropping the specified number (default 1) of bytes from the front.
    /// Returns an empty Span if there were no more bytes than that to start with.
    ///
    /// @param count Number of bytes to drop from the front
    ///
    /// @returns The subspan
    constexpr ConstByteSpan DropFront(size_t count = 1) const { return Base::DropFront(count); }

    /// Returns a subspan dropping the specified number (default 1) of bytes from the back.
    /// Returns an empty Span if there were no more bytes than that to start with.
    ///
    /// @param count Number of bytes to drop from the back
    ///
    /// @returns The subspan
    constexpr ConstByteSpan DropBack(size_t count = 1) const { return Base::DropBack(count); }

    /// Equivalent to a reinterpret_cast<const T*> on this Span's Data() pointer with an added bounds-check.
    /// If this Span isn't large enough to fit a "T" object then nullptr is returned.
    /// The caller must check for null before dereferencing.
    ///
    /// @returns A pointer interpreting this Span as the given type, or nullptr if this Span is too small.
    template<typename T>
    const T* Cast() const
    {
        // The C++ spec only lets us manipulate a type's underlying "object representation" if it's trivially copyable.
        static_assert(std::is_trivially_copyable_v<T> || std::is_void_v<T>,
                      "Only trivially copyable types can be stored as a sequence of bytes!");

        // Even if "T" is trivially copyable, it's still undefined behavior to do this cast if we don't meet alignment.
        // Void is exempt because it has no alignment requirement.
        if constexpr (std::is_void_v<T> == false)
        {
            PAL_ASSERT(VoidPtrIsPow2Aligned(Data(), alignof(T)));
        }

        return (ElementSize<T>() <= SizeInBytes()) ? reinterpret_cast<const T*>(Data()) : nullptr;
    }

    /// Templated conversion of this typeless Span to a typed subspan.
    /// Equivalent to a reinterpret_cast<const T*> on this Span's Data() pointer bundled with a NumElements units
    /// conversion.
    ///
    /// Any trailing bytes at the end of this span which cannot form a full "T" element are dropped.
    /// This can result in an empty span even if the original span was non-empty.
    ///
    /// @returns A subspan with typed elements and NumElements truncated down to the nearest sizeof(T).
    template<typename T>
    Span<const T> CastSpan() const
    {
        // The C++ spec only lets us manipulate a type's underlying "object representation" if it's trivially copyable.
        static_assert(std::is_trivially_copyable_v<T>,
                      "Only trivially copyable types can be stored as a sequence of bytes!");

        // Even if "T" is trivially copyable, it's still undefined behavior to do this cast if we don't meet alignment.
        PAL_ASSERT(VoidPtrIsPow2Aligned(Data(), alignof(T)));

        // Any existing Span must satisfy the null data invariant so we can call the optimized constructor.
        return Span<const T>(reinterpret_cast<const T*>(Data()), SizeInBytes() / ElementSize<T>(),
                             _detail::NullInvariantHolds{});
    }

private:
    // An internal constructor which doesn't do the "(pData != nullptr) ? numElements : 0" validation.
    // This is an optimization for cases where we're guaranteed that this invariant must already hold.
    template<typename T>
    ConstByteSpan(const T* pData, size_t numElements, _detail::NullInvariantHolds)
        : Base(reinterpret_cast<const std::byte*>(pData), ElementSize<T>() * numElements, _detail::NullInvariantHolds{})
    {
        // The C++ spec only lets us manipulate a type's underlying "object representation" if it's trivially copyable.
        // Void is permitted because it is already a typeless view of raw bytes.
        static_assert(std::is_trivially_copyable_v<T> || std::is_void_v<T>,
                      "Only trivially copyable types can be stored as a sequence of bytes!");
    }

    friend class ByteSpan;
};

/**
 ***********************************************************************************************************************
 * @brief Span derived class for mutable byte buffers.
 ***********************************************************************************************************************
 */
class ByteSpan : public Span<std::byte>
{
public:
    using Base = Span<std::byte>;

    /// Factory constructor from any pointer and length.
    ///
    /// @param [in] pData        Pointer to the start of the array
    /// @param [in] numElements  Number of elements in the array
    ///
    /// @returns newly constructed span over the data
    template<Concepts::NonConst T>
    static ByteSpan FromArray(T* pData, size_t numElements)
    { return Base(reinterpret_cast<std::byte*>(pData), numElements * sizeof(T)); }

    /// Factory constructor from any pointer and length.
    ///
    /// @param [in] pData        Pointer to the start of the array
    /// @param [in] numBytes     Number of bytes in the array
    ///
    /// @returns newly constructed span over the data
    static ByteSpan FromBytes(void* pData, size_t numBytes)
    { return Base(reinterpret_cast<std::byte*>(pData), numBytes); }

    /// Constructor from nothing. This allows you to use {} to mean an empty Span.
    constexpr ByteSpan() : Base() {}

    /// Template constructor from any Span
    ///
    /// @param [in] src Other Span<T> to view the bytes of
    template<Concepts::NonConst T>
    constexpr ByteSpan(const Span<T>& src) : ByteSpan(src.Data(), src.NumElements(), _detail::NullInvariantHolds{}) {}

    /// Template constructor from any C++ array
    ///
    /// @param [in] src C++ array
    template<Concepts::NonConst T, size_t NumElements>
    constexpr ByteSpan(T(& src)[NumElements]) : ByteSpan(src, NumElements, _detail::NullInvariantHolds{}) {}

    /// Constructor from any single value
    ///
    /// @param [in] src Single value
    template<Concepts::NonConst T>
    requires ((std::is_pointer_v<T> == false) && (_detail::IsSpan<T> == false))
    constexpr explicit ByteSpan(T& src) : ByteSpan(&src, 1, _detail::NullInvariantHolds{}) {}

    /// Implicitly convert this ByteSpan to its read-only equivalent.
    ///
    /// @returns The same span, but of const bytes
    operator ConstByteSpan() const
    {
        // Any existing Span must satisfy the null data invariant so we can call the optimized constructor.
        return ConstByteSpan(Data(), NumElements(), _detail::NullInvariantHolds{});
    }

    /// Returns a "subspan", a view over a subset range of bytes.
    ///
    /// The "offset" and "count" are both clamped to ensure that the subspan stays in bounds. If "offset" is greater
    /// than or equal to the length of this span, then an empty span is returned.
    ///
    /// Note that count = size_t(-1) is equivalent to C++20 std::dynamic_extent, which the C++20 std::span::subspan
    /// uses in the same way to mean "take the remainder of the bytes from offset". This is a natural result of
    /// the boundary clamping logic.
    ///
    /// @param offset  Zero-based offset to start the subspan at
    /// @param count   Number of bytes in the subspan, or size_t(-1) for the remainder of the bytes from offset
    ///
    /// @returns The subspan
    constexpr ByteSpan Subspan(size_t offset, size_t count) const { return Base::Subspan(offset, count); }

    /// Returns a subspan dropping the specified number (default 1) of bytes from the front.
    /// Returns an empty Span if there were no more bytes than that to start with.
    ///
    /// @param count Number of bytes to drop from the front
    ///
    /// @returns The subspan
    constexpr ByteSpan DropFront(size_t count = 1) const { return Base::DropFront(count); }

    /// Returns a subspan dropping the specified number (default 1) of bytes from the back.
    /// Returns an empty Span if there were no more bytes than that to start with.
    ///
    /// @param count Number of bytes to drop from the back
    ///
    /// @returns The subspan
    constexpr ByteSpan DropBack(size_t count = 1) const { return Base::DropBack(count); }

    /// Assigns the bytes in this Span to the underlying bytes of the "src" Span, which may have any element type.
    /// This modifies this Span's underlying data, not the Span object itself.
    ///
    /// The Spans may have different lengths; the lesser of the two byte counts determines how much is copied.
    ///
    /// @warning The two Spans absolutely must **not** overlap!
    ///
    /// @param [in] src  Specifies the address and length of the source range.
    template<typename T>
    constexpr void AssignSpan(const Span<T>& src) const { Base::AssignSpan(ConstByteSpan(src)); }

    /// Fills the bytes in this Span with the given 8-bit pattern.
    /// This modifies this Span's underlying data, not the Span object itself.
    /// This function does nothing if this Span is empty.
    void Memset(uint8 pattern) const
    {
        // It's illegal to pass a null pointer into memset, even is the count is zero.
        if (HasData())
        {
            std::memset(Data(), pattern, SizeInBytes());
        }
    }

    /// Equivalent to a reinterpret_cast<T*> on this Span's Data() pointer with an added bounds-check.
    /// If this Span isn't large enough to fit a "T" object then nullptr is returned.
    /// The caller must check for null before dereferencing.
    ///
    /// @returns A pointer interpreting this Span as the given type, or nullptr if this Span is too small.
    template<typename T>
    T* Cast() const
    {
        // The C++ spec only lets us manipulate a type's underlying "object representation" if it's trivially copyable.
        static_assert(std::is_trivially_copyable_v<T> || std::is_void_v<T>,
                      "Only trivially copyable types can be stored as a sequence of bytes!");

        // Even if "T" is trivially copyable, it's still undefined behavior to do this cast if we don't meet alignment.
        // Void is exempt because it has no alignment requirement.
        if constexpr (std::is_void_v<T> == false)
        {
            PAL_ASSERT(VoidPtrIsPow2Aligned(Data(), alignof(T)));
        }

        return (ElementSize<T>() <= SizeInBytes()) ? reinterpret_cast<T*>(Data()) : nullptr;
    }

    /// Templated conversion of this typeless Span to a typed subspan.
    /// Equivalent to a reinterpret_cast<T*> on this Span's Data() pointer bundled with a NumElements units conversion.
    ///
    /// Any trailing bytes at the end of this span which cannot form a full "T" element are dropped.
    /// This can result in an empty span even if the original span was non-empty.
    ///
    /// @returns A subspan with typed elements and NumElements truncated down to the nearest sizeof(T).
    template<typename T>
    Span<T> CastSpan() const
    {
        // The C++ spec only lets us manipulate a type's underlying "object representation" if it's trivially copyable.
        static_assert(std::is_trivially_copyable_v<T>,
                      "Only trivially copyable types can be stored as a sequence of bytes!");

        // Even if "T" is trivially copyable, it's still undefined behavior to do this cast if we don't meet alignment.
        PAL_ASSERT(VoidPtrIsPow2Aligned(Data(), alignof(T)));

        // Any existing Span must satisfy the null data invariant so we can call the optimized constructor.
        return Span<T>(reinterpret_cast<T*>(Data()), SizeInBytes() / ElementSize<T>(),
                       _detail::NullInvariantHolds{});
    }

private:
    // An internal constructor which doesn't do the "(pData != nullptr) ? numElements : 0" validation.
    // This is an optimization for cases where we're guaranteed that this invariant must already hold.
    template<typename T>
    ByteSpan(T* pData, size_t numElements, _detail::NullInvariantHolds)
        : Base(reinterpret_cast<std::byte*>(pData), ElementSize<T>() * numElements, _detail::NullInvariantHolds{})
    {
        // The C++ spec only lets us manipulate a type's underlying "object representation" if it's trivially copyable.
        // Void is permitted because it is already a typeless view of raw bytes.
        static_assert(std::is_trivially_copyable_v<T> || std::is_void_v<T>,
                      "Only trivially copyable types can be stored as a sequence of bytes!");
    }
};

/**
 ***********************************************************************************************************************
 * @brief An Unaligned<T> can hold any trivially copyable value in a way that is alignment agnostic.
 *
 * This class exists so that @ref Span, @ref SpanWriter, and @ref SpanReader have feature parity between aligned values
 * and unaligned values. However, there is nothing Span-specific in its implementation so it may be used more generally.
 *
 * To see how this is useful, let's say we have a Span<const void> of 32-bit integers that we know doesn't meet the
 * required 4-byte alignment; that means calling Span<const void>::CastSpan<uint32>() would invoke undefined behavior.
 * Rather than iteratively split this memory into 4-byte long subspans and manipulate the bytes manually, we can call
 * CastSpan<Unaligned<uint32>>() to work in terms of a type that doesn't result in undefined behavior. For example:
 *
 *     for (const Unaligned<uint32>& unValue : span.CastSpan<Unaligned<uint32>>())
 *     {
 *         const uint32 value = unValue.Read(); // Explicitly copy the unaligned bytes into aligned space.
 *         ...
 *     }
 *
 * This has many benefits, including:
 *   1. It uses very few bounds checks because we encode each unaligned item's size in its type. In the example above
 *      there are only two bounds checks: a Span size check in CastSpan, and a hidden for-loop conditional.
 *   2. It's easier to both write and read the code because we're using more natural concepts (arrays and for-loops)
 *      instead of manual Span manipulations.
 *   3. SpanReader and SpanWriter can define "GetUnaligned" functions which suballocate unaligned space without also
 *      reading/writing it. The returned Unaligned<T> can be accessed later without another bounds check.
 *
 * Note that this class purposefully does not define any constructors so that it's trivially constructable. This is
 * what makes it legal (or as legal as possible) to cast arbitrary bytes to Unaligned<T>.
 ***********************************************************************************************************************
 */
template<typename T>
class Unaligned
{
    // This makes the casts easier to understand.
    using Storage = uint8[sizeof(T)];
public:
    /// Copies the object representation (bytes) of the source object into our internal storage.
    ///
    /// @param [in] src  Copy this object's underlying bytes.
    void Write(const T& src) { AssignArray(m_storage, reinterpret_cast<const Storage&>(src)); }

    /// Copies the object representation (bytes) in our internal storage into the destination object.
    ///
    /// @param [out] dst  Overwrite this object's underlying bytes with the bytes in our internal storage.
    void Read(T& dst) const { AssignArray(reinterpret_cast<Storage&>(dst), m_storage); }

    /// Returns an aligned "T" value containing a copy of the object representation (bytes) in our internal storage.
    /// This is a helper function for use-cases where we don't mind returning "T" by value and T can be instantiated.
    ///
    /// @returns The copied, properly aligned "T" value.
    template<typename U = T>
    std::enable_if_t<std::is_array_v<U> == false, T> Read() const
    {
        // If this fails to compile because "T" lacks a default constructor, use the other version of Read() instead.
        T value;
        Read(value);
        return value;
    }

private:
    Storage m_storage;

    // The C++ spec only lets us manipulate a type's underlying "object representation" if it's trivially copyable.
    static_assert(std::is_trivially_copyable_v<T>,
                  "Only trivially copyable types can be stored as a sequence of bytes!");
};

} // Util
