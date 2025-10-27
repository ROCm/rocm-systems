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

#pragma once

#include <ddPlatform.h>

#if !DD_PLATFORM_WINDOWS_KM
#include <type_traits>
#endif
#include <cstring>

namespace DevDriver
{
    template <typename T, size_t defaultCapacity = 8>
    class Vector
    {
    public:
        class Iterator;

        // The capacity this Vector can hold without allocating extra space.
        static constexpr size_t DefaultCapacity = defaultCapacity;

        // Standard constructor
        explicit Vector(const AllocCb& allocCb)
            : m_pData(m_data)
            , m_size(0)
            , m_capacity(defaultCapacity)
            , m_allocCb(allocCb)
        {
        }

        // Move constructor
        Vector(Vector &&rhs)
            : m_pData(m_data) // default initialize it to the default allocation
            , m_size(Platform::Exchange(rhs.m_size, (size_t)0)) // move the rhs size value into ours
            , m_capacity(defaultCapacity) // initialize the capacity to default
            , m_allocCb(rhs.m_allocCb) // copy the allocator callback
        {
            // if the vector will fit inside the default allocation, move it into it
            if (m_size <= defaultCapacity)
            {
                for (size_t index = 0; index < m_size; index++)
                {
                    m_data[index] = Platform::Move(rhs.m_pData[index]);
                }
            }
            else // otherwise, we want to move the allocation + replace the capacity
            {
                m_pData = Platform::Exchange(rhs.m_pData, rhs.m_data);
                m_capacity = Platform::Exchange(rhs.m_capacity, defaultCapacity);
            }
        }

        // Destructor
        ~Vector()
        {
            Clear();
        }

        void operator=(Vector&& rhs)
        {
            Swap(rhs);
        }

        // Convenience methods
        size_t Size() const { return m_size; }
        size_t Capacity() const { return m_capacity; }
        bool IsEmpty() const { return (m_size == 0); }

        // Subscript operator
        T& operator[](size_t index) { DD_ASSERT(index < m_size); return m_pData[index]; }
        const T& operator[](size_t index) const { DD_ASSERT(index < m_size); return m_pData[index]; }

        // Insert elements into the back of the Vector
        template <class... Args>
        bool PushBack(Args&&... args)
        {
            bool result = false;
            Reserve(m_size + 1);
            if (m_size < m_capacity)
            {
                m_pData[m_size] = T(Platform::Forward<Args>(args)...);
                ++m_size;
                result = true;
            }
            return result;
        }

        // Insert elements from another Vector to the back of the Vector
        bool Append(const Vector<T>& other)
        {
            return Append(other.Data(), other.Size());
        }

        template <size_t Len>
        bool Append(const T (&buffer)[Len])
        {
            return Append(buffer, Len);
        }

        // This is un-used by default, but may be overloaded for some Ts
        bool Append(const T* pTs);

        // Insert elements from a buffer to the back of the Vector
        // An empty slice (countOfTs == 0) is effectively a no-op
        bool Append(const T* pTs, size_t countOfTs)
        {
            // Check that we get a valid pointer. If this fires, we'll crash but this is more visible than just crashing
            // in a memcpy below.
            if (countOfTs != 0)
            {
                DD_ASSERT(pTs != nullptr);
            }

            // Pre-allocate all the new elements, since we know how many there are.
            const size_t oldSize = Grow(countOfTs);

            // Some types can be bulk-transferred with a memcpy.
            // Instead of letting the compiler guess, we dictate when dealing with Pods.
            if (Platform::IsPod<T>::Value)
            {
                memcpy(&m_pData[oldSize], pTs, (sizeof(T) * countOfTs));
            }
            else
            {
                for (size_t i = 0; i < countOfTs; ++i)
                {
                    m_pData[oldSize + i] = pTs[i];
                }
            }

            // Pretend Grow() cannot fail, since we cannot check allocation failure with it right now.
            return true;
        }

        // Pop elements out of the Vector
        bool PopBack(T* pData)
        {
            bool result = !IsEmpty();
            if (result)
            {
                --m_size;
                if (pData != nullptr)
                {
                    *pData = Platform::Move(m_pData[m_size]);
                }
            }
            return result;
        }

        // Pop elements out of the Vector
        bool PopFront(T* pData)
        {
            bool result = !IsEmpty();
            if (result)
            {
                if (pData != nullptr)
                {
                    *pData = Platform::Move(m_pData[0]);
                }

                --m_size;

                if (m_size > 0)
                {
                    for (size_t i = 0; i < m_size; i++)
                    {
                        m_pData[i] = Platform::Move(m_pData[i + 1]);
                    }
                }
            }
            return result;
        }

        // Remove the object at the specified index. Does not maintain order.
        void Remove(size_t index)
        {
            DD_ASSERT(index < m_size);

            const size_t lastIndex = m_size - 1;

            // If the index is the last index, we move the last element into it's place
            if (index != lastIndex)
            {
                m_pData[index] = Platform::Move(m_pData[lastIndex]);
            }
            // Otherwise, if it is the last element and not a POD we replace it with a default constructed object
            else if (!Platform::IsPod<T>::Value)
            {
                m_pData[index] = T();
            }

            --m_size;
        }

        // Remove all instances of the specified object from the vector. Does not maintain order.
        size_t Remove(const T& object)
        {
            size_t numRemoved = 0;

            for (size_t index = m_size; index > 0; index--)
            {
                if (m_pData[index - 1] == object)
                {
                    Remove(index - 1);
                    numRemoved++;
                }
            }
            return numRemoved;
        }

        // Free all memory
        void Clear()
        {
            if (m_pData != m_data)
            {
                // If the object is not a POD we explicitly destroy all objects prior to freeing the allocation.
                if (!Platform::IsPod<T>::Value)
                {
                    for (size_t i = 0; i < m_capacity; i++)
                    {
                        m_pData[i].~T();
                    }
                }
                DD_FREE(m_pData, m_allocCb);
                m_pData = m_data;
                m_capacity = defaultCapacity;
                m_size = 0;
            }
            else
            {
                Reset();
            }
        }

        // Clears all objects stored, but doesn't free memory.
        void Reset()
        {
            // If the object is not a POD we need to destroy all instances and replace them with default constructed
            // instances.
            if (!Platform::IsPod<T>::Value)
            {
                for (size_t index = 0; index < m_size; index++)
                {
                    m_pData[index] = T();
                }
            }
            m_size = 0;
        }

        // Swaps the contents of the current vector with the provided vector
        void Swap(Vector& rhs)
        {
            // If we can, we swap allocations directly
            if ((m_pData != m_data) && (rhs.m_pData != rhs.m_data))
            {
                m_pData = Platform::Exchange(rhs.m_pData, m_pData);
            }
            // Else if the other object is using the default allocation we move it's contents here
            // and give ownership of our allocation to it
            else if (m_pData != m_data)
            {
                for (size_t index = 0; index < rhs.m_size; index++)
                {
                    m_data[index] = Platform::Move(rhs.m_data[index]);
                }
                rhs.m_pData = Platform::Exchange(m_pData, m_data);
            }
            // Else if this object is using the default allocation we move our data into it's allocation
            // and take ownership of our allocation to it
            else if (rhs.m_pData != rhs.m_data)
            {
                for (size_t index = 0; index < m_size; index++)
                {
                    rhs.m_data[index] = Platform::Move(m_data[index]);
                }
                m_pData = Platform::Exchange(rhs.m_pData, rhs.m_data);
            }
            // Otherwise we just exchange all the objects that we need to
            else
            {
                for (size_t index = 0; index < Platform::Max(m_size, rhs.m_size); index++)
                {
                    m_data[index] = Platform::Exchange(rhs.m_data[index], m_data[index]);
                }
            }

            // Finally, we exchange the rest of the data
            m_allocCb = Platform::Exchange(rhs.m_allocCb, m_allocCb);
            m_capacity = Platform::Exchange(rhs.m_capacity, m_capacity);
            m_size = Platform::Exchange(rhs.m_size, m_size);
        }

        // Get a pointer to the beginning of the data
        //
        // Returns nullptr if there is no data available
        const T* Data() const
        {
            return (Size() != 0) ? m_pData : nullptr;
        }

        // Get a pointer to the beginning of the data
        //
        // Returns nullptr if there is no data available
        T* Data()
        {
            return (Size() != 0) ? m_pData : nullptr;
        }

        // Allocates enough memory to hold the specified number of elements
        void Reserve(size_t newSize)
        {
            if (m_capacity < newSize)
            {
                const size_t newCapacity = Platform::Pow2Pad(Platform::Max(newSize, (size_t)1));
                const size_t allocSize = sizeof(T) * newCapacity;
                T* pData = static_cast<T*>(DD_MALLOC(allocSize, alignof(T), m_allocCb));

                DD_ASSERT(pData != nullptr);

                // If the struct is not a POD, then we need to construct objects
                if (is_type_trivial() == false)
                {
                    size_t i = 0;
                    // First, we move all existing objects into the vector.
                    for (; i < m_size; i++)
                    {
                        new(&pData[i]) T(Platform::Move(m_pData[i]));
                    }
                    // Then we construct new objects with the remaining memory.
                    for (; i < newCapacity; i++)
                    {
                        new(&pData[i]) T();
                    }
                }
                // Otherwise, we just copy the existing data into the new vector and call it good.
                else
                {
                    // Need to use reinterpret_cast here because gcc can't seem to evaluate
                    // `is_trivial_v` at compile-time, thus generating a no-class-memaccess warning.
                    // `if constexpr` fixes the issue, but AMDLOG's toolchain doesn't support c++17.
                    std::memcpy(reinterpret_cast<void*>(pData), m_pData, m_size * sizeof(T));
                }

                if (m_pData != m_data)
                {
                    // If the object wasn't a POD we need to destroy all instances before freeing the memory.
                    if (!Platform::IsPod<T>::Value)
                    {
                        for (size_t i = 0; i < m_capacity; i++)
                        {
                            m_pData[i].~T();
                        }
                    }
                    DD_FREE(m_pData, m_allocCb);
                }
                m_pData = pData;
                m_capacity = newCapacity;
            }
        }

        // Resizes the vector. Implicitly destroys objects if newSize is smaller than the existing size.
        void Resize(size_t newSize)
        {
            // TODO: Reserve should return whether allocation failed
            Reserve(newSize);

            // If the object isn't a POD and we are shrinking the size, we need to replace destroyed objects with
            // default constructed instances.
            if (!Platform::IsPod<T>::Value)
            {
                for (size_t i = newSize; i < m_size; i++)
                {
                    m_pData[i] = T();
                }
            }
            m_size = newSize;
        }

        // Resizes the vector, zeroing additional elements
        //
        // Warning: This will break badly if your type cannot be safely memset() to 0!
        void ResizeAndZero(size_t newSize)
        {
            // TODO: Reserve should return whether allocation failed
            Reserve(newSize);

            if (newSize > m_size)
            {
                memset(&m_pData[m_size], 0, (newSize - m_size) * sizeof(T));
            }

            m_size = newSize;
        }

        // Grows the vector by the specified number of elements and returns the previous size
        size_t Grow(size_t numElements)
        {
            const size_t oldSize = m_size;

            Resize(m_size + numElements);

            return oldSize;
        }

        // Iterator creation function
        Iterator Begin() const
        {
            return CreateIterator(0);
        }

        // Iterator creation function
        constexpr Iterator End() const
        {
            return Iterator(nullptr, 0);
        }

        // Iterator creation function
        Iterator CreateIterator(size_t index) const
        {
            if (index < m_size)
                return Iterator(this, index);
            return End();
        }

        // Finds the first index for the provided object
        Iterator Find(const T& object) const
        {
            auto it = Begin();
            for (; it != End(); ++it)
            {
                if (*it == object)
                {
                    break;
                }
            }
            return it;
        }

        // Removes the element represented by the provided iterator. Does not maintain order.
        Iterator Remove(const Iterator& it)
        {
            DD_ASSERT(it.m_pContainer == this);

            Remove(it.m_index);
            if (it.m_index < m_size)
                return it;

            return End();
        }

        /// Returns the internal allocator
        const AllocCb& GetAllocCb() const { return m_allocCb; }

    private:
        // Disallow copy construct.
        Vector(Vector& rhs) = delete;

        // This indirection fixes the warning comparision of a constant with another constant. This should be
        // replace with `if constexpr` once AMDLog upgrades to support C++17.
        constexpr bool is_type_trivial()
        {
#if !DD_PLATFORM_WINDOWS_KM
            return std::is_trivial_v<T>;
#else
            // <type_traits> not available in kernel so treat every type as nontrivial
            return false;
#endif
        }

        T m_data[defaultCapacity];
        T* m_pData;
        size_t m_size;
        size_t m_capacity;
        AllocCb m_allocCb;
    };

    // Iterator class for the Vector type
    template <typename T, size_t defaultCapacity>
    class Vector<T, defaultCapacity>::Iterator
    {
        friend Vector;
    public:
        // Comparison operators
        bool operator==(const Iterator& rhs) const
        {
            return ((m_pContainer == rhs.m_pContainer) && (m_index == rhs.m_index));
        }

        bool operator!=(const Iterator& rhs) const
        {
            return ((m_pContainer != rhs.m_pContainer) || (m_index != rhs.m_index));
        }

        // Prefix operator to increment the iterator
        Iterator& operator++()
        {
            if (m_pContainer != nullptr)
            {
                m_index += 1;
                if (m_index >= m_pContainer->m_size)
                {
                    m_index = 0;
                    m_pContainer = nullptr;
                }
            }
            return *this;
        }

        // Addition operator to add to the iterator
        Iterator& operator+(size_t value)
        {
            if (m_pContainer != nullptr)
            {
                m_index += value;
                if (m_index >= m_pContainer->m_size)
                {
                    m_index = 0;
                    m_pContainer = nullptr;
                }
            }
            return *this;
        }

        // Indirection operator
        T& operator*() const
        {
            DD_ASSERT(m_pContainer != nullptr);
            return m_pContainer->m_pData[m_index];
        }

        // Member of pointer operator. Returns a pointer to the object in the shared container.
        T* operator->() const
        {
            DD_ASSERT(m_pContainer != nullptr);
            return &m_pContainer->m_pData[m_index];
        }
    private:
        // Constructor is private to ensure it cannot be created by anything other than the Vector itself
        Iterator(const Vector* pContainer, size_t index) :
            m_pContainer(pContainer),
            m_index(index)
        {
        };

        const Vector* m_pContainer;
        size_t  m_index;
    };

    //
    // functions necessary for C++ ranged based for loop support
    //

    // Implement begin() function for range-based for loops
    template <typename T, size_t defaultCapacity>
    inline typename Vector<T, defaultCapacity>::Iterator begin(Vector<T, defaultCapacity>& rhs)
    {
        return rhs.Begin();
    }

    // Implement end() function for range-based for loops
    template <typename T, size_t defaultCapacity>
    inline constexpr typename Vector<T, defaultCapacity>::Iterator end(const Vector<T, defaultCapacity>& rhs)
    {
        return rhs.End();
    }

    // Specialized functions for using Vector<> like a String
    template <>
    inline bool Vector<char>::Append(const char* pStr)
    {
        return Append(pStr, strlen(pStr));
    }

    template <>
    template <size_t Len>
    inline bool Vector<char>::Append(const char (&str)[Len])
    {
        return Append(str, strlen(str));
    }

} // DevDriver
