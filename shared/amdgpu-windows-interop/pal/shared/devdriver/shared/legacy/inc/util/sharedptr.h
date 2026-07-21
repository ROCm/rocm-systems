/*
 ***********************************************************************************************************************
 *
 *  Copyright (c) 2021-2026 Advanced Micro Devices, Inc. All Rights Reserved.
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

namespace DevDriver
{
    // Helper structure that sets value to true if T is not abstract and is constructable using the provided
    // arguments, otherwise it returns false. We use this to prevent the Create() function from being defined
    // for classes that are not creatable. This in turn prevents code from being generated that causes
    // Container to be defined, which leads to compile errors even if the client application never attempts
    // to directly create an object of that type.
    template<typename T, typename... Args>
    struct CanConstruct
    {
        DD_STATIC_CONST bool Value = !Platform::IsAbstract<T>::Value && Platform::IsConstructible<T, Args...>::Value;
    };

    // SharedPointerBase is the common parent class used by SharedPointer<>
    // It implements common functions (e.g., pointer management) but cannot ever be
    // used directly. The purpose of this separation is to enable SharedPointer to perform
    // typecasts between derived and base types by casting through SharedPointerBase.
    class SharedPointerBase
    {
        // SharedPointer is a subclass, but all need to have access to the protected + private members
        template<typename T>
        friend class SharedPointer;
        // public functions available to all subclasses
    public:
        // check to see if the class has been set
        bool IsNull() const;

        // clear the pointer and, if required, delete the underlying allocation
        void Clear()
        {
            if (m_pContainer != nullptr)
            {
                if (m_pContainer->Release() == 0)
                {
                    // ContainerBase has been declared with a virtual destructor, which guarantees
                    // that the specific ContainerBase subclass destructor is called
                    DD_DELETE(m_pContainer, m_pContainer->GetAllocCb());
                }
                m_pContainer = nullptr;
                m_pObject = nullptr;
            }
        }

    protected:
        // Inner class that provides a standardized reference counted container interface
        // Subclassed by SharedPointer to include an actual object
        class ContainerBase
        {
        public:
            using AtomicValueType = decltype(+Platform::Atomic{});
            static_assert(sizeof(AtomicValueType) == sizeof(Platform::Atomic), "Atomic storage must match Platform::Atomic size.");
            static_assert(alignof(AtomicValueType) == alignof(Platform::Atomic), "Atomic storage must match Platform::Atomic alignment.");

            // Construct container and initialize ref count to zero. This class should never be
            // constructed directly by anything other than a subclass.
            ContainerBase(const AllocCb &allocCb);

            ContainerBase& operator=(const ContainerBase&) = delete;
            ContainerBase& operator=(ContainerBase&&) = delete;

            // Destroy the container. Since this class is never directly created, this ensures
            // subclasses (and the contained object) are always destroyed correctly.
            virtual ~ContainerBase()
            {
                DD_ASSERT(RefCountValue() == 0);
                DD_PRINT(LogLevel::Never, "Deleted reference counted container %i", RefCountValue());
            }

            // Increments the reference count of the container
            int32 Retain(void)
            {
                DD_ASSERT(RefCountValue() >= 0);
                int32 result = Platform::AtomicIncrement(RefCountAtomic());
                DD_ASSERT(result >= 1);
                DD_PRINT(LogLevel::Never, "Incremented reference count: %i", result);
                return result;
            }

            // Decrements the reference count of the container
            int32 Release(void)
            {
                int32 result = Platform::AtomicDecrement(RefCountAtomic());
                DD_ASSERT(result >= 0);
                DD_PRINT(LogLevel::Never, "Decremented reference count: %i", result);
                return result;
            }

            // Returns the reference count of the container
            int32 QueryReferenceCount(void) const;

            // Retrieve the allocator callbacks so it can be destroyed
            const AllocCb& GetAllocCb() const { return m_allocCb; }
        private:
            Platform::Atomic* RefCountAtomic()
            {
                return &m_refCount;
            }

            const Platform::Atomic* RefCountAtomic() const
            {
                return &m_refCount;
            }

            int32 RefCountValue() const
            {
                return Platform::AtomicAdd(const_cast<Platform::Atomic*>(RefCountAtomic()), 0);
            }

            // Allocator callbacks
            const AllocCb       m_allocCb;
            // Reference count
            AtomicValueType     m_refCount;

        };

        // Default constructor that initializes an empty shared pointer base.
        SharedPointerBase();

        // Initialize the object using the provided pointer
        SharedPointerBase(ContainerBase* pContainer, void* pObject);

        // Copy constructor copies the container pointer and increments the reference count
        SharedPointerBase(const SharedPointerBase &right);

        // Move constructor takes the container pointer and clears the other container's pointer
        SharedPointerBase(SharedPointerBase &&right);

        SharedPointerBase& operator=(const SharedPointerBase&) = delete;
        SharedPointerBase& operator=(SharedPointerBase&&) = delete;

        // On deletion of the object clear the pointer
        ~SharedPointerBase()
        {
            Clear();
        }
    private:
        // Pointer to the shared container
        ContainerBase* m_pContainer;
        // Pointer to the object inside the shared container. We keep a copy of this to allow
        // direct access to the object since we might not know the actual parent type of it.
        void*          m_pObject;
    };

    template <typename T>
    class SharedPointer : public SharedPointerBase
    {
    public:
        // Create SharedPointer object with the default constructor
        SharedPointer() : SharedPointerBase() {};

        SharedPointer(const SharedPointer<T>&) = default;

        // Copy conversion constructor. Creates a new object if you can cast from type U to type T.
        template <typename U, typename = typename Platform::EnableIf<Platform::IsConvertible<U*, T*>::Value>::Type>
        SharedPointer(const SharedPointer<U> &right)
            : SharedPointerBase(Platform::Forward<const SharedPointerBase>(right))
        {
        }

        // Move conversion constructor. Takes ownership of the shared container if you can cast from type U to type T.
        template <typename U, typename = typename Platform::EnableIf<Platform::IsConvertible<U*, T*>::Value>::Type>
        SharedPointer(SharedPointer<U> &&right)
            : SharedPointerBase(Platform::Forward<SharedPointerBase>(right))
        {
        }

        // Assignment operator to allow copy + swap idiom
        SharedPointer<T> &operator= (SharedPointer<T> right)
        {
            m_pContainer = Platform::Exchange(right.m_pContainer, m_pContainer);
            m_pObject = Platform::Exchange(right.m_pObject, m_pObject);
            return *this;
        }

        // Indirection operator. Returns a const reference to the object in the shared container.
        // This operator is unsafe to use if the container hasn't been allocated.
        T& operator*() const
        {
            DD_ASSERT(m_pObject != nullptr);
            return *Get();
        }

        // Member of pointer operator. Returns a pointer to the object in the shared container.
        // This operator is unsafe to use if the container hasn't been allocated.
        T* operator->() const
        {
            DD_ASSERT(m_pObject != nullptr);
            return Get();
        }

        // Templated comparison operator. Allows comparing shared pointer objects so long as U is convertable to T.
        template <typename U, typename = typename Platform::EnableIf<Platform::IsConvertible<U*, T*>::Value>::Type>
        bool operator== (const SharedPointer< U >&right) const
        {
            return m_pObject == right.m_pObject;
        }

        // Templated comparison operator. Allows comparing shared pointer objects so long as U is convertable to T.
        template <typename U, typename = typename Platform::EnableIf<Platform::IsConvertible<U*, T*>::Value>::Type>
        bool operator!= (const SharedPointer< U >&right) const
        {
            return m_pObject != right.m_pObject;
        }

        // Get a pointer to the contained object
        T* Get() const
        {
            return static_cast<T* const>(m_pObject);
        }

        // Returns the reference count for the container
        int32 QueryReferenceCount() const
        {
            return m_pContainer->QueryReferenceCount();
        }

        // Create a SharedPointer using the provided allocator callbacks and arguments
        // This function is only valid if the class is not a valid class
        template<typename... Args,
            typename = typename Platform::EnableIf<CanConstruct<T, Args...>::Value>::Type>
            static SharedPointer<T> Create(const AllocCb& allocCb, Args&&... args)
        {
            SharedPointer result;
            Container *pContainer =
                DD_NEW(Container, allocCb)(allocCb, Platform::Forward<Args>(args)...);

            if (pContainer != nullptr)
            {
                result = SharedPointer(pContainer, &pContainer->m_object);
            }
            return result;
        }
    private:
        // Templated Container class that inherents the type from the outer (SharedPointer) class
        class Container : public ContainerBase
        {
        public:
            // Constructor that initializes ContainerBase class and the object using the provided parameters
            template<typename... Args>
            explicit Container(const AllocCb& allocCb, Args&&... args)
                : ContainerBase(allocCb)
                , m_object(Platform::Forward<Args>(args)...)
            {
            }

            // Actual object that the SharedPointer instance encapsulates
            T m_object;
        };

        // Private constructor to allow direct initialization using an externally created Container
        explicit SharedPointer(Container* pContainer, T* pObject)
            : SharedPointerBase(static_cast<ContainerBase*>(pContainer), pObject)
        {
        }
    };
} // DevDriver
