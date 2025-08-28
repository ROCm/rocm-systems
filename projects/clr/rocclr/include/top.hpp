/* Copyright (c) 2008 - 2021 Advanced Micro Devices, Inc.

 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in
 all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE. */

#ifndef TOP_HPP_
#define TOP_HPP_

#if defined(__ARM_ARCH_7__) || defined(__ARM_ARCH_7A__) || defined(__arm) || defined(__arm__) ||   \
    defined(_M_ARM) || defined(__aarch64__) || defined(_M_ARM64)
#define ATI_ARCH_ARM
#elif defined(i386) || defined(__i386) || defined(__i386__) || defined(_M_IX86) ||                 \
    defined(__x86__) || defined(__x86_64) || defined(__x86_64__) || defined(__amd64) ||            \
    defined(__amd64__) || defined(_M_X64) || defined(_M_AMD64)
#define ATI_ARCH_X86
#endif

#if defined(ATI_ARCH_ARM)
#define __EXPORTED_HEADERS__ 1
#endif /*ATI_ARCH_ARM*/

#ifdef _WIN32
// Disable unneeded features of <windows.h> for efficiency.
#define NODRAWTEXT 1
#define NOMINMAX 1
#define WIN32_LEAN_AND_MEAN 1
#endif /*_WIN32*/

#include "utils/macros.hpp"
#include "CL/opencl.h"

#if defined(CL_VERSION_2_0)
/* Deprecated in OpenCL 2.0 */
#define CL_DEVICE_QUEUE_PROPERTIES 0x102A
#define CL_DEVICE_HOST_UNIFIED_MEMORY 0x1035
#endif

#if defined(ATI_ARCH_X86)
#include <xmmintrin.h>
#endif /*ATI_ARCH_X86*/

#include <atomic>
#include <cstdint>
#include <cstddef>
#include <new>

typedef unsigned char* address;
typedef const unsigned char* const_address;
typedef void* pointer;
typedef const void* const_pointer;
typedef unsigned int uint;
typedef unsigned long ulong;
typedef const char* cstring;

#if defined(_WIN32)
#if defined(_WIN64)
typedef __int64 ssize_t;
#else   // !_WIN64
typedef __int32 ssize_t;
#endif  // !_WIN64
#endif  /*_WIN32*/

#ifdef _WIN32
#define SIZE_T_FMT "%Iu"
#define PTR_FMT "0x%p"
#if _MSC_VER < 1900
#define snprintf sprintf_s
#endif
#define ROCCLR_INIT_PRIORITY(priority)
#else /*!_WIN32*/
#define SIZE_T_FMT "%zu"
#define PTR_FMT "%p"
#define ROCCLR_INIT_PRIORITY(priority) __attribute__((init_priority(priority)))
#endif /*!_WIN32*/

typedef uint32_t cl_mem_fence_flags;

//! \cond ignore
#define _BAD_INT32 0xBAADBAAD
#define _BAD_INT64 0XBAADBAADBAADBAADLL
#define _BAD_INTPTR LP64_SWITCH(_BAD_INT32, _BAD_INT64)

const pointer badPointer = (pointer)(intptr_t)_BAD_INTPTR;
const address badAddress = (address)(intptr_t)_BAD_INTPTR;
//! \endcond

constexpr size_t Ki = 1024;
constexpr size_t Mi = Ki * Ki;
constexpr size_t Gi = Ki * Ki * Ki;

constexpr size_t K = 1000;
constexpr size_t M = K * K;
constexpr size_t G = K * K * K;

#include "utils/debug.hpp"

//! \addtogroup Utils

//! Namespace for AMD's OpenCL platform
namespace amd { /*@{*/

//! \brief The default Null object type (!= void*);
struct Null {};

//! \brief Return a const Null object (null)
inline const Null null() { return Null(); }

/*! \brief Equivalent to a namespace (All member functions are static).
 */
class AllStatic {
  WINDOWS_SWITCH(public, private) : AllStatic() { ShouldNotCallThis(); }
  AllStatic(const AllStatic&) { ShouldNotCallThis(); }
  ~AllStatic() { ShouldNotCallThis(); }
};

/*! \brief For embedded objects.
 */
class EmbeddedObject {
  WINDOWS_SWITCH(public, private) : void * operator new(size_t) {
    ShouldNotCallThis();
    return badPointer;
  }
  void operator delete(void*) { ShouldNotCallThis(); }
};

/*! \brief For stack allocated objects.
 */
class StackObject {
  WINDOWS_SWITCH(public, private) : void * operator new(size_t) {
    ShouldNotCallThis();
    return badPointer;
  }
  void operator delete(void*) { ShouldNotCallThis(); }
};

/*! \brief for objects allocated in a dedicate memory pool.
  the standard 'new' should not be called,
  only the in place version 'new (allocation_pointer) <class>()'
  , delete should only invoke the destructors and not release memory
 */
class MemoryPoolObject {
 public:
  void* operator new(size_t) {
    ShouldNotCallThis();
    return badPointer;
  }
  void* operator new(size_t size, void* address) { return address; }
  void operator delete(void*) {}
  void operator delete(void*, void* address) {}
};

/*! \brief For objects allocated on the C-heap.
 */
class HeapObject {
 public:
  void* operator new(size_t size);
  void operator delete(void* obj);
  void* operator new(size_t size, size_t extSize) {
    return HeapObject::operator new(size + extSize);
  };
  void operator delete(void* obj, size_t extSize) { HeapObject::operator delete(obj); }
};

/*! \brief For all reference counted objects.
 */
class ReferenceCountedObject {
  std::atomic<uint> referenceCount_;

 protected:
  virtual ~ReferenceCountedObject() {}
  virtual bool terminate() { return true; }

 public:
  ReferenceCountedObject() : referenceCount_(1) {}

  void* operator new(size_t size) { return ::operator new(size); }
  void operator delete(void* p) { return ::operator delete(p); }
  void* operator new(size_t size, size_t extSize) {
    return ReferenceCountedObject::operator new(size + extSize);
  };
  void operator delete(void* obj, size_t extSize) { ReferenceCountedObject::operator delete(obj); }

  uint referenceCount() const { return referenceCount_.load(std::memory_order_relaxed); }

  uint retain();
  uint release();
};

/*@}*/  // namespace amd
}  // namespace amd

#undef min  // using std::min
#undef max  // using std::max

#endif /*TOP_HPP_*/
