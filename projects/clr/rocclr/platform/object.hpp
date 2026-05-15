/* Copyright (c) 2008 - 2024 Advanced Micro Devices, Inc.

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

#ifndef OBJECT_HPP_
#define OBJECT_HPP_

#include <set>

#include "top.hpp"
#include "os/alloc.hpp"
#include "thread/monitor.hpp"
#include "utils/util.hpp"
#ifdef _WIN32
#include <windows.h>
#include <d3d9.h>
#include <d3d10_1.h>
#include <CL/cl_d3d10.h>
#include <CL/cl_d3d11.h>
#include <CL/cl_dx9_media_sharing.h>
#endif
#include <CL/cl_icd.h>


#define KHR_CL_TYPES_DO(F)                                                                         \
  /* OpenCL type          Runtime type */                                                          \
  F(cl_context, Context)                                                                           \
  F(cl_event, Event)                                                                               \
  F(cl_command_queue, CommandQueue)                                                                \
  F(cl_kernel, Kernel)                                                                             \
  F(cl_program, Program)                                                                           \
  F(cl_device_id, Device)                                                                          \
  F(cl_mem, Memory)                                                                                \
  F(cl_sampler, Sampler)

#define AMD_CL_TYPES_DO(F)                                                                         \
  F(cl_counter_amd, Counter)                                                                       \
  F(cl_perfcounter_amd, PerfCounter)                                                               \
  F(cl_threadtrace_amd, ThreadTrace)

#define CL_TYPES_DO(F)                                                                             \
  KHR_CL_TYPES_DO(F)                                                                               \
  AMD_CL_TYPES_DO(F)

// Forward declare ::cl_* types and amd::Class types
//

#define DECLARE_CL_TYPES(CL, AMD)                                                                  \
  namespace amd {                                                                                  \
  class AMD;                                                                                       \
  }

CL_TYPES_DO(DECLARE_CL_TYPES);

#undef DECLARE_CL_TYPES

typedef struct _cl_icd_dispatch cl_icd_dispatch;

#define DECLARE_CL_TYPES(CL, AMD)                                                                  \
  typedef struct _##CL {                                                                           \
    cl_icd_dispatch* dispatch;                                                                     \
  }* CL;

AMD_CL_TYPES_DO(DECLARE_CL_TYPES);

#undef DECLARE_CL_TYPES

namespace amd {

// Define the cl_*_type tokens for type checking.
//

#define DEFINE_CL_TOKENS(CL, ignored) T##CL,

enum cl_token { Tinvalid = 0, CL_TYPES_DO(DEFINE_CL_TOKENS) numTokens };

#undef DEFINE_CL_TOKENS

const size_t RuntimeObjectAlignment = NextPowerOfTwo<numTokens>::value;

//! \cond ignore
template <typename T> struct as_internal {
  typedef void type;
};

template <typename T> struct as_external {
  typedef void type;
};

template <typename T> struct class_token {
  static const cl_token value = Tinvalid;
};

#define DEFINE_CL_TRAITS(CL, AMD)                                                                  \
                                                                                                   \
  template <> struct class_token<AMD> {                                                            \
    static const cl_token value = T##CL;                                                           \
  };                                                                                               \
                                                                                                   \
  template <> struct as_internal<_##CL> {                                                          \
    typedef AMD type;                                                                              \
  };                                                                                               \
  template <> struct as_internal<const _##CL> {                                                    \
    typedef AMD const type;                                                                        \
  };                                                                                               \
                                                                                                   \
  template <> struct as_external<AMD> {                                                            \
    typedef _##CL type;                                                                            \
  };                                                                                               \
  template <> struct as_external<const AMD> {                                                      \
    typedef _##CL const type;                                                                      \
  };

CL_TYPES_DO(DEFINE_CL_TRAITS);

#undef DEFINE_CL_TRAITS
//! \endcond

struct ICDDispatchedObject {
#ifdef __HIP_PLATFORM_AMD__
  static inline cl_icd_dispatch icdVendorDispatch_[] = {0};
#else
  static cl_icd_dispatch icdVendorDispatch_[];
#endif
  const cl_icd_dispatch* const dispatch_;

 protected:
  ICDDispatchedObject() : dispatch_(icdVendorDispatch_) {}

 public:
  static bool isValidHandle(const void* handle) { return handle != NULL; }

  const void* handle() const { return static_cast<const ICDDispatchedObject*>(this); }
  void* handle() { return static_cast<ICDDispatchedObject*>(this); }

  template <typename T> static const T* fromHandle(const void* handle) {
    return static_cast<const T*>(reinterpret_cast<const ICDDispatchedObject*>(handle));
  }
  template <typename T> static T* fromHandle(void* handle) {
    return static_cast<T*>(reinterpret_cast<ICDDispatchedObject*>(handle));
  }
};

/*! \brief For all OpenCL/Runtime objects.
 */
class RuntimeObject : public ReferenceCountedObject, public ICDDispatchedObject {
 public:
  enum ObjectType {
    ObjectTypeContext = 0,
    ObjectTypeDevice = 1,
    ObjectTypeMemory = 2,
    ObjectTypeKernel = 3,
    ObjectTypeCounter = 4,
    ObjectTypePerfCounter = 5,
    ObjectTypeEvent = 6,
    ObjectTypeProgram = 7,
    ObjectTypeQueue = 8,
    ObjectTypeSampler = 9,
    ObjectTypeThreadTrace = 10,
    ObjectTypeVMMAlloc = 11
  };

  virtual ObjectType objectType() const = 0;
};

template <typename T> class SharedReference : public EmbeddedObject {
 private:
  T& reference_;

 private:
  // do not copy shared references.
  SharedReference<T>& operator=(const SharedReference<T>& sref);

 public:
  explicit SharedReference(T& reference) : reference_(reference) { reference_.retain(); }

  ~SharedReference() { reference_.release(); }

  T& operator()() const { return reference_; }
};

/*! \brief A 1,2 or 3D coordinate.
 *!
 *! Note, dimensionality is only defined for sizes, and is given by the number
 *! of non-zero elements. (i.e. a 1D line is not the same as a 2D plane with width 1)
 */

struct Coord3D {
  size_t c[3];

  Coord3D(size_t d0, size_t d1 = 0, size_t d2 = 0) {
    c[0] = d0;
    c[1] = d1;
    c[2] = d2;
  }
  const size_t& operator[](size_t idx) const {
    assert(idx < 3);
    return c[idx];
  }
  bool operator==(const Coord3D& rhs) const {
    return c[0] == rhs.c[0] && c[1] == rhs.c[1] && c[2] == rhs.c[2];
  }
  explicit operator size_t*() { return &c[0]; }
};

template <class T> class SysmemPool {
 public:
  SysmemPool() : chunk_access_(true) /* Sysmem Pool Lock */ {}
  ~SysmemPool() {
    if (free_chunk_num_ != max_chunk_idx_) {
      for (int i = 0; i < kActiveAllocSize; ++i) {
        // Free any active chunks
        if (active_allocs_[i] != nullptr) {
          auto chunk = active_allocs_[i]->base_;
          // Check if this chunk contains unreleased memory objects
          if ((chunk->busy_ + chunk->free_) != kAllocChunkSize) {
            LogPrintfError("Unreleased slots in sysmem pool %ld",
                           kAllocChunkSize - (chunk->busy_ + chunk->free_));
          }
          delete chunk;
          free_chunk_num_++;
        }
      }
      // Validate if sysmempool released all memory
      if (free_chunk_num_ != max_chunk_idx_) {
        LogPrintfError("Unreleased chunk in sysmem pool %ld", max_chunk_idx_ - free_chunk_num_);
      }
    }
  }
  void* Alloc(size_t size) {
    guarantee(size <= sizeof(T), "Bigger size than pool allows!");
    size_t current = current_alloc_++;
    auto idx = current / kAllocChunkSize;
    while (idx >= max_chunk_idx_) {
      ScopedLock lock(chunk_access_);
      // Second check in a case of multiple waiters
      if (idx == max_chunk_idx_) {
        auto allocs = new MemoryObject[kAllocChunkSize];
        // Save the base in the first slot of all allocations
        allocs[0].base_ = new AllocChunk(allocs);
        // Check if the overwritten chunk has still empty slots
        if (active_allocs_[idx % kActiveAllocSize] != nullptr) {
          auto stale = active_allocs_[idx % kActiveAllocSize]->base_;
          if (stale->busy_ != kAllocChunkSize) {
            // The pool contains the stale slots, hence make sure it's marked as free
            auto freed = stale->free_ - stale->busy_;
            if (freed == 0) {
              delete stale;
              free_chunk_num_++;
            }
          }
        }
        // Keep the chunk in the list of active chunks
        active_allocs_[idx % kActiveAllocSize] = allocs;
        max_chunk_idx_++;
      } else if ((idx < max_chunk_idx_) && ((max_chunk_idx_ - idx) >= kActiveAllocSize)) {
        // If a wait was very long, then drop the old slot and find a more recent one
        current = current_alloc_++;
        idx = current / kAllocChunkSize;
      }
    }

    // Find a slot in the active pool of allocations
    auto chunk_idx = idx % kActiveAllocSize;
    MemoryObject* obj = &active_allocs_[chunk_idx][current % kAllocChunkSize];
    // Save the chunk allocation
    obj->base_ = active_allocs_[chunk_idx]->base_;
    obj->base_->busy_++;
    return &obj->object_;
  }

  void Free(void* ptr) {
#if IS_WINDOWS
    auto obj = reinterpret_cast<MemoryObject*>(reinterpret_cast<address>(ptr) -
                                               offsetof(MemoryObject, object_));
#else
    auto obj =
        reinterpret_cast<MemoryObject*>(reinterpret_cast<address>(ptr) - sizeof(AllocChunk*));
#endif
    auto freed = --obj->base_->free_;
    // If it's the last slot in the chunk, then release memory
    if (freed == 0) {
      auto base = obj->base_;
      {
        // Make sure active chunks don't have a stale pointer
        ScopedLock lock(chunk_access_);
        for (int i = 0; i < kActiveAllocSize; ++i) {
          if (base->allocs_ == active_allocs_[i]) {
            active_allocs_[i] = nullptr;
            break;
          }
        }
      }
      delete base;
      free_chunk_num_++;
    }
  }

 private:
  static constexpr size_t kAllocChunkSize = 2048;  //!< The total number of allocations in a chunk
  static constexpr size_t kActiveAllocSize = 32;   //!< The number of active chunks
  struct AllocChunk;
  struct MemoryObject {
    AllocChunk* base_;  //!< The chunk information for this memory object
    T object_;          //!< Allocated user object
    MemoryObject() {}
  };
  struct AllocChunk {
    MemoryObject* allocs_;        //! Array of allocations
    std::atomic<uint32_t> busy_;  //! The number of commands still available for usage
    std::atomic<uint32_t> free_;  //! The number of commands still available for usage
    AllocChunk(MemoryObject* alloc) : allocs_(alloc), busy_(0), free_(kAllocChunkSize) {}
    ~AllocChunk() { delete[] allocs_; }
  };

  std::atomic<uint64_t> current_alloc_ = 0;             //!< Current allocation, global index
  std::atomic<size_t> max_chunk_idx_ = 0;               //!< Current max chunk index
  size_t free_chunk_num_ = 0;                           //!< The number of freed chunks
  amd::Monitor chunk_access_;                           //!< Lock for the chunk list access
  MemoryObject* active_allocs_[kActiveAllocSize] = {};  //!< Active chunks for fast access
};

}  // namespace amd

template <typename CL> typename amd::as_internal<CL>::type* as_amd(CL* cl_obj) {
  return cl_obj == NULL ? NULL
                        : amd::RuntimeObject::fromHandle<typename amd::as_internal<CL>::type>(
                              static_cast<void*>(cl_obj));
}

template <typename AMD> typename amd::as_external<AMD>::type* as_cl(AMD* amd_obj) {
  return amd_obj == NULL ? NULL
                         : static_cast<typename amd::as_external<AMD>::type*>(amd_obj->handle());
}

template <typename CL> bool is_valid(CL* handle) {
  return amd::as_internal<CL>::type::isValidHandle(handle);
}

#endif /*OBJECT_HPP_*/
