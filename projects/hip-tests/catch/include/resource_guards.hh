/*
Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
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
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#pragma once

#include <hip_array_common.hh>
#include <hip_test_common.hh>
#include <hip/hip_runtime_api.h>

enum class LinearAllocs {
  malloc,
  mallocAndRegister,
  hipHostMalloc,
  hipMalloc,
  hipMallocManaged,
  noAlloc
};

inline std::string to_string(const LinearAllocs allocation_type) {
  switch (allocation_type) {
    case LinearAllocs::malloc:
      return "malloc";
    case LinearAllocs::mallocAndRegister:
      return "malloc + hipHostRegister";
    case LinearAllocs::hipHostMalloc:
      return "hipHostMalloc";
    case LinearAllocs::hipMalloc:
      return "hipMalloc";
    case LinearAllocs::hipMallocManaged:
      return "hipMallocManaged";
    default:
      return "unknown alloc type";
  }
}

template <typename T> class LinearAllocGuard {
 public:
  LinearAllocGuard() = default;

  LinearAllocGuard(const LinearAllocs allocation_type, const size_t size,
                   const unsigned int flags = 0u)
    : allocation_type_{allocation_type},
      size_{size} {
    switch (allocation_type_) {
      case LinearAllocs::malloc:
        ptr_ = host_ptr_ = reinterpret_cast<T*>(malloc(size));
        break;
      case LinearAllocs::mallocAndRegister:
        host_ptr_ = reinterpret_cast<T*>(malloc(size));
        HIP_CHECK(hipHostRegister(host_ptr_, size, flags));
        HIP_CHECK(hipHostGetDevicePointer(reinterpret_cast<void**>(&ptr_), host_ptr_, 0u));
        break;
      case LinearAllocs::hipHostMalloc:
        HIP_CHECK(hipHostMalloc(reinterpret_cast<void**>(&ptr_), size, flags));
        host_ptr_ = ptr_;
        break;
      case LinearAllocs::hipMalloc:
        HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&ptr_), size));
        break;
      case LinearAllocs::hipMallocManaged:
        HIP_CHECK(hipMallocManaged(reinterpret_cast<void**>(&ptr_), size, flags ? flags : 1u));
        host_ptr_ = ptr_;
        break;
      case LinearAllocs::noAlloc:
        break;
    }
  }

  LinearAllocGuard(const LinearAllocGuard&) = delete;

  LinearAllocGuard(LinearAllocGuard&& o) { *this = std::move(o); }

  LinearAllocGuard& operator=(LinearAllocGuard&& o) {
    if (this != &o) {
      dealloc();

      allocation_type_ = o.allocation_type_;
      ptr_ = o.ptr_;
      host_ptr_ = o.host_ptr_;
      size_ = o.size_;

      o.allocation_type_ = LinearAllocs::noAlloc;
      o.ptr_ = nullptr;
      o.host_ptr_ = nullptr;
      o.size_ = 0;
    }

    return *this;
  }

  ~LinearAllocGuard() { dealloc(); }

  T* ptr() const { return ptr_; };
  T* host_ptr() const { return host_ptr_; }
  size_t size_bytes() const { return size_; }

 private:
  LinearAllocs allocation_type_ = LinearAllocs::noAlloc;
  T* ptr_ = nullptr;
  T* host_ptr_ = nullptr;
  size_t size_ = 0;

  void dealloc() {
    if (ptr_ == nullptr) {
      return;
    }
    // No Catch macros, don't want to possibly throw in the destructor
    if (ptr_ != nullptr) {
      switch (allocation_type_) {
        case LinearAllocs::noAlloc:
          break;
        case LinearAllocs::malloc:
          free(ptr_);
          break;
        case LinearAllocs::mallocAndRegister:
          // Cast to void to suppress nodiscard warnings
          static_cast<void>(hipHostUnregister(host_ptr_));
          free(host_ptr_);
          break;
        case LinearAllocs::hipHostMalloc:
          static_cast<void>(hipHostFree(ptr_));
          break;
        case LinearAllocs::hipMalloc:
        case LinearAllocs::hipMallocManaged:
          static_cast<void>(hipFree(ptr_));
      }
    }
  }
};

template <typename T> class LinearAllocGuardMultiDim {
 protected:
  LinearAllocGuardMultiDim(hipExtent extent) : extent_{extent} {}

  ~LinearAllocGuardMultiDim() { static_cast<void>(hipFree(pitched_ptr_.ptr)); }

 public:
  T* ptr() const { return reinterpret_cast<T*>(pitched_ptr_.ptr); };

  size_t pitch() const { return pitched_ptr_.pitch; }

  hipExtent extent() const { return extent_; }

  hipPitchedPtr pitched_ptr() const { return pitched_ptr_; }

  size_t width() const { return extent_.width; }

  size_t width_logical() const { return extent_.width / sizeof(T); }

  size_t height() const { return extent_.height; }

 public:
  hipPitchedPtr pitched_ptr_;
  const hipExtent extent_;
};

template <typename T, bool unaligned = false>
class LinearAllocGuard2D : public LinearAllocGuardMultiDim<T> {
 public:
  LinearAllocGuard2D(const size_t width_logical, const size_t height)
      : LinearAllocGuardMultiDim<T>{make_hipExtent(width_logical * sizeof(T), height, 1)} {
    if (unaligned) {
      this->pitched_ptr_.pitch = width_logical * sizeof(T);
      HIP_CHECK(hipMalloc(&this->pitched_ptr_.ptr, this->pitched_ptr_.pitch * height));
    } else {
      HIP_CHECK(hipMallocPitch(&this->pitched_ptr_.ptr, &this->pitched_ptr_.pitch,
                               this->extent_.width, this->extent_.height));
    }
  }

  LinearAllocGuard2D(const LinearAllocGuard2D&) = delete;
  LinearAllocGuard2D(LinearAllocGuard2D&&) = delete;
};

template <typename T> class LinearAllocGuard3D : public LinearAllocGuardMultiDim<T> {
 public:
  LinearAllocGuard3D(const size_t width_logical, const size_t height, const size_t depth)
      : LinearAllocGuardMultiDim<T>{make_hipExtent(width_logical * sizeof(T), height, depth)} {
    HIP_CHECK(hipMalloc3D(&this->pitched_ptr_, this->extent_));
  }

  LinearAllocGuard3D(const hipExtent extent) : LinearAllocGuardMultiDim<T>(extent) {
    HIP_CHECK(hipMalloc3D(&this->pitched_ptr_, this->extent_));
  }

  LinearAllocGuard3D(const LinearAllocGuard3D&) = delete;
  LinearAllocGuard3D(LinearAllocGuard3D&&) = delete;

  size_t depth() const { return this->extent_.depth; }
};

template <typename T> class ArrayAllocGuard {
 public:
  // extent should contain logical width
  ArrayAllocGuard(const hipExtent extent, const unsigned int flags = 0u) : extent_{extent} {
    hipChannelFormatDesc desc = hipCreateChannelDesc<T>();
    HIP_CHECK(hipMalloc3DArray(&ptr_, &desc, extent_, flags));
  }

  ~ArrayAllocGuard() { static_cast<void>(hipFreeArray(ptr_)); }

  ArrayAllocGuard(const ArrayAllocGuard&) = delete;
  ArrayAllocGuard(ArrayAllocGuard&&) = delete;

  hipArray_t ptr() const { return ptr_; }

  hipExtent extent() const { return extent_; }

 private:
  hipArray_t ptr_ = nullptr;
  const hipExtent extent_;
};

template <typename T> class MipmappedArrayAllocGuard {
 public:
  // extent should contain logical width
  MipmappedArrayAllocGuard(const hipExtent extent, const unsigned int levels,
                           const unsigned int flags)
      : extent_{extent}, levels_{levels} {
    hipChannelFormatDesc desc = hipCreateChannelDesc<T>();
    HIP_CHECK(hipMallocMipmappedArray(&ptr_, &desc, extent_, levels_, flags));
  }

  MipmappedArrayAllocGuard(const hipExtent extent, const unsigned int flags = 0u)
      : MipmappedArrayAllocGuard{extent, 1, flags} {}

  ~MipmappedArrayAllocGuard() { static_cast<void>(hipFreeMipmappedArray(ptr_)); }

  MipmappedArrayAllocGuard(const MipmappedArrayAllocGuard&) = delete;
  MipmappedArrayAllocGuard(MipmappedArrayAllocGuard&&) = delete;

  hipMipmappedArray_t ptr() const { return ptr_; }

  hipArray_t GetLevel(unsigned int level) {
    hipArray_t ret;
    HIP_CHECK(hipGetMipmappedArrayLevel(&ret, ptr_, level));
    return ret;
  }

  hipExtent extent() const { return extent_; }

  unsigned int levels() const { return levels_; }

 private:
  hipMipmappedArray_t ptr_ = nullptr;
  const hipExtent extent_;
  const unsigned int levels_;
};

template <typename T> class DrvArrayAllocGuard {
 public:
  // extent should contain width in bytes
  DrvArrayAllocGuard(const hipExtent extent, const unsigned int flags = 0u) : extent_{extent} {
    HIP_ARRAY3D_DESCRIPTOR desc{};
    using vec_info = vector_info<T>;
    desc.Format = vec_info::format;
    desc.NumChannels = vec_info::size;
    desc.Width = extent_.width / sizeof(T);
    desc.Height = extent_.height;
    desc.Depth = extent_.depth;
    desc.Flags = flags;
    HIP_CHECK(hipArray3DCreate(&ptr_, &desc));
  }

  ~DrvArrayAllocGuard() { static_cast<void>(hipArrayDestroy(ptr_)); }

  DrvArrayAllocGuard(const DrvArrayAllocGuard&) = delete;
  DrvArrayAllocGuard(DrvArrayAllocGuard&&) = delete;

  hipArray_t ptr() const { return ptr_; }

  hipExtent extent() const { return extent_; }

 private:
  hipArray_t ptr_ = nullptr;
  const hipExtent extent_;
};

enum class Streams { nullstream, perThread, created, withFlags, withPriority };

class StreamGuard {
 public:
  StreamGuard() = default;

  StreamGuard(const Streams stream_type, unsigned int flags = hipStreamDefault, int priority = 0)
      : stream_type_{stream_type}, flags_{flags}, priority_{priority} {
    switch (stream_type_) {
      case Streams::nullstream:
        stream_ = nullptr;
        break;
      case Streams::perThread:
        stream_ = hipStreamPerThread;
        break;
      case Streams::created:
        HIP_CHECK(hipStreamCreate(&stream_));
        break;
      case Streams::withFlags:
        HIP_CHECK(hipStreamCreateWithFlags(&stream_, flags_));
        break;
      case Streams::withPriority:
        HIP_CHECK(hipStreamCreateWithPriority(&stream_, flags_, priority_));
        break;
    }
  }

  StreamGuard(const StreamGuard&) = delete;

  StreamGuard(StreamGuard&& o) { *this = std::move(o); }

  StreamGuard& operator=(StreamGuard&& o) {
    if (this != &o) {
      if (stream_type_ >= Streams::created) {
        static_cast<void>(hipStreamDestroy(stream_));
      }

      stream_type_ = o.stream_type_;
      flags_ = o.flags_;
      priority_ = o.priority_;
      stream_ = o.stream_;

      o.stream_type_ = Streams::nullstream;
      o.flags_ = 0u;
      o.priority_ = 0;
      o.stream_ = nullptr;
    }

    return *this;
  }

  ~StreamGuard() {
    if (stream_type_ >= Streams::created && stream_ != nullptr) {
      static_cast<void>(hipStreamDestroy(stream_));
    }
  }

  hipStream_t stream() const { return stream_; }

 private:
  Streams stream_type_ = Streams::nullstream;
  unsigned int flags_ = 0u;
  int priority_ = 0;
  hipStream_t stream_ = nullptr;
};

class EventsGuard {
 public:
  EventsGuard(size_t N) : events_(N) {
    for (auto& e : events_) HIP_CHECK(hipEventCreate(&e));
  }

  EventsGuard(const EventsGuard&) = delete;
  EventsGuard(EventsGuard&&) = delete;

  ~EventsGuard() {
    for (auto& e : events_) {
      static_cast<void>(hipEventDestroy(e));
    }
  }

  hipEvent_t& operator[](int index) { return events_[index]; }

  operator hipEvent_t() const { return events_.at(0); }

  std::vector<hipEvent_t>& event_list() { return events_; }

 private:
  std::vector<hipEvent_t> events_;
};

class StreamsGuard {
 public:
  StreamsGuard(size_t N) : streams_(N) {
    for (auto& s : streams_) HIP_CHECK(hipStreamCreate(&s));
  }

  StreamsGuard(const StreamsGuard&) = delete;
  StreamsGuard(StreamsGuard&&) = delete;

  ~StreamsGuard() {
    for (auto& s : streams_) static_cast<void>(hipStreamDestroy(s));
  }

  hipStream_t& operator[](int index) { return streams_[index]; }

  operator hipStream_t() const { return streams_.at(0); }

  std::vector<hipStream_t>& stream_list() { return streams_; }

 private:
  std::vector<hipStream_t> streams_;
};

enum class MemPools { dev_default, created };

class MemPoolGuard {
 public:
  MemPoolGuard(const MemPools mempool_type, int device,
               hipMemAllocationHandleType handle_type = hipMemHandleTypeNone)
      : mempool_type_{mempool_type}, device_{device}, handle_type_{handle_type} {
    switch (mempool_type_) {
      case MemPools::dev_default:
        HIP_CHECK(hipDeviceGetDefaultMemPool(&mempool_, device_));
        break;
      case MemPools::created:
        hipMemPoolProps pool_props;
        memset(&pool_props, 0, sizeof(pool_props));
        pool_props.allocType = hipMemAllocationTypePinned;
        pool_props.handleTypes = handle_type_;
        pool_props.location.type = hipMemLocationTypeDevice;
        pool_props.location.id = device_;
        pool_props.win32SecurityAttributes = nullptr;

        HIP_CHECK(hipMemPoolCreate(&mempool_, &pool_props));
    }
  }

  MemPoolGuard(const MemPoolGuard&) = delete;
  MemPoolGuard(MemPoolGuard&&) = delete;

  ~MemPoolGuard() {
    if (mempool_type_ == MemPools::created) {
      static_cast<void>(hipMemPoolDestroy(mempool_));
    } else {
      // Reset max states for default mem pool, so subtests won't fail
      uint64_t value = 0;
      HIP_CHECK(hipMemPoolSetAttribute(mempool_, hipMemPoolAttrUsedMemHigh, &value));
      HIP_CHECK(hipMemPoolSetAttribute(mempool_, hipMemPoolAttrReservedMemHigh, &value));
    }
  }

  hipMemPool_t mempool() const { return mempool_; }

 private:
  const MemPools mempool_type_;
  int device_;
  hipMemAllocationHandleType handle_type_;
  hipMemPool_t mempool_;
};
