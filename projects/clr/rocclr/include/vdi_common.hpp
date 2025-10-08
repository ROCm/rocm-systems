/* Copyright (c) 2020 - 2021 Advanced Micro Devices, Inc.

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

#ifndef VDI_COMMON_HPP_
#define VDI_COMMON_HPP_

#include "top.hpp"
#include "platform/runtime.hpp"
#include "platform/command.hpp"
#include "platform/memory.hpp"
#include "thread/thread.hpp"
#include "platform/commandqueue.hpp"

#include <vector>
#include <utility>

namespace device = amd::device;

//! \cond ignore
namespace amd {

template <typename T> class NotNullWrapper {
 private:
  T* const ptrOrNull_;

 protected:
  explicit NotNullWrapper(T* ptrOrNull) : ptrOrNull_(ptrOrNull) {}

 public:
  void operator=(T value) const {
    if (ptrOrNull_ != NULL) {
      *ptrOrNull_ = value;
    }
  }
};

template <typename T> class NotNullReference : protected NotNullWrapper<T> {
 public:
  explicit NotNullReference(T* ptrOrNull) : NotNullWrapper<T>(ptrOrNull) {}

  const NotNullWrapper<T>& operator*() const { return *this; }
};

}  // namespace amd

template <typename T> inline amd::NotNullReference<T> not_null(T* ptrOrNull) {
  return amd::NotNullReference<T>(ptrOrNull);
}

#define RUNTIME_ENTRY_RET(ret, func, args) CL_API_ENTRY ret CL_API_CALL func args {
#define RUNTIME_ENTRY_RET_NOERRCODE(ret, func, args) CL_API_ENTRY ret CL_API_CALL func args {
#define RUNTIME_ENTRY(ret, func, args) CL_API_ENTRY ret CL_API_CALL func args {
#define RUNTIME_ENTRY_VOID(ret, func, args) CL_API_ENTRY ret CL_API_CALL func args {
#define RUNTIME_EXIT                                                                               \
  /* FIXME_lmoriche: we should check to thread->lastError here! */                                 \
  }

namespace amd {

namespace detail {

template <typename T> struct ParamInfo {
  static inline std::pair<const void*, size_t> get(const T& param) {
    return std::pair<const void*, size_t>(&param, sizeof(T));
  }
};

template <> struct ParamInfo<const char*> {
  static inline std::pair<const void*, size_t> get(const char* param) {
    return std::pair<const void*, size_t>(param, strlen(param) + 1);
  }
};

template <int N> struct ParamInfo<char[N]> {
  static inline std::pair<const void*, size_t> get(const char* param) {
    return std::pair<const void*, size_t>(param, strlen(param) + 1);
  }
};

}  // namespace detail

struct PlatformIDS {
  const cl_icd_dispatch* dispatch_;
};
class PlatformID {
 public:
  static inline PlatformIDS Platform = {amd::ICDDispatchedObject::icdVendorDispatch_};
};
#define AMD_PLATFORM (reinterpret_cast<cl_platform_id>(&amd::PlatformID::Platform))

}  // namespace amd

#endif /* _VDI_COMMON_H */
