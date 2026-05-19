/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

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
