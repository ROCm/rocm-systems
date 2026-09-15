// MIT License
//
// Copyright (c) 2017-2026 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#ifndef AQLPROFILE_SRC_CORE_ERROR_STRING_UTIL_H_
#define AQLPROFILE_SRC_CORE_ERROR_STRING_UTIL_H_

#include <string>

namespace aql_profile {

// Return a C string for `message` that stays valid until the next call on the
// same thread (the strerror()/dlerror() contract). The message is copied into
// thread-local storage rather than aliasing a caller-owned buffer that may be
// overwritten or reallocated after this function returns.
inline const char* StableErrorString(const std::string& message) {
  // Copy into thread-local storage so the returned pointer stays valid until the
  // next call on this thread, independent of the caller's buffer lifetime.
  thread_local std::string storage;
  storage = message;
  return storage.c_str();
}

}  // namespace aql_profile

#endif  // AQLPROFILE_SRC_CORE_ERROR_STRING_UTIL_H_
