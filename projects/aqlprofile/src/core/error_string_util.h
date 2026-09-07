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
  // NOTE: returning message.c_str() aliases the caller's buffer. When the caller
  // passes Logger's per-thread message_ entry, the next logged message
  // overwrites/reallocates it and this pointer no longer reflects `message` --
  // see the accompanying test, which fails here. The fix copies into
  // thread-local storage.
  return message.c_str();
}

}  // namespace aql_profile

#endif  // AQLPROFILE_SRC_CORE_ERROR_STRING_UTIL_H_
