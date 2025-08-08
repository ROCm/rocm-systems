////////////////////////////////////////////////////////////////////////////////
//
// The University of Illinois/NCSA
// Open Source License (NCSA)
//
// Copyright (c) 2014-2020, Advanced Micro Devices, Inc. All rights reserved.
//
// Developed by:
//
//                 AMD Research and AMD HSA Software Development
//
//                 Advanced Micro Devices, Inc.
//
//                 www.amd.com
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to
// deal with the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
//  - Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimers.
//  - Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimers in
//    the documentation and/or other materials provided with the distribution.
//  - Neither the names of Advanced Micro Devices, Inc,
//    nor the names of its contributors may be used to endorse or promote
//    products derived from this Software without specific prior written
//    permission.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
// THE CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
// OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
// ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS WITH THE SOFTWARE.
//
////////////////////////////////////////////////////////////////////////////////

#ifndef HSA_RUNTIME_CORE_INC_EXCEPTIONS_H
#define HSA_RUNTIME_CORE_INC_EXCEPTIONS_H

#include <exception>
#include <string>

#include "core/inc/hsa_internal.h"

namespace rocr {
namespace AMD {

/// @brief Exception type which carries an error code to return to the user.
class hsa_exception : public std::exception {
 public:
  hsa_exception(hsa_status_t error, const char* description) : err_(error), desc_(description) {}
  hsa_status_t error_code() const noexcept { return err_; }
  const char* what() const noexcept override { return desc_.c_str(); }

 private:
  hsa_status_t err_;
  std::string desc_;
};

/// @brief Holds and invokes callbacks, capturing any execptions and forwarding those to the user
/// after unwinding the runtime stack.
template <class F> class callback_t;
template <class R, class... Args> class callback_t<R (*)(Args...)> {
 public:
  typedef R (*func_t)(Args...);

  callback_t() : function(nullptr) {}

  // Should not be marked explicit.
  callback_t(func_t function_ptr) : function(function_ptr) {}
  callback_t& operator=(func_t function_ptr) { function = function_ptr; return *this; }

  bool operator==(func_t function_ptr) { return function == function_ptr; }
  bool operator!=(func_t function_ptr) { return function != function_ptr; }

  // Allows common function pointer idioms, such as if( func != nullptr )...
  // without allowing silent reversion to the original function pointer type.
  operator void*() { return reinterpret_cast<void*>(function); }

  R operator()(Args... args) {
    try {
      return function(args...);
    } catch (...) {
      throw std::nested_exception();
    }
  }

 private:
  func_t function;
};

}  // namespace amd
}  // namespace rocr

#endif  // HSA_RUNTIME_CORE_INC_EXCEPTIONS_H
