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

#ifndef HSA_RUNTME_CORE_INC_CHECKED_H_
#define HSA_RUNTME_CORE_INC_CHECKED_H_

#include <stdint.h>
#include <stdlib.h>

namespace rocr {
namespace core {

/// @brief Compares type codes and pointers to check object validity.  Used for cast validation.
template <uint64_t code, bool multiProcess = false> class Check final {
 public:
  typedef Check<code> CheckType;

  Check() { object_ = uintptr_t(this) ^ uintptr_t(code); }
  Check(const Check&) { object_ = uintptr_t(this) ^ uintptr_t(code); }
  Check(Check&&) { object_ = uintptr_t(this) ^ uintptr_t(code); }

  ~Check() { object_ = uintptr_t(NULL); }

  const Check& operator=(Check&& rhs) { return *this; }
  const Check& operator=(const Check& rhs) { return *this; }

  bool IsValid() const {
    return object_ == (uintptr_t(this) ^ uintptr_t(code));
  }

  uint64_t check_code() const { return code; }

 private:
  uintptr_t object_;
};

template <uint64_t code> class Check<code, true> final {
 public:
  typedef Check<code> CheckType;

  Check() { object_ = uintptr_t(code); }
  Check(const Check&) { object_ = uintptr_t(code); }
  Check(Check&&) { object_ = uintptr_t(code); }

  const Check& operator=(Check&& rhs) { return *this; }
  const Check& operator=(const Check& rhs) { return *this; }

  bool IsValid() const { return object_ == uintptr_t(code); }

  uint64_t check_code() const { return code; }

 private:
  uintptr_t object_;
};

/// @brief Base class for validating objects.
template <uint64_t code> class Checked {
 public:
  typedef Checked<code> CheckedType;

  bool IsValid() const { return id.IsValid(); }

  virtual ~Checked() {}

 private:
  Check<code, false> id;
};

}  // namespace core
}  // namespace rocr

#endif  // header guard
