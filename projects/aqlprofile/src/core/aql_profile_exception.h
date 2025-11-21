// MIT License
//
// Copyright (c) 2017-2025 Advanced Micro Devices, Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#ifndef SRC_CORE_AQL_PROFILE_EXCEPTION_H_
#define SRC_CORE_AQL_PROFILE_EXCEPTION_H_

#include <string.h>

#include <sstream>
#include <string>

namespace aql_profile {

class aql_profile_exc_msg : public std::exception {
 public:
  explicit aql_profile_exc_msg(const std::string& msg) : str_(msg) {}
  virtual const char* what() const throw() { return str_.c_str(); }

 protected:
  std::string str_;
};

template <typename T>
class aql_profile_exc_val : public std::exception {
 public:
  aql_profile_exc_val(const std::string& msg, const T& val) {
    std::ostringstream oss;
    oss << msg << "(" << val << ")";
    str_ = oss.str();
  }
  virtual const char* what() const throw() { return str_.c_str(); }

 protected:
  std::string str_;
};
}  // namespace aql_profile

#endif  // SRC_CORE_AQL_PROFILE_EXCEPTION_H_
