/* Copyright (c) 2018-2022 Advanced Micro Devices, Inc.

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

#ifndef SRC_UTIL_EXCEPTION_H_
#define SRC_UTIL_EXCEPTION_H_

#include <exception>
#include <sstream>
#include <string>

#define EXC_ABORT(error, stream)                                                                   \
  do {                                                                                             \
    std::ostringstream oss;                                                                        \
    oss << __FUNCTION__ << "(), " << stream;                                                       \
    std::cout << oss.str() << std::endl;                                                           \
    abort();                                                                                       \
  } while (0)

#define EXC_RAISING(error, stream)                                                                 \
  do {                                                                                             \
    std::ostringstream oss;                                                                        \
    oss << __FUNCTION__ << "(), " << stream;                                                       \
    throw roctracer::util::exception(error, oss.str());                                            \
  } while (0)

#define HCC_EXC_RAISING(error, stream)                                                             \
  do {                                                                                             \
    EXC_RAISING(error, "HCC error: " << stream);                                                   \
  } while(0)

#define HIP_EXC_RAISING(error, stream)                                                             \
  do {                                                                                             \
    EXC_RAISING(error, "HIP error: " << stream);                                                   \
  } while(0)

namespace roctracer {
namespace util {

class exception : public std::exception {
 public:
  explicit exception(const uint32_t& status, const std::string& msg) : status_(status), str_(msg) {}
  const char* what() const throw() { return str_.c_str(); }
  uint32_t status() const throw() { return status_; }

 protected:
  const uint32_t status_;
  const std::string str_;
};

}  // namespace util
}  // namespace roctracer

#endif  // SRC_UTIL_EXCEPTION_H_
