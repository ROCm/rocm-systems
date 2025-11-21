/******************************************************************************
Copyright (c) 2018 Advanced Micro Devices, Inc. All rights reserved.

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
*******************************************************************************/

#ifndef SRC_UTIL_EXCEPTION_H_
#define SRC_UTIL_EXCEPTION_H_

#include <hsa/hsa.h>
#include <hsa/hsa_ven_amd_aqlprofile.h>

#include <exception>
#include <sstream>
#include <string>

#define EXC_ABORT(error, stream)                                                                   \
  do {                                                                                             \
    const char* hsa_err_str = NULL;                                                                \
    if (hsa_status_string(error, &hsa_err_str) != HSA_STATUS_SUCCESS) hsa_err_str = NULL;          \
    std::ostringstream oss;                                                                        \
    oss << "error(" << error << ") \"" << __FUNCTION__ << "(), " << stream << "\"" << std::endl;   \
    if (hsa_err_str != NULL) oss << hsa_err_str << std::endl;                                      \
    std::cout << oss.str() << std::flush;                                                          \
    abort();                                                                                       \
  } while (0)

#define EXC_RAISING(error, stream)                                                                 \
  do {                                                                                             \
    std::ostringstream oss;                                                                        \
    oss << __FUNCTION__ << "(), " << stream;                                                       \
    throw rocprofiler::util::exception(error, oss.str());                                          \
  } while (0)

#define AQL_EXC_RAISING(error, stream)                                                             \
  do {                                                                                             \
    const char* error_string = NULL;                                                               \
    hsa_ven_amd_aqlprofile_error_string(&error_string);                                            \
    EXC_RAISING(error, stream << ", " << error_string);                                            \
  } while (0)

namespace rocprofiler {
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
}  // namespace rocprofiler

#endif  // SRC_UTIL_EXCEPTION_H_
