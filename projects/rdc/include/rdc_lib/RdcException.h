/*
Copyright (c) 2019 - present Advanced Micro Devices, Inc. All rights reserved.

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

#ifndef INCLUDE_RDC_LIB_RDCEXCEPTION_H_
#define INCLUDE_RDC_LIB_RDCEXCEPTION_H_

#include <exception>
#include <string>

#include "rdc/rdc.h"

namespace amd {
namespace rdc {

class RdcException : public std::exception {
 public:
  RdcException(rdc_status_t error, const std::string description)
      : err_(error), desc_(description) {}
  rdc_status_t error_code() const noexcept { return err_; }
  const char* what() const noexcept override { return desc_.c_str(); }

 private:
  rdc_status_t err_;
  std::string desc_;
};

}  // namespace rdc
}  // namespace amd

#endif  // INCLUDE_RDC_LIB_RDCEXCEPTION_H_
