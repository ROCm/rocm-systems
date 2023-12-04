/*
Copyright (c) 2021 - present Advanced Micro Devices, Inc. All rights reserved.

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

#ifndef COMMON_RDC_CAPABILITIES_H_
#define COMMON_RDC_CAPABILITIES_H_

#include <sys/capability.h>

namespace amd {
namespace rdc {

int GetCapability(cap_value_t cap, cap_flag_t cap_type, bool* enabled);
int ModifyCapability(cap_value_t cap, cap_flag_t cap_type, bool enable);

struct ScopedCapability {
  ScopedCapability(cap_value_t cp, cap_flag_t cpt) : cap_(cp), cap_type_(cpt), error_(0) {
    error_ = ModifyCapability(cap_, cap_type_, true);
  }
  ~ScopedCapability() { error_ = ModifyCapability(cap_, cap_type_, false); }
  void Relinquish(void) { error_ = ModifyCapability(cap_, cap_type_, false); }
  int error(void) { return error_; }

 private:
  cap_value_t cap_;
  cap_flag_t cap_type_;
  int error_;
};

}  // namespace rdc
}  // namespace amd

#endif  // COMMON_RDC_CAPABILITIES_H_
