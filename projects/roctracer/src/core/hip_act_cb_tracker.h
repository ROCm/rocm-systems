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

#ifndef CORE_HIP_ACT_CB_TRACKER_H_
#define CORE_HIP_ACT_CB_TRACKER_H_

#include <map>

namespace roctracer {
enum { API_CB_MASK = 0x1, ACT_CB_MASK = 0x2 };

class hip_act_cb_tracker_t {
  private:
  std::map<uint32_t, uint32_t> data;

 public:
  uint32_t enable_check(uint32_t op, uint32_t mask) { return data[op] |= mask; }

  uint32_t disable_check(uint32_t op, uint32_t mask) { return data[op] &= ~mask; }
};  // hip_act_cb_tracker_t
};  // namespace roctracer

#endif  // CORE_HIP_ACT_CB_TRACKER_H_
