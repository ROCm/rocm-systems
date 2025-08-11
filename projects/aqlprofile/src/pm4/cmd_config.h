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

#ifndef SRC_PM4_CMD_CONFIG_H_
#define SRC_PM4_CMD_CONFIG_H_

#include <stdint.h>
#include "pm4/trace_config.h"

namespace pm4_builder {
// Counters vector class
class counters_vector : public std::vector<counter_des_t> {
 public:
  typedef std::vector<counter_des_t> Parent;

  counters_vector() : Parent(), attr_(0) {}

  void push_back(const counter_des_t& des) {
    Parent::push_back(des);
    attr_ |= des.block_info->attr;
  }

  uint32_t get_attr() const { return attr_; }

 private:
  uint32_t attr_;
};

}  // namespace pm4_builder

#endif  // SRC_PM4_CMD_CONFIG_H_
