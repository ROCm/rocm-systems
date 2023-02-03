/* Copyright (c) 2022 Advanced Micro Devices, Inc.

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

#ifndef SRC_CORE_COUNTERS_BASIC_BASIC_COUNTER_H_
#define SRC_CORE_COUNTERS_BASIC_BASIC_COUNTER_H_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <map>
#include <unordered_map>
#include <vector>

#include "src/core/counters/counter.h"

#define ASSERTM(exp, msg) assert(((void)msg, exp))

namespace Counter {

class BasicCounter : Counter {
 public:
  BasicCounter(uint64_t event_id, std::string block_id, std::string name, std::string description,
               std::string gpu_name);
  ~BasicCounter();

  uint64_t GetEventId();
  std::string GetBlockId();
  std::string GetName();
  uint64_t GetBasicCounterID();
  bool GetValue(uint64_t* value, int64_t instance_id);
  uint64_t GetValue(int64_t instance_id = -1);

  uint64_t avr(int64_t instances_count);
  uint64_t max(int64_t instances_count);
  uint64_t min(int64_t instances_count);
  uint64_t sum(int64_t instances_count);

 private:
  void* counter_hw_info;
  std::unordered_map<int64_t, uint64_t> instances_values_;
  uint64_t event_id_;
  std::string block_id_;
};

uint64_t operator+(BasicCounter counter, const uint64_t number);
uint64_t operator*(BasicCounter counter, const uint64_t number);
uint64_t operator/(BasicCounter counter, const uint64_t number);
uint64_t operator-(BasicCounter counter, const uint64_t number);
uint64_t operator^(BasicCounter counter, const uint64_t number);

uint64_t operator+(BasicCounter counter1, BasicCounter counter2);
uint64_t operator*(BasicCounter counter1, BasicCounter counter2);
uint64_t operator/(BasicCounter counter1, BasicCounter counter2);
uint64_t operator-(BasicCounter counter1, BasicCounter counter2);
uint64_t operator^(BasicCounter counter1, BasicCounter counter2);

BasicCounter* GetGeneratedBasicCounter(uint64_t id);
void ClearBasicCounters();

uint64_t GetBasicCounter(const char* name, const char* gpu_name);

}  // namespace Counter

#endif  // SRC_CORE_COUNTERS_BASIC_BASIC_COUNTER_H_
