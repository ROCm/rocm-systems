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

#ifndef SRC_CORE_COUNTERS_COUNTER_H_
#define SRC_CORE_COUNTERS_COUNTER_H_

#include <atomic>
#include <cstdint>
#include <string>
#include <unordered_map>

namespace Counter {

class Counter {
 public:
  Counter(std::string name, std::string description, std::string gpu_name);
  ~Counter();
  void AddCounterToCounterMap();
  uint64_t GetCounterID();
  void GenerateCounterID();
  std::string GetName();
  std::string GetDescription();

 private:
  uint64_t counter_id_;
  std::string name_;
  std::string description_;
  std::string gpu_name_;
};

std::string GetCounterName(uint64_t descriptor);

}  // namespace Counter

#endif  // SRC_CORE_COUNTERS_COUNTER_H_
