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

#include "counter.h"

#include <cassert>

#include "src/api/rocprofiler_singleton.h"

namespace Counter {

std::atomic<uint64_t> COUNTER_COUNT{1};

Counter::Counter(std::string name, std::string description, std::string gpu_name)
    : name_(name), description_(description), gpu_name_(gpu_name) {}

Counter::~Counter() {}

void Counter::AddCounterToCounterMap() { GenerateCounterID(); }

uint64_t Counter::GetCounterID() { return counter_id_; }
void Counter::GenerateCounterID() {
  counter_id_ = COUNTER_COUNT.fetch_add(1, std::memory_order_release);
}

std::string Counter::GetName() { return name_; }
std::string Counter::GetDescription() { return description_; }

}  // namespace Counter
