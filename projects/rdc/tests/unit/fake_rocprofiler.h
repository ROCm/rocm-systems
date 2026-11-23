/*
Copyright (c) 2026 - present Advanced Micro Devices, Inc. All rights reserved.

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

// The controls of the link-time fake of rocprofiler-sdk and of the HSA calls the ROCp module
// makes (fake_rocprofiler.cc). The fake has GPU agents 0..agents-1 whose KFD ids are the fake
// AMD SMI's (0x1000 + index), five counters with one instance each, and contexts that the
// device counting service samples. Its hsa_init() registers the module as a tool the first time,
// as rocprofiler-sdk does, which makes the module's samplers.
#ifndef TESTS_UNIT_FAKE_ROCPROFILER_H_
#define TESTS_UNIT_FAKE_ROCPROFILER_H_

#include <cstdint>
#include <string>
#include <vector>

namespace rdc_fake_rocp {

// The calls a test can fail.
enum class Call { HsaInit, QueryAgents, Start, Sample, Stop, CounterInfoV1 };

// Fails the calls of one agent (every agent with agent -1): `count` calls from the `from`-th
// one (1 = the first), or every call from it with count 0.
struct Failure {
  Call call;
  int agent = -1;
  uint32_t from = 1;
  uint32_t count = 0;
};

struct Config {
  uint32_t agents = 2;
  uint32_t simd_per_cu = 4;
  std::vector<Failure> failures;
};

// Sets the fake up; call it before anything calls hsa_init().
void configure(const Config& config);

// The value the fake samples for a counter on an agent: 1000 * (agent + 1) + the counter's
// number, so that each agent's readings differ.
double value_of(uint32_t agent, const std::string& counter);

// Calls of a kind so far, the failed ones included.
uint32_t calls(Call call);

// hsa_init() calls that succeeded, and hsa_shut_down() calls (each, whether ROCr had a reference).
uint32_t hsa_inits();
uint32_t hsa_shut_downs();

// Whether a context is started and not stopped.
bool any_context_active();

}  // namespace rdc_fake_rocp

#endif  // TESTS_UNIT_FAKE_ROCPROFILER_H_
