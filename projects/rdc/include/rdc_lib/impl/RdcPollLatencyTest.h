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
#ifndef INCLUDE_RDC_LIB_IMPL_RDCPOLLLATENCYTEST_H_
#define INCLUDE_RDC_LIB_IMPL_RDCPOLLLATENCYTEST_H_

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>

#include "rdc/rdc.h"

namespace amd {
namespace rdc {

//! Settings of the poll latency check. The test config can override them, for
//! example "samples=50;interval_ms=1000".
struct PollLatencyConfig {
  uint32_t samples = 20;       //!< Number of timed polls
  uint32_t interval_ms = 100;  //!< Time from the start of one poll to the next
};

//! Field values that one read returned, by outcome.
struct PollValueCounts {
  uint32_t ok = 0;
  uint32_t not_supported = 0;
  uint32_t failed = 0;
};

//! Reads every field of one GPU once. Sets *elapsed_us to the time the read
//! took and adds the values it returned to *counts. Fields it returns no
//! value for count as failed.
using PollReadFn =
    std::function<rdc_status_t(uint32_t gpu_index, uint64_t* elapsed_us, PollValueCounts* counts)>;

//! Clock that spaces the polls. Unit tests replace it so that they do not sleep.
struct PollClock {
  std::function<std::chrono::steady_clock::time_point()> now;
  std::function<void(std::chrono::steady_clock::time_point)> wait_until;
};

//! The steady clock, waiting with std::this_thread::sleep_until.
PollClock real_poll_clock();

//! Parses the test config into *cfg. A null or empty config keeps *cfg.
//! Returns RDC_ST_BAD_PARAMETER and sets *error for an unknown setting, a
//! value out of range, or samples x interval_ms over one hour; *cfg is then
//! unchanged.
rdc_status_t parse_poll_latency_config(const char* config, size_t config_size,
                                       PollLatencyConfig* cfg, std::string* error);

//! Reads each GPU once untimed, then polls all GPUs cfg.samples times,
//! cfg.interval_ms apart, and reports the poll and read times in *result.
//! field_count is the number of fields that one read of one GPU returns.
rdc_status_t run_poll_latency_check(const uint32_t* gpu_index, uint32_t gpu_count,
                                    uint32_t field_count, const PollLatencyConfig& cfg,
                                    const PollReadFn& read, const PollClock& clock,
                                    rdc_diag_test_result_t* result);

}  // namespace rdc
}  // namespace amd

#endif  // INCLUDE_RDC_LIB_IMPL_RDCPOLLLATENCYTEST_H_
