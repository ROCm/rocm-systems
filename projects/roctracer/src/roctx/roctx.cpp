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

#include "roctx.h"
#include "roctracer_roctx.h"
#include "ext/prof_protocol.h"

#include <atomic>
#include <cassert>

namespace {

std::atomic<int (*)(activity_domain_t domain, uint32_t operation_id, void* data)> report_activity;
thread_local int nested_range_level{0};

void ReportActivity(roctx_api_id_t operation_id, const char* message = nullptr,
                    roctx_range_id_t id = {}) {
  auto function = report_activity.load(std::memory_order_relaxed);
  if (!function) return;

  roctx_api_data_t api_data{};
  switch (operation_id) {
    case ROCTX_API_ID_roctxMarkA:
      api_data.args.roctxMarkA.message = message;
      break;
    case ROCTX_API_ID_roctxRangePushA:
      api_data.args.roctxRangePushA.message = message;
      break;
    case ROCTX_API_ID_roctxRangePop:
      break;
    case ROCTX_API_ID_roctxRangeStartA:
      api_data.args.roctxRangeStartA.message = message;
      api_data.args.roctxRangeStartA.id = id;
      break;
    case ROCTX_API_ID_roctxRangeStop:
      api_data.args.roctxRangeStop.id = id;
      break;
    default:
      assert(!"should not reach here");
  }
  function(ACTIVITY_DOMAIN_ROCTX, operation_id, &api_data);
}

}  // namespace

ROCTX_API uint32_t roctx_version_major() { return ROCTX_VERSION_MAJOR; }
ROCTX_API uint32_t roctx_version_minor() { return ROCTX_VERSION_MINOR; }

ROCTX_API void roctxMarkA(const char* message) { ReportActivity(ROCTX_API_ID_roctxMarkA, message); }

ROCTX_API int roctxRangePushA(const char* message) {
  ReportActivity(ROCTX_API_ID_roctxRangePushA, message);
  return nested_range_level++;
}

ROCTX_API int roctxRangePop() {
  ReportActivity(ROCTX_API_ID_roctxRangePop);
  if (nested_range_level == 0) return -1;
  return --nested_range_level;
}

ROCTX_API roctx_range_id_t roctxRangeStartA(const char* message) {
  static std::atomic<roctx_range_id_t> start_stop_range_id(1);
  auto range_id = start_stop_range_id++;
  ReportActivity(ROCTX_API_ID_roctxRangeStartA, message, range_id);
  return range_id;
}

ROCTX_API void roctxRangeStop(roctx_range_id_t range_id) {
  ReportActivity(ROCTX_API_ID_roctxRangeStop, nullptr, range_id);
}

extern "C" ROCTX_EXPORT void roctxRegisterTracerCallback(int (*function)(activity_domain_t domain,
                                                                         uint32_t operation_id,
                                                                         void* data)) {
  report_activity.store(function, std::memory_order_relaxed);
}
