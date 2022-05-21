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

#include "callback_table.h"

namespace {

roctracer::CallbackTable<ACTIVITY_DOMAIN_ROCTX, ROCTX_API_ID_NUMBER> callbacks;
thread_local int nested_range_level(0);

}  // namespace

uint32_t ROCTX_API roctx_version_major() { return ROCTX_VERSION_MAJOR; }
uint32_t ROCTX_API roctx_version_minor() { return ROCTX_VERSION_MINOR; }

void ROCTX_API roctxMarkA(const char* message) {
  roctx_api_data_t api_data{};
  api_data.args.roctxMarkA.message = message;
  callbacks.Invoke(ROCTX_API_ID_roctxMarkA, &api_data);
}

int ROCTX_API roctxRangePushA(const char* message) {
  roctx_api_data_t api_data{};
  api_data.args.roctxRangePushA.message = message;
  callbacks.Invoke(ROCTX_API_ID_roctxRangePushA, &api_data);

  return nested_range_level++;
}

int ROCTX_API roctxRangePop() {
  roctx_api_data_t api_data{};
  callbacks.Invoke(ROCTX_API_ID_roctxRangePop, &api_data);

  if (nested_range_level == 0) return -1;
  return --nested_range_level;
}

roctx_range_id_t ROCTX_API roctxRangeStartA(const char* message) {
  static std::atomic<roctx_range_id_t> start_stop_range_id(1);

  roctx_api_data_t api_data{};
  api_data.args.roctxRangeStartA.message = message;
  callbacks.Invoke(ROCTX_API_ID_roctxRangeStartA, &api_data);

  return start_stop_range_id++;
}

void ROCTX_API roctxRangeStop(roctx_range_id_t rangeId) {
  roctx_api_data_t api_data{};
  api_data.args.roctxRangeStop.id = rangeId;
  callbacks.Invoke(ROCTX_API_ID_roctxRangeStop, &api_data);
}

extern "C" bool ROCTX_EXPORT RegisterApiCallback(uint32_t op, void* callback, void* arg) {
  if (op >= ROCTX_API_ID_NUMBER) return false;
  callbacks.Set(op, reinterpret_cast<activity_rtapi_callback_t>(callback), arg);
  return true;
}

extern "C" bool ROCTX_EXPORT RemoveApiCallback(uint32_t op) {
  if (op >= ROCTX_API_ID_NUMBER) return false;
  callbacks.Set(op, nullptr, nullptr);
  return true;
}