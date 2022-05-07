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

#include "inc/roctx.h"
#include "inc/roctracer_roctx.h"

#include <cassert>

#include "inc/ext/prof_protocol.h"
#include "core/callback_table.h"
#include "util/exception.h"
#include "util/logger.h"

#define PUBLIC_API __attribute__((visibility("default")))

#define API_METHOD_PREFIX try {
#define API_METHOD_SUFFIX_NRET                                                                     \
  }                                                                                                \
  catch (const std::exception& e) {                                                                \
    ERR_LOGGING(__FUNCTION__ << "(), " << e.what());                                               \
  }

#define API_METHOD_CATCH(X)                                                                        \
  }                                                                                                \
  catch (const std::exception& e) {                                                                \
    ERR_LOGGING(__FUNCTION__ << "(), " << e.what());                                               \
    return X;                                                                                      \
  }                                                                                                \
  assert(false && "should not reach here");

////////////////////////////////////////////////////////////////////////////////
// Library errors enumeration
typedef enum {
  ROCTX_STATUS_SUCCESS = 0,
  ROCTX_STATUS_ERROR = 1,
} roctx_status_t;

///////////////////////////////////////////////////////////////////////////////////////////////////
// Library implementation
//
namespace {

roctracer::CallbackTable<ACTIVITY_DOMAIN_ROCTX, ROCTX_API_ID_NUMBER> callbacks;
thread_local int range_level(0);

}  // namespace

// Logger instantiation
roctracer::util::Logger::mutex_t roctracer::util::Logger::mutex_;
std::atomic<roctracer::util::Logger*> roctracer::util::Logger::instance_{};

///////////////////////////////////////////////////////////////////////////////////////////////////
// Public library methods
//

PUBLIC_API uint32_t roctx_version_major() { return ROCTX_VERSION_MAJOR; }
PUBLIC_API uint32_t roctx_version_minor() { return ROCTX_VERSION_MINOR; }

PUBLIC_API void roctxMarkA(const char* message) {
  API_METHOD_PREFIX
  roctx_api_data_t api_data{};
  api_data.args.roctxMarkA.message = message;
  callbacks.Invoke(ROCTX_API_ID_roctxMarkA, &api_data);
  API_METHOD_SUFFIX_NRET
}

PUBLIC_API int roctxRangePushA(const char* message) {
  API_METHOD_PREFIX
  roctx_api_data_t api_data{};
  api_data.args.roctxRangePushA.message = message;
  callbacks.Invoke(ROCTX_API_ID_roctxRangePushA, &api_data);

  return range_level++;
  API_METHOD_CATCH(-1);
}

PUBLIC_API int roctxRangePop() {
  API_METHOD_PREFIX

  roctx_api_data_t api_data{};
  callbacks.Invoke(ROCTX_API_ID_roctxRangePop, &api_data);

  if (range_level == 0) EXC_RAISING(ROCTX_STATUS_ERROR, "Pop from empty stack!");
  return --range_level;
  API_METHOD_CATCH(-1)
}

PUBLIC_API roctx_range_id_t roctxRangeStartA(const char* message) {
  API_METHOD_PREFIX
  static std::atomic<roctx_range_id_t> roctx_range_counter(1);

  roctx_api_data_t api_data{};
  api_data.args.roctxRangeStartA.message = message;
  callbacks.Invoke(ROCTX_API_ID_roctxRangeStartA, &api_data);

  return roctx_range_counter++;
  API_METHOD_CATCH(-1)
}

PUBLIC_API void roctxRangeStop(roctx_range_id_t rangeId) {
  API_METHOD_PREFIX
  roctx_api_data_t api_data{};
  api_data.args.roctxRangeStop.id = rangeId;
  callbacks.Invoke(ROCTX_API_ID_roctxRangeStop, &api_data);
  API_METHOD_SUFFIX_NRET
}

extern "C" PUBLIC_API bool RegisterApiCallback(uint32_t op, void* callback, void* arg) {
  if (op >= ROCTX_API_ID_NUMBER) return false;
  callbacks.Set(op, reinterpret_cast<activity_rtapi_callback_t>(callback), arg);
  return true;
}

extern "C" PUBLIC_API bool RemoveApiCallback(uint32_t op) {
  if (op >= ROCTX_API_ID_NUMBER) return false;
  callbacks.Set(op, nullptr, nullptr);
  return true;
}