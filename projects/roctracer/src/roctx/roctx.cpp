/*
Copyright (c) 2018 Advanced Micro Devices, Inc. All rights reserved.

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

#include "inc/roctx.h"
#include "inc/roctracer_roctx.h"

#include <string.h>

#include "inc/ext/prof_protocol.h"
#include "util/exception.h"
#include "util/logger.h"

#define PUBLIC_API __attribute__((visibility("default")))
#define CONSTRUCTOR_API __attribute__((constructor))
#define DESTRUCTOR_API __attribute__((destructor))

#define API_METHOD_PREFIX                                                                          \
  roctx_status_t err = ROCTX_STATUS_SUCCESS;                                                       \
  try {

#define API_METHOD_SUFFIX                                                                          \
  }                                                                                                \
  catch (std::exception & e) {                                                                     \
    ERR_LOGGING(__FUNCTION__ << "(), " << e.what());                                               \
    err = roctx::GetExcStatus(e);                                                                  \
  }                                                                                                \
  return (err == ROCTX_STATUS_SUCCESS) ? 0 : -1;

#define API_METHOD_SUFFIX_NRET                                                                     \
  }                                                                                                \
  catch (std::exception & e) {                                                                     \
    ERR_LOGGING(__FUNCTION__ << "(), " << e.what());                                               \
    err = roctx::GetExcStatus(e);                                                                  \
  }                                                                                                \
  (void)err;                                                                                       \

#define API_METHOD_CATCH(X)                                                                        \
  }                                                                                                \
  catch (std::exception & e) {                                                                     \
    ERR_LOGGING(__FUNCTION__ << "(), " << e.what());                                               \
  }                                                                                                \
  (void)err;                                                                                       \
  return X;

static inline uint32_t GetPid() { return syscall(__NR_getpid); }
static inline uint32_t GetTid() { return syscall(__NR_gettid); }

////////////////////////////////////////////////////////////////////////////////
// Library errors enumaration
typedef enum {
  ROCTX_STATUS_SUCCESS = 0,
  ROCTX_STATUS_ERROR = 1,
} roctx_status_t;

///////////////////////////////////////////////////////////////////////////////////////////////////
// Library implementation
//
namespace roctx {

roctx_status_t GetExcStatus(const std::exception& e) {
  const roctracer::util::exception* roctx_exc_ptr = dynamic_cast<const roctracer::util::exception*>(&e);
  return (roctx_exc_ptr) ? static_cast<roctx_status_t>(roctx_exc_ptr->status()) : ROCTX_STATUS_ERROR;
}

// callbacks table
extern cb_table_t cb_table;
}  // namespace roctx

// Logger instantiation
roctracer::util::Logger::mutex_t roctracer::util::Logger::mutex_;
std::atomic<roctracer::util::Logger*> roctracer::util::Logger::instance_{};

///////////////////////////////////////////////////////////////////////////////////////////////////
// Public library methods
//
extern "C" {

PUBLIC_API uint32_t roctx_version_major() { return ROCTX_VERSION_MAJOR; }
PUBLIC_API uint32_t roctx_version_minor() { return ROCTX_VERSION_MINOR; }

PUBLIC_API const char* roctracer_error_string() {
  return strdup(roctracer::util::Logger::LastMessage().c_str());
}

PUBLIC_API void roctxMarkA(const char* message) {
  API_METHOD_PREFIX
  roctx_api_data_t api_data{};
  api_data.args.roctxMarkA.message = strdup(message);
  activity_rtapi_callback_t api_callback_fun = NULL;
  void* api_callback_arg = NULL;
  roctx::cb_table.get(ROCTX_API_ID_roctxMarkA, &api_callback_fun, &api_callback_arg);
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_ROCTX, ROCTX_API_ID_roctxMarkA, &api_data, api_callback_arg);
  API_METHOD_SUFFIX_NRET
}

PUBLIC_API int roctxRangePushA(const char* message) {
  API_METHOD_PREFIX
  EXC_ABORT(ROCTX_STATUS_ERROR, "method is not implemented");
  API_METHOD_SUFFIX
}

PUBLIC_API int roctxRangePop() {
  API_METHOD_PREFIX
  EXC_ABORT(ROCTX_STATUS_ERROR, "method is not implemented");
  API_METHOD_SUFFIX
}

}  // extern "C"
