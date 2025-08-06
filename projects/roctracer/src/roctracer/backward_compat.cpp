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

#include "roctracer.h"

extern "C" {

// Deprecated functions:
ROCTRACER_API int roctracer_load() { return 1; }
ROCTRACER_API void roctracer_unload() {}
ROCTRACER_API void roctracer_flush_buf() {}
ROCTRACER_API void roctracer_mark(const char*) {}

ROCTRACER_API roctracer_status_t roctracer_enable_callback(roctracer_rtapi_callback_t callback,
                                                           void* user_data) {
  for (uint32_t domain = 0; domain < ACTIVITY_DOMAIN_NUMBER; ++domain)
    if (auto status =
            roctracer_enable_domain_callback((roctracer_domain_t)domain, callback, user_data);
        status != ROCTRACER_STATUS_SUCCESS)
      return status;
  return ROCTRACER_STATUS_SUCCESS;
}

ROCTRACER_API roctracer_status_t roctracer_disable_callback() {
  for (uint32_t domain = 0; domain < ACTIVITY_DOMAIN_NUMBER; ++domain)
    if (auto status = roctracer_disable_domain_callback((roctracer_domain_t)domain);
        status != ROCTRACER_STATUS_SUCCESS)
      return status;
  return ROCTRACER_STATUS_SUCCESS;
}

ROCTRACER_API roctracer_status_t roctracer_enable_activity_expl(roctracer_pool_t* pool) {
  for (uint32_t domain = 0; domain < ACTIVITY_DOMAIN_NUMBER; ++domain)
    if (auto status = roctracer_enable_domain_activity_expl((roctracer_domain_t)domain, pool);
        status != ROCTRACER_STATUS_SUCCESS)
      return status;
  return ROCTRACER_STATUS_SUCCESS;
}

ROCTRACER_API roctracer_status_t roctracer_enable_activity() {
  for (uint32_t domain = 0; domain < ACTIVITY_DOMAIN_NUMBER; ++domain)
    if (auto status = roctracer_enable_domain_activity((roctracer_domain_t)domain);
        status != ROCTRACER_STATUS_SUCCESS)
      return status;
  return ROCTRACER_STATUS_SUCCESS;
}

ROCTRACER_API roctracer_status_t roctracer_disable_activity() {
  for (uint32_t domain = 0; domain < ACTIVITY_DOMAIN_NUMBER; ++domain)
    if (auto status = roctracer_disable_domain_activity((roctracer_domain_t)domain);
        status != ROCTRACER_STATUS_SUCCESS)
      return status;
  return ROCTRACER_STATUS_SUCCESS;
}

}  // extern "C"
