/*
Copyright (c) 2015-2016 Advanced Micro Devices, Inc. All rights reserved.

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

#include <sstream>
#include <string>

#include <stdio.h>

#include <inc/roctracer_hsa.h>
#include <inc/ext/hsa_rt_utils.hpp>

#define PUBLIC_API __attribute__((visibility("default")))
#define CONSTRUCTOR_API __attribute__((constructor))
#define DESTRUCTOR_API __attribute__((destructor))

typedef hsa_rt_utils::Timer::timestamp_t timestamp_t;
hsa_rt_utils::Timer timer(NULL);
thread_local timestamp_t begin_timestamp = 0;

// HSA API callback function
void hsa_api_callback(
    uint32_t domain,
    uint32_t cid,
    const void* callback_data,
    void* arg)
{
  (void)arg;

  const hsa_api_data_t* data = reinterpret_cast<const hsa_api_data_t*>(callback_data);

  if (data->phase == ACTIVITY_API_PHASE_ENTER) {
    begin_timestamp = timer.timestamp_fn_ns();
  } else {
    const timestamp_t end_timestamp = (cid == HSA_API_ID_hsa_shut_down) ? begin_timestamp : timer.timestamp_fn_ns();
    std::ostringstream os;
    os << '(' << begin_timestamp << ":" << end_timestamp << ") " << hsa_api_data_pair_t(cid, *data);
    fprintf(stdout, "%s\n", os.str().c_str());
  }
}

extern "C" {
// HSA-runtime tool on-load method
PUBLIC_API bool OnLoad(HsaApiTable* table, uint64_t runtime_version, uint64_t failed_tool_count,
                       const char* const* failed_tool_names) {
  timer.init(table->core_->hsa_system_get_info_fn);
  roctracer_enable_callback(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_ANY, hsa_api_callback, NULL);
  return true;
}
}
