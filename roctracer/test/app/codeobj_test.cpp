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

#include <cstring>
#include <cstdio>
#include <cstdlib>

#include "roctracer.h"
#include "roctracer_hsa.h"

namespace {
// Check returned HSA API status
inline void CHECK(roctracer_status_t status) {
  if (status != ROCTRACER_STATUS_SUCCESS) {
    fprintf(stderr, "ERROR: %s\n", roctracer_error_string());
    abort();
  }
}

// codeobj callback
void CodeObjectCallback(uint32_t domain, uint32_t cid, const void* data, void* arg) {
  const hsa_evt_data_t* evt_data = reinterpret_cast<const hsa_evt_data_t*>(data);
  fprintf(stdout,
          "codeobj_callback domain(%u) cid(%u): load_base(0x%lx) load_size(0x%lx) "
          "load_delta(0x%lx) uri(\"%s\") unload(%d)\n",
          domain, cid, evt_data->codeobj.load_base, evt_data->codeobj.load_size,
          evt_data->codeobj.load_delta, evt_data->codeobj.uri, evt_data->codeobj.unload);
}

}  // namespace

#include <hsa/hsa_api_trace.h>

extern "C" {
// The HSA_AMD_TOOL_PRIORITY variable must be a constant value type initialized by the loader
// itself, not by code during _init. 'extern const' seems to do that although that is not a
// guarantee.
ROCTRACER_EXPORT extern const uint32_t HSA_AMD_TOOL_PRIORITY = 1050;

// HSA-runtime tool on-load method
ROCTRACER_EXPORT bool OnLoad(HsaApiTable* table, uint64_t runtime_version,
                             uint64_t failed_tool_count, const char* const* failed_tool_names) {
  CHECK(roctracer_enable_op_callback(ACTIVITY_DOMAIN_HSA_EVT, HSA_EVT_ID_CODEOBJ,
                                     CodeObjectCallback, nullptr));
  return true;
}

ROCTRACER_EXPORT void OnUnload() {
  CHECK(roctracer_disable_domain_callback(ACTIVITY_DOMAIN_HSA_EVT));
}

}  // extern "C"
