/******************************************************************************
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
*******************************************************************************/

#include <string.h>
#include <stdio.h>
#include <stdlib.h>

#include "inc/roctracer.h"
#include "inc/roctracer_hsa.h"
#include <rocprofiler/rocprofiler.h>

#define PUBLIC_API __attribute__((visibility("default")))
#define CONSTRUCTOR_API __attribute__((constructor))
#define DESTRUCTOR_API __attribute__((destructor))

// Check returned HSA API status
void check_status(roctracer_status_t status) {
  if (status != ROCTRACER_STATUS_SUCCESS) {
    const char* error_string = roctracer_error_string();
    fprintf(stderr, "ERROR: %s\n", error_string);
    abort();
  }
}

// codeobj callback
void codeobj_callback(uint32_t domain, uint32_t cid, const void* data, void* arg) {
  const hsa_evt_data_t* evt_data = reinterpret_cast<const hsa_evt_data_t*>(data);
  const char* uri = evt_data->codeobj.uri;
  printf("codeobj_callback domain(%u) cid(%u): load_base(0x%lx) load_size(0x%lx) load_delta(0x%lx) uri(\"%s\")\n",
    domain,
    cid,
    evt_data->codeobj.load_base,
    evt_data->codeobj.load_size,
    evt_data->codeobj.load_delta,
    uri);
  free((void*)uri);
  fflush(stdout);
}

void initialize() {
  roctracer_status_t status = roctracer_enable_op_callback(ACTIVITY_DOMAIN_HSA_EVT, HSA_EVT_ID_CODEOBJ, codeobj_callback, NULL);
  check_status(status);
}

void cleanup() {
  roctracer_status_t status = roctracer_disable_domain_callback(ACTIVITY_DOMAIN_HSA_EVT);
  check_status(status);
}

// Tool constructor
extern "C" PUBLIC_API void OnLoadToolProp(rocprofiler_settings_t* settings) {
  // Enable HSA events intercepting
  settings->hsa_intercepting = 1;
  // Initialize profiling
  initialize();
}

// Tool destructor
extern "C" PUBLIC_API void OnUnloadTool() {
  // Final resources cleanup
  cleanup();
}

extern "C" CONSTRUCTOR_API void constructor() {
  printf("constructor\n"); fflush(stdout);
}

extern "C" DESTRUCTOR_API void destructor() {
  OnUnloadTool();
}
