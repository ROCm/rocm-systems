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

// Test-only stand-in for replies rdci cannot get from a well-behaved rdcd, for
// tests/ci/test_fake_gpu_rdci.py. Preloaded into rdci (never into rdcd), it wraps these RDC
// client calls and passes everything else through:
//
//   RDC_FAKE_CLIENT_FAIL=<call>[=<status>]
//       rdc_field_watch or rdc_group_field_destroy returns <status> (a number; default
//       RDC_ST_UNKNOWN_ERROR) without asking rdcd.
//   RDC_FAKE_CLIENT_NAME=<text>
//       rdc_group_gpu_get_info's and rdc_group_field_get_info's replies carry <text> as the
//       group's name, and rdc_job_get_stats's reply as its first process's name (a reply with no
//       process gets one).
//
// It links no RDC library: it finds the real calls with dlsym(RTLD_NEXT).

#include <dlfcn.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "rdc/rdc.h"

namespace {

template <typename Fn>
Fn real(const char* name) {
  void* sym = dlsym(RTLD_NEXT, name);
  if (sym == nullptr) {
    fprintf(stderr, "fake RDC client: no %s after this library\n", name);
    abort();
  }
  return reinterpret_cast<Fn>(sym);
}

// The status RDC_FAKE_CLIENT_FAIL gives `call`; RDC_ST_OK when it names no call or another.
rdc_status_t injected_failure(const char* call) {
  const char* rule = getenv("RDC_FAKE_CLIENT_FAIL");
  if (rule == nullptr) {
    return RDC_ST_OK;
  }
  const size_t length = strlen(call);
  if (strncmp(rule, call, length) != 0 || (rule[length] != '\0' && rule[length] != '=')) {
    return RDC_ST_OK;
  }
  if (rule[length] == '\0') {
    return RDC_ST_UNKNOWN_ERROR;
  }
  return static_cast<rdc_status_t>(strtol(rule + length + 1, nullptr, 0));
}

// Copies RDC_FAKE_CLIENT_NAME into a name buffer of `size` bytes; false when it is not set.
bool inject_name(char* name, size_t size) {
  const char* text = getenv("RDC_FAKE_CLIENT_NAME");
  if (text == nullptr) {
    return false;
  }
  const size_t length = strnlen(text, size - 1);
  memcpy(name, text, length);
  name[length] = '\0';
  return true;
}

}  // namespace

extern "C" {

rdc_status_t rdc_field_watch(rdc_handle_t p_rdc_handle, rdc_gpu_group_t group_id,
                             rdc_field_grp_t field_group_id, uint64_t update_freq,
                             double max_keep_age, uint32_t max_keep_samples) {
  if (rdc_status_t status = injected_failure("rdc_field_watch")) {
    return status;
  }
  return real<decltype(&rdc_field_watch)>("rdc_field_watch")(
      p_rdc_handle, group_id, field_group_id, update_freq, max_keep_age, max_keep_samples);
}

rdc_status_t rdc_group_field_destroy(rdc_handle_t p_rdc_handle,
                                     rdc_field_grp_t rdc_field_group_id) {
  if (rdc_status_t status = injected_failure("rdc_group_field_destroy")) {
    return status;
  }
  return real<decltype(&rdc_group_field_destroy)>("rdc_group_field_destroy")(p_rdc_handle,
                                                                             rdc_field_group_id);
}

rdc_status_t rdc_group_gpu_get_info(rdc_handle_t p_rdc_handle, rdc_gpu_group_t p_rdc_group_id,
                                    rdc_group_info_t* p_rdc_group_info) {
  rdc_status_t status = real<decltype(&rdc_group_gpu_get_info)>("rdc_group_gpu_get_info")(
      p_rdc_handle, p_rdc_group_id, p_rdc_group_info);
  if (status == RDC_ST_OK) {
    inject_name(p_rdc_group_info->group_name, sizeof(p_rdc_group_info->group_name));
  }
  return status;
}

rdc_status_t rdc_group_field_get_info(rdc_handle_t p_rdc_handle, rdc_field_grp_t rdc_field_group_id,
                                      rdc_field_group_info_t* field_group_info) {
  rdc_status_t status = real<decltype(&rdc_group_field_get_info)>("rdc_group_field_get_info")(
      p_rdc_handle, rdc_field_group_id, field_group_info);
  if (status == RDC_ST_OK) {
    inject_name(field_group_info->group_name, sizeof(field_group_info->group_name));
  }
  return status;
}

rdc_status_t rdc_job_get_stats(rdc_handle_t p_rdc_handle, const char job_id[64],
                               rdc_job_info_t* p_job_info) {
  rdc_status_t status =
      real<decltype(&rdc_job_get_stats)>("rdc_job_get_stats")(p_rdc_handle, job_id, p_job_info);
  if (status != RDC_ST_OK) {
    return status;
  }
  rdc_process_status_info_t& process = p_job_info->processes[0];
  if (inject_name(process.process_name, sizeof(process.process_name)) &&
      p_job_info->num_processes == 0) {
    p_job_info->num_processes = 1;
    process.pid = 4242;
    process.start_time = 1000000;
    process.stop_time = 2000000;
  }
  return status;
}

}  // extern "C"
