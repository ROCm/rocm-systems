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

// RdcModuleMgrImpl makes its telemetry and diagnostic modules on first use. Two
// threads that ask for one at the same time must get the same module, with no data
// race: rdcd serves its clients from several threads, so two clients' first
// diagnostics can come at once. RdcModuleMgrImpl.cc and the two modules it makes
// are compiled in; the AMD SMI, ROCr and ROCp modules below stand in for the real
// ones and offer nothing.

#include <gtest/gtest.h>

#include <atomic>
#include <cstring>
#include <thread>

#include "rdc/rdc.h"
#include "rdc_lib/impl/RdcModuleMgrImpl.h"
#include "rdc_lib/rdc_common.h"

// The stand-ins below leave the modules' private fields unused.
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-private-field"
#endif
#include "rdc_lib/impl/RdcRocpLib.h"
#include "rdc_lib/impl/RdcRocrLib.h"
#include "rdc_lib/impl/RdcSmiLib.h"
#if defined(__clang__)
#pragma clang diagnostic pop
#endif

// The two bootstrap functions the modules call (RdcBootStrap.cc is not compiled in);
// weak, because another race test can define them too.
__attribute__((weak)) char* strncpy_with_null(char* dest, const char* src, size_t n) {
  if (n == 0) {
    return dest;
  }
  strncpy(dest, src, n - 1);
  dest[n - 1] = '\0';
  return dest;
}
__attribute__((weak)) const char* field_id_string(rdc_field_t) { return "field"; }

namespace amd {
namespace rdc {

RdcSmiLib::RdcSmiLib(const RdcMetricFetcherPtr& mf)
    : metric_fetcher_(mf), bulk_fetch_enabled_(false) {}
rdc_status_t RdcSmiLib::rdc_telemetry_fields_query(uint32_t[MAX_NUM_FIELDS],
                                                   uint32_t* field_count) {
  *field_count = 0;
  return RDC_ST_OK;
}
rdc_status_t RdcSmiLib::rdc_telemetry_fields_value_get(rdc_gpu_field_t*, uint32_t,
                                                       rdc_field_value_f, void*) {
  return RDC_ST_NOT_SUPPORTED;
}
rdc_status_t RdcSmiLib::rdc_telemetry_fields_watch(rdc_gpu_field_t*, uint32_t) {
  return RDC_ST_NOT_SUPPORTED;
}
rdc_status_t RdcSmiLib::rdc_telemetry_fields_unwatch(rdc_gpu_field_t*, uint32_t) {
  return RDC_ST_NOT_SUPPORTED;
}
rdc_status_t RdcSmiLib::rdc_diag_test_cases_query(rdc_diag_test_cases_t[MAX_TEST_CASES],
                                                  uint32_t* test_case_count) {
  *test_case_count = 0;
  return RDC_ST_OK;
}
rdc_status_t RdcSmiLib::rdc_test_case_run(rdc_diag_test_cases_t, uint32_t[RDC_MAX_NUM_DEVICES],
                                          uint32_t, const char*, size_t, rdc_diag_test_result_t*,
                                          rdc_diag_callback_t*) {
  return RDC_ST_NOT_SUPPORTED;
}
rdc_status_t RdcSmiLib::rdc_diagnostic_run(const rdc_group_info_t&, rdc_diag_level_t, const char*,
                                           size_t, rdc_diag_response_t*, rdc_diag_callback_t*) {
  return RDC_ST_NOT_SUPPORTED;
}
rdc_status_t RdcSmiLib::rdc_diag_init(uint64_t) { return RDC_ST_OK; }
rdc_status_t RdcSmiLib::rdc_diag_destroy() { return RDC_ST_OK; }

RdcRocrLib::RdcRocrLib()
    : test_case_run_(nullptr),
      diag_test_cases_query_(nullptr),
      diag_init_(nullptr),
      diag_destroy_(nullptr) {}
RdcRocrLib::~RdcRocrLib() = default;
rdc_status_t RdcRocrLib::rdc_diag_test_cases_query(rdc_diag_test_cases_t[MAX_TEST_CASES],
                                                   uint32_t* test_case_count) {
  *test_case_count = 0;
  return RDC_ST_OK;
}
rdc_status_t RdcRocrLib::rdc_test_case_run(rdc_diag_test_cases_t, uint32_t[RDC_MAX_NUM_DEVICES],
                                           uint32_t, const char*, size_t, rdc_diag_test_result_t*,
                                           rdc_diag_callback_t*) {
  return RDC_ST_NOT_SUPPORTED;
}
rdc_status_t RdcRocrLib::rdc_diagnostic_run(const rdc_group_info_t&, rdc_diag_level_t, const char*,
                                            size_t, rdc_diag_response_t*, rdc_diag_callback_t*) {
  return RDC_ST_NOT_SUPPORTED;
}
rdc_status_t RdcRocrLib::rdc_diag_init(uint64_t) { return RDC_ST_OK; }
rdc_status_t RdcRocrLib::rdc_diag_destroy() { return RDC_ST_OK; }

RdcRocpLib::RdcRocpLib()
    : telemetry_fields_query_(nullptr),
      telemetry_fields_value_get_(nullptr),
      telemetry_fields_watch_(nullptr),
      telemetry_fields_unwatch_(nullptr),
      rdc_module_init_(nullptr),
      rdc_module_destroy_(nullptr) {}
RdcRocpLib::~RdcRocpLib() = default;
rdc_status_t RdcRocpLib::rdc_telemetry_fields_query(uint32_t[MAX_NUM_FIELDS],
                                                    uint32_t* field_count) {
  *field_count = 0;
  return RDC_ST_OK;
}
rdc_status_t RdcRocpLib::rdc_telemetry_fields_value_get(rdc_gpu_field_t*, uint32_t,
                                                        rdc_field_value_f, void*) {
  return RDC_ST_NOT_SUPPORTED;
}
rdc_status_t RdcRocpLib::rdc_telemetry_fields_watch(rdc_gpu_field_t*, uint32_t) {
  return RDC_ST_NOT_SUPPORTED;
}
rdc_status_t RdcRocpLib::rdc_telemetry_fields_unwatch(rdc_gpu_field_t*, uint32_t) {
  return RDC_ST_NOT_SUPPORTED;
}

}  // namespace rdc
}  // namespace amd

namespace {

using amd::rdc::RdcDiagnosticPtr;
using amd::rdc::RdcModuleMgrImpl;
using amd::rdc::RdcTelemetryPtr;

TEST(ModuleMgrRace, FirstModulesFromTwoThreads) {
  for (int round = 0; round < 20; round++) {
    RdcModuleMgrImpl mgr(nullptr);
    std::atomic<int> ready{0};
    RdcDiagnosticPtr diagnostic[2];
    RdcTelemetryPtr telemetry[2];
    auto ask = [&](int i) {
      ready++;
      while (ready.load() < 2) {
      }
      diagnostic[i] = mgr.get_diagnostic_module();
      telemetry[i] = mgr.get_telemetry_module();
    };
    std::thread first(ask, 0);
    std::thread second(ask, 1);
    first.join();
    second.join();
    ASSERT_NE(diagnostic[0], nullptr);
    ASSERT_NE(telemetry[0], nullptr);
    EXPECT_EQ(diagnostic[0], diagnostic[1]) << "round " << round << ": two diagnostic modules";
    EXPECT_EQ(telemetry[0], telemetry[1]) << "round " << round << ": two telemetry modules";
  }
}

}  // namespace
