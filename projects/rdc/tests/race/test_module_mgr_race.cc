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

// RdcModuleMgrImpl builds its dispatcher and its telemetry module when the
// matching getter is first called. rdcd calls both getters from its request
// threads, so two first requests at once read and write one member. Each case
// here calls one getter from two threads and expects them to get the one
// module the manager built.
//
// The file defines link-time fakes of every class the manager builds and of
// the logger it writes through, because the binary links no RDC library. So no
// amd-smi call, no dlopen and no GPU is reached. test_race_sanity.cc has the
// binary's main().

#include <gtest/gtest.h>

#include <atomic>
#include <thread>  // NOLINT(build/c++11)

#include "rdc_lib/RdcLibraryLoader.h"
#include "rdc_lib/RdcLogger.h"
#include "rdc_lib/impl/RdcDiagnosticModule.h"
#include "rdc_lib/impl/RdcModuleMgrImpl.h"
#include "rdc_lib/impl/RdcRocpLib.h"
#include "rdc_lib/impl/RdcRocrLib.h"
#include "rdc_lib/impl/RdcSmiLib.h"
#include "rdc_lib/impl/RdcTelemetryModule.h"

namespace {

// The modules the fake constructors below have built. Each case reads the one
// it is about, so that a module built twice is seen.
std::atomic<int> g_diagnostic_modules_built{0};
std::atomic<int> g_telemetry_modules_built{0};

}  // namespace

namespace amd {
namespace rdc {

// The fakes. Every class RdcModuleMgrImpl.cc constructs needs its constructor,
// its destructor and each of its virtual methods here, and so does every class
// one of them holds by value. Only the two module constructors do anything.

RdcSmiLib::RdcSmiLib(const RdcMetricFetcherPtr& mf)
    : metric_fetcher_(mf), bulk_fetch_enabled_(false) {}

rdc_status_t RdcSmiLib::rdc_telemetry_fields_query(uint32_t[MAX_NUM_FIELDS],
                                                   uint32_t* field_count) {
  *field_count = 0;
  return RDC_ST_OK;
}

rdc_status_t RdcSmiLib::rdc_telemetry_fields_value_get(rdc_gpu_field_t*, uint32_t,
                                                       rdc_field_value_f, void*) {
  return RDC_ST_OK;
}

rdc_status_t RdcSmiLib::rdc_telemetry_fields_watch(rdc_gpu_field_t*, uint32_t) { return RDC_ST_OK; }

rdc_status_t RdcSmiLib::rdc_telemetry_fields_unwatch(rdc_gpu_field_t*, uint32_t) {
  return RDC_ST_OK;
}

rdc_status_t RdcSmiLib::rdc_diag_test_cases_query(rdc_diag_test_cases_t[MAX_TEST_CASES],
                                                  uint32_t* test_case_count) {
  *test_case_count = 0;
  return RDC_ST_OK;
}

rdc_status_t RdcSmiLib::rdc_test_case_run(rdc_diag_test_cases_t, uint32_t[RDC_MAX_NUM_DEVICES],
                                          uint32_t, const char*, size_t, rdc_diag_test_result_t*,
                                          rdc_diag_callback_t*) {
  return RDC_ST_OK;
}

rdc_status_t RdcSmiLib::rdc_diagnostic_run(const rdc_group_info_t&, rdc_diag_level_t, const char*,
                                           size_t, rdc_diag_response_t*, rdc_diag_callback_t*) {
  return RDC_ST_OK;
}

rdc_status_t RdcSmiLib::rdc_diag_init(uint64_t) { return RDC_ST_OK; }

rdc_status_t RdcSmiLib::rdc_diag_destroy() { return RDC_ST_OK; }

RdcRocrLib::RdcRocrLib()
    : test_case_run_(nullptr),
      diag_test_cases_query_(nullptr),
      diag_init_(nullptr),
      diag_destroy_(nullptr) {}

RdcRocrLib::~RdcRocrLib() {}

rdc_status_t RdcRocrLib::rdc_diag_test_cases_query(rdc_diag_test_cases_t[MAX_TEST_CASES],
                                                   uint32_t* test_case_count) {
  *test_case_count = 0;
  return RDC_ST_OK;
}

rdc_status_t RdcRocrLib::rdc_test_case_run(rdc_diag_test_cases_t, uint32_t[RDC_MAX_NUM_DEVICES],
                                           uint32_t, const char*, size_t, rdc_diag_test_result_t*,
                                           rdc_diag_callback_t*) {
  return RDC_ST_OK;
}

rdc_status_t RdcRocrLib::rdc_diagnostic_run(const rdc_group_info_t&, rdc_diag_level_t, const char*,
                                            size_t, rdc_diag_response_t*, rdc_diag_callback_t*) {
  return RDC_ST_OK;
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

RdcRocpLib::~RdcRocpLib() {}

rdc_status_t RdcRocpLib::rdc_telemetry_fields_query(uint32_t[MAX_NUM_FIELDS],
                                                    uint32_t* field_count) {
  *field_count = 0;
  return RDC_ST_OK;
}

rdc_status_t RdcRocpLib::rdc_telemetry_fields_value_get(rdc_gpu_field_t*, uint32_t,
                                                        rdc_field_value_f, void*) {
  return RDC_ST_OK;
}

rdc_status_t RdcRocpLib::rdc_telemetry_fields_watch(rdc_gpu_field_t*, uint32_t) {
  return RDC_ST_OK;
}

rdc_status_t RdcRocpLib::rdc_telemetry_fields_unwatch(rdc_gpu_field_t*, uint32_t) {
  return RDC_ST_OK;
}

RdcDiagnosticModule::RdcDiagnosticModule(std::list<RdcDiagnosticPtr> diagnostic_modules)
    : diagnostic_modules_(diagnostic_modules) {
  g_diagnostic_modules_built.fetch_add(1);
}

rdc_status_t RdcDiagnosticModule::rdc_diag_test_cases_query(rdc_diag_test_cases_t[MAX_TEST_CASES],
                                                            uint32_t* test_case_count) {
  *test_case_count = 0;
  return RDC_ST_OK;
}

rdc_status_t RdcDiagnosticModule::rdc_test_case_run(rdc_diag_test_cases_t,
                                                    uint32_t[RDC_MAX_NUM_DEVICES], uint32_t,
                                                    const char*, size_t, rdc_diag_test_result_t*,
                                                    rdc_diag_callback_t*) {
  return RDC_ST_OK;
}

rdc_status_t RdcDiagnosticModule::rdc_diagnostic_run(const rdc_group_info_t&, rdc_diag_level_t,
                                                     const char*, size_t, rdc_diag_response_t*,
                                                     rdc_diag_callback_t*) {
  return RDC_ST_OK;
}

rdc_status_t RdcDiagnosticModule::rdc_diag_init(uint64_t) { return RDC_ST_OK; }

rdc_status_t RdcDiagnosticModule::rdc_diag_destroy() { return RDC_ST_OK; }

RdcTelemetryModule::RdcTelemetryModule(std::list<RdcTelemetryPtr> telemetry_modules)
    : telemetry_modules_(telemetry_modules) {
  g_telemetry_modules_built.fetch_add(1);
}

rdc_status_t RdcTelemetryModule::rdc_telemetry_fields_query(uint32_t[MAX_NUM_FIELDS],
                                                            uint32_t* field_count) {
  *field_count = 0;
  return RDC_ST_OK;
}

rdc_status_t RdcTelemetryModule::rdc_telemetry_fields_value_get(rdc_gpu_field_t*, uint32_t,
                                                                rdc_field_value_f, void*) {
  return RDC_ST_OK;
}

rdc_status_t RdcTelemetryModule::rdc_telemetry_fields_watch(rdc_gpu_field_t*, uint32_t) {
  return RDC_ST_OK;
}

rdc_status_t RdcTelemetryModule::rdc_telemetry_fields_unwatch(rdc_gpu_field_t*, uint32_t) {
  return RDC_ST_OK;
}

// RdcRocrLib and RdcRocpLib hold a loader by value, which no fake here uses.
RdcLibraryLoader::RdcLibraryLoader() : libHandler_(nullptr) {}

RdcLibraryLoader::~RdcLibraryLoader() {}

// The manager's RDC_LOG lines go through this logger. The level leaves the
// debug line out and keeps an error line, as rdc_bootstrap's logger does
// without RDC_DEBUG_LEVEL set.
RdcLogger::RdcLogger(std::ostream& os) : os_(os), log_level_(RDC_ERROR) {}

std::string RdcLogger::get_log_header(uint32_t, const char*, int) { return "RDC "; }

}  // namespace rdc
}  // namespace amd

namespace {

// Calls `get` on a fresh manager from two threads that start together, and
// writes what each got. Both threads wait on one flag, so neither orders the
// other: ThreadSanitizer reports the race whatever their timing.
template <typename Ptr, typename Getter>
void get_from_two_threads(Getter get, Ptr* first, Ptr* second) {
  amd::rdc::RdcModuleMgrImpl module_mgr{amd::rdc::RdcMetricFetcherPtr{}};
  std::atomic<bool> start{false};
  std::thread one([&]() {
    while (!start.load()) {
    }
    *first = get(&module_mgr);
  });
  std::thread two([&]() {
    while (!start.load()) {
    }
    *second = get(&module_mgr);
  });
  start.store(true);
  one.join();
  two.join();
}

}  // namespace

// Admission case (F4): before the fix the two threads read and write
// rdc_diagnostic_module_ with no lock, so ThreadSanitizer reports a data race
// on the member and the binary exits with 66. They may also get two modules.
TEST(ModuleMgrRace, TwoFirstRequestsGetOneDiagnosticModule) {
  g_diagnostic_modules_built.store(0);
  amd::rdc::RdcDiagnosticPtr first;
  amd::rdc::RdcDiagnosticPtr second;
  get_from_two_threads([](amd::rdc::RdcModuleMgrImpl* mgr) { return mgr->get_diagnostic_module(); },
                       &first, &second);
  EXPECT_NE(first, nullptr);
  EXPECT_EQ(first, second);
  EXPECT_EQ(g_diagnostic_modules_built.load(), 1);
}

// Admission case (F4): the same for rdc_telemetry_module_, which rdcd's watch
// table reaches from its own threads.
TEST(ModuleMgrRace, TwoFirstRequestsGetOneTelemetryModule) {
  g_telemetry_modules_built.store(0);
  amd::rdc::RdcTelemetryPtr first;
  amd::rdc::RdcTelemetryPtr second;
  get_from_two_threads([](amd::rdc::RdcModuleMgrImpl* mgr) { return mgr->get_telemetry_module(); },
                       &first, &second);
  EXPECT_NE(first, nullptr);
  EXPECT_EQ(first, second);
  EXPECT_EQ(g_telemetry_modules_built.load(), 1);
}
