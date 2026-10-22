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

// The cases that need the ROCr module as a shared library, which this binary
// dlopens and dlcloses as rdcd's loader does, with the HSA fake in a library
// beside it. They are here rather than in rdc_rocr_diag_tests, which builds the
// module's sources into the test binary: a module whose code is part of the
// executable cannot be unmapped, and the library this one loads is built so that
// a dlclose without a pin does unmap it. Neither library is installed.
//
// Each case runs in a forked child of its own, which loads the libraries itself,
// so that no case rests on another: a pin cannot be undone and the loader case
// plants an error.
//
// The pin case and its control load the module themselves, run a check through it
// and close it again, which is what rdcd's loader does when a program that embeds
// RDC stops it.

#define RDC_FAKE_HSA_HARNESS 1

#include <dlfcn.h>
#include <gtest/gtest.h>
#include <unistd.h>

#include <cstdio>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "fake_hsa.h"
#include "rdc/rdc.h"
#include "rdc_lib/RdcDiagnosticLibInterface.h"
#include "rdc_lib/RdcLibraryLoader.h"

namespace {

// Fails the case naming the library the build should have made beside it.
void require_library(const char* path) {
  ASSERT_EQ(::access(path, R_OK), 0) << "the build made no " << path;
}

// A check's result is half a megabyte, so a case keeps it on the heap.
using ResultPtr = std::unique_ptr<rdc_diag_test_result_t>;

// The module as the loader gives it: its entry point, its test hooks and the fake's controls,
// which one handle finds because the module links the fake.
struct Module {
  void* handle = nullptr;
  rdc_status_t (*test_case_run)(rdc_diag_test_cases_t, uint32_t*, uint32_t, const char*, size_t,
                                rdc_diag_test_result_t*, rdc_diag_callback_t*) = nullptr;
  uint64_t (*set_wait_deadline_ms)(uint64_t) = nullptr;
  void (*set_kernel_directories)(const char* const*, size_t) = nullptr;
  void (*fake_reset)(void) = nullptr;
  void (*fake_set_kernel_mode)(int) = nullptr;
};

// The symbol, which the library must carry: its own text is what the case reports, as a null
// dlerror() streamed into a log line would leave std::cout bad.
template <typename T>
T symbol_of(void* handle, const char* name) {
  void* const found = ::dlsym(handle, name);
  EXPECT_NE(found, nullptr) << "the library has no " << name;
  return reinterpret_cast<T>(found);
}

Module load_module() {
  Module module;
  module.handle = ::dlopen(RDC_ROCR_PIN_MODULE_PATH, RTLD_NOW | RTLD_LOCAL);
  EXPECT_NE(module.handle, nullptr) << "the module could not be loaded";
  if (module.handle == nullptr) {
    return module;
  }
  module.test_case_run =
      symbol_of<decltype(module.test_case_run)>(module.handle, "rdc_diag_test_case_run");
  module.set_wait_deadline_ms = symbol_of<decltype(module.set_wait_deadline_ms)>(
      module.handle, "rdc_rocr_test_set_wait_deadline_ms");
  module.set_kernel_directories = symbol_of<decltype(module.set_kernel_directories)>(
      module.handle, "rdc_rocr_test_set_kernel_directories");
  module.fake_reset = symbol_of<decltype(module.fake_reset)>(module.handle, "rdc_fake_hsa_reset");
  module.fake_set_kernel_mode = symbol_of<decltype(module.fake_set_kernel_mode)>(
      module.handle, "rdc_fake_hsa_set_kernel_mode");
  return module;
}

// The kernel files a check loads, in the case's own working directory, which the module is told
// to search: it does not search the working directory of its own accord.
void write_kernel_files(const Module& module) {
  const char* const names[] = {"binary_search_kernels.hsaco", "gpuReadWrite_kernels.hsaco"};
  for (const char* name : names) {
    std::FILE* const file = std::fopen(name, "wb");
    ASSERT_NE(file, nullptr) << name;
    ASSERT_GE(std::fputs("hsaco", file), 0) << name;
    ASSERT_EQ(std::fclose(file), 0) << name;
  }
  const char* const here[] = {"."};
  module.set_kernel_directories(here, 1);
}

// One check on those GPUs, through the module this binary loaded.
std::string run_check(const Module& module, const std::vector<uint32_t>& ids,
                      rdc_diag_test_result_t* result) {
  std::vector<uint32_t> entities = ids;
  EXPECT_EQ(
      module.test_case_run(RDC_DIAG_SYS_MEM_CHECK, entities.data(),
                           static_cast<uint32_t>(entities.size()), nullptr, 0, result, nullptr),
      RDC_ST_OK);
  std::string texts;
  for (uint32_t row = 0; row < result->per_gpu_result_count; row++) {
    texts += result->gpu_results[row].gpu_result.msg;
    texts += "\n";
  }
  return texts;
}

// Whether the module is still mapped: an RTLD_NOLOAD dlopen finds a library that is, and
// answers nothing for one that is not.
bool module_is_mapped() {
  void* const found = ::dlopen(RDC_ROCR_PIN_MODULE_PATH, RTLD_LAZY | RTLD_LOCAL | RTLD_NOLOAD);
  if (found == nullptr) {
    return false;
  }
  ::dlclose(found);
  return true;
}

}  // namespace

// rdc_rocr_fake_hsa is linked with --defsym=rdc_zero_symbol=0, so the library
// exports that name as an absolute symbol of value 0: dlsym answers with a null
// pointer and sets no error, which is the one way a symbol the loader asks for
// resolves to nothing without dlerror() having anything to say. A name the
// library does not carry sets one, as the contrast shows.
TEST(RocrLoader, AZeroSymbolResolvesToNothingWithoutAnError) {
  RDC_ROCR_CASE({
    require_library(RDC_ROCR_FAKE_HSA_PATH);
    void* handle = ::dlopen(RDC_ROCR_FAKE_HSA_PATH, RTLD_NOW | RTLD_LOCAL);
    ASSERT_NE(handle, nullptr) << ::dlerror();

    ::dlerror();
    void* zero = ::dlsym(handle, "rdc_zero_symbol");
    const char* no_error = ::dlerror();
    EXPECT_EQ(zero, nullptr);
    EXPECT_EQ(no_error, nullptr) << "dlerror: " << no_error;

    ::dlerror();
    void* missing = ::dlsym(handle, "rdc_a_symbol_no_library_carries");
    const char* error = ::dlerror();
    EXPECT_EQ(missing, nullptr);
    EXPECT_NE(error, nullptr);

    // RdcLibraryLoader reports a symbol that resolved to nothing as a failure to
    // load the module, whatever dlerror() had to say about it.
    amd::rdc::RdcLibraryLoader loader;
    ASSERT_EQ(loader.load(RDC_ROCR_FAKE_HSA_PATH, RTLD_NOW), RDC_ST_OK);
    void* symbol = nullptr;
    EXPECT_EQ(loader.load_symbol(&symbol, "rdc_zero_symbol"), RDC_ST_FAIL_LOAD_MODULE);
    EXPECT_EQ(symbol, nullptr);
    // The log line that call writes is what the case below checks; on the code
    // F22 fixes it streams dlerror()'s null pointer, which leaves std::cout bad.
    EXPECT_TRUE(std::cout.good());
    std::cout.clear();
    EXPECT_EQ(loader.unload(), RDC_ST_OK);
  });
}

// The line that failure writes: F22's loader clears dlerror() before it asks for
// the symbol and reads it after, and writes its own text alone when there is
// nothing to add, so that a stale error from an earlier call cannot be reported
// as this one's and a null one cannot silence every later line of the process
// (f22/exp/zerosym_lsttdev6.txt).
TEST(RocrLoader, AZeroSymbolIsReportedWithoutADlerrorText) {
  RDC_ROCR_CASE({
    require_library(RDC_ROCR_FAKE_HSA_PATH);
    amd::rdc::RdcLibraryLoader loader;
    ASSERT_EQ(loader.load(RDC_ROCR_FAKE_HSA_PATH, RTLD_NOW), RDC_ST_OK);

    // An error another call left pending and nobody read, as a loader's earlier work leaves one.
    void* handle = ::dlopen(RDC_ROCR_FAKE_HSA_PATH, RTLD_NOW | RTLD_LOCAL | RTLD_NOLOAD);
    ASSERT_NE(handle, nullptr);
    (void)::dlsym(handle, "rdc_a_symbol_no_library_carries");

    ::testing::internal::CaptureStdout();
    void* symbol = reinterpret_cast<void*>(0x1);
    const rdc_status_t status = loader.load_symbol(&symbol, "rdc_zero_symbol");
    std::string line = ::testing::internal::GetCapturedStdout();
    while (!line.empty() && (line.back() == '\n' || line.back() == '\r')) {
      line.pop_back();
    }

    EXPECT_EQ(status, RDC_ST_FAIL_LOAD_MODULE);
    EXPECT_EQ(symbol, nullptr);
    EXPECT_TRUE(std::cout.good());

    // One line, which is the loader's own text and ends with the symbol's name: the stale error
    // is not reported as this call's, and no null text reaches the log.
    EXPECT_EQ(line.find('\n'), std::string::npos) << line;
    EXPECT_NE(line.find("RdcLibraryLoader: Fail to load the symbol rdc_zero_symbol"),
              std::string::npos)
        << line;
    ASSERT_GE(line.size(), std::string("rdc_zero_symbol").size());
    EXPECT_EQ(line.substr(line.size() - std::string("rdc_zero_symbol").size()), "rdc_zero_symbol")
        << line;
    EXPECT_EQ(line.find("rdc_a_symbol_no_library_carries"), std::string::npos) << line;

    // A symbol the library does not carry has an error of its own, which the line does carry.
    ::testing::internal::CaptureStdout();
    void* missing = nullptr;
    EXPECT_EQ(loader.load_symbol(&missing, "rdc_a_symbol_no_library_carries"),
              RDC_ST_FAIL_LOAD_MODULE);
    const std::string reported = ::testing::internal::GetCapturedStdout();
    EXPECT_NE(reported.find("undefined symbol"), std::string::npos) << reported;
    EXPECT_TRUE(std::cout.good());

    // A line written after them arrives whole.
    ::testing::internal::CaptureStdout();
    std::cout << "the log still works" << std::endl;
    EXPECT_EQ(::testing::internal::GetCapturedStdout(), "the log still works\n");

    EXPECT_EQ(loader.unload(), RDC_ST_OK);
  });
}

// The module pins itself when it abandons a check's work, so that its record and
// its queue error callback stay mapped after the loader's dlclose: the check it
// could not stop is a failure in every later run and the other GPUs are skipped,
// which is what would be lost with the module (M: the before run has no pin
// target, and T4's mutation drops the pin).
TEST(RocrPin, AnAbandonedCheckKeepsTheModuleMapped) {
  RDC_ROCR_CASE({
    require_library(RDC_ROCR_FAKE_HSA_PATH);
    require_library(RDC_ROCR_PIN_MODULE_PATH);

    Module module = load_module();
    ASSERT_NE(module.handle, nullptr);
    ASSERT_NE(module.test_case_run, nullptr);
    ASSERT_NE(module.set_wait_deadline_ms, nullptr);
    ASSERT_NE(module.fake_set_kernel_mode, nullptr);

    module.fake_reset();
    write_kernel_files(module);
    // A kernel that never completes, with a deadline of a few hints rather than ten seconds.
    EXPECT_EQ(module.set_wait_deadline_ms(300), 10u * 1000u);
    module.fake_set_kernel_mode(RDC_FAKE_KERNEL_NEVER);

    ResultPtr abandoned(new rdc_diag_test_result_t());
    const std::string left = run_check(module, {1}, abandoned.get());
    EXPECT_EQ(abandoned->status, RDC_DIAG_RESULT_FAIL) << left;
    EXPECT_NE(left.find("The kernel did not complete in 300 ms."), std::string::npos) << left;

    ASSERT_EQ(::dlclose(module.handle), 0);
    ASSERT_TRUE(module_is_mapped()) << "the dlclose unmapped the module, and its record with it";

    // The module the loader finds again is the one that holds the record.
    Module kept = load_module();
    ASSERT_NE(kept.handle, nullptr);
    ASSERT_NE(kept.test_case_run, nullptr);
    ResultPtr later(new rdc_diag_test_result_t());
    const std::string texts = run_check(kept, {0, 1}, later.get());
    ASSERT_EQ(later->per_gpu_result_count, 2u) << texts;
    EXPECT_NE(std::string(later->gpu_results[1].gpu_result.msg)
                  .find("An earlier check could not stop its work on this GPU"),
              std::string::npos)
        << texts;
    EXPECT_NE(
        std::string(later->gpu_results[0].gpu_result.msg).find("The ROCr checks are disabled"),
        std::string::npos)
        << texts;
    EXPECT_NE(std::string(later->gpu_results[0].gpu_result.msg).find("on GPU 1;"),
              std::string::npos)
        << texts;
  });
}

// The control of the case above: a module that abandoned nothing pinned nothing, and the
// loader's dlclose unmaps it, as it would unmap it under a live queue without the pin. It is
// neither an F case nor an M case: it holds on F9's sources and with the pin dropped, and what
// it rules out is a module that pins itself where nothing was abandoned, which would leave the
// case above proving nothing.
TEST(RocrPin, AModuleThatAbandonedNothingIsUnmappedByTheDlclose) {
  RDC_ROCR_CASE({
    require_library(RDC_ROCR_FAKE_HSA_PATH);
    require_library(RDC_ROCR_PIN_MODULE_PATH);

    Module module = load_module();
    ASSERT_NE(module.handle, nullptr);
    ASSERT_NE(module.test_case_run, nullptr);

    module.fake_reset();
    write_kernel_files(module);
    ResultPtr result(new rdc_diag_test_result_t());
    const std::string texts = run_check(module, {1}, result.get());
    EXPECT_EQ(result->status, RDC_DIAG_RESULT_PASS) << texts;

    // The leak check of this case is made here, before the dlclose: unmapping a library makes
    // everything it still holds unreachable, which LeakSanitizer would report as leaked.
    EXPECT_FALSE(rdc_fake_hsa::leaks_reported()) << "the case leaked before the dlclose";

    ASSERT_EQ(::dlclose(module.handle), 0);
    EXPECT_FALSE(module_is_mapped())
        << "the module stayed mapped with nothing abandoned, so the pin case proves nothing here";
  });
}
