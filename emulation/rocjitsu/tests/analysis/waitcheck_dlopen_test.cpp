// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "../tools/waitcheck_fixture.h"
#include "rocjitsu/analysis/rj_waitcheck.h"

#include <gtest/gtest.h>

#include <dlfcn.h>
#include <link.h>

#include <array>
#include <atomic>
#include <filesystem>
#include <string_view>
#include <thread>

namespace {

using OptionsInit = rj_status_t (*)(rj_waitcheck_options_t *, size_t);
using ResultInit = rj_status_t (*)(rj_waitcheck_result_t *, size_t);
using Analyze = rj_status_t (*)(const void *, size_t, const rj_waitcheck_options_t *,
                                rj_waitcheck_result_t *);
using DiagnosticCodeName = const char *(*)(rj_waitcheck_diagnostic_code_t);

template <typename Function> Function load_symbol(void *handle, const char *name) {
  dlerror();
  void *symbol = dlsym(handle, name);
  EXPECT_EQ(dlerror(), nullptr) << name;
  EXPECT_NE(symbol, nullptr) << name;
  return reinterpret_cast<Function>(symbol);
}

bool is_object_loaded(std::string_view needle) {
  struct State {
    std::string_view needle;
    bool found = false;
  } state{needle};
  dl_iterate_phdr(
      [](dl_phdr_info *info, size_t, void *data) {
        auto &state = *static_cast<State *>(data);
        if (std::string_view(info->dlpi_name).find(state.needle) != std::string_view::npos) {
          state.found = true;
          return 1;
        }
        return 0;
      },
      &state);
  return state.found;
}

size_t thread_count() {
  size_t count = 0;
  for ([[maybe_unused]] const auto &entry : std::filesystem::directory_iterator("/proc/self/task"))
    ++count;
  return count;
}

struct CallbackState {
  size_t count = 0;
  std::thread::id calling_thread;
  bool valid = true;
};

void capture_diagnostic(const rj_waitcheck_diagnostic_t *diagnostic, void *user_data) {
  auto &state = *static_cast<CallbackState *>(user_data);
  state.valid &= std::this_thread::get_id() == state.calling_thread;
  state.valid &= diagnostic->code == ROCJITSU_WAITCHECK_DIAGNOSTIC_WAIT_COUNTER;
  ++state.count;
}

TEST(WaitcheckDlopenTest, AnalyzeManyBuffersAcrossRepeatedLoadUnloadCycles) {
  constexpr size_t kCycles = 12;
  constexpr size_t kWorkerCount = 4;
  constexpr size_t kAnalysesPerWorker = 4;
  const auto gfx1200 = rocjitsu::waitcheck_test::make_gfx1200_missing_wait_code_object();
  const auto gfx1150 = rocjitsu::waitcheck_test::make_gfx1150_missing_wait_code_object();
  const size_t initial_thread_count = thread_count();

  ASSERT_FALSE(is_object_loaded("librocjitsu.so"));
  ASSERT_FALSE(is_object_loaded("libhsa-runtime64.so"));

  for (size_t cycle = 0; cycle < kCycles; ++cycle) {
    void *handle = dlopen(ROCJITSU_WAITCHECK_LIBRARY_PATH, RTLD_NOW | RTLD_LOCAL);
    ASSERT_NE(handle, nullptr) << dlerror();
    EXPECT_TRUE(is_object_loaded("librocjitsu_waitcheck.so"));
    EXPECT_FALSE(is_object_loaded("librocjitsu.so"));
    EXPECT_FALSE(is_object_loaded("libhsa-runtime64.so"));
    EXPECT_EQ(dlsym(handle, "hsa_init"), nullptr);
    EXPECT_EQ(dlsym(handle, "rj_vm_create"), nullptr);

    const OptionsInit options_init = load_symbol<OptionsInit>(handle, "rj_waitcheck_options_init");
    const ResultInit result_init = load_symbol<ResultInit>(handle, "rj_waitcheck_result_init");
    const Analyze analyze = load_symbol<Analyze>(handle, "rj_waitcheck_analyze");
    const DiagnosticCodeName diagnostic_code_name =
        load_symbol<DiagnosticCodeName>(handle, "rj_waitcheck_diagnostic_code_name");
    ASSERT_STREQ(diagnostic_code_name(ROCJITSU_WAITCHECK_DIAGNOSTIC_WAIT_COUNTER), "wait-counter");

    // Exercise DSO-owned TLS on the persistent test thread before concurrently
    // checking additional buffers on short-lived worker threads.
    CallbackState main_callback_state{.calling_thread = std::this_thread::get_id()};
    rj_waitcheck_options_t main_options;
    ASSERT_EQ(options_init(&main_options, sizeof(main_options)), ROCJITSU_STATUS_SUCCESS);
    main_options.diagnostic_callback = capture_diagnostic;
    main_options.user_data = &main_callback_state;
    rj_waitcheck_result_t main_result;
    ASSERT_EQ(result_init(&main_result, sizeof(main_result)), ROCJITSU_STATUS_SUCCESS);
    ASSERT_EQ(analyze(gfx1200.data(), gfx1200.size(), &main_options, &main_result),
              ROCJITSU_STATUS_SUCCESS);
    EXPECT_EQ(main_callback_state.count, 1u);
    EXPECT_TRUE(main_callback_state.valid);

    std::atomic<size_t> failures = 0;
    std::array<std::thread, kWorkerCount> workers;
    for (size_t worker_index = 0; worker_index < workers.size(); ++worker_index) {
      workers[worker_index] = std::thread([&, worker_index] {
        const auto &image = worker_index % 2 == 0 ? gfx1200 : gfx1150;
        const rj_waitcheck_target_t target = worker_index % 2 == 0
                                                 ? ROCJITSU_WAITCHECK_TARGET_GFX1200
                                                 : ROCJITSU_WAITCHECK_TARGET_GFX1150;
        for (size_t analysis_index = 0; analysis_index < kAnalysesPerWorker; ++analysis_index) {
          CallbackState callback_state{.calling_thread = std::this_thread::get_id()};
          rj_waitcheck_options_t options;
          rj_waitcheck_result_t result;
          if (options_init(&options, sizeof(options)) != ROCJITSU_STATUS_SUCCESS ||
              result_init(&result, sizeof(result)) != ROCJITSU_STATUS_SUCCESS) {
            ++failures;
            continue;
          }
          options.diagnostic_callback = capture_diagnostic;
          options.user_data = &callback_state;
          if (analyze(image.data(), image.size(), &options, &result) != ROCJITSU_STATUS_SUCCESS ||
              result.target != target || result.passed != 0 || result.diagnostics_observed != 1 ||
              callback_state.count != 1 || !callback_state.valid) {
            ++failures;
          }
        }
      });
    }
    for (std::thread &worker : workers)
      worker.join();
    EXPECT_EQ(failures.load(), 0u);

    ASSERT_EQ(dlclose(handle), 0) << dlerror();
    void *still_loaded =
        dlopen(ROCJITSU_WAITCHECK_LIBRARY_PATH, RTLD_NOW | RTLD_LOCAL | RTLD_NOLOAD);
    EXPECT_EQ(still_loaded, nullptr) << "waitcheck DSO retained after dlclose";
    if (still_loaded != nullptr)
      dlclose(still_loaded);
    EXPECT_EQ(thread_count(), initial_thread_count);
    EXPECT_FALSE(is_object_loaded("libhsa-runtime64.so"));
  }
}

} // namespace
