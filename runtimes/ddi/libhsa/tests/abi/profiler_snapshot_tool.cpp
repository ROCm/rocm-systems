// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Deliberately independent C ABI declaration: the runtime package does not
// require rocprofiler-register headers merely to verify its dynamic boundary.
#define AMD_INTERNAL_BUILD
#include <hsa/hsa_api_trace.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <mutex>
#include <thread>

using Mutator = int (*)(void *, void **, uint64_t, uint64_t);
struct Provider {
  size_t size;
  uint64_t version;
  void *data;
  uint32_t (*update)(void *, Mutator, void *);
  uint32_t (*run_cleanup)(void *, uint64_t, int (*)(void *), void *);
};
static_assert(sizeof(Provider) == 40);

namespace {
Provider provider{};
std::atomic<uint32_t> registrations{0}, refreshes{0}, quiesces{0};
std::atomic<uint32_t> active_updates{0}, failed_quiesce{0}, nested_status{0};
std::atomic<decltype(CoreApiTable::hsa_system_get_info_fn)> original{nullptr};
decltype(CoreApiTable::hsa_signal_create_fn) signal_create = nullptr;
decltype(CoreApiTable::hsa_signal_destroy_fn) signal_destroy = nullptr;
decltype(CoreApiTable::hsa_signal_store_screlease_fn) signal_store = nullptr;
decltype(CoreApiTable::hsa_signal_wait_scacquire_fn) signal_wait = nullptr;
decltype(AmdExtTable::hsa_amd_signal_async_handler_fn) signal_handler = nullptr;
std::atomic<uint32_t> callback_status{HSA_STATUS_ERROR};
std::mutex pause_mutex;
std::condition_variable pause_cv;
bool paused = false;
bool released = false;
uint64_t last_generation = 0;
bool expect_status(const char *stage, uint32_t actual, uint32_t expected = 0) {
  if (actual == expected)
    return true;
  std::fprintf(stderr, "profiler fixture %s: got 0x%x, expected 0x%x\n", stage,
               actual, expected);
  return false;
}

hsa_status_t query(hsa_system_info_t attribute, void *value, uint16_t marker) {
  auto status = original.load(std::memory_order_acquire)(attribute, value);
  if (status == HSA_STATUS_SUCCESS &&
      attribute == HSA_SYSTEM_INFO_VERSION_MAJOR)
    *static_cast<uint16_t *>(value) = marker;
  return status;
}
hsa_status_t alpha(hsa_system_info_t attribute, void *value) {
  return query(attribute, value, 111);
}
hsa_status_t beta(hsa_system_info_t attribute, void *value) {
  return query(attribute, value, 222);
}
int mutate(void *data, void **tables, uint64_t count, uint64_t) {
  if (count != 1 || !tables || !tables[0])
    return -1;
  active_updates.fetch_add(1);
  auto *table = static_cast<HsaApiTable *>(tables[0]);
  auto expected = decltype(CoreApiTable::hsa_system_get_info_fn){};
  original.compare_exchange_strong(expected,
                                   table->core_->hsa_system_get_info_fn);
  if (!signal_create) {
    signal_create = table->core_->hsa_signal_create_fn;
    signal_destroy = table->core_->hsa_signal_destroy_fn;
    signal_store = table->core_->hsa_signal_store_screlease_fn;
    signal_wait = table->core_->hsa_signal_wait_scacquire_fn;
    signal_handler = table->amd_ext_->hsa_amd_signal_async_handler_fn;
  }
  auto mode = reinterpret_cast<uintptr_t>(data);
  table->core_->hsa_system_get_info_fn = mode == 1 ? beta : alpha;
  if (mode == 2) {
    std::unique_lock lock(pause_mutex);
    paused = true;
    pause_cv.notify_all();
    pause_cv.wait(lock, [] { return released; });
    uint16_t version = 0;
    // This nested call must still inherit the admitted generation after its
    // public admission has closed on the shutdown thread.
    nested_status = original.load()(HSA_SYSTEM_INFO_VERSION_MAJOR, &version);
  }
  active_updates.fetch_sub(1);
  return 0;
}
int cleanup(void *) {
  uint16_t version = 0;
  if (original.load()(HSA_SYSTEM_INFO_VERSION_MAJOR, &version) !=
      HSA_STATUS_SUCCESS)
    return -1;
  // A native cleanup lease never grants table-update admission.
  return provider.update(provider.data, mutate, nullptr) == 9 ? 0 : -1;
}
bool signal_callback(hsa_signal_value_t, void *) {
  uint16_t version = 0;
  callback_status = original.load()(HSA_SYSTEM_INFO_VERSION_MAJOR, &version);
  return false;
}
int signal_cleanup(void *data) {
  auto signal = *static_cast<hsa_signal_t *>(data);
  signal_store(signal, 0);
  return signal_wait(signal, HSA_SIGNAL_CONDITION_EQ, 0, 0,
                     HSA_WAIT_STATE_ACTIVE) == 0
             ? 0
             : -1;
}
} // namespace

extern "C" {
uint32_t rocprofiler_register_library_api_table_v2(const char *, uint32_t (*)(),
                                                   uint32_t,
                                                   const Provider *input,
                                                   uint64_t *id) {
  if (!input || input->size != sizeof(Provider) || input->version != 2)
    return 8;
  provider = *input;
  ++registrations;
  *id = 17;
  return provider.update(provider.data, mutate, nullptr);
}
uint32_t rocprofiler_register_refresh_library_api_table_v2(uint64_t id) {
  if (id != 17)
    return 8;
  ++refreshes;
  return provider.update(provider.data, mutate, nullptr);
}
uint32_t
rocprofiler_register_prepare_library_api_table_v2(uint64_t id,
                                                  uint64_t generation) {
  if (id != 17 || active_updates.load() != 0)
    return 6;
  hsa_signal_t signal{};
  callback_status = HSA_STATUS_ERROR;
  if (!expect_status("prepare signal create",
                     signal_create(1, 0, nullptr, &signal)))
    return 6;
  if (!expect_status("prepare signal handler",
                     signal_handler(signal, HSA_SIGNAL_CONDITION_EQ, 0,
                                    signal_callback, nullptr)))
    return 6;
  uint32_t result = 6;
  std::thread worker([&] {
    uint16_t version = 0;
    // Ordinary callers remain outside the generation during preparation.
    if (!expect_status("outsider query",
                       original.load()(HSA_SYSTEM_INFO_VERSION_MAJOR, &version),
                       HSA_STATUS_ERROR_NOT_INITIALIZED))
      return;
    if (last_generation && provider.run_cleanup(provider.data, last_generation,
                                                cleanup, nullptr) != 9)
      return;
    result = provider.run_cleanup(provider.data, generation, cleanup, nullptr);
    expect_status("delegated query", result);
    if (result == 0)
      result = provider.run_cleanup(provider.data, generation, signal_cleanup,
                                    &signal);
    expect_status("delegated signal store/wait", result);
  });
  worker.join();
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(2);
  while (callback_status == HSA_STATUS_ERROR &&
         std::chrono::steady_clock::now() < deadline)
    std::this_thread::yield();
  if (!expect_status("async callback query", callback_status) ||
      !expect_status("prepare signal destroy", signal_destroy(signal)))
    result = 6;
  last_generation = generation;
  return result;
}
uint32_t rocprofiler_register_quiesce_library_api_table_v2(uint64_t id) {
  if (id != 17 || active_updates.load() != 0 || failed_quiesce.load())
    return 6;
  if (!expect_status("nested final cleanup",
                     provider.run_cleanup(provider.data, last_generation,
                                          cleanup, nullptr)))
    return 6;
  uint32_t outsider = 0;
  std::thread worker([&] {
    outsider =
        provider.run_cleanup(provider.data, last_generation, cleanup, nullptr);
  });
  worker.join();
  if (!expect_status("closed worker cleanup", outsider, 9))
    return 6;
  ++quiesces;
  return 0;
}
uint32_t test_update(uint32_t mode) {
  return provider.update(provider.data, mutate,
                         reinterpret_cast<void *>(uintptr_t(mode)));
}
uint32_t test_count(uint32_t index) {
  switch (index) {
  case 0:
    return registrations.load();
  case 1:
    return refreshes.load();
  case 2:
    return quiesces.load();
  case 3:
    return nested_status.load();
  default:
    return 0;
  }
}
void test_wait_paused() {
  std::unique_lock lock(pause_mutex);
  pause_cv.wait(lock, [] { return paused; });
}
void test_release() {
  std::lock_guard lock(pause_mutex);
  released = true;
  pause_cv.notify_all();
}
void test_fail_quiesce() { failed_quiesce = 1; }
}
