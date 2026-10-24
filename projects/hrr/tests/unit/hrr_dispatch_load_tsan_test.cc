/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

// Races HRR's dispatch slot store against the HIP entry-point slot load.
//
// This binary is built with -fsanitize=thread. Reader threads call through one slot of a
// dispatch table with hip::LoadDispatchSlot, the helper behind every entry point in
// hip_table_interface.cpp. One writer thread installs and uninstalls a shim in that slot
// with the release store HRR capture uses (store_slot in hip_capture.cpp). The shim reads
// state that the writer set before the first install, as HRR shims read g_real_table.
//
// With an ordinary read in LoadDispatchSlot, ThreadSanitizer reports a race on the slot.
// With a relaxed load it reports a race on the shim state. The acquire load passes.

#include "hip_dispatch_load.hpp"

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>

extern "C" const char* __tsan_default_options() { return "halt_on_error=1:exitcode=66"; }

namespace {

constexpr int kReaders = 4;
constexpr int kCycles = 2000;

// One slot laid out like a HipDispatchTable entry (hip_api_trace.hpp): a size, then
// function pointers. Only hip_dispatch_load.hpp is needed, so the test builds with any host
// compiler, without the HIP headers or libamdhip64.
using GetDeviceCountFn = int (*)(int*);
struct DispatchTable {
  std::size_t size;
  GetDeviceCountFn hipGetDeviceCount_fn;
};

DispatchTable g_table{};
GetDeviceCountFn g_real = nullptr;  // written once before install, read by the shim
std::atomic<bool> g_done{false};
std::atomic<long> g_real_calls{0};
std::atomic<long> g_shim_calls{0};

int real_get_device_count(int* count) {
  *count = 1;
  g_real_calls.fetch_add(1, std::memory_order_relaxed);
  return 0;
}

int shim_get_device_count(int* count) {
  g_shim_calls.fetch_add(1, std::memory_order_relaxed);
  return g_real(count);
}

// The store HRR capture uses on Linux (store_slot in hip_capture.cpp).
void store_slot(GetDeviceCountFn* slot, GetDeviceCountFn value) {
  __atomic_store_n(slot, value, __ATOMIC_RELEASE);
}

void reader() {
  while (!g_done.load(std::memory_order_acquire)) {
    int count = 0;
    if (hip::LoadDispatchSlot(g_table.hipGetDeviceCount_fn)(&count) != 0 || count != 1)
      std::abort();
  }
}

// Waits until a reader has made a call counted by `calls`. The load is relaxed, so the wait
// gives ThreadSanitizer no happens-before edge between the writer and the readers.
bool wait_for_call(const std::atomic<long>& calls) {
  auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(60);
  while (calls.load(std::memory_order_relaxed) == 0) {
    if (std::chrono::steady_clock::now() > deadline) return false;
    std::this_thread::yield();
  }
  return true;
}

}  // namespace

int main() {
  g_table.size = sizeof(g_table);
  g_table.hipGetDeviceCount_fn = real_get_device_count;

  std::vector<std::thread> readers;
  for (int i = 0; i < kReaders; ++i) readers.emplace_back(reader);

  // Like hip_capture_init: snapshot the real table first, then install. A reader has to
  // call the real function before the install and the shim after it, so the race is
  // exercised on any schedule.
  bool exercised = wait_for_call(g_real_calls);
  g_real = real_get_device_count;
  store_slot(&g_table.hipGetDeviceCount_fn, shim_get_device_count);
  exercised = exercised && wait_for_call(g_shim_calls);
  for (int i = 0; exercised && i < kCycles; ++i) {
    store_slot(&g_table.hipGetDeviceCount_fn, real_get_device_count);
    std::this_thread::yield();
    store_slot(&g_table.hipGetDeviceCount_fn, shim_get_device_count);
    std::this_thread::yield();
  }
  g_done.store(true, std::memory_order_release);
  for (auto& t : readers) t.join();

  std::printf("calls: %ld through the shim, %ld in total\n", g_shim_calls.load(),
              g_real_calls.load());
  if (!exercised) {
    std::fprintf(stderr, "no reader called through the slot in 60 s; race not exercised\n");
    return 1;
  }
  return 0;
}
