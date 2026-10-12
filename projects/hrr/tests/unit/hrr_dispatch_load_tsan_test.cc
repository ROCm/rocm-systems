/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

// Races HRR's dispatch slot store against the HIP entry-point slot load.
//
// This binary is built with -fsanitize=thread. Reader threads call through one slot of a
// real HipDispatchTable with hip::LoadDispatchSlot, the helper behind every entry point in
// hip_table_interface.cpp. One writer thread installs and uninstalls a shim in that slot
// with the release store HRR capture uses (store_slot in hip_capture.cpp). The shim reads
// state that the writer set before the first install, as HRR shims read g_real_table.
//
// With an ordinary read in LoadDispatchSlot, ThreadSanitizer reports a race on the slot.
// With a relaxed load it reports a race on the shim state. The acquire load passes.

#include <hip/amd_detail/hip_api_trace.hpp>

#include "hip_dispatch_load.hpp"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>

extern "C" const char* __tsan_default_options() { return "halt_on_error=1:exitcode=66"; }

namespace {

constexpr int kReaders = 4;
constexpr int kCycles = 2000;

HipDispatchTable g_table{};
t_hipGetDeviceCount g_real = nullptr;  // written once before install, read by the shim
std::atomic<bool> g_done{false};
std::atomic<long> g_real_calls{0};
std::atomic<long> g_shim_calls{0};

hipError_t real_get_device_count(int* count) {
  *count = 1;
  g_real_calls.fetch_add(1, std::memory_order_relaxed);
  return hipSuccess;
}

hipError_t shim_get_device_count(int* count) {
  g_shim_calls.fetch_add(1, std::memory_order_relaxed);
  return g_real(count);
}

// The store HRR capture uses on Linux (store_slot in hip_capture.cpp).
void store_slot(t_hipGetDeviceCount* slot, t_hipGetDeviceCount value) {
  __atomic_store_n(slot, value, __ATOMIC_RELEASE);
}

void reader() {
  while (!g_done.load(std::memory_order_acquire)) {
    int count = 0;
    if (hip::LoadDispatchSlot(g_table.hipGetDeviceCount_fn)(&count) != hipSuccess || count != 1)
      std::abort();
  }
}

}  // namespace

int main() {
  g_table.size = sizeof(g_table);
  g_table.hipGetDeviceCount_fn = real_get_device_count;

  std::vector<std::thread> readers;
  for (int i = 0; i < kReaders; ++i) readers.emplace_back(reader);

  // Like hip_capture_init: snapshot the real table first, then install.
  g_real = real_get_device_count;
  for (int i = 0; i < kCycles; ++i) {
    store_slot(&g_table.hipGetDeviceCount_fn, shim_get_device_count);
    std::this_thread::yield();
    store_slot(&g_table.hipGetDeviceCount_fn, real_get_device_count);
    std::this_thread::yield();
  }
  g_done.store(true, std::memory_order_release);
  for (auto& t : readers) t.join();

  long real_calls = g_real_calls.load();
  long shim_calls = g_shim_calls.load();
  std::printf("calls: %ld through the shim, %ld in total\n", shim_calls, real_calls);
  if (shim_calls == 0 || shim_calls == real_calls) {
    std::fprintf(stderr, "readers never saw both slot values; the race was not exercised\n");
    return 1;
  }
  return 0;
}
