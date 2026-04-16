//===-- KernargPool.cpp - Pre-allocated kernarg pool impl -----------------===//
//
// Part of the AegisBit Project
//
//===----------------------------------------------------------------------===//

#include "aegisbit/KernargPool.h"
#include "aegisbit/RuntimeConfig.h"

#ifdef AEGISBIT_HAS_GPU
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>
#endif

#include <chrono>
#include <cstring>
#include <string>

namespace aegisbit {

KernargPool::~KernargPool() {
  // Make sure the background thread can't outlive `this`. In normal
  // TracingEngine shutdown joinInit() is called explicitly before destruction
  // to preserve the finalize ordering invariant.
  joinInit();
}

void KernargPool::startInitAsync() {
  bool Expected = false;
  if (!InitStarted.compare_exchange_strong(Expected, true))
    return; // Already started

  InitThread = std::thread([this]() {
    // Small delay to let the main thread proceed.
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    // The pool manager is thread-safe; it will lazy-initialize if needed.
    Pools.ensureDiscovered();
    ensurePopulated();
  });
  RuntimeConfig::getInstance().log("Started background kernarg pool init");
}

void KernargPool::joinInit() {
  if (InitThread.joinable()) {
    InitThread.join();
    RuntimeConfig::getInstance().log("Kernarg pool init thread joined");
  }
}

void KernargPool::ensurePopulated() {
  std::lock_guard<std::mutex> Lock(Mutex);

  if (Populated)
    return;

  RuntimeConfig &Cfg = RuntimeConfig::getInstance();
  Cfg.log("Pre-allocating GPU kernarg pool from HSA...");

#ifdef AEGISBIT_HAS_GPU
  if (!Pools.isInitialized() || Pools.kernargPool() == 0) {
    Cfg.log("HSA pools not available, skipping kernarg pool allocation");
    Populated = true;
    return;
  }

  hsa_amd_memory_pool_t pool = {Pools.kernargPool()};

  size_t Created = 0;
  for (size_t i = 0; i < kPoolSize; ++i) {
    void *Ptr = nullptr;
    hsa_status_t status =
        hsa_amd_memory_pool_allocate(pool, kMaxBufferSize, 0, &Ptr);
    if (status == HSA_STATUS_SUCCESS && Ptr != nullptr) {
      std::memset(Ptr, 0, kMaxBufferSize);
      Buffers.push_back(Ptr);
      ++Created;
    } else {
      Cfg.log("Failed to pre-allocate kernarg " + std::to_string(i) +
              ": HSA allocation failed");
      break; // Stop on first failure
    }
  }
  Cfg.log("Pre-allocated " + std::to_string(Created) +
          " GPU kernarg buffers from HSA pool");
#else
  Cfg.log("GPU not available, kernarg pool not allocated");
#endif

  Populated = true;
}

void *KernargPool::acquire(size_t Size) {
  if (Size > kMaxBufferSize) {
    RuntimeConfig::getInstance().log("Kernarg size " + std::to_string(Size) +
                                     " exceeds max " +
                                     std::to_string(kMaxBufferSize));
    return nullptr;
  }

  std::lock_guard<std::mutex> Lock(Mutex);
  if (Buffers.empty())
    return nullptr;

  void *Ptr = Buffers.back();
  Buffers.pop_back();
  return Ptr;
}

void KernargPool::release(void *Ptr) {
  if (!Ptr)
    return;

  std::lock_guard<std::mutex> Lock(Mutex);
  // Cap the pool so misuse can't grow it unboundedly. In practice the pool
  // should never exceed 2x the initial size; if it does we simply leak the
  // excess pointer (safer than freeing from a callback context).
  if (Buffers.size() < kPoolSize * 2)
    Buffers.push_back(Ptr);
}

void KernargPool::clear() {
  std::lock_guard<std::mutex> Lock(Mutex);
  Buffers.clear();
}

size_t KernargPool::size() const {
  std::lock_guard<std::mutex> Lock(Mutex);
  return Buffers.size();
}

} // namespace aegisbit
