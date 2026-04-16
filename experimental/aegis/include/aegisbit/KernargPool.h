//===-- aegisbit/KernargPool.h - Pre-allocated kernarg pool -----*- C++ -*-===//
//
// Part of the AegisBit Project
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Small pre-allocated pool of GPU-accessible kernarg buffers used by
/// `TracingEngine::onDispatch` to avoid allocating in the HSA dispatch
/// callback. Buffers are drawn from the kernarg pool discovered by
/// `HSAPoolManager` and are filled on a background thread so the first
/// dispatch doesn't pay the HSA allocation cost.
///
/// Extracted from TracingEngine. The pool is intentionally leaked during
/// `TracingEngine::finalize()` (HSA free during __cxa_finalize corrupts the
/// glibc heap); the finalize sequence therefore calls `joinInit()` first to
/// reap the init thread, then `clear()` to drop the vector without freeing
/// the underlying pointers.
///
//===----------------------------------------------------------------------===//

#ifndef AEGISBIT_KERNARG_POOL_H
#define AEGISBIT_KERNARG_POOL_H

#include "aegisbit/HSAPoolManager.h"

#include <atomic>
#include <cstddef>
#include <mutex>
#include <thread>
#include <vector>

namespace aegisbit {

class KernargPool {
public:
  /// Construct with a reference to the pool manager used to source kernarg
  /// memory. The manager must outlive the pool. Non-const because the pool
  /// triggers `ensureDiscovered()` on its background init thread.
  explicit KernargPool(HSAPoolManager &Pools) : Pools(Pools) {}

  ~KernargPool();

  /// Maximum per-buffer size; `acquire(size > maxBufferSize())` returns null.
  static constexpr size_t kPoolSize = 16;
  static constexpr size_t kMaxBufferSize = 4096; // 4KB per kernarg
  static constexpr size_t maxBufferSize() { return kMaxBufferSize; }

  /// Kick off background initialization. HSA pools are assumed to already be
  /// (or become) discovered by the time the background thread runs; the
  /// thread itself triggers discovery if needed via the pool manager.
  /// Idempotent — the second call is a no-op.
  void startInitAsync();

  /// Block until the background init thread finishes, if it was started.
  /// Safe to call multiple times.
  void joinInit();

  /// Synchronously populate the pool if not already populated. Called from
  /// the background thread, but may be called directly in tests.
  void ensurePopulated();

  /// Acquire a buffer of at least `Size` bytes. Returns nullptr if the pool
  /// is empty or `Size > maxBufferSize()`.
  void *acquire(size_t Size);

  /// Return a buffer to the pool. No-op for nullptr.
  void release(void *Ptr);

  /// Drop all buffer pointers without freeing HSA memory. Used by finalize.
  void clear();

  /// Current pool depth (for stats/tests).
  size_t size() const;

private:
  HSAPoolManager &Pools;

  mutable std::mutex Mutex;
  std::vector<void *> Buffers;
  bool Populated = false;

  std::atomic<bool> InitStarted{false};
  std::thread InitThread;
};

} // namespace aegisbit

#endif // AEGISBIT_KERNARG_POOL_H
