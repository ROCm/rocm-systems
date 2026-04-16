//===-- aegisbit/TraceBufferAllocator.h - Trace buffer alloc ----*- C++ -*-===//
//
// Part of the AegisBit Project
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Allocates GPU-visible trace buffers for the instrumented path.
///
/// A trace buffer consists of a record/counter region and an 8-byte write-
/// offset counter, both sourced from an HSA pool exposed by
/// `HSAPoolManager`. The allocator prefers the fine-grained pool so that
/// device-side atomics work, and falls back to the kernarg pool when that
/// allocation fails. When fine-grained (VRAM-backed) memory is used, the CPU
/// agent is explicitly granted access so that the host can memset/readback.
///
/// Extracted from TracingEngine. Allocation results are intentionally leaked
/// during `TracingEngine::finalize()` (HSA free during __cxa_finalize
/// corrupts glibc heap), so `freeBuffer` is provided for completeness but is
/// not invoked by the engine today.
///
//===----------------------------------------------------------------------===//

#ifndef AEGISBIT_TRACE_BUFFER_ALLOCATOR_H
#define AEGISBIT_TRACE_BUFFER_ALLOCATOR_H

#include "aegisbit/HSAPoolManager.h"

#include <cstddef>
#include <tuple>

namespace aegisbit {

class TraceBufferAllocator {
public:
  /// Construct with a reference to the pool manager the allocator draws
  /// memory from. The manager must outlive the allocator.
  explicit TraceBufferAllocator(const HSAPoolManager &Pools) : Pools(Pools) {}

  /// Allocate a trace buffer + write-offset pair of `Size` bytes.
  /// Returns `{trace_buf, write_offset, actual_size, supports_gpu_atomics}`.
  /// All nullptr / 0 on failure (caller must check).
  std::tuple<void *, void *, size_t, bool> allocate(size_t Size);

  /// Free a previously allocated trace buffer + write-offset pair.
  /// No-op for null pointers. Not invoked by TracingEngine during shutdown
  /// because calling into HSA during __cxa_finalize corrupts the heap.
  void freeBuffer(void *TraceBufferPtr, void *WriteOffsetPtr);

private:
  const HSAPoolManager &Pools;
};

} // namespace aegisbit

#endif // AEGISBIT_TRACE_BUFFER_ALLOCATOR_H
