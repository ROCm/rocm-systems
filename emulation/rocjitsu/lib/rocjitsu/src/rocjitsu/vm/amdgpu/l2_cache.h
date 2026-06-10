// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file l2_cache.h
/// @brief L2 cache component shared per XCD.

#ifndef ROCJITSU_VM_AMDGPU_L2_CACHE_H_
#define ROCJITSU_VM_AMDGPU_L2_CACHE_H_

#include "rocjitsu/vm/amdgpu/mtype.h"
#include "simdojo/components/cache.h"
#include "simdojo/sim/component.h"
#include "simdojo/sim/message.h"

#include <array>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <utility>

namespace rocjitsu {
namespace amdgpu {

class GpuMemory; // Forward declaration for backing store writeback.

/// @brief L2 cache component shared per Accelerator Complex Die (XCD).
///
/// 128-byte lines, 2048 sets, 16-way set-associative = 4MB (default).
/// In functional mode, writes are kept visible through the backing store
/// (HBM controller or memory-side cache) via the requester port.
///
/// Mtype-aware behavior:
///   - UC: Bypass L2, forward directly to backing store.
///   - CC: Allocate in L2, MOESI coherence state tracking for CPU-GPU
///         shared memory. Write-through on CC stores.
///   - RW/WB: Allocate in L2, with functional-mode stores written through to
///            the backing store so completed dispatches are globally visible.
///   - NT: Allocate in L2 (L1 is bypassed, but L2 still caches).
///
/// Serves as the backing store for both L1 Scalar (K$) and L1 Vector (V$).
///
/// Provides structural ports for the topology graph (IN for CU L1 miss
/// requests, OUT for HBM/fabric traffic).
///
/// THREAD SAFETY: The L2's read(), write(), fetch_line(), writeback_line(),
/// ensure_line(), and flush_line() methods are NOT thread-safe. They operate on
/// the underlying CacheStore without locking. Only atomic_rmw() is protected
/// (via striped mutexes for cross-CU atomics to the same L2).
///
/// All CUs sharing an L2 instance MUST be assigned to the same simulation
/// partition. This invariant ensures that only one worker thread accesses the
/// L2's non-atomic paths at any given time. Violating this constraint (e.g.,
/// placing CUs connected to the same L2 in different partitions) will cause
/// data races on the cache data structure. The partitioner must enforce this
/// constraint.
class L2Cache : public simdojo::Component {
public:
  static constexpr uint32_t LINE_SIZE_BITS = 7; // 128 bytes
  static constexpr uint32_t NUM_SETS = 2048;
  static constexpr uint32_t ASSOCIATIVITY = 16;

  using CacheStore = simdojo::Cache<LINE_SIZE_BITS, NUM_SETS, ASSOCIATIVITY>;
  static constexpr uint32_t LINE_SIZE = CacheStore::LINE_SIZE;

  /// @brief Construct an L2Cache component.
  /// @param name Human-readable name (e.g., "xcd0.l2").
  explicit L2Cache(std::string name) : simdojo::Component(std::move(name)) {
    req_port_ = add_port(std::make_unique<simdojo::Port>(
        "req", 0, this, simdojo::PortDirection::OUT, simdojo::PortProtocol::MEMORY));
  }

  /// @brief Set the requester port used to reach the backing store.
  /// @param port The OUT port connected to the MSC or HBM controller.
  void set_req_port(simdojo::Port *port) { req_port_ = port; }

  /// @brief Set the backing memory for direct writeback in functional mode.
  /// @param mem GpuMemory instance (used when req_port_ has no link).
  void set_backing_memory(GpuMemory *mem) { backing_memory_ = mem; }

  /// @brief Read a cache line worth of data (or partial line).
  ///
  /// Used by L1 controllers to fetch on miss. In functional mode this refetches
  /// from backing memory to avoid stale clean lines across independent L2s.
  /// @param addr The starting address (line-aligned or not).
  /// @param dst Destination buffer.
  /// @param size Number of bytes to read.
  /// @param mtype Memory type for caching policy.
  void read(uint64_t addr, uint8_t *dst, uint32_t size, Mtype mtype = Mtype::RW, uint32_t vmid = 0);

  /// @brief Write data to L2 (and possibly through to HBM).
  ///
  /// Used by L1 for write-through (CC) and write-back evictions.
  /// @param addr The starting address.
  /// @param src Source data.
  /// @param size Number of bytes to write.
  /// @param mtype Memory type for caching policy.
  void write(uint64_t addr, const uint8_t *src, uint32_t size, Mtype mtype = Mtype::RW,
             uint32_t vmid = 0);

  /// @brief Fetch an entire cache line into the given buffer.
  ///
  /// Convenience method for L1 fills. Returns a full LINE_SIZE-byte line
  /// at the line-aligned address containing addr.
  /// @param addr Any address within the desired cache line.
  /// @param[out] line_buf Buffer of at least LINE_SIZE bytes.
  void fetch_line(uint64_t addr, uint8_t *line_buf, uint32_t vmid = 0);

  /// @brief Write back a full cache line from L1 eviction.
  /// @param line_addr Line-aligned address.
  /// @param[in] data Full cache line data (LINE_SIZE bytes).
  /// @param mtype Memory type for caching policy.
  void writeback_line(uint64_t line_addr, const uint8_t *data, Mtype mtype = Mtype::RW,
                      uint32_t vmid = 0);

  /// @brief Perform an atomic read-modify-write on a cache line.
  ///
  /// @details Ensures the line is present in L2, then calls the provided
  /// function with a pointer to the line data and the byte offset within
  /// the line. The entire operation is serialized under a mutex so that
  /// concurrent atomic RMWs (from different simulation threads in
  /// multi-threaded mode) are correctly ordered, matching real hardware
  /// L2 arbitration.
  ///
  /// @param addr The memory address of the atomic target.
  /// @param size Access size in bytes (4 or 8).
  /// @param fn Callback: fn(line_data_ptr, line_offset). Must read the
  ///           old value, compute the new value, and write it in place.
  template <typename F> void atomic_rmw(uint64_t addr, uint32_t size, F &&fn, uint32_t vmid = 0) {
    uint32_t stripe = (addr >> LINE_SIZE_BITS) & (ATOMIC_STRIPE_COUNT - 1);
    std::lock_guard<std::mutex> lock(atomic_stripes_[stripe]);
    ensure_line(addr, vmid);

    uint32_t offset = CacheStore::line_offset(addr);
    uint8_t *line = cache_.line_data_for_write(addr);
    assert(line != nullptr && "ensure_line must guarantee hit");

    fn(line, offset);

    send_backing(addr, line + offset, size, simdojo::MessageOp::WRITE, vmid);

    simdojo::CacheTag *ctag = nullptr;
    cache_.lookup(addr, &ctag);
    if (ctag) {
      ctag->coherence = simdojo::CoherenceState::MODIFIED;
      ctag->dirty = false;
    }
  }

  /// @brief Flush a single L2 line (writeback if dirty, then invalidate).
  ///
  /// Used by CC reads to ensure cross-XCD coherence: dirty data is
  /// written back to the backing store before the line is invalidated,
  /// so a subsequent ensure_line() refetch gets the latest data.
  /// @param addr Any address within the target cache line.
  void flush_line(uint64_t addr, uint32_t vmid = 0);

  /// @brief Invalidate all L2 lines.
  void invalidate_all() { cache_.invalidate_all(); }

  /// @brief Invalidate L2 lines covering an address range.
  /// @details Used after host/SDMA writes to ensure GPU reads reload from
  /// backing store. No writeback — the backing store already has latest data.
  void invalidate_range(uint64_t addr, uint32_t size);

  /// @brief Flush all dirty L2 lines to HBM and invalidate.
  void flush_all(uint32_t vmid = 0);

  /// @brief Create a completer port for a CU connection (one per CU).
  /// @param name Name suffix for the port (used for port naming).
  /// @returns Pointer to the newly created completer port.
  simdojo::Port *create_cpl_port(const std::string &name) {
    auto port_id = static_cast<simdojo::PortID>(cpl_ports_.size() + 1);
    auto port = std::make_unique<simdojo::Port>(
        "cpl_" + name, port_id, this, simdojo::PortDirection::IN, simdojo::PortProtocol::MEMORY);
    auto *raw = add_port(std::move(port));
    cpl_ports_.push_back(raw);
    return raw;
  }

  /// @brief Return the requester port (for topology wiring to HBM/fabric).
  /// @returns Pointer to the requester port.
  simdojo::Port *req_port() { return req_port_; }

  /// @brief Return all completer ports (CU-facing).
  /// @returns Const reference to the vector of completer ports.
  const std::vector<simdojo::Port *> &cpl_ports() const { return cpl_ports_; }

private:
  void ensure_line(uint64_t addr, uint32_t vmid = 0);
  void send_backing(uint64_t addr, uint8_t *data, uint32_t size, simdojo::MessageOp op,
                    uint32_t vmid = 0);

  static constexpr uint32_t ATOMIC_STRIPE_COUNT = 64;

  CacheStore cache_;
  simdojo::Port *req_port_ = nullptr;
  GpuMemory *backing_memory_ = nullptr; ///< Direct writeback path (functional mode).
  /// @brief Striped locks for atomic RMW serialization. Each stripe covers
  /// a range of cache lines, allowing atomics to different lines to proceed
  /// in parallel (matching real L2 arbitration behavior).
  std::array<std::mutex, ATOMIC_STRIPE_COUNT> atomic_stripes_;
  std::vector<simdojo::Port *> cpl_ports_;
  uint64_t write_count_ = 0; ///< Debug: total L2 writes (for trace).
};

} // namespace amdgpu
} // namespace rocjitsu

#endif // ROCJITSU_VM_AMDGPU_L2_CACHE_H_
