// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file l2_cache.h
/// @brief L2 cache component shared per XCD.

#ifndef ROCJITSU_VM_AMDGPU_L2_CACHE_H_
#define ROCJITSU_VM_AMDGPU_L2_CACHE_H_

#include "rocjitsu/vm/amdgpu/device_cache_coherence.h"
#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "rocjitsu/vm/amdgpu/mtype.h"
#include "simdojo/components/cache.h"
#include "simdojo/sim/component.h"
#include "simdojo/sim/message.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <utility>

namespace rocjitsu {
namespace amdgpu {

/// @brief L2 cache component shared per Accelerator Complex Die (XCD).
///
/// 128-byte lines, 2048 sets, 16-way set-associative = 4MB (default).
/// In functional mode, writes are kept visible through the backing store
/// (HBM controller or memory-side cache) via the requester port.
///
/// Mtype-aware behavior:
///   - UC: Flush/invalidate any resident L2 line, then bypass L2 and
///         forward directly to backing store.
///   - CC: Allocate in L2, MOESI coherence state tracking for CPU-GPU
///         shared memory. Write-through on CC stores.
///   - RW/WB: Allocate in L2, with functional-mode stores written through to
///            the backing store so completed dispatches are globally visible.
///   - NT: Allocate in L2 (L1 is bypassed, but L2 still caches).
///
/// Ordinary cache entries are keyed by (VMID, GPU virtual line), not resolved
/// backing identity. Different aliases of the same backing line therefore keep
/// independent cached copies. Functional write() stores update backing memory
/// but do not invalidate other aliases; CC/UC access or explicit cache
/// maintenance is required before an alias refetches that data. Dirty lines are
/// written back through their owning VMID. Atomic RMW is intentionally stronger:
/// L2Cache coordinates all registered device caches before the backing RMW,
/// above GPU-VA-to-host mapping differences.
///
/// Serves as the backing store for both L1 Scalar (K$) and L1 Vector (V$).
///
/// Provides structural ports for the topology graph (IN for CU L1 miss
/// requests, OUT for HBM/fabric traffic).
///
/// @par Thread safety
/// Public cache operations are thread-safe. Normal line operations take a
/// shared maintenance lock followed by a per-set mutex, so CPU dispatch workers
/// sharing an XCD-local L2 can make progress on independent cache sets.
/// Whole-cache maintenance takes the maintenance lock exclusively, avoiding
/// both concurrent set access and TSan's fixed lock-tracker limit. Each line
/// operation is atomic with respect to other operations on that set; a request
/// spanning multiple lines is intentionally not an atomic snapshot of the full
/// range.
class L2Cache : public simdojo::Component {
public:
  static constexpr uint32_t LINE_SIZE_BITS = 7; // 128 bytes
  static constexpr uint32_t NUM_SETS = 2048;
  static constexpr uint32_t ASSOCIATIVITY = 16;

  using CacheStore = simdojo::Cache<LINE_SIZE_BITS, NUM_SETS, ASSOCIATIVITY>;
  static constexpr uint32_t LINE_SIZE = CacheStore::LINE_SIZE;

  /// @brief Construct an L2Cache component.
  /// @param name Human-readable name (e.g., "xcd0.l2").
  explicit L2Cache(std::string name);
  ~L2Cache() override;

  L2Cache(const L2Cache &) = delete;
  L2Cache &operator=(const L2Cache &) = delete;
  L2Cache(L2Cache &&) = delete;
  L2Cache &operator=(L2Cache &&) = delete;

  /// @brief Set the requester port used to reach the backing store.
  /// @param port The OUT port connected to the MSC or HBM controller.
  void set_req_port(simdojo::Port *port) { req_port_ = port; }

  /// @brief Set the backing memory for direct writeback in functional mode.
  /// @param mem GpuMemory instance (used when req_port_ has no link).
  void set_backing_memory(GpuMemory *mem) { backing_memory_ = mem; }

  uint64_t backing_read_transactions() const {
    return backing_read_transactions_.load(std::memory_order_relaxed);
  }
  uint64_t backing_write_transactions() const {
    return backing_write_transactions_.load(std::memory_order_relaxed);
  }

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

  uint64_t write_count() const { return write_count_.load(std::memory_order_relaxed); }

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

  /// @brief Write back only the modified bytes from a full L1 line snapshot.
  /// @param line_addr Line-aligned address.
  /// @param[in] data Full cache line data (LINE_SIZE bytes).
  /// @param dirty_offset First modified byte within the line.
  /// @param dirty_size Number of modified bytes.
  /// @param mtype Memory type for caching policy.
  /// @param vmid Owning process address space.
  void writeback_line(uint64_t line_addr, const uint8_t *data, uint32_t dirty_offset,
                      uint32_t dirty_size, Mtype mtype = Mtype::RW, uint32_t vmid = 0);

  /// @brief Perform an atomic read-modify-write on a cache line.
  ///
  /// @details Establishes a process-wide device coherence boundary: dirty
  /// scalar and L2 state is published, the device coherence epoch is advanced,
  /// and every L2 remains quiescent through the backing RMW. Cached clean lines
  /// are discarded lazily on their next ordinary access.
  ///
  /// @param addr The memory address of the atomic target.
  /// @param size Access size in bytes (4 or 8).
  /// @param fn Callback: fn(line_data_ptr, line_offset). Must read the
  ///           old value, compute the new value, and write it in place.
  template <typename F> void atomic_rmw(uint64_t addr, uint32_t size, F &&fn, uint32_t vmid = 0) {
    [[maybe_unused]] auto boundary = DeviceCacheCoherence::instance().acquire_atomic_boundary();

    if (backing_memory_) {
      backing_read_transactions_.fetch_add(1, std::memory_order_relaxed);
      backing_write_transactions_.fetch_add(1, std::memory_order_relaxed);
      backing_memory_->atomic_rmw(addr, size, [&](uint8_t *target) { fn(target, 0); }, vmid);
    } else {
      assert((size == sizeof(uint32_t) || size == sizeof(uint64_t)) &&
             "L2 atomic size must be 4 or 8 bytes");
      std::array<uint8_t, sizeof(uint64_t)> target{};
      send_backing(addr, target.data(), size, simdojo::MessageOp::READ, vmid);
      fn(target.data(), 0);
      send_backing(addr, target.data(), size, simdojo::MessageOp::WRITE, vmid);
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
  void invalidate_all() {
    auto maintenance_lock = acquire_cache_maintenance();
    cache_.invalidate_all();
    clear_all_dirty_bytes();
  }

  /// @brief Invalidate L2 lines covering an address range.
  /// @details Used after a host/SDMA write has completed. Bytes inside the
  /// range are authoritative in backing memory. Dirty bytes outside a partial
  /// line update are preserved before the target VMID's line is invalidated.
  /// @param vmid Owning process address space to invalidate.
  void invalidate_range(uint64_t addr, uint32_t size, uint32_t vmid);

  /// @brief Flush all dirty L2 lines to HBM and invalidate.
  /// @param vmid Ignored. Each dirty line is written back under its own owning
  /// vmid (recorded in the line tag), so a caller-supplied vmid cannot be
  /// correct for a global flush. Retained only for call-site signature symmetry.
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
  friend class DeviceCacheCoherence;

  static constexpr uint64_t MAX_INVALIDATE_RANGE_SET_LOCKS = 64;

  std::mutex &set_mutex(uint64_t addr) const { return set_mutexes_[CacheStore::set_index(addr)]; }

  class SetRangeLocks {
  public:
    SetRangeLocks(std::array<std::mutex, NUM_SETS> &mutexes, uint64_t line_start,
                  uint64_t line_count)
        : mutexes_(mutexes) {
      std::array<bool, NUM_SETS> seen{};
      for (uint64_t i = 0; i < line_count; ++i) {
        const uint64_t line_addr = line_start + i * LINE_SIZE;
        uint32_t set = CacheStore::set_index(line_addr);
        if (!seen[set]) {
          seen[set] = true;
          sets_[count_++] = set;
        }
      }

      std::sort(sets_.begin(), sets_.begin() + count_);
      try {
        for (size_t i = 0; i < count_; ++i) {
          mutexes_[sets_[i]].lock();
          ++locked_;
        }
      } catch (...) {
        unlock_all();
        throw;
      }
    }

    SetRangeLocks(const SetRangeLocks &) = delete;
    SetRangeLocks &operator=(const SetRangeLocks &) = delete;

    ~SetRangeLocks() { unlock_all(); }

  private:
    void unlock_all() {
      while (locked_ > 0) {
        --locked_;
        mutexes_[sets_[locked_]].unlock();
      }
    }

    std::array<std::mutex, NUM_SETS> &mutexes_;
    std::array<uint32_t, NUM_SETS> sets_{};
    size_t count_ = 0;
    size_t locked_ = 0;
  };

  SetRangeLocks lock_sets_for_range(uint64_t line_start, uint64_t line_count) const {
    return SetRangeLocks(set_mutexes_, line_start, line_count);
  }

  std::shared_lock<std::shared_mutex> acquire_cache_access();
  std::unique_lock<std::shared_mutex> acquire_cache_maintenance();
  void synchronize_epoch_locked();
  void ensure_line(uint64_t addr, uint32_t vmid = 0);
  void flush_line_locked(uint64_t addr, uint32_t vmid = 0);
  void flush_dirty_locked();
  using DirtyMask = std::array<uint64_t, LINE_SIZE / 64>;
  void mark_dirty_bytes(uint64_t line_addr, uint32_t offset, uint32_t size, uint32_t vmid);
  bool clear_dirty_bytes(uint64_t line_addr, uint32_t offset, uint32_t size, uint32_t vmid);
  void publish_dirty_bytes(uint64_t line_addr, const uint8_t *data, uint32_t vmid,
                           uint32_t discard_offset = 0, uint32_t discard_size = 0);
  void clear_all_dirty_bytes();
  void invalidate_range_locked(uint64_t addr, uint32_t size, uint32_t vmid, uint64_t line_start,
                               uint64_t line_count);
  void send_backing(uint64_t addr, uint8_t *data, uint32_t size, simdojo::MessageOp op,
                    uint32_t vmid = 0);

  CacheStore cache_;
  mutable std::shared_mutex maintenance_mutex_;
  mutable std::array<std::mutex, NUM_SETS> set_mutexes_;
  simdojo::Port *req_port_ = nullptr;
  GpuMemory *backing_memory_ = nullptr; ///< Direct writeback path (functional mode).
  uint64_t coherence_epoch_ = 0;
  std::mutex dirty_bytes_mutex_;
  std::map<std::pair<uint32_t, uint64_t>, DirtyMask> dirty_bytes_;
  std::atomic<bool> has_dirty_lines_{false};
  std::vector<simdojo::Port *> cpl_ports_;
  std::atomic<uint64_t> write_count_ = 0; ///< Debug: total L2 writes (for trace).
  // Relaxed atomics: independent cache operations can update these counters
  // concurrently; the values are diagnostic only.
  std::atomic<uint64_t> backing_read_transactions_{0};
  std::atomic<uint64_t> backing_write_transactions_{0};
};

} // namespace amdgpu
} // namespace rocjitsu

#endif // ROCJITSU_VM_AMDGPU_L2_CACHE_H_
