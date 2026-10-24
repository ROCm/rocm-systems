// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/vm/amdgpu/device_cache_coherence.h"
#include "rocjitsu/vm/amdgpu/gpu_vm.h"
#include "simdojo/components/cache.h"
#include "simdojo/sim/component.h"
#include "simdojo/sim/message.h"

#include <array>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace rocjitsu {
namespace amdgpu {

class GpuMemory;
class PreparedCanonicalWriteback;
class MemorySideCacheTestAccess;
class WriterPreferredAccessGateTestAccess;

/// @brief Shared access gate that gives queued exclusive owners priority.
///
/// @details This type is internal to the memory-side cache implementation but
/// is kept separate so its writer-preference invariant can be tested directly.
class WriterPreferredAccessGate {
public:
  void lock_shared();
  bool try_lock_shared();
  void unlock_shared();
  void lock();
  void unlock();

private:
  friend class WriterPreferredAccessGateTestAccess;

  std::mutex mutex_;
  std::condition_variable cv_;
  uint32_t active_readers_ = 0;
  uint32_t waiting_writers_ = 0;
  bool writer_active_ = false;
};

/// @brief Memory-side cache component sitting between L2 and HBM on each IOD.
///
/// @details Write-through, write-allocate cache, and no mtype awareness. All traffic from
/// L2 is already filtered (UC bypasses L2, so it never reaches the MSC).
/// On miss, fetches from the backing store via the requester port (typically HbmController).
/// Stores are pushed straight to the backing store and the line is left clean
/// (EXCLUSIVE), so the daemon's process_vm_readv bridge sees writes immediately;
/// the for_each_dirty writeback in flush_all() is retained as a safety net.
///
/// Provides structural ports for the topology graph. The backing store is reached
/// through the requester port (req), which is connected to the HBM controller via a link.
///
/// @par Thread safety
/// @c read(), @c write(), and @c flush_all() are thread-safe. Reads and writes
/// acquire the writer-preference access gate in shared mode before acquiring a
/// stripe mutex by cache set index. A waiting flush blocks new shared entrants,
/// then acquires the gate exclusively after active accesses finish. This
/// guarantees flush progress while preserving concurrent access to different
/// cache sets. The required lock order is access gate before stripe; no path
/// may acquire them in the opposite order.
class MemorySideCache : public simdojo::Component {
public:
  static constexpr uint32_t LINE_SIZE_BITS = 7; // 128 bytes
  static constexpr uint32_t NUM_SETS = 65536;   // 65536 sets x 16-way x 128B = 128MB
  static constexpr uint32_t ASSOCIATIVITY = 16;

  using CacheStore = simdojo::Cache<LINE_SIZE_BITS, NUM_SETS, ASSOCIATIVITY>;
  static constexpr uint32_t LINE_SIZE = CacheStore::LINE_SIZE;

  /// @brief Number of lock stripes. Power of 2 for fast masking.
  /// 256 stripes covers 65536 sets (256 sets per stripe).
  static constexpr uint32_t STRIPE_COUNT = 256;

  explicit MemorySideCache(
      std::string name,
      std::shared_ptr<DeviceCacheCoherence> coherence = std::make_shared<DeviceCacheCoherence>(),
      GpuMemory *legacy_maintenance_memory = nullptr);
  ~MemorySideCache() override;

  /// @brief Rebind this cache to a device-local coherence domain before use.
  void set_coherence_domain(std::shared_ptr<DeviceCacheCoherence> coherence);
  const std::shared_ptr<DeviceCacheCoherence> &coherence_domain() const { return coherence_; }
  /// @brief Set the raw VMID-0 backing used by out-of-band maintenance.
  void set_legacy_maintenance_memory(GpuMemory *memory) { legacy_maintenance_memory_ = memory; }
  /// @brief Set the VM service used by maintenance for nonzero VMIDs.
  /// @details Ordinary MSC traffic continues through its requester port; this
  /// pointer exists only for a quiesced maintenance operation.
  void set_legacy_maintenance_vm(GpuVm *gpu_vm) { legacy_maintenance_vm_ = gpu_vm; }
  /// @brief Set the VM service used to identify nonzero-VMID cache lines and
  /// validate speculative fills. Without a VM service, those requests bypass
  /// the cache so dirty data cannot be published through a reused numeric VMID.
  void set_gpu_vm(GpuVm *gpu_vm) { gpu_vm_ = gpu_vm; }

  VmAccessOutcome read(uint64_t addr, uint8_t *dst, uint32_t size, uint32_t vmid = 0);
  VmAccessOutcome write(uint64_t addr, const uint8_t *src, uint32_t size, uint32_t vmid = 0);

  /// Probe only resident, byte-valid data under the supplied raw grant key.
  /// The span must be nonempty and contained in one MSC line. False is a miss;
  /// this method performs no VM admission or backing request.
  [[nodiscard]] bool try_read_peer_cached(uint64_t addr, std::span<uint8_t> dst, uint32_t vmid,
                                          uint64_t admission_generation, uint64_t cache_identity);

  /// Optionally fill one MSC line fragment from a canonical peer read.
  /// The caller must hold exact ingress, current-VM, and canonical-line
  /// freshness authority through this call. The span must be nonempty and
  /// contained in one line. A dirty-line refusal leaves tags and replacement
  /// order unchanged. This method never publishes a dirty victim or sends
  /// backing traffic.
  [[nodiscard]] bool try_fill_peer_cached(uint64_t addr, std::span<const uint8_t> src,
                                          uint32_t vmid, uint64_t admission_generation,
                                          uint64_t cache_identity);
  VmAccessOutcome atomic_modify(uint64_t addr, uint32_t size,
                                const simdojo::MemoryAtomicMutation &mutation, uint32_t vmid = 0);

  /// @brief Flush all dirty lines to the backing store and invalidate.
  VmAccessOutcome flush_all();

  /// @brief Prepare a byte range for an external write in one VMID.
  /// @details Before the external write commits, publish dirty resident lines
  /// overlapping the range in full, then invalidate every overlapping line.
  /// Unrelated lines and VMIDs remain resident. On transient admission or
  /// backing failure, retain the dirty line and return the failure for retry.
  /// If VM admission is absent or stale, discard the line without writing
  /// through a revoked grant. Earlier lines in a failed range may already be
  /// invalidated. A zero size changes nothing. Ranges past UINT64_MAX stop at
  /// the last addressable line; large ranges scan bounded cache storage. The
  /// external-write owner must exclude alias refills before this pre-write
  /// flush until invalidate_range_after_external_write() completes and the
  /// post-effect invalidation is acknowledged.
  VmAccessOutcome flush_range(uint64_t addr, uint32_t size, uint32_t vmid);

  /// @brief Publish whole dirty canonical lines through a prepared grant.
  /// @details A dirty line must fit entirely in one retained backing view with
  /// matching VM identity and a nonzero grant generation. The caller holds the
  /// prepared ingress permit until this method releases the access gate. Failed
  /// writes and refused VM effects retain the line for retry.
  VmAccessOutcome flush_range(uint64_t addr, uint32_t size, uint32_t vmid,
                              const PreparedCanonicalWriteback &prepared);

  /// @brief Discard resident lines after an external write has committed.
  /// @details Drops dirty and clean lines overlapping the range in @p vmid
  /// without publishing their contents. The external-write owner must exclude
  /// alias refills from before the pre-write flush through this post-effect
  /// drop and its acknowledgment. Call without a DeviceCacheMaintenanceLease:
  /// that lease already owns the MSC access gate. Unrelated lines and VMIDs
  /// remain resident. A zero size changes nothing, and ranges past UINT64_MAX
  /// stop at the last addressable line. Large ranges scan bounded cache
  /// storage. This operation takes only the MSC access gate; it does not
  /// acquire a maintenance lease, enter an effect, or access backing/VM.
  void invalidate_range_after_external_write(uint64_t addr, uint32_t size, uint32_t vmid);

  void initialize() override {
    for (const std::unique_ptr<simdojo::Port> &port : ports()) {
      if (port->direction() == simdojo::PortDirection::IN && !port->recv_event()->has_handler()) {
        port->recv_event()->set_handler([this](simdojo::Tick, simdojo::Message *message) {
          simdojo::MessageHeader &header = message->header();
          auto *data = reinterpret_cast<uint8_t *>(message->payload());
          VmAccessOutcome outcome = VmAccessOutcome::Malformed;
          if (header.op == simdojo::MessageOp::READ)
            outcome = read(header.addr, data, header.size_bytes, header.vmid);
          else if (header.op == simdojo::MessageOp::WRITE)
            outcome = write(header.addr, data, header.size_bytes, header.vmid);
          else if (header.op == simdojo::MessageOp::ATOMIC) {
            auto *mutation = reinterpret_cast<simdojo::MemoryAtomicMutation *>(message->payload());
            outcome = mutation != nullptr
                          ? atomic_modify(header.addr, header.size_bytes, *mutation, header.vmid)
                          : VmAccessOutcome::Malformed;
          }
          if (header.completion_status != nullptr)
            *header.completion_status = message_status(outcome);
          header.op = simdojo::MessageOp::RESPONSE;
        });
      }
    }
  }

  simdojo::Port *req_port() { return req_; }

  simdojo::Port *create_cpl_port(const std::string &src_name) {
    simdojo::PortID port_id = static_cast<simdojo::PortID>(cpl_ports_.size() + 1);
    auto port =
        std::make_unique<simdojo::Port>("cpl_" + src_name, port_id, this,
                                        simdojo::PortDirection::IN, simdojo::PortProtocol::MEMORY);
    simdojo::Port *raw = add_port(std::move(port));
    raw->recv_event()->set_handler([this](simdojo::Tick, simdojo::Message *message) {
      simdojo::MessageHeader &header = message->header();
      auto *data = reinterpret_cast<uint8_t *>(message->payload());
      VmAccessOutcome outcome = VmAccessOutcome::Malformed;
      if (header.op == simdojo::MessageOp::READ)
        outcome = read(header.addr, data, header.size_bytes, header.vmid);
      else if (header.op == simdojo::MessageOp::WRITE)
        outcome = write(header.addr, data, header.size_bytes, header.vmid);
      else if (header.op == simdojo::MessageOp::ATOMIC) {
        auto *mutation = reinterpret_cast<simdojo::MemoryAtomicMutation *>(message->payload());
        outcome = mutation != nullptr
                      ? atomic_modify(header.addr, header.size_bytes, *mutation, header.vmid)
                      : VmAccessOutcome::Malformed;
      }
      if (header.completion_status != nullptr)
        *header.completion_status = message_status(outcome);
      header.op = simdojo::MessageOp::RESPONSE;
    });
    cpl_ports_.push_back(raw);
    return raw;
  }

private:
  friend class DeviceCacheCoherence;
  friend class MemorySideCacheTestAccess;
  friend class KfdPreparedProviderTestAccess;

  [[nodiscard]] VmAccessOutcome cache_partial_bytes(uint64_t addr, const uint8_t *src,
                                                    uint32_t size, uint32_t vmid,
                                                    uint64_t admission_generation,
                                                    const GpuVmAccess *access);
  [[nodiscard]] VmAccessOutcome ensure_line(uint64_t addr, uint32_t vmid,
                                            uint64_t admission_generation,
                                            const GpuVmAccess *access, bool fetch_on_miss = true);
  [[nodiscard]] bool guarded_dirty_allocation_victim(uint64_t replacement_addr) const;
  [[nodiscard]] VmAccessAdmission admit_access(uint64_t addr, uint32_t size, uint32_t vmid,
                                               VmAccessKind access, bool report_fault = true) const;
  [[nodiscard]] VmAccessOutcome flush_dirty_locked();
  [[nodiscard]] VmAccessOutcome flush_dirty_to_legacy_backing_locked();
  /// @brief Mark a dirty resident before publishing its tag to another cache operation.
  void mark_line_dirty(simdojo::CacheTag &tag);
  [[nodiscard]] VmAccessOutcome publish_dirty_line(uint64_t line_addr, const simdojo::CacheTag &tag,
                                                   uint8_t *data, bool legacy_maintenance = false);
  /// @brief Visit overlapping resident lines while the MSC access gate is held exclusively.
  /// @details The visitor returns false to skip all later visitor calls.
  template <typename Visitor>
  void visit_resident_lines_in_range(uint64_t addr, uint32_t size, uint32_t vmid,
                                     Visitor &&visitor);
  GpuVm *cache_vm() const { return gpu_vm_ != nullptr ? gpu_vm_ : legacy_maintenance_vm_; }
  bool has_legacy_maintenance_backing() const {
    return legacy_maintenance_memory_ != nullptr && legacy_maintenance_vm_ != nullptr;
  }

  /// @brief Send a read or write request to the backing store via the req port.
  [[nodiscard]] VmAccessOutcome send_backing(uint64_t addr, uint8_t *data, uint32_t size,
                                             simdojo::MessageOp op, uint32_t vmid);
  [[nodiscard]] VmAccessOutcome send_atomic_backing(uint64_t addr, uint32_t size,
                                                    const simdojo::MemoryAtomicMutation &mutation,
                                                    uint32_t vmid);
  [[nodiscard]] static simdojo::MessageStatus message_status(VmAccessOutcome outcome);
  [[nodiscard]] static VmAccessOutcome access_outcome(simdojo::MessageStatus status);

  /// @brief Return the stripe index for a given address.
  uint32_t stripe_index(uint64_t addr) const {
    return CacheStore::set_index(addr) & (STRIPE_COUNT - 1);
  }

  mutable std::array<std::mutex, STRIPE_COUNT> stripes_;
  WriterPreferredAccessGate access_gate_;
  CacheStore cache_;
  // Shared-gate accesses on distinct stripes may create dirty lines concurrently.
  // Full maintenance only reads or clears this hint with the gate held exclusively.
  std::atomic<bool> may_have_dirty_lines_{false};
  uint64_t full_dirty_scans_ = 0;
  uint64_t skipped_clean_scans_ = 0;
  std::shared_ptr<DeviceCacheCoherence> coherence_;
  GpuMemory *legacy_maintenance_memory_ = nullptr;
  GpuVm *legacy_maintenance_vm_ = nullptr;
  GpuVm *gpu_vm_ = nullptr;
  simdojo::Port *req_ = nullptr;
  std::vector<simdojo::Port *> cpl_ports_;
};

} // namespace amdgpu
} // namespace rocjitsu
