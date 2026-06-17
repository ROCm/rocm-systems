// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_VM_AMDGPU_MEMORY_SIDE_CACHE_H_
#define ROCJITSU_VM_AMDGPU_MEMORY_SIDE_CACHE_H_

#include "simdojo/components/cache.h"
#include "simdojo/sim/component.h"
#include "simdojo/sim/message.h"

#include <array>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <utility>

namespace rocjitsu {
namespace amdgpu {

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
/// All public methods are thread-safe. Striped locking (by cache set index)
/// serializes access because multiple XCDs assigned to the same IOD may share
/// this MSC from different partition threads. Stripes allow concurrent access
/// to different cache sets, eliminating contention for non-overlapping addresses.
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

  explicit MemorySideCache(std::string name) : simdojo::Component(std::move(name)) {
    req_ = add_port(std::make_unique<simdojo::Port>("req", 0, this, simdojo::PortDirection::OUT,
                                                    simdojo::PortProtocol::MEMORY));
  }

  void read(uint64_t addr, uint8_t *dst, uint32_t size, uint32_t vmid = 0);
  void write(uint64_t addr, const uint8_t *src, uint32_t size, uint32_t vmid = 0);

  /// @brief Flush all dirty lines to the backing store and invalidate.
  void flush_all();

  void initialize() override {
    for (auto &p : ports()) {
      if (p->direction() == simdojo::PortDirection::IN && !p->recv_event()->has_handler()) {
        p->recv_event()->set_handler([this](simdojo::Tick, simdojo::Message *msg) {
          auto &hdr = msg->header();
          auto *data = reinterpret_cast<uint8_t *>(msg->payload());
          if (hdr.op == simdojo::MessageOp::READ)
            read(hdr.addr, data, hdr.size_bytes, hdr.vmid);
          else if (hdr.op == simdojo::MessageOp::WRITE)
            write(hdr.addr, data, hdr.size_bytes, hdr.vmid);
          hdr.op = simdojo::MessageOp::RESPONSE;
        });
      }
    }
  }

  simdojo::Port *req_port() { return req_; }

  simdojo::Port *create_cpl_port(const std::string &src_name) {
    auto port_id = static_cast<simdojo::PortID>(cpl_ports_.size() + 1);
    auto port =
        std::make_unique<simdojo::Port>("cpl_" + src_name, port_id, this,
                                        simdojo::PortDirection::IN, simdojo::PortProtocol::MEMORY);
    auto *raw = add_port(std::move(port));
    raw->recv_event()->set_handler([this](simdojo::Tick, simdojo::Message *msg) {
      auto &hdr = msg->header();
      auto *data = reinterpret_cast<uint8_t *>(msg->payload());
      if (hdr.op == simdojo::MessageOp::READ)
        read(hdr.addr, data, hdr.size_bytes, hdr.vmid);
      else if (hdr.op == simdojo::MessageOp::WRITE)
        write(hdr.addr, data, hdr.size_bytes, hdr.vmid);
      hdr.op = simdojo::MessageOp::RESPONSE;
    });
    cpl_ports_.push_back(raw);
    return raw;
  }

private:
  void ensure_line(uint64_t addr, uint32_t vmid);

  /// @brief Send a read or write request to the backing store via the req port.
  void send_backing(uint64_t addr, uint8_t *data, uint32_t size, simdojo::MessageOp op,
                    uint32_t vmid);

  /// @brief Return the stripe index for a given address.
  uint32_t stripe_index(uint64_t addr) const {
    return CacheStore::set_index(addr) & (STRIPE_COUNT - 1);
  }

  mutable std::array<std::mutex, STRIPE_COUNT> stripes_;
  mutable std::mutex flush_mutex_; ///< Exclusive lock for flush_all (must not race with stripes).
  CacheStore cache_;
  simdojo::Port *req_ = nullptr;
  std::vector<simdojo::Port *> cpl_ports_;
};

} // namespace amdgpu
} // namespace rocjitsu

#endif // ROCJITSU_VM_AMDGPU_MEMORY_SIDE_CACHE_H_
