// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file sdma_block_model.h
/// @brief Firmware-free SDMA startup registers.

#pragma once

#include "rocjitsu/vm/amdgpu/pci/ip_block_model.h"
#include "rocjitsu/vm/amdgpu/pci/ip_discovery.h"
#include "rocjitsu/vm/amdgpu/sdma_queue_runner.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

namespace rocjitsu {

class SoC;

namespace amdgpu {
class QueueBackend;
enum class QueueDoorbellDisposition : uint8_t;
} // namespace amdgpu

/// @brief The SDMA status needed after the driver primes its synthetic ucode.
class SdmaBlockModel final : public IpBlockModel {
public:
  struct UserQueue {
    uint64_t ring_base = 0;
    uint64_t read_pointer_address = 0;
    uint64_t ring_bytes = 0;
    uint64_t doorbell_offset = 0;
    uint32_t process_id = 0;
    uint32_t engine = 0;
    amdgpu::AddressSpaceHandle address_space;
    std::optional<uint64_t> initial_read_pointer = std::nullopt;
  };

  [[nodiscard]] static std::unique_ptr<SdmaBlockModel> create(const IpBlock &block,
                                                              IpRegisterWindow registers);

  [[nodiscard]] std::vector<RegisterClaim> claims() const override;
  bool reset() override;
  /// @brief Release all retained queue execution and VM-access state.
  void teardown_queues();
  void attach_soc(SoC *soc) { soc_ = soc; }
  [[nodiscard]] std::shared_ptr<amdgpu::QueueBackend>
  make_user_queue_backend(PciMemoryAccess &memory, uint32_t engine);
  [[nodiscard]] bool map_user_queue(UserQueue queue);
  [[nodiscard]] bool update_user_queue(uint64_t doorbell_offset, uint64_t ring_base,
                                       uint64_t ring_bytes, uint32_t queue_percentage);
  [[nodiscard]] amdgpu::QueueDoorbellDisposition
  notify_user_queue(uint64_t doorbell_offset, uint64_t write_pointer, PciMemoryAccess &memory);
  [[nodiscard]] bool unmap_user_queue(uint64_t doorbell_offset);
  [[nodiscard]] DoorbellDisposition observe_doorbell_write(uint64_t byte_offset, uint64_t value,
                                                           std::size_t width,
                                                           PciMemoryAccess &memory) override;

private:
  struct Queue {
    uint64_t ring_base = 0;
    uint64_t read_pointer_address = 0;
    uint64_t ring_bytes = 0;
    uint64_t read_pointer = 0;
    uint64_t doorbell_offset = 0;
    uint32_t register_offset = 0;
    uint32_t process_id = 0;
    uint32_t engine = 0;
    bool active = false;
    bool register_backed = true;
    bool faulted = false;
    amdgpu::AddressSpaceHandle address_space;
    std::optional<uint64_t> initial_read_pointer = std::nullopt;
    std::unique_ptr<amdgpu::SdmaQueueRunner> runner;
  };

  explicit SdmaBlockModel(IpRegisterWindow registers);

  [[nodiscard]] Queue configured_queue(uint32_t engine) const;
  [[nodiscard]] amdgpu::QueueDoorbellDisposition process_queue(Queue &queue, uint64_t write_pointer,
                                                               PciMemoryAccess &memory);
  [[nodiscard]] amdgpu::SdmaExecutor make_executor(const Queue &queue, PciMemoryAccess &memory);
  [[nodiscard]] static bool has_in_flight_state(const Queue &queue);
  std::vector<Queue> user_queues_;
  std::array<std::optional<Queue>, 2> register_queues_;
  SoC *soc_ = nullptr;
};

} // namespace rocjitsu
