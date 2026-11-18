// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file sdma_block_model.h
/// @brief Firmware-free SDMA startup registers.

#pragma once

#include "rocjitsu/vm/amdgpu/gpu_queue_registry.h"
#include "rocjitsu/vm/amdgpu/pci/ip_block_model.h"
#include "rocjitsu/vm/amdgpu/pci/ip_discovery.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

namespace rocjitsu {

class SoC;

namespace amdgpu {
class SdmaQueueBindingFactory;
enum class QueueSubmissionStatus : uint8_t;
} // namespace amdgpu

/// @brief The SDMA status needed after the driver primes its synthetic ucode.
class SdmaBlockModel final : public IpBlockModel {
public:
  [[nodiscard]] static std::unique_ptr<SdmaBlockModel> create(const IpBlock &block,
                                                              IpRegisterWindow registers);

  [[nodiscard]] std::vector<RegisterClaim> claims() const override;
  bool reset() override;
  /// @brief Release all retained queue execution and VM-access state.
  void teardown_queues();
  void attach_soc(SoC *soc) { soc_ = soc; }
  void
  attach_queue_binding_factory(std::shared_ptr<amdgpu::SdmaQueueBindingFactory> binding_factory) {
    queue_binding_factory_ = std::move(binding_factory);
  }
  /// @brief Publish asynchronous scheduler progress on the PCI owner thread.
  void update_queue_progress(uint32_t engine, uint64_t consumer_cursor, bool terminal);
  [[nodiscard]] DoorbellDisposition observe_doorbell_write(uint64_t byte_offset, uint64_t value,
                                                           std::size_t width,
                                                           PciMemoryAccess &memory) override;

private:
  class Queue {
  public:
    uint64_t ring_base = 0;
    uint64_t read_pointer_address = 0;
    uint64_t ring_bytes = 0;
    uint64_t read_pointer = 0;
    uint64_t doorbell_offset = 0;
    uint32_t register_offset = 0;
    uint32_t process_id = 0;
    uint32_t engine = 0;
    bool active = false;
    bool faulted = false;
    amdgpu::AddressSpaceHandle address_space;
    amdgpu::QueueHandle queue_handle;
    std::optional<uint64_t> initial_read_pointer = std::nullopt;
  };

  explicit SdmaBlockModel(IpRegisterWindow registers);

  [[nodiscard]] Queue configured_queue(uint32_t engine) const;
  [[nodiscard]] amdgpu::QueueSubmissionStatus process_queue(Queue &queue, uint64_t write_pointer);
  std::array<std::optional<Queue>, 2> register_queues_;
  SoC *soc_ = nullptr;
  std::shared_ptr<amdgpu::SdmaQueueBindingFactory> queue_binding_factory_;
};

} // namespace rocjitsu
