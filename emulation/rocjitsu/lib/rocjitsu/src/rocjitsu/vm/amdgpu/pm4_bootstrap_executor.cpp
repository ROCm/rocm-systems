// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/amdgpu/pm4_bootstrap_executor.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>

namespace rocjitsu::amdgpu {

namespace {

constexpr uint32_t kPacketType3 = 3;
constexpr uint32_t kPacket3Nop = 0x10;
constexpr uint32_t kPacket3SetUconfigRegister = 0x79;
constexpr uint64_t kPacket3SetUconfigRegisterStart = 0xc000;

Pm4BootstrapOutcome to_bootstrap_outcome(VmAccessOutcome outcome) {
  switch (outcome) {
  case VmAccessOutcome::Complete:
    return Pm4BootstrapOutcome::Complete;
  case VmAccessOutcome::Unavailable:
    return Pm4BootstrapOutcome::Unavailable;
  case VmAccessOutcome::Faulted:
    return Pm4BootstrapOutcome::Faulted;
  case VmAccessOutcome::Malformed:
    return Pm4BootstrapOutcome::Malformed;
  }
  return Pm4BootstrapOutcome::Malformed;
}

VmAccessOutcome read_ring(const GpuVmAccess &access, uint64_t ring_base, uint64_t ring_dwords,
                          uint64_t pointer, std::span<std::byte> bytes) {
  const uint64_t ring_bytes = ring_dwords * sizeof(uint32_t);
  const uint64_t offset = (pointer % ring_dwords) * sizeof(uint32_t);
  const std::size_t first =
      std::min<std::size_t>(bytes.size(), static_cast<std::size_t>(ring_bytes - offset));
  VmAccessOutcome outcome = access.read(ring_base + offset, bytes.first(first));
  if (outcome != VmAccessOutcome::Complete || first == bytes.size())
    return outcome;
  return access.read(ring_base, bytes.subspan(first));
}

} // namespace

Pm4BootstrapResult Pm4BootstrapExecutor::execute(const GpuVmAccess &access, uint64_t ring_base,
                                                 uint64_t ring_dwords, uint64_t read_pointer,
                                                 uint64_t write_pointer) const {
  Pm4BootstrapResult result{.outcome = Pm4BootstrapOutcome::Malformed,
                            .read_pointer = read_pointer};
  if (ring_base == 0 || ring_dwords == 0 ||
      ring_dwords > std::numeric_limits<uint64_t>::max() / sizeof(uint32_t) ||
      ring_base > std::numeric_limits<uint64_t>::max() - ring_dwords * sizeof(uint32_t) ||
      write_pointer < read_pointer || write_pointer - read_pointer > ring_dwords) {
    return result;
  }

  while (result.read_pointer != write_pointer) {
    std::array<std::byte, sizeof(uint32_t)> raw_header{};
    const VmAccessOutcome header_outcome =
        read_ring(access, ring_base, ring_dwords, result.read_pointer, raw_header);
    if (header_outcome != VmAccessOutcome::Complete) {
      result.outcome = to_bootstrap_outcome(header_outcome);
      return result;
    }

    result.packet_header = std::bit_cast<uint32_t>(raw_header);
    const uint32_t type = result.packet_header >> 30;
    const uint32_t opcode = (result.packet_header >> 8) & 0xff;
    // The startup sequence uses the driver's special one-dword NOP encoding;
    // its count field is padding, not a packet length to fetch.
    if (type == kPacketType3 && opcode == kPacket3Nop) {
      ++result.read_pointer;
      continue;
    }

    const uint32_t packet_dwords = ((result.packet_header >> 16) & 0x3fff) + 2;
    if (type != kPacketType3 || opcode != kPacket3SetUconfigRegister || packet_dwords != 3 ||
        packet_dwords > write_pointer - result.read_pointer) {
      result.outcome = Pm4BootstrapOutcome::UnsupportedPacket;
      return result;
    }

    std::array<std::byte, 3 * sizeof(uint32_t)> raw_packet{};
    const VmAccessOutcome packet_outcome =
        read_ring(access, ring_base, ring_dwords, result.read_pointer, raw_packet);
    if (packet_outcome != VmAccessOutcome::Complete) {
      result.outcome = to_bootstrap_outcome(packet_outcome);
      return result;
    }
    const std::array<uint32_t, 3> packet = std::bit_cast<std::array<uint32_t, 3>>(raw_packet);
    result.register_dword = kPacket3SetUconfigRegisterStart + packet[1];
    if (!callbacks_.write_uconfig_register ||
        !callbacks_.write_uconfig_register(result.register_dword, packet[2])) {
      result.outcome = Pm4BootstrapOutcome::RegisterWriteRejected;
      return result;
    }
    result.read_pointer += packet_dwords;
  }

  result.outcome = Pm4BootstrapOutcome::Complete;
  return result;
}

} // namespace rocjitsu::amdgpu
