// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/amdgpu/pm4_packet_processor.h"

namespace rocjitsu::amdgpu {
namespace {

constexpr uint32_t kPacketType3 = 3;
constexpr uint32_t kPacket3Nop = 0x10;
constexpr uint32_t kPacket3SetUconfigRegister = 0x79;
constexpr uint64_t kPacket3SetUconfigRegisterStart = 0xc000;

} // namespace

Pm4PacketProcessResult Pm4PacketProcessor::process(Request request) const {
  Pm4PacketProcessResult result;
  if (request.available_dwords.empty()) {
    result.packet.status = PacketProcessStatus::NeedInput;
    result.packet.required_bytes = sizeof(uint32_t);
    return result;
  }

  result.packet_header = request.available_dwords.front();
  const uint32_t type = result.packet_header >> 30;
  const uint32_t opcode = (result.packet_header >> 8) & 0xff;

  // The supported compute-queue stream uses the driver's special one-dword NOP
  // encoding, whose count field is padding rather than a packet length.
  if (type == kPacketType3 && opcode == kPacket3Nop) {
    result.packet = {.status = PacketProcessStatus::Complete,
                     .retirement = PacketRetirement::Retire,
                     .retirement_bytes = sizeof(uint32_t)};
    return result;
  }

  const uint32_t packet_dwords = ((result.packet_header >> 16) & 0x3fff) + 2;
  if (type != kPacketType3 || opcode != kPacket3SetUconfigRegister || packet_dwords != 3) {
    result.packet.status = PacketProcessStatus::Unsupported;
    result.diagnostic = Pm4PacketDiagnostic::UnsupportedPacket;
    return result;
  }

  result.packet.retirement_bytes = packet_dwords * sizeof(uint32_t);
  if (request.available_dwords.size() < packet_dwords) {
    result.packet.status = PacketProcessStatus::NeedInput;
    result.packet.required_bytes = result.packet.retirement_bytes;
    return result;
  }

  result.register_dword = kPacket3SetUconfigRegisterStart + request.available_dwords[1];
  if (!callbacks_.write_uconfig_register) {
    result.packet.status = PacketProcessStatus::Unsupported;
    result.diagnostic = Pm4PacketDiagnostic::RegisterWriteRejected;
    return result;
  }

  const Pm4RegisterWriteStatus write_status =
      callbacks_.write_uconfig_register(result.register_dword, request.available_dwords[2]);
  if (write_status != Pm4RegisterWriteStatus::Complete) {
    result.packet.status =
        write_status == Pm4RegisterWriteStatus::Blocked   ? PacketProcessStatus::Blocked
        : write_status == Pm4RegisterWriteStatus::Faulted ? PacketProcessStatus::Faulted
                                                          : PacketProcessStatus::Unsupported;
    if (write_status == Pm4RegisterWriteStatus::Rejected)
      result.diagnostic = Pm4PacketDiagnostic::RegisterWriteRejected;
    return result;
  }

  result.packet.status = PacketProcessStatus::Complete;
  result.packet.retirement = PacketRetirement::Retire;
  return result;
}

} // namespace rocjitsu::amdgpu
