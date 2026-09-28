// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/amdgpu/pci/ip_block_model.h"

#include "rocjitsu/vm/amdgpu/pci/mmio_registers.h"
#include "util/log.h"

#include <algorithm>
#include <format>
#include <utility>

namespace rocjitsu {

IpRegisterWindow::IpRegisterWindow(RegisterAperture &registers, std::span<const uint64_t> segments,
                                   std::string owner)
    : registers_(&registers), segments_(segments), owner_(std::move(owner)) {}

std::optional<uint64_t> IpRegisterWindow::resolve(uint32_t segment_index, uint32_t dword) const {
  if (segment_index >= segments_.size()) {
    util::Logger::warn(std::format(
        "{}: segment {} was asked for, but the discovery table publishes {} for this block, so "
        "its registers cannot be answered",
        owner_, segment_index, segments_.size()));
    return std::nullopt;
  }
  const uint64_t segment = segments_[segment_index];
  // A segment equal to the aperture's dword count already names the dword one
  // past the end, so this is not a `>`. Refusing here is what stops a segment
  // near the top of the range multiplying into a small byte offset that would
  // pass the aperture bounds test below and land on an unrelated register.
  if (segment >= registers_->dwords()) {
    util::Logger::warn(
        std::format("{}: the block at segment {:#x} is not within a {}-byte register aperture, so "
                    "its registers cannot be answered",
                    owner_, segment, registers_->bytes()));
    return std::nullopt;
  }
  const uint64_t at = byte_offset_of_dword(segment + dword);
  if (!registers_->holds(at)) {
    return std::nullopt;
  }
  return at;
}

bool IpRegisterWindow::define(uint32_t segment_index, uint32_t dword, uint32_t value) {
  const std::optional<uint64_t> at = resolve(segment_index, dword);
  return at.has_value() && registers_->define(*at, value);
}

bool IpRegisterWindow::define_read_only(uint32_t segment_index, uint32_t dword, uint32_t value) {
  const std::optional<uint64_t> at = resolve(segment_index, dword);
  return at.has_value() && registers_->define_read_only(*at, value);
}

uint32_t IpRegisterWindow::read(uint32_t segment_index, uint32_t dword) const {
  const std::optional<uint64_t> at = resolve(segment_index, dword);
  return at.has_value() ? registers_->value(*at) : 0;
}

bool IpRegisterWindow::write(uint32_t segment_index, uint32_t dword, uint32_t value) {
  const std::optional<uint64_t> at = resolve(segment_index, dword);
  return at.has_value() && registers_->store(*at, value);
}

IpBlockModel::IpBlockModel(std::string what, IpRegisterWindow registers)
    : registers_(std::move(registers)), what_(std::move(what)) {}

std::string overlapping_claim(std::vector<ClaimedRegisters> claims) {
  // Sorted rather than compared pairwise: the map is built once at construction
  // over a handful of blocks, and sorting makes the check linear in the thing
  // that is being built anyway.
  std::ranges::sort(claims, {}, &ClaimedRegisters::first_dword);
  for (std::size_t sorted_index = 1; sorted_index < claims.size(); ++sorted_index) {
    const ClaimedRegisters &before = claims[sorted_index - 1];
    const ClaimedRegisters &after = claims[sorted_index];
    if (before.count == 0 || after.count == 0) {
      continue;
    }
    if (before.first_dword + before.count > after.first_dword) {
      return std::format("{} claims {} register(s) from dword {:#x} and {} claims {} from {:#x}, "
                         "which overlap; whichever defined them last would answer for both",
                         before.owner, before.count, before.first_dword, after.owner, after.count,
                         after.first_dword);
    }
  }
  return {};
}

} // namespace rocjitsu
