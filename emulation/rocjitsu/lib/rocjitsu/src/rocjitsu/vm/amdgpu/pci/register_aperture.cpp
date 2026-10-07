// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/amdgpu/pci/register_aperture.h"

#include "rocjitsu/vm/amdgpu/pci/mmio_registers.h"
#include "util/log.h"

#include <format>
#include <utility>

namespace rocjitsu {

RegisterAperture::RegisterAperture(std::string owner, uint64_t bytes)
    : owner_(std::move(owner)), bytes_(bytes) {
  const auto count = static_cast<std::size_t>(dword_index_of(bytes));
  values_.assign(count, 0);
  modelled_.assign(count, false);
  read_only_.assign(count, false);
}

std::size_t RegisterAperture::index_of(uint64_t byte_offset) const {
  if ((byte_offset % kRegisterBytes) != 0 || byte_offset > bytes_ ||
      kRegisterBytes > bytes_ - byte_offset) {
    return values_.size();
  }
  return static_cast<std::size_t>(dword_index_of(byte_offset));
}

bool RegisterAperture::holds(uint64_t byte_offset) const {
  const std::lock_guard lock(*mutex_);
  return index_of(byte_offset) != values_.size();
}

bool RegisterAperture::define(uint64_t byte_offset, uint32_t value) {
  const std::lock_guard lock(*mutex_);
  const std::size_t index = index_of(byte_offset);
  if (index == values_.size()) {
    util::Logger::warn(
        std::format("{}: register at byte {:#x} is outside the {}-byte register aperture and will "
                    "not be modelled",
                    owner_, byte_offset, bytes_));
    return false;
  }
  values_[index] = value;
  modelled_[index] = true;
  read_only_[index] = false;
  return true;
}

bool RegisterAperture::define_read_only(uint64_t byte_offset, uint32_t value) {
  const std::lock_guard lock(*mutex_);
  const std::size_t index = index_of(byte_offset);
  if (index == values_.size()) {
    util::Logger::warn(
        std::format("{}: register at byte {:#x} is outside the {}-byte register aperture and will "
                    "not be modelled",
                    owner_, byte_offset, bytes_));
    return false;
  }
  values_[index] = value;
  modelled_[index] = true;
  read_only_[index] = true;
  return true;
}

bool RegisterAperture::modelled(uint64_t byte_offset) const {
  const std::lock_guard lock(*mutex_);
  const std::size_t index = index_of(byte_offset);
  return index != values_.size() && modelled_[index];
}

bool RegisterAperture::read_only(uint64_t byte_offset) const {
  const std::lock_guard lock(*mutex_);
  const std::size_t index = index_of(byte_offset);
  return index != values_.size() && read_only_[index];
}

uint32_t RegisterAperture::value(uint64_t byte_offset) const {
  const std::lock_guard lock(*mutex_);
  const std::size_t index = index_of(byte_offset);
  return index == values_.size() ? 0 : values_[index];
}

bool RegisterAperture::store(uint64_t byte_offset, uint32_t value) {
  const std::lock_guard lock(*mutex_);
  const std::size_t index = index_of(byte_offset);
  if (index == values_.size() || !modelled_[index]) {
    return false;
  }
  values_[index] = value;
  return true;
}

} // namespace rocjitsu
