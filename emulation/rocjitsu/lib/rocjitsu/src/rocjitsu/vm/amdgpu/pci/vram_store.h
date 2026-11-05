// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file vram_store.h
/// @brief The device's video memory: a descriptor a guest can map, and the
///        window it maps.
///
/// @details Separate from the device because it is the one part of it that owns
/// operating-system resources. Keeping the descriptor, the mapping and their
/// lifetimes in one object means the device cannot half-release them, and means
/// the register model being split into per-IP-block units does not have to carry
/// memory ownership along with it.
///
/// The whole of memory is backed, sparsely; only the aperture is mapped. The
/// driver reaches past the aperture through the indirect window, which is why
/// the file is larger than the mapping and why reads and writes here are by
/// file offset rather than through the pointer.
///
/// Linux-only, like the device that owns it: sharing memory with a VMM by
/// descriptor is how every POSIX transport does it, and there is no portable
/// equivalent.

#pragma once

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <optional>
#include <span>
#include <string>

namespace rocjitsu {

struct VramAtomicCompareExchangeResult {
  uint64_t observed = 0;
  bool exchanged = false;
};

struct VramAtomicLoadResult {
  uint64_t value = 0;
};

/// @brief Memfd-backed video memory, with a mapped window onto its start.
class VramStore final {
public:
  /// @brief Back @p bytes of memory and map its first @p aperture_bytes.
  ///
  /// @details Reports its own failures against @p name and leaves the object
  /// unusable rather than throwing, matching how the device reports a
  /// configuration it cannot present.
  ///
  /// @param[in] name Owner's name, for diagnostics.
  /// @param[in] bytes Total memory to back.
  /// @param[in] aperture_bytes Window to map, which must not exceed @p bytes.
  VramStore(const std::string &name, uint64_t bytes, uint64_t aperture_bytes);
  ~VramStore();

  // Owns a descriptor and a mapping, and hands the descriptor to a transport.
  VramStore(const VramStore &) = delete;
  VramStore &operator=(const VramStore &) = delete;
  VramStore(VramStore &&) = delete;
  VramStore &operator=(VramStore &&) = delete;

  /// @brief Whether memory was backed and mapped.
  [[nodiscard]] bool usable() const { return fd_ >= 0 && mapping_ != nullptr; }

  /// @brief The descriptor to share with a transport, or negative when unusable.
  [[nodiscard]] int fd() const { return fd_; }

  /// @brief The mapped window, empty when unusable.
  [[nodiscard]] std::span<std::byte> aperture() const {
    return mapping_ == nullptr ? std::span<std::byte>{}
                               : std::span<std::byte>(mapping_, aperture_bytes_);
  }

  /// @brief Read one register-sized word at @p offset.
  /// @param[in] offset Byte offset into memory, which may lie past the aperture.
  /// @param[out] value The word read.
  /// @retval false The offset is outside memory, or the read failed.
  [[nodiscard]] bool read(uint64_t offset, uint32_t &value) const;

  /// @brief Read a whole buffer at @p at, resuming after a short read.
  /// @param[out] bytes Buffer to fill.
  /// @param[in] at Byte offset into memory.
  /// @retval false The read failed or memory ended before the buffer did.
  [[nodiscard]] bool read_all(std::span<std::byte> bytes, uint64_t at) const;

  /// @brief Write one register-sized word at @p offset.
  /// @param[in] offset Byte offset into memory, which may lie past the aperture.
  /// @param[in] value The word to store.
  /// @retval false The offset is outside memory, or the write failed.
  bool write(uint64_t offset, uint32_t value);

  /// @brief Store a whole buffer at @p at, resuming after a short write.
  ///
  /// @details A single write is permitted to transfer less than asked for, and
  /// a partial store of something the driver parses -- a discovery table, whose
  /// signature sits at the front -- surfaces as a malformed record rather than
  /// as an absent one, which is the harder failure to read.
  ///
  /// @param[in] bytes Buffer to store.
  /// @param[in] at Byte offset into memory.
  /// @retval false The store failed or memory would take no more.
  [[nodiscard]] bool write_all(std::span<const std::byte> bytes, uint64_t at);

  /// @brief Perform a strong 4- or 8-byte compare/exchange on the backing.
  /// @returns Empty for an invalid range, width, alignment, or backing.
  [[nodiscard]] std::optional<VramAtomicCompareExchangeResult>
  compare_exchange(uint64_t offset, uint32_t width, uint64_t expected, uint64_t desired);
  [[nodiscard]] std::optional<VramAtomicLoadResult> atomic_load(uint64_t offset, uint32_t width);
  [[nodiscard]] bool atomic_store(uint64_t offset, uint32_t width, uint64_t value);

private:
  uint64_t bytes_ = 0;
  uint64_t aperture_bytes_ = 0;
  int fd_ = -1;
  std::byte *mapping_ = nullptr;
  mutable std::mutex access_mutex_;
};

} // namespace rocjitsu
