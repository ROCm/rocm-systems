// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file vfio_user_client.h
/// @brief A minimal vfio-user client, for driving the transport in tests.
///
/// @details The transport is the part of the PCI front end that a unit test
/// cannot reach by calling a device directly: everything interesting about it
/// happens in response to protocol messages from a VMM. Booting a guest to
/// produce those messages takes a minute and cannot easily provoke the cases
/// worth testing, such as a client that shares memory the transport must
/// decline, so this client speaks the protocol directly.
///
/// It implements only the framing and the handful of messages the tests need,
/// and deliberately uses nothing but the public wire definitions: a test that
/// depended on the library's internal transport helpers would break when they
/// change, which is exactly the sort of coupling tests should not add.

#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace rocjitsu::test {

/// @brief What a shared memory window lets the device do.
/// @details Translated to the protocol's DMA region flags, so a test can
/// advertise exactly one direction and check that the transport enforces it.
enum class DmaProtection { ReadOnly, WriteOnly, ReadWrite };

/// @brief A client connection to a vfio-user server.
class VfioUserClient {
public:
  /// @brief Close the connection, if one is open.
  ~VfioUserClient();

  /// @brief Connect to @p socket_path and negotiate a protocol version.
  /// @param[in] socket_path Filesystem path of the server's AF_UNIX socket.
  /// @retval true The server accepted the connection and the version handshake.
  [[nodiscard]] bool connect(const std::string &socket_path);

  /// @brief Read the number of regions and interrupts the device advertises.
  /// @param[out] region_count Regions reported.
  /// @param[out] irq_count Interrupt types reported.
  [[nodiscard]] bool device_info(uint32_t &region_count, uint32_t &irq_count);

  /// @brief Ask how many vectors an interrupt index offers.
  /// @param[in] index VFIO interrupt index, e.g. VFIO_PCI_INTX_IRQ_INDEX.
  /// @param[out] count Vectors the server reports for it.
  /// @retval false The server refused the request.
  [[nodiscard]] bool irq_info(uint32_t index, uint32_t &count);

  /// @brief Arm one vector of @p index to signal @p fd.
  /// @details The descriptor travels with the message, which is the only way
  /// the server can signal a client it shares no memory with.
  /// @param[in] index VFIO interrupt index, e.g. VFIO_PCI_MSIX_IRQ_INDEX.
  /// @param[in] fd Eventfd the server should write to when it triggers.
  /// @retval false The server refused the request.
  [[nodiscard]] bool arm_irq(uint32_t index, int fd);

  /// @brief Read the size and flags of one region.
  /// @param[in] region Region index.
  /// @param[out] size Region size in bytes.
  /// @param[out] flags Region flags, using the VFIO_REGION_INFO_FLAG_* values.
  [[nodiscard]] bool region_info(uint32_t region, uint64_t &size, uint32_t &flags);

  /// @brief Read from a device region.
  /// @param[in] region Region index; the configuration space has its own index.
  /// @param[in] offset Byte offset within the region.
  /// @param[out] into Buffer to fill; its size is the access width.
  /// @retval false The server refused the access.
  [[nodiscard]] bool region_read(uint32_t region, uint64_t offset, std::span<std::byte> into);

  /// @brief Write to a device region.
  /// @param[in] region Region index.
  /// @param[in] offset Byte offset within the region.
  /// @param[in] from Bytes to write; their size is the access width.
  /// @retval false The server refused the access.
  [[nodiscard]] bool region_write(uint32_t region, uint64_t offset,
                                  std::span<const std::byte> from);

  /// @brief Share a memory window with the device.
  /// @param[in] iova Guest-physical base address to advertise.
  /// @param[in] size Window length in bytes.
  /// @param[in] fd Descriptor backing the window, or negative to share it
  ///               without one. The request still succeeds -- the protocol has
  ///               no way to refuse it -- but the transport keeps such a window
  ///               away from the device, so a true return does not mean the
  ///               device can reach it.
  /// @param[in] fd_offset Offset into @p fd of the window base.
  /// @param[in] protection Which directions the window permits; read/write by
  ///                       default, matching what every earlier caller asked for.
  [[nodiscard]] bool dma_map(uint64_t iova, uint64_t size, int fd, uint64_t fd_offset,
                             DmaProtection protection = DmaProtection::ReadWrite);

  /// @brief Withdraw a previously shared window.
  /// @param[in] iova Guest-physical base address of the window.
  /// @param[in] size Window length in bytes.
  [[nodiscard]] bool dma_unmap(uint64_t iova, uint64_t size);

private:
  [[nodiscard]] bool send(uint16_t command, std::span<const std::byte> payload, int fd);
  [[nodiscard]] bool receive(std::vector<std::byte> &payload);
  [[nodiscard]] bool request(uint16_t command, std::span<const std::byte> payload, int fd,
                             std::vector<std::byte> &reply);

  /// @brief Close the socket if open and mark it closed. Idempotent.
  /// @details Failure paths call this so a retried connect does not leak one
  /// descriptor per attempt.
  void close_socket();

  int socket_ = -1;
  uint16_t next_message_id_ = 1;
};

} // namespace rocjitsu::test
