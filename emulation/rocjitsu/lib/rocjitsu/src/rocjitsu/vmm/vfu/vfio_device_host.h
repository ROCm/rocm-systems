// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file vfio_device_host.h
/// @brief Binds a @ref simdojo::PciDevice to a VMM over the vfio-user protocol.
///
/// @details This is the only part of rocjitsu that knows libvfio-user exists. It
/// reads the bus shape a device declares, programs it into a libvfio-user
/// context, serves the socket a VMM connects to, and turns each protocol message
/// into a call on the device. In the other direction it implements the two sinks
/// a device needs, so the device can raise interrupts and reach guest memory
/// while remaining transport-agnostic.
///
/// libvfio-user is explicitly not thread safe, so every entry point into it is
/// serialized here. That matters because interrupts and guest-memory access
/// originate from whichever thread the device runs work on, while protocol
/// messages are served from the thread that called @ref
/// rocjitsu::VfioDeviceHost::run.
///
/// The serialization is reentrant on purpose. Servicing a protocol message runs
/// device hooks on the serving thread, and a device answering a doorbell write
/// by reading a command buffer and raising a completion is the ordinary case,
/// not an abuse: the library documents scatter-gather access from within a
/// device callback. What must be prevented is two *threads* in the library at
/// once, so the lock excludes other threads while allowing the serving thread to
/// re-enter.
///
/// A device can only reach guest memory the client shared mmap-ably. A window
/// that must instead be fetched by sending the client a command is refused,
/// because unless the client negotiated a second socket for those, the command
/// travels on the same connection it may itself be sending on, which the library
/// documents as known to break; whether that socket was negotiated is not
/// exposed, and the peer sends when it likes, so the collision is not confined
/// to any one thread or moment. A window the client shared as a descriptor to be
/// read with file I/O would in fact be safe, but the public API gives no way to
/// tell it apart from one needing protocol traffic, so it is declined too. In
/// practice a VMM shares guest RAM mmap-ably and the supported case is the
/// ordinary one.
///
/// A device that shares a BAR by descriptor is served to one client only. The
/// mapping outlives the connection and cannot be taken back, so a later client
/// would inherit whatever the first left behind, and the first could keep
/// reading and writing it.
///
/// Teardown has a required order, because the device holds this host through
/// raw sink pointers: stop the serving thread, join it, then destroy the host.
/// @ref rocjitsu::VfioDeviceHost::detach clears the sinks so a device that
/// outlives its transport cannot call into freed state, but it does not wait for
/// a device's own worker threads; a device with workers must quiesce them
/// before its transport is destroyed.

#pragma once

#include "rocjitsu/vmm/vfu/guest_region_map.h"
#include "simdojo/components/pci_device.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <mutex>
#include <optional>
#include <span>
#include <stop_token>
#include <string>
#include <vector>

struct vfu_ctx;
struct dma_sg;

namespace rocjitsu {

/// @brief Serves one @ref simdojo::PciDevice to a VMM over a vfio-user socket.
class VfioDeviceHost final : public simdojo::IrqSink, public simdojo::DmaEngine {
public:
  /// @brief Construct a host for @p device listening on @p socket_path.
  /// @param[in] socket_path Filesystem path of the AF_UNIX socket to serve.
  /// @param[in] device Device to expose; must outlive this host.
  VfioDeviceHost(std::string socket_path, simdojo::PciDevice &device);
  ~VfioDeviceHost() override;

  /// @brief Build the libvfio-user context from the device's declared bus shape.
  /// @retval true The device is realized and the socket is ready for a client.
  /// @retval false Setup failed; the reason has been logged.
  [[nodiscard]] bool build();

  /// @brief Why serving ended.
  enum class ServeResult {
    Stopped, ///< An orderly end: the stop token was signalled, or the sole
             ///< client of a descriptor-backed device disconnected.
    Failed   ///< The transport broke; the reason has been logged.
  };

  /// @brief Serve the socket until @p stop_token is signalled or serving fails.
  /// @param[in] stop_token Cooperative stop for the serving thread.
  /// @returns Why serving ended.
  /// @details Blocks, and accepts one client at a time. What a disconnect means
  /// depends on the device. One whose BARs all trap can be served again, so this
  /// returns to waiting and a VMM may be restarted against a running server. One
  /// that shares a BAR by descriptor cannot: the mapping outlives the connection
  /// and cannot be reclaimed, so serving ends and this returns @ref
  /// ServeResult::Stopped.
  ///
  /// A caller must observe the result either way: once this returns, the socket
  /// is no longer served, and a process that keeps running is advertising a
  /// device nobody is behind.
  [[nodiscard]] ServeResult run(std::stop_token stop_token);

  /// @brief Detach this host from the device.
  /// @details Clears the sinks the device holds, so a device that outlives its
  /// transport cannot call into freed state. Called from the destructor, and
  /// callable earlier during an orderly shutdown.
  void detach();

  /// @brief Device this host serves, for the protocol callbacks to dispatch to.
  [[nodiscard]] simdojo::PciDevice &device() { return device_; }

  /// @brief Ask for @p work to run on the serving thread, one request at a time.
  ///
  /// @details The device is otherwise touched only by protocol callbacks, which
  /// all arrive on the serving thread. A thread that wants to make the device
  /// act on its own -- to raise an interrupt for something the guest did not
  /// ask for -- hands the work here rather than reaching into the device, so
  /// that invariant holds rather than being widened to a second thread.
  ///
  /// Deliberately not "run it under the transport's lock". That lock is held
  /// across protocol dispatch, which can block reading from a stalled client,
  /// so waiting for it would let one guest make the caller unresponsive -- and
  /// the caller here is the thread that handles shutdown signals.
  ///
  /// This is deliberately NOT a general work queue, and should not become one
  /// without the semantics a producer of completions or faults would need. What
  /// it promises is only what the one-shot request it exists for needs:
  /// - **At most one request is outstanding**, where outstanding means accepted
  ///   but not yet finished -- not merely not yet picked up. A second while one
  ///   is outstanding is REFUSED, so a caller is told rather than having its work
  ///   silently dropped.
  /// - **Latency is bounded by one transport poll**, because the serving thread
  ///   drains before it waits. It is not immediate, and there is no wake-up: a
  ///   caller needing promptness needs a different mechanism.
  /// - **@p work must not throw.** It runs on the serving thread, so an escaping
  ///   exception would terminate the process; this catches and reports instead,
  ///   which keeps a faulty request from taking the device down but does NOT
  ///   make throwing a supported way to report failure.
  /// - **It may never run.** If serving stops first the request is discarded.
  ///
  /// @param[in] work What the serving thread should do.
  /// @retval true The request was accepted and will run, unless serving stops.
  /// @retval false A request is already outstanding; @p work was not taken.
  [[nodiscard]] bool ask_serving_thread(std::function<void()> work);

  /// @brief Record a guest memory window the client registered.
  /// @param[in] region The window, as the transport reported it.
  /// @retval true The window was not already known, so the device should be told.
  /// @retval false An identical window was already recorded.
  /// @details The transport keeps its own list because a transfer crossing two
  /// registrations has to be split at their boundary, and the scatter-gather
  /// entries the library hands back do not expose where that boundary is. A
  /// client may re-register a window it already holds, which the library accepts
  /// and still reports, so insertion is idempotent to keep this list and the
  /// device's view of it from drifting apart.
  [[nodiscard]] bool record_guest_region(const simdojo::DmaRegion &region);

  /// @brief Report that a shared window is not one this transport can serve.
  /// @param[in] region The window being declined.
  /// @details Logged once. A guest whose IOMMU shares memory page by page would
  /// otherwise produce one message per page.
  void note_unreachable_region(const simdojo::DmaRegion &region);

  /// @brief Forget a guest memory window that the client withdrew.
  /// @param[in] region The window being withdrawn.
  void forget_guest_region(const simdojo::DmaRegion &region);

  [[nodiscard]] bool trigger(uint32_t vector) override;
  [[nodiscard]] bool read(uint64_t guest_phys, std::span<std::byte> dst) override;
  [[nodiscard]] bool write(uint64_t guest_phys, std::span<const std::byte> src) override;

private:
  bool copy_guest_memory(uint64_t guest_phys, void *data, std::size_t length, bool to_guest);
  bool copy_one_segment(uint64_t guest_phys, void *data, std::size_t length, bool to_guest);
  bool copy_by_region(uint64_t guest_phys, void *data, std::size_t length, bool to_guest);
  bool transfer_one_sg(dma_sg *sg, void *data, uint64_t guest_phys, std::size_t length,
                       bool to_guest);
  [[nodiscard]] std::size_t bytes_until_region_end(uint64_t guest_phys) const;

  const std::string socket_path_;
  simdojo::PciDevice &device_;

  std::recursive_mutex vfu_mutex_;

  /// @brief The single outstanding request, and the lock guarding it.
  ///
  /// @details Its own lock rather than @ref vfu_mutex_, so handing work over
  /// never waits on protocol dispatch. One slot rather than a queue: see
  /// ask_serving_thread() for why this is not a general work queue.
  std::mutex asked_mutex_;
  std::optional<std::function<void()>> asked_;
  /// @brief Set while the serving thread is running a request it already took.
  bool asked_running_ = false;
  vfu_ctx *ctx_ = nullptr;
  bool attached_ = false;

  /// @brief Whether serving ends when the client disconnects.
  ///
  /// @details Set when the device shares a BAR by descriptor. Closing the socket
  /// cannot revoke a mapping the client already holds, so a second client could
  /// neither be isolated from the first nor given clean state; refusing to serve
  /// one is honest, where quietly reattaching is not.
  bool single_client_ = false;
  GuestRegionMap guest_regions_;
  bool warned_unreachable_region_ = false;
  uint64_t declined_regions_ = 0;
  uint64_t declined_bytes_ = 0;
};

} // namespace rocjitsu
