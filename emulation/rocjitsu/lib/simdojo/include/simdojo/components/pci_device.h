// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file pci_device.h
/// @brief Transport-agnostic PCI function model.
///
/// @details A @ref simdojo::PciDevice is the neutral object that a PCI-attached
/// front end backs. It knows nothing about libvfio-user, KVM, sockets, or any
/// specific VMM: it declares its bus shape (identity, BARs, interrupt count) and
/// reacts to guest-driven events (BAR access, DMA map/unmap, reset). A transport binds to it by
/// injecting an @ref simdojo::IrqSink and a @ref simdojo::DmaEngine, then translating its own wire
/// protocol into calls on the device's virtual hooks.
///
/// This inversion is what lets one device model be driven by more than one
/// transport, and lets device families other than GPUs implement the same
/// interface with no coupling to any one family.
///
/// A PCI function is a component of the simulated machine, so it is a @ref
/// simdojo::Component:
///
/// ```
/// class MyDevice : public simdojo::PciDevice { ... };
/// ```
///
/// The function is the bus face, not the machine behind it. A family that
/// already owns a component subtree -- a GPU is a @ref
/// simdojo::CompositeComponent with memory and per-die children -- keeps that
/// subtree and *refers* to it from its PCI function. Inheriting both would give
/// the result two @ref simdojo::Component bases, and the bus face has no need
/// to own what it speaks for.

#pragma once

#include "simdojo/sim/component.h"

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace simdojo {

/// @brief PCI configuration-space identity used to build the bus.
///
/// @details Populated by the device and read once by the transport to program
/// configuration space. The @ref cls, @ref subcls, and @ref prog_if triple
/// follows the PCI class code (for example 0x12/0x00/0x00 for a processing
/// accelerator, 0x02 for a network controller). A transport and an in-guest
/// driver dispatch on the class, so the device kind is carried here rather than
/// encoded in any type name.
struct PciId {
  uint16_t vendor = 0;        ///< PCI vendor ID.
  uint16_t device = 0;        ///< PCI device ID.
  uint16_t subsys_vendor = 0; ///< Subsystem vendor ID.
  uint16_t subsys = 0;        ///< Subsystem ID.
  uint8_t cls = 0;            ///< PCI base class code.
  uint8_t subcls = 0;         ///< PCI subclass code.
  uint8_t prog_if = 0;        ///< Programming interface byte.
  uint8_t revision = 0;       ///< PCI revision ID.
};

/// @brief A window within a BAR that the guest may map directly.
///
/// @details Real devices mix mappable and trapped memory inside one BAR: a
/// frame-buffer aperture is mapped for speed while a control window within the
/// same BAR must trap so the device sees every access. Each area names a byte
/// range of the BAR that is backed by @ref BarSpec::backing_fd; anything outside
/// every area traps to @ref PciDevice::bar_access.
struct MmapArea {
  uint64_t offset = 0; ///< Byte offset of the window within the BAR.
  uint64_t length = 0; ///< Window length in bytes.
};

/// @brief Neutral description of one Base Address Register.
///
/// @details The transport reads a device's @ref PciDevice::bars() and programs
/// each region. A BAR with a valid @ref backing_fd and at least one entry in
/// @ref mmap_areas may be mapped by the guest for those ranges; every other
/// access traps back to @ref PciDevice::bar_access.
///
/// No transport-specific type appears here, so one specification drives any
/// transport. @ref backing_fd is not an exception: sharing memory with a VMM
/// by descriptor is how every POSIX transport does it, not something vfio-user
/// introduced. The model does assume a POSIX host, and a transport on another
/// one would translate this handle rather than reinterpret the struct.
struct BarSpec {
  int index = 0;                    ///< BAR index, 0 through 5.
  uint64_t size = 0;                ///< Region size in bytes.
  bool mem = true;                  ///< True for a memory BAR, false for an I/O BAR.
  bool prefetch = false;            ///< Prefetchable hint.
  bool is_64bit = false;            ///< True if this BAR is the low half of a 64-bit pair.
  int backing_fd = -1;              ///< Backing file descriptor, or negative to always trap.
  uint64_t fd_offset = 0;           ///< Offset into @ref backing_fd of the region base.
  std::vector<MmapArea> mmap_areas; ///< Directly mappable windows; empty means always trap.
};

/// @brief What a device may do with a shared guest memory window.
/// @details Its own type rather than the host's PROT_* bits: those are
/// `<sys/mman.h>` values, and a model header should not make every reader of a
/// device carry mmap semantics to interpret a field. The transport translates.
enum class DmaAccess : uint32_t {
  None = 0,       ///< Neither readable nor writable by the device.
  Read = 1U << 0, ///< The device may read the window.
  Write = 1U << 1 ///< The device may write the window.
};

/// @brief Combine two access sets.
constexpr DmaAccess operator|(DmaAccess lhs, DmaAccess rhs) {
  return static_cast<DmaAccess>(static_cast<uint32_t>(lhs) | static_cast<uint32_t>(rhs));
}

/// @brief Whether every bit of @p wanted is present in @p have.
constexpr bool contains(DmaAccess have, DmaAccess wanted) {
  return (static_cast<uint32_t>(have) & static_cast<uint32_t>(wanted)) ==
         static_cast<uint32_t>(wanted);
}

/// @brief A guest memory window shared with the device.
///
/// @details Delivered to the device through @ref PciDevice::dma_map when the
/// guest, by way of the transport, makes a region accessible to the device, and
/// withdrawn through @ref PciDevice::dma_unmap. With no virtual IOMMU in the
/// guest the guest-physical address equals the I/O virtual address. A device
/// must reach the window through the injected @ref DmaEngine rather than
/// caching a host pointer, because the transport may withdraw the mapping at
/// any time.
struct DmaRegion {
  uint64_t guest_phys = 0;            ///< Guest-physical base address, equal to the IOVA.
  uint64_t length = 0;                ///< Window length in bytes.
  DmaAccess access = DmaAccess::None; ///< What the device may do with the window.
};

/// @brief Why a device is being reset.
///
/// @details A device does different work for each kind, so the transport
/// reports which one the guest or the bus asked for instead of collapsing them
/// into a single notification.
enum class ResetKind {
  FunctionLevel, ///< PCI function-level reset requested through configuration space.
  Bus,           ///< Bus or platform reset.
  LostConnection ///< The transport lost its peer, so the device must return to a quiet state.
};

/// @brief How a device signals the guest, if at all.
enum class InterruptKind {
  None,    ///< The device raises no interrupts and is advertised with none.
  IntxPin, ///< A single legacy interrupt pin.
  MsiX     ///< Message-signalled interrupts, using @ref InterruptSpec::vectors.
};

/// @brief The interrupt capability a device asks to be advertised with.
///
/// @details Kind and count are declared together because a count alone cannot
/// distinguish "no interrupts" from "one pin". A transport that guesses will
/// advertise an interrupt the device never raises, which a guest driver may then
/// wait on.
struct InterruptSpec {
  InterruptKind kind = InterruptKind::None; ///< What to advertise.
  uint32_t vectors = 0;                     ///< Vector count, for @ref InterruptKind::MsiX.
  /// @brief BAR holding the message table and its pending-bit array.
  ///
  /// @details Message-signalled interrupts are configured through a table in
  /// the device's own memory rather than through configuration space, so the
  /// device decides where that table lives and the transport only reports the
  /// decision. Meaningful only for @ref InterruptKind::MsiX.
  int table_bar = 0;
  uint64_t table_offset = 0;   ///< Byte offset of the table within that BAR.
  uint64_t pending_offset = 0; ///< Byte offset of the pending-bit array, likewise.
};

/// @brief PCI Express capabilities implemented by a device.
///
/// @details These are properties of the function, not of the transport carrying
/// its configuration space. In particular, a transport must not advertise
/// AtomicOp completion merely because it can forward ordinary DMA: software
/// uses these bits to decide whether the device may issue atomic requests.
struct PcieSpec {
  bool atomic_completer_32 = false; ///< Completes 32-bit PCIe AtomicOps.
  bool atomic_completer_64 = false; ///< Completes 64-bit PCIe AtomicOps.
};

/// @brief Sink through which a device raises interrupts toward the guest.
///
/// @details Implemented by the transport and injected into the device so the
/// device can signal completion without knowing how the interrupt reaches the
/// guest.
class IrqSink {
public:
  virtual ~IrqSink() = default;

  /// @brief Raise interrupt vector @p vector toward the guest.
  /// @param[in] vector Zero-based interrupt vector index.
  /// @retval true The interrupt was handed to the transport.
  /// @retval false Delivery failed, for example because no guest is attached.
  [[nodiscard]] virtual bool trigger(uint32_t vector) = 0;
};

/// @brief Result of one transport-mediated DMA access.
///
/// @details Kept transport-neutral so PCI devices can preserve why an access
/// failed without depending on a device-family VM result type.
enum class DmaAccessOutcome : uint8_t {
  Complete,    ///< The complete range was transferred.
  Unavailable, ///< No live peer or transport endpoint can service the request.
  Faulted,     ///< The peer rejected or did not map the requested range.
  Malformed,   ///< The request itself cannot describe a valid transfer.
};

/// @brief Result of one strong compare/exchange through a DMA transport.
struct DmaAtomicCompareExchangeResult {
  DmaAccessOutcome outcome = DmaAccessOutcome::Faulted;
  uint64_t observed = 0;
  bool exchanged = false;
};

/// @brief Result of one indivisible 4- or 8-byte DMA load.
struct DmaAtomicLoadResult {
  DmaAccessOutcome outcome = DmaAccessOutcome::Faulted;
  uint64_t value = 0;
};

/// @brief Engine through which a device reaches guest memory.
///
/// @details Implemented by the transport and injected into the device so the
/// device can read and write guest-physical memory, such as command buffers and
/// completion records, without depending on how the transport resolves and
/// copies it.
class DmaEngine {
public:
  virtual ~DmaEngine() = default;

  /// @brief Read guest memory at @p guest_phys into @p dst.
  /// @param[in] guest_phys Guest-physical source address.
  /// @param[out] dst Destination buffer; its size is the transfer length.
  /// @retval true The full range was read.
  /// @retval false The access failed or fell outside a mapped window.
  [[nodiscard]] virtual bool read(uint64_t guest_phys, std::span<std::byte> dst) = 0;

  /// @brief Write @p src to guest memory at @p guest_phys.
  /// @param[in] guest_phys Guest-physical destination address.
  /// @param[in] src Source bytes; their size is the transfer length.
  /// @retval true The full range was written AND is visible to the guest.
  /// @retval false The access failed or fell outside a mapped window.
  /// @details Writes are ORDERED AND COMPLETE on return. A true result means the
  /// bytes are guest-visible, not merely accepted for later transfer, and two
  /// writes that return in order become visible to the guest in that order.
  ///
  /// This is part of the contract rather than something a caller can arrange,
  /// because a caller cannot: publishing a ring entry and then its write pointer
  /// is only safe if the entry is already visible when the pointer write starts,
  /// and no fence a caller issues can order stores an implementation has merely
  /// queued. A driver reads the pointer and the entry with a barrier between
  /// them, so an implementation that buffers has to flush here to hold up its
  /// side -- an implementation that cannot must fail rather than return true.
  [[nodiscard]] virtual bool write(uint64_t guest_phys, std::span<const std::byte> src) = 0;

  /// @brief Typed form of @ref read used by revocable backing sessions.
  /// @details Existing transports retain their boolean API through the default
  /// implementation. Transports that can distinguish absence, address faults,
  /// and malformed requests override this method.
  [[nodiscard]] virtual DmaAccessOutcome read_outcome(uint64_t guest_phys,
                                                      std::span<std::byte> dst) {
    return read(guest_phys, dst) ? DmaAccessOutcome::Complete : DmaAccessOutcome::Faulted;
  }

  /// @brief Typed form of @ref write used by revocable backing sessions.
  [[nodiscard]] virtual DmaAccessOutcome write_outcome(uint64_t guest_phys,
                                                       std::span<const std::byte> src) {
    return write(guest_phys, src) ? DmaAccessOutcome::Complete : DmaAccessOutcome::Faulted;
  }

  /// @brief Perform one acquire load from naturally aligned guest memory.
  [[nodiscard]] virtual DmaAtomicLoadResult atomic_load(uint64_t guest_phys, uint32_t width) {
    (void)guest_phys;
    (void)width;
    return {};
  }

  /// @brief Perform one release store to naturally aligned guest memory.
  [[nodiscard]] virtual DmaAccessOutcome atomic_store(uint64_t guest_phys, uint32_t width,
                                                      uint64_t value) {
    (void)guest_phys;
    (void)width;
    (void)value;
    return DmaAccessOutcome::Faulted;
  }

  /// @brief Perform one non-spurious atomic compare/exchange in guest memory.
  /// @details The default rejects the request; synthesizing it from read/write
  /// would falsely claim atomicity. Only naturally aligned widths 4 and 8 are
  /// valid for implementations that support it.
  [[nodiscard]] virtual DmaAtomicCompareExchangeResult
  compare_exchange(uint64_t guest_phys, uint32_t width, uint64_t expected, uint64_t desired) {
    (void)guest_phys;
    (void)width;
    (void)expected;
    (void)desired;
    return {};
  }
};

/// @brief The two endpoints supplied by one PCI transport owner.
struct PciTransport {
  IrqSink *irq = nullptr;   ///< Where the device raises interrupts, or nullptr.
  DmaEngine *dma = nullptr; ///< How the device reaches guest memory, or nullptr.
};

/// @brief One generation of access to a transport's downstream peer.
///
/// @details A transport owner can serve several peers over its lifetime. Each
/// peer receives a distinct session so device state retained from an earlier
/// peer can never migrate to a later one. Revocation first refuses new leases,
/// then waits for admitted operations before endpoint storage may be destroyed.
class PciTransportSession final : public std::enable_shared_from_this<PciTransportSession> {
public:
  enum class State : uint8_t { Open, Closing, Revoked };

  class OperationLease {
  public:
    OperationLease() = default;
    OperationLease(const OperationLease &) = delete;
    OperationLease &operator=(const OperationLease &) = delete;

    OperationLease(OperationLease &&other) noexcept
        : session_(std::move(other.session_)), endpoints_(other.endpoints_) {
      other.endpoints_ = {};
    }

    OperationLease &operator=(OperationLease &&other) noexcept {
      if (this == &other)
        return *this;
      release();
      session_ = std::move(other.session_);
      endpoints_ = other.endpoints_;
      other.endpoints_ = {};
      return *this;
    }

    ~OperationLease() { release(); }

    [[nodiscard]] explicit operator bool() const { return session_ != nullptr; }
    [[nodiscard]] DmaEngine *dma() const { return endpoints_.dma; }
    [[nodiscard]] IrqSink *irq() const { return endpoints_.irq; }

    [[nodiscard]] DmaAccessOutcome read(uint64_t guest_phys, std::span<std::byte> dst) const {
      return endpoints_.dma != nullptr ? endpoints_.dma->read_outcome(guest_phys, dst)
                                       : DmaAccessOutcome::Unavailable;
    }

    [[nodiscard]] DmaAccessOutcome write(uint64_t guest_phys,
                                         std::span<const std::byte> src) const {
      return endpoints_.dma != nullptr ? endpoints_.dma->write_outcome(guest_phys, src)
                                       : DmaAccessOutcome::Unavailable;
    }

    [[nodiscard]] DmaAtomicCompareExchangeResult compare_exchange(uint64_t guest_phys,
                                                                  uint32_t width, uint64_t expected,
                                                                  uint64_t desired) const {
      return endpoints_.dma != nullptr
                 ? endpoints_.dma->compare_exchange(guest_phys, width, expected, desired)
                 : DmaAtomicCompareExchangeResult{.outcome = DmaAccessOutcome::Unavailable};
    }

    [[nodiscard]] DmaAtomicLoadResult atomic_load(uint64_t guest_phys, uint32_t width) const {
      return endpoints_.dma != nullptr
                 ? endpoints_.dma->atomic_load(guest_phys, width)
                 : DmaAtomicLoadResult{.outcome = DmaAccessOutcome::Unavailable};
    }

    [[nodiscard]] DmaAccessOutcome atomic_store(uint64_t guest_phys, uint32_t width,
                                                uint64_t value) const {
      return endpoints_.dma != nullptr ? endpoints_.dma->atomic_store(guest_phys, width, value)
                                       : DmaAccessOutcome::Unavailable;
    }

    [[nodiscard]] bool trigger(uint32_t vector) const {
      return endpoints_.irq != nullptr && endpoints_.irq->trigger(vector);
    }

  private:
    friend class PciTransportSession;
    OperationLease(std::shared_ptr<PciTransportSession> session, PciTransport endpoints)
        : session_(std::move(session)), endpoints_(endpoints) {}

    void release() {
      if (session_ == nullptr)
        return;
      session_->release_operation();
      session_.reset();
      endpoints_ = {};
    }

    std::shared_ptr<PciTransportSession> session_;
    PciTransport endpoints_;
  };

  [[nodiscard]] OperationLease acquire() {
    std::lock_guard lock(mutex_);
    if (state_ != State::Open)
      return {};
    ++active_operations_;
    return OperationLease(shared_from_this(), endpoints_);
  }

  /// @brief Refuse future leases without waiting for admitted operations.
  /// @retval true This call changed the state from open to closing.
  bool begin_revoke() {
    std::lock_guard lock(mutex_);
    if (state_ != State::Open)
      return false;
    state_ = State::Closing;
    return true;
  }

  /// @brief Wait for admitted operations and permanently clear the endpoints.
  void wait_until_drained() {
    std::unique_lock lock(mutex_);
    if (state_ == State::Open)
      state_ = State::Closing;
    drained_.wait(lock, [this]() { return active_operations_ == 0; });
    endpoints_ = {};
    state_ = State::Revoked;
  }

  [[nodiscard]] uint64_t generation() const { return generation_; }

  [[nodiscard]] State state() const {
    std::lock_guard lock(mutex_);
    return state_;
  }

private:
  friend class PciDevice;
  PciTransportSession(uint64_t generation, PciTransport endpoints)
      : generation_(generation), endpoints_(endpoints) {}

  void release_operation() {
    std::lock_guard lock(mutex_);
    if (--active_operations_ == 0)
      drained_.notify_all();
  }

  const uint64_t generation_;
  mutable std::mutex mutex_;
  std::condition_variable drained_;
  PciTransport endpoints_;
  State state_ = State::Open;
  uint64_t active_operations_ = 0;
};

/// @brief A simulated PCI function: a component with a transport-agnostic bus face.
///
/// @details A device declares its bus shape once through @ref pci_id, @ref bars,
/// and @ref interrupts. The transport reads those to build the bus, injects
/// its @ref IrqSink and @ref DmaEngine, and then delivers guest events to the
/// hooks below. Identity and placement in the machine come from @ref Component.
///
/// Every hook is invoked on the transport thread and must be serviced from
/// device-local state. Blocking a hook stalls the guest instruction that
/// triggered it, and drivers routinely poll a status register in a tight loop
/// while waiting for hardware, so a slow read is indistinguishable from broken
/// hardware.
class PciDevice : public Component {
public:
  /// @brief Construct a PCI function with the given name and identity.
  /// @param[in] name Human-readable name for this function.
  /// @param[in] id Identity to present in configuration space.
  PciDevice(std::string name, PciId id) : Component(std::move(name)), id_(id) {}
  ~PciDevice() override = default;

  /// @brief Return the PCI configuration-space identity.
  /// @details Fixed for the life of the function, so it is stored rather than
  /// asked for: identity is what the device *is*, not behavior it implements.
  [[nodiscard]] const PciId &pci_id() const { return id_; }

  /// @brief Return the BAR layout, one entry per populated BAR.
  [[nodiscard]] virtual std::vector<BarSpec> bars() const = 0;

  /// @brief Return the interrupt capability to advertise for this device.
  /// @returns The specification; the default advertises no interrupts.
  [[nodiscard]] virtual InterruptSpec interrupts() const { return {}; }

  /// @brief Return the PCI Express capabilities implemented by this device.
  /// @returns The specification; the default advertises no optional features.
  [[nodiscard]] virtual PcieSpec pcie() const { return {}; }

  /// @brief The two sinks a transport gives a device, as one object.
  ///
  /// @details Grouped rather than installed separately because they arrive from
  /// the same transport, share its lifetime, and are used together: an interrupt
  /// is delivered by writing the ring through the engine and *then* raising the
  /// sink. Two independently-published pointers make a half-attached device
  /// representable, and a device with a thread of its own can observe that state.
  ///
  /// Owned by the transport and immutable once published, so a device that has
  /// taken a pointer to one holds a consistent pair for as long as it uses it.
  using Transport = PciTransport;

  /// @brief Claim this device for @p transport.
  ///
  /// @details A device serves one transport at a time. The claim is a single
  /// compare-and-exchange rather than a test followed by a store, because the
  /// two are not the same promise: with a gap between them, two transports both
  /// find the device free, both believe they own it, and both will later detach
  /// it. Publishing one pointer also means there is no moment at which a reader
  /// can see half a transport.
  ///
  /// Either sink inside @p transport may be null, which claims the device for a
  /// transport that cannot do that half of the job; the device declines the
  /// operations needing it rather than dereferencing null.
  ///
  /// @param[in] transport Transport-owned record; must outlive the attachment.
  /// @retval true The device is now attached to @p transport.
  /// @retval false Another transport holds it; this one was not installed.
  [[nodiscard]] bool attach_transport(const Transport *transport) {
    if (transport == nullptr)
      return false;
    std::lock_guard lock(transport_mutex_);
    if (transport_ != nullptr)
      return false;
    std::shared_ptr<PciTransportSession> session = make_transport_session_locked(*transport);
    transport_ = transport;
    transport_session_ = std::move(session);
    return true;
  }

  /// @brief Start a fresh downstream-peer generation for the owning transport.
  /// @details The previous generation must have been revoked first. This split
  /// lets a long-lived transport host serve sequential clients without letting
  /// state captured for one client reach the next.
  [[nodiscard]] std::shared_ptr<PciTransportSession>
  activate_transport_session(const Transport *transport) {
    std::lock_guard lock(transport_mutex_);
    if (transport_ != transport || transport_session_ != nullptr ||
        (closing_transport_session_ != nullptr &&
         closing_transport_session_->state() != PciTransportSession::State::Revoked))
      return {};
    closing_transport_session_.reset();
    transport_session_ = make_transport_session_locked(*transport);
    return transport_session_;
  }

  /// @brief Unpublish and begin revoking the current downstream-peer session.
  /// @returns The closing session for its owner to quiesce and drain, or null.
  [[nodiscard]] std::shared_ptr<PciTransportSession>
  revoke_transport_session(const Transport *transport) {
    std::lock_guard lock(transport_mutex_);
    if (transport_ != transport || transport_session_ == nullptr)
      return {};
    std::shared_ptr<PciTransportSession> closing = std::move(transport_session_);
    (void)closing->begin_revoke();
    closing_transport_session_ = closing;
    return closing;
  }

  /// @brief Release this device, if @p transport is what holds it.
  ///
  /// @details Takes the claimant rather than nothing, so releasing is checked
  /// the way claiming is. An unchecked release lets a transport that never won
  /// the device evict the one that did -- which is how a refused second
  /// transport turns into a broken first one.
  ///
  /// Stopping *later* calls is all this does. A device thread already inside
  /// @ref IrqSink::trigger or @ref DmaEngine::write is unaffected, so a device
  /// with threads of its own must be quiesced before its transport is destroyed;
  /// this is the second half of that, not a substitute for it.
  ///
  /// @param[in] transport The transport releasing its claim.
  /// @retval true The device was held by @p transport and is now free.
  /// @retval false Something else holds it; nothing was changed.
  bool detach_transport(const Transport *transport) {
    std::shared_ptr<PciTransportSession> closing;
    {
      std::lock_guard lock(transport_mutex_);
      if (transport_ != transport)
        return false;
      closing = take_transport_session_for_shutdown_locked();
    }
    if (closing != nullptr)
      closing->wait_until_drained();
    return true;
  }

  /// @brief Whether a transport currently holds this device.
  [[nodiscard]] bool transport_attached() const {
    std::lock_guard lock(transport_mutex_);
    return transport_ != nullptr;
  }

  /// @brief Generation of the currently active downstream peer, or zero.
  [[nodiscard]] uint64_t transport_session_generation() const {
    std::lock_guard lock(transport_mutex_);
    return transport_session_ != nullptr ? transport_session_->generation() : 0;
  }

  /// @brief Service a guest read or write to a BAR.
  /// @param[in] bar BAR index the access targets.
  /// @param[in,out] buf On a write, the bytes the guest stored; on a read, the
  ///                    buffer to fill. Its size is the access width.
  /// @param[in] offset Byte offset within the BAR.
  /// @param[in] write True for a write access, false for a read.
  /// @returns The number of bytes serviced, or a negative value if the access
  ///          was rejected, for example for an unsupported width or alignment.
  [[nodiscard]] virtual int64_t bar_access(int bar, std::span<std::byte> buf, uint64_t offset,
                                           bool write) = 0;

  /// @brief Register a guest memory window as reachable by the device.
  /// @param[in] region The window the guest mapped.
  virtual void dma_map(const DmaRegion &region) = 0;

  /// @brief Withdraw a previously mapped guest memory window.
  /// @param[in] region The window being unmapped.
  /// @details After this returns, the device must not issue further @ref
  /// DmaEngine access to the range.
  virtual void dma_unmap(const DmaRegion &region) = 0;

  /// @brief Return the device to its power-on state.
  /// @param[in] kind Why the reset was requested.
  virtual void reset(ResetKind /*kind*/) {}

protected:
  /// @brief Revoke the current transport while a derived device is still alive.
  ///
  /// @details Retained device state may hold a @ref PciTransportSession and a
  /// non-owning pointer back into the derived device. A derived destructor must
  /// call this before destroying that state. Revocation refuses new leases and
  /// waits for every admitted operation, so a retained session cannot reach
  /// either the transport endpoints or the derived device after this returns.
  void shutdown_transport() {
    std::shared_ptr<PciTransportSession> closing;
    {
      std::lock_guard lock(transport_mutex_);
      closing = take_transport_session_for_shutdown_locked();
    }
    if (closing != nullptr)
      closing->wait_until_drained();
  }

  /// @brief Acquire the current peer generation for one complete operation.
  [[nodiscard]] PciTransportSession::OperationLease acquire_transport() const {
    std::shared_ptr<PciTransportSession> session;
    {
      std::lock_guard lock(transport_mutex_);
      session = transport_session_;
    }
    return session != nullptr ? session->acquire() : PciTransportSession::OperationLease{};
  }

  /// @brief Capture the exact current peer generation for retained device state.
  [[nodiscard]] std::shared_ptr<PciTransportSession> transport_session() const {
    std::lock_guard lock(transport_mutex_);
    return transport_session_;
  }

private:
  [[nodiscard]] std::shared_ptr<PciTransportSession> take_transport_session_for_shutdown_locked() {
    if (transport_session_ != nullptr)
      (void)transport_session_->begin_revoke();
    transport_ = nullptr;
    std::shared_ptr<PciTransportSession> closing = transport_session_ != nullptr
                                                       ? std::move(transport_session_)
                                                       : std::move(closing_transport_session_);
    closing_transport_session_.reset();
    return closing;
  }

  [[nodiscard]] std::shared_ptr<PciTransportSession>
  make_transport_session_locked(const Transport &transport) {
    uint64_t generation = next_transport_generation_++;
    if (generation == 0) {
      generation = next_transport_generation_++;
    }
    return std::shared_ptr<PciTransportSession>(new PciTransportSession(generation, transport));
  }

  PciId id_;
  mutable std::mutex transport_mutex_;
  const Transport *transport_ = nullptr;
  std::shared_ptr<PciTransportSession> transport_session_;
  std::shared_ptr<PciTransportSession> closing_transport_session_;
  uint64_t next_transport_generation_ = 1;
};

} // namespace simdojo
