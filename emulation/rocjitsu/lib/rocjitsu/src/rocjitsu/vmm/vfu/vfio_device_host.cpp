// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vmm/vfu/vfio_device_host.h"

#include "rocjitsu/vm/amdgpu/pci/gpu_pci_device.h"
#include "rocjitsu/vm/fabric_endpoint.h"
#include "rocjitsu/vmm/vfu/bus_plan.h"
#include "rocjitsu/vmm/vfu/vfio_target_grants.h"

#include "util/log.h"

// libvfio-user's headers pull in <stdint.h>, <sys/uio.h> and others before
// opening their own linkage block, so those are included first here, outside the
// wrapper, rather than being dragged into C linkage by it. The C++ spelling is
// used where there is one; the rest are POSIX interfaces with no C++ header.
//
// The wrapper is then only around libvfio-user's own declarations. v0.8 guards
// them itself, which makes this nest harmlessly, but releases before it do not,
// and a consumer building against one of those otherwise has to work around the
// name mangling from outside the project.
#include <cstdint>

#include <sys/queue.h>
#include <sys/uio.h>
#include <syslog.h>
#include <unistd.h>

extern "C" {
#include <libvfio-user.h>
#include <pci_caps/msix.h>
#include <pci_caps/px.h>
}

#include <endian.h>
#include <linux/pci_regs.h>
#include <poll.h>
#include <sys/mman.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <cstring>
#include <format>
#include <limits>
#include <memory>
#include <new>
#include <optional>
#include <utility>
#include <vector>

namespace rocjitsu {
namespace {

/// @brief Longest a serving thread waits for socket activity before rechecking
/// its stop token.
///
/// @details Two things wait on it, not one: shutdown, and any work handed over
/// by ask_serving_thread(), which the serving thread only picks up when this
/// poll returns. Raising it lengthens both, so a request that looks instant
/// from the caller can sit here for the whole interval.
constexpr int kPollTimeoutMs = 100;

/// @brief Bytes one message-table entry occupies: two address words, a data
/// word and a vector-control word.
constexpr uint64_t kMsixEntryBytes = 16;

/// @brief Bytes one word of the pending-bit array occupies; the array is words
/// of 64 bits rather than a byte per vector.
constexpr uint64_t kMsixPendingWordBytes = 8;

/// @brief Scatter-gather entries a transfer is attempted with before the
/// library is asked how many it actually needs.
constexpr std::size_t kInitialSgEntries = 8;

class TerminalPeerTransfer final : public amdgpu::VmDeferredTransfer {
public:
  explicit TerminalPeerTransfer(amdgpu::VmAccessOutcome outcome) : outcome_(outcome) {}
  void submit() override {}
  [[nodiscard]] amdgpu::VmAccessOutcome poll(std::span<std::byte>) override { return outcome_; }
  [[nodiscard]] amdgpu::VmAccessOutcome reconcile_revocation(std::span<std::byte>) override {
    return outcome_;
  }
  void cancel() noexcept override {}

private:
  amdgpu::VmAccessOutcome outcome_;
};

std::optional<amdgpu::VmTransferToken> terminal_peer_token(std::size_t length,
                                                           amdgpu::VmAccessOutcome outcome) {
  return amdgpu::VmTransferToken(std::make_unique<TerminalPeerTransfer>(outcome), length);
}

// Process peer PTEs select the CU cache policy. Until the translated PCI path
// can retain and invalidate those cache tags through the guest's hub/TLB
// invalidation, silently treating a cacheable PTE as uncached would acknowledge
// a memory operation with the wrong visibility semantics. VMID 0 is the GART
// path used by MES and SDMA, which does not issue CU cache fills.
bool unsupported_cacheable_peer_pte(const amdgpu::GpuVmAccess &access,
                                    const amdgpu::VmTranslation &translation) {
  return access.info().vmid != 0 && translation.mtype != amdgpu::Mtype::UC;
}

class PeerBarBacking final : public FabricBacking::Owner {
public:
  explicit PeerBarBacking(VfioTargetGrants::PeerSnapshot snapshot)
      : offset_(snapshot.backing_offset), backing_id_(snapshot.backing_id),
        length_(snapshot.length), bytes_(std::move(snapshot.mapped_bytes)) {}

  [[nodiscard]] uint64_t base_offset() const noexcept override { return offset_; }
  [[nodiscard]] std::size_t extent() const noexcept override { return length_; }
  [[nodiscard]] uint64_t backing_id() const noexcept override { return backing_id_; }
  [[nodiscard]] bool read(uint64_t offset, std::span<std::byte> result) noexcept override {
    if (offset != offset_ || result.size() != length_ || bytes_ == nullptr)
      return false;
    std::memmove(result.data(), bytes_.get(), length_);
    return true;
  }
  [[nodiscard]] bool write(uint64_t offset, std::span<const std::byte> source) noexcept override {
    if (offset != offset_ || source.size() != length_ || bytes_ == nullptr)
      return false;
    std::memmove(bytes_.get(), source.data(), length_);
    return true;
  }

private:
  uint64_t offset_;
  uint64_t backing_id_;
  std::size_t length_;
  std::shared_ptr<std::byte> bytes_;
};

/// Hold every mmap-backed SG entry until a mixed target/ordinary transfer has
/// validated its complete span. Releasing the entries marks writes dirty.
struct PreparedMappedSpan {
  vfu_ctx_t *ctx = nullptr;
  std::size_t segment_count = 0;
  std::unique_ptr<std::byte[]> sgl_storage;
  std::unique_ptr<iovec[]> segments;

  ~PreparedMappedSpan() {
    if (segments != nullptr)
      vfu_sgl_put(ctx, reinterpret_cast<dma_sg_t *>(sgl_storage.get()), segments.get(),
                  segment_count);
  }

  void copy(void *data, bool to_guest) const {
    auto *cursor = static_cast<std::byte *>(data);
    for (std::size_t index = 0; index < segment_count; ++index) {
      const iovec &segment = segments[index];
      if (to_guest)
        std::memcpy(segment.iov_base, cursor, segment.iov_len);
      else
        std::memcpy(cursor, segment.iov_base, segment.iov_len);
      cursor += segment.iov_len;
    }
  }
};

std::unique_ptr<PreparedMappedSpan> prepare_mapped_span(vfu_ctx_t *ctx, uint64_t guest_phys,
                                                        std::size_t length, bool to_guest) {
  const std::size_t sg_entry_size = dma_sg_size();
  if (sg_entry_size == 0 ||
      sg_entry_size > std::numeric_limits<std::size_t>::max() / kInitialSgEntries)
    return nullptr;
  auto prepared = std::unique_ptr<PreparedMappedSpan>(new (std::nothrow) PreparedMappedSpan{});
  if (prepared == nullptr)
    return nullptr;
  prepared->ctx = ctx;
  std::size_t capacity = kInitialSgEntries;
  prepared->sgl_storage.reset(new (std::nothrow) std::byte[sg_entry_size * capacity]{});
  if (prepared->sgl_storage == nullptr)
    return nullptr;

  int nr_sgs = vfu_addr_to_sgl(ctx, reinterpret_cast<vfu_dma_addr_t>(guest_phys), length,
                               reinterpret_cast<dma_sg_t *>(prepared->sgl_storage.get()), capacity,
                               to_guest ? PROT_WRITE : PROT_READ);
  if (nr_sgs < 0) {
    // The library reports an insufficient SG array as -(needed) - 1.
    const std::size_t needed = static_cast<std::size_t>(-nr_sgs - 1);
    if (needed <= capacity || needed > LIBVFIO_USER_MAX_DMA_REGIONS ||
        needed > std::numeric_limits<std::size_t>::max() / sg_entry_size)
      return nullptr;
    capacity = needed;
    prepared->sgl_storage.reset(new (std::nothrow) std::byte[sg_entry_size * capacity]{});
    if (prepared->sgl_storage == nullptr)
      return nullptr;
    nr_sgs = vfu_addr_to_sgl(ctx, reinterpret_cast<vfu_dma_addr_t>(guest_phys), length,
                             reinterpret_cast<dma_sg_t *>(prepared->sgl_storage.get()), capacity,
                             to_guest ? PROT_WRITE : PROT_READ);
  }
  if (nr_sgs <= 0)
    return nullptr;
  prepared->segment_count = static_cast<std::size_t>(nr_sgs);
  for (std::size_t index = 0; index < prepared->segment_count; ++index) {
    auto *segment =
        reinterpret_cast<dma_sg_t *>(prepared->sgl_storage.get() + index * sg_entry_size);
    if (!vfu_sg_is_mappable(ctx, segment)) {
      util::Logger::warn(std::format(
          "vfu: refusing to reach {:#x}+{} because it is outside this transport's mmap-only policy",
          guest_phys, length));
      return nullptr;
    }
  }

  prepared->segments.reset(new (std::nothrow) iovec[prepared->segment_count]{});
  if (prepared->segments == nullptr)
    return nullptr;
  auto *sgl = reinterpret_cast<dma_sg_t *>(prepared->sgl_storage.get());
  if (vfu_sgl_get(ctx, sgl, prepared->segments.get(), prepared->segment_count, 0) < 0) {
    prepared->segments.reset();
    return nullptr;
  }
  std::size_t mapped_length = 0;
  for (std::size_t index = 0; index < prepared->segment_count; ++index) {
    const iovec &segment = prepared->segments[index];
    if (segment.iov_base == nullptr || segment.iov_len == 0 ||
        segment.iov_len > length - mapped_length)
      return nullptr;
    mapped_length += segment.iov_len;
  }
  if (mapped_length != length)
    return nullptr;
  return prepared;
}

VfioDeviceHost &host_of(vfu_ctx_t *ctx) {
  return *static_cast<VfioDeviceHost *>(vfu_get_private(ctx));
}

simdojo::PciDevice &device_of(vfu_ctx_t *ctx) { return host_of(ctx).device(); }

void forward_library_log(vfu_ctx_t * /*ctx*/, int level, const char *message) {
  if (level <= LOG_WARNING) {
    util::Logger::warn(std::format("vfu: {}", message));
  }
}

// libvfio-user identifies a region only by the callback it was registered with,
// so each BAR needs its own function to recover which BAR was accessed.
//
// The library turns a negative return into a protocol error using errno, and a
// device only reports that it rejected the access, so errno is set here rather
// than left holding whatever the last unrelated call put there. C++ exceptions
// must not unwind through the library's C frames either.
template <int Bar>
ssize_t bar_trampoline(vfu_ctx_t *ctx, char *buf, std::size_t count, loff_t offset,
                       const bool is_write) try {
  const std::span bytes(reinterpret_cast<std::byte *>(buf), count);
  const int64_t serviced =
      device_of(ctx).bar_access(Bar, bytes, static_cast<uint64_t>(offset), is_write);
  if (serviced < 0) {
    errno = EINVAL;
    return -1;
  }
  return static_cast<ssize_t>(serviced);
} catch (...) {
  errno = EIO;
  return -1;
}

constexpr std::array<vfu_region_access_cb_t *, 6> kBarTrampolines = {
    &bar_trampoline<0>, &bar_trampoline<1>, &bar_trampoline<2>,
    &bar_trampoline<3>, &bar_trampoline<4>, &bar_trampoline<5>,
};

/// @brief Whether a shared window is backed by a mapping in this process.
///
/// @details The library maps a window in only when the client shared it
/// mmap-ably, so a host address is exactly that signal for ordinary guest RAM.
/// A descriptor-backed window serviced by file I/O and one that must be
/// fetched over the protocol otherwise remain indistinguishable here.
bool is_mmap_backed(const vfu_dma_info_t &info) { return info.vaddr != nullptr; }

simdojo::DmaRegion to_dma_region(const vfu_dma_info_t &info) {
  return {
      .guest_phys = reinterpret_cast<uint64_t>(info.iova.iov_base),
      .length = info.iova.iov_len,
      .access =
          ((info.prot & PROT_READ) != 0 ? simdojo::DmaAccess::Read : simdojo::DmaAccess::None) |
          ((info.prot & PROT_WRITE) != 0 ? simdojo::DmaAccess::Write : simdojo::DmaAccess::None)};
}

void dma_register_trampoline(vfu_ctx_t *ctx, vfu_dma_info_t *info) try {
  const simdojo::DmaRegion region = to_dma_region(*info);
  // Telling the device about a window it could never read would leave it holding
  // a mapping every access fails against. The request itself still succeeds --
  // the protocol offers no way to refuse one window -- so the window is counted
  // and dropped here rather than handed on.
  if (!is_mmap_backed(*info)) {
    if (host_of(ctx).register_standard_peer_map(region))
      return;
    host_of(ctx).note_unreachable_region(region);
    return;
  }
  // A client may re-register a window it already holds; the library reports that
  // as a fresh mapping, so the device is told only about genuinely new ones.
  if (host_of(ctx).record_guest_region(region)) {
    device_of(ctx).dma_map(region);
  }
} catch (...) {
  util::Logger::warn("vfu: device threw while mapping a guest memory window");
}

void dma_unregister_trampoline(vfu_ctx_t *ctx, vfu_dma_info_t *info) try {
  if (!is_mmap_backed(*info)) {
    host_of(ctx).unregister_standard_peer_map(to_dma_region(*info));
    return;
  }
  const simdojo::DmaRegion region = to_dma_region(*info);
  // The library drops the window once this returns whatever the device does, so
  // the transport's own record is cleared even if the device hook throws.
  struct ForgetOnReturn {
    VfioDeviceHost &host;
    const simdojo::DmaRegion &region;
    ~ForgetOnReturn() { host.forget_guest_region(region); }
  } forget{host_of(ctx), region};
  device_of(ctx).dma_unmap(region);
} catch (...) {
  util::Logger::warn("vfu: device threw while unmapping a guest memory window");
}

int quiesce_trampoline(vfu_ctx_t *ctx) try {
  return host_of(ctx).begin_target_quiesce();
} catch (...) {
  errno = EIO;
  return -1;
}

simdojo::ResetKind to_reset_kind(vfu_reset_type_t type) {
  switch (type) {
  case VFU_RESET_PCI_FLR:
    return simdojo::ResetKind::FunctionLevel;
  case VFU_RESET_LOST_CONN:
    return simdojo::ResetKind::LostConnection;
  case VFU_RESET_DEVICE:
    break;
  }
  return simdojo::ResetKind::Bus;
}

int reset_trampoline(vfu_ctx_t *ctx, vfu_reset_type_t type) try {
  host_of(ctx).request_reset(to_reset_kind(type));
  return 0;
} catch (...) {
  errno = EIO;
  return -1;
}

} // namespace

VfioDeviceHost::VfioDeviceHost(std::string socket_path, simdojo::PciDevice &device)
    : socket_path_(std::move(socket_path)), device_(device) {
  quiesce_wake_ = std::make_shared<VfioQuiesceWake>();
  transport_ = {.irq = this, .dma = this};
  owns_device_ = device_.attach_transport(&transport_);
  // Claiming is a host-lifetime reservation. A guest-facing session begins only
  // after libvfio-user accepts a client, so state cannot accidentally bind to
  // this host before there is a peer to own it.
  if (owns_device_) {
    std::shared_ptr<simdojo::PciTransportSession> idle =
        device_.revoke_transport_session(&transport_);
    if (idle != nullptr)
      idle->wait_until_drained();
  }
  if (!owns_device_) {
    util::Logger::warn(std::format(
        "vfu: cannot serve {}: the device is already served by another transport", socket_path_));
  }
}

VfioDeviceHost::~VfioDeviceHost() {
  detach();
  // Serialized like every other entry into the library: destruction can invoke
  // device callbacks, and a stray transport call must not overlap it.
  const std::lock_guard lock(vfu_mutex_);
  if (ctx_ != nullptr) {
    vfu_destroy_ctx(ctx_);
    ctx_ = nullptr;
  }
}

void VfioDeviceHost::detach() {
  // Conditional on having attached in the first place. A host that lost the
  // device to another transport never installed itself, and detaching
  // unconditionally would take the winner's sinks away -- turning a refused
  // second transport into a broken first one.
  if (!owns_device_) {
    return;
  }
  std::shared_ptr<simdojo::PciTransportSession> closing;
  {
    const std::lock_guard lock(state_mutex_);
    attached_ = false;
    disconnect_in_progress_ = false;
    pending_reset_.reset();
    closing = device_.revoke_transport_session(&transport_);
    client_session_.reset();
    guest_regions_.clear();
  }
  if (closing != nullptr && target_grants_ != nullptr)
    target_grants_->revoke_session(target_ordinal_, closing->generation());
  if (closing != nullptr)
    (void)finish_session_reset(simdojo::ResetKind::LostConnection, std::move(closing));

  // Checked against this host's own record, so a host that never won the device
  // cannot release the one that did. detach_transport also drains a session if
  // a future caller reaches this path without the explicit revoke above.
  owns_device_ = !device_.detach_transport(&transport_);
}

VfioDeviceHost::ClientSnapshot VfioDeviceHost::client_snapshot() const {
  const std::lock_guard lock(state_mutex_);
  return {.attached = attached_,
          .disconnect_in_progress = disconnect_in_progress_,
          .completed_disconnects = completed_disconnects_};
}

bool VfioDeviceHost::enable_target_grants(VfioTargetGrants &grants, uint32_t ordinal,
                                          uint32_t gpu_id, FabricEndpoint *fabric) {
  const std::lock_guard lock(state_mutex_);
  if (ctx_ != nullptr || target_grants_ != nullptr || quiesce_wake_->fd() < 0 ||
      !grants.bind(ordinal, *this, gpu_id))
    return false;
  target_grants_ = &grants;
  target_ordinal_ = ordinal;
  target_gpu_id_ = gpu_id;
  fabric_ = fabric;
  return true;
}

std::optional<amdgpu::VmTransferToken>
VfioDeviceHost::prepare_peer_read(uint64_t requester_generation, uint64_t iova, std::size_t length,
                                  const std::shared_ptr<amdgpu::VmTransferEffectGate> &gate,
                                  const amdgpu::GpuVmAccess *executing_access, uint64_t gpu_va,
                                  const amdgpu::VmTranslation *selected_translation,
                                  const std::optional<amdgpu::MemoryPtEpoch> &pt_epoch) {
  if (target_grants_ == nullptr || length == 0)
    return std::nullopt;
  auto prepared =
      target_grants_->prepare_peer(target_ordinal_, requester_generation, iova, length, false);
  if (prepared.kind == VfioTargetGrants::PeerKind::OrdinaryRam)
    return std::nullopt;
  if (prepared.kind != VfioTargetGrants::PeerKind::Ready || fabric_ == nullptr)
    return terminal_peer_token(length, amdgpu::VmAccessOutcome::Faulted);
  if (executing_access == nullptr || selected_translation == nullptr || !pt_epoch || !*pt_epoch)
    return terminal_peer_token(length, amdgpu::VmAccessOutcome::Revoked);
  if (executing_access->retain_effect_gate() != gate)
    return terminal_peer_token(length, amdgpu::VmAccessOutcome::Revoked);
  if (unsupported_cacheable_peer_pte(*executing_access, *selected_translation))
    return terminal_peer_token(length, amdgpu::VmAccessOutcome::Faulted);
  auto snapshot = std::move(prepared.snapshot);
  const FabricEndpoint::TraceOrigin trace_origin{
      .map_id = snapshot.map_id, .gpu_va = gpu_va, .iova = iova};
  const uint32_t owner_gpu_id = snapshot.owner_gpu_id;
  FabricBacking backing(std::make_shared<PeerBarBacking>(snapshot), owner_gpu_id,
                        snapshot.backing_offset, snapshot.length);
  auto captured_admission = target_grants_->capture_admission(snapshot);
  const amdgpu::GpuVmAccess captured_access = *executing_access;
  const amdgpu::VmTranslation selected = *selected_translation;
  auto admission = [captured_admission = std::move(captured_admission), captured_access, selected,
                    gpu_va, iova,
                    length](const FabricBacking &, bool write, amdgpu::MemoryIngressPermit &) {
    if (write)
      return FabricEndpoint::EffectAuthority(
          {.outcome = amdgpu::VmAccessOutcome::Revoked, .permit = {}});
    const amdgpu::VmAccessOutcome outcome = captured_access.validate_deferred_system_span(
        gpu_va, iova, length, amdgpu::VmAccessKind::Read, selected);
    if (outcome != amdgpu::VmAccessOutcome::Complete)
      return FabricEndpoint::EffectAuthority({.outcome = outcome, .permit = {}});
    return FabricEndpoint::EffectAuthority(captured_admission());
  };
  FabricEndpoint::PreEffectAdmission admit_pt = [captured_access, expected = *pt_epoch] {
    return captured_access.admit_pt_rewalk(expected);
  };
  auto transfer = fabric_->prepare_read(target_gpu_id_, owner_gpu_id, std::move(backing), gate,
                                        std::move(admission), std::move(admit_pt), trace_origin);
  if (!transfer)
    return terminal_peer_token(length, transfer.outcome);
  return amdgpu::VmTransferToken(std::move(transfer.operation), length,
                                 amdgpu::VmDeferredWriteMaintenance::Caller,
                                 /*peer_origin=*/true);
}

std::optional<amdgpu::VmTransferToken>
VfioDeviceHost::prepare_peer_write(uint64_t requester_generation, uint64_t iova,
                                   std::span<const std::byte> bytes,
                                   const std::shared_ptr<amdgpu::VmTransferEffectGate> &gate,
                                   const amdgpu::GpuVmAccess *executing_access, uint64_t gpu_va,
                                   const amdgpu::VmTranslation *selected_translation,
                                   const std::optional<amdgpu::MemoryPtEpoch> &pt_epoch) {
  if (target_grants_ == nullptr || bytes.empty())
    return std::nullopt;
  auto prepared =
      target_grants_->prepare_peer(target_ordinal_, requester_generation, iova, bytes.size(), true);
  if (prepared.kind == VfioTargetGrants::PeerKind::OrdinaryRam)
    return std::nullopt;
  if (prepared.kind != VfioTargetGrants::PeerKind::Ready || fabric_ == nullptr)
    return terminal_peer_token(bytes.size(), amdgpu::VmAccessOutcome::Faulted);
  if (executing_access == nullptr || selected_translation == nullptr || !pt_epoch || !*pt_epoch)
    return terminal_peer_token(bytes.size(), amdgpu::VmAccessOutcome::Revoked);
  if (executing_access->retain_effect_gate() != gate)
    return terminal_peer_token(bytes.size(), amdgpu::VmAccessOutcome::Revoked);
  if (unsupported_cacheable_peer_pte(*executing_access, *selected_translation))
    return terminal_peer_token(bytes.size(), amdgpu::VmAccessOutcome::Faulted);
  auto snapshot = std::move(prepared.snapshot);
  const FabricEndpoint::TraceOrigin trace_origin{
      .map_id = snapshot.map_id, .gpu_va = gpu_va, .iova = iova};
  const uint32_t owner_gpu_id = snapshot.owner_gpu_id;
  FabricBacking backing(std::make_shared<PeerBarBacking>(snapshot), owner_gpu_id,
                        snapshot.backing_offset, snapshot.length);
  auto captured_admission = target_grants_->capture_admission(snapshot);
  const amdgpu::GpuVmAccess captured_access = *executing_access;
  const amdgpu::VmTranslation selected = *selected_translation;
  auto admission = [captured_admission = std::move(captured_admission), captured_access, selected,
                    gpu_va, iova, length = bytes.size()](const FabricBacking &, bool write,
                                                         amdgpu::MemoryIngressPermit &) {
    if (!write)
      return FabricEndpoint::EffectAuthority(
          {.outcome = amdgpu::VmAccessOutcome::Revoked, .permit = {}});
    const amdgpu::VmAccessOutcome outcome = captured_access.validate_deferred_system_span(
        gpu_va, iova, length, amdgpu::VmAccessKind::Write, selected);
    if (outcome != amdgpu::VmAccessOutcome::Complete)
      return FabricEndpoint::EffectAuthority({.outcome = outcome, .permit = {}});
    return FabricEndpoint::EffectAuthority(captured_admission());
  };
  FabricEndpoint::PreEffectAdmission admit_pt = [captured_access, expected = *pt_epoch] {
    return captured_access.admit_pt_rewalk(expected);
  };
  auto transfer =
      fabric_->prepare_write(target_gpu_id_, owner_gpu_id, std::move(backing), bytes, gate,
                             std::move(admission), std::move(admit_pt), trace_origin);
  if (!transfer)
    return terminal_peer_token(bytes.size(), transfer.outcome);
  return amdgpu::VmTransferToken(std::move(transfer.operation), bytes.size());
}

int VfioDeviceHost::begin_target_quiesce() {
  if (target_grants_ == nullptr)
    return 0;
  quiesce_gates_ = target_grants_->begin_quiesce(target_ordinal_);
  quiescing_ = true;
  quiesce_pending_ = !target_quiesce_drained();
  if (!quiesce_pending_)
    return 0;
  errno = EBUSY;
  return -1;
}

bool VfioDeviceHost::target_quiesce_drained() const noexcept {
  return VfioTargetGrants::drained(quiesce_gates_);
}

void VfioDeviceHost::end_target_quiesce() noexcept {
  if (!quiescing_)
    return;
  target_grants_->end_quiesce(target_ordinal_);
  quiesce_gates_.clear();
  quiesce_pending_ = false;
  quiescing_ = false;
}

std::shared_ptr<simdojo::PciTransportSession> VfioDeviceHost::target_session_snapshot() const {
  const std::lock_guard lock(state_mutex_);
  return attached_ ? client_session_ : nullptr;
}

bool VfioDeviceHost::register_standard_peer_map(const simdojo::DmaRegion &region) {
  return target_grants_ != nullptr &&
         target_grants_->register_standard_map(target_ordinal_, region).has_value();
}

void VfioDeviceHost::unregister_standard_peer_map(const simdojo::DmaRegion &region) {
  if (target_grants_ == nullptr)
    return;
  const auto session = target_session_snapshot();
  if (session != nullptr)
    target_grants_->unregister_standard_map(target_ordinal_, session->generation(),
                                            region.guest_phys, region.length);
}

void VfioDeviceHost::refresh_peer_bar_window() {
  if (target_grants_ == nullptr || ctx_ == nullptr)
    return;
  std::optional<simdojo::PciBarAddressDecoder::Window> window;
  const auto session = target_session_snapshot();
  // The reset callback closes this generation before its reply. Its grants
  // are drained after vfu_run_ctx releases the library lock, so do not retire
  // the BAR window here: publication can wait on an admitted peer effect.
  if (session != nullptr && session->state() != simdojo::PciTransportSession::State::Open)
    return;
  if (session != nullptr && session->state() == simdojo::PciTransportSession::State::Open) {
    const auto *config = reinterpret_cast<const std::byte *>(vfu_pci_get_config_space(ctx_));
    const auto read16 = [config](std::size_t offset) {
      uint16_t raw = 0;
      std::memcpy(&raw, config + offset, sizeof(raw));
      return le16toh(raw);
    };
    const auto read32 = [config](std::size_t offset) {
      uint32_t raw = 0;
      std::memcpy(&raw, config + offset, sizeof(raw));
      return le32toh(raw);
    };
    const uint32_t low = read32(PCI_BASE_ADDRESS_0);
    const bool memory_enabled = (read16(PCI_COMMAND) & PCI_COMMAND_MEMORY) != 0;
    for (const simdojo::BarSpec &bar : device_.bars()) {
      if (bar.index != 0 || !bar.mem || bar.backing_fd < 0 || bar.mmap_areas.empty() ||
          !memory_enabled || (low & PCI_BASE_ADDRESS_SPACE_IO) != 0)
        continue;
      const bool configured_64bit =
          (low & PCI_BASE_ADDRESS_MEM_TYPE_MASK) == PCI_BASE_ADDRESS_MEM_TYPE_64;
      if (configured_64bit != bar.is_64bit)
        continue;
      uint64_t base = low & PCI_BASE_ADDRESS_MEM_MASK;
      if (configured_64bit)
        base |= static_cast<uint64_t>(read32(PCI_BASE_ADDRESS_1)) << 32;
      window = simdojo::PciBarAddressDecoder::Window{.function = target_ordinal_,
                                                     .bar = 0,
                                                     .base = base,
                                                     .size = bar.size,
                                                     .rights = simdojo::DmaAccess::Read |
                                                               simdojo::DmaAccess::Write,
                                                     .generation = session->generation()};
      break;
    }
  }
  if (window == last_published_peer_bar_)
    return;
  target_grants_->publish_bar_window(target_ordinal_, window);
  last_published_peer_bar_ = window;
}

void VfioDeviceHost::request_reset(simdojo::ResetKind kind) {
  const std::lock_guard lock(state_mutex_);
  if (!pending_reset_.has_value() || kind == simdojo::ResetKind::LostConnection)
    pending_reset_ = kind;
  // A VFIO-user reset reply can be observed before the serving loop has left
  // vfu_run_ctx. Refuse fresh peer leases at the callback boundary; the loop
  // drains admitted effects and retires the generation outside its locks.
  if (client_session_ != nullptr)
    (void)client_session_->begin_revoke();
}

bool VfioDeviceHost::finish_session_reset(
    simdojo::ResetKind kind,
    std::shared_ptr<simdojo::PciTransportSession> closing_session) noexcept {
  // Never called with either host mutex held. Revocation prevents new leases;
  // drain the admitted operations before reset tears down the routes and
  // storage they can reach.
  try {
    if (closing_session != nullptr)
      closing_session->wait_until_drained();
  } catch (...) {
    util::Logger::warn("vfu: failed while draining a revoked PCI transport session");
    return false;
  }
  try {
    device_.reset(kind);
  } catch (const std::exception &error) {
    util::Logger::warn(std::format("vfu: deferred device reset failed: {}", error.what()));
    return false;
  } catch (...) {
    util::Logger::warn("vfu: deferred device reset failed");
    return false;
  }
  return true;
}

bool VfioDeviceHost::ask_serving_thread(std::function<void()> work) {
  // Refused here rather than thrown there. An empty target reaches the serving
  // thread as a bad_function_call, which is reported as work that failed --
  // indistinguishable from work that ran and could not be done -- and it
  // occupies the single slot while doing nothing at all.
  if (!work) {
    return false;
  }
  const std::lock_guard lock(asked_mutex_);
  // One slot, and a refusal rather than a drop. The serving thread can be parked
  // reading from a stalled client while requests keep arriving, so something has
  // to give; telling the caller lets it decide, where silently discarding the
  // 65th of a backlog only looked like it had a policy.
  // Outstanding means accepted but not yet FINISHED, not merely not yet picked
  // up. Freeing the slot the moment the serving thread takes the work would let
  // a second request be accepted while the first is still running, which is not
  // the one-at-a-time contract this promises.
  if (asked_.has_value() || asked_running_) {
    return false;
  }
  asked_ = std::move(work);
  return true;
}

bool VfioDeviceHost::build() {
  const std::lock_guard lock(vfu_mutex_);

  // A host that never won the device would serve a function it cannot raise an
  // interrupt through and cannot reach guest memory with. Refusing here is what
  // turns that into a reported failure rather than a device that answers reads
  // and never completes anything.
  if (!owns_device_) {
    return false;
  }

  // This host is the callback context: dispatching a protocol message needs both
  // the device and the transport's own record of what the client has mapped.
  ctx_ = vfu_create_ctx(VFU_TRANS_SOCK, socket_path_.c_str(), LIBVFIO_USER_FLAG_ATTACH_NB, this,
                        VFU_DEV_TYPE_PCI);
  if (ctx_ == nullptr) {
    util::Logger::warn(std::format("vfu: cannot serve {}: {}", socket_path_, std::strerror(errno)));
    return false;
  }
  vfu_setup_log(ctx_, &forward_library_log, LOG_WARNING);

  const simdojo::PciId id = device_.pci_id();
  if (vfu_pci_init(ctx_, VFU_PCI_TYPE_EXPRESS, PCI_HEADER_TYPE_NORMAL, id.revision) < 0) {
    util::Logger::warn(std::format("vfu: vfu_pci_init failed: {}", std::strerror(errno)));
    return false;
  }

  vfu_pci_set_id(ctx_, id.vendor, id.device, id.subsys_vendor, id.subsys);
  vfu_pci_set_class(ctx_, id.cls, id.subcls, id.prog_if);
  // The revision argument to vfu_pci_init is accepted and ignored by the pinned
  // library, so the byte is written directly or the guest sees revision zero.
  vfu_pci_get_config_space(ctx_)->hdr.rid = id.revision;

  // vfu_pci_init selects the size and kind of configuration space but does not
  // add the PCI Express capability. Without the capability Linux treats the
  // function as conventional PCI, cannot read Device Capabilities 2, and will
  // not enable AtomicOp requests even when the device implements them.
  const simdojo::PcieSpec pcie_spec = device_.pcie();
  const auto supported_width = [](uint8_t width) {
    return width == 1 || width == 2 || width == 4 || width == 8 || width == 12 || width == 16 ||
           width == 32;
  };
  if (pcie_spec.max_link_speed < 1 || pcie_spec.max_link_speed > 5 ||
      pcie_spec.current_link_speed < 1 || pcie_spec.current_link_speed > pcie_spec.max_link_speed ||
      !supported_width(pcie_spec.max_link_width) ||
      !supported_width(pcie_spec.current_link_width) ||
      pcie_spec.current_link_width > pcie_spec.max_link_width) {
    util::Logger::warn("vfu: invalid PCIe link speed or width");
    return false;
  }
  pxcap pcie{};
  pcie.hdr.id = PCI_CAP_ID_EXP;
  pcie.pxcaps.ver = 2;
  pcie.pxcaps.dpt = PCI_EXP_TYPE_ENDPOINT;
  pcie.pxdcap.flrc = 1;
  pcie.pxlcap.stuff = pcie_spec.max_link_speed | (uint32_t{pcie_spec.max_link_width} << 4);
  pcie.pxls.stuff = pcie_spec.current_link_speed | (uint16_t{pcie_spec.current_link_width} << 4);
  pcie.pxlcap2.stuff = (uint32_t{1} << (pcie_spec.max_link_speed + 1)) - 2;
  pcie.pxdcap2.aocs32 = pcie_spec.atomic_completer_32;
  pcie.pxdcap2.aocs64 = pcie_spec.atomic_completer_64;
  if (vfu_pci_add_capability(ctx_, 0, 0, &pcie) < 0) {
    util::Logger::warn(
        std::format("vfu: cannot publish PCI Express capabilities: {}", std::strerror(errno)));
    return false;
  }

  std::array<uint64_t, 6> bar_sizes{};
  for (const simdojo::BarSpec &bar : device_.bars()) {
    const BarRegionPlan plan = plan_bar_region(bar);
    single_client_ = single_client_ || !plan.mmap_areas.empty();
    if (!plan.valid) {
      util::Logger::warn(
          std::format("vfu: device {} declared an unusable BAR{}", device_.name(), bar.index));
      return false;
    }

    int flags = VFU_REGION_FLAG_RW;
    if (bar.mem) {
      flags |= VFU_REGION_FLAG_MEM;
    }
    if (bar.is_64bit) {
      flags |= VFU_REGION_FLAG_64_BITS;
    }
    if (bar.prefetch) {
      flags |= VFU_REGION_FLAG_PREFETCH;
    }

    std::vector<iovec> mmap_areas;
    mmap_areas.reserve(plan.mmap_areas.size());
    for (const simdojo::MmapArea &area : plan.mmap_areas) {
      mmap_areas.push_back({.iov_base = reinterpret_cast<void *>(area.offset),
                            .iov_len = static_cast<std::size_t>(area.length)});
    }

    if (vfu_setup_region(ctx_, VFU_PCI_DEV_BAR0_REGION_IDX + bar.index, bar.size,
                         kBarTrampolines[static_cast<std::size_t>(bar.index)], flags,
                         mmap_areas.empty() ? nullptr : mmap_areas.data(),
                         static_cast<uint32_t>(mmap_areas.size()), plan.backing_fd,
                         plan.fd_offset) < 0) {
      util::Logger::warn(
          std::format("vfu: cannot set up BAR{}: {}", bar.index, std::strerror(errno)));
      return false;
    }
    bar_sizes[static_cast<std::size_t>(bar.index)] = bar.size;
  }

  const InterruptPlan interrupts = plan_interrupts(device_.interrupts());
  if (!interrupts.supported) {
    util::Logger::warn(
        std::format("vfu: device {} declared interrupts this transport cannot advertise, either a "
                    "kind it does not implement or one described in terms a capability cannot "
                    "express",
                    device_.name()));
    return false;
  }
  // Both of these must happen before the device is realized: the interrupt
  // count decides whether a pin is published in configuration space, and a
  // capability added afterwards is not published at all. Neither call reports
  // being too late, so the ordering is the only thing enforcing it.
  if (interrupts.intx_count != 0 &&
      vfu_setup_device_nr_irqs(ctx_, VFU_DEV_INTX_IRQ, interrupts.intx_count) < 0) {
    util::Logger::warn(std::format("vfu: cannot set up interrupts: {}", std::strerror(errno)));
    return false;
  }
  if (interrupts.msix_count != 0) {
    // The capability names a BAR and two offsets, and nothing downstream
    // checks that they describe somewhere the device actually answers. A
    // client refuses a table that runs past its BAR or that overlaps the
    // pending bits, but it does so at the guest's realization and in terms of
    // the guest's own layout, naming none of the offsets involved.
    //
    // plan_interrupts has already bounded the index, so this indexes safely.
    const auto table_bar = static_cast<std::size_t>(interrupts.table_bar);
    const uint64_t table_end = interrupts.table_offset + interrupts.msix_count * kMsixEntryBytes;
    // The pending bits are an array of 64-bit words, not a byte per vector.
    const uint64_t pending_end =
        interrupts.pending_offset + ((interrupts.msix_count + 63) / 64) * kMsixPendingWordBytes;
    const bool overlap =
        interrupts.table_offset < pending_end && interrupts.pending_offset < table_end;
    if (table_end > bar_sizes[table_bar] || pending_end > bar_sizes[table_bar] || overlap) {
      util::Logger::warn(std::format(
          "vfu: device {} puts its message table at {:#x}..{:#x} and its pending bits at "
          "{:#x}..{:#x} of BAR{}, which is {} bytes",
          device_.name(), interrupts.table_offset, table_end, interrupts.pending_offset,
          pending_end, interrupts.table_bar, bar_sizes[table_bar]));
      return false;
    }
    if (vfu_setup_device_nr_irqs(ctx_, VFU_DEV_MSIX_IRQ, interrupts.msix_count) < 0) {
      util::Logger::warn(
          std::format("vfu: cannot set up message interrupts: {}", std::strerror(errno)));
      return false;
    }
    // The table and pending-bit offsets are recorded in units of eight bytes,
    // and the low three bits of each field carry the BAR index instead.
    msixcap msix{};
    msix.hdr.id = PCI_CAP_ID_MSIX;
    msix.mxc.ts = static_cast<uint16_t>(interrupts.msix_count - 1);
    msix.mtab.tbir = static_cast<uint32_t>(interrupts.table_bar);
    msix.mtab.to = static_cast<uint32_t>(interrupts.table_offset >> 3);
    msix.mpba.pbir = static_cast<uint32_t>(interrupts.table_bar);
    msix.mpba.pbao = static_cast<uint32_t>(interrupts.pending_offset >> 3);
    if (vfu_pci_add_capability(ctx_, 0, 0, &msix) < 0) {
      util::Logger::warn(
          std::format("vfu: cannot publish the message table: {}", std::strerror(errno)));
      return false;
    }
  }

  if (vfu_setup_device_dma(ctx_, LIBVFIO_USER_MAX_DMA_REGIONS, &dma_register_trampoline,
                           &dma_unregister_trampoline) < 0) {
    util::Logger::warn(std::format("vfu: cannot set up DMA: {}", std::strerror(errno)));
    return false;
  }

  vfu_setup_host_doorbell_cb(ctx_, [](vfu_ctx_t *ctx, uint64_t gpa, uint64_t datamatch,
                                      uint32_t size, int fd) {
    host_of(ctx).register_host_doorbell(gpa, datamatch, size, fd);
  });

  if (target_grants_ != nullptr)
    vfu_setup_device_quiesce_cb(ctx_, &quiesce_trampoline);

  if (vfu_setup_device_reset_cb(ctx_, &reset_trampoline) < 0) {
    util::Logger::warn(std::format("vfu: cannot set up reset: {}", std::strerror(errno)));
    return false;
  }

  if (vfu_realize_ctx(ctx_) < 0) {
    util::Logger::warn(std::format("vfu: cannot realize device: {}", std::strerror(errno)));
    return false;
  }
  return true;
}

VfioDeviceHost::ServeResult VfioDeviceHost::run(std::stop_token stop_token) {
  {
    const std::lock_guard lock(vfu_mutex_);
    if (ctx_ != nullptr) {
      // Keep the check serialized with destruction, then leave the library
      // lock before entering the serve loop.
    } else {
      util::Logger::warn("vfu: run() called before a successful build()");
      return ServeResult::Failed;
    }
  }

  while (!stop_token.stop_requested()) {
    int poll_fd = -1;
    bool needs_attach = false;
    {
      const std::lock_guard lock(state_mutex_);
      needs_attach = !attached_;
    }
    {
      const std::lock_guard lock(vfu_mutex_);
      poll_fd = vfu_get_poll_fd(ctx_);
    }
    if (poll_fd < 0) {
      util::Logger::warn("vfu: transport has no descriptor to wait on");
      return ServeResult::Failed;
    }

    // Anything another thread handed over runs here, on the serving thread and
    // between protocol messages, so it sees the device the way a callback does.
    // Drained before the wait rather than after it, which bounds the delay at
    // one poll timeout rather than leaving a request until the next message.
    std::optional<std::function<void()>> asked;
    {
      const std::lock_guard asked_lock(asked_mutex_);
      asked.swap(asked_);
      asked_running_ = asked.has_value();
    }
    if (asked.has_value()) {
      const std::lock_guard lock(vfu_mutex_);
      // The request runs on this thread, so an exception escaping it would take
      // the process down rather than the request. Report and carry on serving:
      // a faulty request is not a reason to drop the guest's device.
      try {
        (*asked)();
      } catch (const std::exception &error) {
        util::Logger::warn(
            std::format("vfu: a request for the serving thread failed: {}", error.what()));
      } catch (...) {
        util::Logger::warn("vfu: a request for the serving thread failed");
      }
      const std::lock_guard asked_lock(asked_mutex_);
      asked_running_ = false;
    }

    std::array<pollfd, 2> waits{{{.fd = poll_fd, .events = POLLIN, .revents = 0},
                                 {.fd = quiesce_wake_->fd(), .events = POLLIN, .revents = 0}}};
    const int ready = poll(waits.data(), waits.size(), kPollTimeoutMs);
    if (ready < 0 && errno != EINTR) {
      util::Logger::warn(std::format("vfu: poll failed: {}", std::strerror(errno)));
      return ServeResult::Failed;
    }
    if ((waits[1].revents & POLLIN) != 0) {
      uint64_t completed = 0;
      [[maybe_unused]] ssize_t n = ::read(quiesce_wake_->fd(), &completed, sizeof(completed));
    }
    if (ready <= 0 && !quiesce_pending_) {
      continue;
    }

    // A pending command has no new socket traffic to read. The fabric owner
    // keeps running while this thread waits for its final grant permit; the
    // eventfd wakes us as soon as that permit leaves its effect body.
    if (quiesce_pending_ && !target_quiesce_drained())
      continue;

    if (needs_attach) {
      int attach_result = 0;
      int attach_error = 0;
      {
        const std::lock_guard lock(vfu_mutex_);
        attach_result = vfu_attach_ctx(ctx_);
        attach_error = errno;
      }
      if (attach_result < 0) {
        if (attach_error != EAGAIN && attach_error != EWOULDBLOCK && attach_error != EINTR) {
          util::Logger::warn(
              std::format("vfu: client failed to attach: {}", std::strerror(attach_error)));
        }
        continue;
      }
      std::shared_ptr<simdojo::PciTransportSession> session =
          device_.activate_transport_session(&transport_);
      if (session == nullptr) {
        util::Logger::warn("vfu: client attached but no fresh PCI transport session was available");
        return ServeResult::Failed;
      }
      {
        const std::lock_guard lock(state_mutex_);
        client_session_ = std::move(session);
        attached_ = true;
        // Each client gets its own diagnostics: one that shares everything
        // mmap-ably must not be silently denied an explanation because a previous
        // client already used up the warning.
        warned_unreachable_region_ = false;
        declined_regions_ = 0;
        declined_bytes_ = 0;
      }
      {
        const std::lock_guard lock(vfu_mutex_);
        refresh_peer_bar_window();
      }
      continue;
    }

    int run_result = 0;
    int run_error = 0;
    bool disconnected = false;
    bool stop_after_disconnect = false;
    uint64_t declined_regions = 0;
    uint64_t declined_bytes = 0;
    std::optional<simdojo::ResetKind> reset;
    std::shared_ptr<simdojo::PciTransportSession> closing_session;
    {
      const std::lock_guard lock(vfu_mutex_);
      if (quiesce_pending_) {
        run_result = vfu_device_quiesced(ctx_, 0);
        end_target_quiesce();
      } else {
        run_result = vfu_run_ctx(ctx_);
        if (quiescing_ && !quiesce_pending_)
          end_target_quiesce();
      }
      run_error = errno;
      refresh_peer_bar_window();
    }
    if (run_result < 0 && run_error == EBUSY && quiesce_pending_)
      continue;
    {
      const std::lock_guard lock(state_mutex_);
      reset = std::exchange(pending_reset_, std::nullopt);

      disconnected = run_result < 0 && run_error == ENOTCONN;
      if (reset == simdojo::ResetKind::LostConnection)
        disconnected = true;
      // ENOTCONN can arrive without a reset callback. Revoke the old session
      // before resetting the device or accepting another client in that case.
      if (disconnected)
        reset = simdojo::ResetKind::LostConnection;

      if (reset.has_value()) {
        // Refuse fresh leases before reset. Draining and device teardown happen
        // below, after the callback-state lock and the library-call lock are
        // both released.
        if (disconnected) {
          attached_ = false;
          disconnect_in_progress_ = true;
        }
        closing_session = device_.revoke_transport_session(&transport_);
        client_session_.reset();
      }
      if (disconnected) {
        guest_regions_.clear();
        for (auto &db : host_doorbells_) {
          if (db.fd >= 0) {
            ::close(db.fd);
            db.fd = -1;
          }
        }
        host_doorbells_.clear();
        declined_regions = declined_regions_;
        declined_bytes = declined_bytes_;
        stop_after_disconnect = single_client_;
      }
    }

    if (closing_session != nullptr && target_grants_ != nullptr)
      target_grants_->revoke_session(target_ordinal_, closing_session->generation());

    if (reset.has_value() && !finish_session_reset(*reset, std::move(closing_session)))
      return ServeResult::Failed;

    if (disconnected) {
      // Publish completion only after reset and admitted transport work are
      // complete. The early attached_ clear continues to reject stale DMA/IRQ
      // calls while the node coordinator sees the reset as in progress.
      const std::lock_guard lock(state_mutex_);
      ++completed_disconnects_;
      disconnect_in_progress_ = false;
    }

    if (reset.has_value() && !disconnected) {
      std::shared_ptr<simdojo::PciTransportSession> session =
          device_.activate_transport_session(&transport_);
      if (session == nullptr) {
        util::Logger::warn("vfu: reset completed but no fresh PCI transport session was available");
        return ServeResult::Failed;
      }
      const std::lock_guard lock(state_mutex_);
      client_session_ = std::move(session);
    }

    if (disconnected) {
      if (stop_after_disconnect) {
        util::Logger::warn("vfu: client disconnected; this device shares memory by descriptor, "
                           "which cannot be reclaimed, so serving ends here");
        return ServeResult::Stopped;
      }
      // The client went away and had nothing mapped, so wait for a new one rather
      // than tearing the server down: a VMM may be restarted against a server
      // that keeps running. The next attach receives a new session generation.
      if (declined_regions != 0) {
        util::Logger::warn(std::format(
            "vfu: declined {} guest window(s) totalling {} bytes during that connection "
            "because they were not shared mmap-ably",
            declined_regions, declined_bytes));
      }
      continue;
    }

    if (run_result < 0 && run_error != EAGAIN && run_error != EINTR) {
      util::Logger::warn(std::format("vfu: serving failed: {}", std::strerror(run_error)));
      return ServeResult::Failed;
    }
  }
  return ServeResult::Stopped;
}

void VfioDeviceHost::note_unreachable_region(const simdojo::DmaRegion &region) {
  const std::lock_guard lock(state_mutex_);
  ++declined_regions_;
  declined_bytes_ += region.length;
  if (warned_unreachable_region_) {
    return;
  }
  warned_unreachable_region_ = true;
  util::Logger::warn(std::format(
      "vfu: ignoring the guest window at {:#x}+{} and any others like it: ordinary guest RAM "
      "must be shared mmap-ably, and a count follows when the client disconnects",
      region.guest_phys, region.length));
}

bool VfioDeviceHost::record_guest_region(const simdojo::DmaRegion &region) {
  const std::lock_guard lock(state_mutex_);
  return guest_regions_.insert(region);
}

void VfioDeviceHost::forget_guest_region(const simdojo::DmaRegion &region) {
  const std::lock_guard lock(state_mutex_);
  guest_regions_.erase(region);
}

bool VfioDeviceHost::trigger(uint32_t vector) {
  const std::lock_guard lock(vfu_mutex_);
  const std::lock_guard state_lock(state_mutex_);
  if (ctx_ == nullptr || !attached_) {
    return false;
  }
  return vfu_irq_trigger(ctx_, vector) == 0;
}

bool VfioDeviceHost::read(uint64_t guest_phys, std::span<std::byte> dst) {
  return read_outcome(guest_phys, dst) == simdojo::DmaAccessOutcome::Complete;
}

bool VfioDeviceHost::write(uint64_t guest_phys, std::span<const std::byte> src) {
  return write_outcome(guest_phys, src) == simdojo::DmaAccessOutcome::Complete;
}

simdojo::DmaAccessOutcome VfioDeviceHost::read_outcome(uint64_t guest_phys,
                                                       std::span<std::byte> dst) {
  return copy_guest_memory(guest_phys, dst.data(), dst.size(), /*to_guest=*/false);
}

simdojo::DmaAccessOutcome VfioDeviceHost::write_outcome(uint64_t guest_phys,
                                                        std::span<const std::byte> src) {
  // vfu_sgl_write does not modify the source, but takes it as void*.
  const simdojo::DmaAccessOutcome outcome =
      copy_guest_memory(guest_phys, const_cast<std::byte *>(src.data()), src.size(),
                        /*to_guest=*/true);
  if (outcome == simdojo::DmaAccessOutcome::Complete) {
    // DmaEngine::write() promises the bytes are guest-visible on return, and that
    // writes become visible in the order they return. The copy above lands in
    // memory the guest has mapped, so completeness is already met; this is what
    // makes the ORDER hold, and it belongs here because this is the code that owns
    // the transfer -- a caller cannot order stores on its behalf.
    std::atomic_thread_fence(std::memory_order_release);
    return simdojo::DmaAccessOutcome::Complete;
  }
  // GPA not in DMA address space — MMIO region (e.g. NVMe BAR0 doorbell,
  // another vfio-user device BAR).  Try device_write() which selects the
  // fastest available path: eventfd doorbell → zero-copy memfd → socket
  // DMA_WRITE → vfu_mmio_write.  This lets a GPU SDMA engine ring an
  // emulated NVMe SQ doorbell or write to a peer vfio-user BAR directly
  // without requiring the MMIO GPA to be registered in DMA space.
  if (outcome == simdojo::DmaAccessOutcome::Faulted) {
    const std::lock_guard lock(vfu_mutex_);
    const std::lock_guard state_lock(state_mutex_);
    if (ctx_ != nullptr && attached_ && !quiescing_ &&
        device_write(guest_phys, src.data(), src.size()) == 0) {
      std::atomic_thread_fence(std::memory_order_release);
      return simdojo::DmaAccessOutcome::Complete;
    }
  }
  return outcome;
}

simdojo::DmaAccessOutcome VfioDeviceHost::atomic_destination(uint64_t guest_phys, uint32_t width,
                                                             bool to_guest) const {
  if (target_grants_ == nullptr || client_session_ == nullptr)
    return simdojo::DmaAccessOutcome::Complete;
  const auto peer = target_grants_->prepare_peer(target_ordinal_, client_session_->generation(),
                                                 guest_phys, width, to_guest);
  if (peer.kind == VfioTargetGrants::PeerKind::OrdinaryRam)
    return simdojo::DmaAccessOutcome::Complete;
  // The peer endpoint does not advertise AtomicOp completion. A peer BAR's
  // shared bytes are mappable, but that alone cannot authorize a peer atomic.
  return peer.kind == VfioTargetGrants::PeerKind::Rejected ? peer.outcome
                                                           : simdojo::DmaAccessOutcome::Faulted;
}

simdojo::DmaAtomicLoadResult VfioDeviceHost::atomic_load(uint64_t guest_phys, uint32_t width) {
  if ((width != sizeof(uint16_t) && width != sizeof(uint32_t) && width != sizeof(uint64_t)) ||
      guest_phys % width != 0 || width > std::numeric_limits<uint64_t>::max() - guest_phys) {
    return {.outcome = simdojo::DmaAccessOutcome::Malformed};
  }

  const std::lock_guard lock(vfu_mutex_);
  const std::lock_guard state_lock(state_mutex_);
  if (ctx_ == nullptr || !attached_ || quiescing_)
    return {.outcome = simdojo::DmaAccessOutcome::Unavailable};
  if (const auto outcome = atomic_destination(guest_phys, width, false);
      outcome != simdojo::DmaAccessOutcome::Complete)
    return {.outcome = outcome};

  std::vector<std::byte> sgl_storage(dma_sg_size());
  auto *sgl = reinterpret_cast<dma_sg_t *>(sgl_storage.data());
  if (vfu_addr_to_sgl(ctx_, reinterpret_cast<vfu_dma_addr_t>(guest_phys), width, sgl, 1,
                      PROT_READ) != 1 ||
      !vfu_sg_is_mappable(ctx_, sgl)) {
    return {.outcome = simdojo::DmaAccessOutcome::Faulted};
  }

  iovec segment{};
  if (vfu_sgl_get(ctx_, sgl, &segment, 1, 0) < 0)
    return {.outcome = simdojo::DmaAccessOutcome::Faulted};
  if (segment.iov_base == nullptr || segment.iov_len < width ||
      reinterpret_cast<uintptr_t>(segment.iov_base) % width != 0) {
    vfu_sgl_put(ctx_, sgl, &segment, 1);
    return {.outcome = simdojo::DmaAccessOutcome::Faulted};
  }

  uint64_t value = 0;
  if (width == sizeof(uint64_t)) {
    value = std::atomic_ref<uint64_t>(*static_cast<uint64_t *>(segment.iov_base))
                .load(std::memory_order_acquire);
  } else if (width == sizeof(uint32_t)) {
    value = std::atomic_ref<uint32_t>(*static_cast<uint32_t *>(segment.iov_base))
                .load(std::memory_order_acquire);
  } else {
    value = std::atomic_ref<uint16_t>(*static_cast<uint16_t *>(segment.iov_base))
                .load(std::memory_order_acquire);
  }
  vfu_sgl_put(ctx_, sgl, &segment, 1);
  return {.outcome = simdojo::DmaAccessOutcome::Complete, .value = value};
}

simdojo::DmaAccessOutcome VfioDeviceHost::atomic_store(uint64_t guest_phys, uint32_t width,
                                                       uint64_t value) {
  if ((width != sizeof(uint32_t) && width != sizeof(uint64_t)) || guest_phys % width != 0 ||
      width > std::numeric_limits<uint64_t>::max() - guest_phys) {
    return simdojo::DmaAccessOutcome::Malformed;
  }

  const std::lock_guard lock(vfu_mutex_);
  const std::lock_guard state_lock(state_mutex_);
  if (ctx_ == nullptr || !attached_ || quiescing_)
    return simdojo::DmaAccessOutcome::Unavailable;
  if (const auto outcome = atomic_destination(guest_phys, width, true);
      outcome != simdojo::DmaAccessOutcome::Complete)
    return outcome;

  std::vector<std::byte> sgl_storage(dma_sg_size());
  auto *sgl = reinterpret_cast<dma_sg_t *>(sgl_storage.data());
  if (vfu_addr_to_sgl(ctx_, reinterpret_cast<vfu_dma_addr_t>(guest_phys), width, sgl, 1,
                      PROT_WRITE) != 1 ||
      !vfu_sg_is_mappable(ctx_, sgl)) {
    return simdojo::DmaAccessOutcome::Faulted;
  }

  iovec segment{};
  if (vfu_sgl_get(ctx_, sgl, &segment, 1, 0) < 0)
    return simdojo::DmaAccessOutcome::Faulted;
  if (segment.iov_base == nullptr || segment.iov_len < width ||
      reinterpret_cast<uintptr_t>(segment.iov_base) % width != 0) {
    vfu_sgl_put(ctx_, sgl, &segment, 1);
    return simdojo::DmaAccessOutcome::Faulted;
  }

  if (width == sizeof(uint64_t)) {
    std::atomic_ref<uint64_t>(*static_cast<uint64_t *>(segment.iov_base))
        .store(value, std::memory_order_release);
  } else {
    std::atomic_ref<uint32_t>(*static_cast<uint32_t *>(segment.iov_base))
        .store(static_cast<uint32_t>(value), std::memory_order_release);
  }
  vfu_sgl_put(ctx_, sgl, &segment, 1);
  return simdojo::DmaAccessOutcome::Complete;
}

simdojo::DmaAtomicCompareExchangeResult VfioDeviceHost::compare_exchange(uint64_t guest_phys,
                                                                         uint32_t width,
                                                                         uint64_t expected,
                                                                         uint64_t desired) {
  if ((width != sizeof(uint32_t) && width != sizeof(uint64_t)) || guest_phys % width != 0 ||
      width > std::numeric_limits<uint64_t>::max() - guest_phys) {
    return {.outcome = simdojo::DmaAccessOutcome::Malformed};
  }

  const std::lock_guard lock(vfu_mutex_);
  const std::lock_guard state_lock(state_mutex_);
  if (ctx_ == nullptr || !attached_ || quiescing_)
    return {.outcome = simdojo::DmaAccessOutcome::Unavailable};
  if (const auto outcome = atomic_destination(guest_phys, width, true);
      outcome != simdojo::DmaAccessOutcome::Complete)
    return {.outcome = outcome};

  std::vector<std::byte> sgl_storage(dma_sg_size());
  auto *sgl = reinterpret_cast<dma_sg_t *>(sgl_storage.data());
  if (vfu_addr_to_sgl(ctx_, reinterpret_cast<vfu_dma_addr_t>(guest_phys), width, sgl, 1,
                      PROT_READ | PROT_WRITE) != 1 ||
      !vfu_sg_is_mappable(ctx_, sgl)) {
    return {.outcome = simdojo::DmaAccessOutcome::Faulted};
  }

  iovec segment{};
  if (vfu_sgl_get(ctx_, sgl, &segment, 1, 0) < 0)
    return {.outcome = simdojo::DmaAccessOutcome::Faulted};
  if (segment.iov_base == nullptr || segment.iov_len < width ||
      reinterpret_cast<uintptr_t>(segment.iov_base) % width != 0) {
    vfu_sgl_put(ctx_, sgl, &segment, 1);
    return {.outcome = simdojo::DmaAccessOutcome::Faulted};
  }

  simdojo::DmaAtomicCompareExchangeResult result{
      .outcome = simdojo::DmaAccessOutcome::Complete,
      .observed = expected,
      .exchanged = false,
  };
  if (width == sizeof(uint64_t)) {
    uint64_t observed = expected;
    result.exchanged = std::atomic_ref<uint64_t>(*static_cast<uint64_t *>(segment.iov_base))
                           .compare_exchange_strong(observed, desired, std::memory_order_acq_rel,
                                                    std::memory_order_acquire);
    result.observed = observed;
  } else {
    uint32_t observed = static_cast<uint32_t>(expected);
    result.exchanged =
        std::atomic_ref<uint32_t>(*static_cast<uint32_t *>(segment.iov_base))
            .compare_exchange_strong(observed, static_cast<uint32_t>(desired),
                                     std::memory_order_acq_rel, std::memory_order_acquire);
    result.observed = observed;
  }
  vfu_sgl_put(ctx_, sgl, &segment, 1);
  return result;
}

simdojo::DmaAccessOutcome VfioDeviceHost::copy_guest_memory(uint64_t guest_phys, void *data,
                                                            std::size_t length, bool to_guest) {
  if (length == 0) {
    return simdojo::DmaAccessOutcome::Complete;
  }
  if (length > std::numeric_limits<uint64_t>::max() - guest_phys)
    return simdojo::DmaAccessOutcome::Malformed;

  const std::lock_guard lock(vfu_mutex_);
  const std::lock_guard state_lock(state_mutex_);
  if (ctx_ == nullptr || !attached_ || quiescing_) {
    return simdojo::DmaAccessOutcome::Unavailable;
  }

  VfioTargetGrants::PrepareResult targets;
  if (target_grants_ != nullptr && client_session_ != nullptr) {
    targets = target_grants_->prepare(target_ordinal_, client_session_->generation(), guest_phys,
                                      length, to_guest);
    if (targets.outcome != simdojo::DmaAccessOutcome::Complete)
      return targets.outcome;
  }

  // Any gaps between target intervals must be ordinary mmap-backed guest RAM.
  // Acquire every owner lease and ordinary SG mapping before moving any bytes.
  const uint64_t end = guest_phys + length;
  struct OrdinaryGap {
    uint64_t iova;
    std::unique_ptr<PreparedMappedSpan> mapping;
  };
  std::vector<OrdinaryGap> ordinary;
  const auto prepare_gap = [&](uint64_t start, uint64_t stop) {
    auto mapping =
        prepare_mapped_span(ctx_, start, static_cast<std::size_t>(stop - start), to_guest);
    if (mapping == nullptr)
      return false;
    ordinary.push_back({start, std::move(mapping)});
    return true;
  };
  uint64_t cursor = guest_phys;
  for (const auto &target : targets.segments) {
    if (cursor < target.iova && !prepare_gap(cursor, target.iova))
      return simdojo::DmaAccessOutcome::Faulted;
    cursor = target.iova + target.length;
  }
  if (cursor < end && !prepare_gap(cursor, end))
    return simdojo::DmaAccessOutcome::Faulted;

  // Every remaining copy uses a retained local mapping and cannot report a
  // short BAR read or write after an earlier segment has moved.
  auto *bytes = static_cast<std::byte *>(data);
  std::size_t ordinary_index = 0;
  cursor = guest_phys;
  for (const auto &target : targets.segments) {
    if (cursor < target.iova) {
      const OrdinaryGap &gap = ordinary[ordinary_index++];
      gap.mapping->copy(bytes + (gap.iova - guest_phys), to_guest);
    }
    const auto target_bytes = std::span(bytes + (target.iova - guest_phys), target.length);
    target.copy(target_bytes, to_guest);
    cursor = target.iova + target.length;
  }
  if (cursor < end) {
    const OrdinaryGap &gap = ordinary[ordinary_index];
    gap.mapping->copy(bytes + (gap.iova - guest_phys), to_guest);
  }
  return simdojo::DmaAccessOutcome::Complete;
}

void VfioDeviceHost::register_host_doorbell(uint64_t gpa, uint64_t datamatch, uint32_t size,
                                             int fd) {
  std::lock_guard lock(vfu_mutex_);
  for (auto &db : host_doorbells_) {
    if (db.gpa == gpa && db.datamatch == datamatch && db.size == size) {
      if (db.fd >= 0)
        ::close(db.fd);
      db.fd = fd;
      util::Logger::debug_print([&](std::ostringstream &b) {
        b << std::format("vfu: HOST_DOORBELL_FD updated: gpa={:#x} datamatch={:#x} size={} fd={}",
                         gpa, datamatch, size, fd);
      });
      return;
    }
  }
  host_doorbells_.push_back({gpa, datamatch, size, fd});
  util::Logger::debug_print([&](std::ostringstream &b) {
    b << std::format("vfu: HOST_DOORBELL_FD registered: gpa={:#x} datamatch={:#x} size={} fd={}",
                     gpa, datamatch, size, fd);
  });
}

int VfioDeviceHost::device_write(uint64_t gpa, const void *data, size_t len) {
  // Caller must hold vfu_mutex_.

  // Path 1: eventfd doorbell — kernel-only, no socket round-trip.
  for (const auto &db : host_doorbells_) {
    if (db.gpa == gpa && db.size == static_cast<uint32_t>(len)) {
      const uint64_t val = 1;
      if (::write(db.fd, &val, sizeof(val)) == sizeof(val)) {
        util::Logger::debug_print([&](std::ostringstream &b) {
          b << std::format("vfu: device_write({:#x}): doorbell eventfd fd={}", gpa, db.fd);
        });
        return 0;
      }
      return -errno;
    }
  }

  // Path 2: memfd-backed DMA region — zero-copy memcpy via vfu_sgl_get.
  // Path 3 fallback: DMA-registered but non-memfd — vfu_sgl_write socket msg.
  std::vector<std::byte> sgl_storage(dma_sg_size());
  auto *sgl = reinterpret_cast<dma_sg_t *>(sgl_storage.data());
  if (vfu_addr_to_sgl(ctx_, reinterpret_cast<vfu_dma_addr_t>(gpa), len, sgl, 1, PROT_WRITE) == 1) {
    iovec iov{};
    if (vfu_sgl_get(ctx_, sgl, &iov, 1, 0) == 0) {
      std::memcpy(iov.iov_base, data, len);
      std::atomic_thread_fence(std::memory_order_release);
      vfu_sgl_put(ctx_, sgl, &iov, 1);
      util::Logger::debug_print([&](std::ostringstream &b) {
        b << std::format("vfu: device_write({:#x}): vfu_sgl_get (zero-copy)", gpa);
      });
      return 0;
    }
    // Non-memfd: fall through to vfu_sgl_write.
    if (vfu_sgl_write(ctx_, sgl, 1, const_cast<void *>(data), VFU_SGL_DIRECT_ACCESS) == 0) {
      util::Logger::debug_print([&](std::ostringstream &b) {
        b << std::format("vfu: device_write({:#x}): vfu_sgl_write", gpa);
      });
      return 0;
    }
    return -errno;
  }

  // Path 4: MMIO or unregistered GPA — vfu_mmio_write socket msg.
  util::Logger::debug_print([&](std::ostringstream &b) {
    b << std::format("vfu: device_write({:#x}): vfu_mmio_write (MMIO)", gpa);
  });
  return vfu_mmio_write(ctx_, gpa, data, len);
}

} // namespace rocjitsu
