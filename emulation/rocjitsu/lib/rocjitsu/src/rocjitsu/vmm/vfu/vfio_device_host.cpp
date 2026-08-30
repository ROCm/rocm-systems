// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vmm/vfu/vfio_device_host.h"

#include "rocjitsu/vmm/vfu/bus_plan.h"

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
}

#include <poll.h>
#include <sys/mman.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <cstring>
#include <format>
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

/// @brief Upper bound on segments one transfer may span, so a pathologically
/// fragmented guest cannot make the device allocate without limit.
constexpr std::size_t kMaxSgEntries = 256;

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
/// mmap-ably, so a host address is exactly that signal. It is the only window
/// kind this transport supports, and the only one the public API lets it
/// identify: a descriptor-backed window serviced by file I/O and one that must
/// be fetched over the protocol are indistinguishable from here, since both
/// arrive without a host address.
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
    // Never announced, so there is nothing to withdraw.
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
  device_of(ctx).reset(to_reset_kind(type));
  return 0;
} catch (...) {
  errno = EIO;
  return -1;
}

} // namespace

VfioDeviceHost::VfioDeviceHost(std::string socket_path, simdojo::PciDevice &device)
    : socket_path_(std::move(socket_path)), device_(device) {
  device_.set_irq_sink(this);
  device_.set_dma_engine(this);
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
  device_.set_irq_sink(nullptr);
  device_.set_dma_engine(nullptr);
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
  if (ctx_ == nullptr) {
    util::Logger::warn("vfu: run() called before a successful build()");
    return ServeResult::Failed;
  }

  while (!stop_token.stop_requested()) {
    int poll_fd = -1;
    bool needs_attach = false;
    {
      const std::lock_guard lock(vfu_mutex_);
      needs_attach = !attached_;
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

    pollfd wait = {.fd = poll_fd, .events = POLLIN, .revents = 0};
    const int ready = poll(&wait, 1, kPollTimeoutMs);
    if (ready < 0 && errno != EINTR) {
      util::Logger::warn(std::format("vfu: poll failed: {}", std::strerror(errno)));
      return ServeResult::Failed;
    }
    if (ready <= 0) {
      continue;
    }

    const std::lock_guard lock(vfu_mutex_);
    if (needs_attach) {
      if (vfu_attach_ctx(ctx_) < 0) {
        if (errno != EAGAIN && errno != EWOULDBLOCK && errno != EINTR) {
          util::Logger::warn(std::format("vfu: client failed to attach: {}", std::strerror(errno)));
        }
        continue;
      }
      attached_ = true;
      // Each client gets its own diagnostics: one that shares everything
      // mmap-ably must not be silently denied an explanation because a previous
      // client already used up the warning.
      warned_unreachable_region_ = false;
      declined_regions_ = 0;
      declined_bytes_ = 0;
      continue;
    }

    if (vfu_run_ctx(ctx_) >= 0 || errno == EAGAIN || errno == EINTR) {
      continue;
    }
    if (errno != ENOTCONN) {
      util::Logger::warn(std::format("vfu: serving failed: {}", std::strerror(errno)));
      return ServeResult::Failed;
    }
    if (single_client_) {
      util::Logger::warn("vfu: client disconnected; this device shares memory by descriptor, "
                         "which cannot be reclaimed, so serving ends here");
      return ServeResult::Stopped;
    }
    // The client went away and had nothing mapped, so wait for a new one rather
    // than tearing the server down: a VMM may be restarted against a server
    // that keeps running.
    attached_ = false;
    guest_regions_.clear();
    if (declined_regions_ != 0) {
      util::Logger::warn(
          std::format("vfu: declined {} guest window(s) totalling {} bytes during that connection "
                      "because they were not shared mmap-ably",
                      declined_regions_, declined_bytes_));
    }
  }
  return ServeResult::Stopped;
}

void VfioDeviceHost::note_unreachable_region(const simdojo::DmaRegion &region) {
  const std::lock_guard lock(vfu_mutex_);
  ++declined_regions_;
  declined_bytes_ += region.length;
  if (warned_unreachable_region_) {
    return;
  }
  warned_unreachable_region_ = true;
  util::Logger::warn(std::format(
      "vfu: ignoring the guest window at {:#x}+{} and any others like it: this transport serves "
      "only windows the client shares mmap-ably, and a count follows when the client disconnects",
      region.guest_phys, region.length));
}

bool VfioDeviceHost::record_guest_region(const simdojo::DmaRegion &region) {
  const std::lock_guard lock(vfu_mutex_);
  return guest_regions_.insert(region);
}

void VfioDeviceHost::forget_guest_region(const simdojo::DmaRegion &region) {
  const std::lock_guard lock(vfu_mutex_);
  guest_regions_.erase(region);
}

std::size_t VfioDeviceHost::bytes_until_region_end(uint64_t guest_phys) const {
  return guest_regions_.bytes_until_end(guest_phys);
}

bool VfioDeviceHost::transfer_one_sg(dma_sg *sg, void *data, uint64_t guest_phys,
                                     std::size_t length, bool to_guest) {
  // A segment the transport has mapped is copied locally. This transport serves
  // only mapped windows, so anything else is refused; a window that must be
  // fetched over the protocol could collide with the client's own traffic on a
  // single connection, and one serviced by file I/O cannot be told apart from it
  // through the public API. The multi-segment path
  // does not come through here, but reaches the same conclusion by its own
  // route: the library's scatter-gather mapping call fails with EFAULT for any
  // segment that is not mapped.
  if (!vfu_sg_is_mappable(ctx_, sg)) {
    util::Logger::warn(std::format(
        "vfu: refusing to reach {:#x}+{} because it is outside this transport's mmap-only policy",
        guest_phys, length));
    return false;
  }
  const int result = to_guest ? vfu_sgl_write(ctx_, sg, 1, data, VFU_SGL_DIRECT_ACCESS)
                              : vfu_sgl_read(ctx_, sg, 1, data, VFU_SGL_DIRECT_ACCESS);
  return result == 0;
}

bool VfioDeviceHost::copy_by_region(uint64_t guest_phys, void *data, std::size_t length,
                                    bool to_guest) {
  // Walk the windows the client registered, transferring at most one window at a
  // time. The scatter-gather entries do not expose where one registration ends,
  // but the map notifications did.
  auto *cursor = static_cast<std::byte *>(data);
  uint64_t address = guest_phys;
  std::size_t remaining = length;
  while (remaining > 0) {
    const std::size_t until_end = bytes_until_region_end(address);
    if (until_end == 0) {
      return false;
    }
    const std::size_t chunk = std::min(remaining, until_end);
    if (!copy_one_segment(address, cursor, chunk, to_guest)) {
      return false;
    }
    cursor += chunk;
    address += chunk;
    remaining -= chunk;
  }
  return true;
}

bool VfioDeviceHost::copy_one_segment(uint64_t guest_phys, void *data, std::size_t length,
                                      bool to_guest) {
  std::vector<std::byte> sgl_storage(dma_sg_size());
  auto *sgl = reinterpret_cast<dma_sg_t *>(sgl_storage.data());
  const int prot = to_guest ? PROT_WRITE : PROT_READ;

  if (vfu_addr_to_sgl(ctx_, reinterpret_cast<vfu_dma_addr_t>(guest_phys), length, sgl, 1, prot) !=
      1) {
    return false;
  }
  return transfer_one_sg(sgl, data, guest_phys, length, to_guest);
}

bool VfioDeviceHost::trigger(uint32_t vector) {
  const std::lock_guard lock(vfu_mutex_);
  if (ctx_ == nullptr || !attached_) {
    return false;
  }
  return vfu_irq_trigger(ctx_, vector) == 0;
}

bool VfioDeviceHost::read(uint64_t guest_phys, std::span<std::byte> dst) {
  return copy_guest_memory(guest_phys, dst.data(), dst.size(), /*to_guest=*/false);
}

bool VfioDeviceHost::write(uint64_t guest_phys, std::span<const std::byte> src) {
  // vfu_sgl_write does not modify the source, but takes it as void*.
  if (!copy_guest_memory(guest_phys, const_cast<std::byte *>(src.data()), src.size(),
                         /*to_guest=*/true)) {
    return false;
  }
  // DmaEngine::write() promises the bytes are guest-visible on return, and that
  // writes become visible in the order they return. The copy above lands in
  // memory the guest has mapped, so completeness is already met; this is what
  // makes the ORDER hold, and it belongs here because this is the code that owns
  // the transfer -- a caller cannot order stores on its behalf.
  std::atomic_thread_fence(std::memory_order_release);
  return true;
}

bool VfioDeviceHost::copy_guest_memory(uint64_t guest_phys, void *data, std::size_t length,
                                       bool to_guest) {
  if (length == 0) {
    return true;
  }

  const std::lock_guard lock(vfu_mutex_);
  if (ctx_ == nullptr || !attached_) {
    return false;
  }

  const int prot = to_guest ? PROT_WRITE : PROT_READ;
  std::vector<std::byte> sgl_storage(dma_sg_size() * kInitialSgEntries);
  std::size_t capacity = kInitialSgEntries;

  int nr_sgs = vfu_addr_to_sgl(ctx_, reinterpret_cast<vfu_dma_addr_t>(guest_phys), length,
                               reinterpret_cast<dma_sg_t *>(sgl_storage.data()), capacity, prot);
  if (nr_sgs < 0) {
    // The range is mapped but spans more segments than were offered; the
    // library reports how many it needs as -(needed) - 1.
    const std::size_t needed = static_cast<std::size_t>(-nr_sgs - 1);
    if (needed <= capacity) {
      return false;
    }
    if (needed > kMaxSgEntries) {
      // A heavily fragmented guest can need more entries than are worth holding
      // at once. That is a reason to stream the transfer, not to reject it: a
      // client-side IOMMU can legitimately reflect a large range as thousands of
      // page-sized windows.
      return copy_by_region(guest_phys, data, length, to_guest);
    }
    capacity = needed;
    sgl_storage.assign(dma_sg_size() * capacity, std::byte{0});
    nr_sgs = vfu_addr_to_sgl(ctx_, reinterpret_cast<vfu_dma_addr_t>(guest_phys), length,
                             reinterpret_cast<dma_sg_t *>(sgl_storage.data()), capacity, prot);
  }
  if (nr_sgs <= 0) {
    return false;
  }

  auto *sgl = reinterpret_cast<dma_sg_t *>(sgl_storage.data());
  const auto segment_count = static_cast<std::size_t>(nr_sgs);

  // The message-based helpers carry exactly one segment, so anything spanning a
  // registration boundary has to be copied through the shared mappings instead.
  if (segment_count == 1) {
    return transfer_one_sg(sgl, data, guest_phys, length, to_guest);
  }

  std::vector<iovec> segments(segment_count);
  if (vfu_sgl_get(ctx_, sgl, segments.data(), segment_count, 0) < 0) {
    // The client shared these windows without descriptors, so they cannot be
    // mapped and the message helpers carry only one segment each.
    return copy_by_region(guest_phys, data, length, to_guest);
  }

  auto *cursor = static_cast<std::byte *>(data);
  std::size_t copied = 0;
  for (const iovec &segment : segments) {
    const std::size_t chunk = std::min(segment.iov_len, length - copied);
    if (to_guest) {
      std::memcpy(segment.iov_base, cursor, chunk);
    } else {
      std::memcpy(cursor, segment.iov_base, chunk);
    }
    cursor += chunk;
    copied += chunk;
  }

  // Releasing the mapping is what marks the written pages dirty for migration.
  vfu_sgl_put(ctx_, sgl, segments.data(), segment_count);
  return copied == length;
}

} // namespace rocjitsu
