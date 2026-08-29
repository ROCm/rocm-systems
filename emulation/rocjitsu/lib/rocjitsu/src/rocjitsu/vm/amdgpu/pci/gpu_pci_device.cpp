// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/amdgpu/pci/gpu_pci_device.h"

#include "rocjitsu/vm/amdgpu/pci/ip_discovery.h"
#include "rocjitsu/vm/amdgpu/pci/ip_discovery_profile.h"
#include "rocjitsu/vm/amdgpu/pci/mmio_registers.h"
#include "util/log.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cerrno>
#include <cstring>
#include <format>
#include <limits>
#include <optional>
#include <span>
#include <utility>

namespace rocjitsu {
namespace {

bool is_supported_width(std::size_t width) {
  return width == 1 || width == 2 || width == 4 || width == 8;
}

bool is_power_of_two(uint64_t value) { return value != 0 && (value & (value - 1)) == 0; }

/// @brief Address bits carried by the low index register.
///
/// @details Its top bit is the flag the driver sets to say it is addressing
/// memory rather than a register, so the address continues in the high register
/// from bit 31 rather than bit 32.
constexpr uint32_t kIndirectLowMask = 0x7fffffff;
/// @brief MM_INDEX bit selecting the memory space rather than the register space.
constexpr uint32_t kIndirectMemorySelect = 0x80000000;

/// @brief Address bit at which the high index register begins.
constexpr unsigned kIndirectHighShift = 31;

/// @brief Write a whole buffer at an offset, resuming after a short write.
///
/// @details A single pwrite is permitted to transfer fewer bytes than asked
/// for, and a discovery table is large enough that a partial store would leave
/// the driver reading a truncated table rather than none at all: the signature
/// at the front would be intact, so the failure would surface as a malformed
/// record rather than as an absent table.
/// @param[in] fd File to write to.
/// @param[in] bytes Buffer to store.
/// @param[in] at Byte offset within @p fd.
/// @retval true The whole buffer was stored.
/// @retval false The write failed or the file would take no more.
[[nodiscard]] bool write_all_at(int fd, std::span<const std::byte> bytes, uint64_t at) {
  std::size_t done = 0;
  while (done < bytes.size()) {
    const ssize_t wrote =
        ::pwrite(fd, bytes.data() + done, bytes.size() - done, static_cast<off_t>(at + done));
    if (wrote < 0) {
      if (errno == EINTR) {
        continue;
      }
      return false;
    }
    if (wrote == 0) {
      return false;
    }
    done += static_cast<std::size_t>(wrote);
  }
  return true;
}

/// @brief Where an HDP flush is issued, as a byte offset into the register BAR.
///
/// @details Not a register of any IP block but a hole the bus reserves, which
/// `nbio_v7_11_set_reg_remap` points the driver at on any non-virtualized
/// function with pages of 4 KiB or less. A flush is a write of zero here
/// followed by an unrelated register read to order it, so answering the write
/// is the whole of the model: nothing reads this back.
constexpr uint64_t kHdpFlushHoleOffset = 0x44000;

/// @brief Invalidation engines each memory hub has.
///
/// @details The driver picks one by index and reaches it by stride, so the
/// device answers all of them rather than guessing which. Engine 17 is the one
/// the GART flush uses, but that is a driver convention rather than a property
/// of the hardware.
constexpr uint32_t kInvalidationEngines = 18;

/// @brief Registers between one invalidation engine and the next.
constexpr uint32_t kInvalidationEngineStride = 1;

/// @brief Value a semaphore reads as when the acquire has been granted.
///
/// @details The GC 12.0 flush path acquires an engine's semaphore before
/// writing its request and releases it by writing zero, so this has to answer
/// reads and ignore writes: one that stored the release would grant the first
/// acquire and stall every flush after it. That path takes the semaphore for
/// MMHUB only; both hubs are answered anyway, because eighteen more registers
/// cost nothing and a hub that answered half a handshake would be the harder
/// thing to explain.
///
/// The GC 12.1 path this device publishes today does not take the semaphore at
/// all -- it writes the request and polls the acknowledge -- so these registers
/// are answered for the profile rather than for the boot, and are untouched on
/// gfx1250. They are still worth answering: this class is deliberately not
/// specific to one part, and every other GC 12.x takes the path that does.
///
/// Granting unconditionally is right only while nothing else acquires. The
/// device does not execute rings, so the driver's own lock is all that
/// serializes flushes; a device that ran packets would have a second acquirer
/// and this register would stop meaning what it means on hardware.
constexpr uint32_t kInvalidationSemaphoreHeld = 0x1;

/// @brief One hub's invalidation-register layout, and the version it describes.
///
/// @details These offsets move between versions of the same block -- GC 12.0
/// puts the three at 0x1635/0x1647/0x1659 where 12.1 puts them at
/// 0x1645/0x1657/0x1669 -- so the hardware ID alone does not identify a layout.
/// Matching on the version the published table actually declares is what stops
/// a profile for one version being answered with another's addresses, which
/// would leave the driver polling registers this device never defined.
///
/// One row per layout that has been read out of a header and checked. A block
/// whose version matches no row is refused rather than guessed at: there is no
/// safe default, because a wrong offset is indistinguishable from hardware that
/// never completes a flush.
struct HubInvalidationLayout {
  IpHardwareId id;      ///< Block the layout belongs to.
  uint16_t major;       ///< Block major version it was read from.
  uint16_t minor;       ///< Block minor version it was read from.
  uint32_t semaphore;   ///< Engine 0's semaphore, in dwords from the block's first segment.
  uint32_t request;     ///< Engine 0's request, likewise.
  uint32_t acknowledge; ///< Engine 0's acknowledge, likewise.
};

/// @details GC from `regGCVM_INVALIDATE_ENG0_{SEM,REQ,ACK}` of
/// `gc_12_1_0_offset.h`, MMHUB from `regMMVM_INVALIDATE_ENG0_{SEM,REQ,ACK}` of
/// `mmhub_4_1_0_offset.h`; all `_BASE_IDX 0`, so all relative to the block's
/// first register segment.
constexpr HubInvalidationLayout kHubInvalidationLayouts[] = {
    {IpHardwareId::Gc, 12, 1, 0x1645, 0x1657, 0x1669},
    {IpHardwareId::MmHub, 4, 1, 0x0575, 0x0587, 0x0599},
};

/// @brief The interrupt ring's registers within OSSSYS, in dwords.
///
/// @details `regIH_RB_{CNTL,RPTR,WPTR,BASE,BASE_HI,WPTR_ADDR_HI,WPTR_ADDR_LO}`
/// and `regIH_DOORBELL_RPTR` of `osssys_7_1_0_offset.h`, all `_BASE_IDX 0`.
/// The last matters more than it looks: the ring is created with doorbells
/// unconditionally on, so the driver acknowledges entries by writing a doorbell
/// rather than the read-pointer register. The driver programs them in
/// `ih_v7_0_enable_ring` and reads the pointers back on every interrupt, so
/// they are ordinary storage rather than answers fixed at reset.
constexpr uint32_t kIhRingControl = 0x0080;
constexpr uint32_t kIhRingReadPointer = 0x0081;
constexpr uint32_t kIhRingWritePointer = 0x0082;
constexpr uint32_t kIhRingBase = 0x0083;
constexpr uint32_t kIhRingBaseHigh = 0x0084;
constexpr uint32_t kIhRingWritePointerAddressHigh = 0x0085;
constexpr uint32_t kIhRingWritePointerAddressLow = 0x0086;
constexpr uint32_t kIhRingDoorbell = 0x0087;

/// @brief Bits the driver shifts the ring's address down by before writing it.
///
/// @details The base register holds bits 39:8, so the low eight are implied
/// zero and the ring is at least 256-byte aligned. The address itself does not
/// stop at 39: the high register below continues it from bit 40, and the two
/// together carry 48 bits.
constexpr unsigned kIhRingBaseShift = 8;

/// @brief Address bit at which the high half of the base register continues.
constexpr unsigned kIhRingBaseHighShift = 40;

/// @brief Bits of the write-pointer address carried by its high register.
constexpr uint64_t kIhWritePointerAddressHighMask = 0xffff;

/// @brief Bit selecting the ring within the control register.
constexpr uint32_t kIhRingEnableMask = 1U << 0;

/// @brief Bit asking for an interrupt per entry.
constexpr uint32_t kIhRingInterruptEnableMask = 1U << 17;

/// @brief Where the size field sits within the control register.
///
/// @details The driver stores the base-two logarithm of the ring's size in
/// dwords, so a size of `4 << field` bytes.
constexpr unsigned kIhRingSizeShift = 1;

/// @brief Its width, once shifted down: the field mask is `0x3e`.
constexpr uint32_t kIhRingSizeMask = 0x1f;

/// @brief Where the address-space field sits within the control register.
constexpr unsigned kIhRingSpaceShift = 28;

/// @brief Its width, once shifted down: the field mask is `0x70000000`.
constexpr uint32_t kIhRingSpaceMask = 0x7;

/// @brief Bits of the ring's base carried by its high register.
///
/// @details Narrower than the register's own field, which is seventeen bits,
/// because the driver only ever writes eight of them.
constexpr uint64_t kIhRingBaseHighMask = 0xff;

/// @brief Dwords one interrupt entry occupies, and the bytes that comes to.
///
/// @details The driver advances its read pointer by this much per entry and
/// decodes exactly this many dwords, so an entry of any other size would put
/// every later entry at an offset it does not look at.
constexpr uint32_t kInterruptEntryDwords = 8;
constexpr uint32_t kInterruptEntryBytes = kInterruptEntryDwords * 4;

/// @brief Bits of the write-pointer register that carry the offset.
///
/// @details `IH_RB_WPTR__OFFSET_MASK`. The offset occupies bits 17:2; bits 1:0
/// are the overflow flag, so the register does not carry the low two bits of an
/// address at all. Anything above the field is not part of one either.
constexpr uint32_t kIhWritePointerOffsetMask = 0x0003fffc;

/// @brief Largest ring whose every entry the write-pointer register can name.
///
/// @details The offset field is sixteen bits wide, so it addresses exactly one
/// 256 KiB ring. A larger one would have entries the device could never point
/// at, and worse, the device would wrap at the field width while the driver
/// wrapped at the ring size -- the two would disagree about which entry a
/// pointer names, and the driver would decode the never-written remainder as
/// entries. The size field is five bits and guest-writable, so it can ask for
/// far more than this.
constexpr uint64_t kLargestAddressableRingBytes =
    (uint64_t{kIhWritePointerOffsetMask} & ~uint64_t{kInterruptEntryBytes - 1}) +
    kInterruptEntryBytes;

/// @brief The message vector an entry is announced on.
///
/// @details The device advertises one, so this is it. A second would need the
/// capability to advertise it before anything could be delivered on it.
constexpr uint32_t kInterruptVector = 0;

/// @brief Acknowledge value reporting every VMID's flush already complete.
///
/// @details The driver polls for `1 << vmid` and this device has no translation
/// to invalidate, so every flush is finished before it is asked for. Answering
/// per-VMID instead would mean modelling which VMIDs exist, to no end: the
/// alternative to "already done" is not "done later" but the driver giving up
/// after its timeout and continuing anyway.
constexpr uint32_t kInvalidationComplete = 0xffffffff;

} // namespace

GpuPciDevice::GpuPciDevice(std::string name, const GpuPciDeviceSpec &spec, BarAccessTrace *trace)
    : simdojo::PciDevice(std::move(name), spec.id), spec_(spec), trace_(trace) {
  if (spec_.vram_bytes == 0) {
    // A device with no memory has nothing to present; an aperture cannot stand
    // in for a capacity that was never described.
    util::Logger::warn(std::format("{}: no video memory was configured", this->name()));
    return;
  }
  // A BAR is a power-of-two window by definition; anything else is refused here
  // rather than deeper in a transport that would reject it less clearly.
  for (const auto &[what, size] : {std::pair{"video memory", spec_.vram_aperture_bytes},
                                   std::pair{"doorbell", spec_.doorbell_aperture_bytes},
                                   std::pair{"register", spec_.register_aperture_bytes}}) {
    if (!is_power_of_two(size)) {
      util::Logger::warn(std::format("{}: the {} aperture is {} bytes, which is not a power of two",
                                     this->name(), what, size));
      return;
    }
    if (size < kMinMemoryBarBytes) {
      util::Logger::warn(
          std::format("{}: the {} aperture is {} bytes, below the {}-byte minimum for a memory BAR",
                      this->name(), what, size, kMinMemoryBarBytes));
      return;
    }
  }
  if (spec_.vram_aperture_bytes > spec_.vram_bytes) {
    util::Logger::warn(std::format("{}: the video memory aperture is larger than the memory itself",
                                   this->name()));
    return;
  }
  if (spec_.register_aperture_bytes < kMinRegisterApertureBytes) {
    util::Logger::warn(std::format(
        "{}: a register aperture of {} bytes cannot reach the registers the driver reads before "
        "discovery; at least {} are needed",
        this->name(), spec_.register_aperture_bytes, kMinRegisterApertureBytes));
    return;
  }

  // Video memory is a file so the guest can map it rather than trapping to us
  // for every access, which is the only way an aperture of this size is usable.
  // It stays sparse until written.
  // The driver reads the capacity as a count of megabytes in a 32-bit register,
  // and treats zero and all-ones as "there is no usable memory here". A capacity
  // that cannot be said in those terms would produce a device that looks fine
  // and then fails discovery.
  // The driver reads the capacity as a count of megabytes, so anything finer is
  // unrepresentable and rounds down here. That is not silently lossy: this
  // rounded value is what the device reports, and everything else it derives --
  // including where it later publishes a discovery table -- is derived from the
  // same value rather than from the true byte count, so the driver never
  // computes an address the device did not use. The remainder is simply memory
  // the guest is never told about.
  if ((spec_.vram_bytes & 0xfffffULL) != 0) {
    util::Logger::warn(std::format(
        "{}: {} bytes of video memory is not a whole number of megabytes; reporting {} MiB and "
        "leaving the remaining {} bytes unreachable by the guest",
        this->name(), spec_.vram_bytes, spec_.vram_bytes >> 20, spec_.vram_bytes & 0xfffffULL));
  }
  const uint64_t megabytes = spec_.vram_bytes >> 20;
  if (megabytes == 0 || megabytes >= std::numeric_limits<uint32_t>::max()) {
    util::Logger::warn(std::format(
        "{}: {} bytes of video memory is {} megabytes, which the driver cannot read as a size",
        this->name(), spec_.vram_bytes, megabytes));
    return;
  }
  vram_megabytes_ = static_cast<uint32_t>(megabytes);

  // The whole of memory is backed, sparsely, because the driver reaches past
  // the window through the indirect registers; only the window is shared with
  // the guest.
  vram_fd_ = ::memfd_create("rocjitsu-vram", MFD_CLOEXEC | MFD_ALLOW_SEALING);
  if (vram_fd_ < 0 || ::ftruncate(vram_fd_, static_cast<off_t>(spec_.vram_bytes)) != 0) {
    util::Logger::warn(std::format("{}: cannot back the video memory aperture", this->name()));
    return;
  }
  void *mapped =
      ::mmap(nullptr, spec_.vram_aperture_bytes, PROT_READ | PROT_WRITE, MAP_SHARED, vram_fd_, 0);
  if (mapped == MAP_FAILED) {
    util::Logger::warn(std::format("{}: cannot map the video memory aperture", this->name()));
    ::close(vram_fd_);
    vram_fd_ = -1;
    return;
  }
  vram_ = static_cast<std::byte *>(mapped);

  // The descriptor is shared with a client that could otherwise resize it and
  // leave the server touching memory past the end of the file.
  if (::fcntl(vram_fd_, F_ADD_SEALS, F_SEAL_SHRINK | F_SEAL_GROW | F_SEAL_SEAL) != 0) {
    util::Logger::warn(std::format("{}: cannot seal the video memory backing", this->name()));
    return;
  }
  doorbells_.resize(spec_.doorbell_aperture_bytes);
  msix_table_.resize(kMsixBarBytes);

  // One entry per dword of the aperture.
  const auto register_count = static_cast<uint32_t>(spec_.register_aperture_bytes / 4);
  registers_.assign(register_count, 0);
  modelled_.assign(register_count, false);
  read_only_.assign(register_count, false);
  reset_registers();

  // A device that answers every pre-discovery register and then has nothing at
  // the address those answers point to is worse than one that fails to
  // construct: the driver would read a zero signature and reject it, with the
  // registers all looking correct.
  if (!publish_discovery_table()) {
    return;
  }
  // A device whose flush handshakes cannot be answered looks correct right up
  // until the driver waits on one, and then stalls with every register it read
  // beforehand reporting success. Refusing here is what turns that into a
  // message at construction.
  if (!flushes_answerable_) {
    util::Logger::warn(std::format(
        "{}: this device cannot answer the VM flush handshakes the driver waits on", this->name()));
    return;
  }
  usable_ = true;
}

bool GpuPciDevice::publish_discovery_table() {
  // Guarded rather than left to pwrite returning EBADF, because this is now
  // reachable from the public reset() as well as from a device whose memory
  // never came up, and because reading and writing memory guard the same way.
  if (vram_fd_ < 0) {
    util::Logger::warn(
        std::format("{}: no video memory to publish a discovery table into", name()));
    return false;
  }
  if (spec_.discovery.blocks.empty()) {
    util::Logger::warn(std::format("{}: no discovery profile for this configuration, so a guest "
                                   "driver would find nothing to attach to",
                                   name()));
    return false;
  }
  const IpDiscoveryBuild built = build_ip_discovery_table(spec_.discovery);
  if (!built.ok()) {
    util::Logger::warn(
        std::format("{}: cannot build a discovery table: {}", name(), built.problem));
    return false;
  }
  const IpDiscoveryValidation checked = validate_ip_discovery_table(built.table);
  if (!checked.valid) {
    util::Logger::warn(std::format("{}: would publish a discovery table the driver refuses: {}",
                                   name(), checked.problem));
    return false;
  }
  if (built.table.size() > kDiscoveryTableBytes) {
    util::Logger::warn(std::format("{}: the discovery table is {} bytes, more than the {} the "
                                   "driver reads",
                                   name(), built.table.size(), kDiscoveryTableBytes));
    return false;
  }
  // The driver never learns the byte count. It reads the megabyte count out of
  // RCC_CONFIG_MEMSIZE and computes the address itself, as
  // `(vram_size << 20) - DISCOVERY_TMR_OFFSET` (amdgpu_discovery.c:1996-1997).
  // Publishing at a top-of-memory derived from the unrounded byte count would
  // therefore miss by the remainder for any capacity that is not a whole number
  // of megabytes, and the driver would read zeros and reject the signature with
  // no hint that the address was the problem. Deriving the address from the
  // value the device already committed to in reset_registers() is what keeps
  // the two from drifting; configs/gfx1151.json is one of the capacities that
  // would otherwise miss, by 446464 bytes.
  const uint64_t reported_bytes = static_cast<uint64_t>(vram_megabytes_) << 20;
  if (reported_bytes < kDiscoveryOffsetFromTopOfVram) {
    util::Logger::warn(std::format("{}: {} bytes of video memory has no room for a discovery "
                                   "table {} from the top",
                                   name(), reported_bytes, kDiscoveryOffsetFromTopOfVram));
    return false;
  }

  const uint64_t at = reported_bytes - kDiscoveryOffsetFromTopOfVram;
  if (!write_all_at(vram_fd_, built.table, at)) {
    util::Logger::warn(std::format("{}: cannot store the {}-byte discovery table at {:#x}: {}",
                                   name(), built.table.size(), at, std::strerror(errno)));
    return false;
  }
  return true;
}

GpuPciDevice::~GpuPciDevice() {
  if (vram_ != nullptr) {
    ::munmap(vram_, spec_.vram_aperture_bytes);
  }
  if (vram_fd_ >= 0) {
    ::close(vram_fd_);
  }
}

std::vector<simdojo::BarSpec> GpuPciDevice::bars() const {
  simdojo::BarSpec vram;
  vram.index = kVramBar;
  vram.size = spec_.vram_aperture_bytes;
  vram.mem = true;
  vram.prefetch = true;
  vram.is_64bit = true;
  vram.backing_fd = vram_fd_;
  if (vram_fd_ >= 0) {
    vram.mmap_areas.push_back({.offset = 0, .length = spec_.vram_aperture_bytes});
  }

  simdojo::BarSpec doorbell;
  doorbell.index = kDoorbellBar;
  doorbell.size = spec_.doorbell_aperture_bytes;
  doorbell.mem = true;
  doorbell.prefetch = true;
  doorbell.is_64bit = true;

  // Registers always trap: a doorbell can be a plain store to memory we scan,
  // but a register read has to be answered by the model.
  simdojo::BarSpec registers;
  registers.index = kRegisterBar;
  registers.size = spec_.register_aperture_bytes;
  registers.mem = true;

  // Trapped rather than mapped, so every access is seen. What this stage does
  // with what it sees is nothing beyond storing it: the table and the pending
  // bits are one undifferentiated buffer, the device is told nothing about
  // which vectors the guest has masked, and it sets no pending bit of its own.
  // That is enough for a client that emulates the table itself, which is what
  // this transport's clients do. Modelling ownership properly -- the device
  // setting pending bits for a masked vector and learning mask changes through
  // a state callback -- needs an interface this device does not have yet, and
  // trapping is the precondition for adding it without changing the bus shape.
  simdojo::BarSpec msix;
  msix.index = kMsixBar;
  msix.size = kMsixBarBytes;
  msix.mem = true;

  return {vram, doorbell, registers, msix};
}

void GpuPciDevice::reset_registers() {
  // The registers the driver reads before it can locate the IP discovery table.
  // Reporting no memory, or all-ones, makes it give up before it starts.
  define_register(byte_offset_of(MmioRegister::RccConfigMemsize), vram_megabytes_);
  define_register(byte_offset_of(MmioRegister::Mp0SmnC2pmsg33), kFirmwareInitDoneBit);
  // Zero says this is a physical function with virtualization disabled, which is
  // what lets the driver treat the device as passed through to it whole. The
  // alternative would be to model the SR-IOV mailbox a virtual function reaches
  // its host through, which this device does not have.
  define_register(byte_offset_of(MmioRegister::RccIovFuncIdentifier), 0);
  // Zero here tells the driver the discovery table is not published through
  // these registers, so it looks for it at the top of video memory instead.
  define_register(byte_offset_of(MmioRegister::DriverScratch0), 0);
  define_register(byte_offset_of(MmioRegister::DriverScratch1), 0);
  define_register(byte_offset_of(MmioRegister::DriverScratch2), 0);
  // The indirect window and its data register are how memory outside the
  // aperture is reached, so they are modelled rather than reading as absent.
  define_register(byte_offset_of(MmioRegister::MmIndex), 0);
  define_register(byte_offset_of(MmioRegister::MmIndexHi), 0);
  define_register(byte_offset_of(MmioRegister::MmData), 0);
  // Readable before discovery, like the registers above: it is inside the
  // pre-discovery aperture and is named, so leaving it undefined would report it
  // as a register this device does not model when in fact it has an answer.
  define_register(byte_offset_of(MmioRegister::IpDiscoveryVersion), kIpDiscoveryVersion);

  // Accepting the flush is the whole model; the driver orders it with a read of
  // a different register and never reads this one back.
  define_register(kHdpFlushHoleOffset, 0);

  // Both hubs, at the segments the published table gave them, so the addresses
  // the driver computes and the ones answered here cannot drift apart. A hub
  // the table does not name has no registers to answer.
  //
  // Offsets are from gc_12_1_0_offset.h and mmhub_4_1_0_offset.h. Which driver
  // consumes them takes two steps to answer: GC 12.1.0 adds gmc_v12_0's ip
  // block -- which is the name the guest logs -- and that block's early_init
  // then installs gmc_v12_1's flush functions for this version. The hub
  // register offsets come from gfxhub_v12_1 and mmhub_v4_1_0 either way, since
  // those are set outside that switch.
  flushes_answerable_ = true;
  for (const IpHardwareId id : {IpHardwareId::Gc, IpHardwareId::MmHub}) {
    const IpBlock *hub = published_block(id);
    if (hub == nullptr) {
      // A hub the table does not name has no registers to answer, and the
      // driver will not look for them either.
      continue;
    }
    const HubInvalidationLayout *layout = nullptr;
    for (const HubInvalidationLayout &candidate : kHubInvalidationLayouts) {
      if (candidate.id == id && candidate.major == hub->major && candidate.minor == hub->minor) {
        layout = &candidate;
        break;
      }
    }
    if (layout == nullptr) {
      util::Logger::warn(std::format(
          "{}: no invalidation-register layout is known for hardware id {} version {}.{}, and "
          "answering with another version's addresses would leave the driver polling registers "
          "this device never defined",
          name(), static_cast<uint16_t>(id), hub->major, hub->minor));
      flushes_answerable_ = false;
      continue;
    }
    if (!define_invalidation_engines(hub->register_bases.front(),
                                     {.semaphore = layout->semaphore,
                                      .request = layout->request,
                                      .acknowledge = layout->acknowledge})) {
      flushes_answerable_ = false;
    }
  }

  // The interrupt ring's registers answer as plain storage: the driver writes
  // where it put the ring and reads its own pointers back, so anything the
  // device invented here would be a fact the driver did not state.
  ih_control_offset_.reset();
  if (const IpBlock *osssys = published_block(IpHardwareId::OssSys);
      osssys != nullptr && segment_within_aperture(osssys->register_bases.front())) {
    const uint64_t segment = osssys->register_bases.front();
    for (const uint32_t reg :
         {kIhRingControl, kIhRingReadPointer, kIhRingWritePointer, kIhRingBase, kIhRingBaseHigh,
          kIhRingWritePointerAddressHigh, kIhRingWritePointerAddressLow, kIhRingDoorbell}) {
      const uint64_t at = (segment + reg) * 4;
      if (at + 4 <= spec_.register_aperture_bytes) {
        define_register(at, 0);
      }
    }
    // Recorded only when the dword it names was actually modelled. Recording it
    // for a control register that fell outside the aperture would have every
    // later read of it answer out of an index this device never defined.
    const uint64_t control_at = (segment + kIhRingControl) * 4;
    if (control_at + 4 <= spec_.register_aperture_bytes) {
      ih_control_offset_ = control_at;
    }
    // Cleared with the registers it describes: a reset discards what the driver
    // said, so the next programming has to be reported as freshly as the first.
    announced_interrupt_ring_ = false;
  }
}

const IpBlock *GpuPciDevice::published_block(IpHardwareId id) const {
  for (const IpBlock &block : spec_.discovery.blocks) {
    if (block.hardware_id == id && !block.register_bases.empty()) {
      return &block;
    }
  }
  return nullptr;
}

bool GpuPciDevice::segment_within_aperture(uint64_t segment) const {
  // Checked at the one place a segment becomes an address rather than by each
  // caller: every one turns it into `(segment + register) * 4`, and a large
  // enough segment wraps that multiply into a small address that passes an
  // aperture bounds test and lands on an unrelated register.
  //
  // A segment equal to the aperture's dword count already names the dword one
  // past the end, so this is not a `>`.
  if (segment >= spec_.register_aperture_bytes / 4) {
    util::Logger::warn(
        std::format("{}: the block at segment {:#x} is not within a {}-byte register aperture, "
                    "so its registers cannot be answered",
                    name(), segment, spec_.register_aperture_bytes));
    return false;
  }
  return true;
}

bool GpuPciDevice::deliver_interrupt(const InterruptEntry &entry) {
  const IpBlock *osssys = published_block(IpHardwareId::OssSys);
  if (osssys == nullptr || !segment_within_aperture(osssys->register_bases.front())) {
    return false;
  }
  const uint64_t wptr_register = (osssys->register_bases.front() + kIhRingWritePointer) * 4;
  if (wptr_register + 4 > spec_.register_aperture_bytes) {
    return false;
  }

  const InterruptRing ring = interrupt_ring();
  if (!ring.enabled || ring.bytes < kInterruptEntryBytes ||
      ring.bytes > kLargestAddressableRingBytes) {
    return false;
  }
  // An address in a space this device cannot resolve would be written to
  // whatever guest page happens to sit at that number. That is worse than
  // declining: the driver would carry on waiting while memory it owns was
  // quietly changed underneath it.
  if (ring.space != InterruptRingSpace::BusAddress) {
    return false;
  }
  if (dma_ == nullptr || irq_ == nullptr) {
    return false;
  }

  // Publishing to nowhere is not publishing: the driver reads the pointer from
  // memory, so a ring switched on before that address was given has no way to
  // be told anything, and guest-physical zero is a real page to write over.
  //
  // The ring's own base is the same question and the same answer. A driver may
  // set the enable bit before it has written the base -- the registers are
  // separate and it writes them in its own order -- and zero there means unset,
  // not "the ring is at address zero". Writing entries into guest-physical zero
  // would corrupt whatever the guest keeps in its first page.
  if (ring.base == 0 || ring.wptr_address == 0) {
    return false;
  }

  // The pointers are byte offsets into the ring and wrap with it. Taking the
  // current one from the register rather than from a member keeps the device's
  // idea of it and the driver's the same object rather than two that drift --
  // every driver-side re-init writes this register back to zero.
  //
  // It is a *guest-writable* register, though, and this is where its value
  // becomes an address. So it is put through the field mask the hardware
  // defines and then aligned down to an entry, because the driver's own write-
  // and read-pointer shadows sit immediately after the ring: an entry placed
  // at an unaligned offset would run off the end and overwrite exactly the two
  // words the interrupt protocol depends on.
  const uint64_t stated =
      registers_[static_cast<uint32_t>(wptr_register / 4)] & kIhWritePointerOffsetMask;
  const auto at =
      static_cast<uint32_t>((stated & ~uint64_t{kInterruptEntryBytes - 1}) % ring.bytes);

  std::array<uint32_t, kInterruptEntryDwords> words = {};
  words[0] = static_cast<uint32_t>(entry.client_id) | (static_cast<uint32_t>(entry.source_id) << 8);
  words[4] = entry.data[0];
  words[5] = entry.data[1];
  words[6] = entry.data[2];
  words[7] = entry.data[3];

  std::array<std::byte, kInterruptEntryBytes> raw = {};
  std::memcpy(raw.data(), words.data(), raw.size());
  if (!dma_->write(ring.base + at, raw)) {
    return false;
  }

  // Only now is there something to point at. Publishing the pointer first would
  // invite the driver to read an entry that is not yet there -- the driver orders
  // its read of the ring against its read of this pointer with a barrier, so the
  // device owes it the matching store order.
  //
  // That order is DmaEngine::write()'s to keep, not this function's: it returns
  // only once the bytes are guest-visible, and writes become visible in the order
  // they return. A fence here would not have been enough anyway, since it cannot
  // order stores an implementation has only queued.
  const auto next = static_cast<uint32_t>((uint64_t{at} + kInterruptEntryBytes) % ring.bytes);
  const auto published = std::bit_cast<std::array<std::byte, sizeof(uint32_t)>>(next);
  if (!dma_->write(ring.wptr_address, published)) {
    return false;
  }
  define_register(wptr_register, next);

  // The ring can be switched on while messages for it are switched off, in
  // which case entries accumulate and the driver finds them when it next looks.
  // The driver moves the two bits together, so this is a state it never asks
  // for -- but the field is decoded, and decoding a field and then ignoring it
  // is how a model starts disagreeing with itself.
  if (!ring.raises_messages) {
    return true;
  }
  return irq_->trigger(kInterruptVector);
}

std::string GpuPciDevice::describe(const InterruptRing &ring) {
  // Initialised to the fallback and switched without a default: a fourth space
  // added later is a compiler warning here, while a value cast in from outside
  // the enumerators still renders as something rather than as a null pointer.
  const char *space = "an unstated address space";
  switch (ring.space) {
  case InterruptRingSpace::BusAddress:
    space = "a bus address";
    break;
  case InterruptRingSpace::GpuVirtual:
    space = "a translated address";
    break;
  case InterruptRingSpace::Unset:
    space = "an unstated address space";
    break;
  }
  // Whether it was switched on is part of the state, not a detail: a sized and
  // addressed ring that the driver never enabled is a different situation from
  // a running one, and reading the two the same way sends whoever is
  // diagnosing a silent guest looking in the wrong place.
  return std::format("its interrupt ring at {:#x}, {} bytes, {}, publishing its write pointer to "
                     "{:#x}; it {} an interrupt per entry, and it is {}",
                     ring.base, ring.bytes, space, ring.wptr_address,
                     ring.raises_messages ? "asks for" : "does not ask for",
                     ring.enabled ? "enabled" : "not yet enabled");
}

GpuPciDevice::InterruptRing GpuPciDevice::interrupt_ring() const {
  InterruptRing ring;
  const IpBlock *osssys = published_block(IpHardwareId::OssSys);
  if (osssys == nullptr || !segment_within_aperture(osssys->register_bases.front())) {
    return ring;
  }
  const auto value = [this, base = osssys->register_bases.front()](uint32_t reg) -> uint32_t {
    const uint64_t at = (base + reg) * 4;
    if (at + 4 > spec_.register_aperture_bytes) {
      return 0;
    }
    return registers_[static_cast<uint32_t>(at / 4)];
  };

  ring.base =
      (static_cast<uint64_t>(value(kIhRingBase)) << kIhRingBaseShift) |
      (static_cast<uint64_t>(value(kIhRingBaseHigh) & kIhRingBaseHighMask) << kIhRingBaseHighShift);
  ring.wptr_address = static_cast<uint64_t>(value(kIhRingWritePointerAddressLow)) |
                      ((static_cast<uint64_t>(value(kIhRingWritePointerAddressHigh)) &
                        kIhWritePointerAddressHighMask)
                       << 32);

  const uint32_t control = value(kIhRingControl);
  ring.enabled = (control & kIhRingEnableMask) != 0;
  ring.raises_messages = (control & kIhRingInterruptEnableMask) != 0;
  // The field beside them saying what the addresses above are addresses in.
  // Anything the driver has not written yet reads as zero, which is neither of
  // the two values it uses and is reported as such rather than guessed at.
  switch ((control >> kIhRingSpaceShift) & kIhRingSpaceMask) {
  case static_cast<uint32_t>(InterruptRingSpace::BusAddress):
    ring.space = InterruptRingSpace::BusAddress;
    break;
  case static_cast<uint32_t>(InterruptRingSpace::GpuVirtual):
    ring.space = InterruptRingSpace::GpuVirtual;
    break;
  default:
    ring.space = InterruptRingSpace::Unset;
    break;
  }
  // The field is the logarithm of the size in dwords, so the size is four bytes
  // shifted up by it. A ring the driver has not sized yet reads as zero rather
  // than as the four bytes that shift would otherwise imply.
  const uint32_t size_log = (control >> kIhRingSizeShift) & kIhRingSizeMask;
  ring.bytes = size_log == 0 ? 0 : uint64_t{4} << size_log;
  return ring;
}

bool GpuPciDevice::define_invalidation_engines(uint64_t segment,
                                               const InvalidationEngineBlocks &blocks) {
  // The blocks sit back to back, so a wrong engine count does not overrun the
  // aperture -- it silently writes one block's registers over the next one's,
  // and the flush the overwritten register served stalls again.
  const uint32_t span = kInvalidationEngines * kInvalidationEngineStride;
  if (blocks.semaphore + span > blocks.request || blocks.request + span > blocks.acknowledge) {
    util::Logger::warn(
        std::format("{}: {} invalidation engines do not fit between the semaphore, request and "
                    "acknowledge blocks of the hub at register segment {:#x} (byte {:#x})",
                    name(), kInvalidationEngines, segment, segment * 4));
    return false;
  }

  for (uint32_t engine = 0; engine < kInvalidationEngines; ++engine) {
    const uint32_t offset = engine * kInvalidationEngineStride;
    const uint64_t semaphore = (segment + blocks.semaphore + offset) * 4;
    const uint64_t request = (segment + blocks.request + offset) * 4;
    const uint64_t acknowledge = (segment + blocks.acknowledge + offset) * 4;
    // A hub whose registers fall outside the aperture is reached through an
    // indirect window this device does not model, so there is nothing to define
    // and defining it would corrupt an unrelated register. Addresses rise with
    // the engine, so no later engine is in range once one is not.
    if (acknowledge + 4 > spec_.register_aperture_bytes) {
      util::Logger::warn(std::format(
          "{}: invalidation engine {} of the hub at register segment {:#x} acknowledges at byte "
          "{:#x}, outside the {}-byte register aperture, so its flushes cannot be answered",
          name(), engine, segment, acknowledge, spec_.register_aperture_bytes));
      return false;
    }
    define_read_only_register(semaphore, kInvalidationSemaphoreHeld);
    define_register(request, 0);
    define_read_only_register(acknowledge, kInvalidationComplete);
  }
  return true;
}

uint64_t GpuPciDevice::indirect_address() const {
  const auto low = registers_[static_cast<uint32_t>(byte_offset_of(MmioRegister::MmIndex) / 4)];
  const auto high = registers_[static_cast<uint32_t>(byte_offset_of(MmioRegister::MmIndexHi) / 4)];
  return (static_cast<uint64_t>(low) & kIndirectLowMask) |
         (static_cast<uint64_t>(high) << kIndirectHighShift);
}

bool GpuPciDevice::read_vram(uint64_t offset, uint32_t &value) const {
  if (vram_fd_ < 0 || offset + sizeof(value) > spec_.vram_bytes) {
    return false;
  }
  return ::pread(vram_fd_, &value, sizeof(value), static_cast<off_t>(offset)) ==
         static_cast<ssize_t>(sizeof(value));
}

bool GpuPciDevice::write_vram(uint64_t offset, uint32_t value) {
  if (vram_fd_ < 0 || offset + sizeof(value) > spec_.vram_bytes) {
    return false;
  }
  return ::pwrite(vram_fd_, &value, sizeof(value), static_cast<off_t>(offset)) ==
         static_cast<ssize_t>(sizeof(value));
}

void GpuPciDevice::define_register(uint64_t byte_offset, uint32_t value) {
  const auto index = static_cast<uint32_t>(byte_offset / 4);
  registers_[index] = value;
  modelled_[index] = true;
  read_only_[index] = false;
}

void GpuPciDevice::define_read_only_register(uint64_t byte_offset, uint32_t value) {
  define_register(byte_offset, value);
  read_only_[static_cast<uint32_t>(byte_offset / 4)] = true;
}

int64_t GpuPciDevice::access_registers(std::span<std::byte> buf, uint64_t offset, bool write) {
  // Registers are 32 bits wide and naturally aligned; anything else is a driver
  // bug or a transport bug, and answering it would hide which.
  if (buf.size() != 4 || (offset % 4) != 0 || offset + 4 > spec_.register_aperture_bytes) {
    return -1;
  }

  const auto index = static_cast<uint32_t>(offset / 4);
  const bool modelled = modelled_[index];
  if (trace_ != nullptr) {
    trace_->record(kRegisterBar, offset, buf.size(), write, modelled);
  }

  // Memory outside the aperture is reached by pointing the index registers at
  // an address and then reading or writing the data register.
  if (offset == byte_offset_of(MmioRegister::MmData)) {
    // Bit 31 of MM_INDEX selects which space the window addresses. This stage
    // models the memory side only, so a request for the register side is
    // refused rather than quietly answered out of the framebuffer -- which
    // would hand the driver bytes from an unrelated address and look like
    // working hardware.
    if ((registers_[static_cast<uint32_t>(byte_offset_of(MmioRegister::MmIndex) / 4)] &
         kIndirectMemorySelect) == 0) {
      if (trace_ != nullptr) {
        trace_->record_rejected(kRegisterBar, offset, buf.size(), write);
      }
      return -1;
    }
    uint32_t value = 0;
    if (write) {
      std::memcpy(&value, buf.data(), sizeof(value));
      (void)write_vram(indirect_address(), value);
    } else {
      (void)read_vram(indirect_address(), value);
      std::memcpy(buf.data(), &value, sizeof(value));
    }
    return 4;
  }

  if (write) {
    // A write to something the device does not model is dropped rather than
    // remembered, so a later read still reports absent hardware instead of
    // echoing back whatever the driver put there. A read-only register keeps
    // its value for the same reason: what it reports is a property of the
    // hardware, not state the driver owns.
    if (modelled && !read_only_[index]) {
      uint32_t value = 0;
      std::memcpy(&value, buf.data(), sizeof(value));
      registers_[index] = value;
      // Switching the interrupt ring on is the moment the device learns where
      // to deliver, and it is worth saying once rather than being read back
      // later: a reset clears these registers, so anything asked afterwards
      // finds a device that was never told.
      if (ih_control_offset_ && offset == *ih_control_offset_ && (value & kIhRingEnableMask) != 0 &&
          !announced_interrupt_ring_) {
        announced_interrupt_ring_ = true;
        const InterruptRing ring = interrupt_ring();
        util::Logger::warn(std::format("{}: the driver enabled {}", name(), describe(ring)));
        // An address the device cannot translate is worth saying plainly now
        // rather than leaving for whoever wonders why no interrupt arrived.
        if (ring.space == InterruptRingSpace::GpuVirtual) {
          util::Logger::warn(
              std::format("{}: that ring is behind translation tables this device does not walk, "
                          "so it cannot be reached; the driver places it there whenever firmware "
                          "is loaded through the security processor",
                          name()));
        } else if (ring.space != InterruptRingSpace::BusAddress) {
          util::Logger::warn(std::format(
              "{}: that ring names no address space, so where it is cannot be acted on", name()));
        }
      }
    }
    return 4;
  }

  const uint32_t value = modelled ? registers_[index] : 0;
  std::memcpy(buf.data(), &value, sizeof(value));
  return 4;
}

int64_t GpuPciDevice::access_memory(std::span<std::byte> buf, uint64_t offset, bool write,
                                    std::span<std::byte> backing) {
  if (offset > backing.size() || buf.size() > backing.size() - offset) {
    return -1;
  }
  const auto begin = backing.begin() + static_cast<std::ptrdiff_t>(offset);
  if (write) {
    std::ranges::copy(buf, begin);
  } else {
    std::copy_n(begin, buf.size(), buf.begin());
  }
  return static_cast<int64_t>(buf.size());
}

int64_t GpuPciDevice::bar_access(int bar, std::span<std::byte> buf, uint64_t offset, bool write) {
  if (!is_supported_width(buf.size())) {
    // Deliberately not traced. The trace answers "which registers does this
    // device still not model", and an unsupported width is a malformed access
    // rather than missing hardware: recording it as unmodelled would put an
    // address into that report that may be modelled perfectly well, and send
    // whoever reads it looking for a register that is not the problem.
    return -1;
  }

  switch (bar) {
  case kRegisterBar:
    return access_registers(buf, offset, write);
  case kDoorbellBar:
    return access_memory(buf, offset, write, doorbells_);
  case kMsixBar:
    return access_memory(buf, offset, write, msix_table_);
  case kVramBar:
    // Reached only for a guest that did not map the aperture; a mapped one
    // never traps here.
    if (vram_ == nullptr) {
      return -1;
    }
    return access_memory(buf, offset, write, {vram_, spec_.vram_aperture_bytes});
  default:
    break;
  }

  if (trace_ != nullptr) {
    trace_->record_rejected(bar, offset, buf.size(), write);
  }
  return -1;
}

void GpuPciDevice::dma_map(const simdojo::DmaRegion & /*region*/) {}

void GpuPciDevice::dma_unmap(const simdojo::DmaRegion & /*region*/) {}

void GpuPciDevice::reset(simdojo::ResetKind kind) {
  util::Logger::warn(std::format("{}: reset requested, kind {}", name(), static_cast<int>(kind)));
  // Everything a client could have changed goes back to power-on state. Video
  // memory is not cleared here: a mapping already handed out cannot be taken
  // back, which is why a device that exports one is served to a single client.
  std::ranges::fill(doorbells_, std::byte{0});
  std::ranges::fill(msix_table_, std::byte{0});
  reset_registers();
  // The table is restored rather than assumed intact. On real hardware it lives
  // in memory the security processor reserves and the driver cannot write; here
  // it is ordinary video memory, reachable through MM_DATA, so a guest that
  // scribbles over it once would otherwise make every later bind fail.
  if (!publish_discovery_table()) {
    util::Logger::warn(std::format("{}: the discovery table could not be restored, so a driver "
                                   "binding after this reset will refuse the device",
                                   name()));
  }
}

} // namespace rocjitsu
