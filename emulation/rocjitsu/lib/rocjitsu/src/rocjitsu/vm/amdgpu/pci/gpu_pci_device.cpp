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
#include <cerrno>
#include <cstring>
#include <format>
#include <limits>
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

  // One entry per dword of the aperture.
  const auto register_count = static_cast<uint32_t>(spec_.register_aperture_bytes / 4);
  registers_.assign(register_count, 0);
  modelled_.assign(register_count, false);
  reset_registers();

  // A device that answers every pre-discovery register and then has nothing at
  // the address those answers point to is worse than one that fails to
  // construct: the driver would read a zero signature and reject it, with the
  // registers all looking correct.
  if (!publish_discovery_table()) {
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

  return {vram, doorbell, registers};
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
    // echoing back whatever the driver put there.
    if (modelled) {
      uint32_t value = 0;
      std::memcpy(&value, buf.data(), sizeof(value));
      registers_[index] = value;
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
