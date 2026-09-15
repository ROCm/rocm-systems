// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/amdgpu/gpu_vm.h"

#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "util/log.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cassert>
#include <format>
#include <limits>
#include <type_traits>
#include <utility>

namespace rocjitsu::amdgpu {
namespace {

constexpr uint64_t kGfx12PteValid = uint64_t{1} << 0;
constexpr uint64_t kGfx12PteSystem = uint64_t{1} << 1;
constexpr uint64_t kGfx12PteExecutable = uint64_t{1} << 4;
constexpr uint64_t kGfx12PteReadable = uint64_t{1} << 5;
constexpr uint64_t kGfx12PteWriteable = uint64_t{1} << 6;
constexpr uint64_t kGfx12PdePte = uint64_t{1} << 63;
constexpr std::array<uint32_t, 5> kGfx12PageTableShifts = {48, 39, 30, 21, 12};

constexpr uint64_t gfx12_virtual_address_mask(Gfx12VmConfig config) {
  const auto bits = static_cast<uint8_t>(config.virtual_address_width);
  return (uint64_t{1} << bits) - 1;
}

constexpr uint64_t gfx12_physical_address_mask(Gfx12VmConfig config) {
  const auto bits = static_cast<uint8_t>(config.physical_address_width);
  return (uint64_t{1} << bits) - 1;
}

constexpr uint64_t gfx12_pte_address_mask(Gfx12VmConfig config) {
  constexpr uint64_t kPageOffsetMask = (uint64_t{1} << 12) - 1;
  return gfx12_physical_address_mask(config) & ~kPageOffsetMask;
}

constexpr uint64_t gfx12_pde_address_mask(Gfx12VmConfig config) {
  constexpr uint64_t kPdeFlagMask = (uint64_t{1} << 6) - 1;
  return gfx12_physical_address_mask(config) & ~kPdeFlagMask;
}

constexpr VmAccessOutcome vm_access_outcome(CopyOutcome outcome) {
  switch (outcome) {
  case CopyOutcome::Complete:
    return VmAccessOutcome::Complete;
  case CopyOutcome::Unavailable:
    return VmAccessOutcome::Unavailable;
  case CopyOutcome::Faulted:
    return VmAccessOutcome::Faulted;
  }
  return VmAccessOutcome::Malformed;
}

class LegacyGpuMemoryTranslator final : public AddressSpaceTranslator {
public:
  LegacyGpuMemoryTranslator(GpuMemory &memory, uint32_t vmid) : memory_(&memory), vmid_(vmid) {}

  VmTranslationResult translate(uint64_t address, std::size_t size,
                                VmAccessKind /*access*/) const override {
    if (size == 0 || size - 1 > std::numeric_limits<uint64_t>::max() - address)
      return {.outcome = VmAccessOutcome::Malformed, .translation = {}};
    constexpr uint64_t kPageBytes = 4096;
    return {
        .outcome = VmAccessOutcome::Complete,
        .translation =
            {
                .domain = VmMemoryDomain::Compatibility,
                .address = address,
                .contiguous_bytes = kPageBytes - (address & (kPageBytes - 1)),
                .mtype = memory_->pte_mtype(address, vmid_),
                .permissions = {.readable = true, .writable = true, .executable = true},
            },
    };
  }

private:
  GpuMemory *memory_;
  uint32_t vmid_;
};

class LegacyGpuMemoryBacking final : public PhysicalMemoryAccess {
public:
  LegacyGpuMemoryBacking(GpuMemory &memory, uint32_t vmid) : memory_(&memory), vmid_(vmid) {}

  VmAccessOutcome read(VmMemoryDomain domain, uint64_t address,
                       std::span<std::byte> bytes) override {
    if (domain != VmMemoryDomain::Compatibility)
      return VmAccessOutcome::Malformed;
    return vm_access_outcome(memory_->read_block_strict(
        address, std::span<uint8_t>(reinterpret_cast<uint8_t *>(bytes.data()), bytes.size()),
        vmid_));
  }

  VmAccessOutcome write(VmMemoryDomain domain, uint64_t address,
                        std::span<const std::byte> bytes) override {
    if (domain != VmMemoryDomain::Compatibility)
      return VmAccessOutcome::Malformed;
    return vm_access_outcome(memory_->write_block_strict(
        address,
        std::span<const uint8_t>(reinterpret_cast<const uint8_t *>(bytes.data()), bytes.size()),
        vmid_));
  }

  AtomicLoadResult atomic_load(VmMemoryDomain domain, uint64_t address, uint32_t width) override {
    if (domain != VmMemoryDomain::Compatibility)
      return {.outcome = VmAccessOutcome::Malformed, .value = 0};
    uint64_t value = 0;
    const VmAccessOutcome outcome =
        vm_access_outcome(memory_->atomic_load(address, width, value, vmid_));
    return {.outcome = outcome, .value = outcome == VmAccessOutcome::Complete ? value : 0};
  }

  VmAccessOutcome atomic_store(VmMemoryDomain domain, uint64_t address, uint32_t width,
                               uint64_t value) override {
    if (domain != VmMemoryDomain::Compatibility)
      return VmAccessOutcome::Malformed;
    return memory_->atomic_store(address, width, value, vmid_) == AccessOutcome::Complete
               ? VmAccessOutcome::Complete
               : VmAccessOutcome::Faulted;
  }

  AtomicCompareExchangeResult compare_exchange(VmMemoryDomain domain, uint64_t address,
                                               uint32_t width, uint64_t expected,
                                               uint64_t desired) override {
    if (domain != VmMemoryDomain::Compatibility)
      return {.outcome = VmAccessOutcome::Malformed};
    uint64_t observed = 0;
    bool exchanged = false;
    const AccessOutcome outcome = memory_->atomic_compare_exchange(
        address, width, expected, desired, observed, exchanged, vmid_);
    return {.outcome = outcome == AccessOutcome::Complete ? VmAccessOutcome::Complete
                                                          : VmAccessOutcome::Faulted,
            .observed = observed,
            .exchanged = exchanged};
  }

private:
  GpuMemory *memory_;
  uint32_t vmid_;
};

Mtype gfx12_mtype(uint64_t entry) {
  switch ((entry >> 54) & 0x3) {
  case 1:
    return Mtype::CC;
  case 3:
    return Mtype::UC;
  case 0:
  case 2:
  default:
    return Mtype::RW;
  }
}

template <typename Span>
VmAccessOutcome
access_translated(const AddressSpaceTranslator &translator, PhysicalMemoryAccess &memory,
                  uint64_t address, Span bytes, std::size_t &completed_bytes, VmAccessKind access,
                  const std::function<void(uint64_t, VmAccessKind)> &fault_reporter = {}) {
  if (completed_bytes > bytes.size())
    return VmAccessOutcome::Malformed;
  while (completed_bytes < bytes.size()) {
    const VmTranslationResult translated =
        translator.translate(address + completed_bytes, bytes.size() - completed_bytes, access);
    if (!translated) {
      if (fault_reporter && translated.outcome != VmAccessOutcome::Unavailable)
        fault_reporter(address + completed_bytes, access);
      return translated.outcome;
    }
    if (translated.translation.contiguous_bytes == 0)
      return VmAccessOutcome::Malformed;
    const std::size_t chunk = static_cast<std::size_t>(std::min<uint64_t>(
        bytes.size() - completed_bytes, translated.translation.contiguous_bytes));
    const VmAccessOutcome outcome = [&]() {
      if constexpr (std::is_const_v<typename Span::element_type>) {
        return memory.write(translated.translation.domain, translated.translation.address,
                            bytes.subspan(completed_bytes, chunk));
      } else {
        return memory.read(translated.translation.domain, translated.translation.address,
                           bytes.subspan(completed_bytes, chunk));
      }
    }();
    if (outcome != VmAccessOutcome::Complete)
      return outcome;
    completed_bytes += chunk;
  }
  return VmAccessOutcome::Complete;
}

} // namespace

VmAccessOutcome read_translated(const AddressSpaceTranslator &translator,
                                PhysicalMemoryAccess &memory, uint64_t address,
                                std::span<std::byte> bytes) {
  std::size_t completed_bytes = 0;
  return access_translated(translator, memory, address, bytes, completed_bytes, VmAccessKind::Read);
}

VmAccessOutcome write_translated(const AddressSpaceTranslator &translator,
                                 PhysicalMemoryAccess &memory, uint64_t address,
                                 std::span<const std::byte> bytes) {
  std::size_t completed_bytes = 0;
  return access_translated(translator, memory, address, bytes, completed_bytes,
                           VmAccessKind::Write);
}

VmTranslationResult GpuVmAccess::translate(uint64_t address, std::size_t size,
                                           VmAccessKind access) const {
  VmTranslationResult translated =
      translator_ != nullptr
          ? translator_->translate(address, size, access)
          : VmTranslationResult{.outcome = VmAccessOutcome::Unavailable, .translation = {}};
  report_terminal_fault(address, access, translated.outcome);
  return translated;
}

VmAccessOutcome GpuVmAccess::probe(uint64_t address, std::size_t size, VmAccessKind access) const {
  if (translator_ == nullptr)
    return VmAccessOutcome::Unavailable;
  if (size == 0 || size - 1 > std::numeric_limits<uint64_t>::max() - address) {
    report_terminal_fault(address, access, VmAccessOutcome::Malformed);
    return VmAccessOutcome::Malformed;
  }

  std::size_t completed_bytes = 0;
  while (completed_bytes < size) {
    const VmTranslationResult translated =
        translate(address + completed_bytes, size - completed_bytes, access);
    if (!translated)
      return translated.outcome;
    if (translated.translation.contiguous_bytes == 0) {
      report_terminal_fault(address + completed_bytes, access, VmAccessOutcome::Malformed);
      return VmAccessOutcome::Malformed;
    }
    completed_bytes += static_cast<std::size_t>(
        std::min<uint64_t>(size - completed_bytes, translated.translation.contiguous_bytes));
  }
  return VmAccessOutcome::Complete;
}

void GpuVmAccess::report_terminal_fault(uint64_t address, VmAccessKind access,
                                        VmAccessOutcome outcome) const {
  if (fault_reporter_ && outcome != VmAccessOutcome::Complete &&
      outcome != VmAccessOutcome::Unavailable) {
    fault_reporter_(address, access);
  }
}

VmAccessOutcome GpuVmAccess::read(uint64_t address, std::span<std::byte> bytes,
                                  VmAccessKind access) const {
  std::size_t completed_bytes = 0;
  return read(address, bytes, completed_bytes, access);
}

VmAccessOutcome GpuVmAccess::read(uint64_t address, std::span<std::byte> bytes,
                                  std::size_t &completed_bytes, VmAccessKind access) const {
  if (access != VmAccessKind::Read && access != VmAccessKind::Execute)
    return VmAccessOutcome::Malformed;
  if (translator_ == nullptr || physical_memory_ == nullptr)
    return VmAccessOutcome::Unavailable;
  return access_translated(*translator_, *physical_memory_, address, bytes, completed_bytes, access,
                           fault_reporter_);
}

VmAccessOutcome GpuVmAccess::write(uint64_t address, std::span<const std::byte> bytes) const {
  std::size_t completed_bytes = 0;
  return write(address, bytes, completed_bytes);
}

VmAccessOutcome GpuVmAccess::write(uint64_t address, std::span<const std::byte> bytes,
                                   std::size_t &completed_bytes) const {
  if (translator_ == nullptr || physical_memory_ == nullptr)
    return VmAccessOutcome::Unavailable;
  return access_translated(*translator_, *physical_memory_, address, bytes, completed_bytes,
                           VmAccessKind::Write, fault_reporter_);
}

AtomicLoadResult GpuVmAccess::atomic_load(uint64_t address, uint32_t width) const {
  if ((width != sizeof(uint32_t) && width != sizeof(uint64_t)) || (address & (width - 1)) != 0) {
    return {.outcome = VmAccessOutcome::Malformed, .value = 0};
  }
  if (translator_ == nullptr || physical_memory_ == nullptr)
    return {.outcome = VmAccessOutcome::Unavailable, .value = 0};

  const VmTranslationResult translated = translate(address, width, VmAccessKind::Atomic);
  if (!translated)
    return {.outcome = translated.outcome, .value = 0};
  if (translated.translation.contiguous_bytes < width)
    return {.outcome = VmAccessOutcome::Malformed, .value = 0};
  return physical_memory_->atomic_load(translated.translation.domain,
                                       translated.translation.address, width);
}

VmAccessOutcome GpuVmAccess::atomic_store(uint64_t address, uint32_t width, uint64_t value) const {
  if ((width != sizeof(uint32_t) && width != sizeof(uint64_t)) || (address & (width - 1)) != 0) {
    return VmAccessOutcome::Malformed;
  }
  if (translator_ == nullptr || physical_memory_ == nullptr)
    return VmAccessOutcome::Unavailable;

  const VmTranslationResult translated = translate(address, width, VmAccessKind::Atomic);
  if (!translated)
    return translated.outcome;
  if (translated.translation.contiguous_bytes < width)
    return VmAccessOutcome::Malformed;
  return physical_memory_->atomic_store(translated.translation.domain,
                                        translated.translation.address, width, value);
}

AtomicCompareExchangeResult GpuVmAccess::compare_exchange(uint64_t address, uint32_t width,
                                                          uint64_t expected,
                                                          uint64_t desired) const {
  if ((width != sizeof(uint32_t) && width != sizeof(uint64_t)) || (address & (width - 1)) != 0) {
    return {.outcome = VmAccessOutcome::Malformed};
  }
  if (translator_ == nullptr || physical_memory_ == nullptr)
    return {.outcome = VmAccessOutcome::Unavailable};

  const VmTranslationResult translated = translate(address, width, VmAccessKind::Atomic);
  if (!translated)
    return {.outcome = translated.outcome};
  if (translated.translation.contiguous_bytes < width)
    return {.outcome = VmAccessOutcome::Malformed};
  return physical_memory_->compare_exchange(
      translated.translation.domain, translated.translation.address, width, expected, desired);
}

Gfx12PageTableTranslator::Gfx12PageTableTranslator(std::shared_ptr<PhysicalMemoryAccess> memory,
                                                   uint64_t page_table_base, Gfx12VmConfig config)
    : memory_(std::move(memory)), page_table_base_(page_table_base), config_(config) {}

VmTranslationResult Gfx12PageTableTranslator::translate(uint64_t address, std::size_t size,
                                                        VmAccessKind access) const {
  if (memory_ == nullptr)
    return {.outcome = VmAccessOutcome::Unavailable, .translation = {}};
  if (size == 0 || size - 1 > std::numeric_limits<uint64_t>::max() - address)
    return {.outcome = VmAccessOutcome::Malformed, .translation = {}};

  const uint64_t virtual_address_mask = gfx12_virtual_address_mask(config_);
  if (address > virtual_address_mask || size - 1 > virtual_address_mask - address)
    return {.outcome = VmAccessOutcome::Malformed, .translation = {}};

  const uint64_t pte_address_mask = gfx12_pte_address_mask(config_);
  const uint64_t pde_address_mask = gfx12_pde_address_mask(config_);
  uint64_t table = page_table_base_ & pte_address_mask;
  util::Logger::vm([&](auto &os) {
    os << std::format("gfx12 walk begin va={:#x} size={} access={} root_raw={:#x} root={:#x} "
                      "va_bits={} levels={} pa_bits={}",
                      address, size, static_cast<unsigned>(access), page_table_base_, table,
                      static_cast<unsigned>(config_.virtual_address_width),
                      config_.page_table_levels(),
                      static_cast<unsigned>(config_.physical_address_width));
  });
  if (table == 0)
    return {.outcome = VmAccessOutcome::Faulted, .translation = {}};
  VmMemoryDomain table_domain = VmMemoryDomain::Local;

  const std::size_t first_level = kGfx12PageTableShifts.size() - config_.page_table_levels();
  for (const uint32_t shift : std::span(kGfx12PageTableShifts).subspan(first_level)) {
    const uint64_t index = (address >> shift) & 0x1ff;
    const uint64_t entry_address = table + index * sizeof(uint64_t);
    std::array<std::byte, sizeof(uint64_t)> raw_entry{};
    const VmAccessOutcome read = memory_->read(table_domain, entry_address, raw_entry);
    if (read != VmAccessOutcome::Complete) {
      util::Logger::vm([&](auto &os) {
        os << std::format("gfx12 walk va={:#x} shift={} index={:#x} table_domain={} "
                          "entry_address={:#x} read_outcome={}",
                          address, shift, index, static_cast<unsigned>(table_domain), entry_address,
                          static_cast<unsigned>(read));
      });
      return {.outcome = read, .translation = {}};
    }

    const uint64_t entry = std::bit_cast<uint64_t>(raw_entry);
    util::Logger::vm([&](auto &os) {
      os << std::format("gfx12 walk va={:#x} shift={} index={:#x} table_domain={} "
                        "entry_address={:#x} entry={:#018x} valid={} system={} p={}",
                        address, shift, index, static_cast<unsigned>(table_domain), entry_address,
                        entry, (entry & kGfx12PteValid) != 0, (entry & kGfx12PteSystem) != 0,
                        (entry & kGfx12PdePte) != 0);
    });
    if ((entry & kGfx12PteValid) == 0)
      return {.outcome = VmAccessOutcome::Faulted, .translation = {}};

    const VmMemoryDomain entry_domain =
        (entry & kGfx12PteSystem) != 0 ? VmMemoryDomain::System : VmMemoryDomain::Local;
    if (shift == 12 || (entry & kGfx12PdePte) != 0) {
      const uint64_t page_bytes = uint64_t{1} << shift;
      const uint64_t page_offset = address & (page_bytes - 1);
      const VmPermissions permissions{
          .readable = (entry & kGfx12PteReadable) != 0,
          .writable = (entry & kGfx12PteWriteable) != 0,
          .executable = (entry & kGfx12PteExecutable) != 0,
      };
      if (!permissions.allows(access))
        return {.outcome = VmAccessOutcome::Faulted, .translation = {}};
      const uint64_t physical_address = (entry & pte_address_mask) + page_offset;
      util::Logger::vm([&](auto &os) {
        os << std::format("gfx12 walk complete va={:#x} shift={} domain={} backing={:#x} span={}",
                          address, shift, static_cast<unsigned>(entry_domain), physical_address,
                          page_bytes - page_offset);
      });
      return {
          .outcome = VmAccessOutcome::Complete,
          .translation =
              {
                  .domain = entry_domain,
                  .address = physical_address,
                  .contiguous_bytes = page_bytes - page_offset,
                  .mtype = gfx12_mtype(entry),
                  .permissions = permissions,
              },
      };
    }

    table = entry & pde_address_mask;
    if (table == 0)
      return {.outcome = VmAccessOutcome::Malformed, .translation = {}};
    table_domain = entry_domain;
  }
  return {.outcome = VmAccessOutcome::Malformed, .translation = {}};
}

Gfx12GartTranslator::Gfx12GartTranslator(PhysicalMemoryAccess &memory, uint64_t page_table_base,
                                         uint64_t aperture_start, uint64_t aperture_end,
                                         Gfx12VmConfig config)
    : memory_(&memory), page_table_base_(page_table_base), aperture_start_(aperture_start),
      aperture_end_(aperture_end), config_(config) {}

Gfx12GartTranslator::Gfx12GartTranslator(std::shared_ptr<PhysicalMemoryAccess> memory,
                                         uint64_t page_table_base, uint64_t aperture_start,
                                         uint64_t aperture_end, Gfx12VmConfig config)
    : retained_memory_(std::move(memory)), memory_(retained_memory_.get()),
      page_table_base_(page_table_base), aperture_start_(aperture_start),
      aperture_end_(aperture_end), config_(config) {}

VmTranslationResult Gfx12GartTranslator::translate(uint64_t address, std::size_t size,
                                                   VmAccessKind access) const {
  if (memory_ == nullptr)
    return {.outcome = VmAccessOutcome::Unavailable, .translation = {}};
  if (size == 0 || size - 1 > std::numeric_limits<uint64_t>::max() - address)
    return {.outcome = VmAccessOutcome::Malformed, .translation = {}};

  constexpr uint64_t kPageBytes = 4096;
  const uint64_t aperture_end_page = aperture_end_ / kPageBytes;
  const bool in_aperture = aperture_start_ != 0 && address >= aperture_start_ &&
                           address / kPageBytes <= aperture_end_page;
  if (!in_aperture) {
    uint64_t contiguous_bytes = size;
    if (aperture_start_ != 0 && address < aperture_start_)
      contiguous_bytes = std::min<uint64_t>(contiguous_bytes, aperture_start_ - address);
    return {
        .outcome = VmAccessOutcome::Complete,
        .translation =
            {
                .domain = VmMemoryDomain::Local,
                .address = address,
                .contiguous_bytes = contiguous_bytes,
                .mtype = Mtype::RW,
                .permissions = {.readable = true, .writable = true, .executable = true},
            },
    };
  }

  const uint64_t pte_address_mask = gfx12_pte_address_mask(config_);
  const uint64_t table = page_table_base_ & pte_address_mask;
  if (table == 0)
    return {.outcome = VmAccessOutcome::Faulted, .translation = {}};
  const uint64_t page = (address - aperture_start_) / kPageBytes;
  if (page > (std::numeric_limits<uint64_t>::max() - table) / sizeof(uint64_t))
    return {.outcome = VmAccessOutcome::Malformed, .translation = {}};

  std::array<std::byte, sizeof(uint64_t)> raw_entry{};
  const VmAccessOutcome read =
      memory_->read(VmMemoryDomain::Local, table + page * sizeof(uint64_t), raw_entry);
  if (read != VmAccessOutcome::Complete)
    return {.outcome = read, .translation = {}};
  const uint64_t entry = std::bit_cast<uint64_t>(raw_entry);
  if ((entry & (kGfx12PteValid | kGfx12PteSystem)) != (kGfx12PteValid | kGfx12PteSystem)) {
    return {.outcome = VmAccessOutcome::Faulted, .translation = {}};
  }

  const VmPermissions permissions{.readable = true, .writable = true, .executable = true};
  if (!permissions.allows(access))
    return {.outcome = VmAccessOutcome::Faulted, .translation = {}};
  const uint64_t page_offset = address & (kPageBytes - 1);
  return {
      .outcome = VmAccessOutcome::Complete,
      .translation =
          {
              .domain = VmMemoryDomain::System,
              .address = (entry & pte_address_mask) + page_offset,
              .contiguous_bytes = kPageBytes - page_offset,
              .mtype = gfx12_mtype(entry),
              .permissions = permissions,
          },
  };
}

void GpuVm::set_memory(GpuMemory *memory) {
  std::lock_guard lock(mutex_);
  assert(vmid_handles_.empty() && "GPU memory cannot change while address spaces are registered");
  memory_ = memory;
}

AddressSpaceHandle GpuVm::allocate_locked(Binding binding) {
  uint32_t slot_index = 0;
  if (free_slots_.empty()) {
    slot_index = static_cast<uint32_t>(slots_.size());
    slots_.push_back({});
  } else {
    slot_index = free_slots_.back();
    free_slots_.pop_back();
  }
  Slot &slot = slots_[slot_index];
  slot.binding = std::move(binding);
  return {.slot = slot_index, .generation = slot.generation};
}

GpuVm::Binding *GpuVm::find_locked(AddressSpaceHandle handle) {
  if (!handle || handle.slot >= slots_.size())
    return nullptr;
  Slot &slot = slots_[handle.slot];
  return slot.generation == handle.generation && slot.binding ? &*slot.binding : nullptr;
}

const GpuVm::Binding *GpuVm::find_locked(AddressSpaceHandle handle) const {
  if (!handle || handle.slot >= slots_.size())
    return nullptr;
  const Slot &slot = slots_[handle.slot];
  return slot.generation == handle.generation && slot.binding ? &*slot.binding : nullptr;
}

AddressSpaceHandle GpuVm::register_legacy(uint32_t vmid) {
  std::lock_guard lock(mutex_);
  if (memory_ == nullptr || vmid_handles_.contains(vmid))
    return {};
  auto translator = std::make_shared<LegacyGpuMemoryTranslator>(*memory_, vmid);
  auto physical_memory = std::make_shared<LegacyGpuMemoryBacking>(*memory_, vmid);
  AddressSpaceHandle handle =
      allocate_locked({.vmid = vmid,
                       .translation_epoch = 1,
                       .queue_references = 0,
                       .translator = std::move(translator),
                       .physical_memory = std::move(physical_memory),
                       .fault_reporter =
                           [memory = memory_, vmid](uint64_t address, VmAccessKind) {
                             memory->report_vm_fault(vmid, address);
                           },
                       .legacy_compatibility = true,
                       .device_gart = false});
  vmid_handles_.emplace(vmid, handle);
  return handle;
}

AddressSpaceHandle GpuVm::register_translated(uint32_t vmid,
                                              std::shared_ptr<AddressSpaceTranslator> translator,
                                              std::shared_ptr<PhysicalMemoryAccess> memory) {
  if (translator == nullptr || memory == nullptr)
    return {};
  std::lock_guard lock(mutex_);
  if (vmid_handles_.contains(vmid))
    return {};
  AddressSpaceHandle handle = allocate_locked({.vmid = vmid,
                                               .translation_epoch = 1,
                                               .queue_references = 0,
                                               .translator = std::move(translator),
                                               .physical_memory = std::move(memory),
                                               .fault_reporter = {},
                                               .legacy_compatibility = false,
                                               .device_gart = false});
  vmid_handles_.emplace(vmid, handle);
  return handle;
}

AddressSpaceHandle
GpuVm::register_gfx12_address_space(uint32_t vmid, uint64_t page_table_base,
                                    std::shared_ptr<PhysicalMemoryAccess> memory) {
  if (page_table_base == 0 || memory == nullptr)
    return {};
  auto translator =
      std::make_shared<Gfx12PageTableTranslator>(memory, page_table_base, gfx12_config_);
  return register_translated(vmid, std::move(translator), std::move(memory));
}

bool GpuVm::replace_translated(AddressSpaceHandle handle,
                               std::shared_ptr<AddressSpaceTranslator> translator,
                               std::shared_ptr<PhysicalMemoryAccess> memory) {
  if (translator == nullptr || memory == nullptr)
    return false;
  std::lock_guard lock(mutex_);
  Binding *binding = find_locked(handle);
  if (binding == nullptr || binding->translator == nullptr || binding->legacy_compatibility)
    return false;
  binding->translator = std::move(translator);
  binding->physical_memory = std::move(memory);
  ++binding->translation_epoch;
  if (binding->translation_epoch == 0)
    ++binding->translation_epoch;
  return true;
}

bool GpuVm::replace_gfx12_address_space_root(AddressSpaceHandle handle, uint64_t page_table_base,
                                             std::shared_ptr<PhysicalMemoryAccess> memory) {
  if (page_table_base == 0 || memory == nullptr)
    return false;
  auto translator =
      std::make_shared<Gfx12PageTableTranslator>(memory, page_table_base, gfx12_config_);
  return replace_translated(handle, std::move(translator), std::move(memory));
}

AddressSpaceHandle GpuVm::initialize_gart_address_space() {
  std::lock_guard lock(mutex_);
  if (find_locked(gart_address_space_) != nullptr)
    return gart_address_space_;
  if (vmid_handles_.contains(0))
    return {};

  gart_address_space_ = allocate_locked({.vmid = 0,
                                         .translation_epoch = 1,
                                         .queue_references = 0,
                                         .translator = nullptr,
                                         .physical_memory = nullptr,
                                         .fault_reporter = {},
                                         .legacy_compatibility = false,
                                         .device_gart = true});
  vmid_handles_.emplace(0, gart_address_space_);
  return gart_address_space_;
}

bool GpuVm::publish_gart(const GartConfig &config, std::shared_ptr<PhysicalMemoryAccess> memory) {
  if (memory == nullptr)
    return false;
  auto translator = std::make_shared<Gfx12GartTranslator>(
      memory, config.page_table_base, config.aperture_start, config.aperture_end, gfx12_config_);

  std::lock_guard lock(mutex_);
  Binding *binding = find_locked(gart_address_space_);
  if (binding == nullptr || !binding->device_gart)
    return false;
  binding->translator = std::move(translator);
  binding->physical_memory = std::move(memory);
  ++binding->translation_epoch;
  if (binding->translation_epoch == 0)
    ++binding->translation_epoch;
  return true;
}

bool GpuVm::clear_gart_binding() {
  std::lock_guard lock(mutex_);
  Binding *binding = find_locked(gart_address_space_);
  if (binding == nullptr || !binding->device_gart || binding->queue_references != 0)
    return false;
  binding->translator.reset();
  binding->physical_memory.reset();
  ++binding->translation_epoch;
  if (binding->translation_epoch == 0)
    ++binding->translation_epoch;
  return true;
}

AddressSpaceHandle GpuVm::gart_address_space() const {
  std::lock_guard lock(mutex_);
  return find_locked(gart_address_space_) != nullptr ? gart_address_space_ : AddressSpaceHandle{};
}

bool GpuVm::invalidate(AddressSpaceHandle handle) {
  std::lock_guard lock(mutex_);
  Binding *binding = find_locked(handle);
  if (binding == nullptr)
    return false;
  ++binding->translation_epoch;
  if (binding->translation_epoch == 0)
    ++binding->translation_epoch;
  return true;
}

std::optional<AddressSpaceInfo> GpuVm::retain_queue_address_space(AddressSpaceHandle handle) {
  std::lock_guard lock(mutex_);
  Binding *binding = find_locked(handle);
  if (binding == nullptr || binding->queue_references == UINT32_MAX)
    return std::nullopt;
  ++binding->queue_references;
  return AddressSpaceInfo{.vmid = binding->vmid,
                          .translation_epoch = binding->translation_epoch,
                          .queue_references = binding->queue_references,
                          .external = !binding->legacy_compatibility,
                          .ready = binding->translator != nullptr &&
                                   binding->physical_memory != nullptr};
}

bool GpuVm::retain_queue(AddressSpaceHandle handle) {
  return retain_queue_address_space(handle).has_value();
}

bool GpuVm::release_queue(AddressSpaceHandle handle) {
  std::lock_guard lock(mutex_);
  Binding *binding = find_locked(handle);
  if (binding == nullptr || binding->queue_references == 0)
    return false;
  --binding->queue_references;
  return true;
}

bool GpuVm::unregister_address_space(AddressSpaceHandle handle) {
  std::lock_guard lock(mutex_);
  Binding *binding = find_locked(handle);
  if (binding == nullptr || binding->queue_references != 0 || binding->device_gart)
    return false;
  vmid_handles_.erase(binding->vmid);
  Slot &slot = slots_[handle.slot];
  slot.binding.reset();
  ++slot.generation;
  if (slot.generation == 0)
    ++slot.generation;
  free_slots_.push_back(handle.slot);
  return true;
}

std::optional<GpuVmAccess> GpuVm::snapshot(AddressSpaceHandle handle) const {
  std::lock_guard lock(mutex_);
  const Binding *binding = find_locked(handle);
  if (binding == nullptr)
    return std::nullopt;
  return GpuVmAccess(
      handle,
      {.vmid = binding->vmid,
       .translation_epoch = binding->translation_epoch,
       .queue_references = binding->queue_references,
       .external = !binding->legacy_compatibility,
       .ready = binding->translator != nullptr && binding->physical_memory != nullptr},
      binding->translator, binding->physical_memory, binding->fault_reporter);
}

VmTranslationResult GpuVm::translate(AddressSpaceHandle handle, uint64_t address, std::size_t size,
                                     VmAccessKind access) const {
  const std::optional<GpuVmAccess> access_snapshot = snapshot(handle);
  return access_snapshot
             ? access_snapshot->translate(address, size, access)
             : VmTranslationResult{.outcome = VmAccessOutcome::Faulted, .translation = {}};
}

VmAccessOutcome GpuVm::probe(AddressSpaceHandle handle, uint64_t address, std::size_t size,
                             VmAccessKind access) const {
  const std::optional<GpuVmAccess> access_snapshot = snapshot(handle);
  return access_snapshot ? access_snapshot->probe(address, size, access) : VmAccessOutcome::Faulted;
}

VmAccessOutcome GpuVm::read(AddressSpaceHandle handle, uint64_t address,
                            std::span<std::byte> bytes) const {
  const std::optional<GpuVmAccess> access_snapshot = snapshot(handle);
  return access_snapshot ? access_snapshot->read(address, bytes) : VmAccessOutcome::Faulted;
}

VmAccessOutcome GpuVm::write(AddressSpaceHandle handle, uint64_t address,
                             std::span<const std::byte> bytes) {
  const std::optional<GpuVmAccess> access_snapshot = snapshot(handle);
  return access_snapshot ? access_snapshot->write(address, bytes) : VmAccessOutcome::Faulted;
}

AtomicLoadResult GpuVm::atomic_load(AddressSpaceHandle handle, uint64_t address,
                                    uint32_t width) const {
  const std::optional<GpuVmAccess> access_snapshot = snapshot(handle);
  return access_snapshot ? access_snapshot->atomic_load(address, width)
                         : AtomicLoadResult{.outcome = VmAccessOutcome::Faulted, .value = 0};
}

VmAccessOutcome GpuVm::atomic_store(AddressSpaceHandle handle, uint64_t address, uint32_t width,
                                    uint64_t value) {
  const std::optional<GpuVmAccess> access_snapshot = snapshot(handle);
  return access_snapshot ? access_snapshot->atomic_store(address, width, value)
                         : VmAccessOutcome::Faulted;
}

AtomicCompareExchangeResult GpuVm::compare_exchange(AddressSpaceHandle handle, uint64_t address,
                                                    uint32_t width, uint64_t expected,
                                                    uint64_t desired) {
  const std::optional<GpuVmAccess> access_snapshot = snapshot(handle);
  return access_snapshot ? access_snapshot->compare_exchange(address, width, expected, desired)
                         : AtomicCompareExchangeResult{.outcome = VmAccessOutcome::Faulted};
}

std::optional<AddressSpaceInfo> GpuVm::lookup(AddressSpaceHandle handle) const {
  std::lock_guard lock(mutex_);
  const Binding *binding = find_locked(handle);
  if (binding == nullptr)
    return std::nullopt;
  return AddressSpaceInfo{.vmid = binding->vmid,
                          .translation_epoch = binding->translation_epoch,
                          .queue_references = binding->queue_references,
                          .external = !binding->legacy_compatibility,
                          .ready = binding->translator != nullptr &&
                                   binding->physical_memory != nullptr};
}

std::optional<AddressSpaceHandle> GpuVm::find_vmid(uint32_t vmid) const {
  std::lock_guard lock(mutex_);
  const auto found = vmid_handles_.find(vmid);
  return found == vmid_handles_.end() ? std::nullopt
                                      : std::optional<AddressSpaceHandle>(found->second);
}

std::size_t GpuVm::active_address_spaces() const {
  std::lock_guard lock(mutex_);
  return vmid_handles_.size();
}

uint64_t GpuVm::reset_epoch() const {
  std::lock_guard lock(mutex_);
  return reset_epoch_;
}

bool GpuVm::reset() {
  std::lock_guard lock(mutex_);
  const bool retained = std::ranges::any_of(
      slots_, [](const Slot &slot) { return slot.binding && slot.binding->queue_references != 0; });
  if (retained)
    return false;
  for (uint32_t index = 0; index < slots_.size(); ++index) {
    Slot &slot = slots_[index];
    if (!slot.binding)
      continue;
    slot.binding.reset();
    ++slot.generation;
    if (slot.generation == 0)
      ++slot.generation;
    free_slots_.push_back(index);
  }
  vmid_handles_.clear();
  gart_address_space_ = {};
  ++reset_epoch_;
  if (reset_epoch_ == 0)
    ++reset_epoch_;
  return true;
}

} // namespace rocjitsu::amdgpu
