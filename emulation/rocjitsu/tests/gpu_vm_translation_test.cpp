// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "rocjitsu/vm/amdgpu/gpu_vm.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <shared_mutex>
#include <span>
#include <sys/mman.h>
#include <unistd.h>
#include <vector>

namespace rocjitsu::amdgpu {
namespace {

class TestPhysicalMemory final : public PhysicalMemoryAccess {
public:
  VmAccessOutcome read(VmMemoryDomain domain, uint64_t address,
                       std::span<std::byte> bytes) override {
    ++read_calls;
    const auto &memory = domain == VmMemoryDomain::System ? system : local;
    for (std::size_t index = 0; index < bytes.size(); ++index) {
      const auto found = memory.find(address + index);
      if (found == memory.end())
        return VmAccessOutcome::Unavailable;
      bytes[index] = found->second;
    }
    return VmAccessOutcome::Complete;
  }

  VmAccessOutcome write(VmMemoryDomain domain, uint64_t address,
                        std::span<const std::byte> bytes) override {
    auto &memory = domain == VmMemoryDomain::System ? system : local;
    for (std::size_t index = 0; index < bytes.size(); ++index)
      memory[address + index] = bytes[index];
    return VmAccessOutcome::Complete;
  }

  void store_qword(VmMemoryDomain domain, uint64_t address, uint64_t value) {
    const auto bytes = std::bit_cast<std::array<std::byte, sizeof(value)>>(value);
    EXPECT_EQ(write(domain, address, bytes), VmAccessOutcome::Complete);
  }

  std::map<uint64_t, std::byte> system;
  std::map<uint64_t, std::byte> local;
  std::size_t read_calls = 0;
};

class RecordingFaultReporter final : public MemoryFaultReporter {
public:
  void report_memory_fault(uint32_t vmid, uint64_t address, MemoryFaultCause cause) override {
    vmids.push_back(vmid);
    addresses.push_back(address);
    causes.push_back(cause);
  }

  std::vector<uint32_t> vmids;
  std::vector<uint64_t> addresses;
  std::vector<MemoryFaultCause> causes;
};

class HostPage {
public:
  HostPage() {
    void *mapping = mmap(nullptr, KfdProcess::kPageSize, PROT_READ | PROT_WRITE,
                         MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    data_ = mapping == MAP_FAILED ? nullptr : static_cast<uint8_t *>(mapping);
  }
  HostPage(const HostPage &) = delete;
  HostPage &operator=(const HostPage &) = delete;
  ~HostPage() {
    if (data_ != nullptr)
      munmap(data_, KfdProcess::kPageSize);
  }

  uint8_t *data() const { return data_; }

private:
  uint8_t *data_ = nullptr;
};

constexpr uint64_t kValid = uint64_t{1} << 0;
constexpr uint64_t kSystem = uint64_t{1} << 1;
constexpr uint64_t kExecutable = uint64_t{1} << 4;
constexpr uint64_t kReadable = uint64_t{1} << 5;
constexpr uint64_t kWriteable = uint64_t{1} << 6;
constexpr uint64_t kPdePte = uint64_t{1} << 63;
constexpr std::array<uint64_t, 4> kLowPageTables = {0x2000, 0x3000, 0x4000, 0x5000};
constexpr std::array<uint64_t, 3> kGfx120LowPageTables = {0x2000, 0x3000, 0x4000};

void map_4k(TestPhysicalMemory &memory, uint64_t root, uint64_t virtual_address,
            uint64_t physical_address, uint64_t leaf_flags,
            const std::array<uint64_t, 4> &tables = kLowPageTables) {
  constexpr std::array<uint32_t, 4> shifts = {48, 39, 30, 21};
  uint64_t table = root;
  for (std::size_t level = 0; level < tables.size(); ++level) {
    const uint64_t index = (virtual_address >> shifts[level]) & 0x1ff;
    memory.store_qword(VmMemoryDomain::Local, table + index * sizeof(uint64_t),
                       tables[level] | kValid);
    table = tables[level];
  }
  const uint64_t leaf_index = (virtual_address >> 12) & 0x1ff;
  memory.store_qword(VmMemoryDomain::Local, table + leaf_index * sizeof(uint64_t),
                     physical_address | leaf_flags);
}

void map_gfx120_4k(TestPhysicalMemory &memory, uint64_t root, uint64_t virtual_address,
                   uint64_t physical_address, uint64_t leaf_flags,
                   const std::array<uint64_t, 3> &tables = kGfx120LowPageTables) {
  constexpr std::array<uint32_t, 3> shifts = {39, 30, 21};
  uint64_t table = root;
  for (std::size_t level = 0; level < tables.size(); ++level) {
    const uint64_t index = (virtual_address >> shifts[level]) & 0x1ff;
    memory.store_qword(VmMemoryDomain::Local, table + index * sizeof(uint64_t),
                       tables[level] | kValid);
    table = tables[level];
  }
  const uint64_t leaf_index = (virtual_address >> 12) & 0x1ff;
  memory.store_qword(VmMemoryDomain::Local, table + leaf_index * sizeof(uint64_t),
                     physical_address | leaf_flags);
}

TEST(GpuVmTranslation, Gfx121WalkPreserves52BitLeafPhysicalAddress) {
  auto physical = std::make_shared<TestPhysicalMemory>();
  constexpr uint64_t root = 0x1000;
  constexpr uint64_t virtual_address = 0x5123;
  constexpr uint64_t physical_page = 0x0001'0000'0000'9000;
  map_4k(*physical, root, virtual_address, physical_page,
         kValid | kSystem | kReadable | kWriteable);
  physical->system[physical_page + 0x123] = std::byte{0x5a};
  physical->system[0x9000 + 0x123] = std::byte{0xa5};
  Gfx12PageTableTranslator translator(physical, root);

  const VmTranslationResult translated =
      translator.translate(virtual_address, 1, VmAccessKind::Read);

  ASSERT_TRUE(translated);
  EXPECT_EQ(translated.translation.address, physical_page + 0x123);
  std::array<std::byte, 1> value{};
  EXPECT_EQ(read_translated(translator, *physical, virtual_address, value),
            VmAccessOutcome::Complete);
  EXPECT_EQ(value[0], std::byte{0x5a});
}

TEST(GpuVmTranslation, Gfx121WalkPreserves52BitIntermediateTableAddress) {
  auto physical = std::make_shared<TestPhysicalMemory>();
  constexpr uint64_t root = 0x1000;
  constexpr uint64_t virtual_address = 0x7000;
  constexpr uint64_t physical_page = 0x9000;
  constexpr std::array<uint64_t, 4> high_page_tables = {
      0x0001'0000'0000'2000,
      0x0001'0000'0000'3000,
      0x0001'0000'0000'4000,
      0x0001'0000'0000'5000,
  };
  map_4k(*physical, root, virtual_address, physical_page, kValid | kSystem | kReadable | kWriteable,
         high_page_tables);
  Gfx12PageTableTranslator translator(physical, root);

  const VmTranslationResult translated =
      translator.translate(virtual_address, 1, VmAccessKind::Read);

  ASSERT_TRUE(translated);
  EXPECT_EQ(translated.translation.address, physical_page);
}

TEST(GpuVmTranslation, Gfx121WalkPreserves52BitRootAddress) {
  auto physical = std::make_shared<TestPhysicalMemory>();
  constexpr uint64_t root = 0x0001'0000'0000'1000;
  constexpr uint64_t virtual_address = 0x7000;
  constexpr uint64_t physical_page = 0x9000;
  map_4k(*physical, root, virtual_address, physical_page,
         kValid | kSystem | kReadable | kWriteable);
  Gfx12PageTableTranslator translator(physical, root);

  const VmTranslationResult translated =
      translator.translate(virtual_address, 1, VmAccessKind::Read);

  ASSERT_TRUE(translated);
  EXPECT_EQ(translated.translation.address, physical_page);
}

TEST(GpuVmTranslation, Gfx120ProfileUses48BitPhysicalAddressMask) {
  auto physical = std::make_shared<TestPhysicalMemory>();
  constexpr uint64_t root = 0x1000;
  constexpr uint64_t virtual_address = 0x5123;
  constexpr uint64_t encoded_physical_page = 0x0001'0000'0000'9000;
  map_gfx120_4k(*physical, root, virtual_address, encoded_physical_page,
                kValid | kSystem | kReadable | kWriteable);
  Gfx12PageTableTranslator translator(physical, root, Gfx12VmConfig::gfx12_0());

  const VmTranslationResult translated =
      translator.translate(virtual_address, 1, VmAccessKind::Read);

  ASSERT_TRUE(translated);
  EXPECT_EQ(translated.translation.address, 0x9000u + 0x123u);
}

TEST(GpuVmTranslation, Gfx120ProfileUsesFourLevelPageTable) {
  auto physical = std::make_shared<TestPhysicalMemory>();
  constexpr uint64_t root = 0x1000;
  constexpr uint64_t virtual_address = (uint64_t{1} << 39) | 0x5123;
  constexpr uint64_t physical_page = 0x9000;
  map_gfx120_4k(*physical, root, virtual_address, physical_page, kValid | kSystem | kReadable);
  Gfx12PageTableTranslator translator(physical, root, Gfx12VmConfig::gfx12_0());

  const VmTranslationResult translated =
      translator.translate(virtual_address, 1, VmAccessKind::Read);

  ASSERT_TRUE(translated);
  EXPECT_EQ(translated.translation.address, physical_page + 0x123);
}

TEST(GpuVmTranslation, Gfx12ProfilesRejectAddressesOutsideTheirVirtualAddressWidth) {
  auto gfx120_physical = std::make_shared<TestPhysicalMemory>();
  constexpr uint64_t root = 0x1000;
  map_gfx120_4k(*gfx120_physical, root, 0, 0x9000, kValid | kSystem | kReadable);
  Gfx12PageTableTranslator gfx120(gfx120_physical, root, Gfx12VmConfig::gfx12_0());

  EXPECT_EQ(gfx120.translate(uint64_t{1} << 48, 1, VmAccessKind::Read).outcome,
            VmAccessOutcome::Malformed);
  EXPECT_EQ(gfx120.translate((uint64_t{1} << 48) - 1, 2, VmAccessKind::Read).outcome,
            VmAccessOutcome::Malformed);
  EXPECT_EQ(gfx120_physical->read_calls, 0u);

  auto gfx121_physical = std::make_shared<TestPhysicalMemory>();
  map_4k(*gfx121_physical, root, 0, 0xa000, kValid | kSystem | kReadable);
  Gfx12PageTableTranslator gfx121(gfx121_physical, root, Gfx12VmConfig::gfx12_1());

  // Bits above the 57-bit aperture must not alias an otherwise valid low mapping.
  EXPECT_EQ(gfx121.translate(uint64_t{1} << 57, 1, VmAccessKind::Read).outcome,
            VmAccessOutcome::Malformed);
  EXPECT_EQ(gfx121.translate((uint64_t{1} << 57) - 1, 2, VmAccessKind::Read).outcome,
            VmAccessOutcome::Malformed);
  EXPECT_EQ(gfx121_physical->read_calls, 0u);
}

TEST(GpuVmTranslation, Gfx12WalkSeparatesTranslationFromPhysicalAccess) {
  auto physical = std::make_shared<TestPhysicalMemory>();
  constexpr uint64_t root = 0x1000;
  constexpr uint64_t virtual_address = 0x5123;
  constexpr uint64_t physical_page = 0x9000;
  map_4k(*physical, root, virtual_address, physical_page,
         kValid | kSystem | kReadable | kWriteable | kExecutable);
  physical->system[physical_page + 0x123] = std::byte{0x5a};

  auto translator = std::make_shared<Gfx12PageTableTranslator>(physical, root);
  const VmTranslationResult translated =
      translator->translate(virtual_address, 1, VmAccessKind::Read);

  ASSERT_TRUE(translated);
  EXPECT_EQ(translated.translation.domain, VmMemoryDomain::System);
  EXPECT_EQ(translated.translation.address, physical_page + 0x123);
  EXPECT_EQ(translated.translation.contiguous_bytes, 0x1000u - 0x123u);
  EXPECT_TRUE(translated.translation.permissions.readable);
  EXPECT_TRUE(translated.translation.permissions.writable);
  EXPECT_TRUE(translated.translation.permissions.executable);
}

TEST(GpuVmTranslation, Gfx12WalkEnforcesLeafPermissions) {
  auto physical = std::make_shared<TestPhysicalMemory>();
  constexpr uint64_t root = 0x1000;
  constexpr uint64_t virtual_address = 0x6000;
  map_4k(*physical, root, virtual_address, 0xa000, kValid | kSystem | kReadable);
  Gfx12PageTableTranslator translator(physical, root);

  EXPECT_EQ(translator.translate(virtual_address, 1, VmAccessKind::Read).outcome,
            VmAccessOutcome::Complete);
  EXPECT_EQ(translator.translate(virtual_address, 1, VmAccessKind::Write).outcome,
            VmAccessOutcome::Faulted);
  EXPECT_EQ(translator.translate(virtual_address, 1, VmAccessKind::Execute).outcome,
            VmAccessOutcome::Faulted);
  EXPECT_EQ(translator.translate(virtual_address, 1, VmAccessKind::Atomic).outcome,
            VmAccessOutcome::Faulted);
}

TEST(GpuVmTranslation, RangeProbeRejectsAnUnmappedTrailingPage) {
  auto physical = std::make_shared<TestPhysicalMemory>();
  constexpr uint64_t root = 0x1000;
  constexpr uint64_t first_page = 0x6000;
  constexpr uint64_t physical_page = 0x9000;
  constexpr uint64_t straddle = first_page + 4092;
  map_4k(*physical, root, first_page, physical_page, kValid | kSystem | kReadable | kWriteable);
  const uint64_t second_leaf_index = ((first_page + 4096) >> 12) & 0x1ff;
  physical->store_qword(VmMemoryDomain::Local,
                        kLowPageTables.back() + second_leaf_index * sizeof(uint64_t), 0);

  GpuVm gpu_vm;
  auto translator = std::make_shared<Gfx12PageTableTranslator>(physical, root);
  const AddressSpaceHandle handle = gpu_vm.register_translated(7, translator, physical);
  ASSERT_TRUE(handle);
  const auto access = gpu_vm.snapshot(handle);
  ASSERT_TRUE(access);

  EXPECT_EQ(access->probe(straddle, sizeof(uint32_t), VmAccessKind::Read),
            VmAccessOutcome::Complete);
  EXPECT_EQ(access->probe(straddle, sizeof(uint64_t), VmAccessKind::Read),
            VmAccessOutcome::Faulted);
}

TEST(GpuVmTranslation, Gfx12PdeAsPtePreservesEncodedBaseAndReportsLargePageSpan) {
  auto physical = std::make_shared<TestPhysicalMemory>();
  constexpr uint64_t root = 0x1000;
  constexpr uint64_t virtual_address = 0x201234;
  physical->store_qword(VmMemoryDomain::Local, root, 0x2000 | kValid);
  physical->store_qword(VmMemoryDomain::Local, 0x2000, 0x3000 | kValid);
  physical->store_qword(VmMemoryDomain::Local, 0x3000, 0x4000 | kValid);
  constexpr uint64_t physical_page = 0x0001'0000'0082'3000;
  physical->store_qword(VmMemoryDomain::Local, 0x4000 + sizeof(uint64_t),
                        physical_page | kPdePte | kValid | kReadable | kWriteable);
  Gfx12PageTableTranslator translator(physical, root);

  const VmTranslationResult translated =
      translator.translate(virtual_address, 16, VmAccessKind::Write);

  ASSERT_TRUE(translated);
  EXPECT_EQ(translated.translation.domain, VmMemoryDomain::Local);
  EXPECT_EQ(translated.translation.address, physical_page + 0x1234u);
  EXPECT_EQ(translated.translation.contiguous_bytes, 0x200000u - 0x1234u);
}

TEST(GpuVmTranslation, TypedFailuresDistinguishMissingBackingFromInvalidMapping) {
  auto missing = std::make_shared<TestPhysicalMemory>();
  Gfx12PageTableTranslator unavailable(missing, 0x1000);
  EXPECT_EQ(unavailable.translate(0, 1, VmAccessKind::Read).outcome, VmAccessOutcome::Unavailable);

  auto invalid = std::make_shared<TestPhysicalMemory>();
  invalid->store_qword(VmMemoryDomain::Local, 0x1000, 0);
  Gfx12PageTableTranslator faulted(invalid, 0x1000);
  EXPECT_EQ(faulted.translate(0, 1, VmAccessKind::Read).outcome, VmAccessOutcome::Faulted);
  EXPECT_EQ(faulted.translate(UINT64_MAX, 2, VmAccessKind::Read).outcome,
            VmAccessOutcome::Malformed);
}

TEST(GpuVmTranslation, Gfx12GartRoutesOnlyTheConfiguredApertureThroughSystemMemory) {
  auto physical = std::make_shared<TestPhysicalMemory>();
  constexpr uint64_t table = 0x1000;
  constexpr uint64_t aperture_start = 0x10000;
  constexpr uint64_t system_page = 0x0001'0000'0000'9000;
  physical->store_qword(VmMemoryDomain::Local, table, system_page | kValid | kSystem);
  physical->local[aperture_start - 1] = std::byte{0xaa};
  physical->system[system_page] = std::byte{0xbb};
  Gfx12GartTranslator translator(physical, table, aperture_start, aperture_start + 0xfff);

  std::array<std::byte, 2> value{};
  EXPECT_EQ(read_translated(translator, *physical, aperture_start - 1, value),
            VmAccessOutcome::Complete);
  EXPECT_EQ(value[0], std::byte{0xaa});
  EXPECT_EQ(value[1], std::byte{0xbb});

  const VmTranslationResult local = translator.translate(aperture_start - 1, 2, VmAccessKind::Read);
  ASSERT_TRUE(local);
  EXPECT_EQ(local.translation.domain, VmMemoryDomain::Local);
  EXPECT_EQ(local.translation.address, aperture_start - 1);
  EXPECT_EQ(local.translation.contiguous_bytes, 1u);

  const VmTranslationResult system = translator.translate(aperture_start, 1, VmAccessKind::Read);
  ASSERT_TRUE(system);
  EXPECT_EQ(system.translation.domain, VmMemoryDomain::System);
  EXPECT_EQ(system.translation.address, system_page);
  EXPECT_EQ(system.translation.contiguous_bytes, 4096u);
}

TEST(GpuVmTranslation, GpuVmPropagatesConfiguredPhysicalAddressWidthToGart) {
  GpuVm gpu_vm(nullptr, Gfx12VmConfig::gfx12_0());
  auto physical = std::make_shared<TestPhysicalMemory>();
  constexpr uint64_t table = 0x1000;
  constexpr uint64_t aperture = 0x10000;
  constexpr uint64_t encoded_system_page = 0x0001'0000'0000'9000;
  physical->store_qword(VmMemoryDomain::Local, table, encoded_system_page | kValid | kSystem);
  ASSERT_TRUE(gpu_vm.initialize_gart_address_space());
  ASSERT_TRUE(gpu_vm.publish_gart(
      {.page_table_base = table, .aperture_start = aperture, .aperture_end = aperture + 0xfff},
      physical));

  const VmTranslationResult translated =
      gpu_vm.translate(gpu_vm.gart_address_space(), aperture + 0x123, 1, VmAccessKind::Read);

  ASSERT_TRUE(translated);
  EXPECT_EQ(translated.translation.address, 0x9000u + 0x123u);
}

TEST(GpuVmTranslation, Gfx12GartRejectsNonSystemOrMissingApertureEntries) {
  auto physical = std::make_shared<TestPhysicalMemory>();
  constexpr uint64_t table = 0x1000;
  constexpr uint64_t aperture_start = 0x10000;
  physical->store_qword(VmMemoryDomain::Local, table, 0x9000 | kValid);
  Gfx12GartTranslator translator(physical, table, aperture_start, aperture_start + 0xfff);

  EXPECT_EQ(translator.translate(aperture_start, 1, VmAccessKind::Read).outcome,
            VmAccessOutcome::Faulted);
  EXPECT_EQ(translator.translate(UINT64_MAX, 2, VmAccessKind::Read).outcome,
            VmAccessOutcome::Malformed);
}

TEST(GpuVmTranslation, GpuVmUsesTranslatedBindingAndAdvancesEpochOnRootReplacement) {
  GpuMemory compatibility_memory("memory");
  GpuVm gpu_vm(&compatibility_memory);
  auto physical = std::make_shared<TestPhysicalMemory>();
  map_4k(*physical, 0x1000, 0x7000, 0xb000, kValid | kSystem | kReadable | kWriteable);
  physical->system[0xb000] = std::byte{0x11};
  auto first = std::make_shared<Gfx12PageTableTranslator>(physical, 0x1000);
  const AddressSpaceHandle handle = gpu_vm.register_translated(7, first, physical);
  ASSERT_TRUE(handle);
  const uint64_t old_epoch = gpu_vm.lookup(handle)->translation_epoch;

  std::array<std::byte, 1> value{};
  EXPECT_EQ(gpu_vm.read(handle, 0x7000, value), VmAccessOutcome::Complete);
  EXPECT_EQ(value[0], std::byte{0x11});

  map_4k(*physical, 0x6000, 0x7000, 0xc000, kValid | kSystem | kReadable | kWriteable);
  physical->system[0xc000] = std::byte{0x22};
  auto second = std::make_shared<Gfx12PageTableTranslator>(physical, 0x6000);
  ASSERT_TRUE(gpu_vm.replace_translated(handle, second, physical));
  ASSERT_TRUE(gpu_vm.lookup(handle));
  EXPECT_EQ(gpu_vm.lookup(handle)->translation_epoch, old_epoch + 1);
  EXPECT_EQ(gpu_vm.read(handle, 0x7000, value), VmAccessOutcome::Complete);
  EXPECT_EQ(value[0], std::byte{0x22});

  const uint64_t replacement_epoch = gpu_vm.lookup(handle)->translation_epoch;
  EXPECT_TRUE(gpu_vm.invalidate(handle));
  EXPECT_EQ(gpu_vm.lookup(handle)->translation_epoch, replacement_epoch + 1);

  EXPECT_TRUE(gpu_vm.unregister_address_space(handle));
  EXPECT_EQ(gpu_vm.read(handle, 0x7000, value), VmAccessOutcome::Faulted);
  EXPECT_FALSE(gpu_vm.invalidate(handle));
}

TEST(GpuVmTranslation, DeviceGartPublishesOnInvalidateAndRejectsItsHandleAfterReset) {
  GpuVm gpu_vm;
  auto physical = std::make_shared<TestPhysicalMemory>();
  constexpr uint64_t first_table = 0x1000;
  constexpr uint64_t second_table = 0x2000;
  constexpr uint64_t aperture = 0x10000;
  physical->store_qword(VmMemoryDomain::Local, first_table, 0x9000 | kValid | kSystem);
  physical->store_qword(VmMemoryDomain::Local, second_table, 0xa000 | kValid | kSystem);
  physical->system[0x9000] = std::byte{0x11};
  physical->system[0xa000] = std::byte{0x22};

  const AddressSpaceHandle handle = gpu_vm.initialize_gart_address_space();
  ASSERT_TRUE(handle);
  ASSERT_TRUE(gpu_vm.lookup(handle));
  const uint64_t initial_epoch = gpu_vm.lookup(handle)->translation_epoch;
  std::array<std::byte, 1> value{};
  EXPECT_EQ(gpu_vm.read(handle, aperture, value), VmAccessOutcome::Unavailable);

  ASSERT_TRUE(gpu_vm.publish_gart({.page_table_base = first_table,
                                   .aperture_start = aperture,
                                   .aperture_end = aperture + 0xfff},
                                  physical));
  EXPECT_EQ(gpu_vm.gart_address_space(), handle);
  ASSERT_TRUE(gpu_vm.lookup(handle));
  EXPECT_EQ(gpu_vm.lookup(handle)->translation_epoch, initial_epoch + 1);
  EXPECT_EQ(gpu_vm.read(handle, aperture, value), VmAccessOutcome::Complete);
  EXPECT_EQ(value[0], std::byte{0x11});

  ASSERT_TRUE(gpu_vm.publish_gart({.page_table_base = second_table,
                                   .aperture_start = aperture,
                                   .aperture_end = aperture + 0xfff},
                                  physical));
  EXPECT_EQ(gpu_vm.gart_address_space(), handle);
  EXPECT_EQ(gpu_vm.lookup(handle)->translation_epoch, initial_epoch + 2);
  EXPECT_EQ(gpu_vm.read(handle, aperture, value), VmAccessOutcome::Complete);
  EXPECT_EQ(value[0], std::byte{0x22});
  EXPECT_FALSE(gpu_vm.unregister_address_space(handle));

  ASSERT_TRUE(gpu_vm.reset());
  EXPECT_EQ(gpu_vm.read(handle, aperture, value), VmAccessOutcome::Faulted);
  const AddressSpaceHandle replacement = gpu_vm.initialize_gart_address_space();
  ASSERT_TRUE(replacement);
  EXPECT_EQ(replacement.slot, handle.slot);
  EXPECT_NE(replacement.generation, handle.generation);
  EXPECT_EQ(gpu_vm.read(replacement, aperture, value), VmAccessOutcome::Unavailable);
}

TEST(GpuVmTranslation, TranslatedBindingDoesNotDependOnLegacyMemoryOrNonzeroMetadata) {
  GpuVm gpu_vm;
  auto physical = std::make_shared<TestPhysicalMemory>();
  map_4k(*physical, 0x1000, 0x7000, 0xb000, kValid | kSystem | kReadable | kWriteable);
  physical->system[0xb000] = std::byte{0x3c};
  auto translator = std::make_shared<Gfx12PageTableTranslator>(physical, 0x1000);

  const AddressSpaceHandle handle = gpu_vm.register_translated(0, translator, physical);

  ASSERT_TRUE(handle);
  ASSERT_TRUE(gpu_vm.lookup(handle));
  EXPECT_EQ(gpu_vm.lookup(handle)->vmid, 0u);
  std::array<std::byte, 1> value{};
  EXPECT_EQ(gpu_vm.read(handle, 0x7000, value), VmAccessOutcome::Complete);
  EXPECT_EQ(value[0], std::byte{0x3c});
  EXPECT_TRUE(gpu_vm.unregister_address_space(handle));
}

TEST(GpuVmTranslation, LegacyBindingUsesTheSharedVmInterfaceWithoutClaimingPhysicalIdentity) {
  GpuMemory memory("memory");
  KfdProcess::PageTable page_table;
  std::shared_mutex page_table_mutex;
  constexpr uint32_t vmid = 7;
  constexpr uint64_t virtual_address = 0x4123;
  std::array<uint8_t, KfdProcess::kPageSize> backing{};
  backing[0x123] = 0x5a;
  page_table[virtual_address >> KfdProcess::kPageShift] = {backing.data(), Mtype::CC};
  memory.register_process(vmid, &page_table, &page_table_mutex);
  GpuVm gpu_vm(&memory);

  const AddressSpaceHandle handle = gpu_vm.register_legacy(vmid);

  ASSERT_TRUE(handle);
  ASSERT_TRUE(gpu_vm.lookup(handle));
  EXPECT_FALSE(gpu_vm.lookup(handle)->external);
  const VmTranslationResult translated =
      gpu_vm.translate(handle, virtual_address, 1, VmAccessKind::Read);
  ASSERT_TRUE(translated);
  EXPECT_EQ(translated.translation.domain, VmMemoryDomain::Compatibility);
  EXPECT_EQ(translated.translation.address, virtual_address);
  EXPECT_EQ(translated.translation.mtype, Mtype::CC);
  EXPECT_EQ(translated.translation.contiguous_bytes, KfdProcess::kPageSize - 0x123);

  std::array<std::byte, 1> value{};
  EXPECT_EQ(gpu_vm.read(handle, virtual_address, value), VmAccessOutcome::Complete);
  EXPECT_EQ(value[0], std::byte{0x5a});
  value[0] = std::byte{0xa5};
  EXPECT_EQ(gpu_vm.write(handle, virtual_address, value), VmAccessOutcome::Complete);
  EXPECT_EQ(backing[0x123], 0xa5);

  EXPECT_TRUE(gpu_vm.unregister_address_space(handle));
  memory.unregister_process(vmid);
}

TEST(GpuVmTranslation, LegacyBackingRetriesUntilPageTableMappingIsPublished) {
  GpuMemory memory("memory");
  KfdProcess::PageTable page_table;
  std::shared_mutex page_table_mutex;
  constexpr uint32_t vmid = 7;
  constexpr uint64_t virtual_address = 0x4000;
  GpuVm gpu_vm(&memory);
  memory.register_process(vmid, &page_table, &page_table_mutex);
  const AddressSpaceHandle handle = gpu_vm.register_legacy(vmid);
  ASSERT_TRUE(handle);

  std::array<std::byte, 4> value{};
  EXPECT_EQ(gpu_vm.read(handle, virtual_address, value), VmAccessOutcome::Unavailable);
  EXPECT_EQ(gpu_vm.write(handle, virtual_address, value), VmAccessOutcome::Unavailable);

  std::array<uint8_t, KfdProcess::kPageSize> backing{};
  backing[0] = 0x5a;
  {
    std::unique_lock lock(page_table_mutex);
    page_table[virtual_address >> KfdProcess::kPageShift] = {backing.data(), Mtype::CC};
  }

  EXPECT_EQ(gpu_vm.read(handle, virtual_address, value), VmAccessOutcome::Complete);
  EXPECT_EQ(value[0], std::byte{0x5a});
  value[0] = std::byte{0xa5};
  EXPECT_EQ(gpu_vm.write(handle, virtual_address, value), VmAccessOutcome::Complete);
  EXPECT_EQ(backing[0], 0xa5);

  EXPECT_TRUE(gpu_vm.unregister_address_space(handle));
  memory.unregister_process(vmid);
}

TEST(GpuVmTranslation, LegacyBackingPreservesMappedPageClipping) {
  GpuMemory memory("memory");
  KfdProcess::PageTable page_table;
  std::shared_mutex page_table_mutex;
  constexpr uint32_t vmid = 7;
  constexpr uint64_t virtual_address = 0x4000;
  constexpr size_t mapped_bytes = 64;
  std::array<uint8_t, mapped_bytes> backing{};
  std::ranges::fill(backing, uint8_t{0x5a});
  page_table[virtual_address >> KfdProcess::kPageShift] = {backing.data(), Mtype::CC,
                                                           backing.size(), 0};
  memory.register_process(vmid, &page_table, &page_table_mutex);
  GpuVm gpu_vm(&memory);
  const AddressSpaceHandle handle = gpu_vm.register_legacy(vmid);
  ASSERT_TRUE(handle);

  std::array<std::byte, mapped_bytes * 2> read_value{};
  std::ranges::fill(read_value, std::byte{0xff});
  EXPECT_EQ(gpu_vm.read(handle, virtual_address, read_value), VmAccessOutcome::Complete);
  EXPECT_TRUE(std::all_of(read_value.begin(), read_value.begin() + mapped_bytes,
                          [](std::byte value) { return value == std::byte{0x5a}; }));
  EXPECT_TRUE(std::all_of(read_value.begin() + mapped_bytes, read_value.end(),
                          [](std::byte value) { return value == std::byte{0}; }));

  std::array<std::byte, mapped_bytes * 2> write_value{};
  std::ranges::fill(write_value, std::byte{0xa5});
  EXPECT_EQ(gpu_vm.write(handle, virtual_address, write_value), VmAccessOutcome::Complete);
  EXPECT_TRUE(std::ranges::all_of(backing, [](uint8_t value) { return value == 0xa5; }));

  EXPECT_TRUE(gpu_vm.unregister_address_space(handle));
  memory.unregister_process(vmid);
}

TEST(GpuVmTranslation, LegacyBackingFaultsForInaccessibleMappedPage) {
  GpuMemory memory("memory");
  KfdProcess::PageTable page_table;
  std::shared_mutex page_table_mutex;
  constexpr uint32_t vmid = 7;
  constexpr uint64_t virtual_address = 0x4000;
  HostPage backing;
  ASSERT_NE(backing.data(), nullptr);
  page_table[virtual_address >> KfdProcess::kPageShift] = {backing.data(), Mtype::CC};
  memory.register_process(vmid, &page_table, &page_table_mutex);
  RecordingFaultReporter reporter;
  memory.set_memory_fault_reporter(&reporter);
  GpuVm gpu_vm(&memory);
  const AddressSpaceHandle handle = gpu_vm.register_legacy(vmid);
  ASSERT_TRUE(handle);
  ASSERT_EQ(mprotect(backing.data(), KfdProcess::kPageSize, PROT_NONE), 0);

  std::array<std::byte, 4> value{};
  EXPECT_EQ(gpu_vm.read(handle, virtual_address, value), VmAccessOutcome::Faulted);
  EXPECT_EQ(gpu_vm.write(handle, virtual_address, value), VmAccessOutcome::Faulted);
  EXPECT_EQ(reporter.vmids, (std::vector<uint32_t>{vmid, vmid}));
  EXPECT_EQ(reporter.addresses, (std::vector<uint64_t>{virtual_address, virtual_address}));

  ASSERT_EQ(mprotect(backing.data(), KfdProcess::kPageSize, PROT_READ | PROT_WRITE), 0);
  memory.set_memory_fault_reporter(nullptr);
  EXPECT_TRUE(gpu_vm.unregister_address_space(handle));
  memory.unregister_process(vmid);
}

TEST(GpuVmTranslation, StrictLegacyBackingReportsWrappingRange) {
  GpuMemory memory("memory");
  constexpr uint32_t vmid = 7;
  constexpr uint64_t address = UINT64_MAX - 1;
  RecordingFaultReporter reporter;
  memory.set_memory_fault_reporter(&reporter);
  std::array<uint8_t, 4> value{};

  EXPECT_EQ(memory.read_block_strict(address, value, vmid), CopyOutcome::Faulted);
  EXPECT_EQ(memory.write_block_strict(address, value, vmid), CopyOutcome::Faulted);
  EXPECT_EQ(reporter.vmids, (std::vector<uint32_t>{vmid, vmid}));
  EXPECT_EQ(reporter.addresses, (std::vector<uint64_t>{address, address}));
}

TEST(GpuVmTranslation, LegacyVmidZeroGetsGenerationSafePassthroughBinding) {
  GpuMemory memory("memory");
  memory.set_passthrough(true);
  GpuVm gpu_vm(&memory);
  HostPage backing;
  ASSERT_NE(backing.data(), nullptr);
  backing.data()[0] = 0x5a;

  const AddressSpaceHandle handle = gpu_vm.register_legacy(0);

  ASSERT_TRUE(handle);
  ASSERT_TRUE(gpu_vm.lookup(handle));
  EXPECT_EQ(gpu_vm.lookup(handle)->vmid, 0u);
  EXPECT_FALSE(gpu_vm.lookup(handle)->external);
  EXPECT_FALSE(gpu_vm.initialize_gart_address_space());

  std::array<std::byte, 1> value{};
  const uint64_t address = reinterpret_cast<uint64_t>(backing.data());
  EXPECT_EQ(gpu_vm.read(handle, address, value), VmAccessOutcome::Complete);
  EXPECT_EQ(value[0], std::byte{0x5a});
  value[0] = std::byte{0xa5};
  EXPECT_EQ(gpu_vm.write(handle, address, value), VmAccessOutcome::Complete);
  EXPECT_EQ(backing.data()[0], 0xa5);

  EXPECT_TRUE(gpu_vm.unregister_address_space(handle));
  const AddressSpaceHandle gart = gpu_vm.initialize_gart_address_space();
  ASSERT_TRUE(gart);
  EXPECT_EQ(gart.slot, handle.slot);
  EXPECT_NE(gart.generation, handle.generation);
}

} // namespace
} // namespace rocjitsu::amdgpu
