// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "async_vm_test_support.h"
#include "legacy_gpu_memory_fixture.h"
#include "rocjitsu/kmd/linux/kfd_process.h"
#include "rocjitsu/kmd/linux/legacy_gpu_vm.h"
#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "rocjitsu/vm/amdgpu/gpu_vm.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <future>
#include <map>
#include <memory>
#include <mutex>
#include <span>
#include <sys/mman.h>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>
#include <utility>
#include <vector>

#if defined(__SANITIZE_ADDRESS__)
#define RJ_GPU_VM_TRANSLATION_TEST_WITH_ASAN 1
#elif defined(__has_feature)
#if __has_feature(address_sanitizer)
#define RJ_GPU_VM_TRANSLATION_TEST_WITH_ASAN 1
#endif
#endif

namespace rocjitsu::amdgpu {
namespace {

#if defined(RJ_GPU_VM_TRANSLATION_TEST_WITH_ASAN)
constexpr bool kSanitizedVmMetadata = true;
#else
constexpr bool kSanitizedVmMetadata = false;
#endif

class TestPhysicalMemory : public PhysicalMemoryAccess {
public:
  VmAccessOutcome read(VmMemoryDomain domain, uint64_t address,
                       std::span<std::byte> bytes) override {
    ++read_calls;
    const auto &memory = domain == VmMemoryDomain::System ? system : local;
    std::vector<std::byte> staged(bytes.size());
    for (std::size_t index = 0; index < bytes.size(); ++index) {
      const auto found = memory.find(address + index);
      if (found == memory.end())
        return VmAccessOutcome::Unavailable;
      staged[index] = found->second;
    }
    std::ranges::copy(staged, bytes.begin());
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

class DeferredSystemProbeMemory final : public TestPhysicalMemory {
public:
  [[nodiscard]] std::optional<VmTransferToken>
  prepare_deferred_system_read(uint64_t address, std::size_t length, VmAccessKind access,
                               const std::shared_ptr<VmTransferEffectGate> &) override {
    ++read_prepares;
    last_address = address;
    last_length = length;
    EXPECT_EQ(access, VmAccessKind::Read);
    return std::nullopt;
  }
  [[nodiscard]] std::optional<VmTransferToken>
  prepare_deferred_system_write(uint64_t address, std::span<const std::byte> bytes,
                                const std::shared_ptr<VmTransferEffectGate> &) override {
    ++write_prepares;
    last_address = address;
    last_length = bytes.size();
    return std::nullopt;
  }
  [[nodiscard]] std::optional<VmTransferToken>
  prepare_deferred_system_read(uint64_t address, std::size_t length, VmAccessKind access,
                               const std::shared_ptr<VmTransferEffectGate> &gate,
                               const GpuVmAccess &executing_access, uint64_t gpu_va,
                               const VmTranslation &selected_translation) override {
    ++context_read_prepares;
    seen_cache_identity = executing_access.cache_identity();
    seen_cache_namespace = executing_access.cache_namespace();
    seen_translation = selected_translation;
    seen_gate = gate;
    last_gpu_va = gpu_va;
    last_address = address;
    last_length = length;
    last_kind = access;
    return std::nullopt;
  }
  [[nodiscard]] std::optional<VmTransferToken>
  prepare_deferred_system_write(uint64_t address, std::span<const std::byte> bytes,
                                const std::shared_ptr<VmTransferEffectGate> &gate,
                                const GpuVmAccess &executing_access, uint64_t gpu_va,
                                const VmTranslation &selected_translation) override {
    ++context_write_prepares;
    seen_cache_identity = executing_access.cache_identity();
    seen_cache_namespace = executing_access.cache_namespace();
    seen_translation = selected_translation;
    seen_gate = gate;
    last_gpu_va = gpu_va;
    last_address = address;
    last_length = bytes.size();
    last_kind = VmAccessKind::Write;
    return std::nullopt;
  }

  std::size_t read_prepares = 0;
  std::size_t write_prepares = 0;
  std::size_t context_read_prepares = 0;
  std::size_t context_write_prepares = 0;
  // Keep only values: retaining the snapshot here would own this memory
  // provider through its translator and form a reference cycle.
  std::optional<uint64_t> seen_cache_identity;
  std::optional<VmCacheNamespace> seen_cache_namespace;
  std::optional<VmTranslation> seen_translation;
  std::shared_ptr<VmTransferEffectGate> seen_gate;
  uint64_t last_gpu_va = 0;
  uint64_t last_address = 0;
  std::size_t last_length = 0;
  VmAccessKind last_kind = VmAccessKind::Read;
};

class ByteTranslator final : public AddressSpaceTranslator {
public:
  VmTranslationResult translate(uint64_t address, std::size_t size,
                                VmAccessKind /*access*/) const override {
    if (size == 0)
      return {.outcome = VmAccessOutcome::Malformed, .translation = {}};
    return {
        .outcome = VmAccessOutcome::Complete,
        .translation = {.domain = VmMemoryDomain::System,
                        .address = address,
                        .contiguous_bytes = 1,
                        .mtype = Mtype::RW,
                        .permissions = {.readable = true, .writable = true, .executable = true}}};
  }
};

class BlockingPhysicalMemory final : public PhysicalMemoryAccess {
public:
  VmAccessOutcome read(VmMemoryDomain, uint64_t address, std::span<std::byte> bytes) override {
    std::unique_lock lock(mutex_);
    if (read_calls_++ == 0) {
      first_read_entered_ = true;
      condition_.notify_all();
      condition_.wait(lock, [this] { return released_; });
    }
    std::ranges::fill(bytes, static_cast<std::byte>(address + 1));
    return VmAccessOutcome::Complete;
  }

  VmAccessOutcome write(VmMemoryDomain, uint64_t, std::span<const std::byte>) override {
    return VmAccessOutcome::Complete;
  }

  bool wait_for_first_read(std::chrono::milliseconds timeout) {
    std::unique_lock lock(mutex_);
    return condition_.wait_for(lock, timeout, [this] { return first_read_entered_; });
  }

  void release() {
    std::lock_guard lock(mutex_);
    released_ = true;
    condition_.notify_all();
  }

private:
  std::mutex mutex_;
  std::condition_variable condition_;
  std::size_t read_calls_ = 0;
  bool first_read_entered_ = false;
  bool released_ = false;
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
  explicit HostPage(size_t size = KfdProcess::kPageSize) : size_(size) {
    void *mapping =
        mmap(nullptr, size_, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    data_ = mapping == MAP_FAILED ? nullptr : static_cast<uint8_t *>(mapping);
  }
  HostPage(const HostPage &) = delete;
  HostPage &operator=(const HostPage &) = delete;
  ~HostPage() {
    if (data_ != nullptr)
      munmap(data_, size_);
  }

  uint8_t *data() const { return data_; }

private:
  size_t size_;
  uint8_t *data_ = nullptr;
};

AddressSpaceHandle register_step_process(LegacyGpuVmAdapter &adapter, KfdProcess &process,
                                         MemoryFaultReporter *reporter = nullptr) {
  return adapter.register_address_space(7,
                                        {.page_table = &process.page_table_,
                                         .page_table_mutex = &process.page_table_mutex_,
                                         .page_table_generation = process.page_table_generation(),
                                         .request_mutex = process.page_table_request_mutex(),
                                         .mutation_epoch = process.page_table_mutation_epoch(),
                                         .page_table_cache_state = process.page_table_cache_state(),
                                         .fault_reporter = reporter});
}

TEST(GpuVmTranslation, DefaultStepsPreserveCustomTransportProgressAndFaultOrigin) {
  class Translator final : public AddressSpaceTranslator {
  public:
    VmTranslationResult translate(uint64_t address, size_t, VmAccessKind) const override {
      addresses.push_back(address);
      if (address == fault_address)
        return {.outcome = VmAccessOutcome::Faulted, .translation = {}};
      return {.outcome = VmAccessOutcome::Complete,
              .translation = {.domain = VmMemoryDomain::System,
                              .address = address + 0x100,
                              .contiguous_bytes = 2,
                              .permissions = {.readable = true, .writable = true}}};
    }
    mutable std::vector<uint64_t> addresses;
    uint64_t fault_address = 0x104;
  };
  GpuVm vm;
  auto translator = std::make_shared<Translator>();
  auto transport = std::make_shared<TestPhysicalMemory>();
  std::vector<uint64_t> faults;
  const auto handle = vm.register_address_space(
      7, translator, transport, [&](uint64_t address, VmAccessKind) { faults.push_back(address); });
  ASSERT_TRUE(handle);
  const auto access = vm.snapshot(handle);
  ASSERT_TRUE(access);
  std::array<std::byte, 6> bytes;
  bytes.fill(std::byte{0x55});
  ASSERT_EQ(transport->write(VmMemoryDomain::System, 0x200, bytes), VmAccessOutcome::Complete);
  bytes.fill(std::byte{0xaa});
  size_t completed = 0;
  EXPECT_EQ(access->read(0x100, bytes, completed), VmAccessOutcome::Faulted);
  EXPECT_EQ(completed, 4u);
  EXPECT_EQ(translator->addresses, (std::vector<uint64_t>{0x100, 0x102, 0x104}));
  EXPECT_EQ(faults, (std::vector<uint64_t>{0x104}));
  EXPECT_EQ(transport->read_calls, 2u);
  EXPECT_EQ(bytes, (std::array<std::byte, 6>{std::byte{0x55}, std::byte{0x55}, std::byte{0x55},
                                             std::byte{0x55}, std::byte{0xaa}, std::byte{0xaa}}));
  translator->fault_address = 0;
  translator->addresses.clear();
  faults.clear();
  transport->system.erase(0x204);
  EXPECT_EQ(access->read(0x100, bytes, completed), VmAccessOutcome::Unavailable);
  EXPECT_EQ(completed, 4u);
  EXPECT_TRUE(faults.empty());
  EXPECT_EQ(translator->addresses, (std::vector<uint64_t>{0x104}));
  EXPECT_EQ(bytes[4], std::byte{0xaa});
  EXPECT_EQ(bytes[5], std::byte{0xaa});
  transport->system[0x204] = std::byte{0x66};
  EXPECT_EQ(access->read(0x100, bytes, completed), VmAccessOutcome::Complete);
  EXPECT_EQ(completed, bytes.size());
  EXPECT_EQ(bytes[4], std::byte{0x66});
  EXPECT_EQ(bytes[5], std::byte{0x55});

  translator->fault_address = 0x104;
  translator->addresses.clear();
  bytes.fill(std::byte{0x99});
  completed = 0;
  EXPECT_EQ(access->write(0x100, bytes, completed), VmAccessOutcome::Faulted);
  EXPECT_EQ(completed, 4u);
  EXPECT_EQ(faults, (std::vector<uint64_t>{0x104}));
  EXPECT_EQ(transport->system[0x203], std::byte{0x99});
  EXPECT_EQ(transport->system[0x204], std::byte{0x66});
}

TEST(GpuVmTranslation, LegacyStepsKeepExtentBoundariesAndRejectOverlappingCoverage) {
  for (const auto owner : {LegacyHostExtentOwner::Driver, LegacyHostExtentOwner::Application}) {
    for (const auto second_offset : {8u, 12u}) {
      for (const auto first_size : {8u, 16u}) {
        SCOPED_TRACE(static_cast<int>(owner));
        SCOPED_TRACE(second_offset);
        SCOPED_TRACE(first_size);
        GpuMemory memory("memory");
        KfdProcess process(7);
        HostPage backing;
        ASSERT_NE(backing.data(), nullptr);
        std::memset(backing.data(), 0x55, 32);
        LegacyPageTableEntry pte;
        pte.host_extents = {{backing.data(), first_size, 0, owner},
                            {backing.data() + 16, 8, second_offset, owner}};
        process.page_table_[4] = std::move(pte);
        RecordingFaultReporter reporter;
        GpuVm vm;
        LegacyGpuVmAdapter adapter(vm, &memory);
        const auto handle = register_step_process(adapter, process, &reporter);
        ASSERT_TRUE(handle);
        const auto access = vm.snapshot(handle);
        ASSERT_TRUE(access);
        std::array<std::byte, 16> bytes;
        bytes.fill(std::byte{0xaa});
        size_t completed = 0;
        EXPECT_EQ(access->read(0x4000, bytes, completed), VmAccessOutcome::Faulted);
        EXPECT_EQ(completed, 0u);
        EXPECT_TRUE(std::ranges::all_of(bytes, [](auto byte) { return byte == std::byte{0xaa}; }));
        EXPECT_EQ(access->write(0x4000, bytes, completed), VmAccessOutcome::Faulted);
        EXPECT_EQ(completed, 0u);
        EXPECT_TRUE(std::ranges::all_of(std::span(backing.data(), 32),
                                        [](auto byte) { return byte == 0x55; }));
        EXPECT_EQ(reporter.addresses, (std::vector<uint64_t>{0x4000, 0x4000}));
      }
    }
  }
}

TEST(GpuVmTranslation, LegacyStepsKeepPageProgressAndResumeWithoutReplayingPrefix) {
  for (const auto owner : {LegacyHostExtentOwner::Driver, LegacyHostExtentOwner::Application}) {
    GpuMemory memory("memory");
    KfdProcess process(7);
    HostPage first, second;
    ASSERT_NE(first.data(), nullptr);
    ASSERT_NE(second.data(), nullptr);
    std::memset(first.data(), 0x55, KfdProcess::kPageSize);
    std::memset(second.data(), 0x66, KfdProcess::kPageSize);
    process.map_pages(0x4000, first.data(), KfdProcess::kPageSize, Mtype::RW, owner);
    GpuVm vm;
    LegacyGpuVmAdapter adapter(vm, &memory);
    const auto handle = register_step_process(adapter, process);
    ASSERT_TRUE(handle);
    const auto access = vm.snapshot(handle);
    ASSERT_TRUE(access);
    std::array<std::byte, 32> bytes;
    bytes.fill(std::byte{0xaa});
    size_t completed = 0;
    EXPECT_EQ(access->read(0x4ff0, bytes, completed), VmAccessOutcome::Unavailable);
    EXPECT_EQ(completed, 16u);
    for (size_t i = 0; i < bytes.size(); ++i)
      EXPECT_EQ(bytes[i], i < 16 ? std::byte{0x55} : std::byte{0xaa});
    std::memset(first.data() + KfdProcess::kPageSize - 16, 0x77, 16);
    process.map_pages(0x5000, second.data(), KfdProcess::kPageSize, Mtype::CC, owner);
    EXPECT_EQ(access->read(0x4ff0, bytes, completed), VmAccessOutcome::Complete);
    EXPECT_EQ(completed, bytes.size());
    EXPECT_EQ(bytes[0], std::byte{0x55});
    EXPECT_EQ(bytes[16], std::byte{0x66});
    process.unmap_pages(0x5000, KfdProcess::kPageSize);
    bytes.fill(std::byte{0x88});
    completed = 0;
    EXPECT_EQ(access->write(0x4ff0, bytes, completed), VmAccessOutcome::Unavailable);
    EXPECT_EQ(completed, 16u);
    std::memset(first.data() + KfdProcess::kPageSize - 16, 0x99, 16);
    process.map_pages(0x5000, second.data(), KfdProcess::kPageSize, Mtype::CC, owner);
    EXPECT_EQ(access->write(0x4ff0, bytes, completed), VmAccessOutcome::Complete);
    EXPECT_EQ(first.data()[KfdProcess::kPageSize - 1], 0x99);
    EXPECT_EQ(second.data()[0], 0x88);
  }
}

TEST(GpuVmTranslation, LegacyApplicationStepPrevalidatesCrossHostPageWrites) {
  GpuMemory memory("memory");
  KfdProcess process(7);
  HostPage backing(2 * KfdProcess::kPageSize);
  ASSERT_NE(backing.data(), nullptr);
  auto *first = backing.data() + KfdProcess::kPageSize - 16;
  std::memset(first, 0x55, 32);
  process.map_pages(0x4000, first, 32, Mtype::RW, LegacyHostExtentOwner::Application);
  RecordingFaultReporter reporter;
  GpuVm vm;
  LegacyGpuVmAdapter adapter(vm, &memory);
  const auto handle = register_step_process(adapter, process, &reporter);
  ASSERT_TRUE(handle);
  const auto access = vm.snapshot(handle);
  ASSERT_TRUE(access);
  ASSERT_EQ(mprotect(backing.data() + KfdProcess::kPageSize, KfdProcess::kPageSize, PROT_READ), 0);
  std::array<std::byte, 32> bytes;
  bytes.fill(std::byte{0xaa});
  size_t completed = 0;
  EXPECT_EQ(access->write(0x4000, bytes, completed), VmAccessOutcome::Faulted);
  EXPECT_EQ(completed, 0u);
  EXPECT_TRUE(std::ranges::all_of(std::span(first, 32), [](auto byte) { return byte == 0x55; }));
  EXPECT_EQ(reporter.addresses, (std::vector<uint64_t>{0x4000}));
  ASSERT_EQ(mprotect(backing.data() + KfdProcess::kPageSize, KfdProcess::kPageSize, PROT_NONE), 0);
  EXPECT_EQ(access->read(0x4000, bytes, completed), VmAccessOutcome::Faulted);
  EXPECT_EQ(completed, 0u);
  EXPECT_TRUE(std::ranges::all_of(bytes, [](auto byte) { return byte == std::byte{0xaa}; }));
  EXPECT_EQ(reporter.addresses, (std::vector<uint64_t>{0x4000, 0x4000}));
  ASSERT_EQ(mprotect(backing.data() + KfdProcess::kPageSize, KfdProcess::kPageSize,
                     PROT_READ | PROT_WRITE),
            0);
}

TEST(GpuVmTranslation, LegacyStepsPreserveClientBackingAndTerminalFaults) {
  GpuMemory memory("memory");
  KfdProcess process(7);
  HostPage backing;
  ASSERT_NE(backing.data(), nullptr);
  backing.data()[0] = 0x55;
  RecordingFaultReporter reporter;
  GpuVm vm;
  LegacyGpuVmAdapter adapter(vm, &memory);
  const auto handle = register_step_process(adapter, process, &reporter);
  ASSERT_TRUE(handle);
  ASSERT_TRUE(adapter.set_client_pid(handle, getpid()));
  const auto access = vm.snapshot(handle);
  ASSERT_TRUE(access);
  const auto address = reinterpret_cast<uintptr_t>(backing.data());
  std::array<std::byte, 1> bytes{std::byte{0xaa}};
  EXPECT_EQ(access->read(address, bytes), VmAccessOutcome::Complete);
  EXPECT_EQ(bytes[0], std::byte{0x55});
  bytes[0] = std::byte{0x66};
  EXPECT_EQ(access->write(address, bytes), VmAccessOutcome::Complete);
  EXPECT_EQ(backing.data()[0], 0x66);
  EXPECT_TRUE(reporter.addresses.empty());
  ASSERT_EQ(mprotect(backing.data(), KfdProcess::kPageSize, PROT_NONE), 0);
  bytes[0] = std::byte{0xaa};
  EXPECT_EQ(access->read(address, bytes), VmAccessOutcome::Faulted);
  EXPECT_EQ(bytes[0], std::byte{0xaa});
  EXPECT_EQ(access->write(address, bytes), VmAccessOutcome::Faulted);
  EXPECT_EQ(reporter.addresses, (std::vector<uint64_t>{address, address}));
  ASSERT_EQ(mprotect(backing.data(), KfdProcess::kPageSize, PROT_READ | PROT_WRITE), 0);
  EXPECT_EQ(backing.data()[0], 0x66);
}

TEST(GpuVmTranslation, LegacyStepFaultCallbacksCanMutatePolicyAndHostMappings) {
  EXPECT_EXIT(
      ([] {
        alarm(10);
        for (const auto owner :
             {LegacyHostExtentOwner::Driver, LegacyHostExtentOwner::Application}) {
          GpuMemory memory("memory");
          KfdProcess process(7);
          HostPage backing;
          if (!backing.data())
            _exit(1);
          backing.data()[0] = 0x55;
          process.map_pages(0x4000, backing.data(), KfdProcess::kPageSize, Mtype::RW, owner);
          class Reporter final : public MemoryFaultReporter {
          public:
            explicit Reporter(KfdProcess &process) : process(process) {}
            void report_memory_fault(uint32_t, uint64_t address, MemoryFaultCause) override {
              if (address != 0x4000)
                _exit(2);
              ++calls;
              process.set_page_mtype(0x4000, KfdProcess::kPageSize, Mtype::CC);
              const auto mapping = rocjitsu::host_mapping_lock().lock_exclusive();
            }
            KfdProcess &process;
            unsigned calls = 0;
          } reporter(process);
          GpuVm vm;
          LegacyGpuVmAdapter adapter(vm, &memory);
          const auto handle = register_step_process(adapter, process, &reporter);
          const auto access = vm.snapshot(handle);
          std::array<std::byte, 1> bytes{std::byte{0xaa}};
          if (!access || access->read(0x4000, bytes) != VmAccessOutcome::Complete)
            _exit(3);
          if (mprotect(backing.data(), KfdProcess::kPageSize, PROT_NONE) != 0)
            _exit(4);
          bytes[0] = std::byte{0xaa};
          if (access->read(0x4000, bytes) != VmAccessOutcome::Faulted || reporter.calls != 1 ||
              bytes[0] != std::byte{0xaa})
            _exit(5);
          if (access->write(0x4000, bytes) != VmAccessOutcome::Faulted || reporter.calls != 2)
            _exit(6);
          if (mprotect(backing.data(), KfdProcess::kPageSize, PROT_READ | PROT_WRITE) != 0 ||
              backing.data()[0] != 0x55)
            _exit(7);
        }
        _exit(0);
      }()),
      ::testing::ExitedWithCode(0), "");
}

TEST(GpuVmTranslation, LegacyStepKeepsOneMappedSpanStableDuringConcurrentRemap) {
  EXPECT_EXIT(
      ([] {
        alarm(10);
        for (const auto owner :
             {LegacyHostExtentOwner::Driver, LegacyHostExtentOwner::Application}) {
          GpuMemory memory("memory");
          KfdProcess process(7);
          HostPage first, second;
          if (!first.data() || !second.data())
            _exit(1);
          std::memset(first.data(), 0x55, KfdProcess::kPageSize);
          std::memset(second.data(), 0xaa, KfdProcess::kPageSize);
          process.map_pages(0x4000, first.data(), KfdProcess::kPageSize, Mtype::RW, owner);
          GpuVm vm;
          LegacyGpuVmAdapter adapter(vm, &memory);
          const auto handle = register_step_process(adapter, process);
          const auto access = vm.snapshot(handle);
          if (!access)
            _exit(2);
          std::atomic<bool> started{false};
          std::thread writer([&] {
            while (!started.load(std::memory_order_acquire))
              std::this_thread::yield();
            for (unsigned i = 0; i < 2000; ++i)
              process.map_pages(0x4000, i % 2 ? first.data() : second.data(), KfdProcess::kPageSize,
                                i % 2 ? Mtype::RW : Mtype::CC, owner);
          });
          started.store(true, std::memory_order_release);
          for (unsigned i = 0; i < 2000; ++i) {
            std::array<std::byte, 512> bytes;
            bytes.fill(std::byte{0xcc});
            const auto outcome = access->read(0x4000, bytes);
            // Continuous remapping may exhaust the bounded metadata
            // retry budget. That existing fail-closed result must not
            // expose any partial read.
            if (kSanitizedVmMetadata && outcome == VmAccessOutcome::Faulted &&
                std::ranges::all_of(bytes, [](auto byte) { return byte == std::byte{0xcc}; }))
              continue;
            if (outcome != VmAccessOutcome::Complete ||
                (bytes[0] != std::byte{0x55} && bytes[0] != std::byte{0xaa}) ||
                !std::ranges::all_of(bytes, [&](auto byte) { return byte == bytes[0]; }))
              _exit(3);
          }
          writer.join();
          process.unmap_pages(0x4000, KfdProcess::kPageSize);
          std::array<std::byte, 1> bytes{std::byte{0xcc}};
          if (access->read(0x4000, bytes) != VmAccessOutcome::Unavailable ||
              bytes[0] != std::byte{0xcc})
            _exit(4);
        }
        _exit(0);
      }()),
      ::testing::ExitedWithCode(0), "");
}

#if defined(RJ_GPU_VM_TRANSLATION_TEST_WITH_ASAN)
TEST(GpuVmTranslation, LegacyStepRevalidatesAfterUnlockedAllocatorMetadataQuery) {
  GpuMemory memory("memory");
  KfdProcess process(7);
  HostPage first, second;
  ASSERT_NE(first.data(), nullptr);
  ASSERT_NE(second.data(), nullptr);
  first.data()[0] = 0x55;
  second.data()[0] = 0xaa;
  process.map_pages(0x4000, first.data(), KfdProcess::kPageSize);
  GpuVm vm;
  LegacyGpuVmAdapter adapter(vm, &memory);
  const auto handle = register_step_process(adapter, process);
  ASSERT_TRUE(handle);
  const auto access = vm.snapshot(handle);
  ASSERT_TRUE(access);
  unsigned calls = 0;
  std::function<void()> remap = [&] {
    if (++calls == 1)
      process.map_pages(0x4000, second.data(), KfdProcess::kPageSize);
  };
  auto *address_space = adapter.address_space(7);
  ASSERT_NE(address_space, nullptr);
  LegacyAddressSpaceTestAccess::set_page_table_unlocked_hook(*address_space, &remap);
  std::array<std::byte, 1> bytes{};
  const auto outcome = access->read(0x4000, bytes);
  LegacyAddressSpaceTestAccess::set_page_table_unlocked_hook(*address_space, nullptr);
  EXPECT_EQ(outcome, VmAccessOutcome::Complete);
  EXPECT_GE(calls, 2u);
  EXPECT_EQ(bytes[0], std::byte{0xaa});
}
#endif

constexpr uint64_t kValid = uint64_t{1} << 0;
constexpr uint64_t kSystem = uint64_t{1} << 1;
constexpr uint64_t kSnooped = uint64_t{1} << 2;
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
                     physical_address | leaf_flags | kPdePte);
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
                     physical_address | leaf_flags | kPdePte);
}

TEST(GpuVmTranslation, Gfx12WalkHonorsSystemMemoryRootFlag) {
  auto physical = std::make_shared<TestPhysicalMemory>();
  constexpr uint64_t root = 0x1000;
  constexpr uint64_t virtual_address = 0x5123;
  constexpr uint64_t physical_page = 0x9000;
  constexpr std::array<uint64_t, 4> tables = {0x2000, 0x3000, 0x4000, 0x5000};
  constexpr std::array<uint32_t, 4> shifts = {48, 39, 30, 21};
  uint64_t table = root;
  for (std::size_t level = 0; level < tables.size(); ++level) {
    const uint64_t index = (virtual_address >> shifts[level]) & 0x1ff;
    physical->store_qword(VmMemoryDomain::System, table + index * sizeof(uint64_t),
                          tables[level] | kValid | kSystem);
    table = tables[level];
  }
  const uint64_t leaf_index = (virtual_address >> 12) & 0x1ff;
  physical->store_qword(VmMemoryDomain::System, table + leaf_index * sizeof(uint64_t),
                        physical_page | kPdePte | kValid | kSystem | kReadable);
  Gfx12PageTableTranslator translator(physical, root | kSystem);

  const VmTranslationResult translated =
      translator.translate(virtual_address, 1, VmAccessKind::Read);

  ASSERT_TRUE(translated);
  EXPECT_EQ(translated.translation.domain, VmMemoryDomain::System);
  EXPECT_EQ(translated.translation.address, physical_page + 0x123);
}

TEST(GpuVmTranslation, Gfx12QueuePreparesOnlyTranslatedSystemData) {
  auto physical = std::make_shared<DeferredSystemProbeMemory>();
  constexpr uint64_t root = 0x1000;
  constexpr uint64_t system_va = 0x8123;
  constexpr uint64_t local_va = 0xa123;
  constexpr uint64_t system_page = 0x9000;
  constexpr uint64_t local_page = 0xb000;
  map_4k(*physical, root, system_va, system_page, kValid | kSystem | kReadable | kWriteable);
  map_4k(*physical, root, local_va, local_page, kValid | kReadable | kWriteable);
  Gfx12PageTableTranslator translator(physical, root);
  auto gate = std::make_shared<VmTransferEffectGate>();

  EXPECT_FALSE(translator.prepare_deferred_read(*physical, system_va, 8, VmAccessKind::Read, gate));
  EXPECT_EQ(physical->read_prepares, 1u);
  EXPECT_EQ(physical->last_address, system_page + 0x123);
  EXPECT_EQ(physical->last_length, 8u);
  const std::array<std::byte, 4> value{std::byte{1}, std::byte{2}, std::byte{3}, std::byte{4}};
  EXPECT_FALSE(translator.prepare_deferred_write(*physical, system_va, value, gate));
  EXPECT_EQ(physical->write_prepares, 1u);
  EXPECT_EQ(physical->last_address, system_page + 0x123);
  EXPECT_EQ(physical->last_length, value.size());

  EXPECT_FALSE(translator.prepare_deferred_read(*physical, local_va, 8, VmAccessKind::Read, gate));
  EXPECT_FALSE(translator.prepare_deferred_write(*physical, local_va, value, gate));
  EXPECT_EQ(physical->read_prepares, 1u);
  EXPECT_EQ(physical->write_prepares, 1u);
  EXPECT_GT(physical->read_calls, 0u) << "page-table/control reads remain synchronous";
}

TEST(GpuVmTranslation, Gfx12PageTablePassesExecutingSnapshotToSystemPreparation) {
  auto physical = std::make_shared<DeferredSystemProbeMemory>();
  constexpr uint64_t root = 0x1000;
  constexpr uint64_t gpu_va = 0x8123;
  constexpr uint64_t iova_page = 0x9000;
  map_4k(*physical, root, gpu_va, iova_page, kValid | kSystem | kReadable | kWriteable);
  physical->system[iova_page + 0x123] = std::byte{0x5a};
  GpuVm vm;
  const AddressSpaceHandle handle = vm.register_gfx12_address_space(7, root, physical);
  ASSERT_TRUE(handle);
  const std::optional<GpuVmAccess> access = vm.snapshot_pinned(handle);
  ASSERT_TRUE(access);

  std::array<std::byte, 1> read{};
  VmTransferToken token;
  std::size_t completed = 0;
  EXPECT_EQ(access->read(gpu_va, read, completed, token), VmAccessOutcome::Complete);
  EXPECT_EQ(read[0], std::byte{0x5a});
  EXPECT_EQ(physical->context_read_prepares, 1u);
  ASSERT_TRUE(physical->seen_cache_identity);
  ASSERT_TRUE(physical->seen_cache_namespace);
  EXPECT_EQ(*physical->seen_cache_identity, access->cache_identity());
  EXPECT_EQ(*physical->seen_cache_namespace, access->cache_namespace());
  EXPECT_EQ(physical->seen_gate, access->retain_effect_gate());
  EXPECT_EQ(physical->last_gpu_va, gpu_va);
  EXPECT_EQ(physical->last_address, iova_page + 0x123);
  EXPECT_EQ(physical->last_length, read.size());
  EXPECT_EQ(physical->last_kind, VmAccessKind::Read);
  ASSERT_TRUE(physical->seen_translation);
  EXPECT_EQ(physical->seen_translation->mtype, Mtype::RW);
  EXPECT_TRUE(physical->seen_translation->permissions.readable);
  EXPECT_TRUE(physical->seen_translation->permissions.writable);
  EXPECT_FALSE(physical->seen_translation->permissions.executable);

  constexpr std::array<std::byte, 1> write{std::byte{0x3c}};
  completed = 0;
  EXPECT_EQ(access->write(gpu_va, write, completed, token), VmAccessOutcome::Complete);
  EXPECT_EQ(physical->context_write_prepares, 1u);
  EXPECT_EQ(physical->last_gpu_va, gpu_va);
  EXPECT_EQ(physical->last_address, iova_page + 0x123);
  EXPECT_EQ(physical->last_length, write.size());
  EXPECT_EQ(physical->last_kind, VmAccessKind::Write);
  ASSERT_TRUE(physical->seen_translation);
  EXPECT_EQ(physical->seen_translation->mtype, Mtype::RW);
  EXPECT_TRUE(physical->seen_translation->permissions.readable);
  EXPECT_TRUE(physical->seen_translation->permissions.writable);
  EXPECT_FALSE(physical->seen_translation->permissions.executable);
  EXPECT_EQ(physical->system[iova_page + 0x123], write[0]);
}

TEST(GpuVmTranslation, Gfx12PeerRewalkDistinguishesNcFromRwPteEncoding) {
  auto physical = std::make_shared<TestPhysicalMemory>();
  constexpr uint64_t root = 0x1000;
  constexpr uint64_t gpu_va = 0x8123;
  constexpr uint64_t iova_page = 0x9000;
  constexpr uint64_t flags = kValid | kSystem | kReadable | kWriteable | kPdePte;
  map_4k(*physical, root, gpu_va, iova_page, flags);

  GpuVm vm;
  const AddressSpaceHandle handle = vm.register_gfx12_address_space(7, root, physical);
  ASSERT_TRUE(handle);
  const auto access = vm.snapshot_pinned(handle);
  ASSERT_TRUE(access);
  const VmTranslationResult first = access->query_translation(gpu_va, 1, VmAccessKind::Read);
  ASSERT_TRUE(first);
  EXPECT_EQ(first.translation.mtype, Mtype::RW);
  EXPECT_EQ(first.translation.gfx12_pte_mtype, Gfx12PteMtype::NC);

  const uint64_t leaf_index = (gpu_va >> 12) & 0x1ff;
  physical->store_qword(VmMemoryDomain::Local,
                        kLowPageTables.back() + leaf_index * sizeof(uint64_t),
                        iova_page | flags | (uint64_t{2} << 54));
  const VmTranslationResult second = access->query_translation(gpu_va, 1, VmAccessKind::Read);
  ASSERT_TRUE(second);
  EXPECT_EQ(second.translation.mtype, first.translation.mtype);
  EXPECT_EQ(second.translation.gfx12_pte_mtype, Gfx12PteMtype::RW);
  EXPECT_EQ(access->validate_deferred_system_span(gpu_va, iova_page + 0x123, 1, VmAccessKind::Read,
                                                  first.translation),
            VmAccessOutcome::Revoked);
}

TEST(GpuVmTranslation, Gfx12WalkRejectsFinalLevelPde) {
  auto physical = std::make_shared<TestPhysicalMemory>();
  constexpr uint64_t root = 0x1000;
  constexpr uint64_t virtual_address = 0x6000;
  constexpr uint64_t physical_page = 0x9000;
  map_4k(*physical, root, virtual_address, physical_page, kValid | kSystem | kReadable);
  const uint64_t leaf_index = (virtual_address >> 12) & 0x1ff;
  physical->store_qword(VmMemoryDomain::Local,
                        kLowPageTables.back() + leaf_index * sizeof(uint64_t),
                        physical_page | kValid | kSystem | kReadable);
  Gfx12PageTableTranslator translator(physical, root);

  EXPECT_EQ(translator.translate(virtual_address, 1, VmAccessKind::Read).outcome,
            VmAccessOutcome::Faulted);
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
  physical->store_qword(VmMemoryDomain::Local, table,
                        system_page | kPdePte | kValid | kSystem | kReadable);
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
  GpuVm gpu_vm(Gfx12VmConfig::gfx12_0());
  auto physical = std::make_shared<TestPhysicalMemory>();
  constexpr uint64_t table = 0x1000;
  constexpr uint64_t aperture = 0x10000;
  constexpr uint64_t encoded_system_page = 0x0001'0000'0000'9000;
  physical->store_qword(VmMemoryDomain::Local, table,
                        encoded_system_page | kPdePte | kValid | kSystem | kReadable);
  ASSERT_TRUE(gpu_vm.initialize_gart_address_space());
  ASSERT_TRUE(gpu_vm.publish_gart(
      {.page_table_base = table, .aperture_start = aperture, .aperture_end = aperture + 0xfff},
      physical));

  const VmTranslationResult translated =
      gpu_vm.translate(gpu_vm.gart_address_space(), aperture + 0x123, 1, VmAccessKind::Read);

  ASSERT_TRUE(translated);
  EXPECT_EQ(translated.translation.address, 0x9000u + 0x123u);
}

TEST(GpuVmTranslation, GpuVmAcceptsEncodedGartPageTableRoot) {
  GpuVm gpu_vm;
  auto physical = std::make_shared<TestPhysicalMemory>();
  constexpr uint64_t table = 0x1000;
  constexpr uint64_t aperture = 0x10000;
  constexpr uint64_t system_page = 0x9000;
  physical->store_qword(VmMemoryDomain::Local, table,
                        system_page | kPdePte | kValid | kSystem | kReadable);
  physical->system[system_page] = std::byte{0x5a};
  ASSERT_TRUE(gpu_vm.initialize_gart_address_space());

  ASSERT_TRUE(gpu_vm.publish_gart({.page_table_base = table | kValid | kSnooped,
                                   .aperture_start = aperture,
                                   .aperture_end = aperture + 0xfff},
                                  physical));

  std::array<std::byte, 1> value{};
  EXPECT_EQ(gpu_vm.read(gpu_vm.gart_address_space(), aperture, value), VmAccessOutcome::Complete);
  EXPECT_EQ(value[0], std::byte{0x5a});
}

TEST(GpuVmTranslation, GpuVmReadsGartPageTableFromEncodedSystemRoot) {
  GpuVm gpu_vm;
  auto physical = std::make_shared<TestPhysicalMemory>();
  constexpr uint64_t table = 0x1000;
  constexpr uint64_t aperture = 0x10000;
  constexpr uint64_t system_page = 0x9000;
  physical->store_qword(VmMemoryDomain::System, table,
                        system_page | kPdePte | kValid | kSystem | kReadable);
  physical->system[system_page] = std::byte{0x5a};
  ASSERT_TRUE(gpu_vm.initialize_gart_address_space());

  ASSERT_TRUE(gpu_vm.publish_gart({.page_table_base = table | kValid | kSystem | kSnooped,
                                   .aperture_start = aperture,
                                   .aperture_end = aperture + 0xfff},
                                  physical));

  std::array<std::byte, 1> value{};
  EXPECT_EQ(gpu_vm.read(gpu_vm.gart_address_space(), aperture, value), VmAccessOutcome::Complete);
  EXPECT_EQ(value[0], std::byte{0x5a});
}

TEST(GpuVmTranslation, GpuVmAcceptsEncodedZeroGartPageTableRoot) {
  GpuVm gpu_vm;
  auto physical = std::make_shared<TestPhysicalMemory>();
  constexpr uint64_t aperture = 0x10000;
  constexpr uint64_t system_page = 0x9000;
  physical->store_qword(VmMemoryDomain::Local, 0,
                        system_page | kPdePte | kValid | kSystem | kReadable);
  physical->system[system_page] = std::byte{0x5a};
  ASSERT_TRUE(gpu_vm.initialize_gart_address_space());

  ASSERT_TRUE(gpu_vm.publish_gart({.page_table_base = kValid | kSnooped,
                                   .aperture_start = aperture,
                                   .aperture_end = aperture + 0xfff},
                                  physical));

  std::array<std::byte, 1> value{};
  EXPECT_EQ(gpu_vm.read(gpu_vm.gart_address_space(), aperture, value), VmAccessOutcome::Complete);
  EXPECT_EQ(value[0], std::byte{0x5a});
}

TEST(GpuVmTranslation, Gfx12GartRejectsNonSystemOrMissingApertureEntries) {
  auto physical = std::make_shared<TestPhysicalMemory>();
  constexpr uint64_t table = 0x1000;
  constexpr uint64_t aperture_start = 0x10000;
  physical->store_qword(VmMemoryDomain::Local, table, 0x9000 | kValid);
  Gfx12GartTranslator translator(physical, table, aperture_start, aperture_start + 0xfff);

  EXPECT_EQ(translator.translate(aperture_start, 1, VmAccessKind::Read).outcome,
            VmAccessOutcome::Faulted);
  physical->store_qword(VmMemoryDomain::Local, table, 0x9000 | kValid | kSystem | kReadable);
  EXPECT_EQ(translator.translate(aperture_start, 1, VmAccessKind::Read).outcome,
            VmAccessOutcome::Faulted);
  EXPECT_EQ(translator.translate(UINT64_MAX, 2, VmAccessKind::Read).outcome,
            VmAccessOutcome::Malformed);
}

TEST(GpuVmTranslation, Gfx12GartUsesByteBoundariesAndPtePermissions) {
  auto physical = std::make_shared<TestPhysicalMemory>();
  constexpr uint64_t table = 0x1000;
  constexpr uint64_t aperture_start = 0x10000;
  constexpr uint64_t aperture_end = aperture_start + 0x37;
  constexpr uint64_t system_page = 0x9000;
  physical->store_qword(VmMemoryDomain::Local, table,
                        system_page | kPdePte | kValid | kSystem | kReadable);
  Gfx12GartTranslator translator(physical, table, aperture_start, aperture_end);

  const VmTranslationResult last_byte = translator.translate(aperture_end, 2, VmAccessKind::Read);
  ASSERT_TRUE(last_byte);
  EXPECT_EQ(last_byte.translation.contiguous_bytes, 1u);
  EXPECT_TRUE(last_byte.translation.permissions.readable);
  EXPECT_FALSE(last_byte.translation.permissions.writable);
  EXPECT_FALSE(last_byte.translation.permissions.executable);
  EXPECT_EQ(translator.translate(aperture_end, 1, VmAccessKind::Write).outcome,
            VmAccessOutcome::Faulted);
  const VmTranslationResult local = translator.translate(aperture_end + 1, 1, VmAccessKind::Read);
  ASSERT_TRUE(local);
  EXPECT_EQ(local.translation.domain, VmMemoryDomain::Local);
  EXPECT_EQ(local.translation.address, aperture_end + 1);
}

TEST(GpuVmTranslation, GpuVmRejectsUnusableGartPublications) {
  GpuVm gpu_vm;
  auto physical = std::make_shared<TestPhysicalMemory>();
  ASSERT_TRUE(gpu_vm.initialize_gart_address_space());

  EXPECT_FALSE(gpu_vm.publish_gart({}, physical));
  EXPECT_FALSE(gpu_vm.publish_gart(
      {.page_table_base = kSnooped, .aperture_start = 0x10000, .aperture_end = 0x10fff}, physical));
  EXPECT_FALSE(gpu_vm.publish_gart(
      {.page_table_base = 0x1008, .aperture_start = 0x10000, .aperture_end = 0x10fff}, physical));
  EXPECT_FALSE(gpu_vm.publish_gart(
      {.page_table_base = 0x1000, .aperture_start = 0x10001, .aperture_end = 0x10fff}, physical));
  EXPECT_FALSE(gpu_vm.publish_gart(
      {.page_table_base = 0x1000, .aperture_start = 0x11000, .aperture_end = 0x10fff}, physical));
  EXPECT_FALSE(gpu_vm.publish_gart(
      {.page_table_base = 0x1000, .aperture_start = 0x10000, .aperture_end = 0x10ffe}, physical));
  ASSERT_TRUE(gpu_vm.lookup(gpu_vm.gart_address_space()));
  EXPECT_FALSE(gpu_vm.lookup(gpu_vm.gart_address_space())->ready);
}

TEST(GpuVmTranslation, GpuVmUsesTranslatedBindingAndAdvancesEpochOnRootReplacement) {
  GpuMemory compatibility_memory("memory");
  GpuVm gpu_vm;
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

TEST(GpuVmTranslation, RootReplacementRotatesCacheWriteProofAndLineage) {
  GpuVm gpu_vm;
  auto physical = std::make_shared<TestPhysicalMemory>();
  auto translator = std::make_shared<ByteTranslator>();
  const AddressSpaceHandle handle = gpu_vm.register_translated(7, translator, physical);
  ASSERT_TRUE(handle);
  auto old = gpu_vm.snapshot_pinned(handle);
  ASSERT_TRUE(old);
  EXPECT_EQ(old->cache_write_sequence(), 0u);
  ASSERT_TRUE(old->admit_access(0x100, 4, VmAccessKind::Write));
  EXPECT_EQ(old->cache_write_sequence(), 1u);

  // An explicit root replacement starts a new proof even when it reuses the
  // translator and physical objects. The retired pinned view keeps its proof.
  ASSERT_TRUE(gpu_vm.replace_translated(handle, translator, physical));
  auto replacement = gpu_vm.snapshot_pinned(handle);
  ASSERT_TRUE(replacement);
  EXPECT_EQ(replacement->cache_write_sequence(), 0u);
  EXPECT_EQ(old->cache_write_sequence(), 1u);
  EXPECT_FALSE(old->same_root_lineage(*replacement));

  ASSERT_TRUE(gpu_vm.invalidate(handle));
  auto invalidated = gpu_vm.snapshot_pinned(handle);
  ASSERT_TRUE(invalidated);
  EXPECT_TRUE(replacement->same_root_lineage(*invalidated));
  EXPECT_EQ(invalidated->cache_write_sequence(), 0u);
}

TEST(GpuVmTranslation, InvalidationDrainsAnInFlightAccessAndRevokesOldSnapshots) {
  GpuVm gpu_vm;
  auto physical = std::make_shared<BlockingPhysicalMemory>();
  auto translator = std::make_shared<ByteTranslator>();
  const AddressSpaceHandle handle = gpu_vm.register_translated(7, translator, physical);
  ASSERT_TRUE(handle);
  const std::optional<GpuVmAccess> old_access = gpu_vm.snapshot(handle);
  ASSERT_TRUE(old_access);

  std::array<std::byte, 2> value{};
  auto read = std::async(std::launch::async,
                         [&] { return old_access->read(0, std::span<std::byte>(value)); });
  const bool first_read_entered = physical->wait_for_first_read(std::chrono::seconds(2));
  if (!first_read_entered)
    physical->release();
  ASSERT_TRUE(first_read_entered);
  auto invalidation = std::async(std::launch::async, [&] { return gpu_vm.invalidate(handle); });
  const auto invalidation_before_release = invalidation.wait_for(std::chrono::milliseconds(50));

  physical->release();
  EXPECT_EQ(read.get(), VmAccessOutcome::Complete);
  EXPECT_EQ(invalidation_before_release, std::future_status::timeout);
  EXPECT_TRUE(invalidation.get());
  EXPECT_EQ(value, (std::array<std::byte, 2>{std::byte{1}, std::byte{2}}));

  std::array<std::byte, 1> stale_value{std::byte{0x5a}};
  EXPECT_EQ(old_access->read(0, stale_value), VmAccessOutcome::Revoked);
  EXPECT_EQ(stale_value[0], std::byte{0x5a});
  const std::optional<GpuVmAccess> current_access = gpu_vm.snapshot(handle);
  ASSERT_TRUE(current_access);
  EXPECT_NE(current_access->cache_namespace(), old_access->cache_namespace());
  EXPECT_EQ(current_access->read(0, stale_value), VmAccessOutcome::Complete);
}

TEST(GpuVmTranslation, AccessBatchRefreshesRevokedSnapshots) {
  // A callback between operations must be able to invalidate the binding while
  // its quantum cache remains alive. Keep this regression bounded.
  const pid_t child = fork();
  ASSERT_GE(child, 0);
  if (child == 0) {
    alarm(10);
    GpuVm gpu_vm;
    auto physical = std::make_shared<TestPhysicalMemory>();
    auto translator = std::make_shared<ByteTranslator>();
    physical->system[0] = std::byte{0x2a};
    const AddressSpaceHandle handle = gpu_vm.register_translated(7, translator, physical);
    if (!handle)
      _exit(1);

    const GpuVmAccessBatchGuard batch;
    const GpuVmAccess *first = gpu_vm.borrow_snapshot_vmid(7);
    if (first == nullptr)
      _exit(2);
    std::array<std::byte, 1> value{};
    if (first->read(0, value) != VmAccessOutcome::Complete || value[0] != std::byte{0x2a})
      _exit(3);
    auto invalidation = std::async(std::launch::async, [&] { return gpu_vm.invalidate(handle); });
    if (!invalidation.get())
      _exit(4);

    const GpuVmAccess *repeated = gpu_vm.borrow_snapshot_vmid(7);
    if (!repeated || repeated == first || repeated->cache_namespace() == first->cache_namespace() ||
        repeated->read(0, value) != VmAccessOutcome::Complete || value[0] != std::byte{0x2a})
      _exit(5);
    // The old pointer stays alive but loses authority to access backing.
    value[0] = std::byte{0x5a};
    if (first->read(0, value) != VmAccessOutcome::Revoked || value[0] != std::byte{0x5a})
      _exit(6);
    if (!gpu_vm.invalidate(handle))
      _exit(7);
    const GpuVmAccess *by_handle = gpu_vm.borrow_snapshot(handle);
    if (!by_handle || by_handle == repeated || !by_handle->is_current() || repeated->is_current())
      _exit(8);
    _exit(0);
  }

  int status = 0;
  ASSERT_EQ(waitpid(child, &status, 0), child);
  ASSERT_TRUE(WIFEXITED(status)) << "child status: " << status;
  EXPECT_EQ(WEXITSTATUS(status), 0);
}

TEST(GpuVmTranslation, AccessBatchRefreshesReusedVmidAndKeepsHandleRoutingDistinct) {
  GpuVm gpu_vm;
  auto first_physical = std::make_shared<TestPhysicalMemory>();
  auto second_physical = std::make_shared<TestPhysicalMemory>();
  auto translator = std::make_shared<ByteTranslator>();
  first_physical->system[0] = std::byte{0x11};
  second_physical->system[0] = std::byte{0x22};
  const AddressSpaceHandle routed = gpu_vm.register_translated(7, translator, first_physical);
  const AddressSpaceHandle unrouted =
      gpu_vm.register_unrouted_address_space(7, translator, second_physical);
  ASSERT_TRUE(routed);
  ASSERT_TRUE(unrouted);

  const GpuVmAccessBatchGuard batch;
  const GpuVmAccess *explicit_handle = gpu_vm.borrow_snapshot(unrouted);
  ASSERT_NE(explicit_handle, nullptr);
  const GpuVmAccess *first = gpu_vm.borrow_snapshot_vmid(7);
  ASSERT_NE(first, nullptr);
  EXPECT_NE(first, explicit_handle);
  std::array<std::byte, 1> value{};
  EXPECT_EQ(first->read(0, value), VmAccessOutcome::Complete);
  EXPECT_EQ(value[0], std::byte{0x11});
  EXPECT_EQ(explicit_handle->read(0, value), VmAccessOutcome::Complete);
  EXPECT_EQ(value[0], std::byte{0x22});

  ASSERT_TRUE(gpu_vm.unregister_address_space(routed));
  EXPECT_EQ(gpu_vm.borrow_snapshot_vmid(7), nullptr);
  const AddressSpaceHandle replacement = gpu_vm.register_translated(7, translator, second_physical);
  ASSERT_TRUE(replacement);
  const GpuVmAccess *repeated = gpu_vm.borrow_snapshot_vmid(7);
  ASSERT_NE(repeated, nullptr);
  EXPECT_NE(repeated, first);
  EXPECT_EQ(repeated->cache_namespace().address_space, replacement);
  EXPECT_FALSE(first->is_current());
  EXPECT_EQ(repeated->read(0, value), VmAccessOutcome::Complete);
  EXPECT_EQ(value[0], std::byte{0x22});
  {
    const GpuVmAccessBatchGuard nested;
    EXPECT_EQ(gpu_vm.borrow_snapshot_vmid(7), repeated);
  }
  EXPECT_EQ(gpu_vm.borrow_snapshot_vmid(7), repeated);
}

TEST(GpuVmTranslation, AccessBatchDistinguishesReusedVmStorage) {
  const GpuVmAccessBatchGuard batch;
  std::optional<GpuVm> gpu_vm(std::in_place);
  auto first_physical = std::make_shared<TestPhysicalMemory>();
  auto second_physical = std::make_shared<TestPhysicalMemory>();
  auto translator = std::make_shared<ByteTranslator>();
  first_physical->system[0] = std::byte{0x11};
  second_physical->system[0] = std::byte{0x22};
  const AddressSpaceHandle first_handle =
      gpu_vm->register_translated(7, translator, first_physical);
  ASSERT_TRUE(first_handle);
  const GpuVmAccess *first = gpu_vm->borrow_snapshot_vmid(7);
  ASSERT_NE(first, nullptr);

  gpu_vm.emplace();
  const AddressSpaceHandle replacement =
      gpu_vm->register_translated(7, translator, second_physical);
  ASSERT_TRUE(replacement);
  ASSERT_EQ(replacement, first_handle);
  const GpuVmAccess *repeated = gpu_vm->borrow_snapshot_vmid(7);
  ASSERT_NE(repeated, nullptr);
  EXPECT_NE(repeated, first);
  std::array<std::byte, 1> value{};
  EXPECT_EQ(repeated->read(0, value), VmAccessOutcome::Complete);
  EXPECT_EQ(value[0], std::byte{0x22});
  const GpuVmAccess *by_handle = gpu_vm->borrow_snapshot(replacement);
  ASSERT_NE(by_handle, nullptr);
  EXPECT_EQ(by_handle->read(0, value), VmAccessOutcome::Complete);
  EXPECT_EQ(value[0], std::byte{0x22});
}

TEST(GpuVmTranslation, ScopedVmidOperationBorrowsTheQuantumSnapshot) {
  GpuVm gpu_vm;
  auto physical = std::make_shared<TestPhysicalMemory>();
  auto translator = std::make_shared<ByteTranslator>();
  physical->system[0] = std::byte{0x2a};
  const AddressSpaceHandle handle = gpu_vm.register_translated(7, translator, physical);
  ASSERT_TRUE(handle);

  const GpuVmAccess *first = nullptr;
  {
    const GpuVmAccessBatchGuard batch;
    EXPECT_TRUE(gpu_vm.with_vmid_snapshot(7, [&](const GpuVmAccess *access) {
      first = access;
      std::array<std::byte, 1> value{};
      return access != nullptr && access->read(0, value) == VmAccessOutcome::Complete &&
             value[0] == std::byte{0x2a};
    }));
    EXPECT_TRUE(
        gpu_vm.with_vmid_snapshot(7, [&](const GpuVmAccess *access) { return access == first; }));
  }

  EXPECT_TRUE(
      gpu_vm.with_vmid_snapshot(7, [](const GpuVmAccess *access) { return access != nullptr; }));
}

TEST(GpuVmTranslation, PublicSnapshotsRefreshQueueMetadataInsideAccessBatch) {
  GpuVm gpu_vm;
  const AddressSpaceHandle handle = gpu_vm.register_translated(
      7, std::make_shared<ByteTranslator>(), std::make_shared<TestPhysicalMemory>());
  ASSERT_TRUE(handle);
  const GpuVmAccessBatchGuard batch;
  const GpuVmAccess *borrowed_handle = gpu_vm.borrow_snapshot(handle);
  const GpuVmAccess *borrowed_vmid = gpu_vm.borrow_snapshot_vmid(7);
  ASSERT_NE(borrowed_handle, nullptr);
  ASSERT_NE(borrowed_vmid, nullptr);
  const auto initial_handle = gpu_vm.snapshot(handle);
  const auto initial_vmid = gpu_vm.snapshot_vmid(7);
  ASSERT_TRUE(initial_handle);
  ASSERT_TRUE(initial_vmid);
  EXPECT_EQ(initial_handle->info().queue_references, 0u);
  EXPECT_EQ(initial_vmid->info().queue_references, 0u);

  ASSERT_TRUE(gpu_vm.retain_queue(handle));
  const auto retained_handle = gpu_vm.snapshot(handle);
  const auto retained_vmid = gpu_vm.snapshot_vmid(7);
  ASSERT_TRUE(retained_handle);
  ASSERT_TRUE(retained_vmid);
  EXPECT_EQ(retained_handle->info().queue_references, 1u);
  EXPECT_EQ(retained_vmid->info().queue_references, 1u);
  EXPECT_EQ(initial_handle->info().queue_references, 0u);
  EXPECT_EQ(initial_vmid->info().queue_references, 0u);

  ASSERT_TRUE(gpu_vm.release_queue(handle));
  const auto released_handle = gpu_vm.snapshot(handle);
  const auto released_vmid = gpu_vm.snapshot_vmid(7);
  ASSERT_TRUE(released_handle);
  ASSERT_TRUE(released_vmid);
  EXPECT_EQ(released_handle->info().queue_references, 0u);
  EXPECT_EQ(released_vmid->info().queue_references, 0u);
  EXPECT_EQ(retained_handle->info().queue_references, 1u);
  EXPECT_EQ(retained_vmid->info().queue_references, 1u);
  EXPECT_TRUE(retained_handle->is_current());
  EXPECT_TRUE(retained_vmid->is_current());
  EXPECT_EQ(gpu_vm.borrow_snapshot(handle), borrowed_handle);
  EXPECT_EQ(gpu_vm.borrow_snapshot_vmid(7), borrowed_vmid);
}

TEST(GpuVmTranslation, AccessBatchAllowsSnapshotMissDuringInvalidation) {
  // Isolate the cross-binding lock-order case so a regression becomes a
  // bounded failure instead of hanging the test process.
  const pid_t child = fork();
  ASSERT_GE(child, 0);
  if (child == 0) {
    alarm(10);
    GpuVm gpu_vm;
    auto first_physical = std::make_shared<TestPhysicalMemory>();
    auto second_physical = std::make_shared<TestPhysicalMemory>();
    auto translator = std::make_shared<ByteTranslator>();
    first_physical->system[0] = std::byte{0x11};
    second_physical->system[0] = std::byte{0x22};
    const AddressSpaceHandle first = gpu_vm.register_translated(7, translator, first_physical);
    const AddressSpaceHandle second =
        gpu_vm.register_unrouted_address_space(8, translator, second_physical);
    if (!first || !second)
      _exit(1);

    std::atomic<bool> invalidation_started{false};
    std::atomic<bool> invalidation_complete{false};
    std::thread invalidator;
    {
      const GpuVmAccessBatchGuard batch;
      auto first_access = gpu_vm.snapshot(first);
      if (!first_access)
        _exit(2);
      std::array<std::byte, 1> value{};
      if (first_access->read(0, value) != VmAccessOutcome::Complete || value[0] != std::byte{0x11})
        _exit(3);

      invalidator = std::thread([&] {
        invalidation_started.store(true, std::memory_order_release);
        const bool invalidated = gpu_vm.invalidate(first);
        invalidation_complete.store(invalidated, std::memory_order_release);
      });
      while (!invalidation_started.load(std::memory_order_acquire))
        std::this_thread::yield();
      // A miss for another binding must make progress even if invalidation
      // has already acquired the registry lock.
      auto second_access = gpu_vm.snapshot(second);
      if (!second_access)
        _exit(5);
      value[0] = std::byte{0};
      if (second_access->read(0, value) != VmAccessOutcome::Complete || value[0] != std::byte{0x22})
        _exit(6);
    }

    invalidator.join();
    if (!invalidation_complete.load(std::memory_order_acquire))
      _exit(7);
    _exit(0);
  }

  int status = 0;
  ASSERT_EQ(waitpid(child, &status, 0), child);
  ASSERT_TRUE(WIFEXITED(status)) << "child status: " << status;
  EXPECT_EQ(WEXITSTATUS(status), 0);
}

TEST(GpuVmTranslation, RevokedSnapshotIsDistinctFromTransientUnavailability) {
  GpuVm gpu_vm;
  auto physical = std::make_shared<TestPhysicalMemory>();
  const AddressSpaceHandle handle =
      gpu_vm.register_translated(7, std::make_shared<ByteTranslator>(), physical);
  ASSERT_TRUE(handle);
  const std::optional<GpuVmAccess> access = gpu_vm.snapshot_pinned(handle);
  ASSERT_TRUE(access);
  ASSERT_TRUE(gpu_vm.invalidate(handle));

  std::array<std::byte, sizeof(uint32_t)> bytes{};
  EXPECT_EQ(access->translate(0, bytes.size(), VmAccessKind::Read).outcome,
            VmAccessOutcome::Revoked);
  EXPECT_EQ(access->query_access(0, bytes.size(), VmAccessKind::Read), VmAccessOutcome::Revoked);
  EXPECT_EQ(access->probe(0, bytes.size(), VmAccessKind::Read), VmAccessOutcome::Revoked);
  EXPECT_EQ(access->read(0, bytes), VmAccessOutcome::Revoked);
  EXPECT_EQ(access->write(0, bytes), VmAccessOutcome::Revoked);
  EXPECT_EQ(access->atomic_load(0, sizeof(uint32_t)).outcome, VmAccessOutcome::Revoked);
  EXPECT_EQ(access->atomic_store(0, sizeof(uint32_t), 1), VmAccessOutcome::Revoked);
  EXPECT_EQ(access->compare_exchange(0, sizeof(uint32_t), 0, 1).outcome, VmAccessOutcome::Revoked);
  EXPECT_EQ(access->atomic_modify(0, sizeof(uint32_t), [](std::span<std::byte>) {}),
            VmAccessOutcome::Revoked);
}

TEST(GpuVmTranslation, DeviceGartPublishesOnInvalidateAndRejectsItsHandleAfterReset) {
  GpuVm gpu_vm;
  auto physical = std::make_shared<TestPhysicalMemory>();
  constexpr uint64_t first_table = 0x1000;
  constexpr uint64_t second_table = 0x2000;
  constexpr uint64_t aperture = 0x10000;
  physical->store_qword(VmMemoryDomain::Local, first_table,
                        0x9000 | kPdePte | kValid | kSystem | kReadable);
  physical->store_qword(VmMemoryDomain::Local, second_table,
                        0xa000 | kPdePte | kValid | kSystem | kReadable);
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
  EXPECT_FALSE(gpu_vm.replace_translated(handle, std::make_shared<ByteTranslator>(), physical));
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
  util::DistributedSharedMutex page_table_mutex;
  constexpr uint32_t vmid = 7;
  constexpr uint64_t virtual_address = 0x4123;
  std::array<uint8_t, KfdProcess::kPageSize> backing{};
  backing[0x123] = 0x5a;
  page_table[virtual_address >> KfdProcess::kPageShift] = {backing.data(), Mtype::CC};
  GpuVm gpu_vm;
  LegacyGpuVmAdapter legacy_vm(gpu_vm, &memory);

  const AddressSpaceHandle handle =
      legacy_vm.register_address_space(vmid, &page_table, &page_table_mutex);

  ASSERT_TRUE(handle);
  ASSERT_TRUE(gpu_vm.lookup(handle));
  EXPECT_TRUE(gpu_vm.lookup(handle)->legacy_cache_compatible);
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
}

TEST(GpuVmTranslation, LegacyPrivateDwordBatchPreflightsRawWritePermission) {
  constexpr uint32_t vmid = 7;
  constexpr uint64_t address = 0x4000;
  std::array<uint8_t, KfdProcess::kPageSize> backing{};
  LegacyPageTable page_table;
  page_table[address >> KfdProcess::kPageShift] = {backing.data(), Mtype::UC, backing.size(), 0};
  page_table[address >> KfdProcess::kPageShift].host_extents.front().owner =
      LegacyHostExtentOwner::DriverSealedRam;
  util::DistributedSharedMutex page_table_mutex;
  auto request_mutex = std::make_shared<util::DistributedSharedMutex>();
  auto gate = std::make_shared<VmRawBackingWriteGate>();
  GpuMemory memory("raw_batch");
  GpuVm gpu_vm;
  LegacyGpuVmAdapter legacy_vm(gpu_vm, &memory);
  bool deny_second = true;
  const AddressSpaceHandle handle = legacy_vm.register_address_space(
      vmid, {.page_table = &page_table,
             .page_table_mutex = &page_table_mutex,
             .request_mutex = request_mutex,
             .access_admission =
                 [&](uint64_t request_address, std::size_t, VmAccessKind) {
                   return VmAccessAdmission{.direct_backing_write_allowed =
                                                !(deny_second && request_address == address + 4)};
                 },
             .raw_write_gate = gate});
  ASSERT_TRUE(handle);
  const auto access = gpu_vm.snapshot(handle);
  ASSERT_TRUE(access);
  const VmAccessAdmission admission = access->admit_access(address + 4, 4, VmAccessKind::Write);
  EXPECT_TRUE(admission);
  EXPECT_FALSE(admission.direct_backing_write_allowed);

  const std::array<uint32_t, 2> values{0x12345678u, 0x87654321u};
  const std::array<VmRamDwordStore, 2> stores{{
      {address, reinterpret_cast<const uint8_t *>(&values[0])},
      {address + 4, reinterpret_cast<const uint8_t *>(&values[1])},
  }};
  EXPECT_FALSE(access->try_write_private_dwords(stores, Mtype::UC, Mtype::UC));
  EXPECT_TRUE(std::ranges::all_of(backing, [](uint8_t value) { return value == 0; }));

  deny_second = false;
  auto closed_store = gate->try_close(address + 4, sizeof(uint32_t));
  ASSERT_TRUE(closed_store);
  EXPECT_FALSE(access->try_write_private_dwords(stores, Mtype::UC, Mtype::UC));
  EXPECT_TRUE(std::ranges::all_of(backing, [](uint8_t value) { return value == 0; }));
  closed_store.reset();

  auto closed_unrelated = gate->try_close(address + KfdProcess::kPageSize, KfdProcess::kPageSize);
  ASSERT_TRUE(closed_unrelated);
  ASSERT_TRUE(access->try_write_private_dwords(stores, Mtype::UC, Mtype::UC));
  EXPECT_EQ(std::memcmp(backing.data(), values.data(), sizeof(values)), 0);
  EXPECT_TRUE(gpu_vm.unregister_address_space(handle));
}

TEST(GpuVmTranslation, RawWriteGateTracksMovedAndOutOfOrderTokens) {
  VmRawBackingWriteGate gate;
  auto first = gate.try_acquire(0x4000, 4);
  auto second = gate.try_acquire(0x4004, 4);
  ASSERT_TRUE(first);
  ASSERT_TRUE(second);
  EXPECT_FALSE(gate.try_close(0x4000, 8));

  auto moved = std::move(*first);
  first.reset();
  moved = VmRawBackingWriteGate::Permit{};
  auto closed_first = gate.try_close(0x4000, 4);
  ASSERT_TRUE(closed_first);
  second.reset();
  EXPECT_FALSE(gate.try_acquire(0x4000, 4));

  auto closed_second = gate.try_close(0x5000, 4);
  ASSERT_TRUE(closed_second);
  *closed_second = std::move(*closed_first);
  closed_first.reset();
  EXPECT_TRUE(gate.try_acquire(0x5000, 4));
  EXPECT_FALSE(gate.try_acquire(0x4000, 4));
  closed_second.reset();
  EXPECT_TRUE(gate.try_acquire(0x4000, 4));
}

TEST(GpuVmTranslation, LegacyRawWritePermitSpansAdmissionAndBackingEffect) {
  constexpr uint32_t vmid = 7;
  constexpr uint64_t address = 0x4000;
  constexpr uint64_t status_address = address + KfdProcess::kPageSize;
  alignas(4096) std::array<uint8_t, 2 * KfdProcess::kPageSize> backing{};
  LegacyPageTable page_table;
  page_table[address >> KfdProcess::kPageShift] = {backing.data(), Mtype::UC, KfdProcess::kPageSize,
                                                   0};
  page_table[status_address >> KfdProcess::kPageShift] = {backing.data() + KfdProcess::kPageSize,
                                                          Mtype::UC, KfdProcess::kPageSize, 0};
  util::DistributedSharedMutex page_table_mutex;
  auto gate = std::make_shared<VmRawBackingWriteGate>();
  std::mutex pause_mutex;
  std::condition_variable pause_condition;
  bool entered = false;
  bool resume = false;
  std::atomic<uint32_t> admissions{0};
  GpuMemory memory("raw_permit");
  GpuVm gpu_vm;
  LegacyGpuVmAdapter legacy_vm(gpu_vm, &memory);
  const AddressSpaceHandle handle = legacy_vm.register_address_space(
      vmid, {.page_table = &page_table,
             .page_table_mutex = &page_table_mutex,
             .request_mutex = {},
             .access_admission =
                 [&](uint64_t, std::size_t, VmAccessKind) {
                   if (admissions.fetch_add(1) == 0) {
                     std::unique_lock lock(pause_mutex);
                     entered = true;
                     pause_condition.notify_all();
                     pause_condition.wait(lock, [&] { return resume; });
                   }
                   return VmAccessAdmission{};
                 },
             .raw_write_gate = gate});
  ASSERT_TRUE(handle);
  const auto access = gpu_vm.snapshot(handle);
  ASSERT_TRUE(access);
  const uint32_t first = 0x12345678u;
  const auto first_bytes = std::as_bytes(std::span(&first, 1));
  VmAccessOutcome writer_outcome = VmAccessOutcome::Malformed;
  std::thread writer([&] { writer_outcome = access->write(address, first_bytes); });
  {
    std::unique_lock lock(pause_mutex);
    const bool paused =
        pause_condition.wait_for(lock, std::chrono::seconds(5), [&] { return entered; });
    EXPECT_TRUE(paused);
    if (paused) {
      EXPECT_FALSE(gate->try_close(address, KfdProcess::kPageSize));
    }
    resume = true;
  }
  pause_condition.notify_all();
  writer.join();
  EXPECT_EQ(writer_outcome, VmAccessOutcome::Complete);
  EXPECT_EQ(std::memcmp(backing.data(), &first, sizeof(first)), 0);

  auto unrelated_writer = gate->try_acquire(status_address, sizeof(uint32_t));
  ASSERT_TRUE(unrelated_writer);
  auto closed = gate->try_close(address, KfdProcess::kPageSize);
  ASSERT_TRUE(closed);
  const uint32_t second = 0x87654321u;
  const auto second_bytes = std::as_bytes(std::span(&second, 1));
  EXPECT_EQ(access->write(address, second_bytes), VmAccessOutcome::Unavailable);
  EXPECT_EQ(std::memcmp(backing.data(), &first, sizeof(first)), 0);

  // Queue exception status lives in another allocation in the same process.
  // Controlled publication must not drop its synchronous atomic notification.
  constexpr uint32_t exception_status = 0x80000001u;
  EXPECT_EQ(access->atomic_store(status_address, sizeof(exception_status), exception_status),
            VmAccessOutcome::Complete);
  EXPECT_EQ(std::memcmp(backing.data() + KfdProcess::kPageSize, &exception_status,
                        sizeof(exception_status)),
            0);

  unrelated_writer.reset();
  closed.reset();
  EXPECT_EQ(access->write(address, second_bytes), VmAccessOutcome::Complete);
  EXPECT_EQ(std::memcmp(backing.data(), &second, sizeof(second)), 0);
  EXPECT_TRUE(gpu_vm.unregister_address_space(handle));
}

TEST(GpuVmTranslation, LegacyProbeRequiresEveryPageToHaveAGpuMapping) {
  GpuMemory memory("memory");
  KfdProcess::PageTable page_table;
  util::DistributedSharedMutex page_table_mutex;
  constexpr uint32_t vmid = 7;
  constexpr uint64_t mapped_address = 0x4000;
  constexpr uint64_t unmapped_address = 0;
  constexpr uint64_t mapped_page_end = mapped_address + KfdProcess::kPageSize;
  std::array<uint8_t, KfdProcess::kPageSize> backing{};
  page_table[mapped_address >> KfdProcess::kPageShift] = {backing.data(), Mtype::CC};
  GpuVm gpu_vm;
  LegacyGpuVmAdapter legacy_vm(gpu_vm, &memory);

  const AddressSpaceHandle handle =
      legacy_vm.register_address_space(vmid, &page_table, &page_table_mutex);
  ASSERT_TRUE(handle);
  const std::optional<GpuVmAccess> access = gpu_vm.snapshot(handle);
  ASSERT_TRUE(access);

  EXPECT_EQ(access->probe(mapped_address, sizeof(uint32_t), VmAccessKind::Read),
            VmAccessOutcome::Complete);
  EXPECT_EQ(access->probe(unmapped_address, sizeof(uint32_t), VmAccessKind::Read),
            VmAccessOutcome::Faulted);
  EXPECT_EQ(access->probe(mapped_page_end - sizeof(uint32_t), sizeof(uint64_t), VmAccessKind::Read),
            VmAccessOutcome::Faulted);

  EXPECT_TRUE(gpu_vm.unregister_address_space(handle));
}

TEST(GpuVmTranslation, LegacyProbeRejectsWritesAndAtomicsToReadOnlyBacking) {
  GpuMemory memory("memory");
  KfdProcess::PageTable page_table;
  util::DistributedSharedMutex page_table_mutex;
  constexpr uint32_t vmid = 7;
  constexpr uint64_t virtual_address = 0x4000;
  HostPage backing;
  ASSERT_NE(backing.data(), nullptr);
  page_table[virtual_address >> KfdProcess::kPageShift] = {backing.data(), Mtype::CC};
  RecordingFaultReporter reporter;
  GpuVm gpu_vm;
  LegacyGpuVmAdapter legacy_vm(gpu_vm, &memory);
  const AddressSpaceHandle handle =
      legacy_vm.register_address_space(vmid, {.page_table = &page_table,
                                              .page_table_mutex = &page_table_mutex,
                                              .page_table_generation = nullptr,
                                              .request_mutex = {},
                                              .client_pid = 0,
                                              .client_mem_fd = -1,
                                              .passthrough = false,
                                              .fault_reporter = &reporter});
  ASSERT_TRUE(handle);
  const std::optional<GpuVmAccess> access = gpu_vm.snapshot(handle);
  ASSERT_TRUE(access);
  ASSERT_EQ(mprotect(backing.data(), KfdProcess::kPageSize, PROT_READ), 0);

  EXPECT_EQ(access->query_access(virtual_address, KfdProcess::kPageSize, VmAccessKind::Read),
            VmAccessOutcome::Complete);
  EXPECT_EQ(access->query_access(virtual_address, KfdProcess::kPageSize, VmAccessKind::Write),
            VmAccessOutcome::Faulted);
  EXPECT_EQ(access->query_access(virtual_address, KfdProcess::kPageSize, VmAccessKind::Atomic),
            VmAccessOutcome::Faulted);
  EXPECT_TRUE(reporter.addresses.empty());

  EXPECT_EQ(access->probe(virtual_address, KfdProcess::kPageSize, VmAccessKind::Write),
            VmAccessOutcome::Faulted);
  EXPECT_EQ(access->probe(virtual_address, KfdProcess::kPageSize, VmAccessKind::Atomic),
            VmAccessOutcome::Faulted);
  EXPECT_EQ(reporter.vmids, (std::vector<uint32_t>{vmid, vmid}));
  EXPECT_EQ(reporter.addresses, (std::vector<uint64_t>{virtual_address, virtual_address}));

  ASSERT_EQ(mprotect(backing.data(), KfdProcess::kPageSize, PROT_READ | PROT_WRITE), 0);
  EXPECT_TRUE(gpu_vm.unregister_address_space(handle));
}

TEST(GpuVmTranslation, QueryAccessDoesNotReportAnExpectedProvisioningMiss) {
  GpuMemory memory("memory");
  KfdProcess::PageTable page_table;
  util::DistributedSharedMutex page_table_mutex;
  constexpr uint32_t vmid = 7;
  constexpr uint64_t unmapped_address = 0x4000;
  RecordingFaultReporter reporter;
  GpuVm gpu_vm;
  LegacyGpuVmAdapter legacy_vm(gpu_vm, &memory);
  const AddressSpaceHandle handle =
      legacy_vm.register_address_space(vmid, {.page_table = &page_table,
                                              .page_table_mutex = &page_table_mutex,
                                              .page_table_generation = nullptr,
                                              .request_mutex = {},
                                              .client_pid = 0,
                                              .client_mem_fd = -1,
                                              .passthrough = false,
                                              .fault_reporter = &reporter});
  ASSERT_TRUE(handle);
  const std::optional<GpuVmAccess> access = gpu_vm.snapshot(handle);
  ASSERT_TRUE(access);

  EXPECT_EQ(access->query_access(unmapped_address, KfdProcess::kPageSize, VmAccessKind::Atomic),
            VmAccessOutcome::Faulted);
  EXPECT_TRUE(reporter.addresses.empty());

  EXPECT_EQ(access->probe(unmapped_address, KfdProcess::kPageSize, VmAccessKind::Atomic),
            VmAccessOutcome::Faulted);
  EXPECT_EQ(reporter.vmids, (std::vector<uint32_t>{vmid}));
  EXPECT_EQ(reporter.addresses, (std::vector<uint64_t>{unmapped_address}));
}

TEST(GpuVmTranslation, LegacyExecuteUsesFetchableCompatibilityBacking) {
  GpuMemory memory("memory");
  KfdProcess::PageTable page_table;
  util::DistributedSharedMutex page_table_mutex;
  constexpr uint32_t vmid = 7;
  constexpr uint64_t executable_address = 0x9000;
  constexpr uint32_t instruction = 0xbf800000;
  memory.write32(executable_address, instruction);
  GpuVm gpu_vm;
  LegacyGpuVmAdapter legacy_vm(gpu_vm, &memory);

  const AddressSpaceHandle handle =
      legacy_vm.register_address_space(vmid, &page_table, &page_table_mutex);
  ASSERT_TRUE(handle);
  const std::optional<GpuVmAccess> access = gpu_vm.snapshot(handle);
  ASSERT_TRUE(access);

  EXPECT_EQ(access->probe(executable_address, sizeof(instruction), VmAccessKind::Execute),
            VmAccessOutcome::Complete);
  std::array<std::byte, sizeof(instruction)> bytes{};
  EXPECT_EQ(access->read(executable_address, bytes, VmAccessKind::Execute),
            VmAccessOutcome::Complete);
  EXPECT_EQ(std::bit_cast<uint32_t>(bytes), instruction);
  EXPECT_EQ(access->probe(0x100, sizeof(instruction), VmAccessKind::Execute),
            VmAccessOutcome::Faulted);

  EXPECT_TRUE(gpu_vm.unregister_address_space(handle));
}

TEST(GpuVmTranslation, LegacySnapshotRetainsStorageButIsRevokedAcrossUnregister) {
  auto original_memory = std::make_shared<GpuMemory>("original-memory");
  auto replacement_memory = std::make_shared<GpuMemory>("replacement-memory");
  KfdProcess::PageTable original_page_table;
  KfdProcess::PageTable replacement_page_table;
  util::DistributedSharedMutex original_page_table_mutex;
  util::DistributedSharedMutex replacement_page_table_mutex;
  constexpr uint32_t vmid = 7;
  constexpr uint64_t virtual_address = 0x4000;
  std::array<uint8_t, KfdProcess::kPageSize> original_backing{};
  std::array<uint8_t, KfdProcess::kPageSize> replacement_backing{};
  original_backing[0] = 0x3c;
  replacement_backing[0] = 0xa5;
  original_page_table[virtual_address >> KfdProcess::kPageShift] = {original_backing.data(),
                                                                    Mtype::CC};
  replacement_page_table[virtual_address >> KfdProcess::kPageShift] = {replacement_backing.data(),
                                                                       Mtype::CC};
  GpuVm gpu_vm;
  LegacyGpuVmAdapter legacy_vm(gpu_vm, original_memory);

  const AddressSpaceHandle original =
      legacy_vm.register_address_space(vmid, &original_page_table, &original_page_table_mutex);
  ASSERT_TRUE(original);
  std::optional<GpuVmAccess> old_access = gpu_vm.snapshot(original);
  ASSERT_TRUE(old_access);
  EXPECT_FALSE(legacy_vm.set_memory(replacement_memory));

  std::array<std::byte, 1> value{};
  EXPECT_EQ(gpu_vm.read(original, virtual_address, value), VmAccessOutcome::Complete);
  EXPECT_EQ(value[0], std::byte{0x3c});

  EXPECT_TRUE(legacy_vm.unregister_address_space(original));
  EXPECT_FALSE(gpu_vm.snapshot(original));
  std::weak_ptr<GpuMemory> old_memory = original_memory;
  original_memory.reset();
  EXPECT_FALSE(old_memory.expired());
  EXPECT_TRUE(legacy_vm.set_memory(replacement_memory));
  const AddressSpaceHandle replacement = legacy_vm.register_address_space(
      vmid, &replacement_page_table, &replacement_page_table_mutex);
  ASSERT_TRUE(replacement);
  EXPECT_EQ(replacement.slot, original.slot);
  EXPECT_NE(replacement.generation, original.generation);
  EXPECT_EQ(gpu_vm.read(replacement, virtual_address, value), VmAccessOutcome::Complete);
  EXPECT_EQ(value[0], std::byte{0xa5});

  value[0] = std::byte{0x5a};
  EXPECT_EQ(old_access->read(virtual_address, value), VmAccessOutcome::Revoked);
  EXPECT_EQ(value[0], std::byte{0x5a});
  ASSERT_TRUE(gpu_vm.snapshot(replacement));
  EXPECT_NE(old_access->cache_namespace(), gpu_vm.snapshot(replacement)->cache_namespace());

  EXPECT_TRUE(legacy_vm.unregister_address_space(replacement));
  old_access.reset();
  EXPECT_TRUE(old_memory.expired());
}

TEST(GpuVmTranslation, LegacyUnregisterRevokesFaultDeliveryFromRetainedSnapshot) {
  GpuMemory memory("memory");
  KfdProcess::PageTable page_table;
  util::DistributedSharedMutex page_table_mutex;
  RecordingFaultReporter reporter;
  GpuVm gpu_vm;
  LegacyGpuVmAdapter legacy_vm(gpu_vm, &memory);
  const AddressSpaceHandle handle =
      legacy_vm.register_address_space(7, {.page_table = &page_table,
                                           .page_table_mutex = &page_table_mutex,
                                           .page_table_generation = nullptr,
                                           .request_mutex = {},
                                           .client_pid = 0,
                                           .client_mem_fd = -1,
                                           .passthrough = false,
                                           .fault_reporter = &reporter});
  ASSERT_TRUE(handle);
  const std::optional<GpuVmAccess> access = gpu_vm.snapshot(handle);
  ASSERT_TRUE(access);

  ASSERT_TRUE(legacy_vm.unregister_address_space(handle));
  EXPECT_EQ(access->probe(0x4000, sizeof(uint32_t), VmAccessKind::Read), VmAccessOutcome::Revoked);
  EXPECT_TRUE(reporter.addresses.empty());
}

TEST(GpuVmTranslation, LegacyAdapterTeardownRevokesFaultDeliveryFromRetainedSnapshot) {
  GpuMemory memory("memory");
  KfdProcess::PageTable page_table;
  util::DistributedSharedMutex page_table_mutex;
  RecordingFaultReporter reporter;
  GpuVm gpu_vm;
  AddressSpaceHandle handle;
  std::optional<GpuVmAccess> access;
  {
    LegacyGpuVmAdapter legacy_vm(gpu_vm, &memory);
    handle = legacy_vm.register_address_space(7, {.page_table = &page_table,
                                                  .page_table_mutex = &page_table_mutex,
                                                  .page_table_generation = nullptr,
                                                  .request_mutex = {},
                                                  .client_pid = 0,
                                                  .client_mem_fd = -1,
                                                  .passthrough = false,
                                                  .fault_reporter = &reporter});
    ASSERT_TRUE(handle);
    access = gpu_vm.snapshot(handle);
    ASSERT_TRUE(access);
  }

  EXPECT_FALSE(gpu_vm.lookup(handle));
  EXPECT_EQ(access->probe(0x4000, sizeof(uint32_t), VmAccessKind::Read), VmAccessOutcome::Revoked);
  EXPECT_TRUE(reporter.addresses.empty());
}

TEST(GpuVmTranslation, LegacyAdapterTeardownPreservesQueueRetainedBindingWithoutFaultSink) {
  GpuMemory memory("memory");
  KfdProcess::PageTable page_table;
  util::DistributedSharedMutex page_table_mutex;
  RecordingFaultReporter reporter;
  GpuVm gpu_vm;
  AddressSpaceHandle handle;
  std::optional<GpuVmAccess> access;
  {
    LegacyGpuVmAdapter legacy_vm(gpu_vm, &memory);
    handle = legacy_vm.register_address_space(7, {.page_table = &page_table,
                                                  .page_table_mutex = &page_table_mutex,
                                                  .page_table_generation = nullptr,
                                                  .request_mutex = {},
                                                  .client_pid = 0,
                                                  .client_mem_fd = -1,
                                                  .passthrough = false,
                                                  .fault_reporter = &reporter});
    ASSERT_TRUE(handle);
    ASSERT_TRUE(gpu_vm.retain_queue(handle));
    access = gpu_vm.snapshot(handle);
    ASSERT_TRUE(access);
  }

  ASSERT_TRUE(gpu_vm.lookup(handle));
  EXPECT_EQ(gpu_vm.lookup(handle)->queue_references, 1u);
  EXPECT_EQ(access->probe(0x4000, sizeof(uint32_t), VmAccessKind::Read), VmAccessOutcome::Faulted);
  EXPECT_TRUE(reporter.addresses.empty());
  EXPECT_TRUE(gpu_vm.release_queue(handle));
  EXPECT_TRUE(gpu_vm.unregister_address_space(handle));
}

TEST(GpuVmTranslation, LegacyMutationPrunesBindingRevokedByGenericVmReset) {
  GpuMemory memory("memory");
  KfdProcess::PageTable page_table;
  util::DistributedSharedMutex page_table_mutex;
  RecordingFaultReporter reporter;
  GpuVm gpu_vm;
  LegacyGpuVmAdapter legacy_vm(gpu_vm, &memory);
  const AddressSpaceHandle handle =
      legacy_vm.register_address_space(7, {.page_table = &page_table,
                                           .page_table_mutex = &page_table_mutex,
                                           .page_table_generation = nullptr,
                                           .request_mutex = {},
                                           .client_pid = 0,
                                           .client_mem_fd = -1,
                                           .passthrough = false,
                                           .fault_reporter = &reporter});
  ASSERT_TRUE(handle);
  const std::optional<GpuVmAccess> access = gpu_vm.snapshot(handle);
  ASSERT_TRUE(access);

  ASSERT_TRUE(gpu_vm.reset());
  const LegacyGpuVmAdapter &const_legacy_vm = legacy_vm;
  EXPECT_EQ(const_legacy_vm.address_space(7), nullptr);
  EXPECT_FALSE(legacy_vm.set_client_pid(handle, 42));
  EXPECT_EQ(legacy_vm.address_space(7), nullptr);
  EXPECT_EQ(access->probe(0x4000, sizeof(uint32_t), VmAccessKind::Read), VmAccessOutcome::Revoked);
  EXPECT_TRUE(reporter.addresses.empty());
}

TEST(GpuVmTranslation, LegacyBackingRetriesUntilPageTableMappingIsPublished) {
  GpuMemory memory("memory");
  KfdProcess::PageTable page_table;
  util::DistributedSharedMutex page_table_mutex;
  constexpr uint32_t vmid = 7;
  constexpr uint64_t virtual_address = 0x4000;
  GpuVm gpu_vm;
  LegacyGpuVmAdapter legacy_vm(gpu_vm, &memory);
  const AddressSpaceHandle handle =
      legacy_vm.register_address_space(vmid, &page_table, &page_table_mutex);
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
}

TEST(GpuVmTranslation, LegacyAtomicsDoNotFallBackToSparseStorage) {
  GpuMemory memory("memory");
  KfdProcess::PageTable page_table;
  util::DistributedSharedMutex page_table_mutex;
  constexpr uint32_t vmid = 7;
  constexpr uint64_t virtual_address = 0x4000;
  constexpr uint32_t sparse_value = 0x11223344;
  memory.write32(virtual_address, sparse_value);
  GpuVm gpu_vm;
  LegacyGpuVmAdapter legacy_vm(gpu_vm, &memory);
  const AddressSpaceHandle handle =
      legacy_vm.register_address_space(vmid, &page_table, &page_table_mutex);
  ASSERT_TRUE(handle);

  EXPECT_EQ(gpu_vm.atomic_store(handle, virtual_address, sizeof(uint32_t), 0xaabbccdd),
            VmAccessOutcome::Unavailable);
  EXPECT_EQ(memory.read32(virtual_address), sparse_value);

  const AtomicCompareExchangeResult exchanged =
      gpu_vm.compare_exchange(handle, virtual_address, sizeof(uint32_t), sparse_value, 0x55667788);
  EXPECT_EQ(exchanged.outcome, VmAccessOutcome::Unavailable);
  EXPECT_FALSE(exchanged.exchanged);
  EXPECT_EQ(exchanged.observed, 0u);
  EXPECT_EQ(memory.read32(virtual_address), sparse_value);

  EXPECT_TRUE(gpu_vm.unregister_address_space(handle));
}

TEST(GpuVmTranslation, LegacyBackingRejectsMappedPageClippingWithoutPartialTransfer) {
  GpuMemory memory("memory");
  KfdProcess::PageTable page_table;
  util::DistributedSharedMutex page_table_mutex;
  constexpr uint32_t vmid = 7;
  constexpr uint64_t virtual_address = 0x4000;
  constexpr size_t mapped_bytes = 64;
  std::array<uint8_t, mapped_bytes> backing{};
  std::ranges::fill(backing, uint8_t{0x5a});
  page_table[virtual_address >> KfdProcess::kPageShift] = {backing.data(), Mtype::CC,
                                                           backing.size(), 0};
  GpuVm gpu_vm;
  LegacyGpuVmAdapter legacy_vm(gpu_vm, &memory);
  const AddressSpaceHandle handle =
      legacy_vm.register_address_space(vmid, &page_table, &page_table_mutex);
  ASSERT_TRUE(handle);

  std::array<std::byte, mapped_bytes * 2> read_value{};
  std::ranges::fill(read_value, std::byte{0xff});
  EXPECT_EQ(gpu_vm.read(handle, virtual_address, read_value), VmAccessOutcome::Faulted);
  EXPECT_TRUE(
      std::ranges::all_of(read_value, [](std::byte value) { return value == std::byte{0xff}; }));

  std::array<std::byte, mapped_bytes * 2> write_value{};
  std::ranges::fill(write_value, std::byte{0xa5});
  EXPECT_EQ(gpu_vm.write(handle, virtual_address, write_value), VmAccessOutcome::Faulted);
  EXPECT_TRUE(std::ranges::all_of(backing, [](uint8_t value) { return value == 0x5a; }));

  EXPECT_TRUE(gpu_vm.unregister_address_space(handle));
}

TEST(GpuVmTranslation, LegacyAtomicTranslationFaultsReportExactlyOncePerOperation) {
  GpuMemory memory("memory");
  KfdProcess::PageTable page_table;
  util::DistributedSharedMutex page_table_mutex;
  constexpr uint32_t vmid = 7;
  constexpr uint64_t virtual_address = 0x4000;
  std::array<uint8_t, sizeof(uint32_t)> backing{};
  page_table[virtual_address >> KfdProcess::kPageShift] = {backing.data(), Mtype::CC,
                                                           backing.size(), 0};
  RecordingFaultReporter reporter;
  GpuVm gpu_vm;
  LegacyGpuVmAdapter legacy_vm(gpu_vm, &memory);
  const AddressSpaceHandle handle =
      legacy_vm.register_address_space(vmid, {.page_table = &page_table,
                                              .page_table_mutex = &page_table_mutex,
                                              .page_table_generation = nullptr,
                                              .request_mutex = {},
                                              .client_pid = 0,
                                              .client_mem_fd = -1,
                                              .passthrough = false,
                                              .fault_reporter = &reporter});
  ASSERT_TRUE(handle);
  const std::optional<GpuVmAccess> access = gpu_vm.snapshot(handle);
  ASSERT_TRUE(access);

  EXPECT_EQ(access->atomic_load(virtual_address, sizeof(uint64_t)).outcome,
            VmAccessOutcome::Faulted);
  EXPECT_EQ(access->atomic_store(virtual_address, sizeof(uint64_t), 1), VmAccessOutcome::Faulted);
  EXPECT_EQ(access->compare_exchange(virtual_address, sizeof(uint64_t), 0, 1).outcome,
            VmAccessOutcome::Faulted);
  uint32_t mutation_calls = 0;
  EXPECT_EQ(access->atomic_modify(virtual_address, sizeof(uint64_t),
                                  [&](std::span<std::byte>) { ++mutation_calls; }),
            VmAccessOutcome::Faulted);

  EXPECT_EQ(mutation_calls, 0u);
  EXPECT_EQ(reporter.vmids, (std::vector<uint32_t>{vmid, vmid, vmid, vmid}));
  EXPECT_EQ(reporter.addresses, (std::vector<uint64_t>{virtual_address, virtual_address,
                                                       virtual_address, virtual_address}));
  EXPECT_TRUE(gpu_vm.unregister_address_space(handle));
}

TEST(GpuVmTranslation, LegacyBackingFaultsForInaccessibleMappedPage) {
  GpuMemory memory("memory");
  KfdProcess::PageTable page_table;
  util::DistributedSharedMutex page_table_mutex;
  constexpr uint32_t vmid = 7;
  constexpr uint64_t virtual_address = 0x4000;
  HostPage backing;
  ASSERT_NE(backing.data(), nullptr);
  page_table[virtual_address >> KfdProcess::kPageShift] = {backing.data(), Mtype::CC};
  RecordingFaultReporter reporter;
  GpuVm gpu_vm;
  LegacyGpuVmAdapter legacy_vm(gpu_vm, &memory);
  const AddressSpaceHandle handle =
      legacy_vm.register_address_space(vmid, {.page_table = &page_table,
                                              .page_table_mutex = &page_table_mutex,
                                              .page_table_generation = nullptr,
                                              .request_mutex = {},
                                              .client_pid = 0,
                                              .client_mem_fd = -1,
                                              .passthrough = false,
                                              .fault_reporter = &reporter});
  ASSERT_TRUE(handle);
  ASSERT_EQ(mprotect(backing.data(), KfdProcess::kPageSize, PROT_NONE), 0);

  std::array<std::byte, 4> value{};
  EXPECT_EQ(gpu_vm.read(handle, virtual_address, value), VmAccessOutcome::Faulted);
  EXPECT_EQ(gpu_vm.write(handle, virtual_address, value), VmAccessOutcome::Faulted);
  EXPECT_EQ(reporter.vmids, (std::vector<uint32_t>{vmid, vmid}));
  EXPECT_EQ(reporter.addresses, (std::vector<uint64_t>{virtual_address, virtual_address}));

  ASSERT_EQ(mprotect(backing.data(), KfdProcess::kPageSize, PROT_READ | PROT_WRITE), 0);
  EXPECT_TRUE(gpu_vm.unregister_address_space(handle));
}

TEST(GpuVmTranslation, FaultedLegacyAtomicModifyDoesNotInvokeMutation) {
  GpuMemory memory("memory");
  KfdProcess::PageTable page_table;
  util::DistributedSharedMutex page_table_mutex;
  constexpr uint32_t vmid = 7;
  constexpr uint64_t virtual_address = 0x4000;
  constexpr uint32_t initial_value = 0x11223344;
  HostPage backing;
  ASSERT_NE(backing.data(), nullptr);
  std::memcpy(backing.data(), &initial_value, sizeof(initial_value));
  page_table[virtual_address >> KfdProcess::kPageShift] = {backing.data(), Mtype::CC};
  RecordingFaultReporter reporter;
  GpuVm gpu_vm;
  LegacyGpuVmAdapter legacy_vm(gpu_vm, &memory);
  const AddressSpaceHandle handle =
      legacy_vm.register_address_space(vmid, {.page_table = &page_table,
                                              .page_table_mutex = &page_table_mutex,
                                              .page_table_generation = nullptr,
                                              .request_mutex = {},
                                              .client_pid = 0,
                                              .client_mem_fd = -1,
                                              .passthrough = false,
                                              .fault_reporter = &reporter});
  ASSERT_TRUE(handle);
  std::optional<GpuVmAccess> access = gpu_vm.snapshot(handle);
  ASSERT_TRUE(access);
  ASSERT_EQ(mprotect(backing.data(), KfdProcess::kPageSize, PROT_READ), 0);

  uint32_t mutation_calls = 0;
  EXPECT_EQ(access->atomic_modify(virtual_address, sizeof(uint32_t),
                                  [&](std::span<std::byte> bytes) {
                                    ++mutation_calls;
                                    std::ranges::fill(bytes, std::byte{0xa5});
                                  }),
            VmAccessOutcome::Faulted);
  EXPECT_EQ(mutation_calls, 0u);
  EXPECT_EQ(reporter.vmids, (std::vector<uint32_t>{vmid}));
  EXPECT_EQ(reporter.addresses, (std::vector<uint64_t>{virtual_address}));

  ASSERT_EQ(mprotect(backing.data(), KfdProcess::kPageSize, PROT_READ | PROT_WRITE), 0);
  uint32_t observed = 0;
  std::memcpy(&observed, backing.data(), sizeof(observed));
  EXPECT_EQ(observed, initial_value);
  EXPECT_TRUE(gpu_vm.unregister_address_space(handle));
}

TEST(GpuVmTranslation, StrictLegacyBackingReportsWrappingRange) {
  GpuMemory memory("memory");
  constexpr uint32_t vmid = 7;
  constexpr uint64_t address = UINT64_MAX - 1;
  RecordingFaultReporter reporter;
  GpuVm gpu_vm;
  LegacyGpuVmAdapter legacy_vm(gpu_vm, &memory);
  LegacyPageTable page_table;
  util::DistributedSharedMutex page_table_mutex;
  const AddressSpaceHandle handle =
      legacy_vm.register_address_space(vmid, {.page_table = &page_table,
                                              .page_table_mutex = &page_table_mutex,
                                              .page_table_generation = nullptr,
                                              .request_mutex = {},
                                              .client_pid = 0,
                                              .client_mem_fd = -1,
                                              .passthrough = false,
                                              .fault_reporter = &reporter});
  ASSERT_TRUE(handle);
  std::array<uint8_t, 4> value{};

  EXPECT_EQ(gpu_vm.read(handle, address, std::as_writable_bytes(std::span(value))),
            VmAccessOutcome::Malformed);
  EXPECT_EQ(gpu_vm.write(handle, address, std::as_bytes(std::span(value))),
            VmAccessOutcome::Malformed);
  EXPECT_EQ(reporter.vmids, (std::vector<uint32_t>{vmid, vmid}));
  EXPECT_EQ(reporter.addresses, (std::vector<uint64_t>{address, address}));
}

TEST(GpuVmTranslation, LegacyVmidZeroGetsGenerationSafePassthroughBinding) {
  GpuMemory memory("memory");
  GpuVm gpu_vm;
  LegacyGpuVmAdapter legacy_vm(gpu_vm, &memory);
  HostPage backing;
  ASSERT_NE(backing.data(), nullptr);
  backing.data()[0] = 0x5a;

  LegacyPageTable page_table;
  util::DistributedSharedMutex page_table_mutex;
  const AddressSpaceHandle handle =
      legacy_vm.register_address_space(0, {.page_table = &page_table,
                                           .page_table_mutex = &page_table_mutex,
                                           .page_table_generation = nullptr,
                                           .request_mutex = {},
                                           .client_pid = 0,
                                           .client_mem_fd = -1,
                                           .passthrough = true,
                                           .fault_reporter = nullptr});

  ASSERT_TRUE(handle);
  ASSERT_TRUE(gpu_vm.lookup(handle));
  EXPECT_EQ(gpu_vm.lookup(handle)->vmid, 0u);
  EXPECT_TRUE(gpu_vm.lookup(handle)->legacy_cache_compatible);
  const AddressSpaceHandle gart = gpu_vm.initialize_gart_address_space();
  ASSERT_TRUE(gart);
  EXPECT_NE(gart, handle);
  EXPECT_EQ(gpu_vm.find_vmid(0), handle);
  EXPECT_EQ(gpu_vm.gart_address_space(), gart);
  EXPECT_EQ(gpu_vm.active_address_spaces(), 2u);

  std::array<std::byte, 1> value{};
  const uint64_t address = reinterpret_cast<uint64_t>(backing.data());
  EXPECT_EQ(gpu_vm.read(handle, address, value), VmAccessOutcome::Complete);
  EXPECT_EQ(value[0], std::byte{0x5a});
  value[0] = std::byte{0xa5};
  EXPECT_EQ(gpu_vm.write(handle, address, value), VmAccessOutcome::Complete);
  EXPECT_EQ(backing.data()[0], 0xa5);

  EXPECT_TRUE(gpu_vm.unregister_address_space(handle));
  EXPECT_EQ(gpu_vm.gart_address_space(), gart);
  EXPECT_EQ(gpu_vm.active_address_spaces(), 1u);
}

TEST(GpuVmTranslation, LegacyHostRangeChecksSeparatelyGrantedPages) {
  constexpr uint32_t vmid = 7;
  constexpr uint64_t gpu_va = 0x40000;
  constexpr size_t page_size = KfdProcess::kPageSize;
  constexpr size_t page_count = 3;
  HostPage backing(page_count * page_size);
  ASSERT_NE(backing.data(), nullptr);
  // Model an ELF header in the last page with its kernel descriptor nearby.
  backing.data()[2 * page_size] = 0x7f;
  backing.data()[2 * page_size + 1] = 'E';
  backing.data()[2 * page_size + 2] = 'L';
  backing.data()[2 * page_size + 3] = 'F';

  LegacyPageTable page_table;
  for (size_t page = 0; page < page_count; ++page)
    page_table[gpu_va / page_size + page] = {backing.data() + page * page_size, Mtype::CC};
  util::DistributedSharedMutex page_table_mutex;
  auto request_mutex = std::make_shared<util::DistributedSharedMutex>();
  bool deny_middle_page = false;
  bool owned_middle_page = false;
  bool checked_request_lock = false;
  bool writer_entered_during_page_check = false;
  const auto page_granted = [&](uint64_t address, size_t size) {
    if (address < gpu_va || address >= gpu_va + page_count * page_size || size == 0)
      return false;
    const size_t offset = address - gpu_va;
    return size <= page_size - offset % page_size && !(deny_middle_page && offset / page_size == 1);
  };

  GpuMemory memory("memory");
  GpuVm gpu_vm;
  LegacyGpuVmAdapter adapter(gpu_vm, &memory);
  const AddressSpaceHandle handle = adapter.register_address_space(
      vmid,
      {.page_table = &page_table,
       .page_table_mutex = &page_table_mutex,
       .request_mutex = request_mutex,
       .access_admission =
           [&](uint64_t address, size_t size, VmAccessKind) {
             return VmAccessAdmission{.outcome = page_granted(address, size)
                                                     ? VmAccessOutcome::Complete
                                                     : VmAccessOutcome::Faulted};
           },
       .data_route =
           [&](uint64_t address, size_t size, VmAccessKind access) {
             if (!checked_request_lock && address == gpu_va && size == page_size &&
                 access == VmAccessKind::Write) {
               checked_request_lock = true;
               writer_entered_during_page_check = std::async(std::launch::async, [&] {
                                                    if (!request_mutex->try_lock())
                                                      return false;
                                                    request_mutex->unlock();
                                                    return true;
                                                  }).get();
             }
             const bool allowed = page_granted(address, size);
             if (allowed && owned_middle_page && access == VmAccessKind::Write &&
                 (address - gpu_va) / page_size == 1)
               return LegacyDataRoute{.route = VmDataRoute::Deferred,
                                      .first_extent_bytes = size,
                                      .candidate = LegacyDeferredCandidate::ControlledLocalWrite};
             return LegacyDataRoute{.route = allowed ? VmDataRoute::Local : VmDataRoute::Faulted,
                                    .first_extent_bytes = allowed ? size : 0};
           }});
  ASSERT_TRUE(handle);
  const auto access = gpu_vm.snapshot(handle);
  ASSERT_TRUE(access);

  EXPECT_EQ(access->host_range(gpu_va + page_size + 0x180),
            (std::pair<uint64_t, uint64_t>{reinterpret_cast<uint64_t>(backing.data()),
                                           page_count * page_size}));
  EXPECT_TRUE(checked_request_lock);
  EXPECT_FALSE(writer_entered_during_page_check);
  owned_middle_page = true;
  EXPECT_EQ(access->host_range(gpu_va + 2 * page_size + 0x180),
            (std::pair<uint64_t, uint64_t>{
                reinterpret_cast<uint64_t>(backing.data()) + 2 * page_size, page_size}));
  EXPECT_EQ(access->host_range(gpu_va + page_size + 0x180), (std::pair<uint64_t, uint64_t>{0, 0}));
  owned_middle_page = false;
  deny_middle_page = true;
  EXPECT_EQ(access->host_range(gpu_va + 2 * page_size + 0x180),
            (std::pair<uint64_t, uint64_t>{
                reinterpret_cast<uint64_t>(backing.data()) + 2 * page_size, page_size}));
  EXPECT_EQ(access->host_range(gpu_va + 0x180),
            (std::pair<uint64_t, uint64_t>{reinterpret_cast<uint64_t>(backing.data()), page_size}));
}

TEST(GpuVmTranslation, LegacyControlledLocalWriteUsesOwnedBoundedTransfer) {
  struct OwnedWrite {
    std::vector<std::byte> payload;
    std::shared_ptr<VmTransferEffectGate> gate;
    uint8_t *target = nullptr;
    std::size_t submissions = 0;
    bool complete = false;
  };
  class OwnedWriteTransfer final : public VmDeferredTransfer {
  public:
    explicit OwnedWriteTransfer(std::shared_ptr<OwnedWrite> write) : write_(std::move(write)) {}
    void submit() override { ++write_->submissions; }
    VmAccessOutcome poll(std::span<std::byte>) override {
      return write_->complete ? VmAccessOutcome::Complete : VmAccessOutcome::Unavailable;
    }
    VmAccessOutcome reconcile_revocation(std::span<std::byte>) override {
      return write_->complete ? VmAccessOutcome::Complete : VmAccessOutcome::Revoked;
    }
    void cancel() noexcept override {}

  private:
    std::shared_ptr<OwnedWrite> write_;
  };

  constexpr uint32_t vmid = 7;
  constexpr uint64_t address = 0x4ffc;
  HostPage first_page;
  HostPage second_page;
  ASSERT_NE(first_page.data(), nullptr);
  ASSERT_NE(second_page.data(), nullptr);
  LegacyPageTable page_table;
  page_table[4] = {first_page.data(), Mtype::CC};
  page_table[5] = {second_page.data(), Mtype::CC};
  util::DistributedSharedMutex page_table_mutex;
  std::vector<std::shared_ptr<OwnedWrite>> prepared;
  GpuMemory memory("memory");
  GpuVm gpu_vm;
  LegacyGpuVmAdapter adapter(gpu_vm, &memory);
  const AddressSpaceHandle handle = adapter.register_address_space(
      vmid, {.page_table = &page_table,
             .page_table_mutex = &page_table_mutex,
             .request_mutex = {},
             .data_route =
                 [](uint64_t request_address, std::size_t, VmAccessKind access) {
                   if (access != VmAccessKind::Write)
                     return LegacyDataRoute{};
                   if (request_address < 0x4000 || request_address >= address + 8)
                     return LegacyDataRoute{};
                   if (request_address < address)
                     return LegacyDataRoute{.first_extent_bytes = address - request_address};
                   return LegacyDataRoute{
                       .route = VmDataRoute::Deferred,
                       // The first chunk crosses a page; later chunks reach a short extent.
                       .first_extent_bytes = request_address < 0x5000 ? 8u : 2u,
                       .candidate = LegacyDeferredCandidate::ControlledLocalWrite};
                 },
             .prepare_deferred_transfer =
                 [&](LegacyDeferredCandidate candidate, uint64_t request_address, std::size_t size,
                     std::span<const std::byte> bytes, bool write,
                     const std::shared_ptr<VmTransferEffectGate> &gate,
                     const GpuVmAccess &executing_access,
                     std::shared_ptr<const LegacyAddressSpace> address_space)
                 -> std::unique_ptr<VmDeferredTransfer> {
               EXPECT_EQ(candidate, LegacyDeferredCandidate::ControlledLocalWrite);
               EXPECT_TRUE(write);
               EXPECT_EQ(bytes.size(), size);
               EXPECT_EQ(gate, executing_access.retain_effect_gate());
               EXPECT_NE(executing_access.cache_identity(), 0u);
               EXPECT_EQ(executing_access.info().vmid, vmid);
               EXPECT_TRUE(address_space->is_range_mapped(request_address, size, vmid));
               auto operation = std::make_shared<OwnedWrite>();
               operation->payload.assign(bytes.begin(), bytes.end());
               operation->gate = gate;
               operation->target = request_address < 0x5000
                                       ? first_page.data() + (request_address - 0x4000)
                                       : second_page.data() + (request_address - 0x5000);
               prepared.push_back(operation);
               return std::make_unique<OwnedWriteTransfer>(std::move(operation));
             }});
  ASSERT_TRUE(handle);
  const auto access = gpu_vm.snapshot(handle);
  ASSERT_TRUE(access);

  const std::array<std::byte, 8> expected = {std::byte{1}, std::byte{2}, std::byte{3},
                                             std::byte{4}, std::byte{5}, std::byte{6},
                                             std::byte{7}, std::byte{8}};
  auto source = expected;
  std::size_t progress = 0;
  EXPECT_EQ(access->classify_data_route(address, source.size(), VmAccessKind::Write),
            VmDataRoute::Deferred);
  EXPECT_EQ(access->admit_access(address, source.size(), VmAccessKind::Write).outcome,
            VmAccessOutcome::Faulted);
  std::array<std::byte, 4> read_back{};
  EXPECT_EQ(access->read(address, read_back), VmAccessOutcome::Complete);
  EXPECT_TRUE(std::ranges::all_of(read_back, [](std::byte value) { return value == std::byte{}; }));
  EXPECT_EQ(access->write(address, source, progress), VmAccessOutcome::Malformed);
  EXPECT_FALSE(access->try_write_contiguous(address, source));
  EXPECT_EQ(access->admit_access(address - 4, 6, VmAccessKind::Write).outcome,
            VmAccessOutcome::Faulted);
  EXPECT_EQ(access->admit_access(address + 8, 1, VmAccessKind::Write).outcome,
            VmAccessOutcome::Faulted);
  EXPECT_FALSE(access->try_write_contiguous(address - 4, std::span(source).first(6)));
  EXPECT_EQ(access->atomic_store(address, sizeof(uint32_t), 0xffffffff), VmAccessOutcome::Faulted);
  EXPECT_EQ(access->atomic_store(address - 4, sizeof(uint64_t), 0xffffffffffffffff),
            VmAccessOutcome::Faulted);
  EXPECT_EQ(access->resolve_host_pointer(address, 1), nullptr);
  EXPECT_EQ(access->resolve_host_pointer(address - 4, 6), nullptr);
  EXPECT_EQ(access->resolve_host_pointer(address - 4, 8), nullptr);
  EXPECT_EQ(access->host_range(address), (std::pair<uint64_t, uint64_t>{0, 0}));
  EXPECT_EQ(access->host_range(address - 4), (std::pair<uint64_t, uint64_t>{0, 0}));
  EXPECT_EQ(progress, 0u);
  EXPECT_TRUE(prepared.empty());
  EXPECT_EQ(first_page.data()[KfdProcess::kPageSize - 1], 0u);
  const std::array<std::byte, 4> local_bytes{std::byte{9}, std::byte{10}, std::byte{11},
                                             std::byte{12}};
  EXPECT_EQ(access->write(address - 8, local_bytes), VmAccessOutcome::Complete);
  for (std::size_t index = 0; index < local_bytes.size(); ++index)
    EXPECT_EQ(first_page.data()[KfdProcess::kPageSize - 12 + index],
              std::to_integer<uint8_t>(local_bytes[index]));

  auto finish = [](const std::shared_ptr<OwnedWrite> &operation) {
    ASSERT_EQ(operation->submissions, 1u);
    ASSERT_TRUE(operation->gate->commit([&] {
      std::memcpy(operation->target, operation->payload.data(), operation->payload.size());
      operation->complete = true;
    }));
  };
  VmTransferToken token;
  EXPECT_EQ(access->write(address, source, progress, token), VmAccessOutcome::Unavailable);
  ASSERT_EQ(prepared.size(), 1u);
  EXPECT_EQ(prepared[0]->payload.size(), 4u);
  EXPECT_EQ(prepared[0]->submissions, 1u);
  EXPECT_EQ(first_page.data()[KfdProcess::kPageSize - 4], 0u);
  source.fill(std::byte{0xff});
  finish(prepared[0]);
  EXPECT_EQ(access->write(address, source, progress, token), VmAccessOutcome::Unavailable);
  EXPECT_EQ(progress, 4u);

  source = expected;
  EXPECT_EQ(access->write(address, source, progress, token), VmAccessOutcome::Unavailable);
  ASSERT_EQ(prepared.size(), 2u);
  EXPECT_EQ(prepared[1]->payload.size(), 2u);
  EXPECT_EQ(prepared[1]->submissions, 1u);
  EXPECT_EQ(second_page.data()[0], 0u);
  source.fill(std::byte{0xff});
  finish(prepared[1]);
  EXPECT_EQ(access->write(address, source, progress, token), VmAccessOutcome::Unavailable);
  EXPECT_EQ(progress, 6u);

  source = expected;
  EXPECT_EQ(access->write(address, source, progress, token), VmAccessOutcome::Unavailable);
  ASSERT_EQ(prepared.size(), 3u);
  EXPECT_EQ(prepared[2]->payload.size(), 2u);
  EXPECT_EQ(prepared[2]->submissions, 1u);
  EXPECT_EQ(second_page.data()[2], 0u);
  source.fill(std::byte{0xff});
  finish(prepared[2]);
  EXPECT_EQ(access->write(address, source, progress, token), VmAccessOutcome::Complete);
  EXPECT_EQ(progress, expected.size());
  EXPECT_FALSE(token.pending());
  for (std::size_t index = 0; index < 4; ++index) {
    EXPECT_EQ(first_page.data()[KfdProcess::kPageSize - 4 + index],
              std::to_integer<uint8_t>(expected[index]));
    EXPECT_EQ(second_page.data()[index], std::to_integer<uint8_t>(expected[index + 4]));
  }
  EXPECT_TRUE(gpu_vm.unregister_address_space(handle));
}

TEST(GpuVmTranslationTest, DeferredTokenOwnsTwoStepReadAndWriteAcrossOwnerEvents) {
  ::rocjitsu::test::AsyncVmHarness sim;
  auto memory = std::make_shared<::rocjitsu::test::AsyncVmMemory>(*sim.owner, 0x200, 0x104, 0x108);
  GpuVm vm;
  const AddressSpaceHandle handle = vm.register_translated(7, memory, memory);
  const auto access = vm.snapshot(handle);
  ASSERT_TRUE(access);
  const std::array<std::byte, 8> initial = {std::byte{1}, std::byte{2}, std::byte{3}, std::byte{4},
                                            std::byte{5}, std::byte{6}, std::byte{7}, std::byte{8}};
  memory->store(0x100, initial);

  std::array<std::byte, 8> read_result{};
  std::size_t read_progress = 0;
  VmTransferToken read_token;
  EXPECT_EQ(access->read(0x100, read_result, read_progress, read_token),
            VmAccessOutcome::Unavailable);
  EXPECT_EQ(read_progress, 4u);
  EXPECT_TRUE(read_token.pending());
  EXPECT_EQ(memory->backing->posts, 1u);
  EXPECT_EQ(memory->backing->read_effects, 0u);
  EXPECT_EQ(read_result[4], std::byte{});
  EXPECT_EQ(access->read(0x100, read_result, read_progress, read_token),
            VmAccessOutcome::Unavailable);
  EXPECT_EQ(memory->backing->posts, 1u);
  (void)sim.engine.step();
  EXPECT_EQ(access->read(0x100, read_result, read_progress, read_token), VmAccessOutcome::Complete);
  EXPECT_EQ(read_progress, initial.size());
  EXPECT_EQ(read_result, initial);
  EXPECT_EQ(memory->backing->read_effects, 1u);
  EXPECT_FALSE(read_token.pending());

  std::array<std::byte, 8> write_source = {std::byte{9},  std::byte{10}, std::byte{11},
                                           std::byte{12}, std::byte{13}, std::byte{14},
                                           std::byte{15}, std::byte{16}};
  const auto original_write = write_source;
  std::size_t write_progress = 0;
  VmTransferToken write_token;
  EXPECT_EQ(access->write(0x100, write_source, write_progress, write_token),
            VmAccessOutcome::Unavailable);
  EXPECT_EQ(write_progress, 4u);
  EXPECT_EQ(memory->backing->posts, 2u);
  EXPECT_EQ(memory->backing->write_effects, 0u);
  write_source.fill(std::byte{0xff});
  (void)sim.engine.step();
  EXPECT_EQ(access->write(0x100, write_source, write_progress, write_token),
            VmAccessOutcome::Complete);
  EXPECT_EQ(write_progress, original_write.size());
  EXPECT_EQ(memory->load(0x100, original_write.size()),
            std::vector<std::byte>(original_write.begin(), original_write.end()));
  EXPECT_EQ(memory->backing->write_effects, 1u);
  EXPECT_EQ(memory->backing->posts, 2u);
}

TEST(GpuVmTranslationTest, TerminalDeferredFaultReportsItsRequestAddressOnce) {
  ::rocjitsu::test::AsyncVmHarness sim;
  // Preparation owns a request, but its owner discovers that the selected
  // backing extent cannot satisfy it after submission.
  auto memory = std::make_shared<::rocjitsu::test::AsyncVmMemory>(*sim.owner, 0x100, 0x100, 0x104);
  GpuVm vm;
  std::vector<std::pair<uint64_t, VmAccessKind>> faults;
  const auto handle =
      vm.register_address_space(7, memory, memory, [&](uint64_t address, VmAccessKind kind) {
        faults.emplace_back(address, kind);
      });
  const auto access = vm.snapshot(handle);
  ASSERT_TRUE(access);
  std::array<std::byte, 4> result{std::byte{0xa5}, std::byte{0xa5}, std::byte{0xa5},
                                  std::byte{0xa5}};
  std::size_t progress = 0;
  VmTransferToken token;
  EXPECT_EQ(access->read(0x100, result, progress, token), VmAccessOutcome::Unavailable);
  ASSERT_TRUE(token.pending());
  ASSERT_TRUE(sim.engine.step());
  EXPECT_EQ(access->read(0x100, result, progress, token), VmAccessOutcome::Faulted);
  EXPECT_FALSE(token.pending());
  EXPECT_EQ(progress, 0u);
  EXPECT_EQ(result[0], std::byte{0xa5});
  EXPECT_EQ(faults, (std::vector<std::pair<uint64_t, VmAccessKind>>{{0x100, VmAccessKind::Read}}));
}

TEST(GpuVmTranslationTest, CommittedDeferredChunkAdvancesOnceAcrossInvalidation) {
  ::rocjitsu::test::AsyncVmHarness sim;
  auto memory = std::make_shared<::rocjitsu::test::AsyncVmMemory>(*sim.owner, 0x200, 0x100, 0x104);
  GpuVm vm;
  const AddressSpaceHandle handle = vm.register_translated(7, memory, memory);
  const auto old_access = vm.snapshot(handle);
  ASSERT_TRUE(old_access);
  const std::array<std::byte, 8> initial = {std::byte{1}, std::byte{2}, std::byte{3}, std::byte{4},
                                            std::byte{5}, std::byte{6}, std::byte{7}, std::byte{8}};
  memory->store(0x100, initial);

  std::array<std::byte, 8> result{};
  std::size_t progress = 0;
  VmTransferToken token;
  EXPECT_EQ(old_access->classify_data_route(0x100, 4, VmAccessKind::Read), VmDataRoute::Deferred);
  EXPECT_EQ(old_access->classify_data_route(0x100, 4, VmAccessKind::Atomic), VmDataRoute::Faulted);
  EXPECT_EQ(old_access->classify_data_route(0x104, 4, VmAccessKind::Read), VmDataRoute::Local);
  EXPECT_EQ(old_access->read(0x100, result, progress, token), VmAccessOutcome::Unavailable);
  EXPECT_EQ(progress, 0u);
  ASSERT_TRUE(sim.engine.step());
  EXPECT_EQ(memory->backing->read_effects, 1u);
  ASSERT_TRUE(vm.invalidate(handle));

  EXPECT_EQ(old_access->read(0x100, result, progress, token), VmAccessOutcome::Revoked);
  EXPECT_EQ(progress, 4u);
  EXPECT_FALSE(token.pending());
  EXPECT_TRUE(std::equal(initial.begin(), initial.begin() + 4, result.begin()));
  const auto current_access = vm.snapshot(handle);
  ASSERT_TRUE(current_access);
  EXPECT_EQ(current_access->read(0x100, result, progress, token), VmAccessOutcome::Complete);
  EXPECT_EQ(progress, initial.size());
  EXPECT_EQ(result, initial);
  EXPECT_EQ(memory->backing->posts, 1u);
  EXPECT_EQ(memory->backing->read_effects, 1u);
}

TEST(GpuVmTranslationTest, DeferredAccessRejectsUntokenizedAndExecutePaths) {
  ::rocjitsu::test::AsyncVmHarness sim;
  auto memory = std::make_shared<::rocjitsu::test::AsyncVmMemory>(*sim.owner, 0x200, 0x100, 0x108);
  GpuVm vm;
  const auto access = vm.snapshot(vm.register_translated(7, memory, memory));
  ASSERT_TRUE(access);
  std::array<std::byte, 4> bytes{};
  std::size_t progress = 0;
  EXPECT_EQ(access->read(0x100, bytes, progress), VmAccessOutcome::Malformed);
  EXPECT_EQ(access->write(0x100, bytes, progress), VmAccessOutcome::Malformed);
  VmTransferToken token;
  EXPECT_EQ(access->read(0x100, bytes, progress, token, VmAccessKind::Execute),
            VmAccessOutcome::Malformed);
  EXPECT_EQ(access->read(0x100, bytes, VmAccessKind::Execute), VmAccessOutcome::Malformed);
  EXPECT_EQ(memory->backing->posts, 0u);
  EXPECT_EQ(memory->backing->read_effects, 0u);
  EXPECT_EQ(memory->backing->write_effects, 0u);
}

TEST(GpuVmTranslationTest, DeferredSubmitRunsAfterVmLeaseAndRevocationFencesLateEffect) {
  ::rocjitsu::test::AsyncVmHarness sim;
  auto memory = std::make_shared<::rocjitsu::test::AsyncVmMemory>(*sim.owner, 0x200, 0x100, 0x108);
  GpuVm vm;
  const AddressSpaceHandle handle = vm.register_translated(7, memory, memory);
  const auto access = vm.snapshot(handle);
  ASSERT_TRUE(access);
  std::promise<void> invalidated;
  std::future<void> completion = invalidated.get_future();
  std::thread revoker;
  bool lease_released_at_submit = false;
  memory->backing->submit_probe = [&] {
    EXPECT_EQ(memory->backing->write_effects, 0u);
    revoker = std::thread([&] {
      EXPECT_TRUE(vm.invalidate(handle));
      invalidated.set_value();
    });
    lease_released_at_submit =
        completion.wait_for(std::chrono::milliseconds(500)) == std::future_status::ready;
  };
  std::array<std::byte, 4> source{std::byte{1}, std::byte{2}, std::byte{3}, std::byte{4}};
  std::size_t progress = 0;
  VmTransferToken token;
  EXPECT_EQ(access->write(0x100, source, progress, token), VmAccessOutcome::Unavailable);
  if (revoker.joinable())
    revoker.join();
  EXPECT_TRUE(lease_released_at_submit);
  EXPECT_EQ(progress, 0u);
  (void)sim.engine.step();
  EXPECT_EQ(memory->backing->posts, 1u);
  EXPECT_EQ(memory->backing->write_effects, 0u);
  EXPECT_EQ(access->write(0x100, source, progress, token), VmAccessOutcome::Revoked);
  EXPECT_FALSE(token.pending());
  EXPECT_GE(memory->backing->cancellations, 1u);
}

TEST(GpuVmTranslationTest, EffectGateDrainsEnteredEffectBeforeRevocationReturns) {
  VmTransferEffectGate gate;
  std::promise<void> entered;
  std::promise<void> release;
  std::promise<void> revoker_started;
  std::promise<void> revoked;
  auto entered_future = entered.get_future();
  auto release_future = release.get_future();
  auto revoker_started_future = revoker_started.get_future();
  auto revoked_future = revoked.get_future();
  std::thread effect([&] {
    EXPECT_TRUE(gate.commit([&] {
      entered.set_value();
      release_future.wait();
    }));
  });
  entered_future.wait();
  std::thread revoker([&] {
    revoker_started.set_value();
    gate.revoke();
    revoked.set_value();
  });
  revoker_started_future.wait();
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(1);
  bool entry_closed = false;
  do {
    entry_closed = !gate.commit([] {});
    if (!entry_closed)
      std::this_thread::yield();
  } while (!entry_closed && std::chrono::steady_clock::now() < deadline);
  EXPECT_TRUE(entry_closed);
  EXPECT_EQ(revoked_future.wait_for(std::chrono::milliseconds(20)), std::future_status::timeout);
  release.set_value();
  effect.join();
  revoker.join();
  EXPECT_EQ(revoked_future.wait_for(std::chrono::milliseconds(0)), std::future_status::ready);
  EXPECT_FALSE(gate.commit([] {}));
}

TEST(GpuVmTranslationTest, RootReplacementFencesPostedWriteBeforeOwnerEvent) {
  ::rocjitsu::test::AsyncVmHarness sim;
  auto memory = std::make_shared<::rocjitsu::test::AsyncVmMemory>(*sim.owner, 0x200, 0x100, 0x104);
  GpuVm vm;
  const AddressSpaceHandle handle = vm.register_translated(7, memory, memory);
  const auto old_access = vm.snapshot(handle);
  ASSERT_TRUE(old_access);
  const std::array<std::byte, 4> source{std::byte{1}, std::byte{2}, std::byte{3}, std::byte{4}};
  std::size_t progress = 0;
  VmTransferToken token;
  EXPECT_EQ(old_access->write(0x100, source, progress, token), VmAccessOutcome::Unavailable);
  EXPECT_TRUE(vm.replace_translated(handle, memory, memory));
  (void)sim.engine.step();
  EXPECT_EQ(memory->backing->posts, 1u);
  EXPECT_EQ(memory->backing->write_effects, 0u);
  EXPECT_EQ(old_access->write(0x100, source, progress, token), VmAccessOutcome::Revoked);
  EXPECT_FALSE(token.pending());
}

TEST(GpuVmTranslationTest, MalformedResumeCancelsPostedWriteBeforeOwnerEffect) {
  ::rocjitsu::test::AsyncVmHarness sim;
  auto memory = std::make_shared<::rocjitsu::test::AsyncVmMemory>(*sim.owner, 0x200, 0x100, 0x104);
  GpuVm vm;
  const auto access = vm.snapshot(vm.register_translated(7, memory, memory));
  ASSERT_TRUE(access);
  const std::array<std::byte, 4> source{std::byte{1}, std::byte{2}, std::byte{3}, std::byte{4}};
  std::size_t progress = 0;
  VmTransferToken token;
  EXPECT_EQ(access->write(0x100, source, progress, token), VmAccessOutcome::Unavailable);
  EXPECT_TRUE(token.pending());
  EXPECT_EQ(memory->backing->posts, 1u);
  progress = source.size() + 1;
  EXPECT_EQ(access->write(0x100, source, progress, token), VmAccessOutcome::Malformed);
  EXPECT_FALSE(token.pending());
  (void)sim.engine.step();
  EXPECT_EQ(memory->backing->write_effects, 0u);
  EXPECT_GE(memory->backing->cancellations, 1u);
}

} // namespace
} // namespace rocjitsu::amdgpu

#undef RJ_GPU_VM_TRANSLATION_TEST_WITH_ASAN
