// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "rocjitsu/vm/amdgpu/hbm_controller.h"
#include "rocjitsu/vm/amdgpu/l1_scalar_cache.h"
#include "rocjitsu/vm/amdgpu/l1_vector_cache.h"
#include "rocjitsu/vm/amdgpu/l2_cache.h"
#include "rocjitsu/vm/amdgpu/memory_side_cache.h"
#include "simdojo/sim/exec_mode.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <barrier>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <linux/memfd.h>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <thread>
#include <unistd.h>
#include <vector>

namespace {

using rocjitsu::amdgpu::GpuMemory;
using rocjitsu::amdgpu::HbmController;
using rocjitsu::amdgpu::L1ScalarCache;
using rocjitsu::amdgpu::L1VectorCache;
using rocjitsu::amdgpu::L2Cache;
using rocjitsu::amdgpu::MemorySideCache;
using rocjitsu::amdgpu::Mtype;

void increment_u32(uint8_t *line, uint32_t offset) {
  uint32_t value = 0;
  std::memcpy(&value, line + offset, sizeof(value));
  ++value;
  std::memcpy(line + offset, &value, sizeof(value));
}

void report_benchmark(std::string_view name, uint64_t operations,
                      std::chrono::steady_clock::duration elapsed) {
  const double total_ns = std::chrono::duration<double, std::nano>(elapsed).count();
  std::cout << "ROCJITSU_BENCHMARK" << " name=" << name << " operations=" << operations
            << " total_ns=" << static_cast<uint64_t>(total_ns) << " ns_per_op=" << std::fixed
            << std::setprecision(3) << total_ns / static_cast<double>(operations) << '\n';
}

void run_cross_l2_atomic_benchmark(std::string_view name, bool same_address) {
  GpuMemory memory("memory");
  L2Cache l2a("l2a");
  L2Cache l2b("l2b");
  l2a.set_backing_memory(&memory);
  l2b.set_backing_memory(&memory);

  constexpr uint32_t kThreads = 8;
  constexpr uint32_t kIterations = 25'000;
  constexpr uint64_t kBase = 0x800000;
  constexpr uint64_t kOperations = static_cast<uint64_t>(kThreads) * kIterations;

  for (uint32_t tid = 0; tid < kThreads; ++tid)
    memory.write32(kBase + static_cast<uint64_t>(tid) * L2Cache::LINE_SIZE, 0);

  std::barrier ready(kThreads + 1);
  std::barrier start(kThreads + 1);
  std::barrier done(kThreads + 1);
  std::array<L2Cache *, 2> l2s = {&l2a, &l2b};
  std::vector<std::thread> workers;
  workers.reserve(kThreads);
  for (uint32_t tid = 0; tid < kThreads; ++tid) {
    workers.emplace_back([&, tid] {
      L2Cache *l2 = l2s[tid % l2s.size()];
      const uint64_t target =
          kBase + (same_address ? 0 : static_cast<uint64_t>(tid) * L2Cache::LINE_SIZE);
      ready.arrive_and_wait();
      start.arrive_and_wait();
      for (uint32_t iteration = 0; iteration < kIterations; ++iteration)
        l2->atomic_rmw(target, sizeof(uint32_t), increment_u32);
      done.arrive_and_wait();
    });
  }

  ready.arrive_and_wait();
  const auto begin = std::chrono::steady_clock::now();
  start.arrive_and_wait();
  done.arrive_and_wait();
  const auto end = std::chrono::steady_clock::now();
  for (auto &worker : workers)
    worker.join();

  if (same_address) {
    EXPECT_EQ(memory.read32(kBase), kOperations);
  } else {
    uint64_t observed_operations = 0;
    for (uint32_t tid = 0; tid < kThreads; ++tid) {
      const uint32_t observed =
          memory.read32(kBase + static_cast<uint64_t>(tid) * L2Cache::LINE_SIZE);
      EXPECT_EQ(observed, kIterations) << "thread=" << tid;
      observed_operations += observed;
    }
    EXPECT_EQ(observed_operations, kOperations);
  }
  report_benchmark(name, kOperations, end - begin);
}

void run_atomic_hierarchy_benchmark(std::string_view name, uint32_t hierarchy_count) {
  GpuMemory memory("memory");
  std::vector<std::unique_ptr<L2Cache>> l2s;
  std::vector<std::unique_ptr<L1ScalarCache>> scalar_l1s;
  std::vector<std::unique_ptr<L1VectorCache>> vector_l1s;
  for (uint32_t i = 0; i < hierarchy_count; ++i) {
    auto l2 = std::make_unique<L2Cache>("l2_" + std::to_string(i));
    l2->set_backing_memory(&memory);
    scalar_l1s.push_back(std::make_unique<L1ScalarCache>(l2.get()));
    vector_l1s.push_back(std::make_unique<L1VectorCache>(l2.get()));
    l2s.push_back(std::move(l2));
  }

  constexpr uint64_t kAddr = 0x880000;
  constexpr uint32_t kIterations = 100'000;
  memory.write32(kAddr, 0);
  const auto begin = std::chrono::steady_clock::now();
  for (uint32_t i = 0; i < kIterations; ++i)
    l2s.front()->atomic_rmw(kAddr, sizeof(uint32_t), increment_u32);
  const auto end = std::chrono::steady_clock::now();

  EXPECT_EQ(memory.read32(kAddr), kIterations);
  report_benchmark(name, kIterations, end - begin);
}

void run_scalar_l1_hit_benchmark() {
  GpuMemory memory("memory");
  L2Cache l2("l2");
  l2.set_backing_memory(&memory);
  L1ScalarCache l1(&l2);
  constexpr uint64_t kAddr = 0x890000;
  constexpr uint32_t kIterations = 1'000'000;
  memory.write32(kAddr, 17);
  uint32_t value = 0;
  l1.load(kAddr, 1, &value);

  const auto begin = std::chrono::steady_clock::now();
  for (uint32_t i = 0; i < kIterations; ++i)
    l1.load(kAddr, 1, &value);
  const auto end = std::chrono::steady_clock::now();

  EXPECT_EQ(value, 17u);
  report_benchmark("scalar_l1_hit", kIterations, end - begin);
}

void run_vector_l1_hit_benchmark() {
  GpuMemory memory("memory");
  L2Cache l2("l2");
  l2.set_backing_memory(&memory);
  L1VectorCache l1(&l2);
  constexpr uint64_t kAddr = 0x8A0000;
  constexpr uint32_t kIterations = 500'000;
  memory.write32(kAddr, 23);
  const uint64_t addrs[1] = {kAddr};
  std::array<uint8_t, sizeof(uint32_t)> value{};
  l1.load(addrs, 1, sizeof(uint32_t), 1, value.data(), Mtype::RW, false, false, 1);

  const auto begin = std::chrono::steady_clock::now();
  for (uint32_t i = 0; i < kIterations; ++i)
    l1.load(addrs, 1, sizeof(uint32_t), 1, value.data(), Mtype::RW, false, false, 1);
  const auto end = std::chrono::steady_clock::now();

  uint32_t observed = 0;
  std::memcpy(&observed, value.data(), sizeof(observed));
  EXPECT_EQ(observed, 23u);
  report_benchmark("vector_l1_hit", kIterations, end - begin);
}

void run_invalidate_range_benchmark(std::string_view name, uint32_t lines, uint32_t iterations) {
  GpuMemory memory("memory");
  L2Cache l2("l2");
  l2.set_backing_memory(&memory);

  constexpr uint64_t kBase = 0xa00000;
  std::array<uint8_t, L2Cache::LINE_SIZE> replacement{};
  std::array<uint8_t, L2Cache::LINE_SIZE> actual{};
  for (uint32_t line = 0; line < lines; ++line) {
    const uint64_t address = kBase + static_cast<uint64_t>(line) * L2Cache::LINE_SIZE;
    replacement.fill(static_cast<uint8_t>(line + 0x40));
    memory.write_block(address, std::span<const uint8_t>(replacement));
  }

  std::chrono::steady_clock::duration elapsed{};
  uint64_t refill_checksum = 0;
  uint64_t expected_refill_checksum = 0;
  for (uint32_t line = 0; line < lines; ++line)
    expected_refill_checksum += static_cast<uint8_t>(line + 0x40);
  expected_refill_checksum *= iterations;

  for (uint32_t iteration = 0; iteration < iterations; ++iteration) {
    // Refill outside the timed interval so every measured call invalidates
    // resident lines rather than repeatedly walking an empty cache.
    for (uint32_t line = 0; line < lines; ++line) {
      const uint64_t address = kBase + static_cast<uint64_t>(line) * L2Cache::LINE_SIZE;
      l2.read(address, actual.data(), actual.size());
      refill_checksum += actual.front();
    }

    const auto begin = std::chrono::steady_clock::now();
    l2.invalidate_range(kBase, static_cast<uint64_t>(lines) * L2Cache::LINE_SIZE, 0);
    elapsed += std::chrono::steady_clock::now() - begin;
  }

  uint64_t checksum = 0;
  uint64_t expected_checksum = 0;
  for (uint32_t line = 0; line < lines; ++line) {
    const uint64_t address = kBase + static_cast<uint64_t>(line) * L2Cache::LINE_SIZE;
    replacement.fill(static_cast<uint8_t>(line + 0x40));
    l2.read(address, actual.data(), actual.size());
    EXPECT_EQ(actual, replacement) << "line=" << line;
    checksum += actual.front();
    expected_checksum += replacement.front();
  }
  EXPECT_EQ(refill_checksum, expected_refill_checksum);
  EXPECT_EQ(checksum, expected_checksum);
  report_benchmark(name, iterations, elapsed);
}

class ScopedFd {
public:
  explicit ScopedFd(int fd) : fd_(fd) {}
  ScopedFd(const ScopedFd &) = delete;
  ScopedFd &operator=(const ScopedFd &) = delete;
  ~ScopedFd() {
    if (fd_ >= 0)
      close(fd_);
  }

  int get() const { return fd_; }

private:
  int fd_;
};

class ScopedMapping {
public:
  ScopedMapping(void *address, size_t size) : address_(address), size_(size) {}
  ScopedMapping(const ScopedMapping &) = delete;
  ScopedMapping &operator=(const ScopedMapping &) = delete;
  ~ScopedMapping() {
    if (address_ != MAP_FAILED)
      munmap(address_, size_);
  }

  uint8_t *data() const { return static_cast<uint8_t *>(address_); }

private:
  void *address_;
  size_t size_;
};

class FunctionalMemoryPort : public simdojo::Component {
public:
  FunctionalMemoryPort(std::string name, uint64_t base, size_t size)
      : simdojo::Component(std::move(name)), base_(base), bytes_(size) {
    port_ = add_port(std::make_unique<simdojo::Port>("memory", 0, this, simdojo::PortDirection::IN,
                                                     simdojo::PortProtocol::MEMORY));
    port_->set_handler([this](simdojo::Tick, simdojo::Message *message) {
      assert(message != nullptr);
      const auto &header = message->header();
      assert(header.addr >= base_);
      const uint64_t offset = header.addr - base_;
      assert(offset + header.size_bytes <= bytes_.size());
      auto *payload = reinterpret_cast<uint8_t *>(message->payload());
      assert(payload != nullptr);
      if (header.op == simdojo::MessageOp::READ) {
        std::memcpy(payload, bytes_.data() + offset, header.size_bytes);
      } else {
        assert(header.op == simdojo::MessageOp::WRITE);
        std::memcpy(bytes_.data() + offset, payload, header.size_bytes);
      }
    });
  }

  simdojo::Port *port() { return port_; }

  void write32(uint64_t addr, uint32_t value) {
    assert(addr >= base_ && addr - base_ + sizeof(value) <= bytes_.size());
    std::memcpy(bytes_.data() + (addr - base_), &value, sizeof(value));
  }

  uint32_t read32(uint64_t addr) const {
    assert(addr >= base_ && addr - base_ + sizeof(uint32_t) <= bytes_.size());
    uint32_t value = 0;
    std::memcpy(&value, bytes_.data() + (addr - base_), sizeof(value));
    return value;
  }

  uint8_t read8(uint64_t addr) const {
    assert(addr >= base_ && addr - base_ < bytes_.size());
    return bytes_[addr - base_];
  }

  void write8(uint64_t addr, uint8_t value) {
    assert(addr >= base_ && addr - base_ < bytes_.size());
    bytes_[addr - base_] = value;
  }

private:
  uint64_t base_;
  std::vector<uint8_t> bytes_;
  simdojo::Port *port_ = nullptr;
};

TEST(L2CacheThreadingTest, ConcurrentDifferentSetWritesArePreserved) {
  GpuMemory memory("memory");
  L2Cache l2("l2");
  l2.set_backing_memory(&memory);

  constexpr uint32_t kThreads = 8;
  constexpr uint32_t kLinesPerThread = 128;
  constexpr uint64_t kBase = 0x100000;

  std::barrier start(kThreads);
  std::vector<std::thread> workers;
  workers.reserve(kThreads);

  for (uint32_t tid = 0; tid < kThreads; ++tid) {
    workers.emplace_back([&, tid] {
      std::array<uint8_t, L2Cache::LINE_SIZE> line{};
      start.arrive_and_wait();
      for (uint32_t i = 0; i < kLinesPerThread; ++i) {
        const uint64_t addr =
            kBase + (static_cast<uint64_t>(i) * kThreads + tid) * L2Cache::LINE_SIZE;
        for (uint32_t b = 0; b < line.size(); ++b)
          line[b] = static_cast<uint8_t>((tid << 4) ^ i ^ b);
        l2.write(addr, line.data(), line.size());
      }
    });
  }

  for (auto &worker : workers)
    worker.join();

  std::array<uint8_t, L2Cache::LINE_SIZE> expected{};
  std::array<uint8_t, L2Cache::LINE_SIZE> actual{};
  for (uint32_t tid = 0; tid < kThreads; ++tid) {
    for (uint32_t i = 0; i < kLinesPerThread; ++i) {
      const uint64_t addr =
          kBase + (static_cast<uint64_t>(i) * kThreads + tid) * L2Cache::LINE_SIZE;
      for (uint32_t b = 0; b < expected.size(); ++b)
        expected[b] = static_cast<uint8_t>((tid << 4) ^ i ^ b);
      l2.read(addr, actual.data(), actual.size());
      EXPECT_EQ(actual, expected) << "addr=0x" << std::hex << addr;
    }
  }
}

TEST(L2CacheThreadingTest, ConcurrentSameSetAccessesPreserveLines) {
  GpuMemory memory("memory");
  L2Cache l2("l2");
  l2.set_backing_memory(&memory);

  constexpr uint32_t kThreads = 8;
  constexpr uint32_t kIterations = 256;
  constexpr uint64_t kBase = 0x180000;
  constexpr uint64_t kSetStride = static_cast<uint64_t>(L2Cache::NUM_SETS) * L2Cache::LINE_SIZE;

  std::barrier start(kThreads);
  std::atomic<uint64_t> mismatches{0};
  std::vector<std::thread> workers;
  workers.reserve(kThreads);
  for (uint32_t tid = 0; tid < kThreads; ++tid) {
    workers.emplace_back([&, tid] {
      const uint64_t addr = kBase + tid * kSetStride;
      std::array<uint8_t, L2Cache::LINE_SIZE> line{};
      std::array<uint8_t, L2Cache::LINE_SIZE> actual{};
      start.arrive_and_wait();
      for (uint32_t iteration = 0; iteration < kIterations; ++iteration) {
        line.fill(static_cast<uint8_t>((tid << 4) ^ iteration));
        l2.write(addr, line.data(), line.size());
        l2.read(addr, actual.data(), actual.size());
        if (actual != line)
          mismatches.fetch_add(1, std::memory_order_relaxed);
      }
    });
  }

  for (auto &worker : workers)
    worker.join();

  EXPECT_EQ(mismatches.load(std::memory_order_relaxed), 0u);
}

TEST(L2CacheThreadingTest, ConcurrentAtomicRmwSameLineIsSerialized) {
  GpuMemory memory("memory");
  L2Cache l2("l2");
  l2.set_backing_memory(&memory);

  constexpr uint32_t kThreads = 8;
  constexpr uint32_t kIterations = 1000;
  constexpr uint64_t kTarget = 0x200000;

  memory.write32(kTarget, 0);

  std::barrier start(kThreads);
  std::vector<std::thread> workers;
  workers.reserve(kThreads);

  for (uint32_t tid = 0; tid < kThreads; ++tid) {
    workers.emplace_back([&] {
      start.arrive_and_wait();
      for (uint32_t i = 0; i < kIterations; ++i) {
        l2.atomic_rmw(kTarget, sizeof(uint32_t), [](uint8_t *line, uint32_t offset) {
          uint32_t value = 0;
          std::memcpy(&value, line + offset, sizeof(value));
          ++value;
          std::memcpy(line + offset, &value, sizeof(value));
        });
      }
    });
  }

  for (auto &worker : workers)
    worker.join();

  EXPECT_EQ(memory.read32(kTarget), kThreads * kIterations);
}

TEST(L2CacheThreadingTest, CrossL2AtomicRmwSameAddressIsSerialized) {
  GpuMemory memory("memory");
  L2Cache l2a("l2a");
  L2Cache l2b("l2b");
  l2a.set_backing_memory(&memory);
  l2b.set_backing_memory(&memory);

  constexpr uint32_t kThreads = 8;
  constexpr uint32_t kIterations = 1000;
  constexpr uint64_t kTarget = 0x280000;

  memory.write32(kTarget, 0);

  std::array<L2Cache *, 2> l2s = {&l2a, &l2b};
  std::barrier start(kThreads);
  std::vector<std::thread> workers;
  workers.reserve(kThreads);

  for (uint32_t tid = 0; tid < kThreads; ++tid) {
    workers.emplace_back([&, tid] {
      auto *l2 = l2s[tid % l2s.size()];
      start.arrive_and_wait();
      for (uint32_t i = 0; i < kIterations; ++i) {
        l2->atomic_rmw(kTarget, sizeof(uint32_t), [](uint8_t *line, uint32_t offset) {
          uint32_t value = 0;
          std::memcpy(&value, line + offset, sizeof(value));
          ++value;
          std::memcpy(line + offset, &value, sizeof(value));
        });
      }
    });
  }

  for (auto &worker : workers)
    worker.join();

  EXPECT_EQ(memory.read32(kTarget), kThreads * kIterations);
}

TEST(L2CacheThreadingTest, CrossL2AtomicRmwAliasedVasIsSerialized) {
  GpuMemory memory("memory");
  L2Cache l2a("l2a");
  L2Cache l2b("l2b");
  l2a.set_backing_memory(&memory);
  l2b.set_backing_memory(&memory);

  constexpr uint32_t kVmidA = 7;
  constexpr uint32_t kVmidB = 8;
  constexpr uint64_t kVaA = 0x100000;
  constexpr uint64_t kVaB = 0x201000;
  constexpr uint64_t kOffset = 64;
  constexpr uint32_t kThreads = 8;
  constexpr uint32_t kIterations = 1000;

  rocjitsu::KfdProcess process_a(kVmidA);
  rocjitsu::KfdProcess process_b(kVmidB);
  std::array<uint8_t, GpuMemory::PAGE_SIZE> backing{};
  process_a.map_pages(kVaA, backing.data(), backing.size());
  process_b.map_pages(kVaB, backing.data(), backing.size());
  memory.register_process(kVmidA, &process_a.page_table_, &process_a.page_table_mutex_,
                          process_a.page_table_generation());
  memory.register_process(kVmidB, &process_b.page_table_, &process_b.page_table_mutex_,
                          process_b.page_table_generation());

  std::barrier start(kThreads);
  std::vector<std::thread> workers;
  workers.reserve(kThreads);
  for (uint32_t tid = 0; tid < kThreads; ++tid) {
    workers.emplace_back([&, tid] {
      L2Cache &l2 = (tid & 1) ? l2b : l2a;
      const uint64_t addr = ((tid & 1) ? kVaB : kVaA) + kOffset;
      const uint32_t vmid = (tid & 1) ? kVmidB : kVmidA;
      start.arrive_and_wait();
      for (uint32_t i = 0; i < kIterations; ++i) {
        l2.atomic_rmw(
            addr, sizeof(uint32_t),
            [](uint8_t *line, uint32_t offset) {
              uint32_t value = 0;
              std::memcpy(&value, line + offset, sizeof(value));
              ++value;
              std::memcpy(line + offset, &value, sizeof(value));
            },
            vmid);
      }
    });
  }

  for (auto &worker : workers)
    worker.join();

  uint32_t actual = 0;
  std::memcpy(&actual, backing.data() + kOffset, sizeof(actual));
  EXPECT_EQ(actual, kThreads * kIterations);
}

TEST(L2CacheThreadingTest, CrossL2AtomicRmwDistinctSharedMappingsIncludesDirtyWriteback) {
  constexpr size_t kMappingSize = GpuMemory::PAGE_SIZE;
  const int raw_fd = static_cast<int>(syscall(SYS_memfd_create, "l2_atomic_alias", MFD_CLOEXEC));
  ASSERT_GE(raw_fd, 0);
  ScopedFd fd(raw_fd);
  ASSERT_EQ(ftruncate(fd.get(), static_cast<off_t>(kMappingSize)), 0);

  void *raw_mapping_a =
      mmap(nullptr, kMappingSize, PROT_READ | PROT_WRITE, MAP_SHARED, fd.get(), 0);
  ASSERT_NE(raw_mapping_a, MAP_FAILED);
  ScopedMapping mapping_a(raw_mapping_a, kMappingSize);
  void *raw_mapping_b =
      mmap(nullptr, kMappingSize, PROT_READ | PROT_WRITE, MAP_SHARED, fd.get(), 0);
  ASSERT_NE(raw_mapping_b, MAP_FAILED);
  ScopedMapping mapping_b(raw_mapping_b, kMappingSize);
  ASSERT_NE(mapping_a.data(), mapping_b.data());

  constexpr uint32_t kVmidA = 17;
  constexpr uint32_t kVmidB = 18;
  constexpr uint64_t kVaA = 0x310000;
  constexpr uint64_t kVaB = 0x420000;
  constexpr uint32_t kOffset = 64;
  rocjitsu::KfdProcess process_a(kVmidA);
  rocjitsu::KfdProcess process_b(kVmidB);
  process_a.map_pages(kVaA, mapping_a.data(), kMappingSize, Mtype::CC);
  process_b.map_pages(kVaB, mapping_b.data(), kMappingSize, Mtype::CC);

  GpuMemory memory("memory");
  memory.register_process(kVmidA, &process_a.page_table_, &process_a.page_table_mutex_,
                          process_a.page_table_generation());
  memory.register_process(kVmidB, &process_b.page_table_, &process_b.page_table_mutex_,
                          process_b.page_table_generation());
  L2Cache l2a("l2a");
  L2Cache l2b("l2b");
  l2a.set_backing_memory(&memory);
  l2b.set_backing_memory(&memory);

  std::array<uint8_t, L2Cache::LINE_SIZE> dirty_line{};
  constexpr uint32_t kDirtyValue = 40;
  std::memcpy(dirty_line.data() + kOffset, &kDirtyValue, sizeof(kDirtyValue));
  l2a.writeback_line(kVaA, dirty_line.data(), Mtype::RW, kVmidA);

  auto increment = [](uint8_t *line, uint32_t offset) {
    uint32_t value = 0;
    std::memcpy(&value, line + offset, sizeof(value));
    ++value;
    std::memcpy(line + offset, &value, sizeof(value));
  };

  // Both virtual addresses resolve to the same MAP_SHARED bytes. The first
  // atomic boundary to enter must publish l2a's dirty line before either RMW,
  // and both RMWs must remain serialized across the distinct L2 objects.
  std::barrier start(3);
  std::thread first_atomic([&] {
    start.arrive_and_wait();
    l2a.atomic_rmw(kVaA + kOffset, sizeof(uint32_t), increment, kVmidA);
  });
  std::thread second_atomic([&] {
    start.arrive_and_wait();
    l2b.atomic_rmw(kVaB + kOffset, sizeof(uint32_t), increment, kVmidB);
  });
  start.arrive_and_wait();

  first_atomic.join();
  second_atomic.join();

  uint32_t actual = 0;
  std::memcpy(&actual, mapping_a.data() + kOffset, sizeof(actual));
  EXPECT_EQ(actual, 42u);
}

TEST(DeviceCacheCoherenceTest, ScalarWriteThroughSurvivesAliasedRemoteAtomic) {
  constexpr size_t kMappingSize = GpuMemory::PAGE_SIZE;
  const int raw_fd =
      static_cast<int>(syscall(SYS_memfd_create, "scalar_atomic_alias", MFD_CLOEXEC));
  ASSERT_GE(raw_fd, 0);
  ScopedFd fd(raw_fd);
  ASSERT_EQ(ftruncate(fd.get(), static_cast<off_t>(kMappingSize)), 0);

  void *raw_scalar_mapping =
      mmap(nullptr, kMappingSize, PROT_READ | PROT_WRITE, MAP_SHARED, fd.get(), 0);
  ASSERT_NE(raw_scalar_mapping, MAP_FAILED);
  ScopedMapping scalar_mapping(raw_scalar_mapping, kMappingSize);
  void *raw_atomic_mapping =
      mmap(nullptr, kMappingSize, PROT_READ | PROT_WRITE, MAP_SHARED, fd.get(), 0);
  ASSERT_NE(raw_atomic_mapping, MAP_FAILED);
  ScopedMapping atomic_mapping(raw_atomic_mapping, kMappingSize);
  ASSERT_NE(scalar_mapping.data(), atomic_mapping.data());

  constexpr uint32_t kScalarVmid = 21;
  constexpr uint32_t kAtomicVmid = 22;
  constexpr uint64_t kScalarVa = 0x510000;
  constexpr uint64_t kAtomicVa = 0x620000;
  constexpr uint64_t kLineOffset = L2Cache::LINE_SIZE;
  constexpr uint64_t kAtomicAddr = kAtomicVa + kLineOffset;
  constexpr uint64_t kScalarAddr = kScalarVa + kLineOffset + sizeof(uint32_t);
  constexpr uint32_t kScalarValue = 0xA5A5A5A5;

  rocjitsu::KfdProcess scalar_process(kScalarVmid);
  rocjitsu::KfdProcess atomic_process(kAtomicVmid);
  scalar_process.map_pages(kScalarVa, scalar_mapping.data(), kMappingSize, Mtype::RW);
  atomic_process.map_pages(kAtomicVa, atomic_mapping.data(), kMappingSize, Mtype::RW);

  GpuMemory memory("memory");
  memory.register_process(kScalarVmid, &scalar_process.page_table_,
                          &scalar_process.page_table_mutex_,
                          scalar_process.page_table_generation());
  memory.register_process(kAtomicVmid, &atomic_process.page_table_,
                          &atomic_process.page_table_mutex_,
                          atomic_process.page_table_generation());
  L2Cache scalar_l2("scalar_l2");
  L2Cache atomic_l2("atomic_l2");
  rocjitsu::amdgpu::L1ScalarCache scalar_l1(&scalar_l2);
  scalar_l2.set_backing_memory(&memory);
  atomic_l2.set_backing_memory(&memory);

  scalar_l1.store(kScalarAddr, /*num_dwords=*/1, &kScalarValue, kScalarVmid);
  atomic_l2.atomic_rmw(kAtomicAddr, sizeof(uint32_t), increment_u32, kAtomicVmid);
  scalar_l1.writeback_all(kScalarVmid);

  uint32_t atomic_value = 0;
  uint32_t scalar_value = 0;
  std::memcpy(&atomic_value, scalar_mapping.data() + kLineOffset, sizeof(atomic_value));
  std::memcpy(&scalar_value, scalar_mapping.data() + kLineOffset + sizeof(uint32_t),
              sizeof(scalar_value));
  EXPECT_EQ(atomic_value, 1u);
  EXPECT_EQ(scalar_value, kScalarValue);
}

TEST(DeviceCacheCoherenceTest, DisjointDirtyL2AliasesSurviveRemoteAtomic) {
  constexpr size_t kMappingSize = GpuMemory::PAGE_SIZE;
  const int raw_fd =
      static_cast<int>(syscall(SYS_memfd_create, "l2_dirty_atomic_alias", MFD_CLOEXEC));
  ASSERT_GE(raw_fd, 0);
  ScopedFd fd(raw_fd);
  ASSERT_EQ(ftruncate(fd.get(), static_cast<off_t>(kMappingSize)), 0);

  void *raw_mapping_a =
      mmap(nullptr, kMappingSize, PROT_READ | PROT_WRITE, MAP_SHARED, fd.get(), 0);
  ASSERT_NE(raw_mapping_a, MAP_FAILED);
  ScopedMapping mapping_a(raw_mapping_a, kMappingSize);
  void *raw_mapping_b =
      mmap(nullptr, kMappingSize, PROT_READ | PROT_WRITE, MAP_SHARED, fd.get(), 0);
  ASSERT_NE(raw_mapping_b, MAP_FAILED);
  ScopedMapping mapping_b(raw_mapping_b, kMappingSize);
  void *raw_mapping_atomic =
      mmap(nullptr, kMappingSize, PROT_READ | PROT_WRITE, MAP_SHARED, fd.get(), 0);
  ASSERT_NE(raw_mapping_atomic, MAP_FAILED);
  ScopedMapping mapping_atomic(raw_mapping_atomic, kMappingSize);
  ASSERT_NE(mapping_a.data(), mapping_b.data());
  ASSERT_NE(mapping_a.data(), mapping_atomic.data());
  ASSERT_NE(mapping_b.data(), mapping_atomic.data());

  constexpr uint32_t kVmidA = 31;
  constexpr uint32_t kVmidB = 32;
  constexpr uint32_t kAtomicVmid = 33;
  constexpr uint64_t kVaA = 0x710000;
  constexpr uint64_t kVaB = 0x820000;
  constexpr uint64_t kAtomicVa = 0x930000;
  constexpr uint64_t kLineOffset = L2Cache::LINE_SIZE;
  constexpr uint32_t kValueA = 0x11112222;
  constexpr uint32_t kValueB = 0x33334444;
  constexpr uint32_t kOffsetA = sizeof(uint32_t);
  constexpr uint32_t kOffsetB = 2 * sizeof(uint32_t);

  rocjitsu::KfdProcess process_a(kVmidA);
  rocjitsu::KfdProcess process_b(kVmidB);
  rocjitsu::KfdProcess atomic_process(kAtomicVmid);
  process_a.map_pages(kVaA, mapping_a.data(), kMappingSize, Mtype::RW);
  process_b.map_pages(kVaB, mapping_b.data(), kMappingSize, Mtype::RW);
  atomic_process.map_pages(kAtomicVa, mapping_atomic.data(), kMappingSize, Mtype::RW);

  GpuMemory memory("memory");
  memory.register_process(kVmidA, &process_a.page_table_, &process_a.page_table_mutex_,
                          process_a.page_table_generation());
  memory.register_process(kVmidB, &process_b.page_table_, &process_b.page_table_mutex_,
                          process_b.page_table_generation());
  memory.register_process(kAtomicVmid, &atomic_process.page_table_,
                          &atomic_process.page_table_mutex_,
                          atomic_process.page_table_generation());
  L2Cache l2a("l2a");
  L2Cache l2b("l2b");
  L2Cache atomic_l2("atomic_l2");
  l2a.set_backing_memory(&memory);
  l2b.set_backing_memory(&memory);
  atomic_l2.set_backing_memory(&memory);

  std::array<uint8_t, L2Cache::LINE_SIZE> line_a{};
  std::array<uint8_t, L2Cache::LINE_SIZE> line_b{};
  std::memcpy(line_a.data() + kOffsetA, &kValueA, sizeof(kValueA));
  std::memcpy(line_b.data() + kOffsetB, &kValueB, sizeof(kValueB));
  l2a.writeback_line(kVaA + kLineOffset, line_a.data(), kOffsetA, sizeof(kValueA), Mtype::RW,
                     kVmidA);
  l2b.writeback_line(kVaB + kLineOffset, line_b.data(), kOffsetB, sizeof(kValueB), Mtype::RW,
                     kVmidB);

  atomic_l2.atomic_rmw(kAtomicVa + kLineOffset, sizeof(uint32_t), increment_u32, kAtomicVmid);

  uint32_t atomic_value = 0;
  uint32_t value_a = 0;
  uint32_t value_b = 0;
  std::memcpy(&atomic_value, mapping_a.data() + kLineOffset, sizeof(atomic_value));
  std::memcpy(&value_a, mapping_a.data() + kLineOffset + kOffsetA, sizeof(value_a));
  std::memcpy(&value_b, mapping_a.data() + kLineOffset + kOffsetB, sizeof(value_b));
  EXPECT_EQ(atomic_value, 1u);
  EXPECT_EQ(value_a, kValueA);
  EXPECT_EQ(value_b, kValueB);
}

TEST(DeviceCacheCoherenceTest, ConcurrentScalarLoadsTrackRepeatedAtomicEpochs) {
  GpuMemory memory("memory");
  L2Cache scalar_l2("scalar_l2");
  L2Cache atomic_l2("atomic_l2");
  rocjitsu::amdgpu::L1ScalarCache scalar_l1(&scalar_l2);
  scalar_l2.set_backing_memory(&memory);
  atomic_l2.set_backing_memory(&memory);

  constexpr uint64_t kAddr = 0xA600;
  constexpr uint32_t kAtomicIterations = 256;
  constexpr uint32_t kLoadIterations = 2048;
  memory.write32(kAddr, 0);
  uint32_t initially_cached = 1;
  scalar_l1.load(kAddr, /*num_dwords=*/1, &initially_cached);
  ASSERT_EQ(initially_cached, 0u);

  std::mutex start_mutex;
  std::condition_variable start_cv;
  bool atomic_ready = false;
  bool scalar_ready = false;
  std::atomic<bool> synchronization_failed{false};
  std::atomic<bool> nonmonotonic_load{false};

  auto wait_for_peer = [&](bool &self_ready, const bool &peer_ready) {
    std::unique_lock lock(start_mutex);
    self_ready = true;
    start_cv.notify_all();
    if (!start_cv.wait_for(lock, std::chrono::seconds(1), [&] { return peer_ready; }))
      synchronization_failed.store(true, std::memory_order_relaxed);
  };

  std::thread atomic([&] {
    wait_for_peer(atomic_ready, scalar_ready);
    for (uint32_t iteration = 0; iteration < kAtomicIterations; ++iteration) {
      atomic_l2.atomic_rmw(kAddr, sizeof(uint32_t), increment_u32);
      std::this_thread::yield();
    }
  });
  std::thread scalar([&] {
    wait_for_peer(scalar_ready, atomic_ready);
    uint32_t previous = 0;
    for (uint32_t iteration = 0; iteration < kLoadIterations; ++iteration) {
      uint32_t observed = 0;
      scalar_l1.load(kAddr, /*num_dwords=*/1, &observed);
      if (observed < previous || observed > kAtomicIterations)
        nonmonotonic_load.store(true, std::memory_order_relaxed);
      previous = observed;
      std::this_thread::yield();
    }
  });
  atomic.join();
  scalar.join();

  EXPECT_FALSE(synchronization_failed.load(std::memory_order_relaxed));
  EXPECT_FALSE(nonmonotonic_load.load(std::memory_order_relaxed));
  uint32_t final_value = 0;
  scalar_l1.load(kAddr, /*num_dwords=*/1, &final_value);
  EXPECT_EQ(final_value, kAtomicIterations);
}

TEST(L2CacheTest, FunctionalLinkedPortAtomicRmwUpdatesBacking) {
  constexpr uint64_t kBase = 0xB00000;
  constexpr uint64_t kAddr = kBase + 20;
  L2Cache l2("l2");
  FunctionalMemoryPort backing("backing", kBase, 2 * L2Cache::LINE_SIZE);
  simdojo::Link link(/*id=*/0, l2.req_port(), backing.port(), /*latency=*/0);
  link.set_exec_mode(simdojo::ExecMode::FUNCTIONAL);
  l2.req_port()->set_link(&link);
  backing.port()->set_link(&link);

  backing.write8(kAddr - 1, 0xA5);
  backing.write32(kAddr, 41);
  backing.write8(kAddr + sizeof(uint32_t), 0x5A);
  uint32_t callback_old_value = 0;
  l2.atomic_rmw(kAddr, sizeof(uint32_t), [&](uint8_t *target, uint32_t offset) {
    EXPECT_EQ(offset, 0u);
    std::memcpy(&callback_old_value, target, sizeof(callback_old_value));
    const uint32_t replacement = callback_old_value + 1;
    std::memcpy(target, &replacement, sizeof(replacement));
  });

  EXPECT_EQ(callback_old_value, 41u);
  EXPECT_EQ(backing.read32(kAddr), 42u);
  EXPECT_EQ(backing.read8(kAddr - 1), 0xA5);
  EXPECT_EQ(backing.read8(kAddr + sizeof(uint32_t)), 0x5A);
}

TEST(L2CacheTest, LinkedPortAtomicRmwRefreshesStaleMemorySideAlias) {
  constexpr uint32_t kVmidA = 41;
  constexpr uint32_t kVmidB = 42;
  constexpr uint64_t kVaA = 0xC10000;
  constexpr uint64_t kVaB = 0xD20000;
  constexpr uint32_t kOffset = 64;
  constexpr uint32_t kPublishedValue = 40;

  rocjitsu::KfdProcess process_a(kVmidA);
  rocjitsu::KfdProcess process_b(kVmidB);
  std::array<uint8_t, GpuMemory::PAGE_SIZE> backing{};
  process_a.map_pages(kVaA, backing.data(), backing.size(), Mtype::RW);
  process_b.map_pages(kVaB, backing.data(), backing.size(), Mtype::RW);

  GpuMemory memory("memory");
  memory.register_process(kVmidA, &process_a.page_table_, &process_a.page_table_mutex_,
                          process_a.page_table_generation());
  memory.register_process(kVmidB, &process_b.page_table_, &process_b.page_table_mutex_,
                          process_b.page_table_generation());

  HbmController hbm("hbm", &memory);
  MemorySideCache msc("msc");
  L2Cache l2a("l2a");
  L2Cache l2b("l2b");

  simdojo::Port *l2a_msc_port = msc.create_cpl_port("l2a");
  simdojo::Port *l2b_msc_port = msc.create_cpl_port("l2b");
  simdojo::Link msc_hbm_link(/*id=*/0, msc.req_port(), hbm.cpl_port(), /*latency=*/0);
  simdojo::Link l2a_msc_link(/*id=*/1, l2a.req_port(), l2a_msc_port, /*latency=*/0);
  simdojo::Link l2b_msc_link(/*id=*/2, l2b.req_port(), l2b_msc_port, /*latency=*/0);
  for (simdojo::Link *link : {&msc_hbm_link, &l2a_msc_link, &l2b_msc_link})
    link->set_exec_mode(simdojo::ExecMode::FUNCTIONAL);
  msc.req_port()->set_link(&msc_hbm_link);
  hbm.cpl_port()->set_link(&msc_hbm_link);
  l2a.req_port()->set_link(&l2a_msc_link);
  l2a_msc_port->set_link(&l2a_msc_link);
  l2b.req_port()->set_link(&l2b_msc_link);
  l2b_msc_port->set_link(&l2b_msc_link);

  uint32_t stale_value = 1;
  l2b.read(kVaB + kOffset, reinterpret_cast<uint8_t *>(&stale_value), sizeof(stale_value),
           Mtype::RW, kVmidB);
  ASSERT_EQ(stale_value, 0u);

  std::array<uint8_t, L2Cache::LINE_SIZE> dirty_line{};
  std::memcpy(dirty_line.data() + kOffset, &kPublishedValue, sizeof(kPublishedValue));
  l2a.writeback_line(kVaA, dirty_line.data(), kOffset, sizeof(kPublishedValue), Mtype::RW, kVmidA);

  uint32_t callback_old_value = 0;
  l2b.atomic_rmw(
      kVaB + kOffset, sizeof(uint32_t),
      [&](uint8_t *target, uint32_t offset) {
        std::memcpy(&callback_old_value, target + offset, sizeof(callback_old_value));
        const uint32_t replacement = callback_old_value + 1;
        std::memcpy(target + offset, &replacement, sizeof(replacement));
      },
      kVmidB);

  EXPECT_EQ(callback_old_value, kPublishedValue);
  uint32_t host_value = 0;
  std::memcpy(&host_value, backing.data() + kOffset, sizeof(host_value));
  EXPECT_EQ(host_value, kPublishedValue + 1);

  uint32_t alias_a_value = 0;
  uint32_t alias_b_value = 0;
  l2a.read(kVaA + kOffset, reinterpret_cast<uint8_t *>(&alias_a_value), sizeof(alias_a_value),
           Mtype::RW, kVmidA);
  l2b.read(kVaB + kOffset, reinterpret_cast<uint8_t *>(&alias_b_value), sizeof(alias_b_value),
           Mtype::RW, kVmidB);
  EXPECT_EQ(alias_a_value, kPublishedValue + 1);
  EXPECT_EQ(alias_b_value, kPublishedValue + 1);
}

TEST(L2CacheTest, AliasedVasRequireCoherenceBoundary) {
  GpuMemory memory("memory");
  L2Cache l2("l2");
  l2.set_backing_memory(&memory);

  constexpr uint32_t kVmidA = 7;
  constexpr uint32_t kVmidB = 8;
  constexpr uint64_t kVaA = 0x100000;
  constexpr uint64_t kVaB = 0x201000;

  rocjitsu::KfdProcess process_a(kVmidA);
  rocjitsu::KfdProcess process_b(kVmidB);
  std::array<uint8_t, GpuMemory::PAGE_SIZE> backing{};
  process_a.map_pages(kVaA, backing.data(), backing.size());
  process_b.map_pages(kVaB, backing.data(), backing.size());
  memory.register_process(kVmidA, &process_a.page_table_, &process_a.page_table_mutex_,
                          process_a.page_table_generation());
  memory.register_process(kVmidB, &process_b.page_table_, &process_b.page_table_mutex_,
                          process_b.page_table_generation());

  std::array<uint8_t, L2Cache::LINE_SIZE> initial{};
  std::array<uint8_t, L2Cache::LINE_SIZE> replacement{};
  std::array<uint8_t, L2Cache::LINE_SIZE> dirty{};
  std::array<uint8_t, L2Cache::LINE_SIZE> actual{};
  initial.fill(0x11);
  replacement.fill(0x22);
  dirty.fill(0x33);

  memory.write_block(kVaA, std::span<const uint8_t>(initial), kVmidA);
  l2.read(kVaA, actual.data(), actual.size(), Mtype::RW, kVmidA);
  ASSERT_EQ(actual, initial);
  l2.read(kVaB, actual.data(), actual.size(), Mtype::RW, kVmidB);
  ASSERT_EQ(actual, initial);

  l2.write(kVaB, replacement.data(), replacement.size(), Mtype::RW, kVmidB);
  l2.read(kVaA, actual.data(), actual.size(), Mtype::RW, kVmidA);
  EXPECT_EQ(actual, initial);
  l2.read(kVaA, actual.data(), actual.size(), Mtype::CC, kVmidA);
  EXPECT_EQ(actual, replacement);

  l2.writeback_line(kVaA, dirty.data(), Mtype::RW, kVmidA);
  memory.read_block(kVaB, std::span<uint8_t>(actual), kVmidB);
  EXPECT_EQ(actual, replacement);
  l2.flush_line(kVaA, kVmidA);
  memory.read_block(kVaB, std::span<uint8_t>(actual), kVmidB);
  EXPECT_EQ(actual, dirty);

  l2.read(kVaB, actual.data(), actual.size(), Mtype::RW, kVmidB);
  EXPECT_EQ(actual, replacement);
  l2.read(kVaB, actual.data(), actual.size(), Mtype::CC, kVmidB);
  EXPECT_EQ(actual, dirty);
}

TEST(L2CacheThreadingTest, ConcurrentFlushAllPreservesDirtyWritebacks) {
  GpuMemory memory("memory");
  L2Cache l2("l2");
  l2.set_backing_memory(&memory);

  constexpr uint32_t kWriterThreads = 4;
  constexpr uint32_t kLinesPerThread = 8;
  constexpr uint32_t kIterations = 64;
  constexpr uint64_t kBase = 0x300000;

  std::atomic<uint32_t> active_writers{0};
  std::barrier start(kWriterThreads + 1);
  std::vector<std::thread> workers;
  workers.reserve(kWriterThreads);

  for (uint32_t tid = 0; tid < kWriterThreads; ++tid) {
    workers.emplace_back([&, tid] {
      std::array<uint8_t, L2Cache::LINE_SIZE> line{};
      start.arrive_and_wait();
      active_writers.fetch_add(1, std::memory_order_release);
      for (uint32_t iteration = 0; iteration < kIterations; ++iteration) {
        for (uint32_t i = 0; i < kLinesPerThread; ++i) {
          const uint64_t addr =
              kBase + (static_cast<uint64_t>(i) * kWriterThreads + tid) * L2Cache::LINE_SIZE;
          for (uint32_t b = 0; b < line.size(); ++b)
            line[b] = static_cast<uint8_t>((tid << 5) ^ iteration ^ i ^ b);
          l2.writeback_line(addr, line.data());
        }
        std::this_thread::yield();
      }
    });
  }

  std::thread flusher([&] {
    start.arrive_and_wait();
    while (active_writers.load(std::memory_order_acquire) < kWriterThreads)
      std::this_thread::yield();
    for (uint32_t i = 0; i < 4; ++i) {
      l2.flush_all();
      std::this_thread::yield();
    }
  });

  for (auto &worker : workers)
    worker.join();
  flusher.join();

  l2.flush_all();

  std::array<uint8_t, L2Cache::LINE_SIZE> expected{};
  std::array<uint8_t, L2Cache::LINE_SIZE> actual{};
  for (uint32_t tid = 0; tid < kWriterThreads; ++tid) {
    for (uint32_t i = 0; i < kLinesPerThread; ++i) {
      const uint64_t addr =
          kBase + (static_cast<uint64_t>(i) * kWriterThreads + tid) * L2Cache::LINE_SIZE;
      for (uint32_t b = 0; b < expected.size(); ++b)
        expected[b] = static_cast<uint8_t>((tid << 5) ^ (kIterations - 1) ^ i ^ b);
      memory.read_block(addr, std::span<uint8_t>(actual));
      EXPECT_EQ(actual, expected) << "addr=0x" << std::hex << addr;
    }
  }
}

TEST(L2CacheTest, InvalidateRangeClampsAtAddressSpaceEnd) {
  GpuMemory memory("memory");
  L2Cache l2("l2");
  l2.set_backing_memory(&memory);

  constexpr uint64_t kLineAddr = std::numeric_limits<uint64_t>::max() - (L2Cache::LINE_SIZE - 1);
  constexpr uint64_t kRangeAddr =
      std::numeric_limits<uint64_t>::max() - (L2Cache::LINE_SIZE / 2 - 1);
  std::array<uint8_t, L2Cache::LINE_SIZE> initial{};
  std::array<uint8_t, L2Cache::LINE_SIZE> replacement{};
  std::array<uint8_t, L2Cache::LINE_SIZE> actual{};
  initial.fill(0x11);
  replacement.fill(0x22);

  memory.write_block(kLineAddr, std::span<const uint8_t>(initial));
  l2.read(kLineAddr, actual.data(), actual.size());
  ASSERT_EQ(actual, initial);

  memory.write_block(kLineAddr, std::span<const uint8_t>(replacement));
  l2.invalidate_range(kRangeAddr, L2Cache::LINE_SIZE, 0);
  l2.read(kLineAddr, actual.data(), actual.size());

  EXPECT_EQ(actual, replacement);
}

TEST(L2CacheTest, InvalidateRangeRefreshesOnlyCoveredSetsAndZeroSizeIsNoOp) {
  GpuMemory memory("memory");
  L2Cache l2("l2");
  l2.set_backing_memory(&memory);

  constexpr uint64_t kBase = 0x400000;
  constexpr uint32_t kLines = 5;
  std::array<std::array<uint8_t, L2Cache::LINE_SIZE>, kLines> initial{};
  std::array<std::array<uint8_t, L2Cache::LINE_SIZE>, kLines> replacement{};
  std::array<uint8_t, L2Cache::LINE_SIZE> actual{};

  for (uint32_t line = 0; line < kLines; ++line) {
    initial[line].fill(static_cast<uint8_t>(0x10 + line));
    replacement[line].fill(static_cast<uint8_t>(0x80 + line));
    const uint64_t addr = kBase + static_cast<uint64_t>(line) * L2Cache::LINE_SIZE;
    memory.write_block(addr, std::span<const uint8_t>(initial[line]));
    l2.read(addr, actual.data(), actual.size());
    ASSERT_EQ(actual, initial[line]);
    memory.write_block(addr, std::span<const uint8_t>(replacement[line]));
  }

  l2.invalidate_range(kBase + L2Cache::LINE_SIZE + 32, 2 * L2Cache::LINE_SIZE, 0);

  for (uint32_t line = 0; line < kLines; ++line) {
    const uint64_t addr = kBase + static_cast<uint64_t>(line) * L2Cache::LINE_SIZE;
    l2.read(addr, actual.data(), actual.size());
    const auto &expected = (line >= 1 && line <= 3) ? replacement[line] : initial[line];
    EXPECT_EQ(actual, expected) << "line=" << line;
  }

  l2.invalidate_range(kBase, 0, 0);
  l2.read(kBase, actual.data(), actual.size());
  EXPECT_EQ(actual, initial[0]);
}

TEST(L2CacheTest, InvalidateRangeOver128LinesUsesExclusiveMaintenance) {
  GpuMemory memory("memory");
  L2Cache l2("l2");
  l2.set_backing_memory(&memory);

  constexpr uint64_t kBase = 0x700000;
  constexpr uint32_t kLines = 129;
  std::array<uint8_t, L2Cache::LINE_SIZE> initial{};
  std::array<uint8_t, L2Cache::LINE_SIZE> replacement{};
  std::array<uint8_t, L2Cache::LINE_SIZE> actual{};

  for (uint32_t line = 0; line < kLines; ++line) {
    const uint64_t addr = kBase + static_cast<uint64_t>(line) * L2Cache::LINE_SIZE;
    initial.fill(static_cast<uint8_t>(line));
    replacement.fill(static_cast<uint8_t>(line + 0x40));
    memory.write_block(addr, std::span<const uint8_t>(initial));
    l2.read(addr, actual.data(), actual.size());
    ASSERT_EQ(actual, initial) << "line=" << line;
    memory.write_block(addr, std::span<const uint8_t>(replacement));
  }

  l2.invalidate_range(kBase, kLines * L2Cache::LINE_SIZE, 0);

  for (uint32_t line = 0; line < kLines; ++line) {
    const uint64_t addr = kBase + static_cast<uint64_t>(line) * L2Cache::LINE_SIZE;
    replacement.fill(static_cast<uint8_t>(line + 0x40));
    l2.read(addr, actual.data(), actual.size());
    EXPECT_EQ(actual, replacement) << "line=" << line;
  }
}

TEST(L2CacheTest, InvalidateRangeOnlyAffectsRequestedVmid) {
  GpuMemory memory("memory");
  L2Cache l2("l2");
  l2.set_backing_memory(&memory);

  constexpr uint32_t kVmidA = 7;
  constexpr uint32_t kVmidB = 8;
  constexpr uint64_t kAddr = 0x500000;
  rocjitsu::KfdProcess process_a(kVmidA);
  rocjitsu::KfdProcess process_b(kVmidB);
  std::array<uint8_t, GpuMemory::PAGE_SIZE> backing_a{};
  std::array<uint8_t, GpuMemory::PAGE_SIZE> backing_b{};
  process_a.map_pages(kAddr, backing_a.data(), backing_a.size());
  process_b.map_pages(kAddr, backing_b.data(), backing_b.size());
  memory.register_process(kVmidA, &process_a.page_table_, &process_a.page_table_mutex_,
                          process_a.page_table_generation());
  memory.register_process(kVmidB, &process_b.page_table_, &process_b.page_table_mutex_,
                          process_b.page_table_generation());

  std::array<uint8_t, L2Cache::LINE_SIZE> initial_a{};
  std::array<uint8_t, L2Cache::LINE_SIZE> initial_b{};
  std::array<uint8_t, L2Cache::LINE_SIZE> dirty_a{};
  std::array<uint8_t, L2Cache::LINE_SIZE> replacement_b{};
  std::array<uint8_t, L2Cache::LINE_SIZE> actual{};
  initial_a.fill(0x11);
  initial_b.fill(0x22);
  dirty_a.fill(0x33);
  replacement_b.fill(0x44);

  memory.write_block(kAddr, std::span<const uint8_t>(initial_a), kVmidA);
  memory.write_block(kAddr, std::span<const uint8_t>(initial_b), kVmidB);
  l2.read(kAddr, actual.data(), actual.size(), Mtype::RW, kVmidA);
  ASSERT_EQ(actual, initial_a);
  l2.read(kAddr, actual.data(), actual.size(), Mtype::RW, kVmidB);
  ASSERT_EQ(actual, initial_b);
  l2.writeback_line(kAddr, dirty_a.data(), Mtype::RW, kVmidA);

  memory.write_block(kAddr, std::span<const uint8_t>(replacement_b), kVmidB);
  l2.invalidate_range(kAddr, L2Cache::LINE_SIZE, kVmidB);
  l2.read(kAddr, actual.data(), actual.size(), Mtype::RW, kVmidB);
  EXPECT_EQ(actual, replacement_b);

  l2.flush_line(kAddr, kVmidA);
  memory.read_block(kAddr, std::span<uint8_t>(actual), kVmidA);
  EXPECT_EQ(actual, dirty_a);
}

TEST(L2CacheTest, InvalidateRangePreservesDirtyBytesOutsidePartialWrite) {
  GpuMemory memory("memory");
  L2Cache l2("l2");
  l2.set_backing_memory(&memory);

  constexpr uint64_t kAddr = 0x600000;
  constexpr uint32_t kOffset = 32;
  std::array<uint8_t, L2Cache::LINE_SIZE> initial{};
  std::array<uint8_t, L2Cache::LINE_SIZE> dirty{};
  std::array<uint8_t, 16> host_write{};
  std::array<uint8_t, L2Cache::LINE_SIZE> expected{};
  std::array<uint8_t, L2Cache::LINE_SIZE> actual{};
  initial.fill(0x11);
  dirty.fill(0x22);
  host_write.fill(0x33);
  expected = dirty;
  std::memcpy(expected.data() + kOffset, host_write.data(), host_write.size());

  memory.write_block(kAddr, std::span<const uint8_t>(initial));
  l2.read(kAddr, actual.data(), actual.size());
  ASSERT_EQ(actual, initial);
  l2.writeback_line(kAddr, dirty.data());

  memory.write_block(kAddr + kOffset, std::span<const uint8_t>(host_write));
  l2.invalidate_range(kAddr + kOffset, host_write.size(), 0);

  memory.read_block(kAddr, std::span<uint8_t>(actual));
  EXPECT_EQ(actual, expected);
  l2.read(kAddr, actual.data(), actual.size());
  EXPECT_EQ(actual, expected);
}

TEST(L2CacheBenchmark, CrossL2SameAddress) {
  run_cross_l2_atomic_benchmark("cross_l2_same_address", true);
}

TEST(L2CacheBenchmark, CrossL2IndependentAddresses) {
  run_cross_l2_atomic_benchmark("cross_l2_independent_addresses", false);
}

TEST(L2CacheBenchmark, AtomicOneHierarchy) {
  run_atomic_hierarchy_benchmark("atomic_hierarchies_1", 1);
}

TEST(L2CacheBenchmark, AtomicTwoHierarchies) {
  run_atomic_hierarchy_benchmark("atomic_hierarchies_2", 2);
}

TEST(L2CacheBenchmark, AtomicFourHierarchies) {
  run_atomic_hierarchy_benchmark("atomic_hierarchies_4", 4);
}

TEST(L2CacheBenchmark, AtomicEightHierarchies) {
  run_atomic_hierarchy_benchmark("atomic_hierarchies_8", 8);
}

TEST(L2CacheBenchmark, ScalarL1Hit) { run_scalar_l1_hit_benchmark(); }

TEST(L2CacheBenchmark, VectorL1Hit) { run_vector_l1_hit_benchmark(); }

TEST(L2CacheBenchmark, InvalidateThreeLines) {
  run_invalidate_range_benchmark("invalidate_3_lines", 3, 200'000);
}

TEST(L2CacheBenchmark, Invalidate129Lines) {
  run_invalidate_range_benchmark("invalidate_129_lines", 129, 5'000);
}

TEST(GpuMemoryTest, BlockAccessHandlesPageBoundaries) {
  GpuMemory memory("memory");

  constexpr uint64_t kAddr = 0x3ff0;
  std::array<uint8_t, 64> input{};
  std::array<uint8_t, 64> output{};
  for (uint32_t i = 0; i < input.size(); ++i)
    input[i] = static_cast<uint8_t>(i * 3);

  memory.write_block(kAddr, std::span<const uint8_t>(input));
  memory.read_block(kAddr, std::span<uint8_t>(output));

  EXPECT_EQ(output, input);
}

} // namespace
