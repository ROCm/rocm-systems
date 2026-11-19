// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/kmd/linux/kfd_process.h"
#include "rocjitsu/kmd/linux/legacy_gpu_vm.h"
#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "rocjitsu/vm/amdgpu/gpu_vm.h"
#include "rocjitsu/vm/amdgpu/l1_scalar_cache.h"
#include "rocjitsu/vm/amdgpu/l1_vector_cache.h"
#include "rocjitsu/vm/amdgpu/l2_cache.h"
#include "rocjitsu/vm/plugins/plugin_config_resolver.h"
#include "util/statistics.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <span>
#include <sstream>
#include <string>
#include <string_view>

namespace {

using namespace rocjitsu::amdgpu;

std::shared_ptr<stats::Scope> make_cache_scope(std::string name) {
  auto config = std::make_shared<stats::Config>();
  config->enabled = true;
  config->default_domain = "model_events";
  return std::make_shared<stats::Scope>(name, name, config);
}

uint64_t read_statistic(const stats::Scope &scope, std::string_view name, int bucket = -1) {
  std::ostringstream output;
  scope.write_samples(output);
  std::istringstream lines(output.str());
  std::string line;
  while (std::getline(lines, line)) {
    flexbuffers::Builder builder;
    if (!rocjitsu::plugin_detail::flexbuffer_from_json(line, builder)) {
      ADD_FAILURE() << "invalid statistics JSON: " << line;
      continue;
    }
    const flexbuffers::Map record = flexbuffers::GetRoot(builder.GetBuffer()).AsMap();
    if (record["metric_id"].AsString().str() != scope.name() + ":" + std::string(name))
      continue;
    if (bucket >= 0 && (record["record"].AsString().str() != "histogram_bucket" ||
                        record["bucket_index"].AsInt32() != bucket))
      continue;
    const std::string value = record[bucket >= 0 ? "count" : "value"].AsString().str();
    uint64_t parsed = 0;
    const std::from_chars_result result =
        std::from_chars(value.data(), value.data() + value.size(), parsed);
    EXPECT_EQ(result.ec, std::errc{});
    EXPECT_EQ(result.ptr, value.data() + value.size());
    return parsed;
  }
  ADD_FAILURE() << "missing statistic " << name;
  return 0;
}

TEST(CacheStatisticsTest, VectorColdWarmAndL2FillHitsHaveIndependentScopes) {
  constexpr uint64_t kAddress = 0x1000;
  constexpr uint32_t kValue = 0x13579bdf;
  GpuMemory memory("memory");
  memory.write32(kAddress, kValue);
  L2Cache l2("l2");
  l2.set_backing_memory(&memory);
  L1VectorCache vector(&l2);
  const std::shared_ptr<stats::Scope> l1_scope = make_cache_scope("vector");
  const std::shared_ptr<stats::Scope> l2_scope = make_cache_scope("l2");
  vector.set_statistics(l1_scope);
  l2.set_statistics(l2_scope);
  uint32_t observed = 0;
  for (unsigned attempt = 0; attempt < 2; ++attempt) {
    ASSERT_EQ(vector.load(&kAddress, 1, sizeof(observed), 1, reinterpret_cast<uint8_t *>(&observed),
                          Mtype::RW, false, false, 1),
              VmAccessOutcome::Complete);
    EXPECT_EQ(observed, kValue);
  }
  EXPECT_EQ(read_statistic(*l1_scope, "cache.read_requests"), 2u);
  EXPECT_EQ(read_statistic(*l1_scope, "cache.read_bytes"), 8u);
  EXPECT_EQ(read_statistic(*l1_scope, "cache.read_miss_chunks"), 1u);
  EXPECT_EQ(read_statistic(*l1_scope, "cache.read_hit_chunks"), 1u);
  EXPECT_EQ(read_statistic(*l2_scope, "cache.read_requests"), 1u);
  EXPECT_EQ(read_statistic(*l2_scope, "cache.read_miss_chunks"), 1u);
  EXPECT_EQ(l2.backing_read_transactions(), 1u);

  vector.invalidate_all();
  ASSERT_EQ(vector.load(&kAddress, 1, sizeof(observed), 1, reinterpret_cast<uint8_t *>(&observed),
                        Mtype::RW, false, false, 1),
            VmAccessOutcome::Complete);
  EXPECT_EQ(read_statistic(*l1_scope, "cache.read_miss_chunks"), 2u);
  EXPECT_EQ(read_statistic(*l2_scope, "cache.read_hit_chunks"), 1u);
  EXPECT_EQ(l2.backing_read_transactions(), 1u);
}

TEST(CacheStatisticsTest, VectorBypassesDoNotBecomeMisses) {
  constexpr uint64_t kAddress = 0x1000;
  GpuMemory memory("memory");
  memory.write32(kAddress, 42);
  L2Cache l2("l2");
  l2.set_backing_memory(&memory);
  L1VectorCache vector(&l2);
  const std::shared_ptr<stats::Scope> scope = make_cache_scope("vector");
  vector.set_statistics(scope);
  for (Mtype mtype : {Mtype::UC, Mtype::CC, Mtype::RW}) {
    uint32_t value = 0;
    ASSERT_EQ(vector.load(&kAddress, 1, sizeof(value), 1, reinterpret_cast<uint8_t *>(&value),
                          mtype, false, true, 1),
              VmAccessOutcome::Complete);
    EXPECT_EQ(value, 42u);
  }
  EXPECT_EQ(read_statistic(*scope, "cache.read_requests"), 3u);
  EXPECT_EQ(read_statistic(*scope, "cache.read_bypass_chunks"), 3u);
  EXPECT_EQ(read_statistic(*scope, "cache.read_hit_chunks"), 0u);
  EXPECT_EQ(read_statistic(*scope, "cache.read_miss_chunks"), 0u);
}

TEST(CacheStatisticsTest, ScalarDwordAndByteLoadsPreserveTheirChunkGranularity) {
  constexpr uint64_t kAddress = 0x1000;
  GpuMemory memory("memory");
  memory.write32(kAddress, 17);
  memory.write32(kAddress + 4, 29);
  L2Cache l2("l2");
  l2.set_backing_memory(&memory);
  L1ScalarCache scalar(&l2);
  const std::shared_ptr<stats::Scope> scope = make_cache_scope("scalar");
  scalar.set_statistics(scope);
  std::array<uint32_t, 2> words{};
  ASSERT_EQ(scalar.load(kAddress, words.size(), words.data()), VmAccessOutcome::Complete);
  EXPECT_EQ(words, (std::array<uint32_t, 2>{17, 29}));
  EXPECT_EQ(read_statistic(*scope, "cache.read_miss_chunks"), 1u);
  EXPECT_EQ(read_statistic(*scope, "cache.read_hit_chunks"), 1u);
  ASSERT_EQ(scalar.load_bytes(kAddress, sizeof(words), reinterpret_cast<uint8_t *>(words.data())),
            VmAccessOutcome::Complete);
  EXPECT_EQ(read_statistic(*scope, "cache.read_hit_chunks"), 2u);
  EXPECT_EQ(read_statistic(*scope, "cache.read_requests"), 2u);
  EXPECT_EQ(read_statistic(*scope, "cache.read_bytes"), 16u);
  ASSERT_EQ(scalar.load(kAddress, 1, words.data(), 0, false, Mtype::RW, true),
            VmAccessOutcome::Complete);
  ASSERT_EQ(scalar.load_bytes(kAddress, sizeof(uint32_t), reinterpret_cast<uint8_t *>(words.data()),
                              0, Mtype::UC),
            VmAccessOutcome::Complete);
  EXPECT_EQ(read_statistic(*scope, "cache.read_bypass_chunks"), 2u);
  EXPECT_EQ(read_statistic(*scope, "cache.read_miss_chunks"), 1u);
}

TEST(CacheStatisticsTest, CrossingAnL2LineCountsTwoChunksAndOneRequest) {
  constexpr uint64_t kAddress = 0x1000 + L2Cache::LINE_SIZE - 2;
  constexpr uint32_t kValue = 0x98765432;
  GpuMemory memory("memory");
  memory.write32(kAddress, kValue);
  L2Cache l2("l2");
  l2.set_backing_memory(&memory);
  const std::shared_ptr<stats::Scope> scope = make_cache_scope("l2");
  l2.set_statistics(scope);
  uint32_t value = 0;
  ASSERT_EQ(l2.read(kAddress, reinterpret_cast<uint8_t *>(&value), sizeof(value)),
            VmAccessOutcome::Complete);
  EXPECT_EQ(value, kValue);
  EXPECT_EQ(read_statistic(*scope, "cache.read_requests"), 1u);
  EXPECT_EQ(read_statistic(*scope, "cache.read_bytes"), 4u);
  EXPECT_EQ(read_statistic(*scope, "cache.read_miss_chunks"), 2u);
  ASSERT_EQ(l2.read(kAddress, reinterpret_cast<uint8_t *>(&value), sizeof(value)),
            VmAccessOutcome::Complete);
  EXPECT_EQ(read_statistic(*scope, "cache.read_hit_chunks"), 2u);
  EXPECT_EQ(read_statistic(*scope, "cache.read_size", 2), 2u);
  EXPECT_EQ(read_statistic(*scope, "cache.read_size.count"), 2u);
}

TEST(CacheStatisticsTest, PeerProbesCountDemandWhileFillsDoNot) {
  constexpr uint64_t kAddress = 0x1000;
  constexpr std::array<uint8_t, 4> kBytes{1, 3, 5, 7};
  GpuMemory memory("memory");
  L2Cache l2("l2");
  l2.set_backing_memory(&memory);
  L1VectorCache vector(&l2);
  L1ScalarCache scalar(&l2);
  const auto check_probe = [&](auto &cache, const std::string &name) {
    const std::shared_ptr<stats::Scope> scope = make_cache_scope(name);
    cache.set_statistics(scope);
    std::array<uint8_t, 4> observed{};
    EXPECT_FALSE(cache.try_read_peer_cached(kAddress, observed, 0, 0, 0));
    ASSERT_TRUE(cache.try_fill_peer_cached(kAddress, kBytes, 0, 0, 0));
    EXPECT_TRUE(cache.try_read_peer_cached(kAddress, observed, 0, 0, 0));
    EXPECT_EQ(observed, kBytes);
    EXPECT_EQ(read_statistic(*scope, "cache.read_requests"), 2u);
    EXPECT_EQ(read_statistic(*scope, "cache.read_miss_chunks"), 1u);
    EXPECT_EQ(read_statistic(*scope, "cache.read_hit_chunks"), 1u);
  };
  check_probe(vector, "vector");
  check_probe(scalar, "scalar");
  check_probe(l2, "l2");
}

TEST(CacheStatisticsTest, UnavailablePeerProbeDoesNotClaimACacheMiss) {
  L1VectorCache cache;
  const std::shared_ptr<stats::Scope> scope = make_cache_scope("vector");
  cache.set_statistics(scope);
  std::array<uint8_t, 4> bytes{};
  EXPECT_FALSE(cache.try_read_peer_cached(0x1000, bytes, 0, 0, 0));
  EXPECT_EQ(read_statistic(*scope, "cache.read_requests"), 1u);
  EXPECT_EQ(read_statistic(*scope, "cache.peer_probe_unavailable"), 1u);
  EXPECT_EQ(read_statistic(*scope, "cache.read_miss_chunks"), 0u);
  EXPECT_EQ(read_statistic(*scope, "cache.read_hit_chunks"), 0u);
}

TEST(CacheStatisticsTest, CurrentLegacyGrantFastPathStillCountsWarmVectorReads) {
  constexpr uint64_t kAddress = 0x100000;
  constexpr uint32_t kVmid = 7;
  std::array<uint8_t, GpuMemory::PAGE_SIZE> page{};
  const uint32_t expected = 0x12345678;
  std::memcpy(page.data(), &expected, sizeof(expected));
  rocjitsu::KfdProcess process(kVmid);
  process.map_pages(kAddress, page.data(), page.size(), Mtype::RW);
  GpuMemory memory("memory");
  GpuVm vm;
  LegacyGpuVmAdapter adapter(vm, &memory);
  const AddressSpaceHandle handle = adapter.register_address_space(
      kVmid, {.page_table = &process.page_table_,
              .page_table_mutex = &process.page_table_mutex_,
              .page_table_generation = process.page_table_generation(),
              .request_mutex = process.page_table_request_mutex(),
              .mutation_epoch = process.page_table_mutation_epoch()});
  ASSERT_TRUE(handle);
  L2Cache l2("l2");
  l2.set_backing_memory(&memory);
  l2.set_gpu_vm(&vm);
  L1VectorCache vector(&l2);
  vector.set_gpu_vm(&vm);
  const std::shared_ptr<stats::Scope> scope = make_cache_scope("vector");
  vector.set_statistics(scope);
  for (unsigned attempt = 0; attempt < 2; ++attempt) {
    uint32_t value = 0;
    ASSERT_EQ(vector.load(&kAddress, 1, sizeof(value), 1, reinterpret_cast<uint8_t *>(&value),
                          Mtype::RW, false, false, 1, kVmid),
              VmAccessOutcome::Complete);
    EXPECT_EQ(value, expected);
  }
  EXPECT_EQ(read_statistic(*scope, "cache.read_requests"), 2u);
  EXPECT_EQ(read_statistic(*scope, "cache.read_miss_chunks"), 1u);
  EXPECT_EQ(read_statistic(*scope, "cache.read_hit_chunks"), 1u);
  EXPECT_EQ(l2.backing_read_transactions(), 1u);
}

class StatisticsLocalTranslator final : public AddressSpaceTranslator {
public:
  VmTranslationResult translate(uint64_t address, size_t size, VmAccessKind) const override {
    if (address >= 4096 || size == 0 || size > 4096 - address)
      return {.outcome = VmAccessOutcome::Faulted, .translation = {}};
    return {.outcome = VmAccessOutcome::Complete,
            .translation = {.domain = VmMemoryDomain::Local,
                            .address = address,
                            .contiguous_bytes = 4096 - address,
                            .mtype = Mtype::RW,
                            .gfx12_pte_mtype = Gfx12PteMtype::NC,
                            .permissions = {.readable = true, .writable = true}}};
  }
};

class StatisticsLocalBacking final : public PhysicalMemoryAccess {
public:
  bool supports_translated_local_cache() const override { return true; }
  MemoryIngressAdmission admit_pt_rewalk() const override {
    return {.outcome = VmAccessOutcome::Complete, .permit = {}};
  }
  VmAccessOutcome read(VmMemoryDomain domain, uint64_t address,
                       std::span<std::byte> bytes) override {
    if (domain != VmMemoryDomain::Local || address > 4096 || bytes.size() > 4096 - address)
      return VmAccessOutcome::Faulted;
    if (unavailable)
      return VmAccessOutcome::Unavailable;
    std::fill(bytes.begin(), bytes.end(), std::byte{0x3a});
    return VmAccessOutcome::Complete;
  }
  VmAccessOutcome write(VmMemoryDomain, uint64_t, std::span<const std::byte>) override {
    return VmAccessOutcome::Faulted;
  }
  bool unavailable = false;
};

TEST(CacheStatisticsTest, TranslatedLocalReadsCountRetriesAndWarmHits) {
  auto coherence = std::make_shared<DeviceCacheCoherence>();
  auto backing = std::make_shared<StatisticsLocalBacking>();
  auto translator = std::make_shared<StatisticsLocalTranslator>();
  GpuVm vm;
  const AddressSpaceHandle handle = vm.register_unrouted_address_space(77, translator, backing);
  ASSERT_TRUE(handle);
  const auto access = vm.snapshot(handle);
  ASSERT_TRUE(access);
  L2Cache l2("l2", coherence);
  l2.set_gpu_vm(&vm);
  L1VectorCache vector(&l2);
  vector.set_gpu_vm(&vm);
  const std::shared_ptr<stats::Scope> l1_scope = make_cache_scope("vector");
  const std::shared_ptr<stats::Scope> l2_scope = make_cache_scope("l2");
  vector.set_statistics(l1_scope);
  l2.set_statistics(l2_scope);
  std::array<uint8_t, 4> bytes{};
  backing->unavailable = true;
  EXPECT_EQ(vector.load_translated_local(*access, 0x80, bytes, Mtype::RW, false, false),
            VmAccessOutcome::Unavailable);
  backing->unavailable = false;
  for (unsigned attempt = 0; attempt < 2; ++attempt)
    ASSERT_EQ(vector.load_translated_local(*access, 0x80, bytes, Mtype::RW, false, false),
              VmAccessOutcome::Complete);
  EXPECT_EQ(bytes, (std::array<uint8_t, 4>{0x3a, 0x3a, 0x3a, 0x3a}));
  EXPECT_EQ(read_statistic(*l1_scope, "cache.read_requests"), 3u);
  EXPECT_EQ(read_statistic(*l1_scope, "cache.read_miss_chunks"), 2u);
  EXPECT_EQ(read_statistic(*l1_scope, "cache.read_hit_chunks"), 1u);
  EXPECT_EQ(read_statistic(*l2_scope, "cache.read_requests"), 2u);
  EXPECT_EQ(read_statistic(*l2_scope, "cache.read_miss_chunks"), 2u);
  ASSERT_EQ(vector.load_translated_local(*access, 0x80, bytes, Mtype::RW, false, true),
            VmAccessOutcome::Complete);
  EXPECT_EQ(read_statistic(*l1_scope, "cache.read_bypass_chunks"), 1u);
  EXPECT_EQ(read_statistic(*l2_scope, "cache.read_hit_chunks"), 1u);
}

} // namespace
