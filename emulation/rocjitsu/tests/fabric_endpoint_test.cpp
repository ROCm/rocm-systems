// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "embedded_schema.h"
#include "rocjitsu/config/config_loader.h"
#include "rocjitsu/kmd/linux/libc_passthrough.h"
#include "rocjitsu/vm/amdgpu/device_cache_coherence.h"
#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "rocjitsu/vm/amdgpu/l2_cache.h"
#include "rocjitsu/vm/amdgpu/mtype.h"
#include "rocjitsu/vm/amdgpu/prepared_canonical_writeback.h"
#include "rocjitsu/vm/fabric_backing_kfd.h"
#include "rocjitsu/vm/fabric_endpoint.h"
#include "rocjitsu/vm/topology_assembly.h"
#include "rocjitsu/vm/virtual_machine.h"
#include "util/diagnostic.h"

#include "simdojo/sim/simulation.h"
#include "simdojo/sim/topology.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <fcntl.h>
#include <future>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <sys/mman.h>
#include <thread>
#include <unistd.h>
#include <utility>
#include <vector>

namespace rocjitsu {
namespace {

constexpr uint32_t kRequester = 101;
constexpr uint32_t kOwner = 202;
constexpr uint32_t kThird = 303;
constexpr uint64_t kVa = 0x100000;

std::vector<config::SystemPeerRoute> fanout_routes() {
  return {{"g0", "g1", kRequester, kOwner, {"e0", "sw", "e1"}},
          {"g0", "g2", kRequester, kThird, {"e0", "sw", "e2"}}};
}

FabricEndpoint::EffectAdmission admit_local_test_effect() {
  return [](const FabricBacking &, bool write, amdgpu::MemoryIngressPermit &) {
    EXPECT_TRUE(write);
    FabricEndpoint::EffectAuthority authority;
    authority.permit = amdgpu::MemoryIngressPermit([] {});
    return authority;
  };
}

std::string assembly_config(std::string_view manifest = {}) {
  std::string json = R"({"num_threads":1,"cpu_dispatch_threads":1,"async_helper_threads":0,
    "vm":{"arch":"cdna5","gpu":{"device":{
      "gpu_id":101,"unique_id":1001,"num_sdma_engines":0}}},
    "topology":{"root":{"name":"soc","type":"soc","children":[
      {"name":"vram","type":"gpu_memory"}]}})";
  if (!manifest.empty()) {
    json += R"(,"system":)";
    json += manifest;
  }
  json += '}';
  return json;
}

constexpr std::string_view kLocalSwitchManifest = R"({
  "nodes":[{"id":"local","gpus":[
    {"id":"g0","gpu_id":101,"unique_id":1001,"location_id":768,"drm_render_minor":128},
    {"id":"g1","gpu_id":202,"unique_id":1002,"location_id":1024,"drm_render_minor":129}
  ]}],
  "endpoints":[
    {"id":"e0","kind":"gpu","gpu":"g0"},
    {"id":"e1","kind":"gpu","gpu":"g1"},
    {"id":"sw","kind":"switch"}
  ],
  "links":[
    {"src":"e0","dst":"sw","link_class":"scale_up"},
    {"src":"e1","dst":"sw","link_class":"scale_up"}
  ]
})";

std::string assembly_coherence_config() {
  std::string json = R"({"num_threads":1,"cpu_dispatch_threads":1,"async_helper_threads":0,
    "exec_mode":"clocked",
    "vm":{"arch":"cdna5","gpu":{"device":{
      "gpu_id":101,"unique_id":1001,"num_sdma_engines":0}}},
    "topology":{"root":{"name":"soc","type":"soc","children":[
      {"name":"vram","type":"gpu_memory"},
      {"name":"iod0","type":"iod","config":[{"key":"num_hbm_stacks","value":"1"}]},
      {"name":"xcd0","type":"xcd","children":[
        {"name":"l2","type":"l2_cache"},
        {"name":"cp","type":"command_processor"},
        {"name":"se0","type":"shader_engine","children":[
          {"name":"cu0","type":"compute_unit"}]}]}]},
      "links":[
        {"src":"xcd0.cp.req_0","dst":"xcd0.se0.cu0.cpl","latency":1,"weight":2},
        {"src":"xcd0.se0.cu0.req","dst":"xcd0.l2.cpl_0","latency":1,"weight":10},
        {"src":"xcd0.l2.req","dst":"iod0.msc.cpl_0","latency":1,"weight":3}]})";
  json += R"(,"system":)";
  json += kLocalSwitchManifest;
  json += '}';
  return json;
}

config::LoadedConfig load_assembly_config(std::string json, std::string_view node_id = {}) {
  util::StringDiagnostic diagnostic;
  auto loaded = config::load_config_from_string_for_node(json, kEmbeddedSchema, node_id,
                                                         diagnostic.emitter());
  if (loaded.failed())
    throw std::runtime_error(diagnostic.message());
  return std::move(loaded.value());
}

constexpr std::string_view kRemoteSharedSwitchManifest = R"({
  "nodes":[
    {"id":"local","gpus":[
      {"id":"g0","gpu_id":101,"unique_id":1001,"location_id":768,"drm_render_minor":128},
      {"id":"g1","gpu_id":202,"unique_id":1002,"location_id":1024,"drm_render_minor":129}
    ]},
    {"id":"remote","gpus":[
      {"id":"g2","gpu_id":303,"unique_id":1003,"location_id":1280,"drm_render_minor":130}
    ]}
  ],
  "endpoints":[
    {"id":"e0","kind":"gpu","gpu":"g0"},
    {"id":"e1","kind":"gpu","gpu":"g1"},
    {"id":"e2","kind":"gpu","gpu":"g2"},
    {"id":"sw","kind":"switch"}
  ],
  "links":[
    {"src":"e0","dst":"sw","link_class":"scale_up"},
    {"src":"e1","dst":"sw","link_class":"scale_up"},
    {"src":"e2","dst":"sw","link_class":"scale_up"}
  ]
})";

class BackingFixture {
public:
  BackingFixture() : process_(1) {
    libc_passthrough().resolve();
    fd_ = libc_passthrough().memfd_create("fabric-endpoint-test", MFD_CLOEXEC | MFD_ALLOW_SEALING);
    if (fd_ < 0 || ::ftruncate(fd_, 4096) != 0 ||
        libc_passthrough().fcntl(fd_, F_ADD_SEALS, F_SEAL_SHRINK) != 0)
      return;
    backing_ = KfdProcess::GpuBacking::create_memfd_alias(fd_, 4096, kOwner);
    if (!backing_)
      return;
    KfdProcess::GpuAllocation allocation;
    allocation.gpu_va = kVa;
    allocation.size = 4096;
    allocation.gpu_id = kOwner;
    allocation.backing_owner = backing_;
    allocation.admission_generations.emplace(kOwner, 1);
    process_.allocations_.emplace(1, std::move(allocation));
    process_.publish_gpu_grants_locked();
  }

  ~BackingFixture() {
    if (fd_ >= 0)
      libc_passthrough().close(fd_);
  }

  bool valid() const { return backing_ != nullptr; }
  KfdProcess::GpuBackingLease lease(size_t offset, size_t size) const {
    return process_.gpu_peer_backing_grant(kOwner, kVa + offset, size).backing;
  }

private:
  int fd_ = -1;
  KfdProcess process_;
  std::shared_ptr<const KfdProcess::GpuBacking> backing_;
};

class VectorBackingOwner final : public FabricBacking::Owner {
public:
  static constexpr uint64_t kBase = 0x400;

  VectorBackingOwner() { bytes_.fill(std::byte{0x7b}); }

  uint64_t base_offset() const noexcept override { return kBase; }
  size_t extent() const noexcept override { return bytes_.size(); }

  bool read(uint64_t offset, std::span<std::byte> result) noexcept override {
    if (!contains(offset, result.size()))
      return false;
    std::copy_n(bytes_.begin() + static_cast<size_t>(offset - kBase), result.size(),
                result.begin());
    ++reads_;
    last_offset_ = offset;
    return true;
  }

  bool write(uint64_t offset, std::span<const std::byte> bytes) noexcept override {
    if (reject_writes_ || !contains(offset, bytes.size()))
      return false;
    std::copy(bytes.begin(), bytes.end(), bytes_.begin() + static_cast<size_t>(offset - kBase));
    ++writes_;
    last_offset_ = offset;
    return true;
  }

  const std::array<std::byte, 16> &bytes_for_testing() const { return bytes_; }
  size_t reads() const { return reads_; }
  size_t writes() const { return writes_; }
  uint64_t last_offset() const { return last_offset_; }
  void reject_writes() { reject_writes_ = true; }

private:
  bool contains(uint64_t offset, size_t size) const noexcept {
    if (offset < kBase)
      return false;
    const uint64_t relative = offset - kBase;
    return relative <= bytes_.size() && size <= bytes_.size() - relative;
  }

  std::array<std::byte, 16> bytes_{};
  size_t reads_ = 0;
  size_t writes_ = 0;
  uint64_t last_offset_ = 0;
  bool reject_writes_ = false;
};

class CacheableBackingOwner final : public FabricBacking::Owner {
public:
  static constexpr uint64_t kBase = 0x400;
  static constexpr size_t kExtent = 0x200;

  explicit CacheableBackingOwner(std::shared_ptr<amdgpu::GpuMemory> memory)
      : memory_(std::move(memory)) {
    memory_->write8(kBase, 0);
  }

  uint64_t base_offset() const noexcept override { return kBase; }
  size_t extent() const noexcept override { return kExtent; }
  uint64_t backing_id() const noexcept override {
    return reinterpret_cast<uintptr_t>(memory_.get());
  }

  bool read(uint64_t offset, std::span<std::byte> result) noexcept override {
    if (!contains(offset, result.size()))
      return false;
    memory_->read_block(offset, {reinterpret_cast<uint8_t *>(result.data()), result.size()});
    ++reads_;
    return true;
  }

  bool write(uint64_t offset, std::span<const std::byte> bytes) noexcept override {
    if (!contains(offset, bytes.size()))
      return false;
    memory_->write_block(offset, {reinterpret_cast<const uint8_t *>(bytes.data()), bytes.size()});
    ++writes_;
    return true;
  }

  size_t reads() const { return reads_; }
  size_t writes() const { return writes_; }

private:
  bool contains(uint64_t offset, size_t size) const noexcept {
    if (offset < kBase)
      return false;
    const uint64_t relative = offset - kBase;
    return relative <= kExtent && size <= kExtent - relative;
  }

  std::shared_ptr<amdgpu::GpuMemory> memory_;
  size_t reads_ = 0;
  size_t writes_ = 0;
};

/// Host-owned control storage mapped into a GPU VM as System memory.
class HostMappedControlMemory final : public amdgpu::AddressSpaceTranslator,
                                      public amdgpu::PhysicalMemoryAccess {
public:
  static constexpr uint64_t kBase = 0x2000;

  amdgpu::VmTranslationResult translate(uint64_t address, size_t size,
                                        amdgpu::VmAccessKind) const override {
    if (!contains(address, size))
      return {};
    return {.outcome = amdgpu::VmAccessOutcome::Complete,
            .translation = {.domain = amdgpu::VmMemoryDomain::System,
                            .address = address,
                            .contiguous_bytes = bytes_.size() - (address - kBase),
                            .mtype = amdgpu::Mtype::UC,
                            .permissions = {.readable = true, .writable = true}}};
  }

  amdgpu::VmAccessOutcome read(amdgpu::VmMemoryDomain domain, uint64_t address,
                               std::span<std::byte> bytes) override {
    if (domain != amdgpu::VmMemoryDomain::System || !contains(address, bytes.size()))
      return amdgpu::VmAccessOutcome::Faulted;
    std::ranges::copy_n(bytes_.begin() + (address - kBase), bytes.size(), bytes.begin());
    return amdgpu::VmAccessOutcome::Complete;
  }

  amdgpu::VmAccessOutcome write(amdgpu::VmMemoryDomain domain, uint64_t address,
                                std::span<const std::byte> bytes) override {
    if (domain != amdgpu::VmMemoryDomain::System || !contains(address, bytes.size()))
      return amdgpu::VmAccessOutcome::Faulted;
    std::ranges::copy(bytes, bytes_.begin() + (address - kBase));
    return amdgpu::VmAccessOutcome::Complete;
  }

private:
  bool contains(uint64_t address, size_t size) const {
    return address >= kBase && address - kBase <= bytes_.size() &&
           size <= bytes_.size() - (address - kBase);
  }

  std::array<std::byte, 2 * sizeof(uint32_t)> bytes_{};
};

class FabricEndpointTest : public ::testing::Test {
protected:
  void SetUp() override {
    ASSERT_TRUE(backing_.valid());
    auto root = std::make_unique<simdojo::CompositeComponent>("root");
    auto fabric = std::make_unique<FabricEndpoint>(fanout_routes());
    endpoint_ = fabric.get();
    root->add_child(std::move(fabric));
    engine_.topology().set_root(std::move(root));
    endpoint_->wire(engine_.topology(), simdojo::ExecMode::FUNCTIONAL);
    engine_.create();
    engine_.register_as_primary();
  }

  void TearDown() override {
    engine_.primary_release();
    engine_.shutdown();
  }

  KfdProcess::GpuBackingLease lease(size_t offset, size_t size) const {
    return backing_.lease(offset, size);
  }
  FabricBacking fabric_lease(size_t offset, size_t size) const {
    return make_fabric_backing(lease(offset, size));
  }

  BackingFixture backing_;
  simdojo::SimulationEngine engine_{{.num_threads = 1}};
  FabricEndpoint *endpoint_ = nullptr;
};

class FabricEndpointClockedTest : public ::testing::Test {
protected:
  void SetUp() override {
    ASSERT_TRUE(backing_.valid());
    auto root = std::make_unique<simdojo::CompositeComponent>("root");
    std::vector<config::SystemPeerRoute> routes = {
        {"g0", "g1", kRequester, kOwner, {"e0", "sw", "e1"}}};
    auto fabric = std::make_unique<FabricEndpoint>(std::move(routes));
    endpoint_ = fabric.get();
    root->add_child(std::move(fabric));
    engine_.topology().set_root(std::move(root));
    endpoint_->wire(engine_.topology(), simdojo::ExecMode::CLOCKED);
    engine_.create();
    engine_.register_as_primary();
  }

  void TearDown() override { stop(); }

  void stop() {
    if (stopped_)
      return;
    engine_.primary_release();
    engine_.shutdown();
    stopped_ = true;
  }

  KfdProcess::GpuBackingLease lease(size_t offset, size_t size) const {
    return backing_.lease(offset, size);
  }
  FabricBacking fabric_lease(size_t offset, size_t size) const {
    return make_fabric_backing(lease(offset, size));
  }

  BackingFixture backing_;
  simdojo::SimulationEngine engine_{{.num_threads = 1}};
  FabricEndpoint *endpoint_ = nullptr;
  bool stopped_ = false;
};

class FabricEndpointLocalTest : public ::testing::Test {
protected:
  void SetUp() override {
    auto root = std::make_unique<simdojo::CompositeComponent>("root");
    auto fabric = std::make_unique<FabricEndpoint>(std::vector<config::SystemPeerRoute>{});
    endpoint_ = fabric.get();
    root->add_child(std::move(fabric));
    engine_.topology().set_root(std::move(root));
    endpoint_->wire(engine_.topology(), simdojo::ExecMode::CLOCKED);
    engine_.create();
    engine_.register_as_primary();
  }

  void TearDown() override {
    engine_.primary_release();
    engine_.shutdown();
  }

  simdojo::SimulationEngine engine_{{.num_threads = 1}};
  FabricEndpoint *endpoint_ = nullptr;
};

class FabricEndpointCoherenceTest : public ::testing::Test {
protected:
  void SetUp() override {
    memory_ = std::make_shared<amdgpu::GpuMemory>("peer_backing");
    owner_ = std::make_shared<CacheableBackingOwner>(memory_);
    requester_domain_ = std::make_shared<amdgpu::DeviceCacheCoherence>();
    owner_domain_ = std::make_shared<amdgpu::DeviceCacheCoherence>();
    requester_l2_ = std::make_unique<amdgpu::L2Cache>("requester_l2", requester_domain_);
    owner_l2_ = std::make_unique<amdgpu::L2Cache>("owner_l2", owner_domain_);
    for (amdgpu::L2Cache *l2 : {requester_l2_.get(), owner_l2_.get()}) {
      l2->set_backing_memory(memory_.get());
      l2->set_legacy_maintenance_memory(memory_.get());
      l2->set_legacy_maintenance_vm(&gpu_vm_);
    }

    auto root = std::make_unique<simdojo::CompositeComponent>("root");
    std::vector<config::SystemPeerRoute> routes = {
        {"g0", "g1", kRequester, kOwner, {"e0", "sw", "e1"}}};
    auto fabric = std::make_unique<FabricEndpoint>(std::move(routes),
                                                   std::vector{requester_domain_, owner_domain_});
    endpoint_ = fabric.get();
    root->add_child(std::move(fabric));
    engine_.topology().set_root(std::move(root));
    endpoint_->wire(engine_.topology(), simdojo::ExecMode::CLOCKED);
    engine_.create();
    engine_.register_as_primary();
  }

  void TearDown() override {
    engine_.primary_release();
    engine_.shutdown();
  }

  FabricBacking backing(uint64_t offset, size_t size) const {
    return FabricBacking(owner_, kOwner, offset, size);
  }

  void step_to_target() {
    ASSERT_TRUE(engine_.step()); // owner submits at tick 0
    ASSERT_TRUE(engine_.step()); // switch receives at tick 1
    ASSERT_TRUE(engine_.step()); // target effect occurs at tick 2
  }

  std::shared_ptr<amdgpu::GpuMemory> memory_;
  std::shared_ptr<CacheableBackingOwner> owner_;
  std::shared_ptr<amdgpu::DeviceCacheCoherence> requester_domain_;
  std::shared_ptr<amdgpu::DeviceCacheCoherence> owner_domain_;
  amdgpu::GpuVm gpu_vm_;
  std::unique_ptr<amdgpu::L2Cache> requester_l2_;
  std::unique_ptr<amdgpu::L2Cache> owner_l2_;
  simdojo::SimulationEngine engine_{{.num_threads = 1}};
  FabricEndpoint *endpoint_ = nullptr;
};

class FabricVmMemory final : public amdgpu::AddressSpaceTranslator,
                             public amdgpu::PhysicalMemoryAccess {
public:
  FabricVmMemory(FabricEndpoint &endpoint, const BackingFixture &backing)
      : endpoint_(endpoint), backing_(backing) {}

  amdgpu::VmTranslationResult translate(uint64_t, size_t, amdgpu::VmAccessKind) const override {
    return {.outcome = amdgpu::VmAccessOutcome::Faulted, .translation = {}};
  }

  amdgpu::VmDataRoute classify_data_route(uint64_t, size_t, amdgpu::VmAccessKind) const override {
    return amdgpu::VmDataRoute::Deferred;
  }

  std::optional<amdgpu::VmTransferToken>
  prepare_deferred_read(amdgpu::PhysicalMemoryAccess &, uint64_t address, size_t size,
                        amdgpu::VmAccessKind access,
                        const std::shared_ptr<amdgpu::VmTransferEffectGate> &gate) const override {
    if (access != amdgpu::VmAccessKind::Read || address < kVa)
      return std::nullopt;
    auto prepared = endpoint_.prepare_read(
        kRequester, kOwner, make_fabric_backing(backing_.lease(address - kVa, size)), gate);
    if (!prepared)
      return std::nullopt;
    return amdgpu::VmTransferToken(std::move(prepared.operation), size);
  }

  std::optional<amdgpu::VmTransferToken>
  prepare_deferred_write(amdgpu::PhysicalMemoryAccess &, uint64_t address,
                         std::span<const std::byte> bytes,
                         const std::shared_ptr<amdgpu::VmTransferEffectGate> &gate) const override {
    if (address < kVa)
      return std::nullopt;
    auto prepared = endpoint_.prepare_write(
        kRequester, kOwner, make_fabric_backing(backing_.lease(address - kVa, bytes.size())), bytes,
        gate);
    if (!prepared)
      return std::nullopt;
    return amdgpu::VmTransferToken(std::move(prepared.operation), bytes.size());
  }

  amdgpu::VmAccessOutcome read(amdgpu::VmMemoryDomain, uint64_t, std::span<std::byte>) override {
    return amdgpu::VmAccessOutcome::Faulted;
  }

  amdgpu::VmAccessOutcome write(amdgpu::VmMemoryDomain, uint64_t,
                                std::span<const std::byte>) override {
    return amdgpu::VmAccessOutcome::Faulted;
  }

private:
  FabricEndpoint &endpoint_;
  const BackingFixture &backing_;
};

TEST_F(FabricEndpointLocalTest, OwnerInboxWritesWithoutRouteOrWireMessages) {
  auto owner = std::make_shared<VectorBackingOwner>();
  auto gate = std::make_shared<amdgpu::VmTransferEffectGate>();
  constexpr uint64_t kOffset = VectorBackingOwner::kBase + 4;
  const std::array<std::byte, 4> payload{std::byte{0x12}, std::byte{0x23}, std::byte{0x34},
                                         std::byte{0x45}};
  EXPECT_FALSE(endpoint_->can_route(kRequester, kOwner));
  EXPECT_TRUE(engine_.topology().links().empty());
  auto write =
      endpoint_->prepare_local_write(kOwner, FabricBacking(owner, kOwner, kOffset, payload.size()),
                                     payload, gate, admit_local_test_effect());
  ASSERT_TRUE(write);
  write.operation->submit();
  EXPECT_EQ(write.operation->poll({}), amdgpu::VmAccessOutcome::Unavailable);

  ASSERT_TRUE(engine_.step());
  EXPECT_EQ(write.operation->poll({}), amdgpu::VmAccessOutcome::Complete);
  EXPECT_EQ(owner->writes(), 1u);
  EXPECT_EQ(owner->last_offset(), kOffset);
  EXPECT_TRUE(std::equal(payload.begin(), payload.end(), owner->bytes_for_testing().begin() + 4));
  const auto diagnostics = endpoint_->diagnostics();
  EXPECT_EQ(diagnostics.submissions, 1u);
  EXPECT_EQ(diagnostics.effects, 1u);
  EXPECT_EQ(diagnostics.request_messages, 0u);
  EXPECT_EQ(diagnostics.response_messages, 0u);
  EXPECT_EQ(diagnostics.owner_thread, diagnostics.handler_thread);

  gate->revoke();
  EXPECT_EQ(write.operation->reconcile_revocation({}), amdgpu::VmAccessOutcome::Complete);
  EXPECT_EQ(owner->writes(), 1u);
}

TEST_F(FabricEndpointLocalTest, OwnerWriteRequiresMatchingBackingAndExtent) {
  auto owner = std::make_shared<VectorBackingOwner>();
  auto gate = std::make_shared<amdgpu::VmTransferEffectGate>();
  const std::array<std::byte, 4> payload{};
  EXPECT_FALSE(endpoint_->prepare_local_write(
      kOwner, FabricBacking(owner, kOwner, VectorBackingOwner::kBase, payload.size()), payload,
      gate, {}));
  EXPECT_FALSE(endpoint_->prepare_local_write(
      0, FabricBacking(owner, 0, VectorBackingOwner::kBase, payload.size()), payload, gate,
      admit_local_test_effect()));
  EXPECT_FALSE(endpoint_->prepare_local_write(
      kOwner, FabricBacking(owner, kRequester, VectorBackingOwner::kBase, payload.size()), payload,
      gate, admit_local_test_effect()));
  EXPECT_EQ(endpoint_
                ->prepare_local_write(
                    kOwner, FabricBacking(owner, kOwner, VectorBackingOwner::kBase, payload.size()),
                    std::span<const std::byte>(payload).first(3), gate, admit_local_test_effect())
                .outcome,
            amdgpu::VmAccessOutcome::Malformed);
  EXPECT_FALSE(endpoint_->prepare_write(
      kOwner, kOwner, FabricBacking(owner, kOwner, VectorBackingOwner::kBase, payload.size()),
      payload, gate));
  EXPECT_EQ(owner->writes(), 0u);
}

TEST_F(FabricEndpointLocalTest, RetryKeepsReservationUntilOwnerEffectFinishes) {
  auto owner = std::make_shared<VectorBackingOwner>();
  auto gate = std::make_shared<amdgpu::VmTransferEffectGate>();
  const std::array<std::byte, 4> payload{std::byte{0x56}, std::byte{0x67}, std::byte{0x78},
                                         std::byte{0x89}};
  int admissions = 0;
  int releases = 0;
  auto admit = [&](const FabricBacking &, bool write, amdgpu::MemoryIngressPermit &reservation) {
    EXPECT_TRUE(write);
    FabricEndpoint::EffectAuthority authority;
    if (++admissions == 1) {
      EXPECT_FALSE(reservation);
      reservation = amdgpu::MemoryIngressPermit([&] { ++releases; });
      authority.outcome = amdgpu::VmAccessOutcome::Unavailable;
    } else {
      EXPECT_EQ(admissions, 2);
      EXPECT_TRUE(reservation);
      EXPECT_EQ(releases, 0);
      authority.permit = amdgpu::MemoryIngressPermit([] {});
    }
    return authority;
  };
  auto write = endpoint_->prepare_local_write(
      kOwner, FabricBacking(owner, kOwner, VectorBackingOwner::kBase, payload.size()), payload,
      gate, admit);
  ASSERT_TRUE(write);
  write.operation->submit();
  ASSERT_TRUE(engine_.step());
  EXPECT_EQ(admissions, 1);
  EXPECT_EQ(releases, 0);
  EXPECT_EQ(owner->writes(), 0u);
  EXPECT_EQ(write.operation->poll({}), amdgpu::VmAccessOutcome::Unavailable);

  ASSERT_TRUE(engine_.step());
  EXPECT_EQ(admissions, 2);
  EXPECT_EQ(releases, 1);
  EXPECT_EQ(owner->writes(), 1u);
  EXPECT_EQ(write.operation->poll({}), amdgpu::VmAccessOutcome::Complete);
  EXPECT_EQ(endpoint_->diagnostics().effects, 1u);
  EXPECT_EQ(endpoint_->diagnostics().request_messages, 0u);
}

TEST_F(FabricEndpointLocalTest, QueuedCancellationAndShutdownNeverWriteBacking) {
  auto owner = std::make_shared<VectorBackingOwner>();
  auto gate = std::make_shared<amdgpu::VmTransferEffectGate>();
  const std::array<std::byte, 4> payload{std::byte{0x92}, std::byte{0x92}, std::byte{0x92},
                                         std::byte{0x92}};
  auto cancelled = endpoint_->prepare_local_write(
      kOwner, FabricBacking(owner, kOwner, VectorBackingOwner::kBase, payload.size()), payload,
      gate, admit_local_test_effect());
  ASSERT_TRUE(cancelled);
  cancelled.operation->submit();
  cancelled.operation->cancel();
  ASSERT_TRUE(engine_.step());
  EXPECT_EQ(cancelled.operation->poll({}), amdgpu::VmAccessOutcome::Revoked);
  EXPECT_EQ(owner->writes(), 0u);

  auto shutdown = endpoint_->prepare_local_write(
      kOwner, FabricBacking(owner, kOwner, VectorBackingOwner::kBase + 4, payload.size()), payload,
      gate, admit_local_test_effect());
  ASSERT_TRUE(shutdown);
  shutdown.operation->submit();
  endpoint_->shutdown();
  EXPECT_EQ(shutdown.operation->poll({}), amdgpu::VmAccessOutcome::Revoked);
  EXPECT_EQ(owner->writes(), 0u);
  EXPECT_EQ(endpoint_->diagnostics().effects, 0u);
}

TEST_F(FabricEndpointLocalTest, ShutdownWaitsForEnteredAdmissionAndReservation) {
  using namespace std::chrono_literals;
  auto owner = std::make_shared<VectorBackingOwner>();
  auto gate = std::make_shared<amdgpu::VmTransferEffectGate>();
  const std::array<std::byte, 4> payload{std::byte{0xa1}, std::byte{0xa1}, std::byte{0xa1},
                                         std::byte{0xa1}};
  std::atomic<int> releases{0};
  std::promise<void> entered;
  auto entered_future = entered.get_future();
  std::promise<void> continue_admission;
  auto continue_future = continue_admission.get_future().share();
  auto admit = [&](const FabricBacking &, bool, amdgpu::MemoryIngressPermit &reservation) {
    reservation = amdgpu::MemoryIngressPermit([&] { releases.fetch_add(1); });
    entered.set_value();
    continue_future.wait();
    FabricEndpoint::EffectAuthority authority;
    authority.outcome = amdgpu::VmAccessOutcome::Unavailable;
    return authority;
  };
  auto write = endpoint_->prepare_local_write(
      kOwner, FabricBacking(owner, kOwner, VectorBackingOwner::kBase, payload.size()), payload,
      gate, admit);
  ASSERT_TRUE(write);
  write.operation->submit();
  auto step = std::async(std::launch::async, [&] { return engine_.step(); });
  if (entered_future.wait_for(5s) != std::future_status::ready) {
    continue_admission.set_value();
    step.wait();
    FAIL() << "owner admission did not acquire the request reservation";
  }

  write.operation->cancel();
  EXPECT_EQ(releases.load(), 0);
  auto shutdown = std::async(std::launch::async, [&] { endpoint_->shutdown(); });
  const auto deadline = std::chrono::steady_clock::now() + 5s;
  bool closed = false;
  while (std::chrono::steady_clock::now() < deadline) {
    auto probe = endpoint_->prepare_local_write(
        kOwner, FabricBacking(owner, kOwner, VectorBackingOwner::kBase + 4, payload.size()),
        payload, gate, admit_local_test_effect());
    if (probe.outcome == amdgpu::VmAccessOutcome::Revoked) {
      closed = true;
      break;
    }
    std::this_thread::yield();
  }
  EXPECT_TRUE(closed);
  EXPECT_EQ(shutdown.wait_for(0ms), std::future_status::timeout);
  EXPECT_EQ(releases.load(), 0);
  continue_admission.set_value();
  EXPECT_TRUE(step.get());
  shutdown.get();
  EXPECT_EQ(releases.load(), 1);
  EXPECT_EQ(write.operation->poll({}), amdgpu::VmAccessOutcome::Revoked);
  EXPECT_EQ(owner->writes(), 0u);
  EXPECT_EQ(endpoint_->diagnostics().effects, 0u);
}

TEST_F(FabricEndpointTest, SwitchFanoutUsesDistinctBidirectionalEdgePorts) {
  // e0-sw, sw-e1, and sw-e2 each need their own request and response link.
  EXPECT_EQ(engine_.topology().links().size(), 6u);
  EXPECT_EQ(endpoint_->routes().size(), 2u);
  EXPECT_TRUE(endpoint_->can_route(kRequester, kOwner));
  EXPECT_TRUE(endpoint_->can_route(kOwner, kRequester));
  EXPECT_FALSE(endpoint_->can_route(kOwner, kThird));
}

TEST_F(FabricEndpointTest, OwnerEventSendsOneRequestAndResponseFromItsOwnThread) {
  ASSERT_TRUE(endpoint_->enable_effect_trace(4));
  EXPECT_FALSE(endpoint_->enable_effect_trace(1));
  auto gate = std::make_shared<amdgpu::VmTransferEffectGate>();
  std::array<std::byte, 4> payload{std::byte{0x13}, std::byte{0x24}, std::byte{0x35},
                                   std::byte{0x46}};
  auto write =
      endpoint_->prepare_write(kRequester, kOwner, fabric_lease(16, payload.size()), payload, gate,
                               {}, {}, {.map_id = 17, .gpu_va = 0x1200, .iova = 0x3400});
  ASSERT_TRUE(write);
  std::thread::id submitter;
  std::thread worker([&] {
    submitter = std::this_thread::get_id();
    write.operation->submit();
  });
  worker.join();
  EXPECT_EQ(write.operation->poll({}), amdgpu::VmAccessOutcome::Unavailable);
  ASSERT_TRUE(engine_.step());
  EXPECT_EQ(write.operation->poll({}), amdgpu::VmAccessOutcome::Complete);
  EXPECT_EQ(write.operation->poll({}), amdgpu::VmAccessOutcome::Complete);
  auto written = lease(16, 4);
  ASSERT_TRUE(written);
  EXPECT_EQ(std::vector<std::byte>(written.bytes().begin(), written.bytes().end()),
            std::vector<std::byte>(payload.begin(), payload.end()));

  auto stats = endpoint_->diagnostics();
  EXPECT_EQ(stats.submissions, 1u);
  EXPECT_EQ(stats.request_messages, 2u);
  EXPECT_EQ(stats.response_messages, 2u);
  EXPECT_EQ(stats.effects, 1u);
  ASSERT_EQ(stats.directed_effects.size(), 4u);
  EXPECT_EQ(stats.directed_effects[0].requester_gpu_id, kRequester);
  EXPECT_EQ(stats.directed_effects[0].owner_gpu_id, kOwner);
  EXPECT_EQ(stats.directed_effects[0].reads, 0u);
  EXPECT_EQ(stats.directed_effects[0].read_bytes, 0u);
  EXPECT_EQ(stats.directed_effects[0].writes, 1u);
  EXPECT_EQ(stats.directed_effects[0].write_bytes, payload.size());
  EXPECT_EQ(stats.directed_effects[1].requester_gpu_id, kOwner);
  EXPECT_EQ(stats.directed_effects[1].owner_gpu_id, kRequester);
  EXPECT_EQ(stats.directed_effects[1].reads + stats.directed_effects[1].writes, 0u);
  EXPECT_EQ(stats.owner_thread, std::this_thread::get_id());
  EXPECT_EQ(stats.handler_thread, std::this_thread::get_id());
  EXPECT_NE(stats.owner_thread, submitter);

  auto read = endpoint_->prepare_read(kRequester, kOwner, fabric_lease(16, payload.size()), gate);
  ASSERT_TRUE(read);
  read.operation->submit();
  ASSERT_TRUE(engine_.step());
  std::fill(written.bytes().begin(), written.bytes().end(), std::byte{0x77});
  std::array<std::byte, 4> result{};
  EXPECT_EQ(read.operation->poll(result), amdgpu::VmAccessOutcome::Complete);
  EXPECT_EQ(result, payload);
  EXPECT_EQ(endpoint_->diagnostics().effects, 2u);
  stats = endpoint_->diagnostics();
  EXPECT_EQ(stats.directed_effects[0].reads, 1u);
  EXPECT_EQ(stats.directed_effects[0].read_bytes, payload.size());
  EXPECT_EQ(stats.directed_effects[0].writes, 1u);
  EXPECT_EQ(stats.directed_effects[0].write_bytes, payload.size());
  EXPECT_EQ(stats.directed_effects[1].reads + stats.directed_effects[1].writes, 0u);

  const std::array<std::byte, 8> larger_payload{std::byte{0x11}, std::byte{0x22}, std::byte{0x33},
                                                std::byte{0x44}, std::byte{0x55}, std::byte{0x66},
                                                std::byte{0x77}, std::byte{0x88}};
  auto larger = endpoint_->prepare_write(
      kRequester, kOwner, fabric_lease(32, larger_payload.size()), larger_payload, gate);
  ASSERT_TRUE(larger);
  larger.operation->submit();
  ASSERT_TRUE(engine_.step());
  EXPECT_EQ(larger.operation->poll({}), amdgpu::VmAccessOutcome::Complete);
  stats = endpoint_->diagnostics();
  EXPECT_EQ(stats.directed_effects[0].writes, 2u);
  EXPECT_EQ(stats.directed_effects[0].write_bytes, payload.size() + larger_payload.size());

  auto reverse_owner = std::make_shared<VectorBackingOwner>();
  const std::array<std::byte, 8> reverse_payload{std::byte{0x51}, std::byte{0x52}, std::byte{0x53},
                                                 std::byte{0x54}, std::byte{0x55}, std::byte{0x56},
                                                 std::byte{0x57}, std::byte{0x58}};
  auto reverse = endpoint_->prepare_write(
      kOwner, kRequester,
      FabricBacking(reverse_owner, kRequester, VectorBackingOwner::kBase, reverse_payload.size()),
      reverse_payload, gate);
  ASSERT_TRUE(reverse);
  reverse.operation->submit();
  ASSERT_TRUE(engine_.step());
  EXPECT_EQ(reverse.operation->poll({}), amdgpu::VmAccessOutcome::Complete);
  stats = endpoint_->diagnostics();
  EXPECT_EQ(stats.directed_effects[1].reads, 0u);
  EXPECT_EQ(stats.directed_effects[1].read_bytes, 0u);
  EXPECT_EQ(stats.directed_effects[1].writes, 1u);
  EXPECT_EQ(stats.directed_effects[1].write_bytes, reverse_payload.size());

  ASSERT_EQ(stats.committed_effects.size(), 4u);
  EXPECT_EQ(stats.effect_trace_dropped, 0u);
  const auto &first = stats.committed_effects[0];
  EXPECT_EQ(first.requester_gpu_id, kRequester);
  EXPECT_EQ(first.owner_gpu_id, kOwner);
  EXPECT_TRUE(first.write);
  EXPECT_NE(first.backing_id, 0u);
  EXPECT_EQ(first.backing_offset, 16u);
  EXPECT_EQ(first.bytes, payload.size());
  EXPECT_EQ(first.origin.map_id, 17u);
  EXPECT_EQ(first.origin.gpu_va, 0x1200u);
  EXPECT_EQ(first.origin.iova, 0x3400u);
  EXPECT_NE(first.request_id, stats.committed_effects[1].request_id);
  EXPECT_FALSE(stats.committed_effects[1].write);
  EXPECT_EQ(stats.committed_effects[1].origin.map_id, 0u);
  EXPECT_EQ(stats.committed_effects[1].backing_id, first.backing_id);
  EXPECT_EQ(stats.committed_effects[1].backing_offset, 16u);
  EXPECT_EQ(stats.committed_effects[2].backing_offset, 32u);
  EXPECT_EQ(stats.committed_effects[2].bytes, larger_payload.size());
  EXPECT_EQ(stats.committed_effects[3].requester_gpu_id, kOwner);
  EXPECT_EQ(stats.committed_effects[3].owner_gpu_id, kRequester);
  EXPECT_EQ(stats.committed_effects[3].backing_id, 0u);
  EXPECT_EQ(stats.committed_effects[3].backing_offset, VectorBackingOwner::kBase);

  auto overflow =
      endpoint_->prepare_write(kRequester, kOwner, fabric_lease(48, payload.size()), payload, gate);
  ASSERT_TRUE(overflow);
  overflow.operation->submit();
  ASSERT_TRUE(engine_.step());
  EXPECT_EQ(overflow.operation->poll({}), amdgpu::VmAccessOutcome::Complete);
  stats = endpoint_->diagnostics();
  EXPECT_EQ(stats.effects, 5u);
  EXPECT_EQ(stats.committed_effects.size(), 4u);
  EXPECT_EQ(stats.effect_trace_dropped, 1u);
}

TEST_F(FabricEndpointTest, NonKfdBackingRetainsOwnerAndBoundsReadWrite) {
  auto owner = std::make_shared<VectorBackingOwner>();
  std::weak_ptr<VectorBackingOwner> lifetime = owner;
  constexpr uint64_t kOffset = VectorBackingOwner::kBase + 5;
  FabricBacking backing(owner, kOwner, kOffset, 4);
  ASSERT_TRUE(backing);
  EXPECT_EQ(backing.owner_gpu_id(), kOwner);
  EXPECT_EQ(backing.offset(), kOffset);
  EXPECT_EQ(backing.extent(), 4u);
  EXPECT_FALSE(FabricBacking(owner, kOwner, VectorBackingOwner::kBase - 1, 4));
  EXPECT_FALSE(FabricBacking(owner, kOwner, VectorBackingOwner::kBase + 14, 4));

  const std::array<std::byte, 3> too_short{};
  EXPECT_FALSE(backing.write(too_short));
  std::array<std::byte, 5> too_long{};
  EXPECT_FALSE(backing.read(too_long));
  EXPECT_EQ(owner->reads(), 0u);
  EXPECT_EQ(owner->writes(), 0u);

  auto gate = std::make_shared<amdgpu::VmTransferEffectGate>();
  auto wrong_owner = endpoint_->prepare_read(kRequester, kThird, backing, gate);
  EXPECT_FALSE(wrong_owner);
  EXPECT_EQ(wrong_owner.outcome, amdgpu::VmAccessOutcome::Faulted);

  const std::array<std::byte, 4> payload{std::byte{0x11}, std::byte{0x22}, std::byte{0x33},
                                         std::byte{0x44}};
  auto write = endpoint_->prepare_write(kRequester, kOwner, backing, payload, gate);
  ASSERT_TRUE(write);
  backing = {};
  owner.reset();
  ASSERT_FALSE(lifetime.expired());

  write.operation->submit();
  ASSERT_TRUE(engine_.step());
  EXPECT_EQ(write.operation->poll({}), amdgpu::VmAccessOutcome::Complete);
  auto retained = lifetime.lock();
  ASSERT_TRUE(retained);
  EXPECT_EQ(retained->writes(), 1u);
  EXPECT_EQ(retained->last_offset(), kOffset);
  const auto &written = retained->bytes_for_testing();
  EXPECT_TRUE(std::equal(payload.begin(), payload.end(), written.begin() + 5));
  EXPECT_EQ(written[4], std::byte{0x7b});
  EXPECT_EQ(written[9], std::byte{0x7b});

  auto read = endpoint_->prepare_read(
      kRequester, kOwner, FabricBacking(retained, kOwner, kOffset, payload.size()), gate);
  ASSERT_TRUE(read);
  retained.reset();
  write.operation.reset();
  ASSERT_FALSE(lifetime.expired());
  read.operation->submit();
  ASSERT_TRUE(engine_.step());
  std::array<std::byte, 4> result{};
  EXPECT_EQ(read.operation->poll(result), amdgpu::VmAccessOutcome::Complete);
  EXPECT_EQ(result, payload);
  retained = lifetime.lock();
  ASSERT_TRUE(retained);
  EXPECT_EQ(retained->reads(), 1u);
  EXPECT_EQ(retained->last_offset(), kOffset);
  retained.reset();
  read.operation.reset();
  EXPECT_TRUE(lifetime.expired());
  EXPECT_EQ(endpoint_->diagnostics().effects, 2u);
}

TEST_F(FabricEndpointTest, ManualEndpointLeavesCacheableBackingUnmanaged) {
  auto memory = std::make_shared<amdgpu::GpuMemory>("manual_peer_backing");
  auto owner = std::make_shared<CacheableBackingOwner>(memory);
  auto domain = std::make_shared<amdgpu::DeviceCacheCoherence>();
  amdgpu::GpuVm gpu_vm;
  amdgpu::L2Cache l2("manual_owner_l2", domain);
  l2.set_backing_memory(memory.get());
  l2.set_legacy_maintenance_memory(memory.get());
  l2.set_legacy_maintenance_vm(&gpu_vm);
  memory->write8(CacheableBackingOwner::kBase, 0x11);
  std::array<uint8_t, amdgpu::L2Cache::LINE_SIZE> dirty_line{};
  dirty_line[0] = 0x5a;
  ASSERT_EQ(l2.writeback_line(CacheableBackingOwner::kBase, dirty_line.data(), amdgpu::Mtype::RW),
            amdgpu::VmAccessOutcome::Complete);
  const uint64_t epoch = domain->current_epoch();

  auto read = endpoint_->prepare_read(kRequester, kOwner,
                                      FabricBacking(owner, kOwner, CacheableBackingOwner::kBase, 1),
                                      std::make_shared<amdgpu::VmTransferEffectGate>());
  ASSERT_TRUE(read);
  read.operation->submit();
  ASSERT_TRUE(engine_.step());
  std::array<std::byte, 1> result{};
  EXPECT_EQ(read.operation->poll(result), amdgpu::VmAccessOutcome::Complete);
  EXPECT_EQ(result[0], std::byte{0x11});
  EXPECT_EQ(domain->current_epoch(), epoch);
  EXPECT_EQ(memory->read8(CacheableBackingOwner::kBase), 0x11);
}

TEST_F(FabricEndpointCoherenceTest, ReadPublishesDirtyCacheableOwnerBeforeBackingEffect) {
  constexpr uint64_t kAddress = CacheableBackingOwner::kBase;
  constexpr uint64_t kRequesterAddress = kAddress + amdgpu::L2Cache::LINE_SIZE;
  memory_->write8(kAddress, 0x11);
  std::array<uint8_t, amdgpu::L2Cache::LINE_SIZE> line{};
  line[0] = 0x5a;
  ASSERT_EQ(owner_l2_->writeback_line(kAddress, line.data(), amdgpu::Mtype::RW),
            amdgpu::VmAccessOutcome::Complete);
  line[0] = 0x6b;
  ASSERT_EQ(requester_l2_->writeback_line(kRequesterAddress, line.data(), amdgpu::Mtype::RW),
            amdgpu::VmAccessOutcome::Complete);
  ASSERT_EQ(memory_->read8(kAddress), 0x11);
  ASSERT_EQ(memory_->read8(kRequesterAddress), 0);
  const uint64_t requester_epoch = requester_domain_->current_epoch();
  const uint64_t owner_epoch = owner_domain_->current_epoch();

  auto read = endpoint_->prepare_read(kRequester, kOwner, backing(kAddress, 1),
                                      std::make_shared<amdgpu::VmTransferEffectGate>());
  ASSERT_TRUE(read);
  read.operation->submit();
  step_to_target();
  EXPECT_EQ(read.operation->poll({}), amdgpu::VmAccessOutcome::Unavailable);
  EXPECT_EQ(memory_->read8(kAddress), 0x5a);
  EXPECT_EQ(memory_->read8(kRequesterAddress), 0x6b);
  EXPECT_GT(requester_domain_->current_epoch(), requester_epoch);
  EXPECT_GT(owner_domain_->current_epoch(), owner_epoch);
  EXPECT_EQ(owner_->reads(), 1u);
  ASSERT_TRUE(engine_.step());
  ASSERT_TRUE(engine_.step());
  std::array<std::byte, 1> result{};
  EXPECT_EQ(read.operation->poll(result), amdgpu::VmAccessOutcome::Complete);
  EXPECT_EQ(result[0], std::byte{0x5a});
}

TEST_F(FabricEndpointCoherenceTest, WriteInvalidatesBothLocalCacheableViewsBeforeReply) {
  constexpr uint64_t kAddress = CacheableBackingOwner::kBase;
  memory_->write8(kAddress, 0x11);
  memory_->write8(kAddress + 1, 0x44);
  for (amdgpu::L2Cache *l2 : {requester_l2_.get(), owner_l2_.get()}) {
    std::array<uint8_t, 2> cached{};
    ASSERT_EQ(l2->read(kAddress, cached.data(), cached.size(), amdgpu::Mtype::RW),
              amdgpu::VmAccessOutcome::Complete);
    ASSERT_EQ(cached, (std::array<uint8_t, 2>{0x11, 0x44}));
  }
  std::array<uint8_t, amdgpu::L2Cache::LINE_SIZE> dirty_line{};
  dirty_line[0] = 0x22;
  dirty_line[1] = 0x66;
  ASSERT_EQ(owner_l2_->writeback_line(kAddress, dirty_line.data(), 0, 2, amdgpu::Mtype::RW),
            amdgpu::VmAccessOutcome::Complete);
  ASSERT_EQ(memory_->read8(kAddress), 0x11);
  ASSERT_EQ(memory_->read8(kAddress + 1), 0x44);
  const uint64_t requester_epoch = requester_domain_->current_epoch();
  const uint64_t owner_epoch = owner_domain_->current_epoch();
  const std::array<std::byte, 1> payload{std::byte{0x33}};
  auto write = endpoint_->prepare_write(kRequester, kOwner, backing(kAddress, payload.size()),
                                        payload, std::make_shared<amdgpu::VmTransferEffectGate>());
  ASSERT_TRUE(write);
  write.operation->submit();
  step_to_target();

  EXPECT_EQ(write.operation->poll({}), amdgpu::VmAccessOutcome::Unavailable);
  EXPECT_EQ(owner_->writes(), 1u);
  EXPECT_EQ(memory_->read8(kAddress), 0x33);
  EXPECT_EQ(memory_->read8(kAddress + 1), 0x66);
  EXPECT_GT(requester_domain_->current_epoch(), requester_epoch);
  EXPECT_GT(owner_domain_->current_epoch(), owner_epoch);
  for (amdgpu::L2Cache *l2 : {requester_l2_.get(), owner_l2_.get()}) {
    std::array<uint8_t, 2> refreshed{};
    ASSERT_EQ(l2->read(kAddress, refreshed.data(), refreshed.size(), amdgpu::Mtype::RW),
              amdgpu::VmAccessOutcome::Complete);
    EXPECT_EQ(refreshed, (std::array<uint8_t, 2>{0x33, 0x66}));
  }
  ASSERT_TRUE(engine_.step());
  ASSERT_TRUE(engine_.step());
  EXPECT_EQ(write.operation->poll({}), amdgpu::VmAccessOutcome::Complete);
  EXPECT_EQ(owner_->writes(), 1u);
}

TEST_F(FabricEndpointCoherenceTest, HostMappedHandshakeCannotHideMissingDataInvalidation) {
  constexpr uint64_t kAddress = CacheableBackingOwner::kBase;
  constexpr uint64_t kReady = HostMappedControlMemory::kBase;
  constexpr uint64_t kDone = kReady + sizeof(uint32_t);
  memory_->write8(kAddress, 0x11);
  uint8_t owner_value = 0;
  ASSERT_EQ(owner_l2_->read(kAddress, &owner_value, 1, amdgpu::Mtype::RW),
            amdgpu::VmAccessOutcome::Complete);
  ASSERT_EQ(owner_value, 0x11);

  auto control = std::make_shared<HostMappedControlMemory>();
  const auto handle = gpu_vm_.register_translated(7, control, control);
  const auto access = gpu_vm_.snapshot(handle);
  ASSERT_TRUE(access);
  const uint64_t owner_epoch = owner_domain_->current_epoch();
  const uint64_t requester_epoch = requester_domain_->current_epoch();
  const uint64_t effects = endpoint_->diagnostics().effects;
  const uint32_t flag = 1;
  uint32_t observed = 0;
  ASSERT_EQ(access->write(kReady, std::as_bytes(std::span(&flag, 1))),
            amdgpu::VmAccessOutcome::Complete);
  ASSERT_EQ(access->read(kReady, std::as_writable_bytes(std::span(&observed, 1))),
            amdgpu::VmAccessOutcome::Complete);
  EXPECT_EQ(observed, flag);
  ASSERT_EQ(access->write(kDone, std::as_bytes(std::span(&flag, 1))),
            amdgpu::VmAccessOutcome::Complete);
  observed = 0;
  ASSERT_EQ(access->read(kDone, std::as_writable_bytes(std::span(&observed, 1))),
            amdgpu::VmAccessOutcome::Complete);
  EXPECT_EQ(observed, flag);
  EXPECT_EQ(endpoint_->diagnostics().effects, effects);
  EXPECT_EQ(owner_domain_->current_epoch(), owner_epoch);
  EXPECT_EQ(requester_domain_->current_epoch(), requester_epoch);

  // A temporary backing change proves the old owner line is still resident
  // after both mapped flag accesses. Restore the seed, then suppress only the
  // data write's invalidation to show the positive probe would fail.
  memory_->write8(kAddress, 0x22);
  owner_value = 0;
  ASSERT_EQ(owner_l2_->read(kAddress, &owner_value, 1, amdgpu::Mtype::RW),
            amdgpu::VmAccessOutcome::Complete);
  EXPECT_EQ(owner_value, 0x11);
  memory_->write8(kAddress, 0x11);
  memory_->write8(kAddress, 0x33);
  owner_value = 0;
  ASSERT_EQ(owner_l2_->read(kAddress, &owner_value, 1, amdgpu::Mtype::RW),
            amdgpu::VmAccessOutcome::Complete);
  EXPECT_EQ(owner_value, 0x11);
  EXPECT_EQ(memory_->read8(kAddress), 0x33);
  EXPECT_EQ(endpoint_->diagnostics().effects, effects);
}

TEST_F(FabricEndpointCoherenceTest, PreflushFailureIsTerminalWithoutBackingEffect) {
  constexpr uint64_t kAddress = CacheableBackingOwner::kBase;
  constexpr uint64_t kPreviouslyDirtyAddress = kAddress + amdgpu::L2Cache::LINE_SIZE;
  memory_->write8(kAddress, 0x11);
  std::array<uint8_t, amdgpu::L2Cache::LINE_SIZE> dirty_line{};
  dirty_line[0] = 0x77;
  ASSERT_EQ(
      requester_l2_->writeback_line(kPreviouslyDirtyAddress, dirty_line.data(), amdgpu::Mtype::RW),
      amdgpu::VmAccessOutcome::Complete);
  ASSERT_EQ(memory_->read8(kPreviouslyDirtyAddress), 0);
  owner_l2_->set_legacy_maintenance_memory(nullptr);
  const uint64_t requester_epoch = requester_domain_->current_epoch();
  const uint64_t owner_epoch = owner_domain_->current_epoch();
  const std::array<std::byte, 1> payload{std::byte{0x44}};
  auto write = endpoint_->prepare_write(kRequester, kOwner, backing(kAddress, payload.size()),
                                        payload, std::make_shared<amdgpu::VmTransferEffectGate>());
  ASSERT_TRUE(write);
  write.operation->submit();
  step_to_target();

  EXPECT_EQ(write.operation->reconcile_revocation({}), amdgpu::VmAccessOutcome::Faulted);
  EXPECT_EQ(write.operation->poll({}), amdgpu::VmAccessOutcome::Unavailable);
  EXPECT_EQ(owner_->writes(), 0u);
  EXPECT_EQ(memory_->read8(kAddress), 0x11);
  EXPECT_EQ(memory_->read8(kPreviouslyDirtyAddress), 0u);
  EXPECT_EQ(requester_domain_->current_epoch(), requester_epoch);
  EXPECT_EQ(owner_domain_->current_epoch(), owner_epoch);
  ASSERT_TRUE(engine_.step());
  ASSERT_TRUE(engine_.step());
  EXPECT_EQ(write.operation->poll({}), amdgpu::VmAccessOutcome::Faulted);
  EXPECT_EQ(write.operation->poll({}), amdgpu::VmAccessOutcome::Faulted);
  EXPECT_EQ(endpoint_->diagnostics().effects, 0u);

  owner_l2_->set_legacy_maintenance_memory(memory_.get());
  auto read = endpoint_->prepare_read(kRequester, kOwner, backing(kAddress, 1),
                                      std::make_shared<amdgpu::VmTransferEffectGate>());
  ASSERT_TRUE(read);
  read.operation->submit();
  step_to_target();
  ASSERT_TRUE(engine_.step());
  ASSERT_TRUE(engine_.step());
  std::array<std::byte, 1> result{};
  EXPECT_EQ(read.operation->poll(result), amdgpu::VmAccessOutcome::Complete);
  EXPECT_EQ(result[0], std::byte{0x11});
  EXPECT_EQ(owner_->writes(), 0u);
  EXPECT_EQ(memory_->read8(kPreviouslyDirtyAddress), 0x77u);
}

TEST_F(FabricEndpointCoherenceTest, CancellationAfterPreflushReleasesEveryLease) {
  using namespace std::chrono_literals;
  constexpr uint64_t kAddress = CacheableBackingOwner::kBase;
  memory_->write8(kAddress, 0x11);
  const uint64_t requester_epoch = requester_domain_->current_epoch();
  const uint64_t owner_epoch = owner_domain_->current_epoch();
  const std::array<std::byte, 1> payload{std::byte{0x55}};
  auto write = endpoint_->prepare_write(kRequester, kOwner, backing(kAddress, payload.size()),
                                        payload, std::make_shared<amdgpu::VmTransferEffectGate>());
  ASSERT_TRUE(write);

  std::promise<void> entered;
  std::future<void> entered_future = entered.get_future();
  std::promise<void> release;
  std::shared_future<void> release_future = release.get_future().share();
  std::thread::id probe_thread;
  endpoint_->set_effect_probe_for_testing([&] {
    probe_thread = std::this_thread::get_id();
    entered.set_value();
    release_future.wait();
  });
  write.operation->submit();
  ASSERT_TRUE(engine_.step());
  ASSERT_TRUE(engine_.step());
  std::future<bool> target = std::async(std::launch::async, [&] { return engine_.step(); });
  if (entered_future.wait_for(5s) != std::future_status::ready) {
    release.set_value();
    target.wait();
    FAIL() << "target effect did not reach the probe";
  }

  write.operation->cancel();
  EXPECT_EQ(write.operation->reconcile_revocation({}), amdgpu::VmAccessOutcome::Revoked);
  release.set_value();
  EXPECT_TRUE(target.get());
  EXPECT_EQ(endpoint_->diagnostics().handler_thread, probe_thread);
  EXPECT_GT(requester_domain_->current_epoch(), requester_epoch);
  EXPECT_GT(owner_domain_->current_epoch(), owner_epoch);
  EXPECT_EQ(owner_->writes(), 0u);
  EXPECT_EQ(memory_->read8(kAddress), 0x11);
  ASSERT_TRUE(engine_.step());
  ASSERT_TRUE(engine_.step());
  EXPECT_EQ(write.operation->poll({}), amdgpu::VmAccessOutcome::Revoked);
  EXPECT_EQ(endpoint_->diagnostics().effects, 0u);
}

TEST_F(FabricEndpointTest, RejectedNonKfdWriteReportsFaultWithoutBackingEffect) {
  ASSERT_TRUE(endpoint_->enable_effect_trace(2));
  auto owner = std::make_shared<VectorBackingOwner>();
  owner->reject_writes();
  const auto original = owner->bytes_for_testing();
  const std::array<std::byte, 4> payload{std::byte{0x11}, std::byte{0x22}, std::byte{0x33},
                                         std::byte{0x44}};
  auto gate = std::make_shared<amdgpu::VmTransferEffectGate>();
  auto write = endpoint_->prepare_write(
      kRequester, kOwner, FabricBacking(owner, kOwner, VectorBackingOwner::kBase + 5, 4), payload,
      gate);
  ASSERT_TRUE(write);
  write.operation->submit();
  ASSERT_TRUE(engine_.step());

  EXPECT_EQ(write.operation->poll({}), amdgpu::VmAccessOutcome::Faulted);
  EXPECT_EQ(write.operation->reconcile_revocation({}), amdgpu::VmAccessOutcome::Faulted);
  EXPECT_EQ(owner->bytes_for_testing(), original);
  EXPECT_EQ(owner->writes(), 0u);
  EXPECT_EQ(endpoint_->diagnostics().effects, 0u);
  EXPECT_EQ(endpoint_->diagnostics().directed_effects[0].write_bytes, 0u);
  EXPECT_TRUE(endpoint_->diagnostics().committed_effects.empty());
  EXPECT_EQ(endpoint_->diagnostics().effect_trace_dropped, 0u);
}

TEST_F(FabricEndpointTest, MissingAndAmbiguousRoutesFaultBeforeSubmission) {
  auto gate = std::make_shared<amdgpu::VmTransferEffectGate>();
  auto missing = endpoint_->prepare_read(kThird, kOwner, fabric_lease(0, 4), gate);
  EXPECT_FALSE(missing);
  EXPECT_EQ(missing.outcome, amdgpu::VmAccessOutcome::Faulted);
  EXPECT_FALSE(endpoint_->can_route(kThird, kOwner));

  std::vector<config::SystemPeerRoute> ambiguous = fanout_routes();
  ambiguous.push_back({"g0", "g1", kRequester, kOwner, {"e0", "alternate", "e1"}});
  simdojo::SimulationEngine engine({.num_threads = 1});
  auto root = std::make_unique<simdojo::CompositeComponent>("root");
  auto duplicate = std::make_unique<FabricEndpoint>(std::move(ambiguous));
  auto *endpoint = duplicate.get();
  root->add_child(std::move(duplicate));
  engine.topology().set_root(std::move(root));
  endpoint->wire(engine.topology(), simdojo::ExecMode::FUNCTIONAL);
  EXPECT_FALSE(endpoint->can_route(kRequester, kOwner));
  engine.create();
  auto denied = endpoint->prepare_read(kRequester, kOwner, fabric_lease(0, 4), gate);
  EXPECT_FALSE(denied);
  EXPECT_EQ(denied.outcome, amdgpu::VmAccessOutcome::Faulted);
  EXPECT_EQ(endpoint->diagnostics().submissions, 0u);
}

TEST_F(FabricEndpointTest, OverlongManualRouteFailsClosedBeforeWiring) {
  config::SystemPeerRoute route{"g0", "g1", kRequester, kOwner, {"e0"}};
  for (size_t i = 0; i < config::kMaxPeerRouteEndpoints - 1; ++i)
    route.endpoints.push_back("sw" + std::to_string(i));
  route.endpoints.push_back("e1");
  ASSERT_EQ(route.endpoints.size(), config::kMaxPeerRouteEndpoints + 1);

  simdojo::SimulationEngine engine({.num_threads = 1});
  auto root = std::make_unique<simdojo::CompositeComponent>("root");
  auto fabric = std::make_unique<FabricEndpoint>(std::vector{std::move(route)});
  auto *endpoint = fabric.get();
  root->add_child(std::move(fabric));
  engine.topology().set_root(std::move(root));
  endpoint->wire(engine.topology(), simdojo::ExecMode::FUNCTIONAL);
  engine.create();
  EXPECT_TRUE(engine.topology().links().empty());
  auto denied = endpoint->prepare_read(kRequester, kOwner, fabric_lease(0, 4),
                                       std::make_shared<amdgpu::VmTransferEffectGate>());
  EXPECT_FALSE(denied);
  EXPECT_EQ(denied.outcome, amdgpu::VmAccessOutcome::Faulted);
}

TEST_F(FabricEndpointTest, QueuedCancellationAndShutdownDrainWithoutEffect) {
  auto gate = std::make_shared<amdgpu::VmTransferEffectGate>();
  auto original = lease(24, 4);
  ASSERT_TRUE(original);
  std::fill(original.bytes().begin(), original.bytes().end(), std::byte{0x11});
  std::array<std::byte, 4> payload{std::byte{0x22}, std::byte{0x22}, std::byte{0x22},
                                   std::byte{0x22}};
  auto cancelled = endpoint_->prepare_write(kRequester, kOwner, fabric_lease(24, 4), payload, gate);
  ASSERT_TRUE(cancelled);
  cancelled.operation->submit();
  cancelled.operation->cancel();
  ASSERT_TRUE(engine_.step());
  EXPECT_EQ(cancelled.operation->poll({}), amdgpu::VmAccessOutcome::Revoked);
  EXPECT_EQ(endpoint_->diagnostics().effects, 0u);
  EXPECT_EQ(original.bytes()[0], std::byte{0x11});
  EXPECT_FALSE(endpoint_->enable_effect_trace(2));

  auto abandoned = endpoint_->prepare_write(kRequester, kOwner, fabric_lease(24, 4), payload, gate);
  ASSERT_TRUE(abandoned);
  abandoned.operation->submit();
  endpoint_->shutdown();
  EXPECT_EQ(abandoned.operation->poll({}), amdgpu::VmAccessOutcome::Revoked);
  EXPECT_EQ(original.bytes()[0], std::byte{0x11});
  EXPECT_EQ(endpoint_->diagnostics().effects, 0u);
}

TEST_F(FabricEndpointTest, ResetRejectsAnOldPreparedRequest) {
  auto gate = std::make_shared<amdgpu::VmTransferEffectGate>();
  auto old = endpoint_->prepare_read(kRequester, kOwner, fabric_lease(32, 4), gate);
  ASSERT_TRUE(old);
  engine_.primary_release();
  engine_.shutdown();
  engine_.create();
  engine_.register_as_primary();

  old.operation->submit();
  EXPECT_EQ(old.operation->poll({}), amdgpu::VmAccessOutcome::Revoked);

  auto fresh = endpoint_->prepare_read(kRequester, kOwner, fabric_lease(32, 4), gate);
  ASSERT_TRUE(fresh);
  fresh.operation->submit();
  ASSERT_TRUE(engine_.step());
  std::array<std::byte, 4> result{};
  EXPECT_EQ(fresh.operation->poll(result), amdgpu::VmAccessOutcome::Complete);
  EXPECT_EQ(endpoint_->diagnostics().effects, 1u);
}

TEST_F(FabricEndpointTest, ShutdownWaitsForAnEnteredTargetEffect) {
  using namespace std::chrono_literals;
  auto gate = std::make_shared<amdgpu::VmTransferEffectGate>();
  auto original = lease(40, 4);
  ASSERT_TRUE(original);
  std::fill(original.bytes().begin(), original.bytes().end(), std::byte{0x11});
  std::array<std::byte, 4> payload{std::byte{0x22}, std::byte{0x22}, std::byte{0x22},
                                   std::byte{0x22}};
  auto write = endpoint_->prepare_write(kRequester, kOwner, fabric_lease(40, 4), payload, gate);
  ASSERT_TRUE(write);

  std::promise<void> entered;
  auto entered_future = entered.get_future();
  std::promise<void> release;
  auto release_future = release.get_future().share();
  endpoint_->set_effect_probe_for_testing([&] {
    entered.set_value();
    release_future.wait();
  });
  write.operation->submit();
  auto owner = std::async(std::launch::async, [&] { return engine_.step(); });
  if (entered_future.wait_for(5s) != std::future_status::ready) {
    release.set_value();
    owner.wait();
    FAIL() << "target effect did not enter";
  }

  auto shutdown = std::async(std::launch::async, [&] { endpoint_->shutdown(); });
  bool closed_admission = false;
  const auto deadline = std::chrono::steady_clock::now() + 5s;
  while (std::chrono::steady_clock::now() < deadline) {
    auto probe = endpoint_->prepare_read(kRequester, kOwner, fabric_lease(40, 4), gate);
    if (probe.outcome == amdgpu::VmAccessOutcome::Revoked) {
      closed_admission = true;
      break;
    }
    std::this_thread::yield();
  }
  EXPECT_TRUE(closed_admission);
  EXPECT_EQ(shutdown.wait_for(0ms), std::future_status::timeout);
  release.set_value();
  shutdown.get();
  EXPECT_TRUE(owner.get());
  EXPECT_EQ(write.operation->poll({}), amdgpu::VmAccessOutcome::Complete);
  EXPECT_EQ(original.bytes()[0], std::byte{0x22});
  EXPECT_EQ(endpoint_->diagnostics().effects, 1u);
}

TEST_F(FabricEndpointTest, CancellationDuringTargetProbePreventsBackingEffect) {
  using namespace std::chrono_literals;
  auto gate = std::make_shared<amdgpu::VmTransferEffectGate>();
  auto backing = lease(44, 4);
  ASSERT_TRUE(backing);
  std::fill(backing.bytes().begin(), backing.bytes().end(), std::byte{0x11});
  const std::array<std::byte, 4> payload{std::byte{0x22}, std::byte{0x22}, std::byte{0x22},
                                         std::byte{0x22}};
  auto write = endpoint_->prepare_write(kRequester, kOwner, fabric_lease(44, 4), payload, gate);
  ASSERT_TRUE(write);

  std::promise<void> entered;
  auto entered_future = entered.get_future();
  std::promise<void> release;
  auto release_future = release.get_future().share();
  endpoint_->set_effect_probe_for_testing([&] {
    entered.set_value();
    release_future.wait();
  });
  write.operation->submit();
  auto owner = std::async(std::launch::async, [&] { return engine_.step(); });
  if (entered_future.wait_for(5s) != std::future_status::ready) {
    release.set_value();
    owner.wait();
    FAIL() << "target effect did not reach the probe";
  }

  write.operation->cancel();
  EXPECT_EQ(write.operation->reconcile_revocation({}), amdgpu::VmAccessOutcome::Revoked);
  EXPECT_EQ(write.operation->poll({}), amdgpu::VmAccessOutcome::Revoked);
  release.set_value();
  EXPECT_TRUE(owner.get());
  EXPECT_EQ(backing.bytes()[0], std::byte{0x11});
  EXPECT_EQ(endpoint_->diagnostics().effects, 0u);
}

TEST_F(FabricEndpointClockedTest, RequestAndResponseAccumulateEveryLinkLatency) {
  auto gate = std::make_shared<amdgpu::VmTransferEffectGate>();
  std::array<std::byte, 4> payload{std::byte{0x12}, std::byte{0x23}, std::byte{0x34},
                                   std::byte{0x45}};
  auto write = endpoint_->prepare_write(kRequester, kOwner, fabric_lease(64, 4), payload, gate);
  ASSERT_TRUE(write);
  write.operation->submit();

  ASSERT_TRUE(engine_.step());
  EXPECT_EQ(engine_.context(0).current_tick(), 0u);
  EXPECT_EQ(endpoint_->diagnostics().request_messages, 0u);
  ASSERT_TRUE(engine_.step());
  EXPECT_EQ(engine_.context(0).current_tick(), 1u);
  EXPECT_EQ(endpoint_->diagnostics().request_messages, 1u);
  EXPECT_EQ(endpoint_->diagnostics().effects, 0u);
  ASSERT_TRUE(engine_.step());
  EXPECT_EQ(engine_.context(0).current_tick(), 2u);
  EXPECT_EQ(endpoint_->diagnostics().request_messages, 2u);
  EXPECT_EQ(endpoint_->diagnostics().effects, 1u);
  EXPECT_EQ(write.operation->poll({}), amdgpu::VmAccessOutcome::Unavailable);
  ASSERT_TRUE(engine_.step());
  EXPECT_EQ(engine_.context(0).current_tick(), 3u);
  EXPECT_EQ(endpoint_->diagnostics().response_messages, 1u);
  EXPECT_EQ(write.operation->poll({}), amdgpu::VmAccessOutcome::Unavailable);
  ASSERT_TRUE(engine_.step());
  EXPECT_EQ(engine_.context(0).current_tick(), 4u);
  EXPECT_EQ(endpoint_->diagnostics().response_messages, 2u);
  EXPECT_EQ(write.operation->poll({}), amdgpu::VmAccessOutcome::Complete);
  EXPECT_EQ(endpoint_->diagnostics().effects, 1u);
}

TEST_F(FabricEndpointClockedTest, RequestReservationSurvivesAdmissionRetryUntilEffectReply) {
  auto gate = std::make_shared<amdgpu::VmTransferEffectGate>();
  const std::array<std::byte, 4> payload{std::byte{0x37}, std::byte{0x37}, std::byte{0x37},
                                         std::byte{0x37}};
  int admissions = 0;
  int releases = 0;
  auto admit = [&](const FabricBacking &, bool write, amdgpu::MemoryIngressPermit &reservation) {
    EXPECT_TRUE(write);
    ++admissions;
    FabricEndpoint::EffectAuthority authority;
    if (admissions == 1) {
      EXPECT_FALSE(reservation);
      reservation = amdgpu::MemoryIngressPermit([&] { ++releases; });
      authority.outcome = amdgpu::VmAccessOutcome::Unavailable;
    } else {
      EXPECT_EQ(admissions, 2);
      EXPECT_TRUE(reservation);
      EXPECT_EQ(releases, 0);
      authority.permit = amdgpu::MemoryIngressPermit([] {});
    }
    return authority;
  };
  auto write =
      endpoint_->prepare_write(kRequester, kOwner, fabric_lease(132, 4), payload, gate, admit);
  ASSERT_TRUE(write);
  EXPECT_EQ(admissions, 0);
  write.operation->submit();
  ASSERT_TRUE(engine_.step());
  ASSERT_TRUE(engine_.step());
  ASSERT_TRUE(engine_.step());
  EXPECT_EQ(admissions, 1);
  EXPECT_EQ(releases, 0);
  EXPECT_EQ(endpoint_->diagnostics().effects, 0u);

  for (int step = 0; step < 8 && admissions < 2; ++step)
    ASSERT_TRUE(engine_.step());
  ASSERT_EQ(admissions, 2);
  EXPECT_EQ(releases, 1);
  EXPECT_EQ(endpoint_->diagnostics().effects, 1u);
  EXPECT_EQ(endpoint_->diagnostics().directed_effects[0].write_bytes, payload.size());
  EXPECT_EQ(write.operation->poll({}), amdgpu::VmAccessOutcome::Unavailable);
  for (int step = 0; step < 8 && write.operation->poll({}) == amdgpu::VmAccessOutcome::Unavailable;
       ++step)
    ASSERT_TRUE(engine_.step());
  EXPECT_EQ(write.operation->poll({}), amdgpu::VmAccessOutcome::Complete);
  EXPECT_EQ(releases, 1);
  EXPECT_TRUE(std::equal(payload.begin(), payload.end(), lease(132, 4).bytes().begin()));
}

TEST_F(FabricEndpointClockedTest, PreEffectRefusalRetriesBeforeEffectAdmissionOrBacking) {
  auto owner = std::make_shared<VectorBackingOwner>();
  auto gate = std::make_shared<amdgpu::VmTransferEffectGate>();
  constexpr uint64_t kOffset = VectorBackingOwner::kBase + 4;
  const std::array<std::byte, 4> payload{std::byte{0x31}, std::byte{0x42}, std::byte{0x53},
                                         std::byte{0x64}};
  bool open = false;
  int pre_admissions = 0;
  int effect_admissions = 0;
  auto admit_pre_effect = [&] {
    ++pre_admissions;
    amdgpu::MemoryIngressAdmission admission;
    if (!open)
      admission.outcome = amdgpu::VmAccessOutcome::Unavailable;
    return admission;
  };
  auto admit_effect = [&](const FabricBacking &, bool write, amdgpu::MemoryIngressPermit &) {
    EXPECT_TRUE(write);
    ++effect_admissions;
    FabricEndpoint::EffectAuthority authority;
    authority.permit = amdgpu::MemoryIngressPermit([] {});
    return authority;
  };
  auto write = endpoint_->prepare_write(kRequester, kOwner,
                                        FabricBacking(owner, kOwner, kOffset, payload.size()),
                                        payload, gate, admit_effect, admit_pre_effect);
  ASSERT_TRUE(write);
  write.operation->submit();
  ASSERT_TRUE(engine_.step()); // source submits
  ASSERT_TRUE(engine_.step()); // switch forwards
  ASSERT_TRUE(engine_.step()); // target refuses before effect admission
  EXPECT_EQ(pre_admissions, 1);
  EXPECT_EQ(effect_admissions, 0);
  EXPECT_EQ(owner->writes(), 0u);
  EXPECT_EQ(endpoint_->diagnostics().effects, 0u);
  EXPECT_EQ(write.operation->poll({}), amdgpu::VmAccessOutcome::Unavailable);

  open = true;
  for (int step = 0; step < 12 && owner->writes() == 0; ++step)
    ASSERT_TRUE(engine_.step());
  ASSERT_EQ(pre_admissions, 2);
  EXPECT_EQ(effect_admissions, 1);
  ASSERT_EQ(owner->writes(), 1u);
  EXPECT_EQ(endpoint_->diagnostics().effects, 1u);
  for (int step = 0; step < 8 && write.operation->poll({}) == amdgpu::VmAccessOutcome::Unavailable;
       ++step)
    ASSERT_TRUE(engine_.step());
  EXPECT_EQ(write.operation->poll({}), amdgpu::VmAccessOutcome::Complete);
  EXPECT_TRUE(std::equal(payload.begin(), payload.end(), owner->bytes_for_testing().begin() + 4));
}

TEST_F(FabricEndpointClockedTest, PreEffectPermitOutlivesCancelledTokenThroughTargetHandler) {
  using namespace std::chrono_literals;
  auto owner = std::make_shared<VectorBackingOwner>();
  auto gate = std::make_shared<amdgpu::VmTransferEffectGate>();
  std::atomic<int> acquisitions{0};
  std::atomic<int> releases{0};
  std::promise<void> entered_validation;
  auto validation_entered = entered_validation.get_future();
  std::promise<void> continue_validation;
  auto validation_continued = continue_validation.get_future().share();
  std::promise<void> entered_effect;
  auto effect_entered = entered_effect.get_future();
  std::promise<void> continue_effect;
  auto effect_continued = continue_effect.get_future().share();
  auto admit_pre_effect = [&] {
    acquisitions.fetch_add(1);
    amdgpu::MemoryIngressAdmission admission;
    admission.permit = amdgpu::MemoryIngressPermit([&] { releases.fetch_add(1); });
    return admission;
  };
  auto admit_effect = [&](const FabricBacking &, bool write, amdgpu::MemoryIngressPermit &) {
    EXPECT_FALSE(write);
    entered_validation.set_value();
    validation_continued.wait();
    FabricEndpoint::EffectAuthority authority;
    authority.permit = amdgpu::MemoryIngressPermit([] {});
    return authority;
  };
  endpoint_->set_effect_probe_for_testing([&] {
    entered_effect.set_value();
    effect_continued.wait();
  });
  constexpr uint64_t kOffset = VectorBackingOwner::kBase + 8;
  auto read = endpoint_->prepare_read(kRequester, kOwner, FabricBacking(owner, kOwner, kOffset, 4),
                                      gate, admit_effect, admit_pre_effect);
  ASSERT_TRUE(read);
  read.operation->submit();
  amdgpu::VmTransferToken token(std::move(read.operation), 4);
  ASSERT_TRUE(token.pending());
  ASSERT_TRUE(engine_.step());
  ASSERT_TRUE(engine_.step());
  auto target = std::async(std::launch::async, [&] { return engine_.step(); });
  if (validation_entered.wait_for(5s) != std::future_status::ready) {
    continue_validation.set_value();
    continue_effect.set_value();
    target.wait();
    FAIL() << "target did not enter effect admission";
  }
  EXPECT_EQ(acquisitions.load(), 1);
  EXPECT_EQ(releases.load(), 0);
  EXPECT_EQ(owner->reads(), 0u);

  continue_validation.set_value();
  if (effect_entered.wait_for(5s) != std::future_status::ready) {
    continue_effect.set_value();
    target.wait();
    FAIL() << "target did not reach the effect probe";
  }
  EXPECT_EQ(releases.load(), 0);
  EXPECT_EQ(owner->reads(), 0u);
  auto cancellation = std::async(std::launch::async, [&] { token.cancel(); });
  if (cancellation.wait_for(5s) != std::future_status::ready) {
    continue_effect.set_value();
    target.wait();
    cancellation.wait();
    FAIL() << "token cancellation blocked behind the target handler";
  }
  cancellation.get();
  EXPECT_FALSE(token.pending());
  EXPECT_EQ(releases.load(), 0);
  EXPECT_EQ(owner->reads(), 0u);

  continue_effect.set_value();
  EXPECT_TRUE(target.get());
  EXPECT_EQ(acquisitions.load(), 1);
  EXPECT_EQ(releases.load(), 1);
  EXPECT_EQ(owner->reads(), 0u);
  EXPECT_EQ(endpoint_->diagnostics().effects, 0u);
}

TEST_F(FabricEndpointClockedTest, CommittedReservedWriteReconcilesOnceBeforeReply) {
  auto owner = std::make_shared<VectorBackingOwner>();
  constexpr uint64_t kOffset = VectorBackingOwner::kBase + 4;
  auto gate = std::make_shared<amdgpu::VmTransferEffectGate>();
  const std::array<std::byte, 4> payload{std::byte{0x29}, std::byte{0x3a}, std::byte{0x4b},
                                         std::byte{0x5c}};
  int admissions = 0;
  int releases = 0;
  auto admit = [&](const FabricBacking &, bool write, amdgpu::MemoryIngressPermit &reservation) {
    EXPECT_TRUE(write);
    EXPECT_FALSE(reservation);
    ++admissions;
    reservation = amdgpu::MemoryIngressPermit([&] { ++releases; });
    FabricEndpoint::EffectAuthority authority;
    authority.permit = amdgpu::MemoryIngressPermit([] {});
    return authority;
  };
  auto write = endpoint_->prepare_write(kRequester, kOwner,
                                        FabricBacking(owner, kOwner, kOffset, payload.size()),
                                        payload, gate, admit);
  ASSERT_TRUE(write);
  write.operation->submit();
  ASSERT_TRUE(engine_.step());
  ASSERT_TRUE(engine_.step());
  ASSERT_TRUE(engine_.step());
  EXPECT_EQ(admissions, 1);
  EXPECT_EQ(releases, 1);
  EXPECT_EQ(owner->writes(), 1u);
  EXPECT_EQ(endpoint_->diagnostics().effects, 1u);
  EXPECT_EQ(write.operation->poll({}), amdgpu::VmAccessOutcome::Unavailable);

  gate->revoke();
  EXPECT_EQ(write.operation->reconcile_revocation({}), amdgpu::VmAccessOutcome::Complete);
  write.operation->cancel();
  EXPECT_EQ(write.operation->poll({}), amdgpu::VmAccessOutcome::Complete);
  for (int step = 0; step < 2; ++step)
    ASSERT_TRUE(engine_.step());
  EXPECT_EQ(admissions, 1);
  EXPECT_EQ(releases, 1);
  EXPECT_EQ(owner->writes(), 1u);
  EXPECT_EQ(endpoint_->diagnostics().effects, 1u);
  const auto &written = owner->bytes_for_testing();
  EXPECT_TRUE(std::equal(payload.begin(), payload.end(), written.begin() + 4));
}

TEST_F(FabricEndpointClockedTest, CancellationAndShutdownWaitForActiveReservationHandler) {
  using namespace std::chrono_literals;
  auto gate = std::make_shared<amdgpu::VmTransferEffectGate>();
  const std::array<std::byte, 4> payload{std::byte{0x48}, std::byte{0x48}, std::byte{0x48},
                                         std::byte{0x48}};
  std::atomic<int> releases{0};
  std::promise<void> entered;
  auto entered_future = entered.get_future();
  std::promise<void> continue_admission;
  auto continue_future = continue_admission.get_future().share();
  auto admit = [&](const FabricBacking &, bool, amdgpu::MemoryIngressPermit &reservation) {
    reservation = amdgpu::MemoryIngressPermit([&] { releases.fetch_add(1); });
    entered.set_value();
    continue_future.wait();
    FabricEndpoint::EffectAuthority authority;
    authority.outcome = amdgpu::VmAccessOutcome::Unavailable;
    return authority;
  };
  auto write =
      endpoint_->prepare_write(kRequester, kOwner, fabric_lease(136, 4), payload, gate, admit);
  ASSERT_TRUE(write);
  write.operation->submit();
  ASSERT_TRUE(engine_.step());
  ASSERT_TRUE(engine_.step());
  auto owner = std::async(std::launch::async, [&] { return engine_.step(); });
  if (entered_future.wait_for(5s) != std::future_status::ready) {
    continue_admission.set_value();
    owner.wait();
    FAIL() << "target admission did not acquire the request reservation";
  }

  write.operation->cancel();
  EXPECT_EQ(releases.load(), 0);
  auto shutdown = std::async(std::launch::async, [&] { endpoint_->shutdown(); });
  bool closed_admission = false;
  const auto deadline = std::chrono::steady_clock::now() + 5s;
  while (std::chrono::steady_clock::now() < deadline) {
    auto probe = endpoint_->prepare_read(kRequester, kOwner, fabric_lease(136, 4), gate);
    if (probe.outcome == amdgpu::VmAccessOutcome::Revoked) {
      closed_admission = true;
      break;
    }
    std::this_thread::yield();
  }
  EXPECT_TRUE(closed_admission);
  EXPECT_EQ(shutdown.wait_for(0ms), std::future_status::timeout);
  EXPECT_EQ(releases.load(), 0);
  continue_admission.set_value();
  EXPECT_TRUE(owner.get());
  shutdown.get();
  EXPECT_EQ(releases.load(), 1);
  EXPECT_EQ(write.operation->poll({}), amdgpu::VmAccessOutcome::Revoked);
  EXPECT_EQ(endpoint_->diagnostics().effects, 0u);
  EXPECT_EQ(lease(136, 4).bytes()[0], std::byte{0});
}

TEST_F(FabricEndpointClockedTest, VmInvalidationReconcilesCommittedWriteBeforeReply) {
  auto memory = std::make_shared<FabricVmMemory>(*endpoint_, backing_);
  amdgpu::GpuVm vm;
  const auto handle = vm.register_translated(7, memory, memory);
  const auto access = vm.snapshot(handle);
  ASSERT_TRUE(access);
  const std::array<std::byte, 4> source{std::byte{0x12}, std::byte{0x23}, std::byte{0x34},
                                        std::byte{0x45}};
  std::size_t progress = 0;
  amdgpu::VmTransferToken token;
  EXPECT_EQ(access->write(kVa + 96, source, progress, token), amdgpu::VmAccessOutcome::Unavailable);
  ASSERT_TRUE(token.pending());
  ASSERT_TRUE(engine_.step());
  ASSERT_TRUE(engine_.step());
  ASSERT_TRUE(engine_.step());
  EXPECT_EQ(endpoint_->diagnostics().effects, 1u);
  EXPECT_EQ(access->write(kVa + 96, source, progress, token), amdgpu::VmAccessOutcome::Unavailable);
  EXPECT_EQ(progress, 0u);

  ASSERT_TRUE(vm.invalidate(handle));
  EXPECT_EQ(access->write(kVa + 96, source, progress, token), amdgpu::VmAccessOutcome::Complete);
  EXPECT_EQ(progress, source.size());
  EXPECT_FALSE(token.pending());
  EXPECT_EQ(endpoint_->diagnostics().effects, 1u);
  EXPECT_TRUE(std::equal(source.begin(), source.end(), lease(96, 4).bytes().begin()));
}

TEST_F(FabricEndpointClockedTest, VmInvalidationRetainsReadBytesFromTargetEffect) {
  auto memory = std::make_shared<FabricVmMemory>(*endpoint_, backing_);
  amdgpu::GpuVm vm;
  const auto handle = vm.register_translated(7, memory, memory);
  const auto access = vm.snapshot(handle);
  ASSERT_TRUE(access);
  auto backing = lease(104, 4);
  ASSERT_TRUE(backing);
  std::fill(backing.bytes().begin(), backing.bytes().end(), std::byte{0x33});
  std::array<std::byte, 4> result{};
  std::size_t progress = 0;
  amdgpu::VmTransferToken token;
  EXPECT_EQ(access->read(kVa + 104, result, progress, token), amdgpu::VmAccessOutcome::Unavailable);
  ASSERT_TRUE(engine_.step());
  ASSERT_TRUE(engine_.step());
  ASSERT_TRUE(engine_.step());
  EXPECT_EQ(endpoint_->diagnostics().effects, 1u);
  EXPECT_EQ(access->read(kVa + 104, result, progress, token), amdgpu::VmAccessOutcome::Unavailable);
  std::fill(backing.bytes().begin(), backing.bytes().end(), std::byte{0x77});

  ASSERT_TRUE(vm.invalidate(handle));
  EXPECT_EQ(access->read(kVa + 104, result, progress, token), amdgpu::VmAccessOutcome::Complete);
  EXPECT_EQ(progress, result.size());
  EXPECT_FALSE(token.pending());
  EXPECT_EQ(result, (std::array<std::byte, 4>{std::byte{0x33}, std::byte{0x33}, std::byte{0x33},
                                              std::byte{0x33}}));
}

TEST_F(FabricEndpointClockedTest, VmInvalidationRevokesQueuedWriteWithoutEffect) {
  auto memory = std::make_shared<FabricVmMemory>(*endpoint_, backing_);
  amdgpu::GpuVm vm;
  const auto handle = vm.register_translated(7, memory, memory);
  const auto access = vm.snapshot(handle);
  ASSERT_TRUE(access);
  const std::array<std::byte, 4> source{std::byte{0x6a}, std::byte{0x6a}, std::byte{0x6a},
                                        std::byte{0x6a}};
  std::size_t progress = 0;
  amdgpu::VmTransferToken token;
  EXPECT_EQ(access->write(kVa + 112, source, progress, token),
            amdgpu::VmAccessOutcome::Unavailable);
  ASSERT_TRUE(token.pending());
  ASSERT_TRUE(vm.invalidate(handle));
  EXPECT_EQ(access->write(kVa + 112, source, progress, token), amdgpu::VmAccessOutcome::Revoked);
  EXPECT_EQ(progress, 0u);
  EXPECT_FALSE(token.pending());
  ASSERT_TRUE(engine_.step());
  EXPECT_EQ(endpoint_->diagnostics().effects, 0u);
  EXPECT_EQ(lease(112, 4).bytes()[0], std::byte{0});
}

TEST_F(FabricEndpointClockedTest, EnteredEffectRemainsPendingUntilPublished) {
  using namespace std::chrono_literals;
  auto memory = std::make_shared<FabricVmMemory>(*endpoint_, backing_);
  amdgpu::GpuVm vm;
  const auto handle = vm.register_translated(7, memory, memory);
  const auto access = vm.snapshot(handle);
  ASSERT_TRUE(access);
  const std::array<std::byte, 4> source{std::byte{0x5b}, std::byte{0x5b}, std::byte{0x5b},
                                        std::byte{0x5b}};
  std::size_t progress = 0;
  amdgpu::VmTransferToken token;
  EXPECT_EQ(access->write(kVa + 120, source, progress, token),
            amdgpu::VmAccessOutcome::Unavailable);
  ASSERT_TRUE(engine_.step());
  ASSERT_TRUE(engine_.step());

  std::promise<void> entered;
  auto entered_future = entered.get_future();
  std::promise<void> release;
  auto release_future = release.get_future().share();
  endpoint_->set_effect_probe_for_testing([&] {
    entered.set_value();
    release_future.wait();
  });
  auto owner = std::async(std::launch::async, [&] { return engine_.step(); });
  if (entered_future.wait_for(5s) != std::future_status::ready) {
    release.set_value();
    owner.wait();
    FAIL() << "target effect did not enter";
  }
  auto invalidation = std::async(std::launch::async, [&] { return vm.invalidate(handle); });
  const auto deadline = std::chrono::steady_clock::now() + 5s;
  while (access->is_current() && std::chrono::steady_clock::now() < deadline)
    std::this_thread::yield();
  EXPECT_FALSE(access->is_current());
  EXPECT_EQ(invalidation.wait_for(0ms), std::future_status::timeout);

  auto reconcile = std::async(std::launch::async,
                              [&] { return access->write(kVa + 120, source, progress, token); });
  const auto before_release = reconcile.wait_for(200ms);
  EXPECT_EQ(before_release, std::future_status::ready);
  if (before_release == std::future_status::ready) {
    EXPECT_EQ(reconcile.get(), amdgpu::VmAccessOutcome::Unavailable);
  }
  release.set_value();
  EXPECT_TRUE(owner.get());
  EXPECT_TRUE(invalidation.get());
  if (before_release != std::future_status::ready)
    (void)reconcile.get();
  EXPECT_EQ(access->write(kVa + 120, source, progress, token), amdgpu::VmAccessOutcome::Complete);
  EXPECT_EQ(progress, source.size());
  EXPECT_EQ(endpoint_->diagnostics().effects, 1u);
  EXPECT_EQ(lease(120, 4).bytes()[0], std::byte{0x5b});
}

TEST_F(FabricEndpointClockedTest, ShutdownRetainsCommittedReadResponse) {
  auto gate = std::make_shared<amdgpu::VmTransferEffectGate>();
  auto backing = lease(72, 4);
  ASSERT_TRUE(backing);
  std::fill(backing.bytes().begin(), backing.bytes().end(), std::byte{0x33});
  auto read = endpoint_->prepare_read(kRequester, kOwner, fabric_lease(72, 4), gate);
  ASSERT_TRUE(read);
  read.operation->submit();
  ASSERT_TRUE(engine_.step()); // owner submits at tick 0
  ASSERT_TRUE(engine_.step()); // switch receives at tick 1
  ASSERT_TRUE(engine_.step()); // target reads at tick 2
  EXPECT_EQ(engine_.context(0).current_tick(), 2u);
  EXPECT_EQ(endpoint_->diagnostics().effects, 1u);
  EXPECT_EQ(read.operation->poll({}), amdgpu::VmAccessOutcome::Unavailable);

  std::fill(backing.bytes().begin(), backing.bytes().end(), std::byte{0x77});
  stop(); // reply has not reached the source
  std::array<std::byte, 4> result{};
  EXPECT_EQ(read.operation->poll(result), amdgpu::VmAccessOutcome::Complete);
  EXPECT_EQ(result, (std::array<std::byte, 4>{std::byte{0x33}, std::byte{0x33}, std::byte{0x33},
                                              std::byte{0x33}}));
}

TEST_F(FabricEndpointClockedTest, ShutdownRetainsCommittedWriteButDropsQueuedWrite) {
  auto gate = std::make_shared<amdgpu::VmTransferEffectGate>();
  std::array<std::byte, 4> payload{std::byte{0x55}, std::byte{0x55}, std::byte{0x55},
                                   std::byte{0x55}};
  auto committed = endpoint_->prepare_write(kRequester, kOwner, fabric_lease(80, 4), payload, gate);
  auto queued = endpoint_->prepare_write(kRequester, kOwner, fabric_lease(88, 4), payload, gate);
  ASSERT_TRUE(committed);
  ASSERT_TRUE(queued);
  committed.operation->submit();
  ASSERT_TRUE(engine_.step());
  ASSERT_TRUE(engine_.step());
  ASSERT_TRUE(engine_.step());
  EXPECT_EQ(engine_.context(0).current_tick(), 2u);
  EXPECT_EQ(committed.operation->poll({}), amdgpu::VmAccessOutcome::Unavailable);
  EXPECT_EQ(lease(80, 4).bytes()[0], std::byte{0x55});
  queued.operation->submit();

  stop(); // committed response in flight; second request still queued
  EXPECT_EQ(committed.operation->poll({}), amdgpu::VmAccessOutcome::Complete);
  EXPECT_EQ(queued.operation->poll({}), amdgpu::VmAccessOutcome::Revoked);
  EXPECT_EQ(lease(88, 4).bytes()[0], std::byte{0});
  EXPECT_EQ(endpoint_->diagnostics().effects, 1u);
  EXPECT_EQ(endpoint_->diagnostics().directed_effects[0].write_bytes, payload.size());
}

class FabricEndpointAssemblyCoherenceTest : public ::testing::Test {
protected:
  void SetUp() override {
    auto loaded = load_assembly_config(assembly_coherence_config());
    ASSERT_EQ(loaded.num_gpus, 2u);
    machine_ = assemble_virtual_machine_topology(loaded, engine_.topology(),
                                                 /*daemon_mode=*/false);
    ASSERT_NE(machine_, nullptr);
    ASSERT_EQ(machine_->num_socs(), 2u);
    endpoint_ = machine_->fabric_endpoint();
    ASSERT_NE(endpoint_, nullptr);
    engine_.create();
    engine_.register_as_primary();
    started_ = true;

    SoC *owner_soc = machine_->soc(1);
    ASSERT_NE(owner_soc, nullptr);
    ASSERT_NE(owner_soc->memory(), nullptr);
    // The engine owns the assembled SoC until TearDown drains every request.
    auto memory =
        std::shared_ptr<amdgpu::GpuMemory>(owner_soc->memory(), [](amdgpu::GpuMemory *) {});
    owner_ = std::make_shared<CacheableBackingOwner>(std::move(memory));
  }

  void TearDown() override {
    if (started_) {
      engine_.primary_release();
      engine_.shutdown();
    }
  }

  void step_to_target() {
    ASSERT_TRUE(engine_.step());
    ASSERT_TRUE(engine_.step());
    ASSERT_TRUE(engine_.step());
  }

  simdojo::SimulationEngine engine_{{.num_threads = 1}};
  VirtualMachine *machine_ = nullptr;
  FabricEndpoint *endpoint_ = nullptr;
  std::shared_ptr<CacheableBackingOwner> owner_;
  bool started_ = false;
};

TEST_F(FabricEndpointAssemblyCoherenceTest,
       PeerReadPublishesOwnerDirtyL2WithoutGuestHdpMaintenance) {
  SoC *requester_soc = machine_->soc(0);
  SoC *owner_soc = machine_->soc(1);
  ASSERT_NE(requester_soc, nullptr);
  ASSERT_NE(owner_soc, nullptr);
  ASSERT_EQ(requester_soc->num_xcds(), 1u);
  ASSERT_EQ(requester_soc->num_iods(), 1u);
  ASSERT_EQ(owner_soc->num_xcds(), 1u);
  ASSERT_EQ(owner_soc->num_iods(), 1u);
  amdgpu::L2Cache *owner_l2 = owner_soc->xcd(0)->l2_cache();
  amdgpu::MemorySideCache *owner_msc = owner_soc->iod(0)->msc();
  ASSERT_NE(owner_l2, nullptr);
  ASSERT_NE(owner_msc, nullptr);
  constexpr uint64_t kAddress = CacheableBackingOwner::kBase;
  owner_soc->memory()->write8(kAddress, 0x11);
  uint8_t cached = 0;
  ASSERT_EQ(owner_msc->read(kAddress, &cached, 1), amdgpu::VmAccessOutcome::Complete);
  ASSERT_EQ(cached, 0x11);

  std::array<uint8_t, amdgpu::L2Cache::LINE_SIZE> dirty_line{};
  dirty_line[0] = 0x5a;
  ASSERT_EQ(owner_l2->writeback_line(kAddress, dirty_line.data(), amdgpu::Mtype::RW),
            amdgpu::VmAccessOutcome::Complete);
  // No guest HDP maintenance occurs between this dirty owner cache fill and
  // the peer read. The endpoint's conservative maintenance publishes it.
  ASSERT_EQ(owner_soc->memory()->read8(kAddress), 0x11);
  const uint64_t requester_epoch = requester_soc->cache_coherence()->current_epoch();
  const uint64_t owner_epoch = owner_soc->cache_coherence()->current_epoch();
  auto gate = std::make_shared<amdgpu::VmTransferEffectGate>();
  auto read =
      endpoint_->prepare_read(kRequester, kOwner, FabricBacking(owner_, kOwner, kAddress, 1), gate);
  ASSERT_TRUE(read);
  read.operation->submit();
  step_to_target();
  EXPECT_EQ(read.operation->poll({}), amdgpu::VmAccessOutcome::Unavailable);
  EXPECT_EQ(owner_soc->memory()->read8(kAddress), 0x5a);
  EXPECT_GT(requester_soc->cache_coherence()->current_epoch(), requester_epoch);
  EXPECT_GT(owner_soc->cache_coherence()->current_epoch(), owner_epoch);
  ASSERT_TRUE(engine_.step());
  ASSERT_TRUE(engine_.step());
  std::array<std::byte, 1> read_result{};
  EXPECT_EQ(read.operation->poll(read_result), amdgpu::VmAccessOutcome::Complete);
  EXPECT_EQ(read_result[0], std::byte{0x5a});

  uint8_t refreshed = 0;
  ASSERT_EQ(owner_msc->read(kAddress, &refreshed, 1), amdgpu::VmAccessOutcome::Complete);
  ASSERT_EQ(refreshed, 0x5a);
  const uint64_t requester_epoch_before_write = requester_soc->cache_coherence()->current_epoch();
  const uint64_t owner_epoch_before_write = owner_soc->cache_coherence()->current_epoch();
  const std::array<std::byte, 1> payload{std::byte{0x33}};
  auto write = endpoint_->prepare_write(
      kRequester, kOwner, FabricBacking(owner_, kOwner, kAddress, payload.size()), payload, gate);
  ASSERT_TRUE(write);
  write.operation->submit();
  step_to_target();
  EXPECT_EQ(write.operation->poll({}), amdgpu::VmAccessOutcome::Unavailable);
  EXPECT_EQ(owner_soc->memory()->read8(kAddress), 0x33);
  EXPECT_GT(requester_soc->cache_coherence()->current_epoch(), requester_epoch_before_write);
  EXPECT_GT(owner_soc->cache_coherence()->current_epoch(), owner_epoch_before_write);
  refreshed = 0;
  EXPECT_EQ(owner_msc->read(kAddress, &refreshed, 1), amdgpu::VmAccessOutcome::Complete);
  EXPECT_EQ(refreshed, 0x33);
  ASSERT_TRUE(engine_.step());
  ASSERT_TRUE(engine_.step());
  EXPECT_EQ(write.operation->poll({}), amdgpu::VmAccessOutcome::Complete);
}

TEST(FabricEndpointAssemblyTest, RetainsExplicitLocalRouteAfterLoadedConfigIsReleased) {
  simdojo::Topology topology;
  VirtualMachine *machine = nullptr;
  {
    auto loaded = load_assembly_config(assembly_config(kLocalSwitchManifest));
    ASSERT_EQ(loaded.num_gpus, 2u);
    ASSERT_EQ(loaded.extra_gpu_builds.size(), 1u);
    machine = assemble_virtual_machine_topology(loaded, topology, /*daemon_mode=*/false);
    ASSERT_NE(machine, nullptr);
    loaded.system_graph = {}; // the VM must retain an independent projection
  }

  ASSERT_EQ(topology.root(), machine);
  ASSERT_NE(machine->driver(), nullptr);
  EXPECT_EQ(machine->peer_routes().node_id(), "local");
  const auto routes = machine->peer_routes().routes();
  ASSERT_EQ(routes.size(), 1u);
  EXPECT_EQ(routes[0].source_gpu, "g0");
  EXPECT_EQ(routes[0].target_gpu, "g1");
  EXPECT_EQ(routes[0].source_gpu_id, kRequester);
  EXPECT_EQ(routes[0].target_gpu_id, kOwner);
  EXPECT_EQ(routes[0].endpoints, (std::vector<std::string>{"e0", "sw", "e1"}));

  const auto *fabric = machine->fabric_endpoint();
  ASSERT_NE(fabric, nullptr);
  ASSERT_EQ(fabric->routes().size(), 1u);
  EXPECT_EQ(fabric->routes()[0].endpoints, routes[0].endpoints);
  EXPECT_EQ(fabric->children().size(), 3u);
  EXPECT_EQ(topology.links().size(), 4u); // two physical edges in both directions
}

TEST(FabricEndpointAssemblyTest, RouteFreeLegacyOwnerResolvesLateGpuIdentityForScopedEffect) {
  auto loaded = load_assembly_config(assembly_config());
  simdojo::SimulationEngine engine{{.num_threads = 1}};
  VirtualMachine *machine =
      assemble_virtual_machine_topology(loaded, engine.topology(), /*daemon_mode=*/false);
  ASSERT_NE(machine, nullptr);
  FabricEndpoint *endpoint = machine->fabric_endpoint();
  ASSERT_NE(endpoint, nullptr);
  ASSERT_TRUE(endpoint->routes().empty());
  auto domain = machine->soc(0)->cache_coherence();
  ASSERT_NE(domain, nullptr);
  engine.create();
  engine.register_as_primary();

  auto owner = std::make_shared<VectorBackingOwner>();
  auto gate = std::make_shared<amdgpu::VmTransferEffectGate>();
  const std::array<std::byte, 4> payload{std::byte{0xb2}, std::byte{0xb2}, std::byte{0xb2},
                                         std::byte{0xb2}};
  bool resolved_dirty_line = false;
  auto admit = [domain, &resolved_dirty_line](const FabricBacking &, bool write,
                                              amdgpu::MemoryIngressPermit &) {
    EXPECT_TRUE(write);
    FabricEndpoint::EffectAuthority authority;
    authority.permit = amdgpu::MemoryIngressPermit([] {});
    authority.cache_aliases.push_back({.domain = {domain, kRequester},
                                       .vmid = 1,
                                       .address = VectorBackingOwner::kBase,
                                       .size = amdgpu::L2Cache::LINE_SIZE,
                                       .admission_generation = 1,
                                       .cache_identity = 1});
    authority.resolve_writeback =
        [&resolved_dirty_line](uint32_t, uint32_t, uint64_t,
                               size_t) -> amdgpu::PreparedCanonicalWriteback {
      resolved_dirty_line = true;
      throw std::runtime_error("unexpected dirty line in route-free endpoint test");
    };
    return authority;
  };
  auto write = endpoint->prepare_local_write(
      kRequester, FabricBacking(owner, kRequester, VectorBackingOwner::kBase, payload.size()),
      payload, gate, admit);
  ASSERT_TRUE(write);
  write.operation->submit();
  ASSERT_TRUE(engine.step());
  EXPECT_EQ(write.operation->poll({}), amdgpu::VmAccessOutcome::Complete);
  EXPECT_EQ(owner->writes(), 1u);
  EXPECT_FALSE(resolved_dirty_line);
  const auto diagnostics = endpoint->diagnostics();
  EXPECT_EQ(diagnostics.effects, 1u);
  ASSERT_EQ(diagnostics.scoped_maintenance.size(), 1u);
  EXPECT_EQ(diagnostics.scoped_maintenance[0].domain_index, 0u);
  EXPECT_EQ(diagnostics.scoped_maintenance[0].successes, 1u);
  EXPECT_EQ(diagnostics.scoped_maintenance[0].admission_timeouts, 0u);
  engine.primary_release();
  engine.shutdown();
}

TEST(FabricEndpointAssemblyTest, RetryablePreparedMaintenanceReleasesPermitAndCommitsOnce) {
  auto loaded = load_assembly_config(assembly_config());
  simdojo::SimulationEngine engine{{.num_threads = 1}};
  VirtualMachine *machine =
      assemble_virtual_machine_topology(loaded, engine.topology(), /*daemon_mode=*/false);
  ASSERT_NE(machine, nullptr);
  FabricEndpoint *endpoint = machine->fabric_endpoint();
  ASSERT_NE(endpoint, nullptr);
  auto domain = machine->soc(0)->cache_coherence();
  ASSERT_NE(domain, nullptr);

  auto memory = std::make_shared<amdgpu::GpuMemory>("maintenance_backing");
  auto owner = std::make_shared<CacheableBackingOwner>(memory);
  amdgpu::GpuVm maintenance_vm;
  amdgpu::L2Cache l2("unselected_l2", domain);
  l2.set_backing_memory(memory.get());
  l2.set_legacy_maintenance_memory(memory.get());
  l2.set_legacy_maintenance_vm(&maintenance_vm);
  constexpr uint32_t kVmid = 73;
  constexpr uint64_t kAddress = CacheableBackingOwner::kBase;
  const uint64_t other_line = kAddress + amdgpu::L2Cache::LINE_SIZE;
  std::array<uint8_t, amdgpu::L2Cache::LINE_SIZE> dirty{};
  dirty.fill(0x7b);
  ASSERT_EQ(l2.writeback_line(other_line, dirty.data(), amdgpu::Mtype::RW, kVmid),
            amdgpu::VmAccessOutcome::Complete);

  engine.create();
  engine.register_as_primary();
  auto gate = std::make_shared<amdgpu::VmTransferEffectGate>();
  const std::array<std::byte, 1> payload{std::byte{0x5a}};
  const FabricBacking backing(owner, kRequester, kAddress, payload.size());
  int admissions = 0;
  int authority_releases = 0;
  int reservation_releases = 0;
  int witness_calls = 0;
  auto admit = [&](const FabricBacking &, bool write, amdgpu::MemoryIngressPermit &reservation) {
    EXPECT_TRUE(write);
    ++admissions;
    if (!reservation)
      reservation = amdgpu::MemoryIngressPermit([&] { ++reservation_releases; });
    FabricEndpoint::EffectAuthority authority;
    authority.permit = amdgpu::MemoryIngressPermit([&] { ++authority_releases; });
    authority.cache_aliases.push_back({.domain = {domain, kRequester},
                                       .vmid = kVmid,
                                       .address = kAddress,
                                       .size = amdgpu::L2Cache::LINE_SIZE,
                                       .admission_generation = 1,
                                       .cache_identity = 1,
                                       .backing_id = backing.backing_id(),
                                       .backing_offset = kAddress});
    authority.resolve_writeback = [](uint32_t, uint32_t, uint64_t,
                                     size_t) -> amdgpu::PreparedCanonicalWriteback {
      throw std::runtime_error("selected line unexpectedly dirty");
    };
    authority.resolve_witness = [&](uint32_t, uint32_t, uint64_t, size_t, uint64_t,
                                    uint64_t) -> std::optional<amdgpu::CanonicalLineWitness> {
      ++witness_calls;
      return std::nullopt;
    };
    return authority;
  };
  auto write = endpoint->prepare_local_write(kRequester, backing, payload, gate, admit);
  ASSERT_TRUE(write);
  write.operation->submit();
  ASSERT_TRUE(engine.step());
  EXPECT_EQ(admissions, 1);
  EXPECT_EQ(witness_calls, 1);
  EXPECT_EQ(authority_releases, 1);
  EXPECT_EQ(reservation_releases, 0);
  EXPECT_EQ(owner->writes(), 0u);
  EXPECT_EQ(write.operation->poll({}), amdgpu::VmAccessOutcome::Unavailable);

  // The unselected line is retired before the next owner tick. The same
  // request and reservation must then pass maintenance and commit once.
  l2.invalidate_all();
  ASSERT_TRUE(engine.step());
  EXPECT_EQ(admissions, 2);
  EXPECT_EQ(authority_releases, 2);
  EXPECT_EQ(reservation_releases, 1);
  EXPECT_EQ(witness_calls, 1);
  EXPECT_EQ(owner->writes(), 1u);
  EXPECT_EQ(write.operation->poll({}), amdgpu::VmAccessOutcome::Complete);
  EXPECT_EQ(endpoint->diagnostics().effects, 1u);
  engine.primary_release();
  engine.shutdown();
}

TEST(FabricEndpointAssemblyTest, LegacyAndRemoteSharedSwitchInstallRouteFreeOwnerEndpoint) {
  simdojo::Topology legacy_topology;
  VirtualMachine *legacy_machine = nullptr;
  {
    auto loaded = load_assembly_config(assembly_config());
    legacy_machine = assemble_virtual_machine_topology(loaded, legacy_topology,
                                                       /*daemon_mode=*/false);
  }
  ASSERT_NE(legacy_machine, nullptr);
  EXPECT_TRUE(legacy_machine->peer_routes().routes().empty());
  ASSERT_NE(legacy_machine->fabric_endpoint(), nullptr);
  EXPECT_TRUE(legacy_machine->fabric_endpoint()->routes().empty());
  EXPECT_TRUE(legacy_machine->fabric_endpoint()->children().empty());
  EXPECT_TRUE(legacy_topology.links().empty());

  simdojo::Topology remote_topology;
  VirtualMachine *remote_machine = nullptr;
  {
    auto loaded = load_assembly_config(assembly_config(kRemoteSharedSwitchManifest), "local");
    ASSERT_EQ(loaded.num_gpus, 2u);
    remote_machine = assemble_virtual_machine_topology(loaded, remote_topology,
                                                       /*daemon_mode=*/false);
    loaded.system_graph = {};
  }
  ASSERT_NE(remote_machine, nullptr);
  EXPECT_TRUE(remote_machine->peer_routes().routes().empty());
  ASSERT_NE(remote_machine->fabric_endpoint(), nullptr);
  EXPECT_TRUE(remote_machine->fabric_endpoint()->routes().empty());
  EXPECT_TRUE(remote_machine->fabric_endpoint()->children().empty());
  EXPECT_TRUE(remote_topology.links().empty());
}

} // namespace
} // namespace rocjitsu
