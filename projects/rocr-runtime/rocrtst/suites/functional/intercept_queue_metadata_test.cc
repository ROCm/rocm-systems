//===- intercept_queue_metadata_test.cc - InterceptQueue metadata tests --===//
//
// SPDX-License-Identifier: NCSA
//
//===----------------------------------------------------------------------===//

#include <cstdlib>
#include <cstring>
#include <memory>
#include <vector>

#include "core/inc/intercept_queue.h"
#include "core/util/atomic_helpers.h"
#include "gtest/gtest.h"
#include "inc/amd_hsa_signal.h"

namespace rocr {
namespace core {

// The host-only target constructs DoorbellSignal without the runtime signal
// implementation. It never exposes or destroys that signal through HSA.
Signal::~Signal() {}
void Signal::registerIpc() {}
void SharedSignalPool_t::clear() {}
SharedSignal* SharedSignalPool_t::alloc() { return nullptr; }
void SharedSignalPool_t::free(SharedSignal*) {}

}  // namespace core
}  // namespace rocr

namespace {

using AqlPacket = rocr::core::AqlPacket;
using MetadataPacket = rocr::core::AqlMetadataPrefetchPacket;
using Queue = rocr::core::Queue;
using SharedQueue = rocr::core::SharedQueue;

constexpr uint32_t kQueueSize = 8;

class QueueStorage {
 protected:
  SharedQueue storage_{};
};

class FakeQueue final : private QueueStorage, public Queue {
 public:
  explicit FakeQueue(bool metadata_supported = true)
      : Queue(&storage_, 0, nullptr), metadata_supported_(metadata_supported) {
    ring_.resize(kQueueSize);
    metadata_ring_.resize(kQueueSize);
    amd_queue_.hsa_queue.size = kQueueSize;
    amd_queue_.hsa_queue.base_address = ring_.data();
    for (auto& metadata : metadata_ring_) metadata = NoopMetadata();
  }

  hsa_status_t Inactivate() override { return HSA_STATUS_SUCCESS; }
  hsa_status_t SetPriority(rocr::HSA::hsa_amd_queue_priority_internal_t) override {
    return HSA_STATUS_SUCCESS;
  }
  uint64_t LoadReadIndexAcquire() override { return read_index_; }
  uint64_t LoadReadIndexRelaxed() override { return read_index_; }
  uint64_t LoadWriteIndexAcquire() override { return write_index_; }
  uint64_t LoadWriteIndexRelaxed() override { return write_index_; }
  void StoreReadIndexRelaxed(uint64_t value) override { read_index_ = value; }
  void StoreReadIndexRelease(uint64_t value) override { read_index_ = value; }
  void StoreWriteIndexRelaxed(uint64_t value) override { write_index_ = value; }
  void StoreWriteIndexRelease(uint64_t value) override { write_index_ = value; }
  uint64_t CasWriteIndexAcqRel(uint64_t expected, uint64_t value) override {
    return CasWriteIndexRelaxed(expected, value);
  }
  uint64_t CasWriteIndexAcquire(uint64_t expected, uint64_t value) override {
    return CasWriteIndexRelaxed(expected, value);
  }
  uint64_t CasWriteIndexRelaxed(uint64_t expected, uint64_t value) override {
    const uint64_t previous = write_index_;
    if (previous == expected) write_index_ = value;
    return previous;
  }
  uint64_t CasWriteIndexRelease(uint64_t expected, uint64_t value) override {
    return CasWriteIndexRelaxed(expected, value);
  }
  uint64_t AddWriteIndexAcqRel(uint64_t value) override { return AddWriteIndexRelaxed(value); }
  uint64_t AddWriteIndexAcquire(uint64_t value) override { return AddWriteIndexRelaxed(value); }
  uint64_t AddWriteIndexRelaxed(uint64_t value) override {
    const uint64_t previous = write_index_;
    write_index_ += value;
    return previous;
  }
  uint64_t AddWriteIndexRelease(uint64_t value) override { return AddWriteIndexRelaxed(value); }
  hsa_status_t SetCUMasking(uint32_t, const uint32_t*) override { return HSA_STATUS_SUCCESS; }
  hsa_status_t GetCUMasking(uint32_t, uint32_t*) override { return HSA_STATUS_SUCCESS; }
  void ExecutePM4(uint32_t*, size_t, hsa_fence_scope_t, hsa_fence_scope_t, hsa_signal_t*) override {
  }
  void SetProfiling(bool) override {}

  hsa_status_t GetInfo(hsa_queue_info_attribute_t attribute, void* value) override {
    switch (attribute) {
      case HSA_AMD_QUEUE_INFO_PREFETCH_METADATA_DISPATCH_PKT_VERSION_MAJOR:
        *reinterpret_cast<uint8_t*>(value) = 0x11;
        return HSA_STATUS_SUCCESS;
      case HSA_AMD_QUEUE_INFO_PREFETCH_METADATA_DISPATCH_PKT_VERSION_MINOR:
        *reinterpret_cast<uint8_t*>(value) = 0x22;
        return HSA_STATUS_SUCCESS;
      case HSA_AMD_QUEUE_INFO_PREFETCH_METADATA_BARRIER_PKT_VERSION_MAJOR:
        *reinterpret_cast<uint8_t*>(value) = 0x33;
        return HSA_STATUS_SUCCESS;
      case HSA_AMD_QUEUE_INFO_PREFETCH_METADATA_BARRIER_PKT_VERSION_MINOR:
        *reinterpret_cast<uint8_t*>(value) = 0x44;
        return HSA_STATUS_SUCCESS;
      case HSA_AMD_QUEUE_INFO_PROPERTIES:
        std::memset(value, 0x55, 8);
        return HSA_STATUS_SUCCESS;
      case HSA_AMD_QUEUE_INFO_PREFETCH_METADATA_RING_BUFFER:
        if (metadata_supported_) {
          *reinterpret_cast<uint64_t*>(value) = reinterpret_cast<uint64_t>(metadata_ring_.data());
          return HSA_STATUS_SUCCESS;
        }
      default:
        return HSA_STATUS_ERROR_INVALID_ARGUMENT;
    }
  }

  MetadataPacket& MetadataAt(uint64_t index) { return metadata_ring_[index & (kQueueSize - 1)]; }
  void Consume(uint64_t count) { read_index_ += count; }

 protected:
  bool _IsA(rtti_t) const override { return false; }

 private:
  static MetadataPacket NoopMetadata() {
    MetadataPacket metadata{};
    metadata.packet.header0.type = HSA_PACKET_TYPE_INVALID;
    metadata.packet.header1.type = HSA_PACKET_TYPE_INVALID;
    metadata.packet.header2.type = HSA_PACKET_TYPE_INVALID;
    metadata.packet.header3.type = HSA_PACKET_TYPE_INVALID;
    return metadata;
  }

  bool metadata_supported_;
  std::vector<AqlPacket> ring_;
  std::vector<MetadataPacket> metadata_ring_;
  uint64_t read_index_ = 0;
  uint64_t write_index_ = 0;
};

class InterceptQueueStorage {
 protected:
  SharedQueue storage_{};
};

class TestInterceptQueue final : private InterceptQueueStorage, public rocr::core::InterceptQueue {
 public:
  explicit TestInterceptQueue(std::unique_ptr<Queue> queue)
      : InterceptQueue(std::move(queue), &storage_) {}
};

class InterceptQueueMetadataTest : public testing::Test {
 protected:
  void SetUp() override {
    rocr::core::BaseShared::SetAllocateAndFree(
        [](size_t size, size_t alignment, uint32_t, int) {
          void* memory = nullptr;
          return posix_memalign(&memory, alignment, size) == 0 ? memory : nullptr;
        },
        [](void* memory) { std::free(memory); });
  }
};

uint16_t PacketHeader(hsa_packet_type_t type, uint16_t flags = 0) {
  return (type << HSA_PACKET_HEADER_TYPE) | flags;
}

hsa_signal_t MakeSignal(amd_signal_t& signal, uint32_t event_id) {
  std::memset(&signal, 0, sizeof(signal));
  signal.event_id = event_id;
  return {reinterpret_cast<uint64_t>(&signal)};
}

AqlPacket MakeDispatch(uint64_t kernel_object, hsa_signal_t completion_signal = {0}) {
  AqlPacket packet{};
  packet.dispatch.header = PacketHeader(HSA_PACKET_TYPE_KERNEL_DISPATCH);
  packet.dispatch.workgroup_size_x = 64;
  packet.dispatch.grid_size_x = 64;
  packet.dispatch.kernel_object = kernel_object;
  packet.dispatch.completion_signal = completion_signal;
  return packet;
}

AqlPacket MakeBarrier() {
  AqlPacket packet{};
  packet.barrier_and.header = PacketHeader(HSA_PACKET_TYPE_BARRIER_AND);
  return packet;
}

AqlPacket MakeExtDispatch(hsa_signal_t completion_signal = {0}) {
  AqlPacket packet{};
  packet.ext_dispatch.header = PacketHeader(HSA_PACKET_TYPE_VENDOR_SPECIFIC);
  packet.ext_dispatch.amd_format = HSA_AMD_PACKET_TYPE_EXT_KERNEL_DISPATCH;
  packet.ext_dispatch.completion_signal = completion_signal;
  return packet;
}

AqlPacket MakeBarrierValue(hsa_signal_t completion_signal = {0}) {
  hsa_amd_barrier_value_packet_t barrier{};
  barrier.header.header = PacketHeader(HSA_PACKET_TYPE_VENDOR_SPECIFIC);
  barrier.header.AmdFormat = HSA_AMD_PACKET_TYPE_BARRIER_VALUE;
  barrier.completion_signal = completion_signal;
  static_assert(sizeof(barrier) == sizeof(AqlPacket), "Unexpected barrier-value packet size");

  AqlPacket packet{};
  std::memcpy(&packet, &barrier, sizeof(packet));
  return packet;
}

AqlPacket MakeAgentDispatch() {
  AqlPacket packet{};
  packet.agent.header = PacketHeader(HSA_PACKET_TYPE_AGENT_DISPATCH);
  packet.agent.completion_signal = {1};
  return packet;
}

MetadataPacket MakeMetadata(uint8_t seed, uint32_t event_id) {
  MetadataPacket metadata;
  std::memset(&metadata, seed, sizeof(metadata));
  metadata.packet.header0.type = HSA_PACKET_TYPE_KERNEL_DISPATCH;
  metadata.packet.header1.type = HSA_PACKET_TYPE_BARRIER_AND;
  metadata.packet.header2.type = HSA_PACKET_TYPE_BARRIER_OR;
  metadata.packet.header3.type = HSA_PACKET_TYPE_VENDOR_SPECIFIC;
  metadata.packet.event_id = event_id;
  return metadata;
}

MetadataPacket NoopMetadata() {
  MetadataPacket metadata{};
  metadata.packet.header0.type = HSA_PACKET_TYPE_INVALID;
  metadata.packet.header1.type = HSA_PACKET_TYPE_INVALID;
  metadata.packet.header2.type = HSA_PACKET_TYPE_INVALID;
  metadata.packet.header3.type = HSA_PACKET_TYPE_INVALID;
  return metadata;
}

void ExpectMetadataEq(const MetadataPacket& actual, const MetadataPacket& expected) {
  EXPECT_EQ(std::memcmp(&actual, &expected, sizeof(actual)), 0);
}

void ExpectNoopMetadata(const MetadataPacket& metadata) {
  ExpectMetadataEq(metadata, NoopMetadata());
}

AqlPacket* ProxyRing(TestInterceptQueue& queue) {
  return reinterpret_cast<AqlPacket*>(queue.amd_queue_.hsa_queue.base_address);
}

MetadataPacket* ProxyMetadataRing(TestInterceptQueue& queue) {
  uint64_t metadata_ring = 0;
  EXPECT_EQ(queue.GetInfo(HSA_AMD_QUEUE_INFO_PREFETCH_METADATA_RING_BUFFER, &metadata_ring),
            HSA_STATUS_SUCCESS);
  return reinterpret_cast<MetadataPacket*>(metadata_ring);
}

void WriteProxyPacket(TestInterceptQueue& queue, uint64_t index, const AqlPacket& packet) {
  const uint64_t slot = index & (queue.amd_queue_.hsa_queue.size - 1);
  ProxyRing(queue)[slot] = packet;
  rocr::atomic::Store(&ProxyRing(queue)[slot].packet.header, packet.packet.header,
                      std::memory_order_release);
}

void PassThrough(const void* packets, uint64_t count, uint64_t, void*,
                 hsa_amd_queue_intercept_packet_writer writer) {
  writer(packets, count);
}

void InsertBarrier(const void* packets, uint64_t count, uint64_t, void*,
                   hsa_amd_queue_intercept_packet_writer writer) {
  std::vector<AqlPacket> output;
  output.reserve(count * 2);
  for (const auto* input = static_cast<const AqlPacket*>(packets); count-- != 0; ++input) {
    output.push_back(MakeBarrier());
    output.push_back(*input);
  }
  writer(output.data(), output.size());
}

struct SignalReplacement {
  hsa_signal_t signal;
};

void ReplaceSignal(const void* packets, uint64_t count, uint64_t, void* data,
                   hsa_amd_queue_intercept_packet_writer writer) {
  std::vector<AqlPacket> output(static_cast<const AqlPacket*>(packets),
                                static_cast<const AqlPacket*>(packets) + count);
  const auto signal = static_cast<SignalReplacement*>(data)->signal;
  for (auto& packet : output) {
    switch (AqlPacket::type(packet.packet.header)) {
      case HSA_PACKET_TYPE_KERNEL_DISPATCH:
        packet.dispatch.completion_signal = signal;
        break;
      case HSA_PACKET_TYPE_VENDOR_SPECIFIC:
        if (packet.amd_vendor.format == HSA_AMD_PACKET_TYPE_EXT_KERNEL_DISPATCH) {
          packet.ext_dispatch.completion_signal = signal;
        } else if (packet.amd_vendor.format == HSA_AMD_PACKET_TYPE_BARRIER_VALUE) {
          reinterpret_cast<hsa_amd_barrier_value_packet_t*>(&packet)->completion_signal = signal;
        }
        break;
      default:
        break;
    }
  }
  writer(output.data(), output.size());
}

void ReversePackets(const void* packets, uint64_t count, uint64_t, void*,
                    hsa_amd_queue_intercept_packet_writer writer) {
  const auto* input = static_cast<const AqlPacket*>(packets);
  ASSERT_EQ(count, 2u);
  AqlPacket output[] = {input[1], input[0]};
  writer(output, 2);
}

void DropFirstPacket(const void* packets, uint64_t count, uint64_t, void*,
                     hsa_amd_queue_intercept_packet_writer writer) {
  ASSERT_GT(count, 1u);
  writer(static_cast<const AqlPacket*>(packets) + 1, count - 1);
}

void ReplaceWithBarrier(const void*, uint64_t count, uint64_t, void*,
                        hsa_amd_queue_intercept_packet_writer writer) {
  std::vector<AqlPacket> output(count, MakeBarrier());
  writer(output.data(), output.size());
}

void ChangeHeaderFlags(const void* packets, uint64_t count, uint64_t, void*,
                       hsa_amd_queue_intercept_packet_writer writer) {
  std::vector<AqlPacket> output(static_cast<const AqlPacket*>(packets),
                                static_cast<const AqlPacket*>(packets) + count);
  output.front().packet.header =
      PacketHeader(HSA_PACKET_TYPE_KERNEL_DISPATCH,
                   (1 << HSA_PACKET_HEADER_BARRIER) |
                       (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_ACQUIRE_FENCE_SCOPE) |
                       (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_RELEASE_FENCE_SCOPE));
  writer(output.data(), output.size());
}

void ChangePacketType(const void* packets, uint64_t count, uint64_t, void*,
                      hsa_amd_queue_intercept_packet_writer writer) {
  std::vector<AqlPacket> output(static_cast<const AqlPacket*>(packets),
                                static_cast<const AqlPacket*>(packets) + count);
  output.front().barrier_and.header = PacketHeader(HSA_PACKET_TYPE_BARRIER_AND);
  output.front().barrier_and.completion_signal = {0};
  writer(output.data(), output.size());
}

void ChangeReservedHeaderBit(const void* packets, uint64_t count, uint64_t, void*,
                             hsa_amd_queue_intercept_packet_writer writer) {
  std::vector<AqlPacket> output(static_cast<const AqlPacket*>(packets),
                                static_cast<const AqlPacket*>(packets) + count);
  output.front().packet.header |= 1u << 13;
  writer(output.data(), output.size());
}

void WriteNothing(const void* packets, uint64_t, uint64_t, void*,
                  hsa_amd_queue_intercept_packet_writer writer) {
  writer(packets, 0);
}

void ExpandPacket(const void* packets, uint64_t count, uint64_t, void*,
                  hsa_amd_queue_intercept_packet_writer writer) {
  ASSERT_EQ(count, 1u);
  const auto* input = static_cast<const AqlPacket*>(packets);
  AqlPacket output[] = {MakeBarrier(), input[0], MakeBarrier()};
  writer(output, 3);
}

bool marker_callback_fired = false;
uint64_t marker_packet_index = 0;

void MarkerCallback(const amd_aql_intercept_marker_t*, hsa_queue_t*, uint64_t packet_index) {
  marker_callback_fired = true;
  marker_packet_index = packet_index;
}

void InsertMarker(const void* packets, uint64_t count, uint64_t, void*,
                  hsa_amd_queue_intercept_packet_writer writer) {
  ASSERT_EQ(count, 1u);
  amd_aql_intercept_marker_t marker{};
  marker.header = PacketHeader(HSA_PACKET_TYPE_VENDOR_SPECIFIC);
  marker.format = AMD_AQL_FORMAT_INTERCEPT_MARKER;
  marker.callback = MarkerCallback;
  AqlPacket output[2] = {};
  static_assert(sizeof(marker) == sizeof(AqlPacket), "Unexpected intercept marker size");
  std::memcpy(&output[0], &marker, sizeof(marker));
  output[1] = *static_cast<const AqlPacket*>(packets);
  writer(output, 2);
}

TEST_F(InterceptQueueMetadataTest, PassthroughPreservesFullMetadata) {
  auto fake = std::make_unique<FakeQueue>();
  auto* fake_queue = fake.get();
  TestInterceptQueue queue(std::move(fake));
  queue.AddInterceptor(PassThrough, nullptr);

  amd_signal_t signal{};
  const MetadataPacket metadata = MakeMetadata(0x11, 41);
  ProxyMetadataRing(queue)[0] = metadata;
  WriteProxyPacket(queue, 0, MakeDispatch(0x1000, MakeSignal(signal, 41)));
  queue.StoreWriteIndexRelaxed(1);
  queue.StoreRelaxed(0);

  EXPECT_EQ(fake_queue->LoadWriteIndexRelaxed(), 1u);
  ExpectMetadataEq(fake_queue->MetadataAt(0), metadata);
}

TEST_F(InterceptQueueMetadataTest, InsertedPacketsGetNoopMetadata) {
  auto fake = std::make_unique<FakeQueue>();
  auto* fake_queue = fake.get();
  TestInterceptQueue queue(std::move(fake));
  queue.AddInterceptor(InsertBarrier, nullptr);

  const MetadataPacket metadata = MakeMetadata(0x22, 0);
  ProxyMetadataRing(queue)[0] = metadata;
  WriteProxyPacket(queue, 0, MakeDispatch(0x2000));
  queue.StoreWriteIndexRelaxed(1);
  queue.StoreRelaxed(0);

  EXPECT_EQ(fake_queue->LoadWriteIndexRelaxed(), 2u);
  ExpectNoopMetadata(fake_queue->MetadataAt(0));
  ExpectMetadataEq(fake_queue->MetadataAt(1), metadata);
}

TEST_F(InterceptQueueMetadataTest, SignalReplacementUpdatesEventId) {
  auto fake = std::make_unique<FakeQueue>();
  auto* fake_queue = fake.get();
  TestInterceptQueue queue(std::move(fake));

  amd_signal_t original_signal{};
  amd_signal_t replacement_signal{};
  SignalReplacement replacement{MakeSignal(replacement_signal, 99)};
  queue.AddInterceptor(ReplaceSignal, &replacement);

  const MetadataPacket metadata = MakeMetadata(0x33, 7);
  ProxyMetadataRing(queue)[0] = metadata;
  WriteProxyPacket(queue, 0, MakeDispatch(0x3000, MakeSignal(original_signal, 7)));
  queue.StoreWriteIndexRelaxed(1);
  queue.StoreRelaxed(0);

  MetadataPacket expected = metadata;
  expected.packet.event_id = 99;
  ExpectMetadataEq(fake_queue->MetadataAt(0), expected);
}

TEST_F(InterceptQueueMetadataTest, ReorderedPacketsFollowMetadata) {
  auto fake = std::make_unique<FakeQueue>();
  auto* fake_queue = fake.get();
  TestInterceptQueue queue(std::move(fake));
  queue.AddInterceptor(ReversePackets, nullptr);

  amd_signal_t first_signal{};
  amd_signal_t second_signal{};
  const MetadataPacket first = MakeMetadata(0x44, 10);
  const MetadataPacket second = MakeMetadata(0x55, 20);
  ProxyMetadataRing(queue)[0] = first;
  ProxyMetadataRing(queue)[1] = second;
  WriteProxyPacket(queue, 0, MakeDispatch(0x4000, MakeSignal(first_signal, 10)));
  WriteProxyPacket(queue, 1, MakeDispatch(0x5000, MakeSignal(second_signal, 20)));
  queue.StoreWriteIndexRelaxed(2);
  queue.StoreRelaxed(0);

  ExpectMetadataEq(fake_queue->MetadataAt(0), second);
  ExpectMetadataEq(fake_queue->MetadataAt(1), first);
}

TEST_F(InterceptQueueMetadataTest, MetadataInfoAttributesAreForwarded) {
  TestInterceptQueue queue(std::make_unique<FakeQueue>());

  const hsa_queue_info_attribute_t attributes[] = {
      HSA_AMD_QUEUE_INFO_PREFETCH_METADATA_DISPATCH_PKT_VERSION_MAJOR,
      HSA_AMD_QUEUE_INFO_PREFETCH_METADATA_DISPATCH_PKT_VERSION_MINOR,
      HSA_AMD_QUEUE_INFO_PREFETCH_METADATA_BARRIER_PKT_VERSION_MAJOR,
      HSA_AMD_QUEUE_INFO_PREFETCH_METADATA_BARRIER_PKT_VERSION_MINOR};
  const uint8_t expected[] = {0x11, 0x22, 0x33, 0x44};
  for (size_t i = 0; i < sizeof(attributes) / sizeof(attributes[0]); ++i) {
    uint8_t value = 0;
    EXPECT_EQ(queue.GetInfo(attributes[i], &value), HSA_STATUS_SUCCESS);
    EXPECT_EQ(value, expected[i]);
  }

  uint8_t properties[8] = {};
  EXPECT_EQ(queue.GetInfo(HSA_AMD_QUEUE_INFO_PROPERTIES, properties), HSA_STATUS_SUCCESS);
  for (uint8_t property : properties) EXPECT_EQ(property, 0x55);
}

TEST_F(InterceptQueueMetadataTest, DeletedPacketsDoNotShiftMetadata) {
  auto fake = std::make_unique<FakeQueue>();
  auto* fake_queue = fake.get();
  TestInterceptQueue queue(std::move(fake));
  queue.AddInterceptor(DropFirstPacket, nullptr);

  amd_signal_t first_signal{};
  amd_signal_t second_signal{};
  ProxyMetadataRing(queue)[0] = MakeMetadata(0x66, 30);
  const MetadataPacket expected = MakeMetadata(0x77, 40);
  ProxyMetadataRing(queue)[1] = expected;
  WriteProxyPacket(queue, 0, MakeDispatch(0x6000, MakeSignal(first_signal, 30)));
  WriteProxyPacket(queue, 1, MakeDispatch(0x7000, MakeSignal(second_signal, 40)));
  queue.StoreWriteIndexRelaxed(2);
  queue.StoreRelaxed(0);

  EXPECT_EQ(fake_queue->LoadWriteIndexRelaxed(), 1u);
  ExpectMetadataEq(fake_queue->MetadataAt(0), expected);
}

TEST_F(InterceptQueueMetadataTest, ReplacedPacketsGetNoopMetadata) {
  auto fake = std::make_unique<FakeQueue>();
  auto* fake_queue = fake.get();
  TestInterceptQueue queue(std::move(fake));
  queue.AddInterceptor(ReplaceWithBarrier, nullptr);

  ProxyMetadataRing(queue)[0] = MakeMetadata(0x88, 0);
  WriteProxyPacket(queue, 0, MakeDispatch(0x8000));
  queue.StoreWriteIndexRelaxed(1);
  queue.StoreRelaxed(0);

  ExpectNoopMetadata(fake_queue->MetadataAt(0));
}

TEST_F(InterceptQueueMetadataTest, MutableHeaderFlagsDoNotBreakMatching) {
  auto fake = std::make_unique<FakeQueue>();
  auto* fake_queue = fake.get();
  TestInterceptQueue queue(std::move(fake));
  queue.AddInterceptor(ChangeHeaderFlags, nullptr);

  amd_signal_t signal{};
  const MetadataPacket metadata = MakeMetadata(0x99, 50);
  ProxyMetadataRing(queue)[0] = metadata;
  WriteProxyPacket(queue, 0, MakeDispatch(0x9000, MakeSignal(signal, 50)));
  queue.StoreWriteIndexRelaxed(1);
  queue.StoreRelaxed(0);

  ExpectMetadataEq(fake_queue->MetadataAt(0), metadata);
}

TEST_F(InterceptQueueMetadataTest, CompletionSignalDoesNotBreakMatching) {
  auto fake = std::make_unique<FakeQueue>();
  auto* fake_queue = fake.get();
  TestInterceptQueue queue(std::move(fake));

  amd_signal_t original_signal{};
  amd_signal_t replacement_signal{};
  SignalReplacement replacement{MakeSignal(replacement_signal, 61)};
  queue.AddInterceptor(ReplaceSignal, &replacement);

  const MetadataPacket metadata = MakeMetadata(0xaa, 60);
  ProxyMetadataRing(queue)[0] = metadata;
  WriteProxyPacket(queue, 0, MakeDispatch(0xa000, MakeSignal(original_signal, 60)));
  queue.StoreWriteIndexRelaxed(1);
  queue.StoreRelaxed(0);

  MetadataPacket expected = metadata;
  expected.packet.event_id = 61;
  ExpectMetadataEq(fake_queue->MetadataAt(0), expected);
}

TEST_F(InterceptQueueMetadataTest, PacketTypeMismatchDoesNotMatch) {
  auto fake = std::make_unique<FakeQueue>();
  auto* fake_queue = fake.get();
  TestInterceptQueue queue(std::move(fake));
  queue.AddInterceptor(ChangePacketType, nullptr);

  ProxyMetadataRing(queue)[0] = MakeMetadata(0xbb, 0);
  WriteProxyPacket(queue, 0, MakeDispatch(0xb000));
  queue.StoreWriteIndexRelaxed(1);
  queue.StoreRelaxed(0);

  ExpectNoopMetadata(fake_queue->MetadataAt(0));
}

TEST_F(InterceptQueueMetadataTest, ReservedHeaderBitsDoNotMatch) {
  auto fake = std::make_unique<FakeQueue>();
  auto* fake_queue = fake.get();
  TestInterceptQueue queue(std::move(fake));
  queue.AddInterceptor(ChangeReservedHeaderBit, nullptr);

  ProxyMetadataRing(queue)[0] = MakeMetadata(0xbc, 0);
  WriteProxyPacket(queue, 0, MakeDispatch(0xb100));
  queue.StoreWriteIndexRelaxed(1);
  queue.StoreRelaxed(0);

  ExpectNoopMetadata(fake_queue->MetadataAt(0));
}

TEST_F(InterceptQueueMetadataTest, ZeroCountSubmissionProducesNoOutput) {
  auto fake = std::make_unique<FakeQueue>();
  auto* fake_queue = fake.get();
  TestInterceptQueue queue(std::move(fake));
  queue.AddInterceptor(WriteNothing, nullptr);

  ProxyMetadataRing(queue)[0] = MakeMetadata(0xcc, 0);
  WriteProxyPacket(queue, 0, MakeDispatch(0xc000));
  queue.StoreWriteIndexRelaxed(1);
  queue.StoreRelaxed(0);

  EXPECT_EQ(fake_queue->LoadWriteIndexRelaxed(), 0u);
}

TEST_F(InterceptQueueMetadataTest, MultipleDoorbellBatchesPreserveMetadata) {
  auto fake = std::make_unique<FakeQueue>();
  auto* fake_queue = fake.get();
  TestInterceptQueue queue(std::move(fake));
  queue.AddInterceptor(PassThrough, nullptr);

  const MetadataPacket first = MakeMetadata(0xdd, 0);
  ProxyMetadataRing(queue)[0] = first;
  WriteProxyPacket(queue, 0, MakeDispatch(0xd000));
  queue.StoreWriteIndexRelaxed(1);
  queue.StoreRelaxed(0);
  ExpectMetadataEq(fake_queue->MetadataAt(0), first);

  fake_queue->Consume(1);
  const MetadataPacket second = MakeMetadata(0xee, 0);
  ProxyMetadataRing(queue)[1] = second;
  WriteProxyPacket(queue, 1, MakeDispatch(0xe000));
  queue.StoreWriteIndexRelaxed(2);
  queue.StoreRelaxed(0);
  ExpectMetadataEq(fake_queue->MetadataAt(1), second);
}

TEST_F(InterceptQueueMetadataTest, RingWraparoundUsesCurrentProxyMetadata) {
  auto fake = std::make_unique<FakeQueue>();
  auto* fake_queue = fake.get();
  TestInterceptQueue queue(std::move(fake));
  queue.AddInterceptor(PassThrough, nullptr);

  for (uint64_t i = 0; i < kQueueSize - 1; ++i) {
    ProxyMetadataRing(queue)[i] = MakeMetadata(static_cast<uint8_t>(i), 0);
    WriteProxyPacket(queue, i, MakeDispatch(0xf000 + i));
  }
  queue.StoreWriteIndexRelaxed(kQueueSize - 1);
  queue.StoreRelaxed(0);
  fake_queue->Consume(kQueueSize - 1);

  const MetadataPacket before_wrap = MakeMetadata(0x12, 0);
  const MetadataPacket after_wrap = MakeMetadata(0x34, 0);
  ProxyMetadataRing(queue)[kQueueSize - 1] = before_wrap;
  ProxyMetadataRing(queue)[0] = after_wrap;
  WriteProxyPacket(queue, kQueueSize - 1, MakeDispatch(0x10000));
  WriteProxyPacket(queue, kQueueSize, MakeDispatch(0x10001));
  queue.StoreWriteIndexRelaxed(kQueueSize + 1);
  queue.StoreRelaxed(0);

  ExpectMetadataEq(fake_queue->MetadataAt(kQueueSize - 1), before_wrap);
  ExpectMetadataEq(fake_queue->MetadataAt(kQueueSize), after_wrap);
}

TEST_F(InterceptQueueMetadataTest, ReusedProxySlotsRequireFreshMetadata) {
  auto fake = std::make_unique<FakeQueue>();
  auto* fake_queue = fake.get();
  TestInterceptQueue queue(std::move(fake));
  queue.AddInterceptor(ReplaceWithBarrier, nullptr);

  const MetadataPacket stale_metadata = MakeMetadata(0x39, 0);
  for (uint64_t i = 0; i < kQueueSize; ++i) {
    if (i == 0)
      ProxyMetadataRing(queue)[i] = stale_metadata;
    else
      ProxyMetadataRing(queue)[i] = NoopMetadata();
    WriteProxyPacket(queue, i, MakeDispatch(0x10100 + i));
    queue.StoreWriteIndexRelaxed(i + 1);
    queue.StoreRelaxed(0);
    fake_queue->Consume(1);
  }

  // Producers with metadata must write it before publishing the AQL header.
  // This reused barrier intentionally has no fresh metadata.
  WriteProxyPacket(queue, kQueueSize, MakeBarrier());
  queue.StoreWriteIndexRelaxed(kQueueSize + 1);
  queue.StoreRelaxed(0);

  ExpectNoopMetadata(fake_queue->MetadataAt(kQueueSize));
}

TEST_F(InterceptQueueMetadataTest, OverflowRetryPreservesMetadata) {
  auto fake = std::make_unique<FakeQueue>();
  auto* fake_queue = fake.get();
  fake_queue->StoreWriteIndexRelaxed(6);
  TestInterceptQueue queue(std::move(fake));
  queue.AddInterceptor(ExpandPacket, nullptr);

  const MetadataPacket metadata = MakeMetadata(0x45, 0);
  ProxyMetadataRing(queue)[0] = metadata;
  WriteProxyPacket(queue, 0, MakeDispatch(0x11000));
  queue.StoreWriteIndexRelaxed(1);
  queue.StoreRelaxed(0);

  EXPECT_EQ(fake_queue->LoadWriteIndexRelaxed(), 7u);
  ExpectNoopMetadata(fake_queue->MetadataAt(6));

  fake_queue->Consume(7);
  queue.StoreRelaxed(0);

  EXPECT_EQ(fake_queue->LoadWriteIndexRelaxed(), 10u);
  ExpectNoopMetadata(fake_queue->MetadataAt(7));
  ExpectMetadataEq(fake_queue->MetadataAt(8), metadata);
  ExpectNoopMetadata(fake_queue->MetadataAt(9));
}

TEST_F(InterceptQueueMetadataTest, UnsupportedMetadataRingIsPassedThrough) {
  auto fake = std::make_unique<FakeQueue>(false);
  auto* fake_queue = fake.get();
  TestInterceptQueue queue(std::move(fake));

  uint64_t metadata_ring = 0xfeedface;
  EXPECT_EQ(queue.GetInfo(HSA_AMD_QUEUE_INFO_PREFETCH_METADATA_RING_BUFFER, &metadata_ring),
            HSA_STATUS_ERROR_INVALID_ARGUMENT);
  EXPECT_EQ(metadata_ring, 0xfeedfaceu);

  WriteProxyPacket(queue, 0, MakeDispatch(0x12000));
  queue.StoreWriteIndexRelaxed(1);
  queue.StoreRelaxed(0);

  EXPECT_EQ(fake_queue->LoadWriteIndexRelaxed(), 1u);
  ExpectNoopMetadata(fake_queue->MetadataAt(0));
}

TEST_F(InterceptQueueMetadataTest, NullCompletionSignalClearsEventId) {
  auto fake = std::make_unique<FakeQueue>();
  auto* fake_queue = fake.get();
  TestInterceptQueue queue(std::move(fake));

  ProxyMetadataRing(queue)[0] = MakeMetadata(0x56, 0x1234);
  WriteProxyPacket(queue, 0, MakeDispatch(0x13000));
  queue.StoreWriteIndexRelaxed(1);
  queue.StoreRelaxed(0);

  MetadataPacket expected = MakeMetadata(0x56, 0);
  ExpectMetadataEq(fake_queue->MetadataAt(0), expected);
}

TEST_F(InterceptQueueMetadataTest, VendorPacketSignalReplacementUpdatesEventId) {
  auto fake = std::make_unique<FakeQueue>();
  auto* fake_queue = fake.get();
  TestInterceptQueue queue(std::move(fake));

  amd_signal_t original_signal{};
  amd_signal_t replacement_signal{};
  SignalReplacement replacement{MakeSignal(replacement_signal, 91)};
  queue.AddInterceptor(ReplaceSignal, &replacement);

  const MetadataPacket metadata = MakeMetadata(0x57, 90);
  ProxyMetadataRing(queue)[0] = metadata;
  WriteProxyPacket(queue, 0, MakeExtDispatch(MakeSignal(original_signal, 90)));
  queue.StoreWriteIndexRelaxed(1);
  queue.StoreRelaxed(0);

  MetadataPacket expected = metadata;
  expected.packet.event_id = 91;
  ExpectMetadataEq(fake_queue->MetadataAt(0), expected);

  fake_queue->Consume(1);
  ProxyMetadataRing(queue)[1] = metadata;
  WriteProxyPacket(queue, 1, MakeBarrierValue(MakeSignal(original_signal, 90)));
  queue.StoreWriteIndexRelaxed(2);
  queue.StoreRelaxed(0);
  ExpectMetadataEq(fake_queue->MetadataAt(1), expected);
}

TEST_F(InterceptQueueMetadataTest, NullVendorCompletionSignalsClearEventId) {
  auto fake = std::make_unique<FakeQueue>();
  auto* fake_queue = fake.get();
  TestInterceptQueue queue(std::move(fake));

  ProxyMetadataRing(queue)[0] = MakeMetadata(0x58, 0x1234);
  ProxyMetadataRing(queue)[1] = MakeMetadata(0x59, 0x5678);
  WriteProxyPacket(queue, 0, MakeExtDispatch());
  WriteProxyPacket(queue, 1, MakeBarrierValue());
  queue.StoreWriteIndexRelaxed(2);
  queue.StoreRelaxed(0);

  ExpectMetadataEq(fake_queue->MetadataAt(0), MakeMetadata(0x58, 0));
  ExpectMetadataEq(fake_queue->MetadataAt(1), MakeMetadata(0x59, 0));
}

TEST_F(InterceptQueueMetadataTest, AgentPacketsDoNotDereferenceCompletionSignals) {
  auto fake = std::make_unique<FakeQueue>();
  auto* fake_queue = fake.get();
  TestInterceptQueue queue(std::move(fake));

  const MetadataPacket metadata = MakeMetadata(0x5a, 0x1234);
  ProxyMetadataRing(queue)[0] = metadata;
  WriteProxyPacket(queue, 0, MakeAgentDispatch());
  queue.StoreWriteIndexRelaxed(1);
  queue.StoreRelaxed(0);

  ExpectMetadataEq(fake_queue->MetadataAt(0), metadata);
}

TEST_F(InterceptQueueMetadataTest, ReorderedNormalizedPacketsFollowMetadata) {
  auto fake = std::make_unique<FakeQueue>();
  auto* fake_queue = fake.get();
  TestInterceptQueue queue(std::move(fake));
  queue.AddInterceptor(ReversePackets, nullptr);

  amd_signal_t first_signal{};
  amd_signal_t second_signal{};
  const MetadataPacket first = MakeMetadata(0x67, 71);
  const MetadataPacket second = MakeMetadata(0x78, 72);
  ProxyMetadataRing(queue)[0] = first;
  ProxyMetadataRing(queue)[1] = second;
  WriteProxyPacket(queue, 0, MakeDispatch(0x14000, MakeSignal(first_signal, 71)));
  WriteProxyPacket(queue, 1, MakeDispatch(0x14000, MakeSignal(second_signal, 72)));
  queue.StoreWriteIndexRelaxed(2);
  queue.StoreRelaxed(0);

  ExpectMetadataEq(fake_queue->MetadataAt(0), second);
  ExpectMetadataEq(fake_queue->MetadataAt(1), first);
}

TEST_F(InterceptQueueMetadataTest, MarkerPacketsDoNotConsumeMetadataSlots) {
  marker_callback_fired = false;
  marker_packet_index = 0;

  auto fake = std::make_unique<FakeQueue>();
  auto* fake_queue = fake.get();
  TestInterceptQueue queue(std::move(fake));
  queue.AddInterceptor(InsertMarker, nullptr);

  amd_signal_t signal{};
  const MetadataPacket metadata = MakeMetadata(0x89, 81);
  ProxyMetadataRing(queue)[0] = metadata;
  WriteProxyPacket(queue, 0, MakeDispatch(0x15000, MakeSignal(signal, 81)));
  queue.StoreWriteIndexRelaxed(1);
  queue.StoreRelaxed(0);

  EXPECT_TRUE(marker_callback_fired);
  EXPECT_EQ(marker_packet_index, 0u);
  EXPECT_EQ(fake_queue->LoadWriteIndexRelaxed(), 1u);
  ExpectMetadataEq(fake_queue->MetadataAt(0), metadata);
}

}  // namespace
