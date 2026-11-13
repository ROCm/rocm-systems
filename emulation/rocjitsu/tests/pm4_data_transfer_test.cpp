// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "async_vm_test_support.h"
#include "rocjitsu/vm/amdgpu/dispatch_entry.h"
#include "rocjitsu/vm/amdgpu/hsa_clock.h"
#include "rocjitsu/vm/amdgpu/pm4/pm4_packet_processor.h"

#include <gtest/gtest.h>

#include <array>
#include <cstring>
#include <span>
#include <vector>

namespace rocjitsu::amdgpu {
namespace {

enum class SubmissionKind { NativeRing, Drm, VendorIb };

class DeferredPm4Fixture {
public:
  DeferredPm4Fixture() {
    sim.engine.register_as_primary();
    memory = std::make_shared<test::AsyncVmMemory>(*sim.owner, 0x3000, 0x1000, 0x2000);
    queue.address_space = vm.register_translated(1, memory, memory);
    queue.process_id = 1;
    queue.packet_format = QueuePacketFormat::Pm4;
    context.arch = ROCJITSU_CODE_ARCH_RDNA4;
    context.flush_caches = [] { return VmAccessOutcome::Complete; };
    context.retry = [&] { queue.command_retry_pending = true; };
    context.wake = [] {};
    context.fault_queue = [&] { queue.faulted = true; };
  }

  ~DeferredPm4Fixture() { sim.engine.primary_release(); }

  void submit(std::span<const uint32_t> words, SubmissionKind kind) {
    const bool ring = kind == SubmissionKind::NativeRing;
    queue.submission_queue = !ring;
    queue.read_pointer_journal = ConsumerCursorJournal(kReadPointer, kInitialCursor, 4);
    for (size_t i = 0; i < words.size(); ++i) {
      const uint64_t offset = ring ? (kInitialCursor + i) % kRingDwords : i;
      memory->store(kRing + offset * 4, std::as_bytes(words.subspan(i, 1)));
    }
    Pm4Submission submission;
    submission.graphics_engine = true;
    submission.allow_dispatch = kind != SubmissionKind::VendorIb;
    submission.buffers.push_back({.address = ring ? kInitialCursor * 4 : kRing,
                                  .dwords = static_cast<uint32_t>(words.size()),
                                  .ring_base = ring ? kRing : 0,
                                  .ring_bytes = ring ? kRingDwords * 4 : 0});
    submission.complete = [&](bool success) {
      EXPECT_TRUE(success);
      ++completions;
    };
    queue.commands.submissions.push_back(std::move(submission));
  }

  void service() {
    queue.command_retry_pending = false;
    process_pm4_packets(queue, &vm, context);
  }

  void finish() {
    while (!completions && !queue.faulted) {
      service();
      if (!completions && !queue.faulted) {
        EXPECT_EQ(queue.read_pointer_journal.cursor(), kInitialCursor);
        ASSERT_TRUE(sim.engine.step());
      }
    }
    EXPECT_FALSE(queue.faulted);
    EXPECT_EQ(completions, 1u);
  }

  static constexpr uint64_t kRing = 0x100;
  static constexpr uint64_t kReadPointer = 0x80;
  static constexpr uint32_t kRingDwords = 16;
  static constexpr uint64_t kInitialCursor = 14;
  // The owner engine outlives every pending request.
  test::AsyncVmHarness sim;
  // Control addresses are local; data in [0x1000, 0x2000) needs owned requests.
  std::shared_ptr<test::AsyncVmMemory> memory;
  // Production translation and deferred-transfer ownership.
  GpuVm vm;
  // Production PM4 submission, pinned snapshot, and cursor state.
  ComputeQueueRecord queue;
  // Synchronous CP services for each interpreter turn.
  Pm4ExecutionContext context;
  // Successful submission callbacks, which must occur once.
  uint32_t completions = 0;
};

constexpr std::array kSubmissionKinds{SubmissionKind::NativeRing, SubmissionKind::Drm,
                                      SubmissionKind::VendorIb};

TEST(Pm4DataTransferTest, DmaSelectorsResumeLocalAndDeferredReadWritePrefixes) {
  for (auto kind : kSubmissionKinds)
    for (uint32_t source_select : {0u, 2u, 3u})
      for (uint32_t destination_select : {0u, 3u})
        for (uint32_t source : {0x800u, 0x1100u})
          for (uint32_t destination : {0x900u, 0x1200u})
            for (uint32_t bytes : {1u, 4u, 7u, 16u}) {
              SCOPED_TRACE(::testing::Message()
                           << int(kind) << ':' << source_select << ':' << destination_select << ':'
                           << source << ':' << destination << ':' << bytes);
              DeferredPm4Fixture fixture;
              constexpr uint32_t kFill = 0x78563412;
              std::vector<std::byte> expected(bytes);
              for (uint32_t i = 0; i < bytes; ++i)
                expected[i] =
                    source_select == 2 ? std::byte(kFill >> ((i % 4) * 8)) : std::byte(i + 1);
              fixture.memory->store(source, expected);
              const std::array packet{0xc0055000u,
                                      (source_select << 29) | (destination_select << 20),
                                      source_select == 2 ? kFill : source,
                                      0u,
                                      destination,
                                      0u,
                                      bytes};
              fixture.submit(packet, kind);
              fixture.finish();
              EXPECT_EQ(fixture.memory->load(destination, bytes), expected);
              EXPECT_EQ(fixture.memory->backing->read_effects,
                        source >= 0x1000 && source_select != 2 ? (bytes + 3) / 4 : 0u);
              EXPECT_EQ(fixture.memory->backing->write_effects,
                        destination >= 0x1000 ? (bytes + 3) / 4 : 0u);
              if (kind == SubmissionKind::NativeRing) {
                EXPECT_EQ(fixture.queue.read_pointer_journal.cursor(),
                          DeferredPm4Fixture::kInitialCursor + packet.size());
              }
            }
}

TEST(Pm4DataTransferTest, CopyDataCapturesEverySourceAndWidthOnce) {
  for (auto kind : kSubmissionKinds)
    for (uint32_t source_select : {0u, 1u, 2u, 5u, 9u})
      for (uint32_t destination_select : {1u, 2u, 5u})
        for (uint32_t bytes : {4u, 8u}) {
          SCOPED_TRACE(::testing::Message() << int(kind) << ':' << source_select << ':'
                                            << destination_select << ':' << bytes);
          DeferredPm4Fixture fixture;
          constexpr uint64_t kValue = 0x78563412deadbeef;
          const auto expected = std::as_bytes(std::span(&kValue, 1)).first(bytes);
          fixture.memory->store(0x1100, expected);
          fixture.queue.commands.sh_registers[0] = static_cast<uint32_t>(kValue);
          fixture.queue.commands.sh_registers[1] = kValue >> 32;
          const uint64_t source = source_select == 0   ? 0x2c00
                                  : source_select == 5 ? kValue
                                                       : 0x1100;
          const std::array packet{0xc0044000u,
                                  source_select | (destination_select << 8) |
                                      (bytes == 8 ? 1u << 16 : 0),
                                  static_cast<uint32_t>(source),
                                  uint32_t(source >> 32),
                                  0x1200u,
                                  0u};
          fixture.submit(packet, kind);
          const uint64_t before = gpu_timestamp(fixture.context.arch);
          fixture.service();
          const uint64_t after = gpu_timestamp(fixture.context.arch);
          ASSERT_FALSE(fixture.queue.faulted);
          ASSERT_EQ(fixture.memory->backing->posts, 1u);
          EXPECT_EQ(fixture.queue.read_pointer_journal.cursor(),
                    DeferredPm4Fixture::kInitialCursor);
          // Neither repeated polling nor later register changes replay admission.
          fixture.queue.commands.sh_registers[0] = 0;
          fixture.queue.commands.sh_registers[1] = 0;
          fixture.service();
          EXPECT_EQ(fixture.memory->backing->posts, 1u);
          fixture.finish();
          const auto result = fixture.memory->load(0x1200, bytes);
          if (source_select == 9) {
            uint64_t value = 0;
            std::memcpy(&value, result.data(), bytes);
            const uint64_t mask = bytes == 4 ? UINT32_MAX : UINT64_MAX;
            // Modular subtraction covers the 32-bit clock's wrap too.
            EXPECT_LE((value - before) & mask, after - before);
          } else {
            EXPECT_EQ(result, std::vector<std::byte>(expected.begin(), expected.end()));
          }
          EXPECT_EQ(fixture.memory->backing->read_effects,
                    source_select == 1 || source_select == 2 ? bytes / 4 : 0u);
          EXPECT_EQ(fixture.memory->backing->write_effects, bytes / 4);
        }
}

TEST(Pm4DataTransferTest, WriteDataOwnsPayloadAndRetiresAfterTheLastChunk) {
  for (auto kind : kSubmissionKinds)
    for (uint32_t destination_select : {1u, 2u, 5u}) {
      DeferredPm4Fixture fixture;
      const std::array packet{
          0xc0053700u, destination_select << 8, 0x1200u, 0u, 0x12345678u, 0x87654321u, 0xdeadbeefu};
      fixture.submit(packet, kind);
      fixture.service();
      ASSERT_FALSE(fixture.queue.faulted);
      EXPECT_EQ(fixture.memory->backing->posts, 1u);
      EXPECT_EQ(fixture.memory->backing->write_effects, 0u);
      const std::vector<std::byte> cleared_ring(DeferredPm4Fixture::kRingDwords * 4);
      fixture.memory->store(DeferredPm4Fixture::kRing, cleared_ring);
      fixture.finish();
      const auto expected = std::as_bytes(std::span(packet).subspan(4));
      EXPECT_EQ(fixture.memory->load(0x1200, expected.size()),
                std::vector<std::byte>(expected.begin(), expected.end()));
      EXPECT_EQ(fixture.memory->backing->write_effects, 3u);
    }
}

TEST(Pm4DataTransferTest, RevocationRetainsTheWrittenPrefixWithoutRetiringThePacket) {
  DeferredPm4Fixture fixture;
  const std::array packet{0xc0053700u, 5u << 8, 0x1200u, 0u, 0x12345678u, 0x87654321u, 0xdeadbeefu};
  fixture.submit(packet, SubmissionKind::NativeRing);
  fixture.service();
  ASSERT_FALSE(fixture.queue.faulted);
  ASSERT_EQ(fixture.memory->backing->posts, 1u);
  (void)fixture.sim.engine.step();
  fixture.service();
  ASSERT_EQ(fixture.memory->backing->write_effects, 1u);
  ASSERT_TRUE(fixture.vm.unregister_address_space(fixture.queue.address_space));
  fixture.service();
  EXPECT_TRUE(fixture.queue.faulted);
  EXPECT_EQ(fixture.completions, 0u);
  EXPECT_EQ(fixture.queue.read_pointer_journal.cursor(), DeferredPm4Fixture::kInitialCursor);
  std::array<uint32_t, 3> expected{packet[4], 0, 0};
  const auto expected_bytes = std::as_bytes(std::span(expected));
  EXPECT_EQ(fixture.memory->load(0x1200, expected_bytes.size()),
            std::vector<std::byte>(expected_bytes.begin(), expected_bytes.end()));
}

TEST(Pm4DataTransferTest, MixedRoutesRetainTheAlreadyReadLocalPrefix) {
  DeferredPm4Fixture fixture;
  constexpr uint32_t kSource = 0xffc;
  constexpr uint32_t kDestination = 0x1ffc;
  const std::array<uint32_t, 3> values{0x12345678u, 0x87654321u, 0xdeadbeefu};
  const auto bytes = std::as_bytes(std::span(values));
  fixture.memory->store(kSource, bytes);
  const std::array packet{
      0xc0055000u, 0u, kSource, 0u, kDestination, 0u, static_cast<uint32_t>(bytes.size())};
  fixture.submit(packet, SubmissionKind::NativeRing);
  fixture.service();
  ASSERT_FALSE(fixture.queue.faulted);
  ASSERT_EQ(fixture.memory->backing->posts, 1u);
  const uint32_t changed_prefix = 0;
  fixture.memory->store(kSource, std::as_bytes(std::span(&changed_prefix, 1)));
  fixture.finish();
  EXPECT_EQ(fixture.memory->load(kDestination, bytes.size()),
            std::vector<std::byte>(bytes.begin(), bytes.end()));
  EXPECT_EQ(fixture.memory->backing->read_effects, 2u);
  EXPECT_EQ(fixture.memory->backing->write_effects, 1u);
}

} // namespace
} // namespace rocjitsu::amdgpu
