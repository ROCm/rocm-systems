/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

// abce_device.h's single-lane path on the host, against plain-memory rings:
// signals, dependencies, handles and inline writes.

#include "abce_device.h"

#include <gtest/gtest.h>

#include <sched.h>

#include <atomic>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "abce_test_harness.hpp"

namespace {

using abce_test::FakeAddress;
using abce_test::IsaByName;
using abce_test::IsaCase;
using abce_test::Walk;
using abce_test::WalkResult;

using Names = std::vector<std::string>;

constexpr uint32_t kRingBytes = 1u << 16;
constexpr uint32_t kRingPoison = 0xdeadbeefu;

struct DeviceViewOptions {
  uint32_t num_engines = 2;
  bool device_atomic_support = true;
};

// Device engines whose rings are plain memory, one handle each. The engine never
// runs, so each ring holds exactly what the submitter wrote, from index 0.
class DeviceView {
 public:
  explicit DeviceView(const IsaCase& isa_case,
                      const DeviceViewOptions& options = DeviceViewOptions())
      : isa_case_(isa_case), rings_(options.num_engines), engines_(options.num_engines),
        handles_(options.num_engines) {
    abce_builder_t builder;
    if (!abce_status_is_ok(abce_builder_initialize(isa_case.isa, nullptr, &builder))) std::abort();
    for (uint32_t engine_idx = 0; engine_idx < options.num_engines; ++engine_idx) {
      RingMemory& memory = rings_[engine_idx];
      memory.buffer.assign(kRingBytes / sizeof(uint32_t), kRingPoison);
      abce_device_ring_t ring{};
      ring.base = memory.buffer.data();
      ring.size = kRingBytes;
      ring.read_ptr = &memory.read_index;
      ring.write_ptr = &memory.write_index;
      ring.doorbell = &memory.doorbell;
      ring.control = &memory.control;
      if (!abce_status_is_ok(abce_device_engine_initialize(
              &builder, &ring, engine_idx, options.device_atomic_support, &engines_[engine_idx])))
        std::abort();
      handles_[engine_idx].engine = &engines_[engine_idx];
      handles_[engine_idx].abi_version = ABCE_DEVICE_ABI_VERSION;
    }
  }

  const abce_device_handle_t* handle(uint32_t engine_idx = 0) const {
    return &handles_[engine_idx];
  }
  abce_device_engine_t& engine(uint32_t engine_idx) { return engines_[engine_idx]; }
  uint64_t write_index(uint32_t engine_idx) const { return rings_[engine_idx].write_index; }
  uint64_t doorbell(uint32_t engine_idx) const { return rings_[engine_idx].doorbell; }
  // The engine has consumed everything published so far.
  void EngineConsumeAll(uint32_t engine_idx) {
    rings_[engine_idx].read_index = rings_[engine_idx].write_index;
  }
  const uint32_t* RingWords(uint32_t engine_idx) const { return rings_[engine_idx].buffer.data(); }
  // For a fake engine on another thread.
  uint32_t* MutableRingWords(uint32_t engine_idx) { return rings_[engine_idx].buffer.data(); }
  uint64_t* DoorbellWord(uint32_t engine_idx) { return &rings_[engine_idx].doorbell; }
  uint64_t* ReadIndexWord(uint32_t engine_idx) { return &rings_[engine_idx].read_index; }
  const abce_ring_control_t& Control(uint32_t engine_idx) const {
    return rings_[engine_idx].control;
  }

  std::vector<uint8_t> Bytes(uint32_t engine_idx) const {
    const uint8_t* base = reinterpret_cast<const uint8_t*>(rings_[engine_idx].buffer.data());
    return std::vector<uint8_t>(base, base + rings_[engine_idx].write_index);
  }

  WalkResult WalkRing(uint32_t engine_idx) const {
    const std::vector<uint8_t> bytes = Bytes(engine_idx);
    return Walk(bytes.data(), bytes.size(), isa_case_.isa);
  }

  template <typename Packet>
  Packet PacketAt(uint32_t engine_idx, size_t byte_offset) const {
    Packet packet;
    std::memcpy(&packet, reinterpret_cast<const uint8_t*>(rings_[engine_idx].buffer.data()) +
                             byte_offset,
                sizeof(packet));
    return packet;
  }

  uint32_t DwordAt(uint32_t engine_idx, size_t byte_offset) const {
    return PacketAt<uint32_t>(engine_idx, byte_offset);
  }

 private:
  struct RingMemory {
    std::vector<uint32_t> buffer;
    uint64_t read_index = 0;
    uint64_t write_index = 0;
    uint64_t doorbell = 0;
    abce_ring_control_t control{};
  };

  IsaCase isa_case_;
  std::vector<RingMemory> rings_;
  std::vector<abce_device_engine_t> engines_;
  std::vector<abce_device_handle_t> handles_;
};

struct alignas(8) SignalWord {
  uint64_t value = 0;
};

abce_status_t SubmitCopy(const DeviceView& view, const abce_device_signal_t* completion,
                         const abce_device_wait_t* dependency = nullptr, size_t size = 4096,
                         uint32_t engine_idx = 0, uint32_t flags = 0) {
  return abce_device_submit_linear(view.handle(engine_idx), FakeAddress(0x20000000),
                                   FakeAddress(0x40000000), size, dependency, completion, flags);
}

abce_status_t SubmitWrite(const DeviceView& view, void* dst, uint64_t value, uint32_t size,
                          const abce_device_signal_t* completion = nullptr,
                          const abce_device_wait_t* dependency = nullptr) {
  return abce_device_submit_write(view.handle(), dst, value, size, dependency, completion, 0);
}

abce_device_signal_t SignalOn(SignalWord* word, uint64_t value, abce_device_signal_op_t op,
                          uint32_t flags = 0) {
  return abce_device_signal_t{&word->value, value, static_cast<uint32_t>(op), flags};
}

uint64_t Join(uint32_t high, uint32_t low) { return (uint64_t{high} << 32) | low; }

//===----------------------------------------------------------------------===//
// Completion
//===----------------------------------------------------------------------===//

TEST(DeviceCompletion, AddOnGfx9IsAnAtomicAfterTheCopy) {
  DeviceView view(IsaByName("gfx942"));
  SignalWord signal;
  const abce_device_signal_t completion = SignalOn(&signal, 3, ABCE_DEVICE_SIGNAL_ADD);
  ASSERT_EQ(SubmitCopy(view, &completion), ABCE_STATUS_OK);

  const WalkResult walk = view.WalkRing(0);
  ASSERT_TRUE(walk.clean);
  ASSERT_EQ(walk.Names(), (Names{"COPY_LINEAR", "ATOMIC"}));
  const auto atomic = view.PacketAt<SDMA_PKT_ATOMIC>(0, walk.packets[1].byte_offset);
  EXPECT_EQ(atomic.HEADER_UNION.operation, SDMA_ATOMIC_ADD64);
  EXPECT_EQ(Join(atomic.ADDR_HI_UNION.addr_63_32, atomic.ADDR_LO_UNION.addr_31_0),
            reinterpret_cast<uintptr_t>(&signal.value));
  EXPECT_EQ(Join(atomic.SRC_DATA_HI_UNION.src_data_63_32, atomic.SRC_DATA_LO_UNION.src_data_31_0),
            3u);
}

TEST(DeviceCompletion, AnUnorderedAddOnGfx125PlusRidesInTheFusedPacket) {
  DeviceView view(IsaByName("gfx1250"));
  SignalWord signal;
  const uint64_t minus_one = ~uint64_t{0};
  const abce_device_signal_t completion =
      SignalOn(&signal, minus_one, ABCE_DEVICE_SIGNAL_ADD, ABCE_DEVICE_SIGNAL_UNORDERED);
  ASSERT_EQ(SubmitCopy(view, &completion), ABCE_STATUS_OK);

  const WalkResult walk = view.WalkRing(0);
  ASSERT_TRUE(walk.clean);
  ASSERT_EQ(walk.Names(), (Names{"COPY_LINEAR_WAITSIGNAL"}));
  EXPECT_EQ(walk.packets[0].dwords, 1u + 6u + 5u) << "header + body + signal, no wait block";
  // The signal block's operation is in its first dword's low 7 bits, followed
  // by the address and the 64-bit operand.
  const size_t signal_block = (1 + 6) * sizeof(uint32_t);
  EXPECT_EQ(view.DwordAt(0, signal_block) & 0x7fu, SDMA_SIGNAL_OP_ADD64);
  EXPECT_EQ(Join(view.DwordAt(0, signal_block + 16), view.DwordAt(0, signal_block + 12)),
            minus_one)
      << "a negative addend is how a caller subtracts";
}

TEST(DeviceCompletion, SubOnGfx9AddsTheNegatedValue) {
  DeviceView view(IsaByName("gfx942"));
  SignalWord signal;
  const abce_device_signal_t completion = SignalOn(&signal, 3, ABCE_DEVICE_SIGNAL_SUB);
  ASSERT_EQ(SubmitCopy(view, &completion), ABCE_STATUS_OK);
  const WalkResult walk = view.WalkRing(0);
  ASSERT_TRUE(walk.clean);
  ASSERT_EQ(walk.Names(), (Names{"COPY_LINEAR", "ATOMIC"}));
  const auto atomic = view.PacketAt<SDMA_PKT_ATOMIC>(0, walk.packets[1].byte_offset);
  EXPECT_EQ(atomic.HEADER_UNION.operation, SDMA_ATOMIC_ADD64);
  EXPECT_EQ(Join(atomic.SRC_DATA_HI_UNION.src_data_63_32, atomic.SRC_DATA_LO_UNION.src_data_31_0),
            uint64_t{0} - 3);
}

TEST(DeviceCompletion, AnUnorderedSubOnGfx125PlusIsTheFusedSub64) {
  DeviceView view(IsaByName("gfx1250"));
  SignalWord signal;
  const abce_device_signal_t completion =
      SignalOn(&signal, 3, ABCE_DEVICE_SIGNAL_SUB, ABCE_DEVICE_SIGNAL_UNORDERED);
  ASSERT_EQ(SubmitCopy(view, &completion), ABCE_STATUS_OK);
  const WalkResult walk = view.WalkRing(0);
  ASSERT_TRUE(walk.clean);
  ASSERT_EQ(walk.Names(), (Names{"COPY_LINEAR_WAITSIGNAL"}));
  const size_t signal_block = (1 + 6) * sizeof(uint32_t);
  EXPECT_EQ(view.DwordAt(0, signal_block) & 0x7fu, SDMA_SIGNAL_OP_SUB64);
  EXPECT_EQ(Join(view.DwordAt(0, signal_block + 16), view.DwordAt(0, signal_block + 12)), 3u);
}

// Ordered by default: the signal is a standalone packet after the copy, which
// lands after everything earlier on the ring.
TEST(DeviceCompletion, ByDefaultAnAddOnGfx125PlusIsAStandaloneAtomicAfterTheCopy) {
  for (const abce_device_signal_op_t op : {ABCE_DEVICE_SIGNAL_ADD, ABCE_DEVICE_SIGNAL_SUB}) {
    DeviceView view(IsaByName("gfx1250"));
    SignalWord signal;
    const abce_device_signal_t completion = SignalOn(&signal, 1, op);
    ASSERT_EQ(SubmitCopy(view, &completion), ABCE_STATUS_OK);
    const WalkResult walk = view.WalkRing(0);
    ASSERT_TRUE(walk.clean);
    EXPECT_EQ(walk.Names(), (Names{"COPY_LINEAR", "ATOMIC"})) << "no fused signal";
  }
  DeviceView gfx942(IsaByName("gfx942"));
  SignalWord signal;
  const abce_device_signal_t unordered =
      SignalOn(&signal, 1, ABCE_DEVICE_SIGNAL_ADD, ABCE_DEVICE_SIGNAL_UNORDERED);
  ASSERT_EQ(SubmitCopy(gfx942, &unordered), ABCE_STATUS_OK);
  EXPECT_EQ(gfx942.WalkRing(0).Names(), (Names{"COPY_LINEAR", "ATOMIC"}))
      << "UNORDERED changes nothing before gfx125";
}

TEST(DeviceCompletion, AnUnorderedSetOnGfx125PlusIsAFusedWrite64) {
  DeviceView view(IsaByName("gfx1250"));
  SignalWord signal;
  const uint64_t value = (uint64_t{7} << 32) | 9u;
  const abce_device_signal_t completion =
      SignalOn(&signal, value, ABCE_DEVICE_SIGNAL_SET, ABCE_DEVICE_SIGNAL_UNORDERED);
  ASSERT_EQ(SubmitCopy(view, &completion), ABCE_STATUS_OK);
  const WalkResult walk = view.WalkRing(0);
  ASSERT_TRUE(walk.clean);
  ASSERT_EQ(walk.Names(), (Names{"COPY_LINEAR_WAITSIGNAL"}));
  const size_t signal_block = (1 + 6) * sizeof(uint32_t);
  EXPECT_EQ(view.DwordAt(0, signal_block) & 0x7fu, SDMA_SIGNAL_OP_WRITE64);
  EXPECT_EQ(Join(view.DwordAt(0, signal_block + 16), view.DwordAt(0, signal_block + 12)), value);
}

// A chunked gfx125+ copy cannot fuse even an UNORDERED signal (each chunk's
// signal would cover that chunk alone), so it follows the chunks standalone.
TEST(DeviceCompletion, AChunkedGfx125PlusCopyEndsWithAStandaloneSignal) {
  const size_t two_gib = size_t{1} << 31;
  for (const abce_device_signal_op_t op : {ABCE_DEVICE_SIGNAL_ADD, ABCE_DEVICE_SIGNAL_SET}) {
    SCOPED_TRACE(op == ABCE_DEVICE_SIGNAL_ADD ? "ADD" : "SET");
    DeviceView view(IsaByName("gfx1250"));
    SignalWord signal;
    SignalWord dependency_word;
    const abce_device_wait_t dependency{&dependency_word.value, 1, ABCE_DEVICE_WAIT_GEQ};
    const abce_device_signal_t completion = SignalOn(&signal, 1, op, ABCE_DEVICE_SIGNAL_UNORDERED);
    ASSERT_EQ(SubmitCopy(view, &completion, &dependency, two_gib),
              ABCE_STATUS_OK);
    const WalkResult walk = view.WalkRing(0);
    ASSERT_TRUE(walk.clean);
    const std::string token_packet = op == ABCE_DEVICE_SIGNAL_ADD ? "ATOMIC" : "FENCE_64";
    EXPECT_EQ(walk.Names(),
              (Names{"COPY_LINEAR_WAITSIGNAL", "COPY_LINEAR", "COPY_LINEAR", token_packet}))
        << "wait on the first chunk only, no fused signal";
  }
}

TEST(DeviceCompletion, SetOnGfx9WritesTheLowWordFirst) {
  DeviceView view(IsaByName("gfx942"));
  SignalWord signal;
  const uint64_t value = (uint64_t{1} << 32) | 5u;
  const abce_device_signal_t completion = SignalOn(&signal, value, ABCE_DEVICE_SIGNAL_SET);
  ASSERT_EQ(SubmitCopy(view, &completion), ABCE_STATUS_OK);

  const WalkResult walk = view.WalkRing(0);
  ASSERT_TRUE(walk.clean);
  ASSERT_EQ(walk.Names(), (Names{"COPY_LINEAR", "FENCE", "FENCE"}));
  const auto low = view.PacketAt<SDMA_PKT_FENCE>(0, walk.packets[1].byte_offset);
  const auto high = view.PacketAt<SDMA_PKT_FENCE>(0, walk.packets[2].byte_offset);
  const uintptr_t signal_address = reinterpret_cast<uintptr_t>(&signal.value);
  EXPECT_EQ(Join(low.ADDR_HI_UNION.addr_63_32, low.ADDR_LO_UNION.addr_31_0), signal_address);
  EXPECT_EQ(low.DATA_UNION.data, 5u);
  EXPECT_EQ(Join(high.ADDR_HI_UNION.addr_63_32, high.ADDR_LO_UNION.addr_31_0),
            signal_address + 4);
  EXPECT_EQ(high.DATA_UNION.data, 1u);
}

TEST(DeviceCompletion, SetOnGfx125PlusIsOneFence64AfterACompactedCopy) {
  DeviceView view(IsaByName("gfx1250"));
  SignalWord signal;
  const abce_device_signal_t completion = SignalOn(&signal, 7, ABCE_DEVICE_SIGNAL_SET);
  ASSERT_EQ(SubmitCopy(view, &completion), ABCE_STATUS_OK);
  const WalkResult walk = view.WalkRing(0);
  ASSERT_TRUE(walk.clean);
  EXPECT_EQ(walk.Names(), (Names{"COPY_LINEAR", "FENCE_64"}));
  EXPECT_EQ(walk.packets[0].dwords, 7u);
}

TEST(DeviceCompletion, IsOptional) {
  for (const char* isa_name : {"gfx942", "gfx1250"}) {
    SCOPED_TRACE(isa_name);
    DeviceView view(IsaByName(isa_name));
    ASSERT_EQ(SubmitCopy(view, nullptr), ABCE_STATUS_OK);
    const WalkResult walk = view.WalkRing(0);
    ASSERT_TRUE(walk.clean);
    EXPECT_EQ(walk.Names(), (Names{"COPY_LINEAR"}));
    EXPECT_EQ(view.doorbell(0), view.write_index(0));
  }
}

TEST(DeviceCompletion, TheKernelNeverStoresToTheCallersSignal) {
  for (const char* isa_name : {"gfx942", "gfx1250"}) {
    for (const abce_device_signal_op_t op :
         {ABCE_DEVICE_SIGNAL_SET, ABCE_DEVICE_SIGNAL_ADD, ABCE_DEVICE_SIGNAL_SUB}) {
      SCOPED_TRACE(std::string(isa_name) + " op " + std::to_string(op));
      DeviceView view(IsaByName(isa_name));
      SignalWord signal;
      signal.value = 0x1234;
      const abce_device_signal_t completion = SignalOn(&signal, 1, op);
      ASSERT_EQ(SubmitCopy(view, &completion), ABCE_STATUS_OK);
      EXPECT_EQ(signal.value, 0x1234u) << "arming would race the engine's update";
    }
  }
}

TEST(DeviceCompletion, AddAndSubNeedDeviceAtomicsButSetDoesNot) {
  for (const char* isa_name : {"gfx942", "gfx1250"}) {
    SCOPED_TRACE(isa_name);
    DeviceViewOptions options;
    options.device_atomic_support = false;
    DeviceView view(IsaByName(isa_name), options);
    SignalWord signal;
    const abce_device_signal_t add = SignalOn(&signal, 1, ABCE_DEVICE_SIGNAL_ADD);
    EXPECT_EQ(SubmitCopy(view, &add), ABCE_STATUS_UNIMPLEMENTED);
    const abce_device_signal_t sub = SignalOn(&signal, 1, ABCE_DEVICE_SIGNAL_SUB);
    EXPECT_EQ(SubmitCopy(view, &sub), ABCE_STATUS_UNIMPLEMENTED);
    EXPECT_EQ(view.write_index(0), 0u) << "a refused submit must not move the ring";
    const abce_device_signal_t set = SignalOn(&signal, 1, ABCE_DEVICE_SIGNAL_SET);
    EXPECT_EQ(SubmitCopy(view, &set), ABCE_STATUS_OK);
  }
}

TEST(DeviceCompletion, RejectsAMisalignedWordAndAnUnknownOperation) {
  DeviceView view(IsaByName("gfx942"));
  alignas(8) uint64_t words[2] = {};
  abce_device_signal_t completion{reinterpret_cast<uint64_t*>(reinterpret_cast<char*>(words) + 4),
                                  1, ABCE_DEVICE_SIGNAL_SET, 0};
  EXPECT_EQ(SubmitCopy(view, &completion), ABCE_STATUS_INVALID_ARGUMENT);
  completion.address = &words[0];
  completion.op = 7;
  EXPECT_EQ(SubmitCopy(view, &completion), ABCE_STATUS_INVALID_ARGUMENT);
  completion.op = ABCE_DEVICE_SIGNAL_SET;
  completion.flags = 1u << 5;
  EXPECT_EQ(SubmitCopy(view, &completion), ABCE_STATUS_INVALID_ARGUMENT) << "unknown flag";
  EXPECT_EQ(view.write_index(0), 0u);
}

//===----------------------------------------------------------------------===//
// Dependency
//===----------------------------------------------------------------------===//

TEST(DeviceDependency, AnUnmetDependencyWaitsWithGreaterOrEqual) {
  {
    DeviceView view(IsaByName("gfx942"));
    SignalWord dependency_word;
    const abce_device_wait_t dependency{&dependency_word.value, 9, ABCE_DEVICE_WAIT_GEQ};
    ASSERT_EQ(SubmitCopy(view, nullptr, &dependency), ABCE_STATUS_OK);
    const WalkResult walk = view.WalkRing(0);
    ASSERT_TRUE(walk.clean);
    ASSERT_EQ(walk.Names(), (Names{"POLL_REGMEM", "COPY_LINEAR"}));
    const auto poll = view.PacketAt<SDMA_PKT_POLL_REGMEM>(0, walk.packets[0].byte_offset);
    EXPECT_EQ(poll.HEADER_UNION.func, SDMA_FUNC_GREATER_EQUAL);
    EXPECT_EQ(poll.VALUE_UNION.value, 9u);
    EXPECT_EQ(poll.MASK_UNION.mask, UINT32_MAX);
  }
  {
    DeviceView view(IsaByName("gfx1250"));
    SignalWord dependency_word;
    const abce_device_wait_t dependency{&dependency_word.value, uint64_t{1} << 40,
                                            ABCE_DEVICE_WAIT_GEQ};
    ASSERT_EQ(SubmitCopy(view, nullptr, &dependency), ABCE_STATUS_OK);
    const WalkResult walk = view.WalkRing(0);
    ASSERT_TRUE(walk.clean);
    ASSERT_EQ(walk.Names(), (Names{"COPY_LINEAR_WAITSIGNAL"}));
    EXPECT_EQ(walk.packets[0].dwords, 1u + 7u + 6u) << "header + wait + body, no signal";
    // The wait block follows the header: the compare function in its first
    // dword's low 3 bits, the address, then the 64-bit reference.
    EXPECT_EQ(view.DwordAt(0, 4) & 0x7u, SDMA_FUNC_GREATER_EQUAL);
    EXPECT_EQ(Join(view.DwordAt(0, 20), view.DwordAt(0, 16)), uint64_t{1} << 40);
  }
}

TEST(DeviceDependency, ACountdownWaitsWithLessOrEqual) {
  {
    DeviceView view(IsaByName("gfx942"));
    SignalWord dependency_word;
    dependency_word.value = 4;
    const abce_device_wait_t dependency{&dependency_word.value, 0, ABCE_DEVICE_WAIT_LEQ};
    ASSERT_EQ(SubmitCopy(view, nullptr, &dependency), ABCE_STATUS_OK);
    const WalkResult walk = view.WalkRing(0);
    ASSERT_TRUE(walk.clean);
    ASSERT_EQ(walk.Names(), (Names{"POLL_REGMEM", "COPY_LINEAR"}));
    const auto poll = view.PacketAt<SDMA_PKT_POLL_REGMEM>(0, walk.packets[0].byte_offset);
    EXPECT_EQ(poll.HEADER_UNION.func, SDMA_FUNC_LESS_EQUAL);
    EXPECT_EQ(poll.VALUE_UNION.value, 0u);
  }
  {
    DeviceView view(IsaByName("gfx1250"));
    SignalWord dependency_word;
    dependency_word.value = 4;
    const abce_device_wait_t dependency{&dependency_word.value, 0, ABCE_DEVICE_WAIT_LEQ};
    ASSERT_EQ(SubmitCopy(view, nullptr, &dependency), ABCE_STATUS_OK);
    EXPECT_EQ(view.DwordAt(0, 4) & 0x7u, SDMA_FUNC_LESS_EQUAL);
  }
  {
    DeviceView view(IsaByName("gfx942"));
    SignalWord dependency_word;
    const abce_device_wait_t reached_zero{&dependency_word.value, 0, ABCE_DEVICE_WAIT_LEQ};
    ASSERT_EQ(SubmitCopy(view, nullptr, &reached_zero, 4096, 0, ABCE_DEVICE_SUBMIT_NONBLOCKING),
              ABCE_STATUS_OK);
    EXPECT_EQ(view.WalkRing(0).Names(), (Names{"COPY_LINEAR"})) << "already at zero";
  }
}

TEST(DeviceDependency, RejectsAnUnknownCompare) {
  DeviceView view(IsaByName("gfx942"));
  SignalWord dependency_word;
  const abce_device_wait_t dependency{&dependency_word.value, 1, 7};
  EXPECT_EQ(SubmitCopy(view, nullptr, &dependency), ABCE_STATUS_INVALID_ARGUMENT);
}

// Checking a blocking submit's word would cost a load from wherever it lives;
// only a non-blocking submit checks, and then queues no wait.
TEST(DeviceDependency, ABlockingSubmitQueuesTheWaitEvenWhenMet) {
  for (const char* isa_name : {"gfx942", "gfx1250"}) {
    SCOPED_TRACE(isa_name);
    DeviceView view(IsaByName(isa_name));
    SignalWord dependency_word;
    dependency_word.value = 10;
    const abce_device_wait_t dependency{&dependency_word.value, 9, ABCE_DEVICE_WAIT_GEQ};
    ASSERT_EQ(SubmitCopy(view, nullptr, &dependency), ABCE_STATUS_OK);
    EXPECT_EQ(view.WalkRing(0).Names(), std::string(isa_name) == "gfx1250"
                                            ? (Names{"COPY_LINEAR_WAITSIGNAL"})
                                            : (Names{"POLL_REGMEM", "COPY_LINEAR"}));
    DeviceView nonblocking(IsaByName(isa_name));
    ASSERT_EQ(SubmitCopy(nonblocking, nullptr, &dependency, 4096, 0,
                         ABCE_DEVICE_SUBMIT_NONBLOCKING),
              ABCE_STATUS_OK);
    EXPECT_EQ(nonblocking.WalkRing(0).Names(), (Names{"COPY_LINEAR"}))
        << "overshoot still counts as met";
  }
}

TEST(DeviceDependency, NeedsToFitIn32BitsWithoutA64BitPoll) {
  SignalWord dependency_word;
  const abce_device_wait_t dependency{&dependency_word.value, uint64_t{1} << 32,
                                          ABCE_DEVICE_WAIT_GEQ};
  DeviceView gfx942(IsaByName("gfx942"));
  EXPECT_EQ(SubmitCopy(gfx942, nullptr, &dependency), ABCE_STATUS_INVALID_ARGUMENT);
  DeviceView gfx1250(IsaByName("gfx1250"));
  EXPECT_EQ(SubmitCopy(gfx1250, nullptr, &dependency), ABCE_STATUS_OK);
}

TEST(DeviceDependency, ComparesUnsigned) {
  SignalWord word;
  const abce_device_wait_t wait{&word.value, 5, ABCE_DEVICE_WAIT_GEQ};
  EXPECT_FALSE(abce_device_wait_holds(&wait, 4));
  EXPECT_TRUE(abce_device_wait_holds(&wait, 5));
  EXPECT_TRUE(abce_device_wait_holds(&wait, 6));
  EXPECT_TRUE(abce_device_wait_holds(&wait, ~uint64_t{0}));

  const abce_device_wait_t countdown{&word.value, 2, ABCE_DEVICE_WAIT_LEQ};
  EXPECT_FALSE(abce_device_wait_holds(&countdown, 3));
  EXPECT_TRUE(abce_device_wait_holds(&countdown, 2));
  EXPECT_TRUE(abce_device_wait_holds(&countdown, 0));
}

//===----------------------------------------------------------------------===//
// Handles
//===----------------------------------------------------------------------===//

TEST(DeviceHandle, EachHandleSubmitsToItsOwnRingInOrder) {
  DeviceView view(IsaByName("gfx942"));
  ASSERT_EQ(SubmitCopy(view, nullptr, nullptr, 4096, /*engine_idx=*/1), ABCE_STATUS_OK);
  ASSERT_EQ(SubmitCopy(view, nullptr), ABCE_STATUS_OK);
  ASSERT_EQ(SubmitCopy(view, nullptr, nullptr, 4096, /*engine_idx=*/1), ABCE_STATUS_OK);
  EXPECT_EQ(view.WalkRing(0).Names(), (Names{"COPY_LINEAR"}));
  EXPECT_EQ(view.WalkRing(1).Names(), (Names{"COPY_LINEAR", "COPY_LINEAR"}))
      << "both on the handle's ring, in submission order";
}

TEST(DeviceHandle, AMismatchedAbiIsRefusedWithoutTouchingTheRing) {
  DeviceView view(IsaByName("gfx942"));
  abce_device_handle_t stale = *view.handle();
  stale.abi_version = ABCE_DEVICE_ABI_VERSION + 1;
  EXPECT_FALSE(abce_device_handle_valid(&stale));
  EXPECT_EQ(abce_device_submit_linear(&stale, FakeAddress(0x20000000), FakeAddress(0x40000000),
                                      4096, nullptr, nullptr, 0),
            ABCE_STATUS_INVALID_ARGUMENT);
  view.engine(0).abi_version = ABCE_DEVICE_ABI_VERSION + 1;
  EXPECT_FALSE(abce_device_handle_valid(view.handle()));
  EXPECT_EQ(SubmitCopy(view, nullptr), ABCE_STATUS_INVALID_ARGUMENT);
  const abce_device_handle_t empty{};
  EXPECT_EQ(abce_device_submit_linear(&empty, FakeAddress(0x20000000), FakeAddress(0x40000000),
                                      4096, nullptr, nullptr, 0),
            ABCE_STATUS_INVALID_ARGUMENT);
  EXPECT_EQ(view.write_index(0), 0u);
}

//===----------------------------------------------------------------------===//
// Combined publish
//===----------------------------------------------------------------------===//

// Producers on several threads share one ring; a fake engine consumes whatever
// the doorbell exposes, decodes it, and poisons it. A frame published before it
// was written decodes as poison; a frame never published stalls the run. All
// threads share two CPUs, so producers are preempted mid-submit and others
// wait their turn behind them.
TEST(DevicePublish, ConcurrentProducersPublishEveryFrameOnceWrittenAndInOrder) {
#if defined(__SANITIZE_THREAD__)
  GTEST_SKIP() << "Ring memory is reused on the engine's say-so, which ThreadSanitizer cannot "
                  "follow through the device ring's relaxed read-index load.";
#endif  // __SANITIZE_THREAD__
  constexpr uint32_t kNumProducers = 8;
  constexpr uint32_t kSubmitsPerProducer = 50000;
  const IsaCase isa_case = IsaByName("gfx942");
  DeviceView view(isa_case);
  std::atomic<uint32_t> num_producers_done{0};
  std::atomic<bool> walk_clean{true};
  std::atomic<bool> doorbell_monotonic{true};
  cpu_set_t two_cpus;
  CPU_ZERO(&two_cpus);
  CPU_SET(0, &two_cpus);
  CPU_SET(1, &two_cpus);
  const auto share_two_cpus = [&two_cpus] { sched_setaffinity(0, sizeof(two_cpus), &two_cpus); };

  std::thread engine([&] {
    share_two_cpus();
    uint32_t* words = view.MutableRingWords(0);
    uint64_t read_index = 0;
    for (;;) {
      const uint64_t doorbell = __atomic_load_n(view.DoorbellWord(0), __ATOMIC_ACQUIRE);
      if (doorbell < read_index) doorbell_monotonic = false;
      if (doorbell == read_index) {
        if (num_producers_done.load() == kNumProducers &&
            __atomic_load_n(&view.Control(0).reserve_cursor, __ATOMIC_ACQUIRE) == read_index)
          return;
        std::this_thread::yield();
        continue;
      }
      while (read_index < doorbell) {
        const uint64_t offset = read_index % kRingBytes;
        const uint64_t segment = std::min<uint64_t>(doorbell - read_index, kRingBytes - offset);
        const WalkResult walk = Walk(words + offset / sizeof(uint32_t), segment, isa_case.isa);
        if (!walk.clean) walk_clean = false;
        std::fill(words + offset / sizeof(uint32_t), words + (offset + segment) / sizeof(uint32_t),
                  kRingPoison);
        read_index += segment;
      }
      __atomic_store_n(view.ReadIndexWord(0), read_index, __ATOMIC_RELEASE);
    }
  });

  std::vector<std::thread> producers;
  std::atomic<bool> submits_ok{true};
  for (uint32_t producer_idx = 0; producer_idx < kNumProducers; ++producer_idx) {
    producers.emplace_back([&, producer_idx] {
      share_two_cpus();
      for (uint32_t submit_idx = 0; submit_idx < kSubmitsPerProducer; ++submit_idx) {
        // Mixed frame sizes, so frames straddle the ring end at varying offsets.
        const abce_status_t status =
            (submit_idx + producer_idx) % 3 == 0
                ? SubmitWrite(view, FakeAddress(0x30000004), submit_idx, 4)
                : SubmitCopy(view, nullptr);
        if (status != ABCE_STATUS_OK) submits_ok = false;
      }
      ++num_producers_done;
    });
  }
  for (std::thread& producer : producers) producer.join();
  engine.join();
  EXPECT_TRUE(submits_ok);
  EXPECT_TRUE(walk_clean) << "the engine saw a frame before it was written";
  EXPECT_TRUE(doorbell_monotonic);
  const abce_ring_control_t& control = view.Control(0);
  EXPECT_EQ(view.doorbell(0), control.reserve_cursor) << "every frame published";
  EXPECT_EQ(control.commit_cursor, control.reserve_cursor) << "publish bit released";
}

//===----------------------------------------------------------------------===//
// Non-blocking submits and signal-only submits
//===----------------------------------------------------------------------===//

TEST(DeviceNonblocking, AFullRingIsNotReadyAndQueuesNothing) {
  DeviceView view(IsaByName("gfx942"));
  uint32_t num_queued = 0;
  abce_status_t status = ABCE_STATUS_OK;
  while ((status = SubmitCopy(view, nullptr, nullptr, 4096, 0, ABCE_DEVICE_SUBMIT_NONBLOCKING)) ==
         ABCE_STATUS_OK)
    ++num_queued;
  EXPECT_EQ(status, ABCE_STATUS_NOT_READY);
  EXPECT_EQ(view.write_index(0), num_queued * sizeof(SDMA_PKT_COPY_LINEAR));
  EXPECT_EQ(view.doorbell(0), view.write_index(0));
  view.EngineConsumeAll(0);
  EXPECT_EQ(SubmitCopy(view, nullptr, nullptr, 4096, 0, ABCE_DEVICE_SUBMIT_NONBLOCKING),
            ABCE_STATUS_OK)
      << "room again once the engine drains";
}

TEST(DeviceNonblocking, AnUnmetDependencyIsNotReadyInsteadOfParkingTheRing) {
  for (const char* isa_name : {"gfx942", "gfx1250"}) {
    SCOPED_TRACE(isa_name);
    DeviceView view(IsaByName(isa_name));
    SignalWord dependency_word;
    const abce_device_wait_t dependency{&dependency_word.value, 1, ABCE_DEVICE_WAIT_GEQ};
    EXPECT_EQ(SubmitCopy(view, nullptr, &dependency, 4096, 0, ABCE_DEVICE_SUBMIT_NONBLOCKING),
              ABCE_STATUS_NOT_READY);
    EXPECT_EQ(view.write_index(0), 0u);
    dependency_word.value = 1;
    ASSERT_EQ(SubmitCopy(view, nullptr, &dependency, 4096, 0, ABCE_DEVICE_SUBMIT_NONBLOCKING),
              ABCE_STATUS_OK);
    EXPECT_EQ(view.WalkRing(0).Names(), (Names{"COPY_LINEAR"})) << "met: no wait packet";
  }
}

TEST(DeviceNonblocking, RejectsAnUnknownSubmitFlag) {
  DeviceView view(IsaByName("gfx942"));
  EXPECT_EQ(SubmitCopy(view, nullptr, nullptr, 4096, 0, 1u << 7), ABCE_STATUS_INVALID_ARGUMENT);
  EXPECT_EQ(view.write_index(0), 0u);
}

TEST(DeviceSignal, IsTheCompletionPacketAloneAfterEverythingEarlier) {
  struct Case {
    const char* isa_name;
    abce_device_signal_op_t op;
    Names packets;
  };
  for (const Case& token_case :
       {Case{"gfx942", ABCE_DEVICE_SIGNAL_ADD, {"COPY_LINEAR", "ATOMIC"}},
        Case{"gfx942", ABCE_DEVICE_SIGNAL_SET, {"COPY_LINEAR", "FENCE", "FENCE"}},
        Case{"gfx1250", ABCE_DEVICE_SIGNAL_ADD, {"COPY_LINEAR", "ATOMIC"}},
        Case{"gfx1250", ABCE_DEVICE_SIGNAL_SET, {"COPY_LINEAR", "FENCE_64"}}}) {
    SCOPED_TRACE(std::string(token_case.isa_name) + " op " + std::to_string(token_case.op));
    DeviceView view(IsaByName(token_case.isa_name));
    SignalWord signal;
    ASSERT_EQ(SubmitCopy(view, nullptr), ABCE_STATUS_OK);
    // UNORDERED has no copy to ride in, so the signal stays standalone.
    const abce_device_signal_t completion =
        SignalOn(&signal, 1, token_case.op, ABCE_DEVICE_SIGNAL_UNORDERED);
    ASSERT_EQ(abce_device_submit_signal(view.handle(), &completion, nullptr, 0), ABCE_STATUS_OK);
    const WalkResult walk = view.WalkRing(0);
    EXPECT_TRUE(walk.clean);
    EXPECT_EQ(walk.Names(), token_case.packets);
  }
}

TEST(DeviceSignal, NeedsACompletionAndTakesADependency) {
  DeviceView view(IsaByName("gfx942"));
  EXPECT_EQ(abce_device_submit_signal(view.handle(), nullptr, nullptr, 0),
            ABCE_STATUS_INVALID_ARGUMENT);
  SignalWord signal;
  SignalWord dependency_word;
  const abce_device_signal_t completion = SignalOn(&signal, 1, ABCE_DEVICE_SIGNAL_ADD);
  const abce_device_wait_t dependency{&dependency_word.value, 1, ABCE_DEVICE_WAIT_GEQ};
  ASSERT_EQ(abce_device_submit_signal_one_lane(view.handle(), &completion, &dependency, 0),
            ABCE_STATUS_OK);
  EXPECT_EQ(view.WalkRing(0).Names(), (Names{"POLL_REGMEM", "ATOMIC"}));
}

//===----------------------------------------------------------------------===//
// Second completion
//===----------------------------------------------------------------------===//

abce_status_t SubmitCopyTwice(const DeviceView& view, const abce_device_signal_t* first,
                              const abce_device_signal_t* second, size_t size = 4096) {
  return abce_device_submit_linear_two_completions(view.handle(), FakeAddress(0x20000000),
                                                   FakeAddress(0x40000000), size, nullptr, first,
                                                   second, 0);
}

TEST(DeviceSecondCompletion, FollowsTheFirstInTheSameFrame) {
  struct Case {
    const char* isa_name;
    abce_device_signal_op_t first;
    abce_device_signal_op_t second;
    Names packets;
  };
  for (const Case& test_case :
       {Case{"gfx942", ABCE_DEVICE_SIGNAL_ADD, ABCE_DEVICE_SIGNAL_ADD,
             {"COPY_LINEAR", "ATOMIC", "ATOMIC"}},
        Case{"gfx942", ABCE_DEVICE_SIGNAL_SET, ABCE_DEVICE_SIGNAL_SET,
             {"COPY_LINEAR", "FENCE", "FENCE", "FENCE", "FENCE"}},
        Case{"gfx942", ABCE_DEVICE_SIGNAL_SET, ABCE_DEVICE_SIGNAL_ADD,
             {"COPY_LINEAR", "FENCE", "FENCE", "ATOMIC"}},
        Case{"gfx1250", ABCE_DEVICE_SIGNAL_ADD, ABCE_DEVICE_SIGNAL_ADD,
             {"COPY_LINEAR", "ATOMIC", "ATOMIC"}},
        Case{"gfx1250", ABCE_DEVICE_SIGNAL_SET, ABCE_DEVICE_SIGNAL_SET,
             {"COPY_LINEAR", "FENCE_64", "FENCE_64"}}}) {
    SCOPED_TRACE(std::string(test_case.isa_name) + " " + std::to_string(test_case.first) + " " +
                 std::to_string(test_case.second));
    DeviceView view(IsaByName(test_case.isa_name));
    SignalWord first_word;
    SignalWord second_word;
    const abce_device_signal_t first = SignalOn(&first_word, 5, test_case.first);
    const abce_device_signal_t second = SignalOn(&second_word, 9, test_case.second);
    ASSERT_EQ(SubmitCopyTwice(view, &first, &second), ABCE_STATUS_OK);

    const WalkResult walk = view.WalkRing(0);
    EXPECT_TRUE(walk.clean);
    EXPECT_EQ(walk.Names(), test_case.packets);
    EXPECT_EQ(view.doorbell(0), view.write_index(0));
  }
}

TEST(DeviceSecondCompletion, WritesTheSameBytesAsACopyThenASeparateSignal) {
  for (const char* isa_name : {"gfx90a", "gfx942", "gfx950", "gfx1250"}) {
    for (const abce_device_signal_op_t first_op : {ABCE_DEVICE_SIGNAL_SET, ABCE_DEVICE_SIGNAL_ADD}) {
      for (const abce_device_signal_op_t second_op :
           {ABCE_DEVICE_SIGNAL_SET, ABCE_DEVICE_SIGNAL_ADD, ABCE_DEVICE_SIGNAL_SUB}) {
        for (const uint32_t flags : {0u, uint32_t{ABCE_DEVICE_SIGNAL_UNORDERED}}) {
          // 2 GiB is above every part's per-packet maximum, so that copy chunks.
          for (const size_t size : {size_t{4096}, size_t{1} << 31}) {
            SCOPED_TRACE(std::string(isa_name) + " ops " + std::to_string(first_op) + " " +
                         std::to_string(second_op) + " flags " + std::to_string(flags) + " size " +
                         std::to_string(size));
            SignalWord first_word;
            SignalWord second_word;
            const abce_device_signal_t first = SignalOn(&first_word, 5, first_op, flags);
            const abce_device_signal_t second = SignalOn(&second_word, 9, second_op);
            DeviceView together(IsaByName(isa_name));
            DeviceView apart(IsaByName(isa_name));
            ASSERT_EQ(SubmitCopyTwice(together, &first, &second, size), ABCE_STATUS_OK);
            ASSERT_EQ(SubmitCopy(apart, &first, nullptr, size), ABCE_STATUS_OK);
            ASSERT_EQ(abce_device_submit_signal(apart.handle(), &second, nullptr, 0),
                      ABCE_STATUS_OK);
            EXPECT_EQ(together.Bytes(0), apart.Bytes(0));
          }
        }
      }
    }
  }
}

TEST(DeviceSecondCompletion, NeedsAFirstCompletionAndIsValidatedLikeIt) {
  DeviceView view(IsaByName("gfx942"));
  SignalWord word;
  const abce_device_signal_t good = SignalOn(&word, 1, ABCE_DEVICE_SIGNAL_SET);
  EXPECT_EQ(SubmitCopyTwice(view, nullptr, &good), ABCE_STATUS_INVALID_ARGUMENT);

  alignas(8) uint64_t words[2] = {};
  abce_device_signal_t second{reinterpret_cast<uint64_t*>(reinterpret_cast<char*>(words) + 4), 1,
                              ABCE_DEVICE_SIGNAL_SET, 0};
  EXPECT_EQ(SubmitCopyTwice(view, &good, &second), ABCE_STATUS_INVALID_ARGUMENT)
      << "misaligned";
  second.address = &words[0];
  second.op = 7;
  EXPECT_EQ(SubmitCopyTwice(view, &good, &second), ABCE_STATUS_INVALID_ARGUMENT)
      << "unknown operation";
  EXPECT_EQ(view.write_index(0), 0u) << "a refused submit must not move the ring";
}

TEST(DeviceSecondCompletion, AnAtomicSecondNeedsDeviceAtomics) {
  DeviceViewOptions options;
  options.device_atomic_support = false;
  DeviceView view(IsaByName("gfx942"), options);
  SignalWord word;
  const abce_device_signal_t set = SignalOn(&word, 1, ABCE_DEVICE_SIGNAL_SET);
  const abce_device_signal_t add = SignalOn(&word, 1, ABCE_DEVICE_SIGNAL_ADD);
  EXPECT_EQ(SubmitCopyTwice(view, &set, &add), ABCE_STATUS_UNIMPLEMENTED);
  EXPECT_EQ(view.write_index(0), 0u);
  EXPECT_EQ(SubmitCopyTwice(view, &set, &set), ABCE_STATUS_OK);
}

TEST(DeviceSecondCompletion, OneLaneEmitsTheSameFrameAsTheGeneralCall) {
  for (const char* isa_name : {"gfx942", "gfx1250"}) {
    SCOPED_TRACE(isa_name);
    SignalWord first_word;
    SignalWord second_word;
    const abce_device_signal_t first = SignalOn(&first_word, 5, ABCE_DEVICE_SIGNAL_SET);
    const abce_device_signal_t second = SignalOn(&second_word, 1, ABCE_DEVICE_SIGNAL_ADD);
    DeviceView general(IsaByName(isa_name));
    DeviceView one_lane(IsaByName(isa_name));
    ASSERT_EQ(SubmitCopyTwice(general, &first, &second), ABCE_STATUS_OK);
    ASSERT_EQ(abce_device_submit_linear_two_completions_one_lane(
                  one_lane.handle(), FakeAddress(0x20000000), FakeAddress(0x40000000), 4096,
                  nullptr, &first, &second, 0),
              ABCE_STATUS_OK);
    EXPECT_EQ(general.Bytes(0), one_lane.Bytes(0));
  }
}

//===----------------------------------------------------------------------===//
// Inline write
//===----------------------------------------------------------------------===//

TEST(DeviceWrite, CarriesItsValueInThePacket) {
  for (const char* isa_name : {"gfx942", "gfx1250"}) {
    SCOPED_TRACE(isa_name);
    DeviceView view(IsaByName(isa_name));
    void* dst = FakeAddress(0x30000000);
    ASSERT_EQ(SubmitWrite(view, dst, 0x0123456789abcdefull, 8), ABCE_STATUS_OK);
    ASSERT_EQ(SubmitWrite(view, dst, 0xc0ffee11u, 4), ABCE_STATUS_OK);
    const WalkResult walk = view.WalkRing(0);
    ASSERT_TRUE(walk.clean);
    ASSERT_EQ(walk.Names(), (Names{"WRITE_LINEAR", "WRITE_LINEAR"}));
    const auto eight = view.PacketAt<SDMA_PKT_WRITE_UNTILED>(0, walk.packets[0].byte_offset);
    EXPECT_EQ(eight.DW_3_UNION.count, 1u) << "count is dwords - 1";
    EXPECT_EQ(Join(eight.DST_ADDR_HI_UNION.dst_addr_63_32, eight.DST_ADDR_LO_UNION.dst_addr_31_0),
              reinterpret_cast<uintptr_t>(dst));
    EXPECT_EQ(view.DwordAt(0, walk.packets[0].byte_offset + 16), 0x89abcdefu);
    EXPECT_EQ(view.DwordAt(0, walk.packets[0].byte_offset + 20), 0x01234567u);
    const auto four = view.PacketAt<SDMA_PKT_WRITE_UNTILED>(0, walk.packets[1].byte_offset);
    EXPECT_EQ(four.DW_3_UNION.count, 0u);
    EXPECT_EQ(view.DwordAt(0, walk.packets[1].byte_offset + 16), 0xc0ffee11u);
  }
}

// CONST_FILL sizes are byte or dword only, so each byte is a one-byte fill,
// with the dependency check left on.
TEST(DeviceWrite, OneAndTwoBytesAreOneByteFillsEach) {
  for (const char* isa_name : {"gfx942", "gfx1250"}) {
    SCOPED_TRACE(isa_name);
    DeviceView view(IsaByName(isa_name));
    ASSERT_EQ(SubmitWrite(view, FakeAddress(0x30000003), 0x5a, 1), ABCE_STATUS_OK);
    ASSERT_EQ(SubmitWrite(view, FakeAddress(0x30000006), 0xbeef, 2), ABCE_STATUS_OK);
    const WalkResult walk = view.WalkRing(0);
    ASSERT_TRUE(walk.clean);
    ASSERT_EQ(walk.Names(), (Names{"CONST_FILL", "CONST_FILL", "CONST_FILL"}));
    std::vector<size_t> fill_offsets;
    for (const auto& packet : walk.packets)
      if (packet.name == "CONST_FILL") fill_offsets.push_back(packet.byte_offset);
    const uint32_t expected_address[] = {0x30000003u, 0x30000006u, 0x30000007u};
    const uint32_t expected_byte[] = {0x5au, 0xefu, 0xbeu};
    for (size_t packet_idx = 0; packet_idx < 3; ++packet_idx) {
      const auto fill = view.PacketAt<SDMA_PKT_CONSTANT_FILL>(0, fill_offsets[packet_idx]);
      EXPECT_EQ(fill.HEADER_UNION.fillsize, 0u) << "byte fill";
      EXPECT_EQ(fill.HEADER_UNION.npd, 0u) << "must not bypass dependency checks";
      EXPECT_EQ(fill.COUNT_UNION.count, 0u) << "one byte";
      EXPECT_EQ(fill.DST_ADDR_LO_UNION.dst_addr_31_0, expected_address[packet_idx]);
      EXPECT_EQ(fill.DATA_UNION.src_data_31_0, expected_byte[packet_idx]);
    }
  }
}

TEST(DeviceWrite, TakesADependencyAndACompletionLikeACopy) {
  {
    DeviceView view(IsaByName("gfx1250"));
    SignalWord dependency_word;
    SignalWord signal;
    const abce_device_wait_t dependency{&dependency_word.value, 1, ABCE_DEVICE_WAIT_GEQ};
    const abce_device_signal_t completion = SignalOn(&signal, 2, ABCE_DEVICE_SIGNAL_SET);
    ASSERT_EQ(SubmitWrite(view, FakeAddress(0x30000000), 1, 4, &completion, &dependency),
              ABCE_STATUS_OK);
    const WalkResult walk = view.WalkRing(0);
    ASSERT_TRUE(walk.clean);
    EXPECT_EQ(walk.Names(), (Names{"POLL_MEM_64", "WRITE_LINEAR", "FENCE_64"}));
  }
  {
    DeviceView view(IsaByName("gfx942"));
    SignalWord signal;
    const abce_device_signal_t completion = SignalOn(&signal, 1, ABCE_DEVICE_SIGNAL_ADD);
    ASSERT_EQ(SubmitWrite(view, FakeAddress(0x30000000), 1, 8, &completion), ABCE_STATUS_OK);
    const WalkResult walk = view.WalkRing(0);
    ASSERT_TRUE(walk.clean);
    EXPECT_EQ(walk.Names(), (Names{"WRITE_LINEAR", "ATOMIC"}));
  }
}

TEST(DeviceWrite, RejectsBadSizesAndMisalignedDestinations) {
  DeviceView view(IsaByName("gfx942"));
  EXPECT_EQ(SubmitWrite(view, FakeAddress(0x30000000), 1, 3), ABCE_STATUS_INVALID_ARGUMENT);
  EXPECT_EQ(SubmitWrite(view, FakeAddress(0x30000000), 1, 16), ABCE_STATUS_INVALID_ARGUMENT);
  EXPECT_EQ(SubmitWrite(view, FakeAddress(0x30000001), 1, 2), ABCE_STATUS_INVALID_ARGUMENT);
  EXPECT_EQ(SubmitWrite(view, FakeAddress(0x30000002), 1, 4), ABCE_STATUS_INVALID_ARGUMENT);
  EXPECT_EQ(SubmitWrite(view, FakeAddress(0x30000002), 1, 8), ABCE_STATUS_INVALID_ARGUMENT);
  EXPECT_EQ(SubmitWrite(view, nullptr, 1, 4), ABCE_STATUS_INVALID_ARGUMENT);
  EXPECT_EQ(view.write_index(0), 0u);
}

//===----------------------------------------------------------------------===//
// Ring wrap and the one-lane entry points
//===----------------------------------------------------------------------===//

// A frame that would straddle the ring end shares its reservation with the
// tail's NOPs: one submit publishes both, and the frame starts at offset zero.
// Four-byte writes: a 20-byte frame does not divide the ring, so one straddles.
TEST(DeviceRing, AFrameThatWouldStraddleTheEndFoldsTheTailPaddingIn) {
  DeviceView view(IsaByName("gfx942"));
  void* dst = FakeAddress(0x30000004);
  const uint64_t frame_bytes = sizeof(SDMA_PKT_WRITE_UNTILED) + sizeof(uint32_t);
  const uint64_t frames_that_fit = kRingBytes / frame_bytes;
  for (uint64_t frame_idx = 0; frame_idx < frames_that_fit; ++frame_idx) {
    view.EngineConsumeAll(0);
    ASSERT_EQ(SubmitWrite(view, dst, 0x5a, 4), ABCE_STATUS_OK);
  }
  const uint64_t tail_start = frames_that_fit * frame_bytes;
  ASSERT_LT(kRingBytes - tail_start, frame_bytes) << "the next frame must straddle";

  view.EngineConsumeAll(0);
  ASSERT_EQ(SubmitWrite(view, dst, 0x5a, 4), ABCE_STATUS_OK);
  EXPECT_EQ(view.write_index(0), kRingBytes + frame_bytes) << "padding and frame in one publish";
  EXPECT_EQ(view.doorbell(0), view.write_index(0));
  const uint32_t* words = view.RingWords(0);
  const WalkResult tail =
      Walk(words + tail_start / sizeof(uint32_t), kRingBytes - tail_start, IsaByName("gfx942").isa);
  EXPECT_TRUE(tail.clean);
  EXPECT_EQ(tail.Names(false), (Names{"NOP"})) << "one counted NOP covers the tail";
  const WalkResult walk = Walk(words, frame_bytes, IsaByName("gfx942").isa);
  EXPECT_TRUE(walk.clean);
  EXPECT_EQ(walk.Names(), (Names{"WRITE_LINEAR"})) << "the frame restarts at offset zero";
}

TEST(DeviceRing, PaddingPastOneCountedNopContinuesAsOneDwordNops) {
  const uint64_t num_words = uint64_t{SDMA_NOP_MAX_DWORDS} + 5;
  std::vector<uint32_t> ring_words(4 * SDMA_NOP_MAX_DWORDS, kRingPoison);
  abce_device_ring_t ring{};
  ring.base = ring_words.data();
  ring.size = ring_words.size() * sizeof(uint32_t);
  abce_ring_reservation_t padding;
  padding.start = 3 * ring.size + sizeof(uint32_t);
  padding.end = padding.start + num_words * sizeof(uint32_t);
  abce_device_ring_pad(&ring, &padding);
  const WalkResult walk =
      Walk(ring_words.data() + 1, num_words * sizeof(uint32_t), IsaByName("gfx942").isa);
  EXPECT_TRUE(walk.clean);
  ASSERT_EQ(walk.packets.size(), 1u + 5u);
  EXPECT_EQ(walk.packets[0].dwords, SDMA_NOP_MAX_DWORDS);
  EXPECT_EQ(walk.Names(false), Names(6, "NOP"));
  EXPECT_EQ(ring_words[0], kRingPoison);
  EXPECT_EQ(ring_words[1 + num_words], kRingPoison) << "nothing past the padding is touched";
}

TEST(DeviceRing, AFrameLargerThanHalfTheRingIsOutOfRange) {
  DeviceView view(IsaByName("gfx942"));
  // 2 TiB is ~2049 packets of ~1 GiB: 56 KiB of frame, which fits the 64 KiB
  // ring but not half of it.
  EXPECT_EQ(SubmitCopy(view, nullptr, nullptr, size_t{1} << 41),
            ABCE_STATUS_OUT_OF_RANGE);
  EXPECT_EQ(view.write_index(0), 0u);
}

// Off a GPU there are no lanes, so the one-lane calls are the general ones.
TEST(DeviceOneLane, EmitsTheSameFramesAsTheGeneralCalls) {
  for (const char* isa_name : {"gfx942", "gfx1250"}) {
    SCOPED_TRACE(isa_name);
    SignalWord signal;
    SignalWord dependency_word;
    const abce_device_signal_t completion = SignalOn(&signal, 1, ABCE_DEVICE_SIGNAL_ADD);
    const abce_device_wait_t dependency{&dependency_word.value, 1, ABCE_DEVICE_WAIT_GEQ};
    DeviceView general(IsaByName(isa_name));
    DeviceView one_lane(IsaByName(isa_name));
    ASSERT_EQ(SubmitCopy(general, &completion, &dependency), ABCE_STATUS_OK);
    ASSERT_EQ(abce_device_submit_linear_one_lane(one_lane.handle(), FakeAddress(0x20000000),
                                                 FakeAddress(0x40000000), 4096, &dependency,
                                                 &completion, 0),
              ABCE_STATUS_OK);
    ASSERT_EQ(SubmitWrite(general, FakeAddress(0x30000000), 7, 8, &completion), ABCE_STATUS_OK);
    ASSERT_EQ(abce_device_submit_write_one_lane(one_lane.handle(), FakeAddress(0x30000000), 7, 8,
                                                nullptr, &completion, 0),
              ABCE_STATUS_OK);
    EXPECT_EQ(general.Bytes(0), one_lane.Bytes(0));
  }
}

//===----------------------------------------------------------------------===//
// Every shape, every device ISA: the frame is exactly what was reserved
//===----------------------------------------------------------------------===//

class DeviceFrames : public ::testing::TestWithParam<std::string> {};

TEST_P(DeviceFrames, EveryShapeDecodesToExactlyTheBytesPublished) {
  const IsaCase isa_case = IsaByName(GetParam());
  SignalWord signal;
  SignalWord dependency_word;
  const abce_device_wait_t dependency{&dependency_word.value, 1, ABCE_DEVICE_WAIT_GEQ};
  const abce_device_signal_t set = SignalOn(&signal, 1, ABCE_DEVICE_SIGNAL_SET);
  const abce_device_signal_t add = SignalOn(&signal, 1, ABCE_DEVICE_SIGNAL_ADD);
  const abce_device_signal_t sub = SignalOn(&signal, 1, ABCE_DEVICE_SIGNAL_SUB);
  for (const abce_device_signal_t* completion : {static_cast<const abce_device_signal_t*>(nullptr),
                                                 &set, &add, &sub}) {
    for (const abce_device_wait_t* wait : {static_cast<const abce_device_wait_t*>(nullptr),
                                           &dependency}) {
      // 2 GiB is above every part's per-packet maximum, so that copy chunks.
      for (const size_t size : {size_t{4096}, size_t{1} << 31}) {
        SCOPED_TRACE(::testing::Message() << "completion " << (completion ? completion->op : 9)
                                          << " wait " << (wait != nullptr) << " size " << size);
        DeviceView view(isa_case);
        ASSERT_EQ(SubmitCopy(view, completion, wait, size),
                  ABCE_STATUS_OK);
        ASSERT_EQ(SubmitWrite(view, FakeAddress(0x30000000), 1, 8, completion, wait),
                  ABCE_STATUS_OK);
        const WalkResult walk = view.WalkRing(0);
        EXPECT_TRUE(walk.clean);
        EXPECT_EQ(view.doorbell(0), view.write_index(0));
      }
    }
  }
}

INSTANTIATE_TEST_SUITE_P(DeviceIsas, DeviceFrames,
                         ::testing::Values("gfx90a", "gfx942", "gfx950", "gfx1250"),
                         [](const ::testing::TestParamInfo<std::string>& info) { return info.param; });

TEST(DeviceFrames, PartsWhoseCopiesNeedGcrGetNoEngine) {
  for (const char* isa_name : {"gfx1030", "gfx1100", "gfx1201"}) {
    SCOPED_TRACE(isa_name);
    abce_builder_t builder;
    ASSERT_EQ(abce_builder_initialize(IsaByName(isa_name).isa, nullptr, &builder), ABCE_STATUS_OK);
    const abce_device_ring_t ring{};
    abce_device_engine_t engine;
    EXPECT_EQ(abce_device_engine_initialize(&builder, &ring, 0, true, &engine),
              ABCE_STATUS_UNIMPLEMENTED);
  }
}

}  // namespace
