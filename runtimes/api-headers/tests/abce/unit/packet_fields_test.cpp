/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

// Field-level checks on individual SDMA packets.
//
// The decoder only knows packet lengths, so it cannot see a wrong field. These
// tests read the fields back and pin each bug class ROCr has had to fix in its
// own SDMA packet builders (see TEST_PLAN.md, "Regression ledger"): a missing
// `sys` bit, a scope left at CU, and a fused packet emitted with its full-struct
// layout where the engine parses it compacted.

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>

#include "abce_test_harness.hpp"

namespace {

using abce_test::FakeAddress;
using abce_test::IsaByName;

constexpr uint32_t kWaitBlockFirstDword = 1;
constexpr uint32_t kWaitBlockDwords = 7;
constexpr uint32_t kSignalBlockDwords = 5;
// Position of the signal block inside the full-struct layout the field unions
// are declared against: header, wait(7), copy body(6), signal(5).
constexpr uint32_t kSignalBlockFullLayoutDword = 14;

abce_builder_t MakeBuilder(const char* isa_name) {
  abce_builder_t builder;
  EXPECT_TRUE(abce_status_is_ok(abce_builder_initialize(IsaByName(isa_name).isa, nullptr, &builder)));
  return builder;
}

uint64_t Reassemble(uint32_t high, uint32_t low) { return (static_cast<uint64_t>(high) << 32) | low; }

abce_wait_signal_params_t WaitAndSignal() {
  abce_wait_signal_params_t params = abce_wait_signal_none();
  params.wait_addr = FakeAddress(0x2000);
  params.signal_addr = FakeAddress(0x3000);
  params.wait_reference = 5;
  params.wait_mask = ~0ull;
  params.boundary_wait_signal = false;
  return params;
}

// The wait and signal blocks are identical across every fused packet kind, so
// one view reads them whatever the body in between is. Built by copying the
// emitted blocks to the offsets the unions are declared at.
struct FusedBlocks {
  SDMA_PKT_COPY_LINEAR_WAITSIGNAL_GFX125PLUS view;
  uint32_t emitted_dwords;
  bool has_wait;
  bool has_signal;
};

FusedBlocks InspectFused(const uint8_t* emitted, abce_isa_version_t isa) {
  FusedBlocks blocks;
  std::memset(&blocks.view, 0, sizeof(blocks.view));
  const uint32_t* words = reinterpret_cast<const uint32_t*>(emitted);
  const abce_decoded_packet_t decoded = abce_decode_sdma_packet(words, 64, isa);
  blocks.emitted_dwords = decoded.dwords;
  blocks.has_wait = (words[0] & (1u << 30)) != 0;
  blocks.has_signal = (words[0] & (1u << 31)) != 0;
  uint32_t* view_words = reinterpret_cast<uint32_t*>(&blocks.view);
  view_words[0] = words[0];
  if (blocks.has_wait)
    std::memcpy(view_words + kWaitBlockFirstDword, words + kWaitBlockFirstDword,
                kWaitBlockDwords * sizeof(uint32_t));
  if (blocks.has_signal)
    std::memcpy(view_words + kSignalBlockFullLayoutDword,
                words + blocks.emitted_dwords - kSignalBlockDwords,
                kSignalBlockDwords * sizeof(uint32_t));
  return blocks;
}

void ExpectWaitAndSignalBlocksAreSystemScope(const FusedBlocks& blocks) {
  ASSERT_TRUE(blocks.has_wait);
  ASSERT_TRUE(blocks.has_signal);
  // ROCr 6691fe3e619: wait_scope was left at 0 (CU), so the poll could miss a
  // host write to the signal.
  EXPECT_EQ(blocks.view.WAIT_FUNCTION_UNION.wait_function, SDMA_FUNC_EQUAL);
  EXPECT_EQ(blocks.view.WAIT_FUNCTION_UNION.wait_scope, SDMA_MEMORY_SCOPE_SYS);
  EXPECT_EQ(Reassemble(blocks.view.WAIT_ADDR_HI_UNION.wait_addr_63_32,
                       blocks.view.WAIT_ADDR_LO_UNION.wait_addr_31_3 << 3),
            reinterpret_cast<uintptr_t>(FakeAddress(0x2000)));
  EXPECT_EQ(Reassemble(blocks.view.WAIT_REFERENCE_HI_UNION.wait_reference_63_32,
                       blocks.view.WAIT_REFERENCE_LO_UNION.wait_reference_31_0),
            5u);
  EXPECT_EQ(blocks.view.SIGNAL_OPERATION_UNION.signal_operation, SDMA_SIGNAL_OP_SUB64);
  EXPECT_EQ(blocks.view.SIGNAL_OPERATION_UNION.signal_scope, SDMA_MEMORY_SCOPE_SYS);
  EXPECT_EQ(Reassemble(blocks.view.SIGNAL_ADDR_HI_UNION.signal_addr_63_32,
                       blocks.view.SIGNAL_ADDR_LO_UNION.signal_addr_31_3 << 3),
            reinterpret_cast<uintptr_t>(FakeAddress(0x3000)));
}

//===----------------------------------------------------------------------===//
// 64-bit poll / fence (ROCr 7f137590661: sys and scope were never set)
//===----------------------------------------------------------------------===//

TEST(PacketFields, Fence64SetsSysAndSystemScope) {
  const abce_builder_t builder = MakeBuilder("gfx1250");
  alignas(16) uint8_t packet[sizeof(SDMA_PKT_FENCE_64B_GFX125PLUS)] = {};
  abce_builder_build_fence_64b(&builder, reinterpret_cast<char*>(packet), FakeAddress(0x1000),
                               0x1122334455667788ull);
  const auto* fence = reinterpret_cast<const SDMA_PKT_FENCE_64B_GFX125PLUS*>(packet);
  EXPECT_EQ(fence->HEADER_UNION.op, SDMA_OP_FENCE);
  EXPECT_EQ(fence->HEADER_UNION.sub_op, SDMA_SUBOP_FENCE_64B);
  EXPECT_EQ(fence->HEADER_UNION.sys, 1u);
  EXPECT_EQ(fence->HEADER_UNION.scope, SDMA_MEMORY_SCOPE_SYS);
  EXPECT_EQ(fence->HEADER_UNION.mtype, 3u);
  EXPECT_EQ(Reassemble(fence->ADDR_HI_UNION.addr_63_32, fence->ADDR_LO_UNION.addr_31_3 << 3),
            reinterpret_cast<uintptr_t>(FakeAddress(0x1000)));
  EXPECT_EQ(Reassemble(fence->DATA_HI_UNION.data_63_32, fence->DATA_LO_UNION.data_31_0),
            0x1122334455667788ull);
}

TEST(PacketFields, Poll64SetsSysAndSystemScopeAndRetriesForever) {
  const abce_builder_t builder = MakeBuilder("gfx1250");
  alignas(16) uint8_t packet[sizeof(SDMA_PKT_POLL_MEM_64B_GFX125PLUS)] = {};
  abce_builder_build_poll_64b(&builder, reinterpret_cast<char*>(packet), FakeAddress(0x1000),
                              0x0000000100000002ull, 0xffffffff00000000ull);
  const auto* poll = reinterpret_cast<const SDMA_PKT_POLL_MEM_64B_GFX125PLUS*>(packet);
  EXPECT_EQ(poll->HEADER_UNION.op, SDMA_OP_POLL_REGMEM);
  EXPECT_EQ(poll->HEADER_UNION.sub_op, SDMA_SUBOP_POLL_MEM_64B);
  EXPECT_EQ(poll->HEADER_UNION.func, SDMA_FUNC_EQUAL);
  EXPECT_EQ(poll->HEADER_UNION.sys, 1u);
  EXPECT_EQ(poll->DW7_UNION.scope, SDMA_MEMORY_SCOPE_SYS);
  EXPECT_EQ(poll->DW7_UNION.retry_count, 0u) << "0 is infinite retry";
  EXPECT_EQ(Reassemble(poll->ADDR_HI_UNION.addr_63_32, poll->ADDR_LO_UNION.addr_31_3 << 3),
            reinterpret_cast<uintptr_t>(FakeAddress(0x1000)));
  EXPECT_EQ(Reassemble(poll->REFERENCE_HI_UNION.reference_63_32,
                       poll->REFERENCE_LO_UNION.reference_31_0),
            0x0000000100000002ull);
  EXPECT_EQ(Reassemble(poll->MASK_HI_UNION.mask_63_32, poll->MASK_LO_UNION.mask_31_0),
            0xffffffff00000000ull);
}

TEST(PacketFields, Fence32OnlyCarriesScopeWhereTheHardwareHasTheField) {
  for (const char* isa_name : {"gfx1201", "gfx1250"}) {
    SCOPED_TRACE(isa_name);
    const abce_builder_t builder = MakeBuilder(isa_name);
    alignas(16) uint8_t packet[sizeof(SDMA_PKT_FENCE_GFX12)] = {};
    uint32_t fence_word = 0;
    abce_builder_build_fence(&builder, reinterpret_cast<char*>(packet), &fence_word, 0xabcd);
    const auto* fence = reinterpret_cast<const SDMA_PKT_FENCE_GFX12*>(packet);
    EXPECT_EQ(fence->HEADER_UNION.sys, 1u);
    EXPECT_EQ(fence->HEADER_UNION.mtype, 3u);
    EXPECT_EQ(fence->HEADER_UNION.scope,
              builder.scope_fields ? SDMA_MEMORY_SCOPE_SYS : SDMA_MEMORY_SCOPE_CU);
    EXPECT_EQ(fence->DATA_UNION.data, 0xabcdu);
  }
}

//===----------------------------------------------------------------------===//
// Fused wait/signal packets (ROCr 6691fe3e619: wait_scope left at CU)
//===----------------------------------------------------------------------===//

TEST(PacketFields, FusedLinearWaitAndSignalAreSystemScope) {
  const abce_builder_t builder = MakeBuilder("gfx1250");
  const abce_wait_signal_params_t params = WaitAndSignal();
  alignas(16) uint8_t packet[256] = {};
  abce_builder_build_wait_signal_copy(&builder, reinterpret_cast<char*>(packet), FakeAddress(0x9000),
                                      FakeAddress(0x8000), 4096, &params);
  const FusedBlocks blocks = InspectFused(packet, IsaByName("gfx1250").isa);
  EXPECT_EQ(blocks.emitted_dwords, 1u + 7 + 6 + 5);
  ExpectWaitAndSignalBlocksAreSystemScope(blocks);
}

TEST(PacketFields, FusedSwapWaitAndSignalAreSystemScope) {
  const abce_builder_t builder = MakeBuilder("gfx1250");
  const abce_wait_signal_params_t params = WaitAndSignal();
  alignas(16) uint8_t packet[256] = {};
  abce_builder_build_wait_signal_swap(&builder, reinterpret_cast<char*>(packet),
                                      FakeAddress(0x9000), FakeAddress(0x8000), 4096, 4096, &params);
  const FusedBlocks blocks = InspectFused(packet, IsaByName("gfx1250").isa);
  ExpectWaitAndSignalBlocksAreSystemScope(blocks);
}

TEST(PacketFields, FusedMulticastWaitAndSignalAreSystemScope) {
  const abce_builder_t builder = MakeBuilder("gfx1250");
  const abce_wait_signal_params_t params = WaitAndSignal();
  void* destinations[3] = {FakeAddress(0x9000), FakeAddress(0xa000), FakeAddress(0xb000)};
  alignas(16) uint8_t packet[256] = {};
  abce_builder_build_multicast_wait_signal_copy(&builder, reinterpret_cast<char*>(packet),
                                                destinations, 3, FakeAddress(0x8000), 4096, &params);
  const FusedBlocks blocks = InspectFused(packet, IsaByName("gfx1250").isa);
  ExpectWaitAndSignalBlocksAreSystemScope(blocks);
}

TEST(PacketFields, FusedIndirectWaitAndSignalAreSystemScope) {
  const abce_builder_t builder = MakeBuilder("gfx1250");
  const abce_wait_signal_params_t params = WaitAndSignal();
  alignas(16) uint8_t packet[256] = {};
  abce_builder_build_wait_signal_indirect_copy(&builder, reinterpret_cast<char*>(packet),
                                               FakeAddress(0x9000), FakeAddress(0x8000), 4096,
                                               /*indirect_src=*/true, /*indirect_dst=*/false, &params);
  const FusedBlocks blocks = InspectFused(packet, IsaByName("gfx1250").isa);
  EXPECT_EQ(blocks.emitted_dwords, 1u + 7 + 6 + 5);
  ExpectWaitAndSignalBlocksAreSystemScope(blocks);
}

//===----------------------------------------------------------------------===//
// Compacted emission (ROCr ff0da4bb350: full-struct layout faulted at 0x0)
//===----------------------------------------------------------------------===//

// The engine parses a fused packet as header, then each block the header says is
// present, with nothing reserved for an absent one. Emitting the full struct
// leaves the copy body where the engine, seeing wait=0, reads the source
// address, and it dereferences zero.
TEST(PacketFields, IndirectCopyIsCompactedWhenWaitAndSignalAreAbsent) {
  const abce_builder_t builder = MakeBuilder("gfx1250");
  const uint64_t source = reinterpret_cast<uintptr_t>(FakeAddress(0x8000));
  const uint64_t destination = reinterpret_cast<uintptr_t>(FakeAddress(0x9000));
  constexpr size_t kSize = 4096;
  constexpr uint32_t kCopyBodyDwords = 6;

  struct Case {
    bool wait;
    bool signal;
  };
  for (const Case shape : {Case{false, false}, Case{true, false}, Case{false, true}, Case{true, true}}) {
    SCOPED_TRACE(::testing::Message() << "wait=" << shape.wait << " signal=" << shape.signal);
    abce_wait_signal_params_t params = WaitAndSignal();
    if (!shape.wait) params.wait_addr = nullptr;
    if (!shape.signal) params.signal_addr = nullptr;

    alignas(16) uint8_t packet[256];
    // Poisoned, not zeroed: fused builders do not rely on the zeroed-buffer contract.
    std::memset(packet, 0xee, sizeof(packet));
    abce_builder_build_wait_signal_indirect_copy(&builder, reinterpret_cast<char*>(packet),
                                                 FakeAddress(0x9000), FakeAddress(0x8000), kSize,
                                                 true, false, &params);
    const uint32_t* words = reinterpret_cast<const uint32_t*>(packet);
    const uint32_t expected_dwords =
        1 + (shape.wait ? kWaitBlockDwords : 0) + kCopyBodyDwords + (shape.signal ? kSignalBlockDwords : 0);
    const abce_decoded_packet_t decoded =
        abce_decode_sdma_packet(words, 64, IsaByName("gfx1250").isa);
    EXPECT_EQ(decoded.dwords, expected_dwords);

    // The copy body starts right after the header or the wait block.
    const uint32_t* copy_body = words + 1 + (shape.wait ? kWaitBlockDwords : 0);
    EXPECT_EQ(copy_body[0], kSize - 1) << "copy_count";
    EXPECT_EQ(Reassemble(copy_body[3], copy_body[2]), source);
    EXPECT_EQ(Reassemble(copy_body[5], copy_body[4]), destination);
  }
}

}  // namespace
