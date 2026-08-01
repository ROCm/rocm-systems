// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "embedded_schema.h"
#include "rocjitsu/config/config_loader.h"
#include "rocjitsu/vm/amdgpu/command_processor.h"
#include "rocjitsu/vm/soc.h"

#include "simdojo/sim/simulation.h"

#include <gtest/gtest.h>

#include <array>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <utility>

namespace {

using namespace rocjitsu;

constexpr uint32_t kFusedCopyDwords = 19;
constexpr uint32_t kLinearCopyDwords = 7;
constexpr uint32_t kBroadcastCopyDwords = 9;
constexpr uint32_t kFence64Dwords = 5;
constexpr uint32_t kCopyBytes = 64;
constexpr uint32_t kOpCopy = 1;
constexpr uint32_t kOpFence = 5;
constexpr uint32_t kSubopFence64 = 2;
constexpr uint32_t kHeaderBroadcast = 1u << 27;
constexpr uint32_t kHeaderNpd = 1u << 28;
constexpr uint32_t kHeaderWait = 1u << 30;
constexpr uint32_t kHeaderSignal = 1u << 31;
constexpr uint32_t kWaitFunctionGe = 5;
constexpr uint32_t kSignalOperationAdd64 = 0x6F;

void write_address(uint32_t *packet, uint32_t lo_dw, uint32_t hi_dw, const void *address) {
  const uint64_t va = reinterpret_cast<uint64_t>(address);
  packet[lo_dw] = static_cast<uint32_t>(va) & ~0x7u;
  packet[hi_dw] = static_cast<uint32_t>(va >> 32);
}

TEST(Cdna4SdmaTest, ProducerPacketsUseOss7Dialect) {
  auto loaded = config::load_config(std::string(CONFIG_DIR) + "/gfx950_cdna4.json",
                                    rocjitsu::kEmbeddedSchema);
  auto *memory = loaded.memory();
  auto *cp = loaded.soc()->xcd(0)->command_processor();
  ASSERT_EQ(cp->sdma_packet_dialect(), amdgpu::SdmaPacketDialect::Oss7);

  auto engine = std::make_unique<simdojo::SimulationEngine>(loaded.engine_config);
  engine->topology().set_root(loaded.take_root());
  loaded.wire_links(engine->topology());
  engine->create();
  memory->set_passthrough(true);

  alignas(8) std::array<uint32_t, 128> ring{};
  alignas(8) uint64_t read_idx = 0;
  alignas(8) uint64_t write_idx = 0;
  alignas(8) std::array<uint64_t, 1> doorbells{};
  alignas(32) std::array<uint8_t, 4 * kCopyBytes> src{};
  alignas(32) std::array<uint8_t, 4 * kCopyBytes> dst{};
  alignas(32) std::array<uint8_t, kCopyBytes> broadcast_dst1{};
  alignas(32) std::array<uint8_t, kCopyBytes> broadcast_dst2{};
  alignas(8) uint64_t signal = 7;
  alignas(8) uint64_t poll = 4;
  alignas(8) uint64_t fence = 0;
  constexpr uint64_t kFenceValue = 0x1122334455667788ULL;
  for (uint32_t i = 0; i < src.size(); ++i)
    src[i] = static_cast<uint8_t>(i ^ 0xA5u);

  amdgpu::HwQueue queue{};
  queue.process_id = 0;
  queue.queue_id = 950;
  queue.ring_base_va = reinterpret_cast<uint64_t>(ring.data());
  queue.ring_size = static_cast<uint32_t>(ring.size() * sizeof(ring[0]));
  queue.read_ptr_va = reinterpret_cast<uint64_t>(&read_idx);
  queue.write_ptr_va = reinterpret_cast<uint64_t>(&write_idx);
  queue.doorbell_base = doorbells.data();
  queue.host_accessible = true;
  queue.is_sdma = true;
  cp->register_queue(std::move(queue));

  // rocSHMEM's fused OSS7 producer always emits the full 19-DWORD struct.
  // Disabling WAIT retains DW1-7 while COPY stays at DW8 and SIGNAL at DW14.
  uint32_t *signal_only = ring.data();
  signal_only[0] = kOpCopy | kHeaderSignal;
  signal_only[8] = kCopyBytes - 1;
  write_address(signal_only, 10, 11, src.data());
  write_address(signal_only, 12, 13, dst.data());
  signal_only[14] = kSignalOperationAdd64;
  write_address(signal_only, 15, 16, &signal);
  signal_only[17] = 3;

  // Disabling SIGNAL retains DW14-18. Stalling this packet also pins the retry
  // boundary after the preceding fixed-size signal-only packet.
  uint32_t *wait_only = ring.data() + kFusedCopyDwords;
  wait_only[0] = kOpCopy | kHeaderWait;
  wait_only[1] = kWaitFunctionGe;
  write_address(wait_only, 2, 3, &poll);
  wait_only[4] = 5;
  wait_only[6] = 0xFFFFFFFFu;
  wait_only[7] = 0xFFFFFFFFu;
  wait_only[8] = kCopyBytes - 1;
  write_address(wait_only, 10, 11, src.data() + kCopyBytes);
  write_address(wait_only, 12, 13, dst.data() + kCopyBytes);

  // Exercise both optional blocks together and leave the destination
  // unresolved for the first attempt. The packet must retry at its own start
  // without copying or applying its signal update.
  uint32_t *wait_and_signal = wait_only + kFusedCopyDwords;
  wait_and_signal[0] = kOpCopy | kHeaderWait | kHeaderSignal;
  wait_and_signal[1] = kWaitFunctionGe;
  write_address(wait_and_signal, 2, 3, &poll);
  wait_and_signal[4] = 5;
  wait_and_signal[6] = 0xFFFFFFFFu;
  wait_and_signal[7] = 0xFFFFFFFFu;
  wait_and_signal[8] = kCopyBytes - 1;
  write_address(wait_and_signal, 10, 11, src.data() + 2 * kCopyBytes);
  wait_and_signal[14] = kSignalOperationAdd64;
  write_address(wait_and_signal, 15, 16, &signal);
  wait_and_signal[17] = 4;

  // A plain OSS7 copy shares sub-op zero with the fused form. With neither
  // WAIT nor SIGNAL set it retains the ordinary 7-DWORD framing. NPD is bit 28,
  // so it must not be mistaken for a legacy broadcast flag.
  uint32_t *plain = wait_and_signal + kFusedCopyDwords;
  plain[0] = kOpCopy | kHeaderNpd;
  plain[1] = kCopyBytes - 1;
  write_address(plain, 3, 4, src.data() + 3 * kCopyBytes);
  write_address(plain, 5, 6, dst.data() + 3 * kCopyBytes);

  // ROCr enables this shared 9-DWORD bit-27 broadcast producer for gfx950.
  uint32_t *broadcast = plain + kLinearCopyDwords;
  broadcast[0] = kOpCopy | kHeaderBroadcast;
  broadcast[1] = kCopyBytes - 1;
  write_address(broadcast, 3, 4, src.data());
  write_address(broadcast, 5, 6, broadcast_dst1.data());
  write_address(broadcast, 7, 8, broadcast_dst2.data());

  // rocSHMEM's OSS7 fence producer uses the extended 5-DWORD 64-bit layout.
  uint32_t *fence64 = broadcast + kBroadcastCopyDwords;
  fence64[0] = kOpFence | (kSubopFence64 << 8);
  write_address(fence64, 1, 2, &fence);
  fence64[3] = static_cast<uint32_t>(kFenceValue);
  fence64[4] = static_cast<uint32_t>(kFenceValue >> 32);

  constexpr uint32_t kSubmittedDwords =
      3 * kFusedCopyDwords + kLinearCopyDwords + kBroadcastCopyDwords + kFence64Dwords;
  write_idx = kSubmittedDwords * sizeof(uint32_t);
  std::atomic_ref<uint64_t>(doorbells[0]).store(write_idx, std::memory_order_release);
  engine->schedule_event_now(cp->doorbell_event());
  ASSERT_TRUE(engine->step());

  EXPECT_EQ(std::atomic_ref<uint64_t>(read_idx).load(std::memory_order_acquire),
            kFusedCopyDwords * sizeof(uint32_t));
  EXPECT_EQ(std::memcmp(dst.data(), src.data(), kCopyBytes), 0);
  EXPECT_EQ(dst[kCopyBytes], 0u);
  EXPECT_EQ(std::atomic_ref<uint64_t>(signal).load(std::memory_order_acquire), 10u);

  std::atomic_ref<uint64_t>(poll).store(5, std::memory_order_release);
  engine->schedule_event_now(cp->doorbell_event());
  ASSERT_TRUE(engine->step());
  EXPECT_EQ(std::atomic_ref<uint64_t>(read_idx).load(std::memory_order_acquire),
            2 * kFusedCopyDwords * sizeof(uint32_t));
  EXPECT_EQ(std::memcmp(dst.data(), src.data(), 2 * kCopyBytes), 0);
  EXPECT_EQ(dst[2 * kCopyBytes], 0u);
  EXPECT_EQ(std::atomic_ref<uint64_t>(signal).load(std::memory_order_acquire), 10u);

  write_address(wait_and_signal, 12, 13, dst.data() + 2 * kCopyBytes);
  engine->schedule_event_now(cp->doorbell_event());
  ASSERT_TRUE(engine->step());
  EXPECT_EQ(std::atomic_ref<uint64_t>(read_idx).load(std::memory_order_acquire), write_idx);
  EXPECT_EQ(std::memcmp(dst.data(), src.data(), src.size()), 0);
  EXPECT_EQ(std::memcmp(broadcast_dst1.data(), src.data(), kCopyBytes), 0);
  EXPECT_EQ(std::memcmp(broadcast_dst2.data(), src.data(), kCopyBytes), 0);
  EXPECT_EQ(std::atomic_ref<uint64_t>(signal).load(std::memory_order_acquire), 14u);
  EXPECT_EQ(std::atomic_ref<uint64_t>(fence).load(std::memory_order_acquire), kFenceValue);

  cp->unregister_queue(/*queue_id=*/950, /*process_id=*/0);
}

} // namespace
