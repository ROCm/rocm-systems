/*
 * Copyright © Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/*
 * Tests for HSA_AMD_AGENT_INFO_MAX_BROADCAST_DESTINATIONS.
 *
 * The attribute tells a caller how many destinations a single
 * HSA_AMD_MEMORY_COPY_OP_LINEAR_BROADCAST submitted to an agent may carry,
 * with zero meaning the agent cannot service a broadcast at all. A query that
 * only reports is worth little, so the tests here check the number against the
 * behaviour it predicts, in both directions: an agent reporting a non-zero
 * ceiling must accept a broadcast up to that count and reject one above it,
 * and an agent reporting zero must reject a broadcast outright.
 *
 * No main() is defined here - these register with the rocrtst64 binary's
 * gtest main, as in the other files in this directory.
 */

#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>

#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include "gtest/gtest.h"

namespace {

// Every agent on the system, split by device type. Collected once so a test
// can make an assertion about all of them rather than about whichever agent
// happened to be found first.
struct AgentSet {
  std::vector<hsa_agent_t> gpus;
  std::vector<hsa_agent_t> cpus;
};

hsa_status_t CollectAgent(hsa_agent_t agent, void* data) {
  AgentSet* set = static_cast<AgentSet*>(data);
  hsa_device_type_t type;
  hsa_status_t err = hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &type);
  if (err != HSA_STATUS_SUCCESS) return err;

  if (type == HSA_DEVICE_TYPE_GPU) {
    set->gpus.push_back(agent);
  } else if (type == HSA_DEVICE_TYPE_CPU) {
    set->cpus.push_back(agent);
  }
  return HSA_STATUS_SUCCESS;
}

std::string AgentName(hsa_agent_t agent) {
  char name[64] = {0};
  hsa_agent_get_info(agent, HSA_AGENT_INFO_NAME, name);
  return std::string(name);
}

// Zero on failure rather than a sentinel: callers size allocations from this
// value, and a UINT32_MAX sentinel would turn a failed query into a terabyte
// allocation request. The EXPECT_EQ already marks the test failed.
uint32_t MaxBroadcastDests(hsa_agent_t agent) {
  uint32_t max_dests = 0;
  EXPECT_EQ(HSA_STATUS_SUCCESS,
            hsa_agent_get_info(
                agent,
                static_cast<hsa_agent_info_t>(HSA_AMD_AGENT_INFO_MAX_BROADCAST_DESTINATIONS),
                &max_dests))
      << "query rejected by agent " << AgentName(agent);
  return max_dests;
}

// Report what an agent is doing instead of asserting it silently. Several of
// the tests below legitimately have nothing to do on an agent whose ceiling is
// zero, and a run where every agent reported zero would pass all six while
// exercising none of them. Printing the ceiling, and printing the counts a
// test actually drove, is what separates a real pass from a vacuous one in the
// log - without it the only way to tell them apart is to query the attribute
// from outside the suite.
void Note(const std::string& text) { std::cout << "[   INFO   ] " << text << std::endl; }

void NoteCeiling(hsa_agent_t agent, uint32_t max_dests) {
  Note(AgentName(agent) + " reports a broadcast ceiling of " + std::to_string(max_dests) +
       (max_dests == 0 ? " (cannot broadcast; the acceptance test asserts it refuses one)" : ""));
}

// The ceiling of the field the operation describes its destination count with.
// Every reported value is bounded by this.
//
// Spelled as the literal the header documents rather than re-derived from
// decltype(num_entries), which is the same expression the runtime answers the
// query with. Deriving it there and here would move both ends together and
// leave nothing to compare against. The static_assert keeps the literal
// honest: widening the field turns this into a build break here instead of a
// bound that silently follows the implementation.
uint32_t ApiCeiling() {
  static_assert(std::numeric_limits<decltype(hsa_amd_memory_copy_op_t::num_entries)>::max() ==
                    65535,
                "num_entries is no longer 16-bit; update the documented broadcast ceiling");
  return 65535;
}

// A broadcast refused within the reported ceiling has two causes that the
// public API cannot tell apart, and only one of them is this feature's bug.
// Name both at the failure site so a red run is triaged against the right one
// instead of being filed as a broadcast regression by default.
std::string RefusalWithinCeilingHint(hsa_agent_t agent, uint32_t max_dests, uint32_t asked) {
  return AgentName(agent) + " reported a ceiling of " + std::to_string(max_dests) +
         " but refused a broadcast of " + std::to_string(asked) +
         ".\nEither the broadcast path regressed, or this agent's host-to-device blit fell"
         " back to a kernel blit because an SDMA queue could not be created - the ceiling"
         " describes how the agent is configured, not whether its DMA engines came up, and"
         " HSA_AMD_AGENT_INFO_MAX_BROADCAST_DESTINATIONS documents that a copy within it may"
         " still be rejected.\nBefore treating this as a regression, check dmesg for SDMA"
         " queue-creation failures and whether another process holds the engines.";
}

// Bound every device wait, so a path that never completes fails the test at a
// named destination count instead of wedging a shared device for the run.
//
// Sixty seconds is sized for silicon, where the widest ramp step finishes in
// milliseconds. Pre-silicon emulation runs several orders of magnitude slower,
// so the same copy makes steady progress and still misses the deadline - a
// throughput result being misreported as an unserviceable ceiling. Rather than
// raise the default and blunt the test everywhere, let the environment widen it
// for those runs only.
uint64_t WaitTimeoutSeconds() {
  constexpr uint64_t kDefaultSeconds = 60;
  const char* env = std::getenv("ROCRTST_WAIT_TIMEOUT_SEC");
  if (env == nullptr) return kDefaultSeconds;
  errno = 0;
  char* end = nullptr;
  const unsigned long long parsed = std::strtoull(env, &end, 10);
  // Ignore anything that is not a clean, non-zero number: a malformed override
  // must not silently disable the bound.
  if (errno != 0 || end == env || *end != '\0' || parsed == 0) return kDefaultSeconds;
  return static_cast<uint64_t>(parsed);
}

uint64_t WaitTimeoutTicks() {
  uint64_t ticks_per_second = 0;
  if (hsa_system_get_info(HSA_SYSTEM_INFO_TIMESTAMP_FREQUENCY, &ticks_per_second) !=
          HSA_STATUS_SUCCESS ||
      ticks_per_second == 0)
    return UINT64_MAX;
  return ticks_per_second * WaitTimeoutSeconds();
}

// Coarse-grained global memory on this agent, reachable by its own engines.
// Returns a zero handle if the agent has none. The callback stops at the first
// usable pool, so iteration reports HSA_STATUS_INFO_BREAK; plain success means
// it ran off the end without finding one, which is why both are accepted and
// the handle is what actually decides.
hsa_amd_memory_pool_t FindAllocatableGlobalPool(hsa_agent_t agent) {
  hsa_amd_memory_pool_t pool = {0};
  const hsa_status_t status = hsa_amd_agent_iterate_memory_pools(
      agent,
      [](hsa_amd_memory_pool_t p, void* data) {
        hsa_amd_segment_t segment;
        if (hsa_amd_memory_pool_get_info(p, HSA_AMD_MEMORY_POOL_INFO_SEGMENT, &segment) !=
            HSA_STATUS_SUCCESS)
          return HSA_STATUS_SUCCESS;
        if (segment != HSA_AMD_SEGMENT_GLOBAL) return HSA_STATUS_SUCCESS;

        bool alloc_allowed = false;
        if (hsa_amd_memory_pool_get_info(p, HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_ALLOWED,
                                         &alloc_allowed) != HSA_STATUS_SUCCESS ||
            !alloc_allowed)
          return HSA_STATUS_SUCCESS;

        *static_cast<hsa_amd_memory_pool_t*>(data) = p;
        return HSA_STATUS_INFO_BREAK;
      },
      &pool);
  EXPECT_TRUE(status == HSA_STATUS_SUCCESS || status == HSA_STATUS_INFO_BREAK)
      << "pool iteration failed on " << AgentName(agent) << " with status " << status;
  return pool;
}

// The tests assert with ASSERT_* inside loops that hold device allocations, so
// releasing them has to survive an early return. These own their handle.
struct PoolAlloc {
  void* ptr = nullptr;
  PoolAlloc() = default;
  PoolAlloc(const PoolAlloc&) = delete;
  PoolAlloc& operator=(const PoolAlloc&) = delete;
  ~PoolAlloc() {
    if (ptr != nullptr) hsa_amd_memory_pool_free(ptr);
  }
  hsa_status_t Allocate(hsa_amd_memory_pool_t pool, size_t bytes) {
    return hsa_amd_memory_pool_allocate(pool, bytes, 0, &ptr);
  }
};

struct ScopedSignal {
  hsa_signal_t signal = {0};
  ScopedSignal() = default;
  ScopedSignal(const ScopedSignal&) = delete;
  ScopedSignal& operator=(const ScopedSignal&) = delete;
  ~ScopedSignal() {
    if (signal.handle != 0) hsa_signal_destroy(signal);
  }
  hsa_status_t Create(hsa_signal_value_t initial) {
    return hsa_signal_create(initial, 0, nullptr, &signal);
  }
};

// Fill in a broadcast of `dsts.size()` destinations from `src`, each `size`
// bytes. Shared so the acceptance and rejection arms submit an identical
// operation and differ only in the count.
hsa_amd_memory_copy_op_t MakeBroadcastOp(hsa_agent_t gpu, const void* src,
                                         std::vector<void*>& dsts,
                                         std::vector<hsa_agent_t>& dst_agents, size_t size,
                                         hsa_signal_t completion) {
  hsa_amd_memory_copy_op_t op;
  std::memset(&op, 0, sizeof(op));
  op.version = HSA_AMD_MEMORY_COPY_OP_VERSION;
  op.type = HSA_AMD_MEMORY_COPY_OP_LINEAR_BROADCAST;
  op.num_entries = static_cast<uint16_t>(dsts.size());
  op.completion_signal = completion;
  op.src = const_cast<void*>(src);
  op.src_agent = gpu;
  op.dst_list = dsts.data();
  op.dst_agent_list = dst_agents.data();
  op.size = size;
  return op;
}

class BroadcastCapabilityQuery : public ::testing::Test {
 protected:
  void SetUp() override {
    ASSERT_EQ(HSA_STATUS_SUCCESS, hsa_init());
    ASSERT_EQ(HSA_STATUS_SUCCESS, hsa_iterate_agents(CollectAgent, &agents_));
    // A machine with no GPU cannot exercise the attribute's non-trivial
    // answer. Fail rather than skip: a skipped test reports as passing here,
    // and a run with no GPU is a broken environment, not a valid result.
    ASSERT_FALSE(agents_.gpus.empty()) << "no GPU agent on this system";
  }

  void TearDown() override { EXPECT_EQ(HSA_STATUS_SUCCESS, hsa_shut_down()); }

  AgentSet agents_;
};

// The attribute must be answerable by every GPU agent, and the answer must be
// a plausible ceiling. It is deliberately not pinned to one value: an agent
// configured so that it cannot service a broadcast reports zero, and that is a
// correct answer rather than a failure. What is always true is that the value
// never exceeds the field the count is carried in, and that a non-zero value
// admits a real broadcast.
TEST_F(BroadcastCapabilityQuery, GpuAgentsReportADocumentedCeiling) {
  constexpr uint32_t kSmallestBroadcast = 2;

  for (hsa_agent_t gpu : agents_.gpus) {
    const uint32_t max_dests = MaxBroadcastDests(gpu);
    NoteCeiling(gpu, max_dests);
    EXPECT_LE(max_dests, ApiCeiling())
        << AgentName(gpu) << " reported " << max_dests
        << ", above the ceiling of the num_entries field " << ApiCeiling();
    EXPECT_TRUE(max_dests == 0 || max_dests >= kSmallestBroadcast)
        << AgentName(gpu) << " reported " << max_dests
        << ", which is neither zero nor enough for a two-destination broadcast";
  }
}

// A CPU agent has no DMA engine to carry a broadcast, so it accepts no
// destinations. Asserted separately because zero from a CPU agent is the
// correct answer rather than a missing implementation.
TEST_F(BroadcastCapabilityQuery, CpuAgentsReportZero) {
  if (agents_.cpus.empty()) return;

  for (hsa_agent_t cpu : agents_.cpus) {
    EXPECT_EQ(0u, MaxBroadcastDests(cpu)) << AgentName(cpu) << " is a CPU agent";
  }
}

// Answering the query must not change the agent. An implementation that
// resolved the answer by instantiating a copy engine would disturb the agent's
// state the first time it was asked, so ask repeatedly and require the answer
// to hold.
TEST_F(BroadcastCapabilityQuery, RepeatedQueriesAgree) {
  for (hsa_agent_t gpu : agents_.gpus) {
    const uint32_t first = MaxBroadcastDests(gpu);
    for (int i = 0; i < 8; ++i) {
      EXPECT_EQ(first, MaxBroadcastDests(gpu))
          << AgentName(gpu) << " changed its answer on query " << i + 1;
    }
  }
}

// The assertion that gives the attribute meaning: the reported ceiling has to
// predict whether a broadcast is actually accepted. A two-destination copy is
// the smallest request that is genuinely a broadcast, so it is the request
// every non-zero ceiling must admit - and, just as load-bearing, the request a
// zero ceiling must refuse. Without the zero arm the test could only be run on
// hardware that reports non-zero; an agent that truthfully reports zero (an
// APU, or any agent under HSA_ENABLE_SDMA=0) would fail a test it satisfies.
TEST_F(BroadcastCapabilityQuery, ReportedCeilingPredictsAcceptance) {
  constexpr size_t kCopyBytes = 4096;
  constexpr uint32_t kNumDests = 2;
  const uint64_t wait_timeout = WaitTimeoutTicks();

  for (hsa_agent_t gpu : agents_.gpus) {
    const uint32_t max_dests = MaxBroadcastDests(gpu);

    hsa_amd_memory_pool_t pool = FindAllocatableGlobalPool(gpu);
    ASSERT_NE(0u, pool.handle) << "no allocatable global pool on " << AgentName(gpu);

    PoolAlloc src;
    PoolAlloc dst0, dst1;
    ASSERT_EQ(HSA_STATUS_SUCCESS, src.Allocate(pool, kCopyBytes));
    ASSERT_EQ(HSA_STATUS_SUCCESS, dst0.Allocate(pool, kCopyBytes));
    ASSERT_EQ(HSA_STATUS_SUCCESS, dst1.Allocate(pool, kCopyBytes));

    // Fill the source with a pattern and clear the destinations, so a
    // completed-but-inert broadcast is distinguishable from a real one.
    std::vector<uint8_t> pattern(kCopyBytes);
    for (size_t i = 0; i < kCopyBytes; ++i) pattern[i] = static_cast<uint8_t>(i & 0xFF);
    ASSERT_EQ(HSA_STATUS_SUCCESS, hsa_memory_copy(src.ptr, pattern.data(), kCopyBytes));
    for (void* d : {dst0.ptr, dst1.ptr}) {
      ASSERT_EQ(HSA_STATUS_SUCCESS, hsa_amd_memory_fill(d, 0, kCopyBytes / sizeof(uint32_t)));
    }

    ScopedSignal completion;
    ASSERT_EQ(HSA_STATUS_SUCCESS, completion.Create(1));

    std::vector<void*> dsts = {dst0.ptr, dst1.ptr};
    std::vector<hsa_agent_t> dst_agents(kNumDests, gpu);
    hsa_amd_memory_copy_op_t op =
        MakeBroadcastOp(gpu, src.ptr, dsts, dst_agents, kCopyBytes, completion.signal);

    const hsa_status_t submit = hsa_amd_memory_async_batch_copy(&op, 1, 0, nullptr);

    if (max_dests == 0) {
      // The agent says it cannot carry a broadcast. It must actually refuse
      // one, otherwise the attribute understates a working capability and a
      // caller would fall back to a slower path for no reason.
      EXPECT_NE(HSA_STATUS_SUCCESS, submit)
          << AgentName(gpu) << " reports a ceiling of 0 but accepted a "
          << kNumDests << "-destination broadcast";
      continue;
    }

    ASSERT_GE(max_dests, kNumDests)
        << AgentName(gpu) << " reports a non-zero ceiling too small for the "
        << "smallest real broadcast";

    EXPECT_EQ(HSA_STATUS_SUCCESS, submit)
        << RefusalWithinCeilingHint(gpu, max_dests, kNumDests);

    if (submit == HSA_STATUS_SUCCESS) {
      ASSERT_EQ(0, hsa_signal_wait_scacquire(completion.signal, HSA_SIGNAL_CONDITION_EQ, 0,
                                             wait_timeout, HSA_WAIT_STATE_BLOCKED))
          << "wait returned without the copy completing on " << AgentName(gpu);

      // Every destination must hold the source bytes; a broadcast that wrote
      // only the first destination would otherwise pass on the status alone.
      for (uint32_t d = 0; d < kNumDests; ++d) {
        std::vector<uint8_t> readback(kCopyBytes, 0);
        ASSERT_EQ(HSA_STATUS_SUCCESS, hsa_memory_copy(readback.data(), dsts[d], kCopyBytes));
        EXPECT_EQ(0, std::memcmp(readback.data(), pattern.data(), kCopyBytes))
            << "destination " << d << " does not hold the source bytes on "
            << AgentName(gpu);
      }
    }
  }
}

// A broadcast is carried either by a dedicated multicast packet or by
// per-destination fan-out, and the packet describes a bounded number of
// destinations. A request past that bound must be routed to fan-out, not
// refused: the reported ceiling carries no transfer size, so if the packet's
// limit could reject a request then the accepted count would silently depend
// on the size that selects the routing, and no single reported number could be
// honest.
//
// The count here deliberately straddles that bound, and the size is kept small
// so that the multicast routing is the one that would otherwise be selected -
// a large transfer takes fan-out anyway and would exercise nothing. The
// destinations are slices of a single allocation rather than separate ones,
// because at this count separate pool allocations dominate the runtime without
// testing anything the slices do not.
TEST_F(BroadcastCapabilityQuery, CountsPastTheMulticastPacketLimitAreRoutedNotRefused) {
  // The multicast packet's num_of_destination field is 10 bits holding
  // (count - 1). Mirrors BlitSdmaBase::kMaxMulticastDsts, which is internal.
  constexpr uint32_t kMulticastPacketLimit = 1024;
  constexpr uint32_t kNumDests = kMulticastPacketLimit + 1;
  constexpr size_t kCopyBytes = 256;
  const uint64_t wait_timeout = WaitTimeoutTicks();

  for (hsa_agent_t gpu : agents_.gpus) {
    const uint32_t max_dests = MaxBroadcastDests(gpu);

    // An agent that cannot broadcast at all, or whose ceiling genuinely sits
    // below this count, is covered by the other tests.
    if (max_dests < kNumDests) {
      Note(AgentName(gpu) + " ceiling " + std::to_string(max_dests) + " is below " +
           std::to_string(kNumDests) + "; nothing to route here");
      continue;
    }
    Note(AgentName(gpu) + " driving " + std::to_string(kNumDests) +
         " destinations, one past what the multicast packet can describe");

    hsa_amd_memory_pool_t pool = FindAllocatableGlobalPool(gpu);
    ASSERT_NE(0u, pool.handle) << "no allocatable global pool on " << AgentName(gpu);

    PoolAlloc src, dst_slab;
    ASSERT_EQ(HSA_STATUS_SUCCESS, src.Allocate(pool, kCopyBytes));
    ASSERT_EQ(HSA_STATUS_SUCCESS, dst_slab.Allocate(pool, size_t(kNumDests) * kCopyBytes));

    std::vector<uint8_t> pattern(kCopyBytes);
    for (size_t i = 0; i < kCopyBytes; ++i) pattern[i] = static_cast<uint8_t>((i * 7 + 3) & 0xFF);
    ASSERT_EQ(HSA_STATUS_SUCCESS, hsa_memory_copy(src.ptr, pattern.data(), kCopyBytes));
    ASSERT_EQ(HSA_STATUS_SUCCESS,
              hsa_amd_memory_fill(dst_slab.ptr, 0,
                                  (size_t(kNumDests) * kCopyBytes) / sizeof(uint32_t)));

    std::vector<void*> dsts(kNumDests);
    for (uint32_t d = 0; d < kNumDests; ++d)
      dsts[d] = static_cast<uint8_t*>(dst_slab.ptr) + size_t(d) * kCopyBytes;
    std::vector<hsa_agent_t> dst_agents(kNumDests, gpu);

    ScopedSignal completion;
    ASSERT_EQ(HSA_STATUS_SUCCESS, completion.Create(1));

    hsa_amd_memory_copy_op_t op =
        MakeBroadcastOp(gpu, src.ptr, dsts, dst_agents, kCopyBytes, completion.signal);

    const hsa_status_t submit = hsa_amd_memory_async_batch_copy(&op, 1, 0, nullptr);
    ASSERT_EQ(HSA_STATUS_SUCCESS, submit)
        << RefusalWithinCeilingHint(gpu, max_dests, kNumDests)
        << "\nThis count is one past what the multicast packet can describe, so it must"
           " fall through to per-destination fan-out rather than be refused.";

    ASSERT_EQ(0, hsa_signal_wait_scacquire(completion.signal, HSA_SIGNAL_CONDITION_EQ, 0,
                                           wait_timeout, HSA_WAIT_STATE_BLOCKED))
        << "wait returned without the copy completing on " << AgentName(gpu);

    // Check the ends and both sides of the packet boundary rather than all
    // kNumDests: a routing that dropped the overflow would leave the entries
    // just past the limit untouched, which is exactly what these catch.
    for (uint32_t d : {0u, 1u, kMulticastPacketLimit - 1, kMulticastPacketLimit}) {
      std::vector<uint8_t> readback(kCopyBytes, 0);
      ASSERT_EQ(HSA_STATUS_SUCCESS, hsa_memory_copy(readback.data(), dsts[d], kCopyBytes));
      EXPECT_EQ(0, std::memcmp(readback.data(), pattern.data(), kCopyBytes))
          << "destination " << d << " of " << kNumDests << " does not hold the source bytes on "
          << AgentName(gpu);
    }
  }
}

// A ceiling that has only ever been exercised at two destinations is an
// advertisement, not a capability. This drives the destination count up to the
// number the agent itself reports, so the attribute is held to its own claim,
// and then one past it, so the number is shown to be the actual boundary
// rather than an arbitrary smaller one.
//
// The destinations are slices of one allocation rather than separate ones: the
// operation cares that the addresses are distinct, and tens of thousands of
// individual pool allocations would measure the allocator instead of the copy.
TEST_F(BroadcastCapabilityQuery, AdvertisedCeilingIsHonoured) {
  // Small enough that an agent routing by transfer size takes its most
  // restrictive route, which is the route the reported ceiling describes.
  constexpr size_t kEntryBytes = 256;
  const uint64_t wait_timeout = WaitTimeoutTicks();

  for (hsa_agent_t gpu : agents_.gpus) {
    const uint32_t max_dests = MaxBroadcastDests(gpu);
    if (max_dests == 0) {
      Note(AgentName(gpu) + " reports no broadcast capability; the acceptance test "
           "covers the refusal this implies");
      continue;  // covered by ReportedCeilingPredictsAcceptance
    }

    std::vector<uint32_t> ramp;
    for (uint32_t n : {2u, 64u, 512u, 4096u, 16384u}) {
      if (n < max_dests) ramp.push_back(n);
    }
    ramp.push_back(max_dests);
    {
      std::string shown;
      for (uint32_t n : ramp) shown += (shown.empty() ? "" : ", ") + std::to_string(n);
      Note(AgentName(gpu) + " driving destination counts " + shown);
    }

    hsa_amd_memory_pool_t pool = FindAllocatableGlobalPool(gpu);
    ASSERT_NE(0u, pool.handle) << "no allocatable global pool on " << AgentName(gpu);

    // One extra entry so the over-the-ceiling probe below has a distinct
    // address for its final destination rather than aliasing another.
    const size_t dst_entries = static_cast<size_t>(max_dests) + 1;

    PoolAlloc src, dst_base;
    ASSERT_EQ(HSA_STATUS_SUCCESS, src.Allocate(pool, kEntryBytes));
    ASSERT_EQ(HSA_STATUS_SUCCESS, dst_base.Allocate(pool, kEntryBytes * dst_entries));

    std::vector<uint8_t> pattern(kEntryBytes);
    for (size_t i = 0; i < kEntryBytes; ++i) pattern[i] = static_cast<uint8_t>((i * 7 + 1) & 0xFF);
    ASSERT_EQ(HSA_STATUS_SUCCESS, hsa_memory_copy(src.ptr, pattern.data(), kEntryBytes));

    auto dst_at = [&](uint32_t index) {
      return static_cast<void*>(static_cast<uint8_t*>(dst_base.ptr) +
                                static_cast<size_t>(index) * kEntryBytes);
    };

    for (uint32_t n : ramp) {
      std::vector<void*> dsts(n);
      std::vector<hsa_agent_t> dst_agents(n, gpu);
      for (uint32_t d = 0; d < n; ++d) dsts[d] = dst_at(d);

      // Clear first, so a broadcast that touches only some destinations is
      // caught rather than passing on residue from the previous ramp step.
      ASSERT_EQ(HSA_STATUS_SUCCESS,
                hsa_amd_memory_fill(dst_base.ptr, 0,
                                    (kEntryBytes * static_cast<size_t>(n)) / sizeof(uint32_t)));

      ScopedSignal completion;
      ASSERT_EQ(HSA_STATUS_SUCCESS, completion.Create(1));

      hsa_amd_memory_copy_op_t op =
          MakeBroadcastOp(gpu, src.ptr, dsts, dst_agents, kEntryBytes, completion.signal);

      ASSERT_EQ(HSA_STATUS_SUCCESS, hsa_amd_memory_async_batch_copy(&op, 1, 0, nullptr))
          << RefusalWithinCeilingHint(gpu, max_dests, n);

      const hsa_signal_value_t remaining = hsa_signal_wait_scacquire(
          completion.signal, HSA_SIGNAL_CONDITION_EQ, 0, wait_timeout, HSA_WAIT_STATE_BLOCKED);
      if (remaining != 0) {
        // The copy is still running. Returning now would destroy src, dst_base
        // and the signal while the engines are mid-write, so the device would
        // fault on freed pages and abort the whole process - turning one named
        // failure into the loss of every test after it. Give it a second,
        // equally bounded window to land before letting the scope go.
        const hsa_signal_value_t after_drain =
            hsa_signal_wait_scacquire(completion.signal, HSA_SIGNAL_CONDITION_EQ, 0,
                                      wait_timeout, HSA_WAIT_STATE_BLOCKED);
        FAIL() << AgentName(gpu) << " did not complete a broadcast of " << n
               << " destinations within the timeout; the advertised ceiling of " << max_dests
               << " is not serviceable at this count. " << remaining
               << " destinations were outstanding at the deadline and " << after_drain
               << " after a second wait of the same length, so compare the two before"
               << " calling this a hang rather than a throughput shortfall";
      }

      std::vector<uint8_t> readback(kEntryBytes * static_cast<size_t>(n), 0);
      ASSERT_EQ(HSA_STATUS_SUCCESS,
                hsa_memory_copy(readback.data(), dst_base.ptr, readback.size()));
      for (uint32_t d = 0; d < n; ++d) {
        ASSERT_EQ(0, std::memcmp(readback.data() + static_cast<size_t>(d) * kEntryBytes,
                                 pattern.data(), kEntryBytes))
            << "destination " << d << " of " << n << " does not hold the source bytes on "
            << AgentName(gpu);
      }
    }

    // One past the ceiling. Only meaningful when the reported value is below
    // the num_entries field's own limit - at the field limit there is no
    // representable larger count to ask for. At this transfer size the agent
    // takes its most restrictive route, so the request the ceiling excludes
    // must be refused rather than quietly serviced by a different route.
    if (max_dests >= ApiCeiling()) {
      Note(AgentName(gpu) + " reports the num_entries field's own limit (" +
           std::to_string(max_dests) + "), so no larger count is representable and the "
           "over-the-ceiling arm cannot be driven on this agent");
    } else {
      const uint32_t over = max_dests + 1;
      std::vector<void*> dsts(over);
      std::vector<hsa_agent_t> dst_agents(over, gpu);
      for (uint32_t d = 0; d < over; ++d) dsts[d] = dst_at(d);

      ScopedSignal completion;
      ASSERT_EQ(HSA_STATUS_SUCCESS, completion.Create(1));

      hsa_amd_memory_copy_op_t op =
          MakeBroadcastOp(gpu, src.ptr, dsts, dst_agents, kEntryBytes, completion.signal);

      const hsa_status_t submit = hsa_amd_memory_async_batch_copy(&op, 1, 0, nullptr);
      EXPECT_NE(HSA_STATUS_SUCCESS, submit)
          << AgentName(gpu) << " advertises a ceiling of " << max_dests
          << " but accepted a broadcast of " << over
          << "; the reported ceiling understates what the agent can do";

      // A rejected submit must not have signalled, so nothing is waited on.
      if (submit == HSA_STATUS_SUCCESS) {
        ASSERT_EQ(0, hsa_signal_wait_scacquire(completion.signal, HSA_SIGNAL_CONDITION_EQ, 0,
                                               wait_timeout, HSA_WAIT_STATE_BLOCKED))
            << AgentName(gpu) << " accepted " << over
            << " destinations but never completed the copy";
      }
    }
  }
}

}  // namespace
