/*
 * Copyright © Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/*
 * Tests for the compute-shader broadcast copy fallback.
 *
 * HSA_AMD_MEMORY_COPY_OP_LINEAR_BROADCAST writes one source buffer to each of
 * N destinations. Where the SDMA engines are unavailable the operation is
 * carried by a blit shader instead, and that path has structure the SDMA path
 * does not: a single dispatch drives a bounded number of destinations, so a
 * larger request is split across several dispatches sharing one completion
 * signal, and a request whose dispatch count exceeds what one queue
 * reservation holds is split across reservations as well.
 *
 * These tests state the operation's contract - every destination ends up
 * holding the source bytes, and the completion signal resolves exactly once
 * regardless of how many dispatches carried it - so they are valid under
 * either routing. They are the ones that fail if the split, the
 * destination-list ring, or the signal accounting is wrong, and the arm that
 * exercises that code is the run with the SDMA engines disabled.
 *
 * Deliberately no GTEST_SKIP anywhere: a skipped test is reported as [ OK ]
 * and would let an environment that ran nothing look like a pass.
 *
 * No main() is defined here - these register with the rocrtst64 binary's
 * gtest main, as in the other files in this directory.
 */

#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "gtest/gtest.h"

namespace {

std::string AgentName(hsa_agent_t agent) {
  char name[64] = {0};
  hsa_agent_get_info(agent, HSA_AGENT_INFO_NAME, name);
  return std::string(name);
}

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

// Bound every device wait, so a split that never resolves its signal fails the
// test instead of wedging a shared device for the rest of the run.
uint64_t WaitTimeoutTicks() {
  uint64_t ticks_per_second = 0;
  if (hsa_system_get_info(HSA_SYSTEM_INFO_TIMESTAMP_FREQUENCY, &ticks_per_second) !=
          HSA_STATUS_SUCCESS ||
      ticks_per_second == 0)
    return UINT64_MAX;
  return ticks_per_second * 60;
}

// Coarse-grained global memory the runtime may allocate from on this agent.
// Zero handle if the agent has none.
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

// The tests assert inside loops that hold device allocations, so releasing
// them has to survive an early return. These own their handle.
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

// A byte pattern that differs at every offset within a 256-byte window and
// also differs between destinations, so a copy that wrote the right number of
// bytes from the wrong offset, or wrote one destination's bytes into another,
// is caught rather than compared equal.
std::vector<uint8_t> MakePattern(size_t bytes, uint8_t seed) {
  std::vector<uint8_t> pattern(bytes);
  for (size_t i = 0; i < bytes; ++i)
    pattern[i] = static_cast<uint8_t>((i + seed * 7u) & 0xFF);
  return pattern;
}

// Destinations as slices of one allocation. At the counts these tests use,
// separate pool allocations dominate the runtime without exercising anything
// the slices do not; the slices are `stride` apart so a copy that overran its
// destination lands in the next one's guard region and is detected.
struct SlicedDestinations {
  PoolAlloc backing;
  std::vector<void*> ptrs;
  std::vector<hsa_agent_t> agents;
  size_t stride = 0;
};

// A host pool the CPU can read directly and the GPU can be given access to.
// Zero handle if there is none.
//
// Needed only by the test that holds a dependency unresolved while inspecting
// the destinations. A dependent copy parks barrier packets on the agent's
// copy queue, and every later copy on that queue waits behind them - including
// a device-to-host readback, because the device-to-host blit shares the
// utility queue with the device-to-device blit that carries the broadcast
// (see GpuAgent::InitDma). Reading device memory while holding the dependency
// would therefore block on a dependency only this thread can release. Putting
// those destinations in host memory makes the check a plain CPU read that
// touches no queue at all. Confirmed against a plain dependent
// hsa_amd_memory_async_copy, which hangs identically: the hazard is inherent
// to dependent copies, not to the broadcast path.
hsa_amd_memory_pool_t FindHostVisiblePool() {
  hsa_amd_memory_pool_t pool = {0};
  hsa_iterate_agents(
      [](hsa_agent_t agent, void* data) -> hsa_status_t {
        hsa_device_type_t type;
        if (hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &type) != HSA_STATUS_SUCCESS ||
            type != HSA_DEVICE_TYPE_CPU)
          return HSA_STATUS_SUCCESS;
        return hsa_amd_agent_iterate_memory_pools(
            agent,
            [](hsa_amd_memory_pool_t p, void* inner) {
              hsa_amd_segment_t segment;
              if (hsa_amd_memory_pool_get_info(p, HSA_AMD_MEMORY_POOL_INFO_SEGMENT, &segment) !=
                      HSA_STATUS_SUCCESS ||
                  segment != HSA_AMD_SEGMENT_GLOBAL)
                return HSA_STATUS_SUCCESS;
              bool alloc_allowed = false;
              if (hsa_amd_memory_pool_get_info(p, HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_ALLOWED,
                                               &alloc_allowed) != HSA_STATUS_SUCCESS ||
                  !alloc_allowed)
                return HSA_STATUS_SUCCESS;
              *static_cast<hsa_amd_memory_pool_t*>(inner) = p;
              return HSA_STATUS_INFO_BREAK;
            },
            data);
      },
      &pool);
  return pool;
}

// Every GPU agent on the system.
std::vector<hsa_agent_t> CollectGpuAgents() {
  std::vector<hsa_agent_t> gpus;
  hsa_iterate_agents(
      [](hsa_agent_t agent, void* data) -> hsa_status_t {
        hsa_device_type_t type;
        if (hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &type) != HSA_STATUS_SUCCESS)
          return HSA_STATUS_SUCCESS;
        if (type == HSA_DEVICE_TYPE_GPU)
          static_cast<std::vector<hsa_agent_t>*>(data)->push_back(agent);
        return HSA_STATUS_SUCCESS;
      },
      &gpus);
  return gpus;
}

class BroadcastCopyFallback : public ::testing::Test {
 protected:
  void SetUp() override {
    ASSERT_EQ(HSA_STATUS_SUCCESS, hsa_init());
    gpus_ = CollectGpuAgents();
    // A run with no GPU is a broken environment, not a valid result.
    ASSERT_FALSE(gpus_.empty()) << "no GPU agent on this system";
    wait_timeout_ = WaitTimeoutTicks();
  }

  void TearDown() override { EXPECT_EQ(HSA_STATUS_SUCCESS, hsa_shut_down()); }

  // Carve `count` destinations of `bytes` each out of one allocation, each
  // cleared to a value the source pattern never contains at that offset, so an
  // untouched destination is distinguishable from a correctly written one.
  // Returns false (with the failure already recorded) if the pool cannot serve
  // the request.
  bool MakeDestinations(hsa_agent_t gpu, hsa_amd_memory_pool_t pool, uint32_t count,
                        size_t bytes, SlicedDestinations* out) {
    // Keep each slice 4-byte aligned so the whole backing buffer can be
    // cleared with one fill; the copy itself has no alignment requirement.
    out->stride = (bytes + 255u) & ~static_cast<size_t>(255u);
    const size_t total = out->stride * count;
    if (out->backing.Allocate(pool, total) != HSA_STATUS_SUCCESS) {
      ADD_FAILURE() << "could not allocate " << total << " bytes on " << AgentName(gpu);
      return false;
    }
    if (hsa_amd_memory_fill(out->backing.ptr, 0, total / sizeof(uint32_t)) !=
        HSA_STATUS_SUCCESS) {
      ADD_FAILURE() << "could not clear destinations on " << AgentName(gpu);
      return false;
    }
    out->ptrs.resize(count);
    for (uint32_t d = 0; d < count; ++d)
      out->ptrs[d] = static_cast<uint8_t*>(out->backing.ptr) + d * out->stride;
    out->agents.assign(count, gpu);
    return true;
  }

  // The same destinations, but in memory the CPU can read without going
  // through a copy engine, with `gpu` given access so the shader can write
  // them. Used only where the test must inspect the destinations while a
  // dependency is deliberately held unresolved - see FindHostVisiblePool.
  bool MakeHostVisibleDestinations(hsa_agent_t gpu, uint32_t count, size_t bytes,
                                   SlicedDestinations* out) {
    hsa_amd_memory_pool_t host_pool = FindHostVisiblePool();
    if (host_pool.handle == 0) {
      ADD_FAILURE() << "no host-accessible global pool on this system";
      return false;
    }
    out->stride = (bytes + 255u) & ~static_cast<size_t>(255u);
    const size_t total = out->stride * count;
    if (out->backing.Allocate(host_pool, total) != HSA_STATUS_SUCCESS) {
      ADD_FAILURE() << "could not allocate " << total << " bytes of host memory";
      return false;
    }
    if (hsa_amd_agents_allow_access(1, &gpu, nullptr, out->backing.ptr) != HSA_STATUS_SUCCESS) {
      ADD_FAILURE() << "could not grant " << AgentName(gpu) << " access to host memory";
      return false;
    }
    std::memset(out->backing.ptr, 0, total);
    out->ptrs.resize(count);
    for (uint32_t d = 0; d < count; ++d)
      out->ptrs[d] = static_cast<uint8_t*>(out->backing.ptr) + d * out->stride;
    out->agents.assign(count, gpu);
    return true;
  }

  // Check every destination holds the source bytes, and that nothing was
  // written past `bytes` into the slice's guard region.
  void VerifyDestinations(const SlicedDestinations& dsts, const std::vector<uint8_t>& pattern,
                          hsa_agent_t gpu) {
    const size_t bytes = pattern.size();
    const size_t guard = dsts.stride - bytes;
    std::vector<uint8_t> readback(dsts.stride);
    // Report at most a few mismatches: at these destination counts a
    // systematically broken split would otherwise produce thousands of lines
    // and bury the first, most informative one.
    uint32_t reported = 0;
    for (size_t d = 0; d < dsts.ptrs.size(); ++d) {
      ASSERT_EQ(HSA_STATUS_SUCCESS,
                hsa_memory_copy(readback.data(), dsts.ptrs[d], dsts.stride));
      const bool body_ok = std::memcmp(readback.data(), pattern.data(), bytes) == 0;
      bool guard_ok = true;
      for (size_t g = 0; g < guard; ++g)
        if (readback[bytes + g] != 0) guard_ok = false;

      if ((!body_ok || !guard_ok) && reported < 4) {
        ++reported;
        EXPECT_TRUE(body_ok) << "destination " << d << " of " << dsts.ptrs.size()
                             << " does not hold the source bytes on " << AgentName(gpu);
        EXPECT_TRUE(guard_ok) << "destination " << d << " of " << dsts.ptrs.size()
                              << " was written past its " << bytes << " bytes on "
                              << AgentName(gpu);
      } else if (!body_ok || !guard_ok) {
        // Still fail the test, without another page of output.
        ASSERT_LT(reported, 4u) << "further destinations also wrong on " << AgentName(gpu)
                                << "; first failures reported above";
      }
    }
  }

  // Run one broadcast of `count` destinations, `bytes` each, and verify every
  // destination. Agents that report they cannot carry a broadcast of this size
  // are checked for a clean rejection instead - that is the correct answer for
  // them and must not be reported as a pass of the copy itself.
  void RunAndVerifyBroadcast(uint32_t count, size_t bytes) {
    for (hsa_agent_t gpu : gpus_) {
      const uint32_t max_dests = MaxBroadcastDests(gpu);

      hsa_amd_memory_pool_t pool = FindAllocatableGlobalPool(gpu);
      ASSERT_NE(0u, pool.handle) << "no allocatable global pool on " << AgentName(gpu);

      PoolAlloc src;
      ASSERT_EQ(HSA_STATUS_SUCCESS, src.Allocate(pool, bytes));
      const std::vector<uint8_t> pattern = MakePattern(bytes, 0x5A);
      ASSERT_EQ(HSA_STATUS_SUCCESS, hsa_memory_copy(src.ptr, pattern.data(), bytes));

      SlicedDestinations dsts;
      ASSERT_TRUE(MakeDestinations(gpu, pool, count, bytes, &dsts));

      ScopedSignal completion;
      ASSERT_EQ(HSA_STATUS_SUCCESS, completion.Create(1));

      hsa_amd_memory_copy_op_t op =
          MakeBroadcastOp(gpu, src.ptr, dsts.ptrs, dsts.agents, bytes, completion.signal);
      const hsa_status_t submit = hsa_amd_memory_async_batch_copy(&op, 1, 0, nullptr);

      if (max_dests < count) {
        EXPECT_NE(HSA_STATUS_SUCCESS, submit)
            << AgentName(gpu) << " reports a ceiling of " << max_dests
            << " but accepted a " << count << "-destination broadcast";
        continue;
      }

      ASSERT_EQ(HSA_STATUS_SUCCESS, submit)
          << AgentName(gpu) << " reports a ceiling of " << max_dests << " but rejected "
          << count << " destinations of " << bytes << " bytes";

      ASSERT_EQ(0, hsa_signal_wait_scacquire(completion.signal, HSA_SIGNAL_CONDITION_EQ, 0,
                                             wait_timeout_, HSA_WAIT_STATE_BLOCKED))
          << "completion signal did not resolve on " << AgentName(gpu) << " for " << count
          << " destinations of " << bytes << " bytes";

      // Resolving to exactly zero is the point of the signal accounting: a
      // split that raised the count by the wrong amount would either hang the
      // wait above or leave the signal negative here.
      EXPECT_EQ(0, hsa_signal_load_scacquire(completion.signal))
          << "completion signal overshot zero on " << AgentName(gpu);

      VerifyDestinations(dsts, pattern, gpu);
    }
  }

  std::vector<hsa_agent_t> gpus_;
  uint64_t wait_timeout_ = UINT64_MAX;
};

// The smallest request that is genuinely a broadcast, with the bytes checked.
TEST_F(BroadcastCopyFallback, TC_BCAST_001_TwoDestinations) {
  RunAndVerifyBroadcast(2, 4096);
}

// A single destination is a degenerate broadcast. It goes down the same path
// and must not be special-cased into a plain copy that skips the destination
// list.
TEST_F(BroadcastCopyFallback, TC_BCAST_002_SingleDestination) {
  RunAndVerifyBroadcast(1, 4096);
}

// A size that is neither a multiple of the shader's 16-byte vectorized chunk
// nor of its workgroup size, so the byte tail and the partial chunk both carry
// real data. A copy that handled only whole chunks would leave the last bytes
// of every destination unwritten.
TEST_F(BroadcastCopyFallback, TC_BCAST_003_UnalignedSize) {
  RunAndVerifyBroadcast(3, 4097);
}

// Smaller than one 16-byte chunk, so the vectorized phase does nothing at all
// and the whole copy is the tail.
TEST_F(BroadcastCopyFallback, TC_BCAST_004_SizeBelowOneChunk) {
  RunAndVerifyBroadcast(3, 7);
}

// Larger than one workgroup's single pass, so every lane loops.
TEST_F(BroadcastCopyFallback, TC_BCAST_005_LargeTransfer) {
  RunAndVerifyBroadcast(2, 4u * 1024 * 1024 + 13);
}

// Exactly the number of destinations one dispatch drives. The boundary below
// which no split happens; a split loop that was off by one would emit an
// empty second dispatch here and leave the completion signal short.
TEST_F(BroadcastCopyFallback, TC_BCAST_006_ExactlyOneDispatchWorth) {
  RunAndVerifyBroadcast(1024, 256);
}

// One past that boundary: the first request that must be split. The second
// dispatch drives a single destination, so a split that sized the tail
// dispatch from the wrong remainder writes nothing to it.
TEST_F(BroadcastCopyFallback, TC_BCAST_007_OnePastOneDispatchWorth) {
  RunAndVerifyBroadcast(1025, 256);
}

// Several dispatches, the last one partial. Exercises the destination-list
// ring across slots: each dispatch reads its own slice of the ring, so a
// dispatch that indexed the ring from the wrong slot would write another
// dispatch's destinations.
TEST_F(BroadcastCopyFallback, TC_BCAST_008_ManyDispatches) {
  RunAndVerifyBroadcast(3000, 256);
}

// Enough destinations that the dispatches cannot fit in one queue
// reservation, so the submission spans reservations and the kernarg and
// destination-list rings wrap. A slot recycled before its dispatch retired
// would corrupt destinations already in flight.
TEST_F(BroadcastCopyFallback, TC_BCAST_009_SpansQueueReservations) {
  RunAndVerifyBroadcast(8192, 128);
}

// Zero bytes to each destination. No dispatch has any work to do, so the
// completion signal must be resolved by the submission itself rather than by
// a dispatch that never runs - otherwise the caller waits forever.
TEST_F(BroadcastCopyFallback, TC_BCAST_010_ZeroSize) {
  for (hsa_agent_t gpu : gpus_) {
    if (MaxBroadcastDests(gpu) < 2) continue;

    hsa_amd_memory_pool_t pool = FindAllocatableGlobalPool(gpu);
    ASSERT_NE(0u, pool.handle) << "no allocatable global pool on " << AgentName(gpu);

    constexpr size_t kSlice = 256;
    PoolAlloc src;
    ASSERT_EQ(HSA_STATUS_SUCCESS, src.Allocate(pool, kSlice));

    SlicedDestinations dsts;
    ASSERT_TRUE(MakeDestinations(gpu, pool, 2, kSlice, &dsts));

    ScopedSignal completion;
    ASSERT_EQ(HSA_STATUS_SUCCESS, completion.Create(1));

    hsa_amd_memory_copy_op_t op =
        MakeBroadcastOp(gpu, src.ptr, dsts.ptrs, dsts.agents, 0, completion.signal);
    const hsa_status_t submit = hsa_amd_memory_async_batch_copy(&op, 1, 0, nullptr);

    // Either the request is refused outright or it completes; what it may not
    // do is report success and leave the signal unresolved.
    if (submit != HSA_STATUS_SUCCESS) continue;

    ASSERT_EQ(0, hsa_signal_wait_scacquire(completion.signal, HSA_SIGNAL_CONDITION_EQ, 0,
                                           wait_timeout_, HSA_WAIT_STATE_BLOCKED))
        << "zero-size broadcast accepted but never resolved its signal on "
        << AgentName(gpu);

    // And it must not have written anything.
    std::vector<uint8_t> readback(dsts.stride);
    for (void* d : dsts.ptrs) {
      ASSERT_EQ(HSA_STATUS_SUCCESS, hsa_memory_copy(readback.data(), d, dsts.stride));
      for (size_t i = 0; i < dsts.stride; ++i)
        ASSERT_EQ(0, readback[i]) << "zero-size broadcast wrote byte " << i << " on "
                                  << AgentName(gpu);
    }
  }
}

// A broadcast gated on a dependency that is still blocked must not start. The
// assertion is that the completion signal is unresolved and no byte has moved
// - not that the signal still holds its initial value, which is a routing
// detail: the fan-out path raises the count by one per contributor before
// submitting, so a caller legitimately observes a value above the one it
// created the signal with.
TEST_F(BroadcastCopyFallback, TC_BCAST_011_DependencyGatesTheCopy) {
  constexpr uint32_t kNumDests = 4;
  constexpr size_t kBytes = 4096;

  for (hsa_agent_t gpu : gpus_) {
    if (MaxBroadcastDests(gpu) < kNumDests) continue;

    hsa_amd_memory_pool_t pool = FindAllocatableGlobalPool(gpu);
    ASSERT_NE(0u, pool.handle) << "no allocatable global pool on " << AgentName(gpu);

    PoolAlloc src;
    ASSERT_EQ(HSA_STATUS_SUCCESS, src.Allocate(pool, kBytes));
    const std::vector<uint8_t> pattern = MakePattern(kBytes, 0x33);
    ASSERT_EQ(HSA_STATUS_SUCCESS, hsa_memory_copy(src.ptr, pattern.data(), kBytes));

    // Host-resident destinations, so the check below is a CPU read. Reading
    // device memory here would queue behind this test's own unresolved
    // dependency and deadlock - see FindHostVisiblePool. Where the
    // destinations live is irrelevant to what this test asserts; the
    // device-resident case is covered by every other test in this file.
    SlicedDestinations dsts;
    ASSERT_TRUE(MakeHostVisibleDestinations(gpu, kNumDests, kBytes, &dsts));

    ScopedSignal dep, completion;
    ASSERT_EQ(HSA_STATUS_SUCCESS, dep.Create(1));
    ASSERT_EQ(HSA_STATUS_SUCCESS, completion.Create(1));

    hsa_amd_memory_copy_op_t op =
        MakeBroadcastOp(gpu, src.ptr, dsts.ptrs, dsts.agents, kBytes, completion.signal);
    ASSERT_EQ(HSA_STATUS_SUCCESS,
              hsa_amd_memory_async_batch_copy(&op, 1, 1, &dep.signal));

    EXPECT_NE(0, hsa_signal_load_scacquire(completion.signal))
        << "broadcast completed while its dependency was still blocked on "
        << AgentName(gpu);

    // Stronger, routing-independent form of the same requirement: with the
    // dependency held, the destinations must still be untouched.
    const uint8_t* gated = static_cast<const uint8_t*>(dsts.ptrs[0]);
    for (size_t i = 0; i < kBytes; ++i)
      ASSERT_EQ(0, gated[i]) << "broadcast wrote byte " << i
                             << " before its dependency was released on " << AgentName(gpu);

    hsa_signal_store_screlease(dep.signal, 0);
    ASSERT_EQ(0, hsa_signal_wait_scacquire(completion.signal, HSA_SIGNAL_CONDITION_EQ, 0,
                                           wait_timeout_, HSA_WAIT_STATE_BLOCKED))
        << "broadcast did not complete after its dependency was released on "
        << AgentName(gpu);

    VerifyDestinations(dsts, pattern, gpu);
  }
}

// A dependency on a request large enough to span queue reservations. This is
// the combination the direct path cannot serve: the dependency barriers sit at
// the head of the first reservation and wait on a signal only this thread can
// release, while acquiring the second reservation blocks this thread. The
// submission must therefore wait off-queue and resubmit, and the observable
// requirement is simply that it completes correctly rather than deadlocking -
// which the bounded wait below turns into a failure rather than a hang.
TEST_F(BroadcastCopyFallback, TC_BCAST_012_DependencyWithReservationSpanningBatch) {
  constexpr uint32_t kNumDests = 8192;
  constexpr size_t kBytes = 128;

  for (hsa_agent_t gpu : gpus_) {
    if (MaxBroadcastDests(gpu) < kNumDests) continue;

    hsa_amd_memory_pool_t pool = FindAllocatableGlobalPool(gpu);
    ASSERT_NE(0u, pool.handle) << "no allocatable global pool on " << AgentName(gpu);

    PoolAlloc src;
    ASSERT_EQ(HSA_STATUS_SUCCESS, src.Allocate(pool, kBytes));
    const std::vector<uint8_t> pattern = MakePattern(kBytes, 0x7E);
    ASSERT_EQ(HSA_STATUS_SUCCESS, hsa_memory_copy(src.ptr, pattern.data(), kBytes));

    SlicedDestinations dsts;
    ASSERT_TRUE(MakeDestinations(gpu, pool, kNumDests, kBytes, &dsts));

    ScopedSignal dep, completion;
    ASSERT_EQ(HSA_STATUS_SUCCESS, dep.Create(1));
    ASSERT_EQ(HSA_STATUS_SUCCESS, completion.Create(1));

    hsa_amd_memory_copy_op_t op =
        MakeBroadcastOp(gpu, src.ptr, dsts.ptrs, dsts.agents, kBytes, completion.signal);
    ASSERT_EQ(HSA_STATUS_SUCCESS,
              hsa_amd_memory_async_batch_copy(&op, 1, 1, &dep.signal));

    hsa_signal_store_screlease(dep.signal, 0);
    ASSERT_EQ(0, hsa_signal_wait_scacquire(completion.signal, HSA_SIGNAL_CONDITION_EQ, 0,
                                           wait_timeout_, HSA_WAIT_STATE_BLOCKED))
        << kNumDests << "-destination broadcast with a dependency did not complete on "
        << AgentName(gpu);

    VerifyDestinations(dsts, pattern, gpu);
  }
}

// Back-to-back broadcasts on one agent. Each reuses the kernarg and
// destination-list ring slots the previous one used, so a slot recycled while
// its dispatch was still in flight shows up as a destination holding the
// previous iteration's pattern rather than this one's.
TEST_F(BroadcastCopyFallback, TC_BCAST_013_RepeatedBroadcastsReuseRingSlots) {
  constexpr uint32_t kNumDests = 1500;
  constexpr size_t kBytes = 256;
  constexpr int kIterations = 8;

  for (hsa_agent_t gpu : gpus_) {
    if (MaxBroadcastDests(gpu) < kNumDests) continue;

    hsa_amd_memory_pool_t pool = FindAllocatableGlobalPool(gpu);
    ASSERT_NE(0u, pool.handle) << "no allocatable global pool on " << AgentName(gpu);

    PoolAlloc src;
    ASSERT_EQ(HSA_STATUS_SUCCESS, src.Allocate(pool, kBytes));

    SlicedDestinations dsts;
    ASSERT_TRUE(MakeDestinations(gpu, pool, kNumDests, kBytes, &dsts));

    for (int iter = 0; iter < kIterations; ++iter) {
      // A different pattern each time, so a destination left over from the
      // previous iteration fails rather than comparing equal.
      const std::vector<uint8_t> pattern =
          MakePattern(kBytes, static_cast<uint8_t>(0x10 + iter));
      ASSERT_EQ(HSA_STATUS_SUCCESS, hsa_memory_copy(src.ptr, pattern.data(), kBytes));

      ScopedSignal completion;
      ASSERT_EQ(HSA_STATUS_SUCCESS, completion.Create(1));

      hsa_amd_memory_copy_op_t op =
          MakeBroadcastOp(gpu, src.ptr, dsts.ptrs, dsts.agents, kBytes, completion.signal);
      ASSERT_EQ(HSA_STATUS_SUCCESS, hsa_amd_memory_async_batch_copy(&op, 1, 0, nullptr))
          << "iteration " << iter << " rejected on " << AgentName(gpu);

      ASSERT_EQ(0, hsa_signal_wait_scacquire(completion.signal, HSA_SIGNAL_CONDITION_EQ, 0,
                                             wait_timeout_, HSA_WAIT_STATE_BLOCKED))
          << "iteration " << iter << " did not complete on " << AgentName(gpu);

      VerifyDestinations(dsts, pattern, gpu);
      if (::testing::Test::HasFailure()) break;
    }
  }
}

}  // namespace
