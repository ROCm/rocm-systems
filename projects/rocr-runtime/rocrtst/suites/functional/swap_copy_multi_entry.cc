/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

// =============================================================================
// Multi-entry Swap Copy Tests (AILIROCR-24)
// =============================================================================
// Coverage for the N-entry form of HSA_AMD_MEMORY_COPY_OP_LINEAR_SWAP, i.e.
// op.num_entries > 0 with src_list / dst_list / dst_agent_list / size_list.
//
// swap_copy_tests.cc only ever sets op.num_entries = 0, so the entire list
// form was untested: the per-entry walk, the split of one batch across more
// than one queue reservation, the completion-signal accounting that split
// requires, and the shader tail-byte phase for sizes that are not a multiple
// of 16. Those are exercised here.
//
// Note on zero-length entries: the API layer rejects size_list[d] == 0 for a
// multi-entry swap, so the zero-size skip inside the blit implementation is
// not reachable from the public API. TC_SWAP_ME_005 pins that rejection
// rather than asserting the skip.
// =============================================================================

#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "common/base_rocr_utils.h"
#include "common/common.h"
#include "common/helper_funcs.h"

#include "common/copy_test_context.h"

namespace {

// Deterministic, entry-distinguishing patterns. Mixing the entry index in
// means a batch that swaps the right number of bytes but pairs the wrong
// entries still fails, which a single shared pattern would not catch.
uint8_t ByteA(uint32_t entry, size_t i) {
  return static_cast<uint8_t>((i * 7 + entry * 31 + 0xAA) & 0xFF);
}

uint8_t ByteB(uint32_t entry, size_t i) {
  return static_cast<uint8_t>((i * 13 + entry * 17 + 0x55) & 0xFF);
}

void FillA(void* buf, uint32_t entry, size_t size) {
  uint8_t* p = static_cast<uint8_t*>(buf);
  for (size_t i = 0; i < size; ++i) p[i] = ByteA(entry, i);
}

void FillB(void* buf, uint32_t entry, size_t size) {
  uint8_t* p = static_cast<uint8_t*>(buf);
  for (size_t i = 0; i < size; ++i) p[i] = ByteB(entry, i);
}

// Returns the offset of the first mismatching byte, or size if the buffer
// matches. Reporting the offset makes a partial-chunk bug legible: a failure
// at a multiple of kMaxSwapBytesPerDispatch or at a 16-byte boundary points
// straight at the chunk walk or the tail phase respectively.
size_t FirstMismatchA(const void* buf, uint32_t entry, size_t size) {
  const uint8_t* p = static_cast<const uint8_t*>(buf);
  for (size_t i = 0; i < size; ++i) {
    if (p[i] != ByteA(entry, i)) return i;
  }
  return size;
}

size_t FirstMismatchB(const void* buf, uint32_t entry, size_t size) {
  const uint8_t* p = static_cast<const uint8_t*>(buf);
  for (size_t i = 0; i < size; ++i) {
    if (p[i] != ByteB(entry, i)) return i;
  }
  return size;
}

// Submit one multi-entry swap op and wait for it. Mirrors the single-entry
// wrapper in swap_copy_tests.cc, but fills the list members of the op unions
// instead of the scalar ones.
hsa_status_t SubmitMultiSwap(const std::vector<void*>& a_list, const std::vector<void*>& b_list,
                             const std::vector<size_t>& size_list,
                             std::vector<hsa_agent_t>& dst_agents, hsa_agent_t src_agent,
                             uint32_t num_dep_signals, const hsa_signal_t* dep_signals,
                             hsa_signal_t out_signal) {
  hsa_amd_memory_copy_op_t op;
  memset(&op, 0, sizeof(op));

  op.version = HSA_AMD_MEMORY_COPY_OP_VERSION;
  op.type = HSA_AMD_MEMORY_COPY_OP_LINEAR_SWAP;
  op.num_entries = static_cast<uint16_t>(a_list.size());
  op.traffic_class = 0;
  op.completion_signal = out_signal;
  op.src_list = const_cast<void**>(a_list.data());
  op.src_agent = src_agent;
  op.dst_list = const_cast<void**>(b_list.data());
  op.dst_agent_list = dst_agents.data();
  op.size_list = const_cast<size_t*>(size_list.data());

  return hsa_amd_memory_async_batch_copy(&op, 1, num_dep_signals, dep_signals);
}

// Owns the two buffer sets for a batch so an ASSERT_* mid-test cannot leak
// them. The tests allocate thousands of buffers; a leak per failure would
// exhaust the pool for every later test in the same process.
class SwapBatch {
 public:
  SwapBatch(HsaTestContext* ctx, const std::vector<size_t>& sizes) : ctx_(ctx), sizes_(sizes) {}

  ~SwapBatch() {
    for (void* p : a_) ctx_->Free(p);
    for (void* p : b_) ctx_->Free(p);
  }

  SwapBatch(const SwapBatch&) = delete;
  SwapBatch& operator=(const SwapBatch&) = delete;

  // Allocates and fills every entry. Returns false if any allocation failed.
  //
  // A failure here must never be turned into a skip by the caller: a skip that
  // stands in for "the buffers could not be obtained" reports the suite green
  // while testing nothing. entries_allocated() says how far
  // it got so the caller can report a real failure with a usable message.
  bool Allocate() {
    // Allocate directly rather than through AllocateGPUBuffer, which discards
    // the hsa_status. Keeping the status is what turns "allocation failed"
    // from an unactionable message into a diagnosis.
    auto alloc = [this](size_t size, void** out) {
      *out = nullptr;
      hsa_status_t st = hsa_amd_memory_pool_allocate(ctx_->cpu_pool, size, 0, out);
      if (st != HSA_STATUS_SUCCESS) {
        alloc_status_ = st;
        return false;
      }
      if (ctx_->gpu_agent.handle != 0) {
        hsa_amd_agents_allow_access(1, &ctx_->gpu_agent, nullptr, *out);
      }
      return true;
    };

    for (size_t e = 0; e < sizes_.size(); ++e) {
      void* pa = nullptr;
      void* pb = nullptr;
      if (!alloc(sizes_[e], &pa) || !alloc(sizes_[e], &pb)) {
        if (pa != nullptr) ctx_->Free(pa);
        if (pb != nullptr) ctx_->Free(pb);
        failed_entry_ = e;
        return false;
      }
      FillA(pa, static_cast<uint32_t>(e), sizes_[e]);
      FillB(pb, static_cast<uint32_t>(e), sizes_[e]);
      a_.push_back(pa);
      b_.push_back(pb);
    }
    dst_agents_.assign(sizes_.size(), ctx_->gpu_agent);
    return true;
  }

  std::vector<void*>& a() { return a_; }
  std::vector<void*>& b() { return b_; }
  std::vector<hsa_agent_t>& dst_agents() { return dst_agents_; }
  const std::vector<size_t>& sizes() const { return sizes_; }

  // Index of the entry whose allocation failed, and how many pairs were
  // obtained before it. Reported on failure so "could not allocate" is
  // distinguishable from "could not allocate this many".
  size_t failed_entry() const { return failed_entry_; }
  size_t entries_allocated() const { return a_.size(); }
  hsa_status_t alloc_status() const { return alloc_status_; }

 private:
  HsaTestContext* ctx_;
  std::vector<size_t> sizes_;
  std::vector<void*> a_;
  std::vector<void*> b_;
  std::vector<hsa_agent_t> dst_agents_;
  size_t failed_entry_ = 0;
  hsa_status_t alloc_status_ = HSA_STATUS_SUCCESS;
};

std::string StatusHex(hsa_status_t st) {
  char buf[16];
  snprintf(buf, sizeof(buf), "%x", static_cast<unsigned>(st));
  return std::string(buf);
}

// Explains an allocation failure in terms the reader can act on. Used with
// ASSERT_TRUE rather than GTEST_SKIP so a device that cannot supply the
// buffers is reported as a failure to test, not as a pass.
std::string AllocFailMsg(const SwapBatch& batch) {
  return "buffer allocation failed at entry " + std::to_string(batch.failed_entry()) + " of " +
         std::to_string(batch.sizes().size()) + " (" +
         std::to_string(batch.entries_allocated()) + " pairs obtained, requested " +
         std::to_string(batch.sizes()[batch.failed_entry()]) + " bytes, status=0x" +
         StatusHex(batch.alloc_status()) + ") - no swap was exercised";
}

}  // namespace

// =============================================================================
// Test Fixture
// =============================================================================

class SwapCopyMultiEntry : public ::testing::Test {
 protected:
  void SetUp() override { ASSERT_TRUE(ctx_.HasGPUAgent()) << "No GPU agent found"; }

  // Runs the batch and verifies that every entry was exchanged. Verification
  // covers every byte of every entry, not a sample: the defects this file
  // exists to catch (a dropped final chunk, a mis-stepped entry walk, an
  // unassigned tail byte) all show up at the edges, which sampling misses.
  void RunAndVerify(SwapBatch* batch) {
    hsa_signal_t signal;
    ASSERT_EQ(HSA_STATUS_SUCCESS, hsa_signal_create(1, 0, nullptr, &signal));

    hsa_status_t status =
        SubmitMultiSwap(batch->a(), batch->b(), batch->sizes(), batch->dst_agents(),
                        ctx_.gpu_agent, 0, nullptr, signal);
    if (status != HSA_STATUS_SUCCESS) {
      hsa_signal_destroy(signal);
      FAIL() << "multi-entry swap submission failed, status=0x" << std::hex << status;
    }

    hsa_signal_value_t left = hsa_signal_wait_scacquire(
        signal, HSA_SIGNAL_CONDITION_LT, 1, kTimeoutNs, HSA_WAIT_STATE_BLOCKED);
    hsa_signal_destroy(signal);
    // A non-zero residue here is the signature of the accounting bug this
    // form is most at risk of: out_signal raised by the dispatch count but
    // decremented fewer times.
    ASSERT_LT(left, 1) << "completion signal never reached 0 (residual " << left
                       << ") - dispatch count and signal decrements disagree";

    for (size_t e = 0; e < batch->sizes().size(); ++e) {
      const size_t sz = batch->sizes()[e];
      const uint32_t ent = static_cast<uint32_t>(e);
      // After the swap, A holds B's pattern and B holds A's.
      ASSERT_EQ(sz, FirstMismatchB(batch->a()[e], ent, sz))
          << "entry " << e << " (size " << sz << "): buffer A did not receive B's contents";
      ASSERT_EQ(sz, FirstMismatchA(batch->b()[e], ent, sz))
          << "entry " << e << " (size " << sz << "): buffer B did not receive A's contents";
    }
  }

  static constexpr uint64_t kTimeoutNs = 60ull * 1000 * 1000 * 1000;

  HsaTestContext ctx_;
};

// =============================================================================
// TC_SWAP_ME_001 - several entries of different sizes in one batch
// =============================================================================
TEST_F(SwapCopyMultiEntry, TC_SWAP_ME_001_ThreeEntries_VariedSizes) {
  std::cout << "\n=== Multi-entry swap, 3 entries of differing sizes ===" << std::endl;

  // 1031 is deliberately neither a multiple of 16 nor a power of two, so the
  // shader's trailing-byte phase has to cover the last 7 bytes.
  SwapBatch batch(&ctx_, {4096, 1031, 65536});
  ASSERT_TRUE(batch.Allocate()) << AllocFailMsg(batch);

  RunAndVerify(&batch);
}

// =============================================================================
// TC_SWAP_ME_002 - sizes that are entirely or almost entirely tail bytes
// =============================================================================
TEST_F(SwapCopyMultiEntry, TC_SWAP_ME_002_UnalignedSmallEntries) {
  std::cout << "\n=== Multi-entry swap, sub-16-byte and unaligned entries ===" << std::endl;

  // 1 and 15 have no whole 16-byte chunk at all; 17 and 31 have exactly one
  // plus a tail. These are the sizes where a grid computed by rounding up to
  // a multiple of 16, instead of summing the two phases, drops bytes.
  SwapBatch batch(&ctx_, {1, 15, 16, 17, 31, 33});
  ASSERT_TRUE(batch.Allocate()) << AllocFailMsg(batch);

  RunAndVerify(&batch);
}

// =============================================================================
// TC_SWAP_ME_003 - more entries than one queue reservation can hold
// =============================================================================
TEST_F(SwapCopyMultiEntry, TC_SWAP_ME_003_ManyEntries_SpansReservations) {
  std::cout << "\n=== Multi-entry swap, 4096 entries spanning queue reservations ==="
            << std::endl;

  // Each entry under kMaxSwapBytesPerDispatch costs exactly one dispatch, so
  // 4096 entries is 4096 dispatches. The internal blit queue is smaller than
  // that on every device tested, which forces the batch to be split across
  // several reservations and wraps the kernarg ring. 4096 * 4KB * 2 is 32MB,
  // small enough to run anywhere.
  const uint32_t kNumEntries = 4096;
  SwapBatch batch(&ctx_, std::vector<size_t>(kNumEntries, 4096));
  ASSERT_TRUE(batch.Allocate()) << AllocFailMsg(batch);

  RunAndVerify(&batch);
}

// =============================================================================
// TC_SWAP_ME_004 - dependent signals gate a batch that spans reservations
// =============================================================================
TEST_F(SwapCopyMultiEntry, TC_SWAP_ME_004_DependentSignal_SpansReservations) {
  std::cout << "\n=== Multi-entry swap gated on a dependent signal ===" << std::endl;

  // The barrier packets ride along in the first reservation only. Using a
  // batch large enough to need several reservations checks that the later
  // ones do not re-emit them and, more importantly, that the dependency
  // still gates the whole batch and not just its first slice.
  const uint32_t kNumEntries = 2048;
  SwapBatch batch(&ctx_, std::vector<size_t>(kNumEntries, 4096));
  ASSERT_TRUE(batch.Allocate()) << AllocFailMsg(batch);

  hsa_signal_t dep;
  ASSERT_EQ(HSA_STATUS_SUCCESS, hsa_signal_create(1, 0, nullptr, &dep));
  hsa_signal_t done;
  ASSERT_EQ(HSA_STATUS_SUCCESS, hsa_signal_create(1, 0, nullptr, &done));

  hsa_status_t status = SubmitMultiSwap(batch.a(), batch.b(), batch.sizes(), batch.dst_agents(),
                                        ctx_.gpu_agent, 1, &dep, done);
  if (status != HSA_STATUS_SUCCESS) {
    hsa_signal_destroy(dep);
    hsa_signal_destroy(done);
    FAIL() << "gated multi-entry swap submission failed, status=0x" << std::hex << status;
  }

  // With the dependency still held, the batch must not complete. A short
  // bounded wait that times out is the expected outcome.
  hsa_signal_value_t early = hsa_signal_wait_scacquire(
      done, HSA_SIGNAL_CONDITION_LT, 1, 200ull * 1000 * 1000, HSA_WAIT_STATE_BLOCKED);
  EXPECT_GE(early, 1) << "batch completed while its dependency was still unsatisfied";

  hsa_signal_store_screlease(dep, 0);

  hsa_signal_value_t left = hsa_signal_wait_scacquire(done, HSA_SIGNAL_CONDITION_LT, 1,
                                                      kTimeoutNs, HSA_WAIT_STATE_BLOCKED);
  hsa_signal_destroy(dep);
  hsa_signal_destroy(done);
  ASSERT_LT(left, 1) << "batch did not complete after its dependency was released (residual "
                     << left << ")";

  for (size_t e = 0; e < batch.sizes().size(); ++e) {
    const size_t sz = batch.sizes()[e];
    const uint32_t ent = static_cast<uint32_t>(e);
    ASSERT_EQ(sz, FirstMismatchB(batch.a()[e], ent, sz)) << "entry " << e << ": A not swapped";
    ASSERT_EQ(sz, FirstMismatchA(batch.b()[e], ent, sz)) << "entry " << e << ": B not swapped";
  }
}

// =============================================================================
// TC_SWAP_ME_005 - a zero-length entry is rejected
// =============================================================================
TEST_F(SwapCopyMultiEntry, TC_SWAP_ME_005_ZeroLengthEntry_ShouldFail) {
  std::cout << "\n=== Multi-entry swap with a zero-length entry ===" << std::endl;

  SwapBatch batch(&ctx_, {4096, 4096, 4096});
  ASSERT_TRUE(batch.Allocate()) << AllocFailMsg(batch);

  std::vector<size_t> sizes = batch.sizes();
  sizes[1] = 0;

  hsa_signal_t signal;
  ASSERT_EQ(HSA_STATUS_SUCCESS, hsa_signal_create(1, 0, nullptr, &signal));
  hsa_status_t status = SubmitMultiSwap(batch.a(), batch.b(), sizes, batch.dst_agents(),
                                        ctx_.gpu_agent, 0, nullptr, signal);
  hsa_signal_destroy(signal);

  ASSERT_NE(HSA_STATUS_SUCCESS, status) << "a zero-length entry should be rejected";
}

// =============================================================================
// TC_SWAP_ME_006 - a null entry pointer is rejected
// =============================================================================
TEST_F(SwapCopyMultiEntry, TC_SWAP_ME_006_NullEntry_ShouldFail) {
  std::cout << "\n=== Multi-entry swap with a null entry pointer ===" << std::endl;

  SwapBatch batch(&ctx_, {4096, 4096, 4096});
  ASSERT_TRUE(batch.Allocate()) << AllocFailMsg(batch);

  // Copy the list so the batch still owns every real pointer and can free
  // them; only the copy carries the null.
  std::vector<void*> a_list = batch.a();
  a_list[2] = nullptr;

  hsa_signal_t signal;
  ASSERT_EQ(HSA_STATUS_SUCCESS, hsa_signal_create(1, 0, nullptr, &signal));
  hsa_status_t status = SubmitMultiSwap(a_list, batch.b(), batch.sizes(), batch.dst_agents(),
                                        ctx_.gpu_agent, 0, nullptr, signal);
  hsa_signal_destroy(signal);

  ASSERT_NE(HSA_STATUS_SUCCESS, status) << "a null entry pointer should be rejected";
}
