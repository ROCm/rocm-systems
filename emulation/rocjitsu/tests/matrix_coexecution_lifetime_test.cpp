// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "async_mma_test_util.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna5/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna5/opcodes.h"
#include "rocjitsu/vm/amdgpu/matrix_coexecution.h"
#include "rocjitsu/vm/amdgpu/mma_admission.h"

#include <gtest/gtest.h>

#include <cerrno>
#include <semaphore>

#if defined(__linux__)
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace {
using namespace rocjitsu;
namespace mc = amdgpu::matrix_coexecution;
using AdmissionCache = amdgpu::MmaAdmissionCache;

// Return the immutable initial fetch for a small accepted or rejected window.
AdmissionCache::Words write_window(amdgpu::GpuMemory &memory, amdgpu::InstructionCache &icache,
                                   uint64_t pc, bool dependent, uint16_t branch_immediate) {
  const auto a =
      cdna5::build_vop3p(cdna5::kVWmmaF3216x16x64Fp8Fp8Vop3p,
                         {.vdst = 64, .src0 = 256, .src1 = 288, .src2 = 320, .opsel_hi = 3});
  const auto b = cdna5::build_vop3p(cdna5::kVWmmaF3216x16x64Fp8Fp8Vop3p,
                                    {.vdst = 96,
                                     .src0 = uint16_t(dependent ? 320 : 256),
                                     .src1 = 288,
                                     .src2 = 352,
                                     .opsel_hi = 3});
  for (unsigned i = 0; i != 2; ++i) {
    memory.write32(pc + 4 * i, a[i]);
    memory.write32(pc + 8 + 4 * i, b[i]);
  }
  memory.write32(pc + 16, cdna5::build_sopp(cdna5::kSBranchSopp, {.simm16 = branch_immediate})[0]);
  icache.invalidate_all();
  AdmissionCache::Words first;
  icache.fetch(memory, pc, reinterpret_cast<uint8_t *>(first.data()));
  return first;
}

TEST(MmaAdmissionCacheTest, BoundsRetainedPlansAndRebuildsAfterEviction) {
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
  decoder->enable_pool();
  AdmissionCache cache(2, 3);
  amdgpu::GpuMemory memory("bounded_plans");
  amdgpu::InstructionCache icache;
  constexpr uint64_t begin = 0x4b0000;
  for (unsigned site = 0; site != 21; ++site) {
    const uint64_t pc = begin + 64 * site;
    const bool dependent = site % 2;
    const auto first = write_window(memory, icache, pc, dependent, 0xffff);
    const auto actual = cache.inspect(*decoder, icache, memory, pc, 0, 128, false, first);
    EXPECT_EQ(actual, dependent ? std::nullopt : std::optional(pc + 8));
    EXPECT_LE(cache.size(), 3u);
  }
  EXPECT_GT(cache.stats.evictions, 0u);
  EXPECT_EQ(cache.size(), 3u);

  // A full cache still serves a retained plan without clearing or decoding.
  const uint64_t last = begin + 64 * 20;
  const auto retained = write_window(memory, icache, last, false, 0xffff);
  const auto evictions = cache.stats.evictions;
  const auto decodes = cache.stats.decodes;
  for (unsigned i = 0; i != 100; ++i)
    EXPECT_EQ(cache.inspect(*decoder, icache, memory, last, 0, 128, false, retained), last + 8);
  EXPECT_EQ(cache.stats.evictions, evictions);
  EXPECT_EQ(cache.stats.decodes, decodes);

  const auto first = write_window(memory, icache, begin, false, 0xffff);
  const auto plans = cache.stats.plans;
  EXPECT_EQ(cache.inspect(*decoder, icache, memory, begin, 0, 128, false, first), begin + 8);
  EXPECT_EQ(cache.stats.plans, plans + 1);
  EXPECT_GT(cache.stats.evictions, evictions);
}

TEST(MmaAdmissionCacheTest, ReplacesChangedCodeWithoutGrowingTheCache) {
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
  decoder->enable_pool();
  AdmissionCache cache(2, 1);
  amdgpu::GpuMemory memory("replaced_code");
  amdgpu::InstructionCache icache;
  constexpr uint64_t pc = 0x4c0000;
  for (unsigned version = 0; version != 24; ++version) {
    // Rebuild one key as its code changes, including previously rejected plans.
    const bool dependent = version % 2;
    const auto first = write_window(memory, icache, pc, dependent, version);
    const auto actual = cache.inspect(*decoder, icache, memory, pc, 0, 128, false, first);
    EXPECT_EQ(actual, dependent ? std::nullopt : std::optional(pc + 8));
    EXPECT_EQ(cache.size(), 1u);
  }
  EXPECT_EQ(cache.stats.evictions, 0u);
  const auto first = write_window(memory, icache, pc, false, 0);
  EXPECT_EQ(cache.inspect(*decoder, icache, memory, pc, 0, 128, false, first), pc + 8);
  const auto decodes = cache.stats.decodes;
  for (unsigned i = 0; i != 100; ++i)
    EXPECT_EQ(cache.inspect(*decoder, icache, memory, pc, 0, 128, false, first), pc + 8);
  EXPECT_EQ(cache.stats.decodes, decodes);
}

TEST(MmaAdmissionCacheTest, RevalidatesAcceptedAndRejectedPlansBeyondTheInitialFetch) {
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
  decoder->enable_pool();
  AdmissionCache cache;
  amdgpu::GpuMemory memory("changed_lookahead");
  amdgpu::InstructionCache icache;
  constexpr uint64_t pc = 0x4d0000;
  const auto a =
      cdna5::build_vop3p(cdna5::kVWmmaF3216x16x64Fp8Fp8Vop3p,
                         {.vdst = 64, .src0 = 256, .src1 = 288, .src2 = 320, .opsel_hi = 3});
  for (unsigned i = 0; i != a.size(); ++i)
    memory.write32(pc + 4 * i, a[i]);
  memory.write32(pc + 8, 0xbf800000);  // s_nop
  memory.write32(pc + 12, 0xbf800000); // s_nop
  memory.write32(pc + 24, cdna5::build_sopp(cdna5::kSBranchSopp, {.simm16 = 0xffff})[0]);
  AdmissionCache::Words first;
  icache.fetch(memory, pc, reinterpret_cast<uint8_t *>(first.data()));

  for (unsigned version = 0; version != 12; ++version) {
    const bool dependent = version % 2;
    const auto b = cdna5::build_vop3p(cdna5::kVWmmaF3216x16x64Fp8Fp8Vop3p,
                                      {.vdst = 96,
                                       .src0 = uint16_t(dependent ? 320 : 256),
                                       .src1 = 288,
                                       .src2 = 352,
                                       .opsel_hi = 3});
    for (unsigned i = 0; i != b.size(); ++i)
      memory.write32(pc + 16 + 4 * i, b[i]);
    icache.invalidate_all();
    const auto expected = dependent ? std::nullopt : std::optional(pc + 16);
    EXPECT_EQ(cache.inspect(*decoder, icache, memory, pc, 0, 128, false, first), expected);
    const auto decodes = cache.stats.decodes;
    const auto plans = cache.stats.plans;
    // Unchanged bytes keep positive and negative decisions, including after
    // dispatch-style I$ invalidation. Neither path should decode again.
    for (unsigned repeat = 0; repeat != 3; ++repeat) {
      icache.invalidate_all();
      EXPECT_EQ(cache.inspect(*decoder, icache, memory, pc, 0, 128, false, first), expected);
    }
    EXPECT_EQ(cache.stats.decodes, decodes);
    EXPECT_EQ(cache.stats.plans, plans);
  }
  EXPECT_EQ(cache.stats.plans, 12u);
}

struct DecodeLifetimes {
  unsigned live = 0, total = 0;
};

class AdmissionDecodeProbe final : public DynamicInstState {
public:
  explicit AdmissionDecodeProbe(DecodeLifetimes &counts)
      : counts_(counts), owner_(std::this_thread::get_id()) {
    ++counts_.live;
    ++counts_.total;
  }
  ~AdmissionDecodeProbe() override {
    EXPECT_EQ(std::this_thread::get_id(), owner_);
    --counts_.live;
  }

private:
  DecodeLifetimes &counts_;
  std::thread::id owner_;
};

class AdmissionTrackingDecoder final : public Decoder {
public:
  explicit AdmissionTrackingDecoder(DecodeLifetimes &counts)
      : counts_(counts), decoder_(Decoder::create(ROCJITSU_CODE_ARCH_CDNA5)) {}
  DecodeResult decode(const rj_code_binary_inst_t *words, const DecodeErrorEmitter &emit) override {
    auto result = decoder_->decode(words, emit);
    if (result.succeeded()) {
      EXPECT_EQ(result.value()->data(), nullptr);
      result.value()->set_data(std::make_unique<AdmissionDecodeProbe>(counts_));
    }
    return result;
  }
  size_t max_instruction_words() const override { return decoder_->max_instruction_words(); }

private:
  DecodeLifetimes &counts_;
  std::unique_ptr<Decoder> decoder_;
};

TEST(MmaAdmissionCacheTest, ReleasesTemporaryDecodesBeforeReturningAndSurvivesThreadMigration) {
  for (bool pooled : {false, true}) {
    AdmissionCache cache;
    amdgpu::GpuMemory memory("decode_lifetimes");
    amdgpu::InstructionCache icache;
    constexpr uint64_t pc = 0x4e0000;
    DecodeLifetimes counts;
    std::thread build([&] {
      AdmissionTrackingDecoder decoder(counts);
      if (pooled)
        decoder.enable_pool();
      for (bool dependent : {false, true}) {
        const auto first = write_window(memory, icache, pc, dependent, 0xffff);
        EXPECT_EQ(cache.inspect(decoder, icache, memory, pc, 0, 128, false, first),
                  dependent ? std::nullopt : std::optional(pc + 8));
        EXPECT_EQ(counts.live, 0u);
      }
    });
    build.join();
    EXPECT_GT(counts.total, 0u);
    EXPECT_EQ(counts.live, 0u);
    const auto total = counts.total;
    // A CU can resume on a different dispatch worker after its decoder and
    // thread-local allocation hooks from the earlier scan have gone away.
    std::thread reuse([&] {
      AdmissionTrackingDecoder decoder(counts);
      if (pooled)
        decoder.enable_pool();
      const auto first = write_window(memory, icache, pc, true, 0xffff);
      EXPECT_FALSE(cache.inspect(decoder, icache, memory, pc, 0, 128, false, first));
      EXPECT_EQ(counts.total, total);
      EXPECT_EQ(counts.live, 0u);
    });
    reuse.join();
  }
}

#if defined(__linux__)
TEST(MatrixCoexecutionTest, ForkChildRejectsInheritedPoolAndDestroysItWithoutJoining) {
  unsigned calls = 0;
  Instruction increment("increment",
                        [](Instruction &, void *opaque) { ++*static_cast<unsigned *>(opaque); });
  auto pool = std::make_unique<mc::SharedPool>(1);
  auto warm = pool->submit(increment, &calls);
  ASSERT_TRUE(warm);
  ASSERT_FALSE(pool->finish(warm));
  ASSERT_EQ(calls, 1u);
  const pid_t child = fork();
  ASSERT_GE(child, 0);
  if (child == 0) {
    alarm(3);
    if (pool->available() || pool->submit(increment, &calls))
      _exit(1);
    std::array<Instruction *, 2> pair{&increment, &increment};
    mma_test::execute_independent(*pool, pair, &calls);
    if (calls != 3)
      _exit(2);
    pool.reset(); // Must not join or destroy inherited std::thread objects.
    mc::SharedPool fresh(1);
    if (fresh.available() || fresh.submit(increment, &calls))
      _exit(3);
    mma_test::execute_independent(fresh, pair, &calls);
    _exit(calls == 5 ? 0 : 4);
  }
  int status = 0;
  pid_t waited;
  do {
    waited = waitpid(child, &status, 0);
  } while (waited < 0 && errno == EINTR);
  ASSERT_EQ(waited, child);
  ASSERT_TRUE(WIFEXITED(status)) << "child signal " << WTERMSIG(status);
  EXPECT_EQ(WEXITSTATUS(status), 0);
  auto parent = pool->submit(increment, &calls);
  ASSERT_TRUE(parent);
  EXPECT_FALSE(pool->finish(parent));
  EXPECT_EQ(calls, 2u);
}

TEST(MatrixCoexecutionTest, ForkChildRejectsInheritedOutstandingTicket) {
  struct Context {
    std::binary_semaphore entered{0};
    std::binary_semaphore release{0};
  } context;
  Instruction blocked("blocked", [](Instruction &, void *opaque) {
    auto &ctx = *static_cast<Context *>(opaque);
    ctx.entered.release();
    ctx.release.acquire();
  });
  auto pool = std::make_unique<mc::SharedPool>(1);
  auto ticket = pool->submit(blocked, &context);
  ASSERT_TRUE(ticket);
  context.entered.acquire();
  const pid_t child = fork();
  if (child == 0) {
    alarm(3);
    if (!pool->ready(ticket) || !pool->finish(ticket))
      _exit(1);
    pool.reset();
    _exit(0);
  }
  context.release.release();
  EXPECT_FALSE(pool->finish(ticket));
  ASSERT_GE(child, 0);
  int status = 0;
  pid_t waited;
  do {
    waited = waitpid(child, &status, 0);
  } while (waited < 0 && errno == EINTR);
  ASSERT_EQ(waited, child);
  ASSERT_TRUE(WIFEXITED(status)) << "child signal " << WTERMSIG(status);
  EXPECT_EQ(WEXITSTATUS(status), 0);
  EXPECT_TRUE(pool->available());
}
#endif
} // namespace
