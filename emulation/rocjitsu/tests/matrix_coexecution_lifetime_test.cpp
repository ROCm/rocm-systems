// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

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
  icache.fetch(memory, pc, 0, reinterpret_cast<uint8_t *>(first.data()));
  return first;
}

TEST(MmaAdmissionCacheTest, BoundsRetainedPlansAndRebuildsAfterEviction) {
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
  decoder->enable_pool();
  AdmissionCache cache(3, 2, {.plans = 3, .decodes = 128});
  amdgpu::GpuMemory memory("bounded_plans");
  amdgpu::InstructionCache icache;
  constexpr uint64_t begin = 0x4b0000;
  for (unsigned site = 0; site != 21; ++site) {
    const uint64_t pc = begin + 64 * site;
    const bool dependent = site % 2;
    const auto first = write_window(memory, icache, pc, dependent, 0xffff);
    const auto actual = cache.inspect(*decoder, icache, memory, pc, 0, 128, false, first);
    EXPECT_EQ(actual, dependent ? std::nullopt : std::optional(pc + 8));
    EXPECT_LE(cache.size().plans, 3u);
    EXPECT_LE(cache.size().decodes, 128u);
  }
  EXPECT_GT(cache.stats.evictions, 0u);
  EXPECT_EQ(cache.size().plans, 3u);

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

TEST(MmaAdmissionCacheTest, BoundsUniqueDecodesAcrossCodeReplacement) {
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
  decoder->enable_pool();
  AdmissionCache cache(3, 2, {.plans = 128, .decodes = 6});
  amdgpu::GpuMemory memory("bounded_decodes");
  amdgpu::InstructionCache icache;
  constexpr uint64_t pc = 0x4c0000;
  for (unsigned version = 0; version != 24; ++version) {
    // The branch lies outside the initial fetch, exercising epoch validation
    // and rebuilding an existing key, including a previously rejected plan.
    const bool dependent = version % 2;
    const auto first = write_window(memory, icache, pc, dependent, version);
    const auto actual = cache.inspect(*decoder, icache, memory, pc, 0, 128, false, first);
    EXPECT_EQ(actual, dependent ? std::nullopt : std::optional(pc + 8));
    EXPECT_EQ(cache.size().plans, 1u);
    EXPECT_LE(cache.size().decodes, 6u);
  }
  EXPECT_GT(cache.stats.evictions, 0u);
  const auto first = write_window(memory, icache, pc, false, 0);
  EXPECT_EQ(cache.inspect(*decoder, icache, memory, pc, 0, 128, false, first), pc + 8);
  const auto decodes = cache.stats.decodes;
  for (unsigned i = 0; i != 100; ++i)
    EXPECT_EQ(cache.inspect(*decoder, icache, memory, pc, 0, 128, false, first), pc + 8);
  EXPECT_EQ(cache.stats.decodes, decodes);
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
    pool->execute(pair, &calls);
    if (calls != 3)
      _exit(2);
    pool.reset(); // Must not join or destroy inherited std::thread objects.
    mc::SharedPool fresh(1);
    if (fresh.available() || fresh.submit(increment, &calls))
      _exit(3);
    mc::execute_private_batch(pair, &calls);
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
