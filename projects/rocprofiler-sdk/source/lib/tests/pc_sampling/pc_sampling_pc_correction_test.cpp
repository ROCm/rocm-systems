// MIT License
//
// Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "lib/output/pc_sampling_pc_correction.hpp"

#include <gtest/gtest.h>

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

using rocprofiler::tool::pc_correction::classify;
using rocprofiler::tool::pc_correction::CodeObjectClassification;
using rocprofiler::tool::pc_correction::CorrectionResult;
using rocprofiler::tool::pc_correction::Instruction;
using rocprofiler::tool::pc_correction::InternalEntry;
using rocprofiler::tool::pc_correction::Kind;
using rocprofiler::tool::pc_correction::PCCorrectionManager;
using stochastic_record_t = rocprofiler::tool::rocprofiler_tool_pc_sampling_stochastic_record_t;

namespace
{
// Pull instructions from a fixed synthetic stream, advancing on each call. The
// builder addresses instructions by voffset; this fake ignores it and returns
// the next instruction in sequence (the builder walks strictly forward), which
// is exactly what add_symbol consumes. Each instruction is given the size the
// test specifies so byte accounting (regular_internal_total_bytes) is exercised.
struct FakeStream
{
    std::vector<std::pair<std::string, uint64_t>> insts;  // {text, size}
    size_t                                        next = 0;

    std::unique_ptr<Instruction> operator()(uint64_t /*voffset*/)
    {
        if(next >= insts.size()) return nullptr;
        auto& [text, size] = insts[next++];
        return std::make_unique<Instruction>(std::string{text}, size);
    }

    // Total byte span of the stream -- pass as the symbol size so the walk
    // consumes every instruction.
    uint64_t total_size() const
    {
        uint64_t sum = 0;
        for(const auto& [text, size] : insts)
            sum += size;
        return sum;
    }
};

// Build a single-symbol classification from a synthetic stream at symbol base 0.
CodeObjectClassification
build_single_symbol(std::vector<std::pair<std::string, uint64_t>> insts)
{
    FakeStream               stream{std::move(insts), 0};
    CodeObjectClassification c;
    c.add_symbol(0, stream.total_size(), [&](uint64_t v) { return stream(v); });
    c.sort();
    return c;
}
}  // namespace

TEST(pc_correction_classify, Classify_NopIsRegular)
{
    EXPECT_EQ(classify("s_nop 0"), Kind::REGULAR_INTERNAL);
}

TEST(pc_correction_classify, Classify_SleepIsRegular)
{
    EXPECT_EQ(classify("s_sleep 1"), Kind::REGULAR_INTERNAL);
}

TEST(pc_correction_classify, Classify_WaitAllVariants)
{
    // All s_wait* variants resolve via the "s_wait" prefix.
    static constexpr std::string_view wait_variants[] = {
        "s_waitcnt 0",
        "s_wait_loadcnt 0",
        "s_wait_storecnt 0",
        "s_wait_kmcnt 0",
        "s_wait_alu 0",
        "s_wait_idle",
    };

    for(auto inst : wait_variants)
    {
        EXPECT_EQ(classify(inst), Kind::REGULAR_INTERNAL) << "inst=" << inst;
    }
}

TEST(pc_correction_classify, Classify_BarrierWaitIsRegular)
{
    EXPECT_EQ(classify("s_barrier_wait -1"), Kind::REGULAR_INTERNAL);
}

TEST(pc_correction_classify, Classify_IcacheInvIsS_icache_inv)
{
    EXPECT_EQ(classify("s_icache_inv"), Kind::S_ICACHE_INV);
}

TEST(pc_correction_classify, Classify_VAluIsExt)
{
    EXPECT_EQ(classify("v_add_f32_e32 v0, v1, v2"), Kind::EXT);
}

TEST(pc_correction_classify, Classify_SAluIsExt)
{
    EXPECT_EQ(classify("s_mov_b32 s0, 1"), Kind::EXT);
}

TEST(pc_correction_classify, Classify_BranchIsExt)
{
    EXPECT_EQ(classify("s_branch 8"), Kind::EXT);
}

TEST(pc_correction_classify, Classify_EndPgmIsExt)
{
    EXPECT_EQ(classify("s_endpgm"), Kind::EXT);
}

TEST(pc_correction_classify, Classify_SetPrioIsExt_Today)
{
    // s_setprio is a candidate internal pending hardware confirmation; until
    // then it must classify as EXT (the safe direction).
    EXPECT_EQ(classify("s_setprio 1"), Kind::EXT);
}

// ------------------------------------------------------------------
// Window-builder tests (CodeObjectClassification::add_symbol).
// All offsets assume 4-byte instructions starting at symbol base 0 unless a
// test specifies otherwise.
// ------------------------------------------------------------------

TEST(pc_correction_build, EmptyChain_NoEntries)
{
    // Back-to-back externals: no internals -> no entries.
    auto c = build_single_symbol({{"v_add", 4}, {"v_mov", 4}});
    EXPECT_TRUE(c.entries.empty());
}

TEST(pc_correction_build, NoEXT1_LeadingInternal)
{
    // Symbol starts with internals (no preceding external) -> has_ext1 false.
    auto c = build_single_symbol({{"s_nop", 4}, {"s_nop", 4}, {"v_add", 4}});
    ASSERT_EQ(c.entries.size(), 2u);
    for(const auto& e : c.entries)
        EXPECT_FALSE(e.window->has_ext1);
}

TEST(pc_correction_build, NoEXT2_TrailingInternal)
{
    // Symbol: EXT@0 s_nop@4 s_nop@8 with no closing external. The trailing
    // internals' window opened on EXT@0 but never closes -> has_ext2 false,
    // which the cascade treats as "drop". This also covers EXT1 legitimately at
    // offset 0 (no magic-zero collision).
    auto c = build_single_symbol({{"v_add", 4}, {"s_nop", 4}, {"s_nop", 4}});
    ASSERT_EQ(c.entries.size(), 2u);
    const auto& w = *c.entries[0].window;
    EXPECT_TRUE(w.has_ext1);
    EXPECT_EQ(w.ext1_offset, 0u);  // opened by EXT@0 -- a valid offset, not a sentinel
    EXPECT_FALSE(w.has_ext2);      // never closed -> trailing internals
}

TEST(pc_correction_build, M_Only_ForwardChain)
{
    // EXT@0 s_nop@4 s_nop@8 EXT@12: regular-only chain, M=2 N=0 B=8.
    auto c = build_single_symbol({{"v_add", 4}, {"s_nop", 4}, {"s_nop", 4}, {"v_mov", 4}});
    ASSERT_EQ(c.entries.size(), 2u);
    const auto& w = *c.entries[0].window;
    EXPECT_EQ(w.M, 2u);
    EXPECT_EQ(w.N, 0u);
    EXPECT_EQ(w.regular_internal_total_bytes, 8u);
    EXPECT_TRUE(w.has_ext1);
    EXPECT_EQ(w.ext1_offset, 0u);  // EXT1 at offset 0 -- valid, not a sentinel
    EXPECT_TRUE(w.has_ext2);
    EXPECT_EQ(w.ext2_offset, 12u);
}

TEST(pc_correction_build, N_Only_BackwardChain)
{
    // EXT@0 inv@4 inv@8 EXT@12: inv-only chain, M=0 N=2 B=0.
    auto c =
        build_single_symbol({{"v_add", 4}, {"s_icache_inv", 4}, {"s_icache_inv", 4}, {"v_mov", 4}});
    ASSERT_EQ(c.entries.size(), 2u);
    const auto& w = *c.entries[0].window;
    EXPECT_EQ(w.M, 0u);
    EXPECT_EQ(w.N, 2u);
    EXPECT_EQ(w.regular_internal_total_bytes, 0u);
    EXPECT_TRUE(w.has_ext1);
    EXPECT_EQ(w.ext1_offset, 0u);
    EXPECT_TRUE(w.has_ext2);
    EXPECT_EQ(w.ext2_offset, 12u);
}

TEST(pc_correction_build, Mixed_OrderAgnostic)
{
    // Order inside the chain doesn't matter, only the counts: both layouts give
    // M=2 N=1 B=8.
    auto a = build_single_symbol(
        {{"v_add", 4}, {"s_nop", 4}, {"s_icache_inv", 4}, {"s_nop", 4}, {"v_mov", 4}});
    auto b = build_single_symbol(
        {{"v_add", 4}, {"s_icache_inv", 4}, {"s_nop", 4}, {"s_nop", 4}, {"v_mov", 4}});

    ASSERT_FALSE(a.entries.empty());
    ASSERT_FALSE(b.entries.empty());
    const auto& wa = *a.entries[0].window;
    const auto& wb = *b.entries[0].window;
    EXPECT_EQ(wa.M, 2u);
    EXPECT_EQ(wa.N, 1u);
    EXPECT_EQ(wa.regular_internal_total_bytes, 8u);
    EXPECT_EQ(wb.M, 2u);
    EXPECT_EQ(wb.N, 1u);
    EXPECT_EQ(wb.regular_internal_total_bytes, 8u);
}

TEST(pc_correction_build, VariableWidthInternal)
{
    // An 8-byte regular internal must contribute its actual size, not a fixed 4.
    auto c = build_single_symbol({{"v_add", 4}, {"s_nop", 8}, {"v_mov", 4}});
    ASSERT_EQ(c.entries.size(), 1u);
    const auto& w = *c.entries[0].window;
    EXPECT_EQ(w.M, 1u);
    EXPECT_EQ(w.regular_internal_total_bytes, 8u);
    EXPECT_EQ(w.ext2_offset, 12u);  // EXT after a 4B + 8B run
}

TEST(pc_correction_build, BackToBackEXTs)
{
    // EXT@0 EXT@4 EXT@8 s_nop@12 EXT@16: only the single internal yields an
    // entry, in the window opened by EXT@8 and closed by EXT@16.
    auto c =
        build_single_symbol({{"v_a", 4}, {"v_b", 4}, {"v_c", 4}, {"s_nop", 4}, {"v_d", 4}});
    ASSERT_EQ(c.entries.size(), 1u);
    const auto& w = *c.entries[0].window;
    EXPECT_EQ(w.M, 1u);
    EXPECT_EQ(w.ext1_offset, 8u);
    EXPECT_EQ(w.ext2_offset, 16u);
    EXPECT_EQ(c.entries[0].offset, 12u);
}

TEST(pc_correction_build, EntriesSortedAcrossSymbols)
{
    // Two symbols added in ascending base order; entries must end up globally
    // sorted by offset after sort().
    CodeObjectClassification c;

    FakeStream s2{{{"v_add", 4}, {"s_nop", 4}, {"v_mov", 4}}, 0};
    c.add_symbol(100, s2.total_size(), [&](uint64_t v) { return s2(v); });

    FakeStream s1{{{"v_add", 4}, {"s_nop", 4}, {"v_mov", 4}}, 0};
    c.add_symbol(0, s1.total_size(), [&](uint64_t v) { return s1(v); });

    c.sort();
    ASSERT_EQ(c.entries.size(), 2u);
    EXPECT_LT(c.entries[0].offset, c.entries[1].offset);
    EXPECT_EQ(c.entries[0].offset, 4u);    // internal in symbol @ base 0
    EXPECT_EQ(c.entries[1].offset, 104u);  // internal in symbol @ base 100
}

TEST(pc_correction_build, FindHitsAndMisses)
{
    // EXT@0 s_nop@4 s_nop@8 EXT@12: internals at 4 and 8.
    auto c = build_single_symbol({{"v_add", 4}, {"s_nop", 4}, {"s_nop", 4}, {"v_mov", 4}});

    EXPECT_TRUE(c.find(4).has_value());
    EXPECT_TRUE(c.find(8).has_value());
    EXPECT_FALSE(c.find(0).has_value());   // external, no entry
    EXPECT_FALSE(c.find(12).has_value());  // external, no entry
    EXPECT_FALSE(c.find(99).has_value());  // out of range
}

// ------------------------------------------------------------------
// Manager publish / lookup / erase tests. These use the publish() seam so no
// comgr-backed decoder is required; the decoder reference is a default-
// constructed (empty) translator that build() never touches here.
// ------------------------------------------------------------------

namespace
{
// A manager bound to an empty decoder. Safe because these tests drive
// publish()/lookup()/erase() directly and never call build().
struct ManagerFixture
{
    rocprofiler::tool::pc_correction::code_obj_decoder_t        decoder_value{};
    rocprofiler::common::Synchronized<
        rocprofiler::tool::pc_correction::code_obj_decoder_t, true>
                        decoder{std::move(decoder_value)};
    PCCorrectionManager mgr{decoder};

    std::shared_ptr<const CodeObjectClassification> make_classification()
    {
        auto c = std::make_shared<CodeObjectClassification>(
            build_single_symbol({{"v_add", 4}, {"s_nop", 4}, {"v_mov", 4}}));
        return c;
    }
};
}  // namespace

TEST(pc_correction_manager, PublishThenLookupHit)
{
    ManagerFixture f;
    f.mgr.publish(42, f.make_classification());

    auto hit = f.mgr.lookup(42, 4);  // s_nop @ offset 4
    ASSERT_TRUE(hit.has_value());
    EXPECT_EQ(hit->offset, 4u);
}

TEST(pc_correction_manager, LookupMissUnknownCO)
{
    ManagerFixture f;
    f.mgr.publish(42, f.make_classification());
    EXPECT_FALSE(f.mgr.lookup(7, 4).has_value());   // unknown CO id
    EXPECT_FALSE(f.mgr.lookup(42, 0).has_value());  // external offset
}

TEST(pc_correction_manager, EraseRemovesClassification)
{
    ManagerFixture f;
    f.mgr.publish(42, f.make_classification());
    ASSERT_TRUE(f.mgr.lookup(42, 4).has_value());
    f.mgr.erase(42);
    EXPECT_FALSE(f.mgr.lookup(42, 4).has_value());
}

TEST(pc_correction_manager, EraseDoesNotInvalidateInFlightReader)
{
    ManagerFixture f;
    f.mgr.publish(42, f.make_classification());

    // Simulate the copy-out: lookup returns a copy of the InternalEntry (which
    // holds a shared_ptr to the window). Erasing the CO must not invalidate it.
    auto entry = f.mgr.lookup(42, 4);
    ASSERT_TRUE(entry.has_value());
    f.mgr.erase(42);

    ASSERT_TRUE(entry->window != nullptr);
    EXPECT_EQ(entry->offset, 4u);  // still valid after erase
}

// ------------------------------------------------------------------
// Concurrency: publish/erase racing against many concurrent lookups. Run under
// TSan to catch data races; without TSan this still exercises the locking.
// ------------------------------------------------------------------

TEST(pc_correction_concurrency, ManyReadersOneWriter)
{
    ManagerFixture f;
    f.mgr.publish(1, f.make_classification());

    constexpr int          kReaders = 8;
    std::atomic<bool>      stop{false};
    std::vector<std::thread> readers;
    readers.reserve(kReaders);
    for(int i = 0; i < kReaders; ++i)
        readers.emplace_back([&] {
            while(!stop.load(std::memory_order_relaxed))
                (void) f.mgr.lookup(1, 4);
        });

    // Writer churns publish/erase on a different CO id concurrently.
    for(int i = 0; i < 1000; ++i)
    {
        f.mgr.publish(2, f.make_classification());
        f.mgr.erase(2);
    }

    stop.store(true, std::memory_order_relaxed);
    for(auto& t : readers)
        t.join();

    EXPECT_TRUE(f.mgr.lookup(1, 4).has_value());
}

// ------------------------------------------------------------------
// Gating tests (PCCorrectionManager::should_correct).
// ------------------------------------------------------------------

namespace
{
// Build a stochastic record with the gating-relevant fields set. wave_issued and
// reason_not_issued are the two HW signals should_correct inspects.
stochastic_record_t
make_record(bool wave_issued, uint32_t reason_not_issued, uint64_t co_id = 0, uint64_t offset = 0)
{
    rocprofiler_pc_sampling_record_stochastic_v0_t rec{};
    rec.wave_issued                 = wave_issued ? 1 : 0;
    rec.snapshot.reason_not_issued  = reason_not_issued;
    rec.pc.code_object_id           = co_id;
    rec.pc.code_object_offset       = offset;
    return stochastic_record_t{rec, /*inst_index*/ 0};
}

// A reason value that is consistent with a healthy internal (anything other than
// ARBITER_NOT_WIN). 0 is the first enum value and is not ARBITER_NOT_WIN.
constexpr uint32_t kReasonInternalOk = 0u;
constexpr uint32_t kReasonArbiterNotWin =
    ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_ARBITER_NOT_WIN;
}  // namespace

TEST(pc_correction_gate, DisabledReturnsFalse)
{
    ManagerFixture f;  // enabled defaults false
    auto           rec = make_record(/*wave_issued*/ true, kReasonInternalOk);
    EXPECT_FALSE(f.mgr.should_correct(rec.pc_sample_record, "s_nop"));
}

TEST(pc_correction_gate, ExtInstructionFalse)
{
    ManagerFixture f;
    f.mgr.set_enabled(true);
    auto rec = make_record(/*wave_issued*/ true, kReasonInternalOk);
    EXPECT_FALSE(f.mgr.should_correct(rec.pc_sample_record, "v_add_f32 v0, v1, v2"));
}

TEST(pc_correction_gate, HealthyInternalFalse)
{
    ManagerFixture f;
    f.mgr.set_enabled(true);
    // On an internal, but signals are consistent with a legitimate internal.
    auto rec = make_record(/*wave_issued*/ false, kReasonInternalOk);
    EXPECT_FALSE(f.mgr.should_correct(rec.pc_sample_record, "s_nop"));
}

TEST(pc_correction_gate, InternalWaveIssuedTrue)
{
    ManagerFixture f;
    f.mgr.set_enabled(true);
    // wave_issued is impossible for an internal -> leaked from adjacent external.
    auto rec = make_record(/*wave_issued*/ true, kReasonInternalOk);
    EXPECT_TRUE(f.mgr.should_correct(rec.pc_sample_record, "s_nop"));
}

TEST(pc_correction_gate, InternalArbiterNotWinTrue)
{
    ManagerFixture f;
    f.mgr.set_enabled(true);
    // reason ARBITER_NOT_WIN is impossible for a never-arbitrated internal.
    auto rec = make_record(/*wave_issued*/ false, kReasonArbiterNotWin);
    EXPECT_TRUE(f.mgr.should_correct(rec.pc_sample_record, "s_icache_inv"));
}

// ------------------------------------------------------------------
// Cascade tests (PCCorrectionManager::correct). Each publishes a classification
// for a single symbol, then corrects a record whose PC lands on a chosen
// internal offset.
// ------------------------------------------------------------------

namespace
{
// Publish `insts` as one symbol at base 0 under `co_id`, then correct a record
// whose PC lands at `offset`. Returns the result and leaves the mutated record
// in `out_rec`.
CorrectionResult
correct_at(PCCorrectionManager&                          mgr,
           uint64_t                                      co_id,
           std::vector<std::pair<std::string, uint64_t>> insts,
           uint64_t                                      offset,
           stochastic_record_t&                          out_rec)
{
    mgr.publish(co_id,
                std::make_shared<CodeObjectClassification>(build_single_symbol(std::move(insts))));
    out_rec = make_record(/*wave_issued*/ true, kReasonInternalOk, co_id, offset);
    return mgr.correct(out_rec);
}
}  // namespace

TEST(pc_correction_cascade, M_Only_ForwardToEXT2)
{
    // EXT@0 s_nop@4 s_nop@8 EXT@12; sample at offset 4 -> forward to EXT2 (12).
    ManagerFixture      f;
    stochastic_record_t rec{make_record(false, 0)};
    auto                result =
        correct_at(f.mgr, 1, {{"v_add", 4}, {"s_nop", 4}, {"s_nop", 4}, {"v_mov", 4}}, 4, rec);
    EXPECT_EQ(result, CorrectionResult::Keep);
    EXPECT_EQ(rec.pc_sample_record.pc.code_object_offset, 12u);
    EXPECT_EQ(rec.original_pc_offset, 4u);
}

TEST(pc_correction_cascade, N_Only_BackwardToEXT1)
{
    // EXT@0 inv@4 inv@8 EXT@12; sample at offset 8 -> backward to EXT1 (0).
    ManagerFixture      f;
    stochastic_record_t rec{make_record(false, 0)};
    auto                result = correct_at(
        f.mgr, 1, {{"v_add", 4}, {"s_icache_inv", 4}, {"s_icache_inv", 4}, {"v_mov", 4}}, 8, rec);
    EXPECT_EQ(result, CorrectionResult::Keep);
    EXPECT_EQ(rec.pc_sample_record.pc.code_object_offset, 0u);
    EXPECT_EQ(rec.original_pc_offset, 8u);
}

TEST(pc_correction_cascade, Mixed_BelowBoundary_BackwardToEXT1)
{
    // EXT@0 inv@4 inv@8 s_nop@12 EXT@16: B=4, boundary = 16-4 = 12. Sample at
    // offset 4 (< 12) -> backward to EXT1 (0).
    ManagerFixture      f;
    stochastic_record_t rec{make_record(false, 0)};
    auto                result = correct_at(
        f.mgr,
        1,
        {{"v_add", 4}, {"s_icache_inv", 4}, {"s_icache_inv", 4}, {"s_nop", 4}, {"v_mov", 4}},
        4,
        rec);
    EXPECT_EQ(result, CorrectionResult::Keep);
    EXPECT_EQ(rec.pc_sample_record.pc.code_object_offset, 0u);
}

TEST(pc_correction_cascade, Mixed_AtOrAboveBoundary_Drop)
{
    // EXT@0 inv@4 s_nop@8 EXT@12: B=4, boundary = 12-4 = 8. Sample at offset 8
    // (>= 8) -> ambiguous -> drop. PC must be left unchanged.
    ManagerFixture      f;
    stochastic_record_t rec{make_record(false, 0)};
    auto                result = correct_at(
        f.mgr, 1, {{"v_add", 4}, {"s_icache_inv", 4}, {"s_nop", 4}, {"v_mov", 4}}, 8, rec);
    EXPECT_EQ(result, CorrectionResult::Drop);
    EXPECT_EQ(rec.pc_sample_record.pc.code_object_offset, 8u);  // unchanged
}

TEST(pc_correction_cascade, LeadingInternalsOrphan_Drop)
{
    // Symbol begins with internals (no EXT1). Sample at offset 0 -> drop.
    ManagerFixture      f;
    stochastic_record_t rec{make_record(false, 0)};
    auto result = correct_at(f.mgr, 1, {{"s_nop", 4}, {"s_nop", 4}, {"v_add", 4}}, 0, rec);
    EXPECT_EQ(result, CorrectionResult::Drop);
}

TEST(pc_correction_cascade, TrailingInternalsOrphan_Drop)
{
    // Symbol ends with internals (no EXT2). Sample at the last internal -> drop.
    ManagerFixture      f;
    stochastic_record_t rec{make_record(false, 0)};
    auto result = correct_at(f.mgr, 1, {{"v_add", 4}, {"s_nop", 4}, {"s_nop", 4}}, 8, rec);
    EXPECT_EQ(result, CorrectionResult::Drop);
}

TEST(pc_correction_cascade, BackwardToEXT1_AtOffsetZero)
{
    // Regression for the has_ext1/has_ext2 flags (vs a zero-offset sentinel):
    // EXT1 legitimately at offset 0 must still be a valid correction target.
    // EXT@0 inv@4 EXT@8; sample at 4 -> backward to EXT1 (0), NOT a drop.
    ManagerFixture      f;
    stochastic_record_t rec{make_record(false, 0)};
    auto                result =
        correct_at(f.mgr, 1, {{"v_add", 4}, {"s_icache_inv", 4}, {"v_mov", 4}}, 4, rec);
    EXPECT_EQ(result, CorrectionResult::Keep);
    EXPECT_EQ(rec.pc_sample_record.pc.code_object_offset, 0u);
}

TEST(pc_correction_cascade, LookupMiss_PassThroughKeep)
{
    // gate said yes but no classification exists for this CO -> keep unchanged.
    ManagerFixture      f;
    stochastic_record_t rec = make_record(/*wave_issued*/ true, kReasonInternalOk, /*co*/ 99, 4);
    auto                result = f.mgr.correct(rec);
    EXPECT_EQ(result, CorrectionResult::Keep);
    EXPECT_EQ(rec.pc_sample_record.pc.code_object_offset, 4u);  // unchanged
    EXPECT_EQ(rec.original_inst_index, -1);                     // no correction stashed
}
