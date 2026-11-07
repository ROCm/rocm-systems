// MIT License
//
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

// Unit tests for the dispatch-log record stream: the
// reader side (preallocate/append/marker/publish, linking), the consumer side
// (copy_out retirement and trim), the decoder (stream_backlog::absorb),
// destruction and fork. Single-threaded tests treat every call boundary as an
// observation point and assert the ownership oracle
// (maps - unmaps == chain + spares) there; the two-thread tests use latched
// rendezvous, never sleeps.

#include "lib/rocprofiler-sdk/kfd/record_stream.hpp"
#include "lib/rocprofiler-sdk/kfd/dlog_drain.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <random>
#include <thread>
#include <vector>

#include <sys/mman.h>

namespace
{
using namespace rocprofiler::kfd;

// The counting mapping policy: counts every map/unmap (attempts and successes
// separately), can fail the k-th map, fills every new page with 0xA5 so reads
// past `published` show, and can record mapping addresses for the fork test.
struct test_mapping
{
    static inline std::atomic<size_t> map_calls{0};        // attempts
    static inline std::atomic<size_t> map_successes{0};    // real mmap succeeded
    static inline std::atomic<size_t> unmap_calls{0};      // attempts
    static inline std::atomic<size_t> unmap_successes{0};  // munmap returned 0
    static inline std::atomic<size_t> fail_map_nth{0};     // fail the k-th map; 0 = never
    static inline std::vector<void*>  mapped_addrs;
    static inline bool                map_recording = false;

    static void reset(size_t fail_nth = 0)
    {
        map_calls.store(0, std::memory_order_relaxed);
        map_successes.store(0, std::memory_order_relaxed);
        unmap_calls.store(0, std::memory_order_relaxed);
        unmap_successes.store(0, std::memory_order_relaxed);
        fail_map_nth.store(fail_nth, std::memory_order_relaxed);
        mapped_addrs.clear();
    }

    // Set the next failing map without disturbing the counters.
    static void set_fail_nth(size_t nth) { fail_map_nth.store(nth, std::memory_order_relaxed); }

    static void record_maps(bool on) { map_recording = on; }

    static void* map(size_t bytes) noexcept
    {
        const size_t nth = map_calls.fetch_add(1, std::memory_order_relaxed) + 1;
        if(fail_map_nth.load(std::memory_order_relaxed) == nth) return nullptr;
        void* const p =
            ::mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if(p == MAP_FAILED) return nullptr;
        std::memset(p, 0xA5, bytes);  // poison: any byte the pool did not write shows
        if(map_recording) mapped_addrs.push_back(p);
        map_successes.fetch_add(1, std::memory_order_relaxed);
        return p;
    }

    static void unmap(void* p, size_t bytes) noexcept
    {
        unmap_calls.fetch_add(1, std::memory_order_relaxed);
        if(::munmap(p, bytes) == 0) unmap_successes.fetch_add(1, std::memory_order_relaxed);
    }
};

using stream_t      = basic_record_stream<test_mapping>;
constexpr size_t kK = stream_t::kK;

// Deterministic 2-party step latch (same shape as the P1 tests): both parties
// call arrive_and_wait(step) with the same 0-based step; a step completes only
// when both have arrived, so a thread can never run ahead into a later phase.
class step_latch
{
public:
    void arrive_and_wait(int step)
    {
        m_done.fetch_add(1, std::memory_order_acq_rel);
        while(m_done.load(std::memory_order_acquire) < static_cast<size_t>(step + 1) * kParties)
            std::this_thread::yield();
    }

private:
    static constexpr size_t kParties = 2;
    std::atomic<size_t>     m_done{0};
};

fw_record
make_record(uint64_t idx)
{
    return fw_record{static_cast<uint32_t>(idx),
                     static_cast<uint32_t>(idx >> 32),
                     kRecStart,
                     0xABCDu,
                     static_cast<uint32_t>(idx & 0xFFFFu)};
}

reader_marker
make_marker(uint64_t epoch = 7, uint64_t seen = 1000)
{
    reader_marker m;
    m.epoch      = epoch;
    m.seen_at_ns = seen;
    return m;
}

uint64_t
slot_ts(const fw_record& r)
{
    return static_cast<uint64_t>(r.ts_lo) | (static_cast<uint64_t>(r.ts_hi) << 32);
}

// Ownership oracle (section 13): every successful map is held by the chain or
// the spare pool. Quiescent accessors only.
void
expect_ownership(const stream_t& st)
{
    const size_t chain  = st.chain_length_quiescent();
    const size_t spares = st.spares_quiescent();
    EXPECT_EQ(test_mapping::map_successes.load(),
              test_mapping::unmap_successes.load() + chain + spares)
        << "chain=" << chain << " spares=" << spares;
}

// copy_out sink that keeps every run and its position for exact checks.
struct copy_sink
{
    std::vector<fw_record> all;
    std::vector<uint64_t>  run_pos;
    std::vector<size_t>    run_len;

    void operator()(const fw_record* p, size_t n, uint64_t pos)
    {
        run_pos.push_back(pos);
        run_len.push_back(n);
        all.insert(all.end(), p, p + n);
    }
};

void
expect_records(const std::vector<fw_record>& got, uint64_t first, size_t count)
{
    ASSERT_EQ(got.size(), count);
    for(size_t i = 0; i < count; ++i)
        ASSERT_EQ(slot_ts(got[i]), first + i) << "record " << i << " of " << count;
}

void
expect_contiguous_runs(const copy_sink& s, uint64_t first)
{
    uint64_t expect = first;
    for(size_t i = 0; i < s.run_pos.size(); ++i)
    {
        ASSERT_EQ(s.run_pos[i], expect) << "run " << i;
        expect += s.run_len[i];
    }
}

// Element-wise record equality: fw_record is a plain ABI aggregate with no
// operator==.
bool
records_equal(const std::vector<fw_record>& a, const std::vector<fw_record>& b)
{
    if(a.size() != b.size()) return false;
    for(size_t i = 0; i < a.size(); ++i)
        if(a[i].ts_lo != b[i].ts_lo || a[i].ts_hi != b[i].ts_hi ||
           a[i].record_type != b[i].record_type || a[i].dispatch_id != b[i].dispatch_id ||
           a[i].doorbell_off != b[i].doorbell_off)
            return false;
    return true;
}

// on_run that feeds stream_backlog::absorb, the production decoder path.
struct backlog_sink
{
    stream_backlog* bl;
    void operator()(const fw_record* p, size_t n, uint64_t pos) const { bl->absorb(p, n, pos); }
};

// --- partial-preallocate rollback ----------------------------------------------

TEST(record_stream, preallocate_failure_rolls_back_everything)
{
    // The 3rd chunk's map fails: the current chunk plus one spare must be freed.
    test_mapping::reset(/*fail_nth=*/3);
    {
        stream_t st;
        EXPECT_FALSE(st.preallocate(4));
        EXPECT_FALSE(st.armed());
        EXPECT_EQ(test_mapping::map_calls.load(), 3u);
        EXPECT_EQ(test_mapping::map_successes.load(), 2u);
        EXPECT_EQ(test_mapping::unmap_calls.load(), 2u);
        EXPECT_EQ(test_mapping::unmap_successes.load(), 2u);
        EXPECT_EQ(st.chain_length_quiescent(), 0u);
        EXPECT_EQ(st.spares_quiescent(), 0u);
        EXPECT_EQ(st.write_pos_quiescent(), 0u);
    }
    EXPECT_EQ(test_mapping::unmap_calls.load(), 2u) << "the destructor must not double free";

    // The first map failing leaves nothing held at all.
    test_mapping::reset(/*fail_nth=*/1);
    {
        stream_t st;
        EXPECT_FALSE(st.preallocate(4));
        EXPECT_EQ(test_mapping::map_successes.load(), 0u);
        EXPECT_EQ(test_mapping::unmap_calls.load(), 0u);
        EXPECT_EQ(st.chain_length_quiescent(), 0u);
        EXPECT_EQ(st.spares_quiescent(), 0u);
    }
    EXPECT_EQ(test_mapping::unmap_calls.load(), 0u);
}

// Fork-child property (the handler itself is not reachable from unit tests):
// abandon_in_child() must cover a stream the setup path left unpublished and
// mid-rollback (preallocate failed after mapping chunks), so the destructor --
// and the child's later teardown -- touch nothing.
TEST(record_stream, abandon_after_failed_preallocate_destructor_touches_nothing)
{
    test_mapping::reset(/*fail_nth=*/4);  // rollback frees 3 mapped chunks
    {
        stream_t st;
        ASSERT_FALSE(st.preallocate(4));
        ASSERT_FALSE(st.armed());  // the unpublished-slot state
        st.abandon_in_child();
        EXPECT_EQ(st.chain_length_quiescent(), 0u);
        EXPECT_EQ(st.spares_quiescent(), 0u);
        const size_t unmaps_at_abandon = test_mapping::unmap_calls.load();
        st.abandon_in_child();  // idempotent, as the child handler may repeat it
        EXPECT_EQ(test_mapping::unmap_calls.load(), unmaps_at_abandon)
            << "abandon_in_child must not unmap";
    }
    EXPECT_EQ(test_mapping::unmap_calls.load(), 3u)
        << "the rollback freed the 3 chunks; "
           "the abandons and the destructor freed nothing more";
    EXPECT_EQ(test_mapping::map_successes.load(), test_mapping::unmap_successes.load());
}

// --- append boundaries; the link happens at write position K -------------------

TEST(record_stream, append_boundaries_match_input_exactly)
{
    const size_t sizes[] = {0, 1, kK - 1, kK, kK + 1, 2 * kK + 7};
    for(size_t n : sizes)
    {
        SCOPED_TRACE(::testing::Message() << "n=" << n);
        std::vector<fw_record> src(n);
        for(size_t i = 0; i < n; ++i)
            src[i] = make_record(i);
        for(int split = 0; split < 2; ++split)
        {
            test_mapping::reset();
            stream_t st;
            ASSERT_TRUE(st.preallocate(8));
            reader_alloc_log log;
            if(split == 0)
            {
                st.append(src.data(), n, log);
            }
            else
            {
                size_t off = 0;
                while(off < n)
                {
                    const size_t c = std::min<size_t>(7, n - off);
                    st.append(src.data() + off, c, log);
                    off += c;
                }
            }
            EXPECT_EQ(st.write_pos_quiescent(), n);
            st.publish();
            // The chain links exactly once per full chunk, at write position K.
            const size_t expect_chain = n == 0 ? 1 : (n - 1) / kK + 1;
            EXPECT_EQ(st.chain_length_quiescent(), expect_chain);
            expect_ownership(st);
            copy_sink sink;
            st.copy_out(sink, 8);
            EXPECT_EQ(st.taken(), n);
            EXPECT_EQ(sink.all.size(), n);
            expect_records(sink.all, 0, n);
            expect_contiguous_runs(sink, 0);
        }
    }
}

// --- a full unlinked tail is never retired --------------------------------------

TEST(record_stream, full_unlinked_tail_is_not_retired)
{
    test_mapping::reset();
    stream_t st;
    ASSERT_TRUE(st.preallocate(1));
    std::vector<fw_record> src(kK);
    for(size_t i = 0; i < src.size(); ++i)
        src[i] = make_record(i);
    reader_alloc_log log;
    st.append(src.data(), src.size(), log);  // fills the chunk exactly; no link yet
    st.publish();
    copy_sink sink;
    st.copy_out(sink, 1);
    EXPECT_EQ(st.taken(), kK);
    EXPECT_EQ(test_mapping::unmap_calls.load(), 0u)
        << "an unlinked chunk is never retired, even when full";
    EXPECT_EQ(st.chain_length_quiescent(), 1u) << "the chunk stays head";
    EXPECT_EQ(st.spares_quiescent(), 1u) << "spares are unchanged";
    // The next append links a successor; a second copy_out retires the chunk.
    std::vector<fw_record> more = {make_record(kK), make_record(kK + 1)};
    st.append(more.data(), more.size(), log);
    st.publish();
    st.copy_out(sink, 1);
    EXPECT_EQ(st.taken(), kK + 2);
    EXPECT_EQ(test_mapping::unmap_calls.load(), 0u)
        << "now that the chunk is linked, it is retired";
    EXPECT_EQ(st.chain_length_quiescent(), 1u);
    EXPECT_EQ(st.spares_quiescent(), 1u);
    expect_ownership(st);
}

// --- coalesced publishes still give one segment per marker --------------------

TEST(record_stream, coalesced_publishes_give_one_segment_per_marker)
{
    test_mapping::reset();
    std::vector<fw_record> src(5);
    for(size_t i = 0; i < src.size(); ++i)
        src[i] = make_record(i);
    const reader_marker m_data = make_marker(7, 111);
    const reader_marker m_idle = make_marker(7, 222);
    reader_alloc_log    log;

    stream_t       a;
    stream_backlog bl_a;
    ASSERT_TRUE(a.preallocate(4));
    a.append(src.data(), src.size(), log);
    a.append_marker(m_data, log);
    a.append_marker(m_idle, log);  // consecutive markers close an empty segment
    a.publish();
    a.copy_out(backlog_sink{&bl_a}, 4);

    stream_t       b;
    stream_backlog bl_b;
    ASSERT_TRUE(b.preallocate(4));
    b.append(src.data(), src.size(), log);
    b.publish();
    b.copy_out(backlog_sink{&bl_b}, 4);
    b.append_marker(m_data, log);  // a marker as the first slot of a run
    b.publish();
    b.copy_out(backlog_sink{&bl_b}, 4);
    b.append_marker(m_idle, log);
    b.publish();
    b.copy_out(backlog_sink{&bl_b}, 4);
    // The mapping counters are global to the TU, so the oracle spans both
    // streams: preallocate is 5 maps each, nothing is freed, nothing leaked.
    EXPECT_EQ(test_mapping::map_successes.load(),
              test_mapping::unmap_successes.load() + a.chain_length_quiescent() +
                  a.spares_quiescent() + b.chain_length_quiescent() + b.spares_quiescent());

    ASSERT_EQ(bl_a.closed.size(), 2u) << "two markers must not merge into one segment";
    ASSERT_EQ(bl_b.closed.size(), 2u);
    for(size_t i = 0; i < 2; ++i)
    {
        const stream_segment& sa = bl_a.closed[i];
        const stream_segment& sb = bl_b.closed[i];
        EXPECT_TRUE(records_equal(sa.records, sb.records));
        EXPECT_EQ(sa.end_pos, sb.end_pos);
        EXPECT_EQ(sa.epoch, sb.epoch);
        EXPECT_EQ(sa.seen_at_ns, sb.seen_at_ns);
    }
    EXPECT_EQ(bl_a.closed[0].end_pos, 6u);
    EXPECT_EQ(bl_a.closed[0].seen_at_ns, 111u);
    EXPECT_EQ(bl_a.closed[1].records.size(), 0u);
    EXPECT_EQ(bl_a.closed[1].end_pos, 7u);
    EXPECT_EQ(bl_a.closed[1].seen_at_ns, 222u);
}

// --- copy_out retires a fully taken chunk and leaves the empty
// successor as head -----------------------------------------------------------

TEST(record_stream, copy_out_retires_and_leaves_empty_successor_as_head)
{
    test_mapping::reset();
    stream_t st;
    ASSERT_TRUE(st.preallocate(1));
    std::vector<fw_record> src(kK);
    for(size_t i = 0; i < src.size(); ++i)
        src[i] = make_record(i);
    reader_alloc_log log;
    st.append(src.data(), src.size(), log);  // fills X exactly
    st.publish();
    // An unpublished append links Y; nothing beyond K is published yet.
    std::vector<fw_record> more = {make_record(kK), make_record(kK + 1)};
    st.append(more.data(), more.size(), log);
    EXPECT_EQ(st.chain_length_quiescent(), 2u);
    EXPECT_EQ(st.published(), kK);
    copy_sink sink;
    st.copy_out(sink, 1);
    EXPECT_EQ(st.taken(), kK) << "copy_out reads to published, never past it";
    EXPECT_EQ(sink.all.size(), kK);
    for(size_t i = 0; i < sink.all.size(); ++i)
        ASSERT_EQ(slot_ts(sink.all[i]), i) << "the 0xA5 poison must never appear";
    EXPECT_EQ(st.chain_length_quiescent(), 1u) << "m_head must have advanced to the empty Y";
    EXPECT_EQ(st.spares_quiescent(), 1u);
    expect_ownership(st);
}

// --- retirement beyond the target frees, ownership stays balanced --------------

TEST(record_stream, retire_beyond_target_frees_and_ownership_stays_balanced)
{
    test_mapping::reset();
    stream_t st;
    ASSERT_TRUE(st.preallocate(2));
    std::vector<fw_record> src(2 * kK + 8);
    for(size_t i = 0; i < src.size(); ++i)
        src[i] = make_record(i);
    reader_alloc_log log;
    // Two links, both served from the spare pool: chain 3, spares 0.
    st.append(src.data(), src.size(), log);
    st.publish();
    copy_sink sink;
    st.copy_out(sink, 1);
    // One retire was pushed, the next retire freed (trim beyond the target).
    EXPECT_EQ(test_mapping::unmap_calls.load(), 1u);
    EXPECT_EQ(st.spares_quiescent(), 1u);
    EXPECT_EQ(st.taken(), 2 * kK + 8);
    expect_ownership(st);

    // Target 0 in copy_out frees every retirable chunk. The appends above
    // drained the pool (the link pops the spare), so nothing is pushed and the
    // unlinked tail stays head.
    std::vector<fw_record> more(kK - 10);
    for(size_t i = 0; i < more.size(); ++i)
        more[i] = make_record(2 * kK + 8 + i);
    st.append(more.data(), more.size(), log);
    std::vector<fw_record> last = {
        make_record(3 * kK - 2), make_record(3 * kK - 1), make_record(3 * kK)};
    st.append(last.data(), last.size(), log);
    st.publish();
    const size_t unmaps = test_mapping::unmap_calls.load();
    st.copy_out(sink, 0);
    EXPECT_EQ(test_mapping::unmap_calls.load(), unmaps + 1);
    EXPECT_EQ(st.spares_quiescent(), 0u);
    expect_ownership(st);
}

// --- a link failure throws std::bad_alloc (failure terminates, as base's did) --

TEST(record_stream, link_failure_throws_bad_alloc)
{
    test_mapping::reset();
    stream_t st;
    ASSERT_TRUE(st.preallocate(1));  // one spare: consumed filling the second chunk
    const size_t           total = 2 * kK + 50;
    std::vector<fw_record> src(total);
    for(size_t i = 0; i < total; ++i)
        src[i] = make_record(i);
    reader_alloc_log log;
    st.append(src.data(), 2 * kK, log);  // fills X, pops the spare, fills Y
    st.publish();
    const uint64_t published_prefix = st.published();
    // Pool empty: the next link must allocate, and the injected map failure
    // surfaces as std::bad_alloc, which terminates the reader as base does.
    test_mapping::set_fail_nth(test_mapping::map_calls.load() + 1);
    EXPECT_THROW(st.append(src.data() + 2 * kK, total - 2 * kK, log), std::bad_alloc);
    test_mapping::set_fail_nth(0);
    EXPECT_EQ(st.write_pos_quiescent(), published_prefix) << "a failed append wrote nothing";
    expect_ownership(st);
    copy_sink sink;
    st.copy_out(sink, 2);
    EXPECT_EQ(st.taken(), published_prefix);
    expect_records(sink.all, 0, published_prefix);
    expect_contiguous_runs(sink, 0);
    st.reset();
    EXPECT_EQ(test_mapping::map_successes.load(), test_mapping::unmap_successes.load())
        << "reset() must free the chain and the spares";

    // Variant (a): the append starts with room left and fails at a later link:
    // the copied portion stays written and unpublished, the throw surfaces.
    {
        test_mapping::reset();
        stream_t st_a;
        ASSERT_TRUE(st_a.preallocate(2));
        std::vector<fw_record> src_a(3 * kK);
        for(size_t i = 0; i < src_a.size(); ++i)
            src_a[i] = make_record(i);
        reader_alloc_log log_a;
        st_a.append(src_a.data(), 2 * kK - 40, log_a);  // fills X, writes K-40 into Y
        st_a.publish();
        const uint64_t published_before = st_a.published();
        const uint64_t written_before   = st_a.write_pos_quiescent();
        ASSERT_EQ(written_before, 2 * kK - 40);
        test_mapping::set_fail_nth(test_mapping::map_calls.load() + 1);
        EXPECT_THROW(st_a.append(src_a.data() + written_before, 2 * kK, log_a), std::bad_alloc);
        test_mapping::set_fail_nth(0);
        // 40 slots filled Y, the last spare filled Z, then the link threw.
        EXPECT_EQ(st_a.write_pos_quiescent(), written_before + kK + 40)
            << "write_pos must advance by exactly the copied portion";
        EXPECT_EQ(st_a.published(), published_before) << "publish() is the caller's job";
        copy_sink sink_a;
        st_a.copy_out(sink_a, 2);
        EXPECT_EQ(st_a.taken(), published_before);
        expect_records(sink_a.all, 0, published_before);
        st_a.reset();
        EXPECT_EQ(test_mapping::map_successes.load(), test_mapping::unmap_successes.load());
    }

    // Variant (b): a marker at a full tail throws with its link and writes
    // nothing: positions, published data and the tail's last record stay put.
    {
        test_mapping::reset();
        stream_t st_b;
        ASSERT_TRUE(st_b.preallocate(0));  // no spares: the marker's link must allocate
        std::vector<fw_record> src_b(kK);
        for(size_t i = 0; i < src_b.size(); ++i)
            src_b[i] = make_record(i);
        reader_alloc_log log_b;
        st_b.append(src_b.data(), src_b.size(), log_b);  // fills X exactly
        st_b.publish();
        const uint64_t published_before = st_b.published();
        test_mapping::set_fail_nth(test_mapping::map_calls.load() + 1);
        EXPECT_THROW(st_b.append_marker(make_marker(9, 4'000), log_b), std::bad_alloc);
        test_mapping::set_fail_nth(0);
        EXPECT_EQ(st_b.write_pos_quiescent(), kK) << "a failed marker wrote nothing";
        EXPECT_EQ(st_b.published(), published_before);
        copy_sink sink_b;
        st_b.copy_out(sink_b, 1);
        EXPECT_EQ(st_b.taken(), kK);
        // expect_records asserts the size and every slot's ts, so the tail's
        // last record must still be intact.
        expect_records(sink_b.all, 0, kK);
        st_b.reset();
        EXPECT_EQ(test_mapping::map_successes.load(), test_mapping::unmap_successes.load());
    }
}

// --- two-thread stress, latched then free-running ------------------------------

TEST(record_stream, two_thread_latched_then_free_running_stress)
{
    test_mapping::reset();
    stream_t st;
    ASSERT_TRUE(st.preallocate(4));

    constexpr size_t   kLatchedRounds = 10;
    constexpr size_t   kRoundRecords  = kK;  // rounds 1-9 are the linking rounds
    constexpr uint64_t kStressRecords = 150'000'000;
    constexpr uint64_t kLatchedTotal  = kLatchedRounds * kRoundRecords;

    std::vector<fw_record> src(2048);
    std::vector<fw_record> round_buf(kRoundRecords);

    reader_alloc_log      log;
    std::atomic<uint64_t> produced{0};
    std::atomic<bool>     producer_done{false};
    size_t                total_markers = 0;
    step_latch            latch;

    auto producer = std::thread{[&]() {
        uint64_t     idx         = 0;
        size_t       markers     = 0;
        uint64_t     stress_left = kStressRecords;
        std::mt19937 rng(987654321u);
        for(size_t round = 0; round < kLatchedRounds; ++round)
        {
            for(size_t i = 0; i < kRoundRecords; ++i)
                round_buf[i] = make_record(round * kRoundRecords + i);
            // Script: rounds 1-4 pop the four spares; rounds 5-9 hit an empty
            // pool and allocate exactly one chunk each.
            for(size_t off = 0; off < kRoundRecords; off += 512)
                st.append(round_buf.data() + off, std::min<size_t>(512, kRoundRecords - off), log);
            st.publish();
            latch.arrive_and_wait(static_cast<int>(2 * round));      // A_k
            latch.arrive_and_wait(static_cast<int>(2 * round + 1));  // B_k
        }
        // Free-running phase (the TSan target): random runs, a marker every few
        // runs, publish per batch. The producer never blocks.
        idx = kLatchedTotal;
        while(stress_left > 0)
        {
            const size_t n =
                static_cast<size_t>(std::min<uint64_t>(1 + rng() % src.size(), stress_left));
            for(size_t i = 0; i < n; ++i)
                src[i] = make_record(idx + i);
            st.append(src.data(), n, log);
            idx += n;
            produced.store(idx, std::memory_order_release);
            stress_left -= n;
            if(rng() % 4 == 0)
            {
                const reader_marker m = make_marker(1, 1000 + markers);
                st.append_marker(m, log);
                ++markers;
            }
            st.publish();
            EXPECT_LE(st.spares_available(), stream_t::pool_t::kCapacity);
            if(st.spares_available() == 0) std::this_thread::yield();
        }
        st.publish();
        total_markers = markers;
        producer_done.store(true, std::memory_order_release);
    }};
    auto consumer = std::thread{[&]() {
        stream_backlog bl;
        uint64_t       verified  = 0;
        size_t         segs_seen = 0;  // test-side count of closed segments (one per marker)
        auto           verify    = [&](const std::vector<fw_record>& recs) {
            for(const fw_record& r : recs)
            {
                EXPECT_EQ(slot_ts(r), verified) << "gap-free data broken at " << verified;
                ++verified;
            }
        };
        auto drain = [&]() {
            segs_seen += bl.closed.size();
            for(auto& seg : bl.closed)
            {
                verify(seg.records);
                bl.recycle_vector(std::move(seg.records));
            }
            bl.closed.clear();
        };
        // Exact spare level after each round's copy_out. Rounds 0-8 copy out
        // with target 0 (every retire freed, successors not yet linked); round
        // 9's copy_out finally retires chunk 8 with target 4, pushing one.
        const size_t expect_spares[kLatchedRounds] = {4, 3, 2, 1, 0, 0, 0, 0, 0, 1};
        for(size_t round = 0; round < kLatchedRounds; ++round)
        {
            latch.arrive_and_wait(static_cast<int>(2 * round));  // A_k
            st.copy_out(backlog_sink{&bl}, round == 9 ? 4 : 0);
            EXPECT_EQ(st.taken(), st.published());
            drain();
            EXPECT_EQ(st.spares_quiescent(), expect_spares[round]);
            expect_ownership(st);
            if(round == kLatchedRounds - 1)
            {
                // Scripted reader allocations: rounds 5-9 allocate exactly once
                // each. Asserted here while the producer is still parked at
                // B_9: it may not touch the non-atomic counters until the
                // rendezvous below releases it into the free-running phase.
                EXPECT_EQ(log.reader_allocs, 5u);
            }
            latch.arrive_and_wait(static_cast<int>(2 * round + 1));  // B_k
        }
        // Free-running: copy out and drain until the producer is done and the
        // backlog is empty.
        while(!producer_done.load(std::memory_order_acquire) || st.taken() < st.published() ||
              !bl.closed.empty())
        {
            st.copy_out(backlog_sink{&bl}, 4);
            drain();
            EXPECT_LE(st.taken(), st.published());
            if(!producer_done.load(std::memory_order_acquire) && st.taken() == st.published())
                std::this_thread::yield();
        }
        verify(bl.open.records);
        bl.open.records.clear();
        EXPECT_EQ(bl.closed.size(), 0u);
        EXPECT_EQ(segs_seen, total_markers) << "every marker must close exactly one segment";
        EXPECT_EQ(verified, kLatchedTotal + kStressRecords);
    }};

    producer.join();
    consumer.join();

    EXPECT_EQ(produced.load(), kLatchedTotal + kStressRecords);
    EXPECT_EQ(st.taken(), st.published());
    EXPECT_EQ(st.published(), kLatchedTotal + kStressRecords + total_markers);
    // The free-running phase adds allocations on top of the scripted five.
    EXPECT_GE(log.reader_allocs, 5u);
    expect_ownership(st);
    test_mapping::reset(0);
}

// --- retire-before-consume rendezvous -------------------------------------------

TEST(record_stream, copy_out_never_retires_before_on_run_returns)
{
    test_mapping::reset();
    stream_t st;
    ASSERT_TRUE(st.preallocate(1));  // one spare: consumed linking Y, so the pool is empty
    constexpr size_t       kIntoY = 8;
    std::vector<fw_record> x_src(kK);
    for(size_t i = 0; i < kK; ++i)
        x_src[i] = make_record(i);
    std::vector<fw_record> y_src(kIntoY);
    for(size_t i = 0; i < kIntoY; ++i)
        y_src[i] = make_record(kK + i);

    step_latch latch;  // 0: setup done; 1: C (consumer parked in on_run); 2: D

    auto producer = std::thread{[&]() {
        reader_alloc_log log;
        st.append(x_src.data(), kK, log);  // fills X; the next append links Y
        st.append(y_src.data(), kIntoY, log);
        st.publish();
        latch.arrive_and_wait(0);
        latch.arrive_and_wait(1);
        // The consumer is parked inside on_run for X's first run and X was NOT
        // pushed, so this link must allocate a fresh successor (+1 map). The
        // three records fill Y's two remaining slots into the successor's slot
        // 0; the third carries the 0x5A pattern, so a mutant that recycled X
        // for the successor overwrites X's record 0.
        std::vector<fw_record> fill(kK - 2 - kIntoY);
        for(size_t i = 0; i < fill.size(); ++i)
            fill[i] = make_record(kK + kIntoY + i);
        st.append(fill.data(), fill.size(), log);
        std::vector<fw_record> last3 = {
            make_record(2 * kK - 2), make_record(2 * kK - 1), make_record(2 * kK)};
        last3[2].dispatch_id = 0x5A5A5A5Au;
        st.append(last3.data(), last3.size(), log);
        st.publish();
        latch.arrive_and_wait(2);
    }};

    auto consumer = std::thread{[&]() {
        latch.arrive_and_wait(0);
        bool saw_first = false;
        st.copy_out(
            [&](const fw_record* p, size_t n, uint64_t pos) {
                if(pos == 0 && !saw_first)
                {
                    saw_first = true;
                    latch.arrive_and_wait(1);  // C: parked holding X's pointer
                    latch.arrive_and_wait(2);  // wait for D
                }
                // Copy and check exactly AFTER the rendezvous: correct code has
                // pushed nothing, so X still holds its original records.
                for(size_t i = 0; i < n; ++i)
                {
                    ASSERT_EQ(slot_ts(p[i]), pos + i)
                        << "X's data was overwritten before on_run returned";
                    ASSERT_EQ(p[i].dispatch_id, 0xABCDu);
                }
            },
            1);
        EXPECT_TRUE(saw_first);
        if(!saw_first)
        {
            // A mutant that skips the first callback would strand the producer
            // at the latch; release it before returning (no join timeout).
            latch.arrive_and_wait(1);
            latch.arrive_and_wait(2);
            return;
        }
        EXPECT_EQ(st.taken(), kK + kIntoY);
        // Drain the rest and check the pattern record and the oracle.
        copy_sink sink;
        st.copy_out(sink, 1);
        EXPECT_EQ(st.taken(), 2 * kK + 1);
        ASSERT_EQ(sink.all.size(), (kK - 2 - kIntoY) + 3u);
        for(size_t i = 0; i + 1 < sink.all.size(); ++i)
            EXPECT_EQ(slot_ts(sink.all[i]), kK + kIntoY + i);
        EXPECT_EQ(slot_ts(sink.all.back()), 2 * kK);
        EXPECT_EQ(sink.all.back().dispatch_id, 0x5A5A5A5Au)
            << "the pattern is the successor's slot 0";
        EXPECT_EQ(test_mapping::map_successes.load(), 3u)
            << "preallocate (2 maps) + the producer's link";
        // Y retires fully taken with the spare slot taken by X, so it is freed.
        EXPECT_EQ(test_mapping::unmap_calls.load(), 1u);
        expect_ownership(st);
    }};

    producer.join();
    consumer.join();
    test_mapping::reset(0);
}

// --- destruction and fork child -------------------------------------------------

TEST(record_stream, destructor_releases_chain_and_abandon_touches_nothing)
{
    // (a) Scope exit without reset(): maps == unmaps.
    test_mapping::reset();
    {
        stream_t st;
        ASSERT_TRUE(st.preallocate(4));
        std::vector<fw_record> src(2 * kK + 5);
        for(size_t i = 0; i < src.size(); ++i)
            src[i] = make_record(i);
        reader_alloc_log log;
        st.append(src.data(), src.size(), log);
        st.publish();  // published but untaken records stay in the chain
        EXPECT_EQ(st.chain_length_quiescent(), 3u);
    }
    EXPECT_EQ(test_mapping::map_successes.load(), test_mapping::unmap_successes.load());

    // (b) abandon_in_child() then scope exit: no munmap at all; the test frees
    // the recorded mappings itself, like the fork child's leak.
    test_mapping::reset();
    test_mapping::record_maps(true);
    {
        stream_t st;
        ASSERT_TRUE(st.preallocate(4));
        std::vector<fw_record> src(2 * kK + 5);
        for(size_t i = 0; i < src.size(); ++i)
            src[i] = make_record(i);
        reader_alloc_log log;
        st.append(src.data(), src.size(), log);
        st.publish();
        st.abandon_in_child();
        EXPECT_EQ(test_mapping::unmap_calls.load(), 0u);
        EXPECT_EQ(st.chain_length_quiescent(), 0u);
        EXPECT_EQ(st.spares_quiescent(), 0u);
    }
    EXPECT_EQ(test_mapping::unmap_calls.load(), 0u) << "abandon_in_child must not munmap";
    // The child owns these mappings now; free them the way the child would.
    const size_t bytes = stream_t::pool_t::map_bytes();
    for(void* p : test_mapping::mapped_addrs)
        ::munmap(p, bytes);
    test_mapping::record_maps(false);
    test_mapping::reset(0);
}

// --- Marker layout: explicit encoded slot and boundary values -----------------

TEST(record_stream, marker_slot_layout_and_boundaries)
{
    // Distinct nonzero high/low words: every field must land in its exact slot.
    const reader_marker m{0x1'0000'0002ull, 0xF'0000'0003ull};
    const fw_record     slot = encode_marker(m);
    EXPECT_EQ(slot.ts_lo, 2u);
    EXPECT_EQ(slot.ts_hi, 1u);
    EXPECT_EQ(slot.record_type, kRecReaderMarker);
    EXPECT_EQ(slot.dispatch_id, 3u) << "seen_at_ns low";
    EXPECT_EQ(slot.doorbell_off, 0xFu) << "seen_at_ns high";

    // Boundary values decode back to the same marker.
    for(uint64_t epoch : {uint64_t(0), uint64_t(1), uint64_t(UINT64_MAX)})
        for(uint64_t seen : {uint64_t(0), uint64_t(1), uint64_t(UINT64_MAX)})
        {
            const reader_marker b{epoch, seen};
            const auto          d = decode_marker(encode_marker(b));
            EXPECT_EQ(d.epoch, b.epoch);
            EXPECT_EQ(d.seen_at_ns, b.seen_at_ns);
        }
}

// --- published / taken / processed are distinct publications -----------------

TEST(record_stream, position_publication_separation_and_repreallocate)
{
    test_mapping::reset();
    stream_t st;
    ASSERT_TRUE(st.preallocate(2));
    std::vector<fw_record> src(100);
    for(size_t i = 0; i < src.size(); ++i)
        src[i] = make_record(i);
    reader_alloc_log log;
    // Appended but unpublished: nothing is published.
    st.append(src.data(), src.size(), log);
    EXPECT_EQ(st.published(), 0u);
    EXPECT_EQ(st.taken(), 0u);
    EXPECT_EQ(st.processed(), 0u);
    st.publish();
    EXPECT_EQ(st.published(), 100u);
    EXPECT_EQ(st.taken(), 0u) << "publish does not move taken";
    EXPECT_EQ(st.processed(), 0u);
    copy_sink sink;
    st.copy_out(sink, 2);
    EXPECT_EQ(st.taken(), 100u);
    EXPECT_EQ(st.processed(), 0u) << "copy-out does not publish processed";
    EXPECT_EQ(sink.all.size(), 100u);
    st.publish_processed(st.taken());
    EXPECT_EQ(st.processed(), 100u);
    // reset()/re-preallocate returns every published position to zero.
    EXPECT_TRUE(st.preallocate(2));
    EXPECT_EQ(st.published(), 0u);
    EXPECT_EQ(st.taken(), 0u);
    EXPECT_EQ(st.processed(), 0u);
    EXPECT_EQ(st.write_pos_quiescent(), 0u);
    expect_ownership(st);
    st.append(src.data(), 40, log);
    st.publish();
    st.copy_out(sink, 2);
    EXPECT_EQ(st.taken(), 40u);
    st.publish_processed(st.taken());
    EXPECT_EQ(st.processed(), 40u);
}

// --- Lifecycle: re-preallocate and failed re-preallocate on a used stream -----

TEST(record_stream, repreallocate_resets_positions_and_leaks_nothing)
{
    test_mapping::reset();
    stream_t st;
    ASSERT_TRUE(st.preallocate(4));
    std::vector<fw_record> src(100);
    for(size_t i = 0; i < src.size(); ++i)
        src[i] = make_record(i);
    reader_alloc_log log;
    st.append(src.data(), src.size(), log);
    st.publish();
    EXPECT_EQ(st.chain_length_quiescent(), 1u);
    EXPECT_EQ(st.published(), 100u);
    // Re-preallocate in place: the old chain and spares are freed, nothing leaks, and
    // every position and state field returns to the default.
    EXPECT_TRUE(st.preallocate(3));
    EXPECT_EQ(
        test_mapping::map_successes.load(),
        test_mapping::unmap_successes.load() + st.chain_length_quiescent() + st.spares_quiescent());
    EXPECT_EQ(st.chain_length_quiescent(), 1u);
    EXPECT_EQ(st.spares_quiescent(), 3u);
    EXPECT_EQ(st.write_pos_quiescent(), 0u);
    EXPECT_EQ(st.published(), 0u);
    EXPECT_EQ(st.taken(), 0u);
    EXPECT_EQ(st.processed(), 0u);
    EXPECT_TRUE(st.armed());
    // The re-preallocated stream starts at position 0 again.
    st.append(src.data(), 50, log);
    st.publish();
    EXPECT_EQ(st.published(), 50u);
    copy_sink sink;
    st.copy_out(sink, 3);
    EXPECT_EQ(st.taken(), 50u);
    expect_records(sink.all, 0, 50);
    // A first-map failure on a used stream leaves a clean stream, not the old one.
    test_mapping::set_fail_nth(test_mapping::map_calls.load() + 1);
    EXPECT_FALSE(st.preallocate(2));
    test_mapping::set_fail_nth(0);
    EXPECT_FALSE(st.armed());
    EXPECT_EQ(st.published(), 0u);
    EXPECT_EQ(st.chain_length_quiescent(), 0u);
    EXPECT_EQ(st.spares_quiescent(), 0u);
    EXPECT_EQ(test_mapping::map_successes.load(), test_mapping::unmap_successes.load());
    // Recovery: a fresh preallocate works.
    EXPECT_TRUE(st.preallocate(2));
    EXPECT_TRUE(st.armed());
    EXPECT_EQ(st.published(), 0u);
    expect_ownership(st);
}

}  // namespace
