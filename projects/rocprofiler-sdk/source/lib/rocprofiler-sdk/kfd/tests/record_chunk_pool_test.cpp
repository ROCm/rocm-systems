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

// P1 unit tests for the dispatch-log record chunk pool: SPSC pop/push, counts,
// static allocate() with its explicit page touch, chunk_deleter, nested chunk.
// Ownership is asserted directly through the counting test_mapping policy; the
// prefault-oracle test runs on the production mapping with getrusage(RUSAGE_THREAD).

#include "lib/rocprofiler-sdk/kfd/record_chunk_pool.hpp"
#include "lib/rocprofiler-sdk/kfd/dlog_drain.hpp"

#include <gtest/gtest.h>

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <thread>
#include <type_traits>
#include <vector>

#include <sys/mman.h>
#include <sys/resource.h>
#include <unistd.h>

namespace
{
using namespace rocprofiler::kfd;

// The design's test seam: the same shape as chunk_mapping plus instrumentation.
// Counts every map/unmap with its length, records the unmapped pointer, can fail
// the k-th map, and fills every new page with 0xA5 so tests can prove exactly
// which bytes allocate() and the pool wrote. Attempts and successful syscalls
// are counted separately so tests can assert the syscall results themselves.
struct test_mapping
{
    static inline std::atomic<size_t>      map_calls{0};      // attempts
    static inline std::atomic<size_t>      map_successes{0};  // real mmap succeeded
    static inline std::atomic<size_t>      map_bytes_total{0};
    static inline std::atomic<size_t>      unmap_calls{0};      // attempts
    static inline std::atomic<size_t>      unmap_successes{0};  // munmap returned 0
    static inline std::atomic<size_t>      unmap_failures{0};   // munmap returned nonzero
    static inline std::atomic<size_t>      unmap_bytes{0};
    static inline std::atomic<const void*> unmap_ptr{nullptr};
    static inline std::atomic<size_t>      fail_map_nth{0};  // fail the k-th map; 0 = never

    static void reset(size_t fail_nth = 0)
    {
        map_calls.store(0, std::memory_order_relaxed);
        map_successes.store(0, std::memory_order_relaxed);
        map_bytes_total.store(0, std::memory_order_relaxed);
        unmap_calls.store(0, std::memory_order_relaxed);
        unmap_successes.store(0, std::memory_order_relaxed);
        unmap_failures.store(0, std::memory_order_relaxed);
        unmap_bytes.store(0, std::memory_order_relaxed);
        unmap_ptr.store(nullptr, std::memory_order_relaxed);
        fail_map_nth.store(fail_nth, std::memory_order_relaxed);
    }

    static void* map(size_t bytes) noexcept
    {
        const size_t nth = map_calls.fetch_add(1, std::memory_order_relaxed) + 1;
        if(fail_map_nth.load(std::memory_order_relaxed) == nth) return nullptr;
        void* const p =
            ::mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if(p == MAP_FAILED) return nullptr;
        std::memset(p, 0xA5, bytes);  // poison: any byte allocate() did not write shows
        map_bytes_total.fetch_add(bytes, std::memory_order_relaxed);
        map_successes.fetch_add(1, std::memory_order_relaxed);
        return p;
    }

    static void unmap(void* p, size_t bytes) noexcept
    {
        unmap_calls.fetch_add(1, std::memory_order_relaxed);
        unmap_bytes.fetch_add(bytes, std::memory_order_relaxed);
        unmap_ptr.store(p, std::memory_order_relaxed);
        if(::munmap(p, bytes) == 0)
            unmap_successes.fetch_add(1, std::memory_order_relaxed);
        else
            unmap_failures.fetch_add(1, std::memory_order_relaxed);
    }
};

using chunk_pool = basic_record_chunk_pool<test_mapping>;

// Deterministic 2-party step latch for a fixed sequence of rendezvous phases.
// Both parties call arrive_and_wait(phase) with the same 0-based phase; the
// phase completes only when both have arrived. The completion target is the
// monotone (phase + 1) * kParties, so a thread can never run ahead into a
// later phase and deadlock the other side.
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

// --- P1-1: layout -----------------------------------------------------------

namespace
{
// ru_minflt for the calling thread; false when getrusage itself fails.
bool
thread_minflt(long* out)
{
    rusage ru{};
    if(getrusage(RUSAGE_THREAD, &ru) != 0) return false;
    *out = static_cast<long>(ru.ru_minflt);
    return true;
}
}  // namespace

TEST(record_chunk_pool, chunk_layout_is_sixty_four_byte_header_plus_packed_slots)
{
    using chunk = chunk_pool::chunk;
    EXPECT_EQ(offsetof(chunk, slots), 64u);
    EXPECT_EQ(sizeof(chunk), 64u + 20u * chunk::kRecords);
    const size_t page  = static_cast<size_t>(::sysconf(_SC_PAGESIZE));
    const size_t bytes = (sizeof(chunk) + page - 1) / page * page;
    EXPECT_EQ(chunk_pool::map_bytes(), bytes);
    EXPECT_TRUE(std::is_trivially_default_constructible_v<fw_record>);
}

// --- P1-2: header-only construction -----------------------------------------

TEST(record_chunk_pool, allocate_constructs_only_the_chunk_header)
{
    test_mapping::reset();
    {
        auto c = chunk_pool::allocate();
        ASSERT_NE(c, nullptr);
        const size_t page = static_cast<size_t>(::sysconf(_SC_PAGESIZE));
        EXPECT_EQ(reinterpret_cast<uintptr_t>(c.get()) % page, 0u)
            << "the mapping must be page-aligned";
        EXPECT_EQ(c->base_pos, 0u);
        EXPECT_EQ(c->next.load(), nullptr);
        // Every page-start byte inside the slots was written 0 by the explicit
        // touch; every other slot byte must still be the 0xA5 poison. Checking
        // the page starts (instead of skipping them) catches an omitted touch
        // and a read-only touch here, independent of any fault-count oracle
        // and of THP/mTHP behavior.
        constexpr size_t kSlotsOff  = offsetof(chunk_pool::chunk, slots);
        const size_t     slot_bytes = sizeof(chunk_pool::chunk) - kSlotsOff;
        const auto*      bytes = static_cast<const uint8_t*>(static_cast<const void*>(c->slots));
        size_t           zero_pages = 0;
        size_t           bad_count  = 0;
        size_t           first_bad  = slot_bytes;
        for(size_t off = 0; off < slot_bytes; ++off)
        {
            const size_t  chunk_off = kSlotsOff + off;
            const bool    is_touch  = chunk_off % page == 0;
            const uint8_t want      = is_touch ? 0 : 0xA5;
            if(is_touch) ++zero_pages;
            if(bytes[off] != want)
            {
                if(first_bad == slot_bytes) first_bad = chunk_off;
                ++bad_count;
            }
        }
        ASSERT_EQ(bad_count, 0u) << "first wrong slot byte at chunk offset " << first_bad;
        // The chunk spans map_bytes()/page pages; page 0's start (offset 0) is in
        // the header, so pages - 1 page starts fall inside the slots.
        EXPECT_EQ(zero_pages, chunk_pool::map_bytes() / page - 1)
            << "the oracle saw no page starts inside the slots";
        EXPECT_EQ(test_mapping::map_calls.load(), 1u);
        EXPECT_EQ(test_mapping::map_successes.load(), 1u);
        EXPECT_EQ(test_mapping::unmap_calls.load(), 0u);
    }
    EXPECT_EQ(test_mapping::unmap_calls.load(), 1u);
}

// --- P1-3: the prefault is a write fault (production mapping) ----------------

TEST(record_chunk_pool, allocate_write_faults_every_page)
{
    using pool         = rocprofiler::kfd::record_chunk_pool;
    const size_t page  = static_cast<size_t>(::sysconf(_SC_PAGESIZE));
    const size_t bytes = pool::map_bytes();

    // Control: a plain mapping faults only when first written. If writing every
    // page adds no fault the minflt oracle is invalid on this kernel, and the
    // test must fail rather than pass vacuously.
    long ctrl_before = 0;
    ASSERT_TRUE(thread_minflt(&ctrl_before));
    void* const ctrl =
        ::mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    ASSERT_NE(ctrl, MAP_FAILED);
    for(size_t off = 0; off < bytes; off += page)
        static_cast<volatile uint8_t*>(ctrl)[off] = 0;
    long ctrl_after = 0;
    ASSERT_TRUE(thread_minflt(&ctrl_after));
    ::munmap(ctrl, bytes);
    const long ctrl_faults = ctrl_after - ctrl_before;
    ASSERT_GE(ctrl_faults, 1)
        << "control mapping took no write fault: minflt oracle invalid on this kernel";

    // allocate() must take the faults itself. Only the minflt reads and
    // allocate() run inside the measured interval; assertions and the write
    // phase afterwards stay outside it.
    long before     = 0;
    long alloc_done = 0;
    ASSERT_TRUE(thread_minflt(&before));
    auto c = pool::allocate();
    ASSERT_TRUE(thread_minflt(&alloc_done));
    const long alloc_faults = alloc_done - before;
    ASSERT_NE(c, nullptr);
    EXPECT_GE(alloc_faults, 1);
    for(size_t off = 0; off < bytes; off += page)
        static_cast<volatile uint8_t*>(static_cast<void*>(c.get()))[off] = 1;
    long after = 0;
    ASSERT_TRUE(thread_minflt(&after));
    const long write_faults = after - alloc_done;
    EXPECT_LE(write_faults, 2) << "allocate() faulted " << alloc_faults
                               << " pages but the write phase still faulted " << write_faults;
}

// --- P1-4: map failure -------------------------------------------------------

TEST(record_chunk_pool, allocate_returns_null_when_map_fails)
{
    test_mapping::reset(/*fail_nth=*/1);
    auto c = chunk_pool::allocate();
    EXPECT_EQ(c, nullptr);
    EXPECT_EQ(test_mapping::map_calls.load(), 1u);
    EXPECT_EQ(test_mapping::map_successes.load(), 0u) << "the injected map failure did not fail";
    EXPECT_EQ(test_mapping::unmap_calls.load(), 0u) << "a failed map must not be unmapped";
}

// --- P1-5: chunk_deleter -----------------------------------------------------

TEST(record_chunk_pool, deleter_unmaps_the_whole_mapping_once)
{
    test_mapping::reset();
    chunk_pool::chunk* raw = nullptr;
    {
        auto c = chunk_pool::allocate();
        ASSERT_NE(c, nullptr);
        raw = c.get();
    }
    EXPECT_EQ(test_mapping::unmap_calls.load(), 1u);
    EXPECT_EQ(test_mapping::unmap_successes.load(), 1u) << "the munmap syscall itself failed";
    EXPECT_EQ(test_mapping::unmap_bytes.load(), chunk_pool::map_bytes());
    EXPECT_EQ(test_mapping::unmap_ptr.load(), static_cast<const void*>(raw))
        << "the chunk address IS the mapping base";
    EXPECT_EQ(test_mapping::map_calls.load(), 1u);
}

// --- P1-6: FIFO and full -----------------------------------------------------

TEST(record_chunk_pool, push_is_fifo_and_rejects_beyond_capacity)
{
    test_mapping::reset();
    chunk_pool pool;
    for(size_t i = 0; i < chunk_pool::kCapacity; ++i)
    {
        auto c = chunk_pool::allocate();
        ASSERT_NE(c, nullptr);
        c->base_pos = i;  // tag every chunk so FIFO order is observable
        EXPECT_TRUE(pool.push(std::move(c)));
        EXPECT_EQ(c, nullptr) << "a successful push moves from `c`";
    }
    EXPECT_EQ(test_mapping::map_calls.load(), chunk_pool::kCapacity);
    EXPECT_EQ(test_mapping::map_successes.load(), chunk_pool::kCapacity);
    // Full: one more push is refused, the caller keeps ownership, nothing is freed.
    auto extra = chunk_pool::allocate();
    ASSERT_NE(extra, nullptr);
    extra->base_pos = chunk_pool::kCapacity;
    EXPECT_FALSE(pool.push(std::move(extra)));
    ASSERT_NE(extra, nullptr) << "a refused push must leave the caller owning `c`";
    EXPECT_EQ(extra->base_pos, chunk_pool::kCapacity) << "a refused push must not move from `c`";
    EXPECT_EQ(test_mapping::unmap_calls.load(), 0u);
    // Pops come back in push order; a pop on empty is nullptr.
    for(size_t i = 0; i < chunk_pool::kCapacity; ++i)
    {
        auto c = pool.pop();
        ASSERT_NE(c, nullptr);
        EXPECT_EQ(c->base_pos, i);
    }
    EXPECT_EQ(pool.pop(), nullptr);
}

// --- P1-7: counts ---------------------------------------------------------

TEST(record_chunk_pool, counts_exact_at_latches_and_bounded_between)
{
    test_mapping::reset();

    // Single-threaded: available() == occupancy() == pushes - pops after every
    // operation, including the pop-on-empty (counts are exact without threads).
    {
        chunk_pool pool;
        auto       a = chunk_pool::allocate();
        ASSERT_NE(a, nullptr);
        EXPECT_TRUE(pool.push(std::move(a)));
        EXPECT_EQ(pool.available(), 1u);
        EXPECT_EQ(pool.occupancy(), 1u);
        auto b = chunk_pool::allocate();
        ASSERT_NE(b, nullptr);
        EXPECT_TRUE(pool.push(std::move(b)));
        EXPECT_EQ(pool.available(), 2u);
        EXPECT_EQ(pool.occupancy(), 2u);
        {
            auto c = pool.pop();
            ASSERT_NE(c, nullptr);
        }
        EXPECT_EQ(pool.available(), 1u);
        EXPECT_EQ(pool.occupancy(), 1u);
        {
            auto c = pool.pop();
            ASSERT_NE(c, nullptr);
        }
        EXPECT_EQ(pool.available(), 0u);
        EXPECT_EQ(pool.occupancy(), 0u);
        EXPECT_EQ(pool.pop(), nullptr) << "pop on empty is nullptr";
        EXPECT_EQ(pool.available(), 0u);
        EXPECT_EQ(pool.occupancy(), 0u);
    }
    EXPECT_EQ(test_mapping::map_calls.load(), 2u);
    EXPECT_EQ(test_mapping::unmap_calls.load(), 2u);
    EXPECT_EQ(test_mapping::unmap_successes.load(), 2u);

    constexpr int    kSteps  = 8;
    constexpr size_t kPushes = 4;
    // The producer pushes at steps 1,2,4,6; the consumer pops at steps 3,5,7.
    const bool prod_push[kSteps] = {false, true, true, false, true, false, true, false};
    const bool cons_pop[kSteps]  = {false, false, false, true, false, true, false, true};

    chunk_pool pool;

    // The push schedule's chunks are allocated on the MAIN thread: an ASSERT in
    // a worker returns from its lambda and would strand the peer at the latch.
    std::array<chunk_pool::chunk_ptr, kPushes> prealloc;
    for(auto& pc : prealloc)
    {
        pc = chunk_pool::allocate();
        ASSERT_NE(pc, nullptr);
    }

    step_latch          latch;
    std::atomic<size_t> pushes{0};
    std::atomic<size_t> pops{0};

    auto producer = std::thread{[&]() {
        size_t my_pushes     = 0;
        size_t pops_at_latch = 0;
        for(int step = 0; step < kSteps; ++step)
        {
            if(prod_push[step])
            {
                auto c      = std::move(prealloc[my_pushes]);
                c->base_pos = my_pushes;
                while(!pool.push(std::move(c)))
                    std::this_thread::yield();
                ++my_pushes;
                pushes.store(my_pushes, std::memory_order_release);
                // Free-running: the producer's accessor is an upper bound.
                EXPECT_LE(pool.occupancy(), my_pushes - pops_at_latch);
            }
            // Quiescent interval: perform the operation, publish, meet at the
            // first rendezvous, sample and assert, then the second rendezvous
            // holds both threads until both have finished sampling, so neither
            // starts the next step's operation inside the interval.
            latch.arrive_and_wait(2 * step);
            pops_at_latch = pops.load(std::memory_order_acquire);
            EXPECT_EQ(pool.occupancy(), my_pushes - pops_at_latch);
            latch.arrive_and_wait(2 * step + 1);
        }
    }};
    auto consumer = std::thread{[&]() {
        size_t my_pops         = 0;
        size_t pushes_at_latch = 0;
        for(int step = 0; step < kSteps; ++step)
        {
            if(cons_pop[step])
            {
                auto c = pool.pop();
                EXPECT_NE(c, nullptr) << "the schedule never pops an empty pool";
                if(c)
                {
                    ++my_pops;
                    pops.store(my_pops, std::memory_order_release);
                }
                // Free-running: the consumer's accessor is a lower bound.
                EXPECT_GE(pool.available(), pushes_at_latch - my_pops);
            }
            latch.arrive_and_wait(2 * step);
            pushes_at_latch = pushes.load(std::memory_order_acquire);
            EXPECT_EQ(pool.available(), pushes_at_latch - my_pops);
            latch.arrive_and_wait(2 * step + 1);
        }
    }};
    producer.join();
    consumer.join();
    // 4 pushes, 3 pops: exactly one spare left, and nothing leaked. Both counts
    // are readable here: the threads are joined.
    EXPECT_EQ(pool.available(), 1u);
    EXPECT_EQ(pool.occupancy(), 1u);
    {
        auto last = pool.pop();
        ASSERT_NE(last, nullptr);
    }
    EXPECT_EQ(pool.pop(), nullptr);
    EXPECT_EQ(test_mapping::map_calls.load(), kPushes + 2u);
    EXPECT_EQ(test_mapping::map_successes.load(), kPushes + 2u);
    EXPECT_EQ(test_mapping::unmap_calls.load(), kPushes + 2u);
    EXPECT_EQ(test_mapping::unmap_successes.load(), kPushes + 2u);
    EXPECT_EQ(test_mapping::unmap_failures.load(), 0u);
}

// --- P1-8: clear / destructor / abandon --------------------------------------

TEST(record_chunk_pool, clear_frees_and_abandon_forgets)
{
    // Explicit clear() unmaps every pooled chunk and empties the pool.
    test_mapping::reset();
    {
        chunk_pool pool;
        for(size_t i = 0; i < 3; ++i)
        {
            auto c = chunk_pool::allocate();
            ASSERT_NE(c, nullptr);
            EXPECT_TRUE(pool.push(std::move(c)));
        }
        EXPECT_EQ(test_mapping::unmap_calls.load(), 0u);
        pool.clear();
        EXPECT_EQ(test_mapping::unmap_calls.load(), 3u);
        EXPECT_EQ(test_mapping::unmap_successes.load(), 3u);
        EXPECT_EQ(test_mapping::unmap_failures.load(), 0u);
        EXPECT_EQ(pool.available(), 0u);
        EXPECT_EQ(pool.occupancy(), 0u);
    }
    EXPECT_EQ(test_mapping::unmap_calls.load(), 3u) << "the pool was already empty";

    // The destructor calls clear().
    test_mapping::reset();
    {
        chunk_pool pool;
        for(size_t i = 0; i < 2; ++i)
        {
            auto c = chunk_pool::allocate();
            ASSERT_NE(c, nullptr);
            EXPECT_TRUE(pool.push(std::move(c)));
        }
    }
    EXPECT_EQ(test_mapping::unmap_calls.load(), 2u);

    // abandon() unmaps nothing and forgets the pool; the caller owns the mappings.
    test_mapping::reset();
    {
        chunk_pool                      pool;
        std::vector<chunk_pool::chunk*> saved;
        for(size_t i = 0; i < 2; ++i)
        {
            auto c = chunk_pool::allocate();
            ASSERT_NE(c, nullptr);
            saved.push_back(c.get());
            EXPECT_TRUE(pool.push(std::move(c)));
        }
        pool.abandon();
        EXPECT_EQ(test_mapping::unmap_calls.load(), 0u) << "abandon() must not munmap";
        EXPECT_EQ(pool.available(), 0u);
        EXPECT_EQ(pool.occupancy(), 0u);
        // The fork child would leak these by design; this test frees them itself.
        for(auto* s : saved)
            chunk_pool::chunk_deleter{}(s);
    }
    EXPECT_EQ(test_mapping::unmap_calls.load(), 2u);
}

// --- P1-9: stress (must run under TSan) ---------------------------------------

TEST(record_chunk_pool, spsc_stress_preserves_order_and_contents)
{
    test_mapping::reset();
    constexpr size_t kIters  = 10'000'000;
    constexpr size_t kChunks = 8;
    // Pool a: the producer pushes, the consumer pops (the pool under test).
    // Pool b: the consumer pushes, the producer pops (returns chunks for refill).
    // Each pool still has one popper and one pusher; together they circulate all
    // 8 chunks through pool a 10M times.
    chunk_pool a;
    chunk_pool b;
    for(size_t i = 0; i < kChunks; ++i)
    {
        auto c = chunk_pool::allocate();
        ASSERT_NE(c, nullptr);
        ASSERT_TRUE(b.push(std::move(c)));
    }
    std::atomic<size_t> producer_waits{0};
    std::atomic<size_t> consumer_waits{0};

    auto producer = std::thread{[&]() {
        for(size_t tag = 0; tag < kIters; ++tag)
        {
            auto c = b.pop();
            while(!c)
            {
                producer_waits.fetch_add(1, std::memory_order_relaxed);
                std::this_thread::yield();
                c = b.pop();
            }
            // Sequence tag in the header, pattern in the first and last slot.
            c->base_pos                                           = tag;
            c->slots[0].dispatch_id                               = static_cast<uint32_t>(tag);
            c->slots[chunk_pool::chunk::kRecords - 1].dispatch_id = static_cast<uint32_t>(tag);
            while(!a.push(std::move(c)))
            {
                producer_waits.fetch_add(1, std::memory_order_relaxed);
                std::this_thread::yield();
            }
        }
    }};
    auto consumer = std::thread{[&]() {
        for(size_t expect = 0; expect < kIters; ++expect)
        {
            auto c = a.pop();
            while(!c)
            {
                consumer_waits.fetch_add(1, std::memory_order_relaxed);
                std::this_thread::yield();
                c = a.pop();
            }
            EXPECT_EQ(c->base_pos, expect) << "spare ring lost FIFO order";
            EXPECT_EQ(c->slots[0].dispatch_id, static_cast<uint32_t>(expect));
            EXPECT_EQ(c->slots[chunk_pool::chunk::kRecords - 1].dispatch_id,
                      static_cast<uint32_t>(expect))
                << "slot bytes were overwritten while the chunk was pooled";
            while(!b.push(std::move(c)))
            {
                consumer_waits.fetch_add(1, std::memory_order_relaxed);
                std::this_thread::yield();
            }
        }
    }};
    producer.join();
    consumer.join();
    // With 8 chunks and one held per thread at most, the threads cannot have run
    // strictly serialized: at least one side must have waited for the other.
    EXPECT_GT(producer_waits.load() + consumer_waits.load(), 0u)
        << "producer waited " << producer_waits.load() << ", consumer waited "
        << consumer_waits.load() << ": no interleaving was observed";
    // Every chunk is back in b (kIters pushes/pops on both sides cancel); free
    // them and check the ownership balance.
    size_t freed = 0;
    while(auto c = a.pop())
    {
        ++freed;
        (void) c;
    }
    while(auto c = b.pop())
    {
        ++freed;
        (void) c;
    }
    EXPECT_EQ(freed, kChunks) << "a chunk was lost or destroyed twice";
    EXPECT_EQ(test_mapping::map_calls.load(), kChunks);
    EXPECT_EQ(test_mapping::map_successes.load(), kChunks);
    EXPECT_EQ(test_mapping::unmap_calls.load(), kChunks);
    EXPECT_EQ(test_mapping::unmap_successes.load(), kChunks);
    EXPECT_EQ(test_mapping::unmap_failures.load(), 0u);
}
}  // namespace
