// Unit tests for cpu_snap::sampling::flat_tid_map.
//
// Covers fast-path slot insert/find/erase, default-constructed value
// via operator[], for_each iteration, and the overflow fallback that
// triggers when more than k_probe_limit tids collide on the same
// home bucket.

#include <cstddef>
#include <gtest/gtest.h>

#include <cstdint>
#include <random>
#include <sched.h>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "cpu_snap/sampling/flat_tid_map.hpp"

using cpu_snap::sampling::flat_tid_map;

namespace
{

constexpr pid_t         k_tid_a              = 1001;
constexpr pid_t         k_tid_b              = 1002;
constexpr int           k_value_a            = 17;
constexpr int           k_value_b            = 42;
constexpr int           k_value_overwrite    = 99;
constexpr std::uint32_t k_rng_seed_random    = 0xC0FFEE;
constexpr std::uint32_t k_rng_seed_for_each  = 0xBEEF1234;
constexpr std::size_t   k_random_tid_count   = 100;
constexpr std::size_t   k_mixed_random_count = 50;
constexpr std::size_t   k_overflow_burst     = 32;
constexpr std::size_t   k_probe_limit        = 8;
constexpr pid_t         k_tid_low_mask       = 0xFFFFFF;
constexpr int           k_value_multiplier   = 7;
constexpr int           k_overflow_value_off = 1000;
constexpr pid_t         k_collision_low_a    = 0x42;
constexpr pid_t         k_collision_low_b    = 0x55;
constexpr pid_t         k_collision_low_c    = 0x77;
constexpr std::size_t   k_overflow_probe_idx = 20;

}  // namespace

TEST(FlatTidMapTest, DefaultConstructedIsEmpty)
{
    flat_tid_map<int> map;
    EXPECT_EQ(map.size(), 0U);
    EXPECT_EQ(map.overflow_size(), 0U);
    EXPECT_EQ(map.find(k_tid_a), nullptr);
}

TEST(FlatTidMapTest, OperatorIndexInsertsDefaultValue)
{
    flat_tid_map<int> map;
    const int&        value = map[k_tid_a];
    EXPECT_EQ(value, 0);
    EXPECT_EQ(map.size(), 1U);
}

TEST(FlatTidMapTest, OperatorIndexReturnsExistingReference)
{
    flat_tid_map<int> map;
    map[k_tid_a] = k_value_b;
    EXPECT_EQ(map[k_tid_a], k_value_b);
    EXPECT_EQ(map.size(), 1U);
}

TEST(FlatTidMapTest, FindReturnsPointerToValueWhenPresent)
{
    flat_tid_map<int> map;
    map[k_tid_a] = k_value_a;

    int* ptr = map.find(k_tid_a);
    ASSERT_NE(ptr, nullptr);
    EXPECT_EQ(*ptr, k_value_a);

    *ptr = k_value_overwrite;
    EXPECT_EQ(map[k_tid_a], k_value_overwrite);
}

TEST(FlatTidMapTest, FindReturnsNullptrWhenAbsent)
{
    flat_tid_map<int> map;
    map[k_tid_a] = 1;
    EXPECT_EQ(map.find(k_tid_b), nullptr);
}

TEST(FlatTidMapTest, FindConstOverloadCompilesAndReturnsValue)
{
    flat_tid_map<int> map;
    map[k_tid_a] = k_value_a;

    const auto& const_map = map;
    const int*  ptr       = const_map.find(k_tid_a);
    ASSERT_NE(ptr, nullptr);
    EXPECT_EQ(*ptr, k_value_a);
    EXPECT_EQ(const_map.find(k_tid_b), nullptr);
}

TEST(FlatTidMapTest, EraseRemovesEntry)
{
    flat_tid_map<int> map;
    map[k_tid_a] = 1;
    map[k_tid_b] = 2;
    EXPECT_EQ(map.size(), 2U);

    map.erase(k_tid_a);
    EXPECT_EQ(map.size(), 1U);
    EXPECT_EQ(map.find(k_tid_a), nullptr);
    ASSERT_NE(map.find(k_tid_b), nullptr);
    EXPECT_EQ(*map.find(k_tid_b), 2);
}

TEST(FlatTidMapTest, EraseAbsentTidIsNoop)
{
    flat_tid_map<int> map;
    map[k_tid_a] = 1;
    map.erase(k_tid_b);
    EXPECT_EQ(map.size(), 1U);
}

TEST(FlatTidMapTest, HundredRandomTidsAllFindable)
{
    flat_tid_map<std::uint64_t> map;
    std::mt19937                rng{ k_rng_seed_random };
    std::vector<pid_t>          inserted;
    std::unordered_set<pid_t>   seen;

    while(inserted.size() < k_random_tid_count)
    {
        const auto tid =
            static_cast<pid_t>(rng() & static_cast<std::uint32_t>(k_tid_low_mask));
        if(tid == 0 || !seen.insert(tid).second) continue;
        map[tid] = static_cast<std::uint64_t>(tid) * k_value_multiplier;
        inserted.push_back(tid);
    }

    EXPECT_EQ(map.size(), inserted.size());
    for(pid_t const tid : inserted)
    {
        const auto* value = map.find(tid);
        ASSERT_NE(value, nullptr) << "tid " << tid << " not findable";
        EXPECT_EQ(*value, static_cast<std::uint64_t>(tid) * k_value_multiplier);
    }
}

TEST(FlatTidMapTest, CollisionBurstTriggersOverflowFallback)
{
    // All tids share the same low byte so they hash to the same home
    // bucket. Once the 8-slot local cluster fills, additional
    // collisions must spill into the overflow map.
    flat_tid_map<int>  map;
    std::vector<pid_t> tids;

    for(std::size_t i = 0; i < k_overflow_burst; ++i)
    {
        const auto tid = static_cast<pid_t>((i << 8) | k_collision_low_a);
        map[tid]       = static_cast<int>(i);
        tids.push_back(tid);
    }

    EXPECT_EQ(map.size(), k_overflow_burst);
    EXPECT_EQ(map.overflow_size(), k_overflow_burst - k_probe_limit);

    for(std::size_t i = 0; i < tids.size(); ++i)
    {
        const int* value = map.find(tids[i]);
        ASSERT_NE(value, nullptr) << "tid " << tids[i] << " not findable";
        EXPECT_EQ(*value, static_cast<int>(i));
    }
}

TEST(FlatTidMapTest, EraseClearsOverflowEntry)
{
    flat_tid_map<int>  map;
    std::vector<pid_t> tids;

    for(std::size_t i = 0; i < k_overflow_burst; ++i)
    {
        const auto tid = static_cast<pid_t>((i << 8) | k_collision_low_b);
        map[tid]       = static_cast<int>(i);
        tids.push_back(tid);
    }

    const pid_t overflow_tid = tids[k_overflow_probe_idx];
    map.erase(overflow_tid);
    EXPECT_EQ(map.find(overflow_tid), nullptr);
    EXPECT_EQ(map.size(), k_overflow_burst - 1);
}

TEST(FlatTidMapTest, ForEachVisitsEveryEntryExactlyOnce)
{
    flat_tid_map<int>              map;
    std::mt19937                   rng{ k_rng_seed_for_each };
    std::unordered_map<pid_t, int> expected;

    for(std::size_t i = 0; i < k_mixed_random_count; ++i)
    {
        pid_t tid = 0;
        while(tid == 0 || expected.contains(tid))
        {
            tid = static_cast<pid_t>(rng() &
                                     static_cast<std::uint32_t>(k_tid_low_mask));
        }
        map[tid]      = static_cast<int>(i);
        expected[tid] = static_cast<int>(i);
    }
    for(std::size_t i = 0; i < 16; ++i)
    {
        const auto tid = static_cast<pid_t>((i << 8) | k_collision_low_c);
        if(expected.contains(tid)) continue;
        const int value = k_overflow_value_off + static_cast<int>(i);
        map[tid]        = value;
        expected[tid]   = value;
    }

    std::unordered_map<pid_t, int> visited;
    map.for_each([&](pid_t tid, int value) {
        EXPECT_EQ(visited.count(tid), 0U) << "tid " << tid << " visited twice";
        visited[tid] = value;
    });

    EXPECT_EQ(visited.size(), expected.size());
    for(const auto& [tid, value] : expected)
    {
        ASSERT_EQ(visited.count(tid), 1U) << "tid " << tid << " missing";
        EXPECT_EQ(visited[tid], value);
    }
}

TEST(FlatTidMapTest, EraseKeepsProbeChainIntactForLaterCollisions)
{
    // tid_home and tid_probed collide in the same probe chain.
    flat_tid_map<int> map;
    constexpr pid_t   tid_home   = 1000;
    constexpr pid_t   tid_probed = 1256;

    map[tid_home]   = 1;
    map[tid_probed] = 2;
    ASSERT_EQ(map.size(), 2U);

    map.erase(tid_home);
    EXPECT_EQ(map.size(), 1U);

    const int* probed_value = map.find(tid_probed);
    ASSERT_NE(probed_value, nullptr) << "erase broke the probe chain";
    EXPECT_EQ(*probed_value, 2);

    // operator[] must not insert a duplicate slot entry for tid_probed.
    EXPECT_EQ(map[tid_probed], 2);
    EXPECT_EQ(map.size(), 1U);
}

TEST(FlatTidMapTest, OperatorIndexDoesNotDuplicateOverflowEntryAfterErase)
{
    // Fill a full probe chain so the 9th colliding tid spills into overflow.
    flat_tid_map<int>  map;
    std::vector<pid_t> slot_tids;
    for(std::size_t i = 0; i < k_probe_limit; ++i)
    {
        const auto tid = static_cast<pid_t>((i << 8) | k_collision_low_a);
        map[tid]       = static_cast<int>(i);
        slot_tids.push_back(tid);
    }
    const auto overflow_tid =
        static_cast<pid_t>((k_probe_limit << 8) | k_collision_low_a);
    map[overflow_tid] = k_overflow_value_off;
    ASSERT_EQ(map.overflow_size(), 1U);

    map.erase(slot_tids.front());
    EXPECT_EQ(map[overflow_tid], k_overflow_value_off);
    EXPECT_EQ(map.overflow_size(), 1U);
    EXPECT_EQ(map.size(), k_probe_limit);
}

TEST(FlatTidMapTest, ForEachConstOverloadCompiles)
{
    flat_tid_map<int> map;
    map[k_tid_a]            = 1;
    map[k_tid_b]            = 2;
    const auto& const_map   = map;
    std::size_t seen_count  = 0;
    int         sum         = 0;
    const_map.for_each([&](pid_t /*tid*/, int value) {
        ++seen_count;
        sum += value;
    });
    EXPECT_EQ(seen_count, 2U);
    EXPECT_EQ(sum, 3);
}
