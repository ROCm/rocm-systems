// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "entity_utility.hpp"

#include <gtest/gtest.h>

#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>

namespace
{

using profiler_hub::entity_utility;

using int_utility_t    = entity_utility<std::unordered_map<int, int>, int>;
using string_utility_t = entity_utility<std::unordered_map<std::string, int>, int>;

// ============================================================================
// Generic (non-string) specialization
// ============================================================================

TEST(entity_utility_generic, hit_after_miss_returns_same_value)
{
    int_utility_t util;
    util.emplace_entity(42, 7);

    // First lookup: cache miss, populates m_last_key/m_last_value.
    EXPECT_EQ(util.get_primary_key_value_for_entity(42), 7);
    // Second lookup: cache hit on the same key.
    EXPECT_EQ(util.get_primary_key_value_for_entity(42), 7);
}

TEST(entity_utility_generic, emplace_invalidates_cache_but_find_repopulates)
{
    int_utility_t util;
    util.emplace_entity(1, 100);

    EXPECT_EQ(util.get_primary_key_value_for_entity(1), 100);

    // Emplacing a different key must reset m_last_key; the next lookup
    // of key=1 must re-populate via find() rather than hand back stale data.
    util.emplace_entity(2, 200);

    EXPECT_EQ(util.get_primary_key_value_for_entity(1), 100);
    EXPECT_EQ(util.get_primary_key_value_for_entity(2), 200);
}

TEST(entity_utility_generic, cache_reset_after_emplace_seen_via_distinct_key)
{
    // unordered_map::emplace is a no-op when the key already exists, so we
    // cannot prove cache invalidation by re-emplacing the same key and
    // expecting a new value. Instead, prove that after an emplace of any
    // key, the cached slot has been reset by reading a *different* key
    // that was inserted after the previous lookup.
    int_utility_t util;
    util.emplace_entity(1, 10);

    EXPECT_EQ(util.get_primary_key_value_for_entity(1), 10);

    util.emplace_entity(2, 20);

    // If emplace_entity did not reset the cache, querying 2 here would
    // still hit the cache for key=1 and return 10. We expect 20.
    EXPECT_EQ(util.get_primary_key_value_for_entity(2), 20);
}

TEST(entity_utility_generic, missing_key_throws_runtime_error)
{
    int_utility_t util;
    util.emplace_entity(1, 10);

    EXPECT_THROW((void) util.get_primary_key_value_for_entity(999), std::runtime_error);
}

// ============================================================================
// std::string specialization
// ============================================================================

TEST(entity_utility_string, hit_after_miss_returns_same_value)
{
    string_utility_t util;
    util.emplace_entity(std::string{ "alpha" }, 1);

    EXPECT_EQ(util.get_primary_key_value_for_entity(std::string{ "alpha" }), 1);
    EXPECT_EQ(util.get_primary_key_value_for_entity(std::string{ "alpha" }), 1);
}

TEST(entity_utility_string, emplace_invalidates_cache_but_find_repopulates)
{
    string_utility_t util;
    util.emplace_entity(std::string{ "alpha" }, 1);

    EXPECT_EQ(util.get_primary_key_value_for_entity(std::string{ "alpha" }), 1);

    util.emplace_entity(std::string{ "beta" }, 2);

    EXPECT_EQ(util.get_primary_key_value_for_entity(std::string{ "alpha" }), 1);
    EXPECT_EQ(util.get_primary_key_value_for_entity(std::string{ "beta" }), 2);
}

TEST(entity_utility_string, cache_reset_after_emplace_seen_via_distinct_key)
{
    string_utility_t util;
    util.emplace_entity(std::string{ "alpha" }, 1);

    EXPECT_EQ(util.get_primary_key_value_for_entity(std::string{ "alpha" }), 1);

    util.emplace_entity(std::string{ "beta" }, 2);

    EXPECT_EQ(util.get_primary_key_value_for_entity(std::string{ "beta" }), 2);
}

TEST(entity_utility_string, string_view_overload_hits_cache)
{
    string_utility_t util;
    util.emplace_entity(std::string{ "gamma" }, 3);

    // Warm the cache via the string_view overload, then re-query the
    // same key via string_view; the second call hits the cache branch.
    EXPECT_EQ(util.get_primary_key_value_for_entity(std::string_view{ "gamma" }), 3);
    EXPECT_EQ(util.get_primary_key_value_for_entity(std::string_view{ "gamma" }), 3);
}

TEST(entity_utility_string, string_and_string_view_share_cache)
{
    string_utility_t util;
    util.emplace_entity(std::string{ "shared" }, 9);

    // Populate cache via const std::string& overload.
    EXPECT_EQ(util.get_primary_key_value_for_entity(std::string{ "shared" }), 9);
    // The string_view overload's cache check compares against m_last_key,
    // so the previously-cached entry should be reused.
    EXPECT_EQ(util.get_primary_key_value_for_entity(std::string_view{ "shared" }), 9);
}

TEST(entity_utility_string, missing_key_string_overload_throws_runtime_error)
{
    string_utility_t util;
    util.emplace_entity(std::string{ "present" }, 1);

    EXPECT_THROW((void) util.get_primary_key_value_for_entity(std::string{ "absent" }),
                 std::runtime_error);
}

TEST(entity_utility_string, missing_key_string_view_overload_throws_runtime_error)
{
    string_utility_t util;
    util.emplace_entity(std::string{ "present" }, 1);

    EXPECT_THROW(
        (void) util.get_primary_key_value_for_entity(std::string_view{ "absent" }),
        std::runtime_error);
}

}  // namespace
