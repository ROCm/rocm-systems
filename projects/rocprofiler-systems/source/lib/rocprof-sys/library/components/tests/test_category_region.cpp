// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Unit tests for the trace-cache argument serialization / caching helpers added to
// category_region.hpp

#include "rocprof-sys/library/components/category_region.hpp"

#include "core/categories.hpp"
#include "core/common_types.hpp"

#include <cstdint>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <memory>
#include <optional>
#include <ostream>
#include <string>
#include <tuple>
#include <utility>

namespace
{
using rocprofsys::argument_info;
using rocprofsys::function_args_t;
using rocprofsys::process_arguments_string;

using region_cache = rocprofsys::utility::category_region<>;
using rocprofsys::utility::entry_key;
using rocprofsys::utility::pending_cache_entry;

auto& map_name_to_args = region_cache::instance().pending_entries();

// ---------------------------------------------------------------------------------------
// test shims
// ---------------------------------------------------------------------------------------

template <typename Tp>
std::string
get_serialized_arg_type()
{
    return region_cache::get_serialized_arg_type<Tp>();
}

template <typename Tp>
std::string
get_serialized_arg_value(Tp&& value)
{
    return region_cache::get_serialized_arg_value(std::forward<Tp>(value));
}

template <typename... Args>
void
append_serialized_arg(Args&&... args)
{
    region_cache::append_serialized_arg(std::forward<Args>(args)...);
}

template <typename... Args>
inline constexpr bool has_trace_cache_arg_pairs_v =
    region_cache::has_trace_cache_arg_pairs_v<Args...>;

template <typename... Args>
std::string
serialize_name_value_pairs(Args&&... args)
{
    return region_cache::serialize_name_value_pairs(std::forward<Args>(args)...);
}

inline std::uint32_t
renumber_serialized_args(std::string& args_str, std::uint32_t next_idx)
{
    return region_cache::renumber_serialized_args(args_str, next_idx);
}

inline std::optional<std::uint32_t>
next_arg_index(const std::string& args_str)
{
    return region_cache::next_arg_index(args_str);
}

template <typename... Args>
std::string
serialize_annotation_args(Args&&... args)
{
    return region_cache::serialize_annotation_args(std::forward<Args>(args)...);
}

template <typename T>
std::string
serialize_return_arg(T&& value)
{
    return region_cache::serialize_return_arg(std::forward<T>(value));
}

template <typename CategoryT, typename... Args>
void
cache_start(const char* name, Args&&... args)
{
    region_cache::instance().cache_start(name, rocprofsys::trait::name<CategoryT>::value,
                                         std::forward<Args>(args)...);
}

template <typename CategoryT>
void
append_cache_args(const char* name, std::string args_str)
{
    region_cache::instance().append_cache_args(
        name, rocprofsys::trait::name<CategoryT>::value, std::move(args_str));
}

// Round-trip the serialized wire string back into structured records so assertions
// do not depend on the exact delimiter layout
function_args_t
parse(const std::string& serialized)
{
    return process_arguments_string(serialized);
}

bool
starts_with(const std::string& value, const std::string& prefix)
{
    return value.rfind(prefix, 0) == 0;
}

// A type that is ostream-streamable but has no fmt formatter, used to exercise the
// fmt::streamed fallback branch in get_serialized_arg_value.
struct streamable_only
{
    int value;

    friend std::ostream& operator<<(std::ostream& os, const streamable_only& self)
    {
        return os << "S(" << self.value << ")";
    }
};
}  // namespace

// ---------------------------------------------------------------------------------------
// get_serialized_arg_type / get_serialized_arg_value
// ---------------------------------------------------------------------------------------

TEST(category_region_serialization, get_serialized_arg_type)
{
    EXPECT_EQ(get_serialized_arg_type<const char*>(), "string");
    EXPECT_EQ(get_serialized_arg_type<char*>(), "string");
    EXPECT_EQ(get_serialized_arg_type<std::string>(), "string");

    // non-string-like types resolve to a demangled, non-empty type that is not "string"
    const auto int_type = get_serialized_arg_type<int>();
    EXPECT_FALSE(int_type.empty());
    EXPECT_NE(int_type, "string");
}

TEST(category_region_serialization, get_serialized_arg_value)
{
    EXPECT_EQ(get_serialized_arg_value(42), "42");
    EXPECT_EQ(get_serialized_arg_value(-7), "-7");
    EXPECT_EQ(get_serialized_arg_value(std::string{ "hello" }), "hello");

    // types without an fmt formatter fall back to fmt::streamed (operator<<)
    EXPECT_EQ(get_serialized_arg_value(streamable_only{ 7 }), "S(7)");
}

// ---------------------------------------------------------------------------------------
// append_serialized_arg
// ---------------------------------------------------------------------------------------

TEST(category_region_serialization, append_serialized_arg_typed)
{
    std::string serialized;
    append_serialized_arg(serialized, 0, "alpha", 7);

    auto args = parse(serialized);
    ASSERT_EQ(args.size(), 1u);
    EXPECT_EQ(args[0].arg_number, 0u);
    EXPECT_EQ(args[0].arg_name, "alpha");
    EXPECT_EQ(args[0].arg_value, "7");
    EXPECT_FALSE(args[0].arg_type.empty());
}

// ---------------------------------------------------------------------------------------
// has_trace_cache_arg_pairs
// ---------------------------------------------------------------------------------------

TEST(category_region_serialization, has_trace_cache_arg_pairs)
{
    // even count with string-like name slots -> true
    EXPECT_TRUE((has_trace_cache_arg_pairs_v<const char*, int>) );
    EXPECT_TRUE((has_trace_cache_arg_pairs_v<const char*, int, std::string, double>) );

    // empty -> false
    EXPECT_FALSE((has_trace_cache_arg_pairs_v<>) );

    // odd count -> false
    EXPECT_FALSE((has_trace_cache_arg_pairs_v<const char*, int, const char*>) );

    // even count but a non-string name slot -> false
    EXPECT_FALSE((has_trace_cache_arg_pairs_v<int, int>) );
}

// ---------------------------------------------------------------------------------------
// serialize_name_value_pairs
// ---------------------------------------------------------------------------------------

TEST(category_region_serialization, serialize_name_value_pairs_valid)
{
    auto args = parse(serialize_name_value_pairs("alpha", 1, "beta", 2, "gamma", 3));
    ASSERT_EQ(args.size(), 3u);

    EXPECT_EQ(args[0].arg_number, 0u);
    EXPECT_EQ(args[0].arg_name, "alpha");
    EXPECT_EQ(args[0].arg_value, "1");

    EXPECT_EQ(args[1].arg_number, 1u);
    EXPECT_EQ(args[1].arg_name, "beta");
    EXPECT_EQ(args[1].arg_value, "2");

    EXPECT_EQ(args[2].arg_number, 2u);
    EXPECT_EQ(args[2].arg_name, "gamma");
    EXPECT_EQ(args[2].arg_value, "3");
}

TEST(category_region_serialization, serialize_name_value_pairs_invalid_returns_empty)
{
    EXPECT_TRUE(serialize_name_value_pairs().empty());        // no args
    EXPECT_TRUE(serialize_name_value_pairs("only").empty());  // odd count
    EXPECT_TRUE(serialize_name_value_pairs(1, 2).empty());    // non-string name slot
}

// An argument value that itself contains the field delimiter ";;" must not corrupt
// the wire format: it should round-trip back as a single record whose value is
// preserved verbatim.
TEST(category_region_serialization, serialize_name_value_pairs_value_with_delimiter)
{
    auto args = parse(serialize_name_value_pairs("path", "a;;b"));
    ASSERT_EQ(args.size(), 1u);
    EXPECT_EQ(args[0].arg_number, 0u);
    EXPECT_EQ(args[0].arg_name, "path");
    EXPECT_EQ(args[0].arg_value, "a;;b");
}

// The escape character itself ('%') and a bare ';' must also survive the round-trip so
// the encoding is lossless, not merely delimiter-safe.
TEST(category_region_serialization, serialize_name_value_pairs_value_with_escape_chars)
{
    auto args = parse(serialize_name_value_pairs("a", std::string{ "50%;done" }, "b",
                                                 std::string{ "x%3By" }));
    ASSERT_EQ(args.size(), 2u);
    EXPECT_EQ(args[0].arg_name, "a");
    EXPECT_EQ(args[0].arg_value, "50%;done");
    EXPECT_EQ(args[1].arg_name, "b");
    // a value that already looks like an escape sequence must not be double-decoded
    EXPECT_EQ(args[1].arg_value, "x%3By");
}

// ---------------------------------------------------------------------------------------
// renumber_serialized_args
// ---------------------------------------------------------------------------------------

TEST(category_region_serialization, renumber_serialized_args)
{
    auto serialized = serialize_name_value_pairs("a", 1, "b", 2);

    const auto renumbered = renumber_serialized_args(serialized, 5);
    EXPECT_EQ(renumbered, 2u);

    auto args = parse(serialized);
    ASSERT_EQ(args.size(), 2u);
    EXPECT_EQ(args[0].arg_number, 5u);
    EXPECT_EQ(args[1].arg_number, 6u);
    // names/values are preserved across renumbering
    EXPECT_EQ(args[0].arg_name, "a");
    EXPECT_EQ(args[0].arg_value, "1");
    EXPECT_EQ(args[1].arg_name, "b");
    EXPECT_EQ(args[1].arg_value, "2");
}

TEST(category_region_serialization, renumber_serialized_args_empty)
{
    std::string empty;
    EXPECT_EQ(renumber_serialized_args(empty, 5), 0u);
    EXPECT_TRUE(empty.empty());
}

// ---------------------------------------------------------------------------------------
// next_arg_index
// ---------------------------------------------------------------------------------------

TEST(category_region_serialization, next_arg_index)
{
    // empty -> 0
    EXPECT_EQ(next_arg_index(""), 0u);
    // records numbered contiguously from 0 -> last idx + 1 == record count
    EXPECT_EQ(next_arg_index(serialize_name_value_pairs("a", 1)), 1u);
    EXPECT_EQ(next_arg_index(serialize_name_value_pairs("a", 1, "b", 2)), 2u);
    EXPECT_EQ(next_arg_index(serialize_name_value_pairs("a", 1, "b", 2, "c", 3)), 3u);

    // reads the last record's idx field, not the record count: a string already
    // renumbered to a non-zero base returns last idx + 1
    auto renumbered = serialize_name_value_pairs("a", 1, "b", 2);
    renumber_serialized_args(renumbered, 5);  // -> indices 5, 6
    EXPECT_EQ(next_arg_index(renumbered), 7u);
}

TEST(category_region_serialization, next_arg_index_malformed_returns_nullopt)
{
    // the leading idx field of the last record is not a number -> cannot continue
    // numbering, so the index is reported as absent rather than guessed
    EXPECT_FALSE(next_arg_index("notanumber;;string;;name;;value;;").has_value());
}

// ---------------------------------------------------------------------------------------
// serialize_annotation_args (variadic gotcha-audit overload)
// ---------------------------------------------------------------------------------------

TEST(category_region_serialization, serialize_annotation_args_variadic)
{
    auto args = parse(serialize_annotation_args(42, std::string{ "hello" }));
    ASSERT_EQ(args.size(), 2u);

    EXPECT_EQ(args[0].arg_number, 0u);
    EXPECT_TRUE(starts_with(args[0].arg_name, "arg0-"));
    EXPECT_EQ(args[0].arg_value, "42");

    EXPECT_EQ(args[1].arg_number, 1u);
    EXPECT_TRUE(starts_with(args[1].arg_name, "arg1-"));
    EXPECT_EQ(args[1].arg_type, "string");
    EXPECT_EQ(args[1].arg_value, "hello");
}

TEST(category_region_serialization, serialize_annotation_args_empty)
{
    EXPECT_TRUE(serialize_annotation_args().empty());
}

// ---------------------------------------------------------------------------------------
// serialize_return_arg
// ---------------------------------------------------------------------------------------

TEST(category_region_serialization, serialize_return_arg)
{
    auto args = parse(serialize_return_arg(0));
    ASSERT_EQ(args.size(), 1u);
    EXPECT_EQ(args[0].arg_number, 0u);
    EXPECT_EQ(args[0].arg_name, "return");
    EXPECT_EQ(args[0].arg_value, "0");
}

// ---------------------------------------------------------------------------------------
// entry_key ordering
// ---------------------------------------------------------------------------------------

TEST(category_region_cache, entry_key_ordering)
{
    entry_key a{ "aaa", "cat" };
    entry_key b{ "bbb", "cat" };  // differs by name
    entry_key c{ "aaa", "dog" };  // same name, differs by category

    // different names are ordered by name
    EXPECT_TRUE(a < b);
    EXPECT_FALSE(b < a);

    // equal names fall back to ordering by category
    EXPECT_TRUE(a < c);
    EXPECT_FALSE(c < a);

    // fully equal keys: neither precedes the other
    entry_key a_copy{ "aaa", "cat" };
    EXPECT_FALSE(a < a_copy);
    EXPECT_FALSE(a_copy < a);
}

// ---------------------------------------------------------------------------------------
// cache_start
// ---------------------------------------------------------------------------------------

TEST(category_region_cache, cache_start_pushes_pending_entry)
{
    using category_t = rocprofsys::category::host;
    const char* name = "start_region";
    entry_key   key{ name, rocprofsys::trait::name<category_t>::value };

    map_name_to_args.clear();
    cache_start<category_t>(name, serialize_name_value_pairs("a", 1, "b", 2));

    auto itr = map_name_to_args.find(key);
    ASSERT_TRUE(itr != map_name_to_args.end());
    ASSERT_EQ(itr->second.size(), 1u);

    const auto& entry = itr->second.back();
    EXPECT_EQ(parse(entry.args).size(), 2u);
    EXPECT_GT(entry.start_ts, 0u);

    auto args = parse(entry.args);
    ASSERT_EQ(args.size(), 2u);
    EXPECT_EQ(args[0].arg_name, "a");
    EXPECT_EQ(args[1].arg_name, "b");

    // a second push for the same key stacks a new (nested) pending entry; the argless
    // overload records zero args
    cache_start<category_t>(name);
    ASSERT_EQ(map_name_to_args[key].size(), 2u);
    EXPECT_EQ(parse(map_name_to_args[key].back().args).size(), 0u);
    EXPECT_TRUE(map_name_to_args[key].back().args.empty());

    map_name_to_args.clear();
}

// ---------------------------------------------------------------------------------------
// append_cache_args
// ---------------------------------------------------------------------------------------

TEST(category_region_cache, append_cache_args_adopts_first_batch_without_renumbering)
{
    using category_t = rocprofsys::category::host;
    const char* name = "adopt_region";
    entry_key   key{ name, rocprofsys::trait::name<category_t>::value };

    map_name_to_args.clear();
    // open entry that has no args yet (e.g. created by an argless start)
    map_name_to_args[key].push_back(pending_cache_entry{ 0, {} });

    // the first batch into an empty entry is adopted verbatim (hot path): its 0-based
    // numbering is preserved and renumber_serialized_args is skipped
    append_cache_args<category_t>(name, serialize_name_value_pairs("a", 1, "b", 2));

    const auto& entry = map_name_to_args[key].back();
    auto        args  = parse(entry.args);
    ASSERT_EQ(args.size(), 2u);
    EXPECT_EQ(args[0].arg_number, 0u);
    EXPECT_EQ(args[0].arg_name, "a");
    EXPECT_EQ(args[1].arg_number, 1u);
    EXPECT_EQ(args[1].arg_name, "b");

    map_name_to_args.clear();
}

TEST(category_region_cache, append_cache_args_appends_and_renumbers)
{
    using category_t = rocprofsys::category::host;
    const char* name = "append_region";
    entry_key   key{ name, rocprofsys::trait::name<category_t>::value };

    map_name_to_args.clear();
    // seed an open entry with one already-serialized arg (numbered 0)
    map_name_to_args[key].push_back(
        pending_cache_entry{ 0, serialize_name_value_pairs("a", 1) });

    // append two more args; their local numbering (0,1) must continue from 1 -> (1,2)
    append_cache_args<category_t>(name, serialize_name_value_pairs("b", 2, "c", 3));

    ASSERT_FALSE(map_name_to_args[key].empty());
    const auto& entry = map_name_to_args[key].back();
    EXPECT_EQ(parse(entry.args).size(), 3u);

    auto args = parse(entry.args);
    ASSERT_EQ(args.size(), 3u);
    EXPECT_EQ(args[0].arg_number, 0u);
    EXPECT_EQ(args[0].arg_name, "a");
    EXPECT_EQ(args[1].arg_number, 1u);
    EXPECT_EQ(args[1].arg_name, "b");
    EXPECT_EQ(args[2].arg_number, 2u);
    EXPECT_EQ(args[2].arg_name, "c");

    map_name_to_args.clear();
}

TEST(category_region_cache, append_cache_args_noop_without_open_entry)
{
    using category_t = rocprofsys::category::host;
    const char* name = "missing_region";
    entry_key   key{ name, rocprofsys::trait::name<category_t>::value };

    map_name_to_args.clear();
    // no open entry -> append is a no-op and must not create one
    append_cache_args<category_t>(name, serialize_name_value_pairs("x", 1));
    EXPECT_TRUE(map_name_to_args.find(key) == map_name_to_args.end());

    // empty args -> no-op even when an entry exists
    map_name_to_args[key].push_back(pending_cache_entry{ 0, {} });
    append_cache_args<category_t>(name, std::string{});
    EXPECT_TRUE(map_name_to_args[key].back().args.empty());
    EXPECT_EQ(parse(map_name_to_args[key].back().args).size(), 0u);

    map_name_to_args.clear();
}

TEST(category_region_cache, append_cache_args_drops_batch_when_existing_args_malformed)
{
    using category_t = rocprofsys::category::host;
    const char* name = "malformed_region";
    entry_key   key{ name, rocprofsys::trait::name<category_t>::value };

    map_name_to_args.clear();
    // open entry whose existing args have a non-numeric leading idx field: the next
    // index cannot be determined, so a subsequent append must be dropped rather than
    // produce colliding indices.
    const std::string malformed = "bad;;string;;x;;1;;";
    map_name_to_args[key].push_back(pending_cache_entry{ 0, malformed });

    append_cache_args<category_t>(name, serialize_name_value_pairs("b", 2));

    // the existing (malformed) args are left untouched and the new batch is not appended
    EXPECT_EQ(map_name_to_args[key].back().args, malformed);

    map_name_to_args.clear();
}

TEST(category_region_cache, cache_start_keys_on_name_and_category)
{
    const char* name = "shared_region";

    map_name_to_args.clear();
    // identical region name pushed under two different categories
    region_cache::instance().cache_start(name, "cat_a",
                                         serialize_name_value_pairs("a", 1));
    region_cache::instance().cache_start(name, "cat_b",
                                         serialize_name_value_pairs("b", 2));

    const entry_key key_a{ name, "cat_a" };
    const entry_key key_b{ name, "cat_b" };

    auto itr_a = map_name_to_args.find(key_a);
    auto itr_b = map_name_to_args.find(key_b);
    ASSERT_TRUE(itr_a != map_name_to_args.end());
    ASSERT_TRUE(itr_b != map_name_to_args.end());

    // the two categories own independent (non-merged) pending stacks
    ASSERT_EQ(itr_a->second.size(), 1u);
    ASSERT_EQ(itr_b->second.size(), 1u);

    auto args_a = parse(itr_a->second.back().args);
    auto args_b = parse(itr_b->second.back().args);
    ASSERT_EQ(args_a.size(), 1u);
    ASSERT_EQ(args_b.size(), 1u);
    EXPECT_EQ(args_a[0].arg_name, "a");
    EXPECT_EQ(args_b[0].arg_name, "b");

    map_name_to_args.clear();
}

TEST(category_region_cache, append_cache_args_is_scoped_to_category)
{
    const char* name = "shared_region";

    map_name_to_args.clear();
    // only cat_a has an open entry
    region_cache::instance().cache_start(name, "cat_a",
                                         serialize_name_value_pairs("a", 1));

    // appending under the same name but a different category must not touch cat_a and
    // must not fabricate an entry for cat_b
    region_cache::instance().append_cache_args(name, "cat_b",
                                               serialize_name_value_pairs("b", 2));

    const entry_key key_a{ name, "cat_a" };
    const entry_key key_b{ name, "cat_b" };
    EXPECT_TRUE(map_name_to_args.find(key_b) == map_name_to_args.end());

    auto itr_a = map_name_to_args.find(key_a);
    ASSERT_TRUE(itr_a != map_name_to_args.end());
    EXPECT_EQ(parse(itr_a->second.back().args).size(), 1u);

    // appending under the matching category extends that category's open entry
    region_cache::instance().append_cache_args(
        name, "cat_a", serialize_name_value_pairs("b", 2, "c", 3));
    const auto& entry = map_name_to_args[key_a].back();
    EXPECT_EQ(parse(entry.args).size(), 3u);

    auto args = parse(entry.args);
    ASSERT_EQ(args.size(), 3u);
    EXPECT_EQ(args[0].arg_number, 0u);
    EXPECT_EQ(args[0].arg_name, "a");
    EXPECT_EQ(args[1].arg_number, 1u);
    EXPECT_EQ(args[1].arg_name, "b");
    EXPECT_EQ(args[2].arg_number, 2u);
    EXPECT_EQ(args[2].arg_name, "c");

    map_name_to_args.clear();
}

// =======================================================================================
// Policy-injected dependency tests
// =======================================================================================

struct gmock_clock
{
    MOCK_METHOD(rocprofsys::utility::timestamp_t, now, (), (const));
};

struct gmock_region_sink
{
    MOCK_METHOD(void, store_region,
                (std::uint64_t thread_id, std::string name, std::uint64_t start_ts,
                 std::uint64_t end_ts, std::string category, std::string args),
                (const));
};

struct gmock_thread_metadata
{
    MOCK_METHOD(std::uint64_t, resolve_current_thread, (), (const));
};

namespace test_globals
{
std::unique_ptr<gmock_clock>           g_clock_gmock;
std::unique_ptr<gmock_region_sink>     g_region_sink_gmock;
std::unique_ptr<gmock_thread_metadata> g_thread_meta_gmock;
}  // namespace test_globals

namespace
{
struct mock_clock_source
{
    rocprofsys::utility::timestamp_t now() const
    {
        return test_globals::g_clock_gmock->now();
    }
};

struct mock_region_sink
{
    void store_region(std::uint64_t thread_id, const char* name, std::uint64_t start_ts,
                      std::uint64_t end_ts, const char* category, const char* args) const
    {
        test_globals::g_region_sink_gmock->store_region(thread_id, name, start_ts, end_ts,
                                                        category, args);
    }
};

struct mock_thread_metadata_source
{
    std::uint64_t resolve_current_thread() const
    {
        return test_globals::g_thread_meta_gmock->resolve_current_thread();
    }
};

struct mock_region_policy
{
    using clock_type           = mock_clock_source;
    using region_sink_type     = mock_region_sink;
    using thread_metadata_type = mock_thread_metadata_source;
};

using mocked_region_cache_t = rocprofsys::utility::category_region<mock_region_policy>;
}  // namespace

class category_region_policy_test : public ::testing::Test
{
protected:
    void SetUp() override
    {
        test_globals::g_clock_gmock       = std::make_unique<gmock_clock>();
        test_globals::g_region_sink_gmock = std::make_unique<gmock_region_sink>();
        test_globals::g_thread_meta_gmock = std::make_unique<gmock_thread_metadata>();
    }

    void TearDown() override
    {
        test_globals::g_clock_gmock.reset();
        test_globals::g_region_sink_gmock.reset();
        test_globals::g_thread_meta_gmock.reset();
    }
};

// ---------------------------------------------------------------------------------------
// policy-injected: cache_start
// ---------------------------------------------------------------------------------------

TEST_F(category_region_policy_test, cache_start_records_injected_clock_without_emitting)
{
    using ::testing::_;
    using ::testing::Return;

    mocked_region_cache_t region;

    EXPECT_CALL(*test_globals::g_clock_gmock, now()).Times(1).WillOnce(Return(1234u));
    // a push must not resolve the thread or emit a region
    EXPECT_CALL(*test_globals::g_thread_meta_gmock, resolve_current_thread()).Times(0);
    EXPECT_CALL(*test_globals::g_region_sink_gmock, store_region(_, _, _, _, _, _))
        .Times(0);

    region.cache_start("region", "cat", serialize_name_value_pairs("a", 1));

    const entry_key key{ "region", "cat" };
    auto            itr = region.pending_entries().find(key);
    ASSERT_TRUE(itr != region.pending_entries().end());
    ASSERT_EQ(itr->second.size(), 1u);
    EXPECT_EQ(itr->second.back().start_ts, 1234u);
}

// ---------------------------------------------------------------------------------------
// policy-injected: cache_stop
// ---------------------------------------------------------------------------------------

TEST_F(category_region_policy_test, cache_stop_emits_region_through_injected_seams)
{
    using ::testing::_;
    using ::testing::Return;

    mocked_region_cache_t region;

    // start consumes the first now() (start_ts), stop consumes the second (end_ts)
    EXPECT_CALL(*test_globals::g_clock_gmock, now())
        .WillOnce(Return(100u))
        .WillOnce(Return(500u));
    EXPECT_CALL(*test_globals::g_thread_meta_gmock, resolve_current_thread())
        .Times(1)
        .WillOnce(Return(42u));
    EXPECT_CALL(
        *test_globals::g_region_sink_gmock,
        store_region(42u, std::string{ "region" }, 100u, 500u, std::string{ "cat" }, _))
        .Times(1);

    region.cache_start("region", "cat", serialize_name_value_pairs("a", 1));
    region.cache_stop("region", "cat");

    // the pending entry is consumed by the stop
    EXPECT_TRUE(region.pending_entries().empty());
}

TEST_F(category_region_policy_test, cache_stop_without_open_entry_touches_no_seams)
{
    using ::testing::_;

    mocked_region_cache_t region;

    EXPECT_CALL(*test_globals::g_clock_gmock, now()).Times(0);
    EXPECT_CALL(*test_globals::g_thread_meta_gmock, resolve_current_thread()).Times(0);
    EXPECT_CALL(*test_globals::g_region_sink_gmock, store_region(_, _, _, _, _, _))
        .Times(0);

    region.cache_stop("missing", "cat");
}

TEST_F(category_region_policy_test, cache_stop_pops_only_the_innermost_frame)
{
    using ::testing::_;
    using ::testing::Return;

    mocked_region_cache_t region;

    // two pushes on the same key (outer start_ts 100, inner start_ts 200); a single stop
    // closes the inner frame (end_ts 300) and must leave the outer frame open.
    EXPECT_CALL(*test_globals::g_clock_gmock, now())
        .WillOnce(Return(100u))   // outer start
        .WillOnce(Return(200u))   // inner start
        .WillOnce(Return(300u));  // stop end_ts
    EXPECT_CALL(*test_globals::g_thread_meta_gmock, resolve_current_thread())
        .Times(1)
        .WillOnce(Return(9u));
    // the popped (innermost) frame is the one emitted: its start_ts 200, not the outer
    // 100
    EXPECT_CALL(
        *test_globals::g_region_sink_gmock,
        store_region(9u, std::string{ "region" }, 200u, 300u, std::string{ "cat" }, _))
        .Times(1);

    region.cache_start("region", "cat", serialize_name_value_pairs("outer", 1));
    region.cache_start("region", "cat", serialize_name_value_pairs("inner", 2));
    region.cache_stop("region", "cat");

    // the outer frame remains open with its original timestamp
    const entry_key key{ "region", "cat" };
    auto            itr = region.pending_entries().find(key);
    ASSERT_TRUE(itr != region.pending_entries().end());
    ASSERT_EQ(itr->second.size(), 1u);
    EXPECT_EQ(itr->second.back().start_ts, 100u);
}

TEST_F(category_region_policy_test, cache_stop_emits_zero_thread_id_when_unresolved)
{
    using ::testing::_;
    using ::testing::Return;

    mocked_region_cache_t region;

    EXPECT_CALL(*test_globals::g_clock_gmock, now())
        .WillOnce(Return(10u))
        .WillOnce(Return(20u));
    // no thread info available -> resolve returns 0, which flows through to the sink
    EXPECT_CALL(*test_globals::g_thread_meta_gmock, resolve_current_thread())
        .Times(1)
        .WillOnce(Return(0u));
    EXPECT_CALL(
        *test_globals::g_region_sink_gmock,
        store_region(0u, std::string{ "region" }, 10u, 20u, std::string{ "cat" }, _))
        .Times(1);

    region.cache_start("region", "cat");
    region.cache_stop("region", "cat");
}

TEST_F(category_region_policy_test, cache_stop_forwards_serialized_args_to_sink)
{
    using ::testing::_;
    using ::testing::Return;

    mocked_region_cache_t region;

    const auto args = serialize_name_value_pairs("a", 1, "b", 2);

    EXPECT_CALL(*test_globals::g_clock_gmock, now())
        .WillOnce(Return(10u))
        .WillOnce(Return(20u));
    EXPECT_CALL(*test_globals::g_thread_meta_gmock, resolve_current_thread())
        .Times(1)
        .WillOnce(Return(3u));
    // the args serialized at start flow verbatim into the sink at stop
    EXPECT_CALL(
        *test_globals::g_region_sink_gmock,
        store_region(3u, std::string{ "region" }, 10u, 20u, std::string{ "cat" }, args))
        .Times(1);

    region.cache_start("region", "cat", args);
    region.cache_stop("region", "cat");
}

// ---------------------------------------------------------------------------------------
// policy-injected: append_cache_args
// ---------------------------------------------------------------------------------------

TEST_F(category_region_policy_test, append_cache_args_touches_no_seams)
{
    using ::testing::_;

    mocked_region_cache_t region;

    EXPECT_CALL(*test_globals::g_clock_gmock, now()).Times(0);
    EXPECT_CALL(*test_globals::g_thread_meta_gmock, resolve_current_thread()).Times(0);
    EXPECT_CALL(*test_globals::g_region_sink_gmock, store_region(_, _, _, _, _, _))
        .Times(0);

    const entry_key key{ "region", "cat" };
    region.pending_entries()[key].push_back(pending_cache_entry{ 0, {} });
    region.append_cache_args("region", "cat", serialize_name_value_pairs("a", 1));

    EXPECT_EQ(parse(region.pending_entries()[key].back().args).size(), 1u);
}

// ---------------------------------------------------------------------------------------
// policy-injected: flush_pending_cached_entries
// ---------------------------------------------------------------------------------------

TEST_F(category_region_policy_test, flush_emits_every_pending_entry_then_clears)
{
    using ::testing::_;
    using ::testing::Return;

    mocked_region_cache_t region;

    // two starts (start_ts 10 and 20) then a single flush end_ts (999)
    EXPECT_CALL(*test_globals::g_clock_gmock, now())
        .WillOnce(Return(10u))
        .WillOnce(Return(20u))
        .WillOnce(Return(999u));
    EXPECT_CALL(*test_globals::g_thread_meta_gmock, resolve_current_thread())
        .Times(1)
        .WillOnce(Return(7u));
    EXPECT_CALL(*test_globals::g_region_sink_gmock,
                store_region(7u, std::string{ "r1" }, 10u, 999u, std::string{ "cat" }, _))
        .Times(1);
    EXPECT_CALL(*test_globals::g_region_sink_gmock,
                store_region(7u, std::string{ "r2" }, 20u, 999u, std::string{ "cat" }, _))
        .Times(1);

    region.cache_start("r1", "cat");
    region.cache_start("r2", "cat");
    region.flush_pending_cached_entries();

    EXPECT_TRUE(region.pending_entries().empty());
}

TEST_F(category_region_policy_test, flush_emits_one_region_per_outstanding_frame)
{
    using ::testing::_;
    using ::testing::Return;

    mocked_region_cache_t region;

    // three nested pushes on one key that are never popped (start_ts 1, 2, 3); flush
    // emits one region per outstanding frame, all sharing the single flush end_ts (100).
    EXPECT_CALL(*test_globals::g_clock_gmock, now())
        .WillOnce(Return(1u))
        .WillOnce(Return(2u))
        .WillOnce(Return(3u))
        .WillOnce(Return(100u));
    EXPECT_CALL(*test_globals::g_thread_meta_gmock, resolve_current_thread())
        .Times(1)
        .WillOnce(Return(5u));
    EXPECT_CALL(*test_globals::g_region_sink_gmock,
                store_region(5u, std::string{ "rec" }, 1u, 100u, std::string{ "cat" }, _))
        .Times(1);
    EXPECT_CALL(*test_globals::g_region_sink_gmock,
                store_region(5u, std::string{ "rec" }, 2u, 100u, std::string{ "cat" }, _))
        .Times(1);
    EXPECT_CALL(*test_globals::g_region_sink_gmock,
                store_region(5u, std::string{ "rec" }, 3u, 100u, std::string{ "cat" }, _))
        .Times(1);

    region.cache_start("rec", "cat");
    region.cache_start("rec", "cat");
    region.cache_start("rec", "cat");
    region.flush_pending_cached_entries();

    EXPECT_TRUE(region.pending_entries().empty());
}

TEST_F(category_region_policy_test, flush_on_empty_map_emits_nothing)
{
    using ::testing::_;
    using ::testing::Return;

    mocked_region_cache_t region;

    // flush unconditionally samples the clock and resolves the thread, but with no
    // pending entries it must not emit any region.
    EXPECT_CALL(*test_globals::g_clock_gmock, now()).Times(1).WillOnce(Return(50u));
    EXPECT_CALL(*test_globals::g_thread_meta_gmock, resolve_current_thread())
        .Times(1)
        .WillOnce(Return(1u));
    EXPECT_CALL(*test_globals::g_region_sink_gmock, store_region(_, _, _, _, _, _))
        .Times(0);

    region.flush_pending_cached_entries();

    EXPECT_TRUE(region.pending_entries().empty());
}
