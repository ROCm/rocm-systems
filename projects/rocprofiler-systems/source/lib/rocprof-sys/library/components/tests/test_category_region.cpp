// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Unit tests for the trace-cache argument serialization / caching helpers added to
// category_region.hpp

#include "rocprof-sys/library/components/category_region.hpp"

#include "core/categories.hpp"
#include "core/common_types.hpp"

#include <cstdint>
#include <gtest/gtest.h>
#include <string>
#include <tuple>
#include <utility>

namespace
{
using rocprofsys::argument_info;
using rocprofsys::function_args_t;
using rocprofsys::process_arguments_string;

// Round-trip the serialized wire string back into structured records so assertions
// do not depend on the exact delimiter layout.
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
}

// ---------------------------------------------------------------------------------------
// append_serialized_arg (both overloads)
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

TEST(category_region_serialization, append_serialized_arg_prestringified)
{
    std::string serialized;
    append_serialized_arg(serialized, 3, "my_type", "beta", "value");

    auto args = parse(serialized);
    ASSERT_EQ(args.size(), 1u);
    EXPECT_EQ(args[0].arg_number, 3u);
    EXPECT_EQ(args[0].arg_type, "my_type");
    EXPECT_EQ(args[0].arg_name, "beta");
    EXPECT_EQ(args[0].arg_value, "value");
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

// ---------------------------------------------------------------------------------------
// count_serialized_args / renumber_serialized_args
// ---------------------------------------------------------------------------------------

TEST(category_region_serialization, count_serialized_args)
{
    EXPECT_EQ(count_serialized_args(""), 0u);
    EXPECT_EQ(count_serialized_args(serialize_name_value_pairs("a", 1)), 1u);
    EXPECT_EQ(count_serialized_args(serialize_name_value_pairs("a", 1, "b", 2, "c", 3)),
              3u);
}

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
// entry_key ordering (used as the std::map key for the pending-entry stacks)
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
// cache_start (pushes a pending entry onto the per-thread stack)
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
    EXPECT_EQ(count_serialized_args(entry.args), 2u);
    EXPECT_GT(entry.start_ts, 0u);

    auto args = parse(entry.args);
    ASSERT_EQ(args.size(), 2u);
    EXPECT_EQ(args[0].arg_name, "a");
    EXPECT_EQ(args[1].arg_name, "b");

    // a second push for the same key stacks a new (nested) pending entry; the argless
    // overload records zero args
    cache_start<category_t>(name);
    ASSERT_EQ(map_name_to_args[key].size(), 2u);
    EXPECT_EQ(count_serialized_args(map_name_to_args[key].back().args), 0u);
    EXPECT_TRUE(map_name_to_args[key].back().args.empty());

    map_name_to_args.clear();
}

// ---------------------------------------------------------------------------------------
// append_cache_args (free function operating on the per-thread pending-entry stack)
// ---------------------------------------------------------------------------------------

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
    EXPECT_EQ(count_serialized_args(entry.args), 3u);

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
    EXPECT_EQ(count_serialized_args(map_name_to_args[key].back().args), 0u);

    map_name_to_args.clear();
}
