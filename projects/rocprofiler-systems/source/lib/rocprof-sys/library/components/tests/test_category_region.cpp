// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Unit tests for the trace-cache argument serialization / caching helpers added to
// category_region.hpp

#include "rocprof-sys/library/components/category_region.hpp"

#include "core/categories.hpp"
#include "core/common_types.hpp"

#include <cstdint>
#include <gtest/gtest.h>
#include <ostream>
#include <string>
#include <tuple>
#include <utility>

namespace
{
using rocprofsys::argument_info;
using rocprofsys::function_args_t;
using rocprofsys::process_arguments_string;

using base = rocprofsys::component::category_region_base;
using rocprofsys::component::entry_key;
using rocprofsys::component::map_name_to_args;
using rocprofsys::component::pending_cache_entry;

// The trace-cache helpers now live as static members of category_region_base. These thin
// forwarding shims keep the unqualified test bodies (and the <category_t> call syntax for
// the cache lifecycle helpers) unchanged.
template <typename Tp>
std::string
get_serialized_arg_type()
{
    return base::get_serialized_arg_type<Tp>();
}

template <typename Tp>
std::string
get_serialized_arg_value(Tp&& value)
{
    return base::get_serialized_arg_value(std::forward<Tp>(value));
}

template <typename... Args>
void
append_serialized_arg(Args&&... args)
{
    base::append_serialized_arg(std::forward<Args>(args)...);
}

template <typename... Args>
inline constexpr bool has_trace_cache_arg_pairs_v =
    base::has_trace_cache_arg_pairs_v<Args...>;

template <typename... Args>
std::string
serialize_name_value_pairs(Args&&... args)
{
    return base::serialize_name_value_pairs(std::forward<Args>(args)...);
}

inline std::uint32_t
renumber_serialized_args(std::string& args_str, std::uint32_t next_idx)
{
    return base::renumber_serialized_args(args_str, next_idx);
}

inline std::uint32_t
next_arg_index(const std::string& args_str)
{
    return base::next_arg_index(args_str);
}

template <typename... Args>
std::string
serialize_annotation_args(Args&&... args)
{
    return base::serialize_annotation_args(std::forward<Args>(args)...);
}

template <typename T>
std::string
serialize_return_arg(T&& value)
{
    return base::serialize_return_arg(std::forward<T>(value));
}

template <typename CategoryT, typename... Args>
void
cache_start(const char* name, Args&&... args)
{
    base::cache_start(name, rocprofsys::trait::name<CategoryT>::value,
                      std::forward<Args>(args)...);
}

template <typename CategoryT>
void
append_cache_args(const char* name, std::string args_str)
{
    base::append_cache_args(name, rocprofsys::trait::name<CategoryT>::value,
                            std::move(args_str));
}

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
// append_cache_args (free function operating on the per-thread pending-entry stack)
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

// ---------------------------------------------------------------------------------------
// category parameter on the cache lifecycle helpers (category_region_base static
// members). After hoisting the helpers into the non-template base, the category is
// threaded through as an explicit argument rather than baked into the template, so the
// per-thread map must key on {name, category} and keep same-named regions in different
// categories distinct.
// ---------------------------------------------------------------------------------------

TEST(category_region_cache, cache_start_keys_on_name_and_category)
{
    const char* name = "shared_region";

    map_name_to_args.clear();
    // identical region name pushed under two different categories
    base::cache_start(name, "cat_a", serialize_name_value_pairs("a", 1));
    base::cache_start(name, "cat_b", serialize_name_value_pairs("b", 2));

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
    base::cache_start(name, "cat_a", serialize_name_value_pairs("a", 1));

    // appending under the same name but a different category must not touch cat_a and
    // must not fabricate an entry for cat_b
    base::append_cache_args(name, "cat_b", serialize_name_value_pairs("b", 2));

    const entry_key key_a{ name, "cat_a" };
    const entry_key key_b{ name, "cat_b" };
    EXPECT_TRUE(map_name_to_args.find(key_b) == map_name_to_args.end());

    auto itr_a = map_name_to_args.find(key_a);
    ASSERT_TRUE(itr_a != map_name_to_args.end());
    EXPECT_EQ(parse(itr_a->second.back().args).size(), 1u);

    // appending under the matching category extends that category's open entry
    base::append_cache_args(name, "cat_a", serialize_name_value_pairs("b", 2, "c", 3));
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
