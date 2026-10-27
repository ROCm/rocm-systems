// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "delivery.hpp"
#include "options/normalize.hpp"
#include "options/parser.hpp"
#include "options/suggest.hpp"
#include "options/table.hpp"
#include "options/validator.hpp"
#include "verb.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

using rocprofsys::rocsys::capability;
using rocprofsys::rocsys::capability_set;
using rocprofsys::rocsys::env_op;
using rocprofsys::rocsys::env_plan;
using rocprofsys::rocsys::k_verbs;
using rocprofsys::rocsys::plan;
using rocprofsys::rocsys::verb_spec;
using rocprofsys::rocsys::options::command_argv;
using rocprofsys::rocsys::options::context;
using rocprofsys::rocsys::options::legacy_conflict;
using rocprofsys::rocsys::options::normalize_choice;
using rocprofsys::rocsys::options::option_spec;
using rocprofsys::rocsys::options::option_table;
using rocprofsys::rocsys::options::parse;
using rocprofsys::rocsys::options::parse_request;
using rocprofsys::rocsys::options::parsed_command;
using rocprofsys::rocsys::options::resolve;
using rocprofsys::rocsys::options::same_choice;
using rocprofsys::rocsys::options::suggest_choice;
using rocprofsys::rocsys::options::validate;
using rocprofsys::rocsys::options::value_kind;

namespace
{
const verb_spec*
find_verb(std::string_view name)
{
    const auto found = std::ranges::find(k_verbs, name, &verb_spec::name);
    return found == k_verbs.end() ? nullptr : &*found;
}

bool
accept_trace(std::string_view token, const context&)
{
    constexpr auto names =
        std::to_array<std::string_view>({ "hip", "mpi", "kernel", "memory-copy" });
    return std::ranges::any_of(
        names, [token](std::string_view name) { return same_choice(token, name); });
}

std::vector<std::string>
trace_choices(const context&)
{
    return { "hip", "mpi", "kernel", "memory-copy" };
}

bool
include_value(std::string_view value)
{
    return value == "all" || value == "mpip" || value == "ompt" || value == "rcclp";
}

constexpr auto k_rocm_spellings = std::to_array<std::string_view>({
    "--rocm",
    "--rocm-domains",
    "--parallel",
});

constexpr auto k_include_spellings =
    std::to_array<std::string_view>({ "-I", "--include", "-E", "--exclude" });

const legacy_conflict k_conflicts[] = {
    {
        .binary    = "rocprof-sys-run",
        .spellings = k_rocm_spellings,
        .reason    = "writes a --traces setting",
    },
    {
        .binary        = "rocprof-sys-run",
        .spellings     = k_include_spellings,
        .matches_value = include_value,
        .reason        = "writes a --traces setting",
    },
};

const option_spec k_traces{
    .name          = "traces",
    .kind          = value_kind::token_list,
    .required_caps = static_cast<capability_set>(capability::honours_rocm_tracing),
    .conflicts     = k_conflicts,
    .min_values    = 1,
    .accepts       = accept_trace,
    .choices       = trace_choices,
};

const option_spec k_dry_run{ .name = "dry-run", .kind = value_kind::flag };

const option_spec k_engine[] = { k_traces, k_dry_run };

const option_spec k_query{
    .name          = "traces",
    .kind          = value_kind::flag,
    .required_caps = static_cast<capability_set>(capability::query_only),
};

std::optional<rocprofsys::rocsys::diagnostic>
resolve_named(std::span<const std::string> values, const verb_spec&, const context&,
              env_plan& out, std::string_view origin)
{
    out.push_back(env_op{ .key    = "ROCPROFSYS_ROCM_DOMAINS",
                          .value  = values.empty() ? std::string{} : values.front(),
                          .origin = std::string{ origin } });
    return std::nullopt;
}

std::optional<rocprofsys::rocsys::diagnostic>
resolve_alpha(std::span<const std::string> values, const verb_spec& verb,
              const context& ctx, env_plan& out)
{
    return resolve_named(values, verb, ctx, out, "--alpha");
}

std::optional<rocprofsys::rocsys::diagnostic>
resolve_beta(std::span<const std::string> values, const verb_spec& verb,
             const context& ctx, env_plan& out)
{
    return resolve_named(values, verb, ctx, out, "--beta");
}

const option_spec k_alpha{ .name    = "alpha",
                           .kind    = value_kind::single,
                           .resolve = resolve_alpha };
const option_spec k_beta{ .name    = "beta",
                          .kind    = value_kind::single,
                          .resolve = resolve_beta };

parse_request
make_request(const verb_spec& verb, bool strip, const std::vector<std::string>& argv,
             std::span<const option_spec> options, const context& ctx)
{
    return { .verb             = &verb,
             .strip_subcommand = strip,
             .argv             = argv,
             .options          = options,
             .ctx              = &ctx };
}
}  // namespace

TEST(options_test, normalize_treats_case_hyphen_and_underscore_alike)
{
    EXPECT_EQ(normalize_choice("Memory-Copy"), "memory_copy");
    EXPECT_TRUE(same_choice("memory-copy", "memory_copy"));
    EXPECT_FALSE(same_choice("hip", "hsa"));
}

TEST(options_test, suggest_choice_picks_one_close_name)
{
    const auto choices = trace_choices(context{});
    EXPECT_EQ(suggest_choice("hpp", choices), "hip");
    EXPECT_EQ(suggest_choice("kernl", choices), "kernel");
    EXPECT_FALSE(suggest_choice("./app", choices).has_value());
    EXPECT_FALSE(suggest_choice("hip", choices).has_value());
}

TEST(options_test, shipped_table_recognizes_dry_run_and_both_traces_rows)
{
    const auto table = option_table();
    ASSERT_EQ(table.size(), 3U);
    EXPECT_EQ(table[0].name, "dry-run");
    EXPECT_EQ(table[0].kind, value_kind::flag);
    EXPECT_EQ(table[0].required_caps, 0U);
    EXPECT_EQ(table[1].name, "traces");
    EXPECT_EQ(table[1].kind, value_kind::token_list);
    EXPECT_EQ(table[1].required_caps,
              static_cast<capability_set>(capability::honours_rocm_tracing));
    EXPECT_EQ(table[1].min_values, 1U);
    EXPECT_EQ(table[2].name, "traces");
    EXPECT_EQ(table[2].kind, value_kind::flag);
    EXPECT_EQ(table[2].required_caps,
              static_cast<capability_set>(capability::query_only));

    const auto* profile = find_verb("profile");
    const auto* avail   = find_verb("avail");
    const auto* causal  = find_verb("causal");
    ASSERT_NE(profile, nullptr);
    ASSERT_NE(avail, nullptr);
    ASSERT_NE(causal, nullptr);
    const context ctx{};
    const auto    argv =
        std::vector<std::string>{ "rocsys", "profile", "--dry-run", "--traces",
                                  "hip",    "--",      "./app" };
    auto parsed = parse(make_request(*profile, true, argv, table, ctx));
    ASSERT_TRUE(parsed);
    ASSERT_EQ(parsed.value().options.size(), 2U);
    EXPECT_EQ(parsed.value().options[0].spec->name, "dry-run");
    EXPECT_EQ(parsed.value().options[1].spec->kind, value_kind::token_list);
    EXPECT_EQ(parsed.value().options[1].values, (std::vector<std::string>{ "hip" }));
    EXPECT_TRUE(parsed.value().passthrough.empty());
    EXPECT_TRUE(validate(parsed.value()));

    const auto avail_argv   = std::vector<std::string>{ "rocsys", "avail", "--traces" };
    auto       avail_parsed = parse(make_request(*avail, true, avail_argv, table, ctx));
    ASSERT_TRUE(avail_parsed);
    EXPECT_TRUE(avail_parsed.value().passthrough.empty());
    EXPECT_EQ(avail_parsed.value().options[0].spec->kind, value_kind::flag);
    EXPECT_TRUE(validate(std::move(avail_parsed.value())));

    const auto valued = std::vector<std::string>{ "rocsys", "avail", "--traces", "hip" };
    const auto valued_parsed = parse(make_request(*avail, true, valued, table, ctx));
    ASSERT_FALSE(valued_parsed);
    EXPECT_NE(valued_parsed.error().error.find("does not take a value"),
              std::string::npos);

    const auto causal_argv =
        std::vector<std::string>{ "rocsys", "causal", "--traces", "hip", "--", "./app" };
    auto causal_parsed = parse(make_request(*causal, true, causal_argv, table, ctx));
    ASSERT_TRUE(causal_parsed);
    auto causal_checked = validate(std::move(causal_parsed.value()));
    ASSERT_FALSE(causal_checked);
    EXPECT_NE(causal_checked.error().error.find("not available for 'causal'"),
              std::string::npos);
}

TEST(options_test, dry_run_after_the_separator_stays_with_the_application)
{
    const auto* profile = find_verb("profile");
    ASSERT_NE(profile, nullptr);
    const context ctx{};
    const auto    argv =
        std::vector<std::string>{ "rocsys", "profile", "--", "./app", "--dry-run" };
    const auto parsed = parse(make_request(*profile, true, argv, option_table(), ctx));
    ASSERT_TRUE(parsed);
    EXPECT_TRUE(parsed.value().options.empty());
    EXPECT_EQ(parsed.value().app,
              (std::vector<std::string>{ "--", "./app", "--dry-run" }));
}

TEST(options_test, dry_run_rejects_a_value)
{
    const auto* profile = find_verb("profile");
    ASSERT_NE(profile, nullptr);
    const context ctx{};
    const auto    argv =
        std::vector<std::string>{ "rocsys", "profile", "--dry-run=yes", "--", "./app" };
    const auto parsed = parse(make_request(*profile, true, argv, option_table(), ctx));
    ASSERT_FALSE(parsed);
    EXPECT_NE(parsed.error().error.find("--dry-run does not take a value"),
              std::string::npos);
}

TEST(options_test, token_list_stops_at_a_flag_and_keeps_the_application)
{
    const auto* profile = find_verb("profile");
    ASSERT_NE(profile, nullptr);
    const context ctx{};
    const auto argv = std::vector<std::string>{ "rocsys",    "profile", "--traces", "hip",
                                                "--verbose", "--",      "./app" };
    const auto parsed = parse(make_request(*profile, true, argv, k_engine, ctx));
    ASSERT_TRUE(parsed);
    ASSERT_EQ(parsed.value().options.size(), 1U);
    EXPECT_EQ(parsed.value().options[0].values, (std::vector<std::string>{ "hip" }));
    EXPECT_EQ(parsed.value().passthrough, (std::vector<std::string>{ "--verbose" }));
    EXPECT_EQ(parsed.value().app, (std::vector<std::string>{ "--", "./app" }));
}

TEST(options_test, commas_spaces_and_repeats_merge_into_one_option)
{
    const auto* profile = find_verb("profile");
    ASSERT_NE(profile, nullptr);
    const context ctx{};
    const auto    argv =
        std::vector<std::string>{ "rocsys",   "profile", "--traces", "Memory-Copy, mpi",
                                  "--traces", "kernel",  "--",       "./app" };
    const auto parsed = parse(make_request(*profile, true, argv, k_engine, ctx));
    ASSERT_TRUE(parsed);
    ASSERT_EQ(parsed.value().options.size(), 1U);
    EXPECT_EQ(parsed.value().options[0].spec, &k_engine[0]);
    EXPECT_EQ(parsed.value().options[0].values,
              (std::vector<std::string>{ "Memory-Copy", "mpi", "kernel" }));
    EXPECT_EQ(parsed.value().app, (std::vector<std::string>{ "--", "./app" }));
}

TEST(options_test, application_token_is_not_swallowed)
{
    const auto* profile = find_verb("profile");
    ASSERT_NE(profile, nullptr);
    const context ctx{};
    const auto    argv =
        std::vector<std::string>{ "rocsys", "profile", "--traces", "hip", "./app" };
    const auto parsed = parse(make_request(*profile, true, argv, k_engine, ctx));
    ASSERT_FALSE(parsed);
    EXPECT_NE(parsed.error().error.find("./app"), std::string::npos);
    EXPECT_NE(parsed.error().hint.find("missing '--'"), std::string::npos);
}

TEST(options_test, unknown_value_suggests_a_name)
{
    const auto* profile = find_verb("profile");
    ASSERT_NE(profile, nullptr);
    const context ctx{};
    const auto    argv =
        std::vector<std::string>{ "rocsys", "profile", "--traces", "hpp", "--", "./app" };
    const auto parsed = parse(make_request(*profile, true, argv, k_engine, ctx));
    ASSERT_FALSE(parsed);
    EXPECT_NE(parsed.error().hint.find("did you mean 'hip'"), std::string::npos);
    EXPECT_NE(parsed.error().hint.find("missing '--'"), std::string::npos);
}

TEST(options_test, missing_list_value_is_rejected)
{
    const auto* profile = find_verb("profile");
    ASSERT_NE(profile, nullptr);
    const context ctx{};
    const auto    argv =
        std::vector<std::string>{ "rocsys", "profile", "--traces", "--", "./app" };
    auto parsed = parse(make_request(*profile, true, argv, k_engine, ctx));
    ASSERT_TRUE(parsed);
    auto checked = validate(std::move(parsed.value()));
    ASSERT_FALSE(checked);
    EXPECT_NE(checked.error().error.find("missing value for --traces"),
              std::string::npos);
}

TEST(options_test, flag_is_removed_and_rejects_a_value)
{
    const auto* profile = find_verb("profile");
    ASSERT_NE(profile, nullptr);
    const context ctx{};
    const auto    argv   = std::vector<std::string>{ "rocsys",    "profile", "--dry-run",
                                                     "--verbose", "--",      "./app" };
    const auto    parsed = parse(make_request(*profile, true, argv, k_engine, ctx));
    ASSERT_TRUE(parsed);
    ASSERT_EQ(parsed.value().options.size(), 1U);
    EXPECT_EQ(parsed.value().options[0].spec->name, "dry-run");
    EXPECT_TRUE(parsed.value().options[0].values.empty());
    EXPECT_EQ(parsed.value().passthrough, (std::vector<std::string>{ "--verbose" }));

    const auto assigned = std::vector<std::string>{ "rocsys", "profile", "--dry-run=1" };
    const auto rejected = parse(make_request(*profile, true, assigned, k_engine, ctx));
    ASSERT_FALSE(rejected);
    EXPECT_NE(rejected.error().error.find("does not take a value"), std::string::npos);
}

TEST(options_test, single_value_accepts_both_spellings)
{
    const option_spec spec{ .name       = "config",
                            .kind       = value_kind::single,
                            .min_values = 1 };
    const option_spec table[] = { spec };
    const auto*       profile = find_verb("profile");
    ASSERT_NE(profile, nullptr);
    const context ctx{};
    const auto    spaced = std::vector<std::string>{ "rocsys", "profile", "--config",
                                                     "/tmp",   "--",      "./app" };
    const auto    parsed = parse(make_request(*profile, true, spaced, table, ctx));
    ASSERT_TRUE(parsed);
    ASSERT_EQ(parsed.value().options.size(), 1U);
    EXPECT_EQ(parsed.value().options[0].values, (std::vector<std::string>{ "/tmp" }));

    const auto attached =
        std::vector<std::string>{ "rocsys", "profile", "--config=/tmp", "--", "./app" };
    const auto attached_parsed =
        parse(make_request(*profile, true, attached, table, ctx));
    ASSERT_TRUE(attached_parsed);
    EXPECT_EQ(attached_parsed.value().options[0].values,
              (std::vector<std::string>{ "/tmp" }));
}

TEST(options_test, capability_rejects_a_verb_that_lacks_the_bit)
{
    const auto* causal  = find_verb("causal");
    const auto* profile = find_verb("profile");
    ASSERT_NE(causal, nullptr);
    ASSERT_NE(profile, nullptr);
    const context ctx{};
    const auto    argv =
        std::vector<std::string>{ "rocsys", "causal", "--traces", "hip", "--", "./app" };
    auto parsed = parse(make_request(*causal, true, argv, k_engine, ctx));
    ASSERT_TRUE(parsed);
    auto checked = validate(std::move(parsed.value()));
    ASSERT_FALSE(checked);
    EXPECT_NE(checked.error().error.find("not available for 'causal'"),
              std::string::npos);

    const auto profile_argv =
        std::vector<std::string>{ "rocsys", "profile", "--traces", "hip", "--", "./app" };
    auto profile_parsed =
        parse(make_request(*profile, true, profile_argv, k_engine, ctx));
    ASSERT_TRUE(profile_parsed);
    auto profile_checked = validate(std::move(profile_parsed.value()));
    ASSERT_TRUE(profile_checked);
}

TEST(options_test, query_only_option_is_limited_to_avail)
{
    const auto* avail   = find_verb("avail");
    const auto* profile = find_verb("profile");
    ASSERT_NE(avail, nullptr);
    ASSERT_NE(profile, nullptr);
    const context     ctx{};
    const option_spec table[] = { k_query };
    const auto avail_argv     = std::vector<std::string>{ "rocsys", "avail", "--traces" };
    auto       avail_parsed   = parse(make_request(*avail, true, avail_argv, table, ctx));
    ASSERT_TRUE(avail_parsed);
    EXPECT_TRUE(validate(std::move(avail_parsed.value())));

    const auto profile_argv =
        std::vector<std::string>{ "rocsys", "profile", "--traces", "--", "./app" };
    auto profile_parsed = parse(make_request(*profile, true, profile_argv, table, ctx));
    ASSERT_TRUE(profile_parsed);
    auto checked = validate(std::move(profile_parsed.value()));
    ASSERT_FALSE(checked);
    EXPECT_NE(checked.error().error.find("not available for 'profile'"),
              std::string::npos);
}

TEST(options_test, legacy_conflict_is_per_binary_and_per_value)
{
    const auto* profile    = find_verb("profile");
    const auto* instrument = find_verb("instrument");
    ASSERT_NE(profile, nullptr);
    ASSERT_NE(instrument, nullptr);
    const context ctx{};

    const auto rocm =
        std::vector<std::string>{ "rocsys",     "profile", "--traces", "hip",
                                  "--rocm=hip", "--",      "./app" };
    auto rocm_parsed = parse(make_request(*profile, true, rocm, k_engine, ctx));
    ASSERT_TRUE(rocm_parsed);
    auto rocm_checked = validate(std::move(rocm_parsed.value()));
    ASSERT_FALSE(rocm_checked);
    EXPECT_NE(rocm_checked.error().error.find("--rocm conflicts with --traces"),
              std::string::npos);

    const auto include_hit =
        std::vector<std::string>{ "rocsys", "profile", "--traces", "hip",
                                  "-I",     "mpip",    "--",       "./app" };
    auto include_parsed = parse(make_request(*profile, true, include_hit, k_engine, ctx));
    ASSERT_TRUE(include_parsed);
    EXPECT_FALSE(validate(std::move(include_parsed.value())));

    const auto include_miss =
        std::vector<std::string>{ "rocsys",      "profile", "--traces", "hip", "-I",
                                  "mutex-locks", "-E",      "amd-smi",  "--",  "./app" };
    auto miss_parsed = parse(make_request(*profile, true, include_miss, k_engine, ctx));
    ASSERT_TRUE(miss_parsed);
    EXPECT_TRUE(validate(std::move(miss_parsed.value())));

    const auto other_binary =
        std::vector<std::string>{ "rocsys",     "instrument", "--traces", "hip",
                                  "--rocm=hip", "--",         "./app" };
    auto other_parsed =
        parse(make_request(*instrument, true, other_binary, k_engine, ctx));
    ASSERT_TRUE(other_parsed);
    EXPECT_TRUE(validate(std::move(other_parsed.value())));

    const auto after_separator = std::vector<std::string>{ "rocsys",   "profile",
                                                           "--traces", "hip",
                                                           "--",       "--rocm" };
    auto       after_parsed =
        parse(make_request(*profile, true, after_separator, k_engine, ctx));
    ASSERT_TRUE(after_parsed);
    EXPECT_TRUE(validate(std::move(after_parsed.value())));
}

TEST(options_test, same_key_folds_equal_values_and_rejects_different_ones)
{
    const auto* profile = find_verb("profile");
    ASSERT_NE(profile, nullptr);
    const context  ctx{};
    parsed_command same;
    same.verb = profile;
    same.options.push_back({ &k_alpha, { "hip_api" } });
    same.options.push_back({ &k_beta, { "hip_api" } });
    const auto folded = resolve(same, ctx);
    ASSERT_TRUE(folded);
    ASSERT_EQ(folded.value().size(), 1U);
    EXPECT_EQ(folded.value().front().key, "ROCPROFSYS_ROCM_DOMAINS");
    EXPECT_EQ(folded.value().front().value, "hip_api");
    EXPECT_EQ(folded.value().front().origin, "--alpha");

    parsed_command different;
    different.verb = profile;
    different.options.push_back({ &k_alpha, { "hip_api" } });
    different.options.push_back({ &k_beta, { "kernel_dispatch" } });
    const auto rejected = resolve(different, ctx);
    ASSERT_FALSE(rejected);
    EXPECT_NE(rejected.error().error.find("--alpha"), std::string::npos);
    EXPECT_NE(rejected.error().error.find("--beta"), std::string::npos);
    EXPECT_NE(rejected.error().error.find("ROCPROFSYS_ROCM_DOMAINS"), std::string::npos);
}

TEST(options_test, empty_plan_keeps_the_rewrite_argv)
{
    const auto* rewrite = find_verb("rewrite");
    ASSERT_NE(rewrite, nullptr);
    const context ctx{};
    const auto    argv   = std::vector<std::string>{ "rocsys", "rewrite", "--", "./app" };
    auto          parsed = parse(make_request(*rewrite, true, argv, option_table(), ctx));
    ASSERT_TRUE(parsed);
    auto checked = validate(std::move(parsed.value()));
    ASSERT_TRUE(checked);
    const auto env = resolve(checked.value(), ctx);
    ASSERT_TRUE(env);
    EXPECT_TRUE(env.value().empty());
    const auto planned =
        plan(*rewrite, false, command_argv("rocsys", checked.value()), env.value());
    ASSERT_TRUE(planned);
    EXPECT_EQ(planned.value().argv, (std::vector<std::string>{ "rocprof-sys-instrument",
                                                               "-o", "--", "./app" }));
}
