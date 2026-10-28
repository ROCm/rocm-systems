// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "delivery.hpp"
#include "options/parser.hpp"
#include "options/table.hpp"
#include "options/traces.hpp"
#include "options/validator.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

using rocprofsys::rocsys::env_op;
using rocprofsys::rocsys::k_verbs;
using rocprofsys::rocsys::plan;
using rocprofsys::rocsys::verb_spec;
using rocprofsys::rocsys::options::command_argv;
using rocprofsys::rocsys::options::context;
using rocprofsys::rocsys::options::disabled_runtime_notes;
using rocprofsys::rocsys::options::domain_query;
using rocprofsys::rocsys::options::parse;
using rocprofsys::rocsys::options::parse_request;
using rocprofsys::rocsys::options::resolve;
using rocprofsys::rocsys::options::validate;

namespace
{
std::vector<std::string> g_choices;
std::vector<std::string> g_supported;
bool                     g_openmp_compiled = true;

const verb_spec*
find_verb(std::string_view name)
{
    const auto found = std::ranges::find(k_verbs, name, &verb_spec::name);
    return found == k_verbs.end() ? nullptr : &*found;
}

std::vector<std::string>
choices()
{
    return g_choices;
}

std::vector<std::string>
supported()
{
    return g_supported;
}

bool
runtime_compiled(std::string_view name)
{
    return name != "openmp" || g_openmp_compiled;
}

context
make_context()
{
    return context{ .domains = domain_query{ .choices          = choices,
                                             .supported        = supported,
                                             .runtime_compiled = runtime_compiled } };
}

void
install_domains()
{
    g_choices   = { "hip_api", "kernel_dispatch", "memory_copy", "memory_allocation",
                    "ompt",    "rccl_api" };
    g_supported = g_choices;
    g_openmp_compiled = true;
}

parse_request
make_request(const verb_spec& verb, const std::vector<std::string>& argv,
             const context& ctx)
{
    return { .verb             = &verb,
             .strip_subcommand = true,
             .argv             = argv,
             .options          = rocprofsys::rocsys::options::option_table(),
             .ctx              = &ctx };
}

const env_op*
find_op(const rocprofsys::rocsys::env_plan& ops, std::string_view key)
{
    const auto found = std::ranges::find(ops, key, &env_op::key);
    return found == ops.end() ? nullptr : &*found;
}

std::string
read_file(const std::filesystem::path& path)
{
    std::ifstream      input{ path };
    std::ostringstream text;
    text << input.rdbuf();
    return text.str();
}

std::string
cli_flag_in(std::string_view text)
{
    constexpr std::string_view k_key = "\"cli_flag\"";
    const auto                 at    = text.find(k_key);
    const auto                 colon = text.find(':', at);
    const auto                 open  = text.find('"', colon + 1);
    const auto                 close = text.find('"', open + 1);
    return std::string{ text.substr(open + 1, close - open - 1) };
}

}  // namespace

TEST(traces_test, hip_and_memory_replace_defaults_and_turn_mpi_off)
{
    install_domains();
    const auto* profile = find_verb("profile");
    ASSERT_NE(profile, nullptr);
    const auto ctx    = make_context();
    const auto argv   = std::vector<std::string>{ "rocsys",     "profile", "--traces",
                                                  "Hip,memory", "--",      "./app" };
    auto       parsed = parse(make_request(*profile, argv, ctx));
    ASSERT_TRUE(parsed) << parsed.error().error;
    auto checked = validate(std::move(parsed.value()));
    ASSERT_TRUE(checked);
    const auto env = resolve(checked.value(), ctx);
    ASSERT_TRUE(env) << env.error().error;
    ASSERT_NE(find_op(env.value(), "ROCPROFSYS_ROCM_DOMAINS"), nullptr);
    EXPECT_EQ(find_op(env.value(), "ROCPROFSYS_ROCM_DOMAINS")->value,
              "hip_api,memory_allocation,memory_copy");
    EXPECT_EQ(find_op(env.value(), "ROCPROFSYS_USE_MPIP")->value, "false");
    EXPECT_EQ(find_op(env.value(), "ROCPROFSYS_USE_OMPT")->value, "false");
    EXPECT_EQ(find_op(env.value(), "ROCPROFSYS_USE_KOKKOSP"), nullptr);
    EXPECT_FALSE(find_op(env.value(), "ROCPROFSYS_USE_MPIP")->embed);
    const auto notes = disabled_runtime_notes(env.value(), nullptr);
    ASSERT_EQ(notes.size(), 1U);
    EXPECT_NE(notes.front().find("MPI (MPIP)"), std::string::npos);
    EXPECT_NE(notes.front().find("'mpi' is not in --traces"), std::string::npos);
}

TEST(traces_test, mpi_only_clears_rocm_domains_and_skips_the_mpi_note)
{
    install_domains();
    const auto* profile = find_verb("profile");
    ASSERT_NE(profile, nullptr);
    const auto ctx = make_context();
    const auto argv =
        std::vector<std::string>{ "rocsys", "profile", "--traces", "MPI", "--", "./app" };
    auto parsed = parse(make_request(*profile, argv, ctx));
    ASSERT_TRUE(parsed) << parsed.error().error;
    auto checked = validate(std::move(parsed.value()));
    ASSERT_TRUE(checked);
    const auto env = resolve(checked.value(), ctx);
    ASSERT_TRUE(env) << env.error().error;
    EXPECT_EQ(find_op(env.value(), "ROCPROFSYS_ROCM_DOMAINS")->value, "");
    EXPECT_EQ(find_op(env.value(), "ROCPROFSYS_USE_MPIP")->value, "true");
    EXPECT_TRUE(disabled_runtime_notes(env.value(), nullptr).empty());
}

TEST(traces_test, ompt_selects_openmp_and_a_shell_override_is_noted)
{
    install_domains();
    const auto* profile = find_verb("profile");
    ASSERT_NE(profile, nullptr);
    const auto ctx    = make_context();
    const auto argv   = std::vector<std::string>{ "rocsys", "profile", "--traces",
                                                  "ompt",   "--",      "./app" };
    auto       parsed = parse(make_request(*profile, argv, ctx));
    ASSERT_TRUE(parsed);
    auto checked = validate(std::move(parsed.value()));
    ASSERT_TRUE(checked);
    const auto env = resolve(checked.value(), ctx);
    ASSERT_TRUE(env);
    EXPECT_EQ(find_op(env.value(), "ROCPROFSYS_ROCM_DOMAINS")->value, "");
    EXPECT_EQ(find_op(env.value(), "ROCPROFSYS_USE_OMPT")->value, "true");

    const auto notes = disabled_runtime_notes(env.value(), [](const char* key) -> char* {
        if(std::string_view{ key } == "ROCPROFSYS_USE_UCX")
        {
            static char value[] = "1";
            return value;
        }
        return nullptr;
    });
    EXPECT_NE(notes.end(), std::ranges::find_if(notes, [](const std::string& note) {
                  return note.find("UCX") != std::string::npos;
              }));
}

TEST(traces_test, shell_false_suppresses_the_library_default_note)
{
    install_domains();
    const auto* profile = find_verb("profile");
    ASSERT_NE(profile, nullptr);
    const auto ctx = make_context();
    const auto argv =
        std::vector<std::string>{ "rocsys", "profile", "--traces", "hip", "--", "./app" };
    auto parsed = parse(make_request(*profile, argv, ctx));
    ASSERT_TRUE(parsed);
    auto checked = validate(std::move(parsed.value()));
    ASSERT_TRUE(checked);
    const auto env = resolve(checked.value(), ctx);
    ASSERT_TRUE(env);
    const auto notes = disabled_runtime_notes(env.value(), [](const char* key) -> char* {
        if(std::string_view{ key } == "ROCPROFSYS_USE_MPIP")
        {
            static char value[] = "false";
            return value;
        }
        return nullptr;
    });
    EXPECT_TRUE(notes.empty());
}

TEST(traces_test, unknown_unavailable_and_kokkos_are_errors)
{
    install_domains();
    g_supported         = { "hip_api" };
    const auto* profile = find_verb("profile");
    ASSERT_NE(profile, nullptr);
    const auto ctx = make_context();

    const auto unknown =
        std::vector<std::string>{ "rocsys", "profile", "--traces", "hpp", "--", "./app" };
    const auto unknown_parsed = parse(make_request(*profile, unknown, ctx));
    ASSERT_FALSE(unknown_parsed);
    EXPECT_NE(unknown_parsed.error().hint.find("did you mean 'hip'"), std::string::npos);
    EXPECT_NE(unknown_parsed.error().hint.find("missing '--'"), std::string::npos);

    const auto kokkos        = std::vector<std::string>{ "rocsys", "profile", "--traces",
                                                         "kokkos", "--",      "./app" };
    const auto kokkos_parsed = parse(make_request(*profile, kokkos, ctx));
    ASSERT_FALSE(kokkos_parsed);
    EXPECT_NE(kokkos_parsed.error().error.find("unknown"), std::string::npos);

    const auto kernel        = std::vector<std::string>{ "rocsys", "profile", "--traces",
                                                         "kernel", "--",      "./app" };
    auto       kernel_parsed = parse(make_request(*profile, kernel, ctx));
    ASSERT_TRUE(kernel_parsed);
    auto kernel_checked = validate(std::move(kernel_parsed.value()));
    ASSERT_TRUE(kernel_checked);
    const auto kernel_env = resolve(kernel_checked.value(), ctx);
    ASSERT_FALSE(kernel_env);
    EXPECT_NE(kernel_env.error().hint.find("headers"), std::string::npos);

    g_openmp_compiled      = false;
    const auto ompt        = std::vector<std::string>{ "rocsys", "profile", "--traces",
                                                       "openmp", "--",      "./app" };
    auto       ompt_parsed = parse(make_request(*profile, ompt, ctx));
    ASSERT_TRUE(ompt_parsed);
    auto ompt_checked = validate(std::move(ompt_parsed.value()));
    ASSERT_TRUE(ompt_checked);
    const auto ompt_env = resolve(ompt_checked.value(), ctx);
    ASSERT_FALSE(ompt_env);
    EXPECT_NE(ompt_env.error().hint.find("not compiled"), std::string::npos);
}

TEST(traces_test, rewrite_embeds_every_trace_setting)
{
    install_domains();
    const auto* rewrite = find_verb("rewrite");
    ASSERT_NE(rewrite, nullptr);
    const auto ctx = make_context();
    const auto argv =
        std::vector<std::string>{ "rocsys", "rewrite", "--traces", "hip", "--", "./app" };
    auto parsed = parse(make_request(*rewrite, argv, ctx));
    ASSERT_TRUE(parsed);
    auto checked = validate(std::move(parsed.value()));
    ASSERT_TRUE(checked);
    const auto env = resolve(checked.value(), ctx);
    ASSERT_TRUE(env);
    EXPECT_TRUE(std::ranges::all_of(env.value(), &env_op::embed));
    const auto planned =
        plan(*rewrite, false, command_argv("rocsys", checked.value()), env.value());
    ASSERT_TRUE(planned);
    EXPECT_TRUE(planned.value().env.empty());
    EXPECT_EQ(
        planned.value().argv,
        (std::vector<std::string>{
            "rocprof-sys-instrument", "--env", "ROCPROFSYS_ROCM_DOMAINS=hip_api", "--env",
            "ROCPROFSYS_USE_MPIP=false", "--env", "ROCPROFSYS_USE_OMPT=false", "--env",
            "ROCPROFSYS_USE_RCCLP=false", "--env", "ROCPROFSYS_USE_UCX=false", "--env",
            "ROCPROFSYS_USE_OPENSHMEM=false", "-o", "--", "./app" }));
}

TEST(traces_test, legacy_flags_conflict_per_binary_and_per_value)
{
    install_domains();
    const auto* profile    = find_verb("profile");
    const auto* instrument = find_verb("instrument");
    ASSERT_NE(profile, nullptr);
    ASSERT_NE(instrument, nullptr);
    const auto ctx = make_context();

    const auto preset =
        std::vector<std::string>{ "rocsys",   "profile",  "--traces", "hip",
                                  "--preset", "balanced", "--",       "./app" };
    auto preset_parsed = parse(make_request(*profile, preset, ctx));
    ASSERT_TRUE(preset_parsed);
    auto preset_checked = validate(std::move(preset_parsed.value()));
    ASSERT_FALSE(preset_checked);
    EXPECT_NE(preset_checked.error().error.find("--preset conflicts with --traces"),
              std::string::npos);
    EXPECT_EQ(preset_checked.error().hint, "use one or the other");

    const auto exported = std::vector<std::string>{
        "rocsys", "profile", "--traces", "hip", "--export-config=out.json", "--", "./app"
    };
    auto export_parsed = parse(make_request(*profile, exported, ctx));
    ASSERT_TRUE(export_parsed);
    auto export_checked = validate(std::move(export_parsed.value()));
    ASSERT_FALSE(export_checked);
    EXPECT_EQ(export_checked.error().hint, "use --dry-run");

    const auto include_miss =
        std::vector<std::string>{ "rocsys",      "profile", "--traces", "hip", "-I",
                                  "mutex-locks", "-E",      "amd-smi",  "--",  "./app" };
    auto miss_parsed = parse(make_request(*profile, include_miss, ctx));
    ASSERT_TRUE(miss_parsed);
    EXPECT_TRUE(validate(std::move(miss_parsed.value())));

    const auto include_hit =
        std::vector<std::string>{ "rocsys", "profile",   "--traces", "hip",
                                  "-I",     "mpip,ompt", "--",       "./app" };
    auto hit_parsed = parse(make_request(*profile, include_hit, ctx));
    ASSERT_TRUE(hit_parsed);
    EXPECT_FALSE(validate(std::move(hit_parsed.value())));

    const auto kokkos_flag =
        std::vector<std::string>{ "rocsys",        "profile", "--traces", "hip",
                                  "--use-kokkosp", "--",      "./app" };
    auto kokkos_parsed = parse(make_request(*profile, kokkos_flag, ctx));
    ASSERT_TRUE(kokkos_parsed);
    EXPECT_TRUE(validate(std::move(kokkos_parsed.value())));

    const auto env_hit =
        std::vector<std::string>{ "rocsys", "instrument", "--traces",
                                  "hip",    "--env",      "ROCPROFSYS_USE_MPIP=1",
                                  "--",     "./app" };
    auto env_parsed = parse(make_request(*instrument, env_hit, ctx));
    ASSERT_TRUE(env_parsed);
    auto env_checked = validate(std::move(env_parsed.value()));
    ASSERT_FALSE(env_checked);
    EXPECT_NE(env_checked.error().error.find("--env conflicts with --traces"),
              std::string::npos);

    const auto env_miss =
        std::vector<std::string>{ "rocsys", "instrument",           "--traces", "hip",
                                  "--env",  "ROCPROFSYS_VERBOSE=1", "--",       "./app" };
    auto env_miss_parsed = parse(make_request(*instrument, env_miss, ctx));
    ASSERT_TRUE(env_miss_parsed);
    EXPECT_TRUE(validate(std::move(env_miss_parsed.value())));
}

TEST(traces_test, every_preset_shortcut_conflicts)
{
    install_domains();
    const auto* profile = find_verb("profile");
    ASSERT_NE(profile, nullptr);
    const auto ctx   = make_context();
    const auto root  = std::filesystem::path{ ROCPROFSYS_PRESET_DIR };
    auto       found = 0;
    for(const auto& entry : std::filesystem::directory_iterator{ root })
    {
        if(entry.path().extension() != ".json" ||
           entry.path().filename() == "schema.json")
        {
            continue;
        }
        const auto flag = cli_flag_in(read_file(entry.path()));
        ASSERT_TRUE(flag.starts_with("--")) << entry.path();
        const auto argv =
            std::vector<std::string>{ "rocsys", "profile", "--traces", "hip",
                                      flag,     "--",      "./app" };
        auto parsed = parse(make_request(*profile, argv, ctx));
        ASSERT_TRUE(parsed) << flag;
        auto checked = validate(std::move(parsed.value()));
        ASSERT_FALSE(checked) << flag;
        EXPECT_NE(checked.error().error.find("conflicts with --traces"),
                  std::string::npos)
            << flag;
        ++found;
    }
    EXPECT_GE(found, 11);
}
