// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Unit tests for unified_memory_processor_t using synthetic agents and a
// recording output sink.

#include "common/env_vars.hpp"
#include "common/tests/filesystem.hpp"
#include "core/categories.hpp"
#include "core/trace_cache/unified_memory_processor.hpp"
#include "unified_memory_test_helpers.hpp"
#include <cstdint>

#include <nlohmann/json.hpp>
#include <timemory/settings.hpp>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <system_error>
#include <vector>

using rocprofsys::agent_manager;
using rocprofsys::agent_type;
using rocprofsys::output_format;
using rocprofsys::trace_cache::kfd_sample;
using rocprofsys::trace_cache::migration_stats;
using rocprofsys::trace_cache::output_file_sink_view;
using rocprofsys::trace_cache::unified_memory_processor_t;
using rocprofsys::trace_cache::detail::kTriggerTable;
using rocprofsys::trace_cache::test::make_cpu_agent;
using rocprofsys::trace_cache::test::make_gpu_agent;
using rocprofsys::trace_cache::test::make_kfd_page_fault_sample;
using rocprofsys::trace_cache::test::make_kfd_page_migrate_sample;
using rocprofsys::trace_cache::test::make_kfd_page_migrate_sample_raw_args;

using ::testing::HasSubstr;

namespace env_vars = rocprofsys::env_vars;

namespace
{

using TestProcessor = unified_memory_processor_t;

struct registered_file
{
    std::string   path;
    output_format format;
};

struct recording_output_sink
{
    void register_file(std::string path, output_format format)
    {
        files.push_back({ std::move(path), format });
    }

    void clear() { files.clear(); }

    std::vector<registered_file> files = {};
};

// The processor reads output config from timemory globals at construction
// time; these RAII guards isolate those globals so each test writes into
// its own mkdtemp() directory.
struct ScopedUseOutputSuffix
{
    const bool previous;
    explicit ScopedUseOutputSuffix(bool desired)
    : previous(tim::settings::use_output_suffix())
    {
        tim::settings::use_output_suffix() = desired;
    }
    ~ScopedUseOutputSuffix() { tim::settings::use_output_suffix() = previous; }
};

struct ScopedOutputPath
{
    const std::string previous;
    explicit ScopedOutputPath(std::string desired)
    : previous(tim::settings::output_path())
    {
        tim::settings::output_path() = std::move(desired);
    }
    ~ScopedOutputPath() { tim::settings::output_path() = previous; }
};

struct ScopedEnv
{
    explicit ScopedEnv(std::string name, std::string desired)
    : env_name(std::move(name))
    {
        const auto* current = getenv(env_name.c_str());
        if(current != nullptr)
        {
            previous = current;
            had_env  = true;
        }
        setenv(env_name.c_str(), desired.c_str(), 1);
    }

    ~ScopedEnv()
    {
        if(had_env)
            setenv(env_name.c_str(), previous.c_str(), 1);
        else
            unsetenv(env_name.c_str());
    }

    std::string env_name;
    std::string previous;
    bool        had_env = false;
};

class UnifiedMemoryProcessorTest : public ::testing::Test
{
protected:
    static constexpr int           kPid  = 12345;
    static constexpr std::uint32_t kCpu0 = 0;
    static constexpr std::uint32_t kGpu1 = 1;
    static constexpr std::uint32_t kGpu2 = 2;

    void SetUp() override
    {
        m_suffix_guard = std::make_unique<ScopedUseOutputSuffix>(true);

        // Match the constructor's HSA_XNACK read so the JSON assertion uses
        // the same observed value.
        const char* xnack      = std::getenv("HSA_XNACK");
        expected_xnack_enabled = (xnack != nullptr && std::strcmp(xnack, "1") == 0);

        char        tmpl[] = "/tmp/rocprofsys_um_test_XXXXXX";
        const char* d      = mkdtemp(tmpl);
        ASSERT_NE(d, nullptr) << "mkdtemp failed";
        tmp_dir = d;

        m_output_path_guard = std::make_unique<ScopedOutputPath>(tmp_dir);

        rebuild_processor();
    }

    void TearDown() override
    {
        processor.reset();
        registry.clear();
        if(!tmp_dir.empty())
        {
            std::error_code ec;
            test_common::fs::remove_all(tmp_dir, ec);
        }
    }

    std::optional<nlohmann::json> read_json_output() const
    {
        auto json_path = get_registered_path(output_format::json);
        if(!json_path.has_value())
        {
            ADD_FAILURE() << "JSON file was not registered";
            return std::nullopt;
        }
        std::ifstream f(*json_path);
        if(!f.is_open())
        {
            ADD_FAILURE() << "JSON file missing on disk: " << *json_path;
            return std::nullopt;
        }
        auto j = nlohmann::json::parse(f, /*cb=*/nullptr, /*allow_exceptions=*/false);
        if(j.is_discarded())
        {
            ADD_FAILURE() << "JSON parse failed for " << *json_path;
            return std::nullopt;
        }
        return j;
    }

    [[nodiscard]] const std::vector<registered_file>& registered_files() const
    {
        return registry.files;
    }

    [[nodiscard]] std::optional<std::string> get_registered_path(
        output_format format) const
    {
        for(const auto& file : registered_files())
        {
            if(file.format == format) return file.path;
        }
        return std::nullopt;
    }

    void rebuild_processor(std::string gpu1_name = "gfx950",
                           std::string gpu2_name = "gfx950")
    {
        processor.reset();

        agent_mgr = std::make_shared<agent_manager>();
        auto cpu0 = make_cpu_agent(kCpu0, "AMD CPU");
        auto gpu1 = make_gpu_agent(kGpu1, std::move(gpu1_name));
        auto gpu2 = make_gpu_agent(kGpu2, std::move(gpu2_name));
        agent_mgr->insert_agent(cpu0);
        agent_mgr->insert_agent(gpu1);
        agent_mgr->insert_agent(gpu2);

        processor = std::make_unique<TestProcessor>(agent_mgr, kPid,
                                                    output_file_sink_view{ registry });
    }

    void feed_h2d_migrate_with_value(double v)
    {
        auto s  = make_kfd_page_migrate_sample(kCpu0, kGpu1, /*size=*/0,
                                               /*duration=*/100, /*device_id=*/0);
        s.value = v;
        processor->handle(s);
    }

    std::shared_ptr<agent_manager> agent_mgr;
    recording_output_sink          registry;
    std::string                    tmp_dir;
    std::unique_ptr<TestProcessor> processor;
    bool                           expected_xnack_enabled = false;

private:
    std::unique_ptr<ScopedUseOutputSuffix> m_suffix_guard;
    std::unique_ptr<ScopedOutputPath>      m_output_path_guard;
};

TEST(MigrationStats, ArithmeticAndSentinels)
{
    migration_stats s;
    EXPECT_EQ(s.count, 0u);
    EXPECT_EQ(s.total_size_bytes, 0u);
    EXPECT_EQ(s.max_size_bytes, 0u);
    EXPECT_EQ(s.min_size_bytes, std::numeric_limits<std::uint64_t>::max());  // sentinel
    EXPECT_DOUBLE_EQ(s.avg_size_bytes(), 0.0);
    EXPECT_DOUBLE_EQ(s.migration_throughput_gbps(), 0.0);  // zero-division guard

    s.add_migration(/*size=*/1000, /*duration_ns=*/100);
    s.add_migration(/*size=*/3000, /*duration_ns=*/300);

    EXPECT_EQ(s.count, 2u);
    EXPECT_EQ(s.total_size_bytes, 4000u);
    EXPECT_EQ(s.total_time_ns, 400u);
    EXPECT_EQ(s.min_size_bytes, 1000u);
    EXPECT_EQ(s.max_size_bytes, 3000u);
    EXPECT_DOUBLE_EQ(s.avg_size_bytes(), 2000.0);
    EXPECT_DOUBLE_EQ(s.migration_throughput_gbps(),
                     10.0);  // 4000 bytes / 400 ns = 10 GB/s
}

TEST(UnifiedMemoryCategory, MigrationThroughputNameAndCompatibilityAlias)
{
    using category_type = tim::category::unified_memory_migration_throughput;

    EXPECT_STREQ(tim::trait::name<category_type>::value,
                 "unified_memory_migration_throughput");
    EXPECT_EQ(rocprofsys::category_enum_id<category_type>::value,
              ROCPROFSYS_CATEGORY_UNIFIED_MEMORY_MIGRATION_THROUGHPUT);
}

TEST_F(UnifiedMemoryProcessorTest, JsonSchemaAlwaysEmitsAllDirections)
{
    processor->handle(make_kfd_page_migrate_sample(kCpu0, kGpu1, /*size=*/4096,
                                                   /*duration=*/1000,
                                                   /*device_id=*/0));
    processor->finalize_processing();

    auto j_opt = read_json_output();
    ASSERT_TRUE(j_opt.has_value());
    auto const& j = *j_opt;
    ASSERT_TRUE(j.contains("devices"));
    ASSERT_EQ(j["devices"].size(), 1u);
    auto const& migrations = j["devices"][0]["migrations"];

    for(const char* dir : { "host_to_device", "device_to_host", "device_to_device" })
    {
        ASSERT_TRUE(migrations.contains(dir)) << "missing direction key: " << dir;
        for(const char* k :
            { "count", "total_size_bytes", "min_size_bytes", "max_size_bytes",
              "avg_size_bytes", "total_time_ns", "migration_throughput_gbps" })
        {
            EXPECT_TRUE(migrations[dir].contains(k)) << dir << " missing stat: " << k;
        }
    }

    EXPECT_EQ(migrations["device_to_host"]["count"], 0u);
    EXPECT_EQ(migrations["device_to_device"]["count"], 0u);
    EXPECT_EQ(migrations["host_to_device"]["count"], 1u);
}

TEST_F(UnifiedMemoryProcessorTest, ClassifyDirectionFromTopology)
{
    processor->handle(make_kfd_page_migrate_sample(kCpu0, kGpu1, 1024, 100, /*dev=*/0));
    processor->handle(make_kfd_page_migrate_sample(kGpu1, kCpu0, 2048, 200, /*dev=*/0));
    processor->handle(make_kfd_page_migrate_sample(kGpu1, kGpu2, 4096, 400, /*dev=*/0));

    processor->finalize_processing();

    auto j_opt = read_json_output();
    ASSERT_TRUE(j_opt.has_value());
    auto const& m = (*j_opt)["devices"][0]["migrations"];
    EXPECT_EQ(m["host_to_device"]["count"], 1u);
    EXPECT_EQ(m["host_to_device"]["total_size_bytes"], 1024u);
    EXPECT_EQ(m["device_to_host"]["count"], 1u);
    EXPECT_EQ(m["device_to_host"]["total_size_bytes"], 2048u);
    EXPECT_EQ(m["device_to_device"]["count"], 1u);
    EXPECT_EQ(m["device_to_device"]["total_size_bytes"], 4096u);
}

TEST_F(UnifiedMemoryProcessorTest,
       HostToDeviceMigrationsBucketByDestinationGpuUnderProducerSemantics)
{
    processor->handle(make_kfd_page_migrate_sample(kCpu0, kGpu1, 1024, 100, /*dev=*/kCpu0,
                                                   agent_type::CPU));
    processor->handle(make_kfd_page_migrate_sample(kCpu0, kGpu2, 2048, 200, /*dev=*/kCpu0,
                                                   agent_type::CPU));

    processor->finalize_processing();

    auto j_opt = read_json_output();
    ASSERT_TRUE(j_opt.has_value());
    auto const& j = *j_opt;
    ASSERT_EQ(j["devices"].size(), 2u);

    bool saw_gpu1 = false;
    bool saw_gpu2 = false;
    for(auto const& dev : j["devices"])
    {
        auto        device_id = dev["device_id"].get<std::uint32_t>();
        auto const& h2d       = dev["migrations"]["host_to_device"];

        if(device_id == kGpu1)
        {
            EXPECT_EQ(h2d["count"], 1u);
            EXPECT_EQ(h2d["total_size_bytes"], 1024u);
            saw_gpu1 = true;
        }
        else if(device_id == kGpu2)
        {
            EXPECT_EQ(h2d["count"], 1u);
            EXPECT_EQ(h2d["total_size_bytes"], 2048u);
            saw_gpu2 = true;
        }
        else
        {
            ADD_FAILURE() << "unexpected GPU bucket id: " << device_id;
        }
    }

    EXPECT_TRUE(saw_gpu1);
    EXPECT_TRUE(saw_gpu2);
}

TEST_F(UnifiedMemoryProcessorTest, ExtractGpuNameResolvesOrFallsBack)
{
    rebuild_processor("gfx950", "");

    processor->handle(make_kfd_page_migrate_sample(kCpu0, kGpu1, 1024, 100, /*dev=*/0));

    processor->handle(make_kfd_page_migrate_sample(kCpu0, kGpu2, 1024, 100, /*dev=*/0));
    processor->finalize_processing();

    auto j_opt = read_json_output();
    ASSERT_TRUE(j_opt.has_value());
    auto const& j = *j_opt;
    ASSERT_EQ(j["devices"].size(), 2u);

    bool saw_resolved = false;
    bool saw_fallback = false;
    for(auto const& dev : j["devices"])
    {
        std::string name = dev["device_name"];
        if(dev["device_id"] == kGpu1)
        {
            EXPECT_THAT(name, HasSubstr("gfx950")) << "name=" << name;
            saw_resolved = true;
        }
        else if(dev["device_id"] == kGpu2)
        {
            EXPECT_THAT(name, ::testing::StartsWith("GPU 2")) << "name=" << name;
            saw_fallback = true;
        }
    }
    EXPECT_TRUE(saw_resolved);
    EXPECT_TRUE(saw_fallback);
}

TEST_F(UnifiedMemoryProcessorTest, AgentLookupThrowFallsBackSafely)
{
    auto sample =
        make_kfd_page_migrate_sample(kCpu0, kGpu1, /*size=*/1024,
                                     /*duration=*/100, /*device_id=*/42, agent_type::CPU);
    EXPECT_NO_THROW(processor->handle(sample));

    processor->finalize_processing();

    auto j_opt = read_json_output();
    ASSERT_TRUE(j_opt.has_value());
    auto const& j = *j_opt;
    ASSERT_EQ(j["devices"].size(), 1u);
    std::string name = j["devices"][0]["device_name"];
    EXPECT_THAT(name, HasSubstr("CPU 42"));
}

TEST_F(UnifiedMemoryProcessorTest, PidSuffixedPathsRegistered)
{
    processor->handle(make_kfd_page_migrate_sample(kCpu0, kGpu1, 1024, 100, /*dev=*/0));
    processor->finalize_processing();

    bool saw_txt  = false;
    bool saw_json = false;
    for(const auto& e : registered_files())
    {
        EXPECT_THAT(e.path, HasSubstr("unified_memory"));
        EXPECT_THAT(e.path, HasSubstr(std::to_string(kPid)));
        if(e.format == output_format::text)
        {
            EXPECT_THAT(e.path, ::testing::EndsWith(".txt"));
            saw_txt = true;
        }
        else if(e.format == output_format::json)
        {
            EXPECT_THAT(e.path, ::testing::EndsWith(".json"));
            saw_json = true;
        }
    }
    EXPECT_TRUE(saw_txt) << "text file not registered";
    EXPECT_TRUE(saw_json) << "json file not registered";
}

TEST_F(UnifiedMemoryProcessorTest, ExplicitOutputPathOverridesBackendDerivedPath)
{
    auto explicit_dir = tmp_dir + "/ump-explicit";
    ASSERT_FALSE(test_common::fs::exists(explicit_dir));
    ScopedEnv ump_output_path{ env_vars::UNIFIED_MEMORY_OUTPUT_PATH, explicit_dir };
    rebuild_processor();

    processor->handle(make_kfd_page_migrate_sample(kCpu0, kGpu1, 1024, 100, /*dev=*/0));
    processor->finalize_processing();

    bool saw_txt  = false;
    bool saw_json = false;
    for(const auto& e : registered_files())
    {
        EXPECT_THAT(e.path, ::testing::HasSubstr(explicit_dir));
        EXPECT_TRUE(test_common::fs::exists(e.path)) << "missing file: " << e.path;
        if(e.format == output_format::text) saw_txt = true;
        if(e.format == output_format::json) saw_json = true;
    }
    EXPECT_TRUE(saw_txt) << "text file not registered";
    EXPECT_TRUE(saw_json) << "json file not registered";
}

TEST_F(UnifiedMemoryProcessorTest, RelativeOutputPathResolvesFromPwd)
{
    const auto  relative_dir = std::string{ "ump-relative" };
    const auto* pwd          = getenv("PWD");
    const auto  base_dir =
        (pwd != nullptr) ? test_common::fs::path{ pwd } : test_common::fs::current_path();
    const auto expected_dir = (base_dir / relative_dir).string();
    test_common::fs::remove_all(expected_dir);
    ASSERT_FALSE(test_common::fs::exists(expected_dir));
    ScopedEnv ump_output_path{ env_vars::UNIFIED_MEMORY_OUTPUT_PATH, relative_dir };
    rebuild_processor();

    processor->handle(make_kfd_page_migrate_sample(kCpu0, kGpu1, 1024, 100, /*dev=*/0));
    processor->finalize_processing();

    bool saw_txt  = false;
    bool saw_json = false;
    for(const auto& e : registered_files())
    {
        EXPECT_THAT(e.path, ::testing::HasSubstr(expected_dir));
        EXPECT_TRUE(test_common::fs::exists(e.path)) << "missing file: " << e.path;
        if(e.format == output_format::text) saw_txt = true;
        if(e.format == output_format::json) saw_json = true;
    }
    EXPECT_TRUE(saw_txt) << "text file not registered";
    EXPECT_TRUE(saw_json) << "json file not registered";

    test_common::fs::remove_all(expected_dir);
}

TEST_F(UnifiedMemoryProcessorTest, ExplicitOutputPathCreatesNestedDirectories)
{
    auto nested_dir = tmp_dir + "/ump-nested/a/b/c";
    ASSERT_FALSE(test_common::fs::exists(nested_dir));
    ScopedEnv ump_output_path{ env_vars::UNIFIED_MEMORY_OUTPUT_PATH, nested_dir };
    rebuild_processor();

    processor->handle(make_kfd_page_migrate_sample(kCpu0, kGpu1, 1024, 100, /*dev=*/0));
    processor->finalize_processing();

    EXPECT_TRUE(test_common::fs::exists(nested_dir)) << "nested dir not created";

    bool saw_txt  = false;
    bool saw_json = false;
    for(const auto& e : registered_files())
    {
        EXPECT_THAT(e.path, ::testing::HasSubstr(nested_dir));
        EXPECT_TRUE(test_common::fs::exists(e.path)) << "missing file: " << e.path;
        if(e.format == output_format::text) saw_txt = true;
        if(e.format == output_format::json) saw_json = true;
    }
    EXPECT_TRUE(saw_txt) << "text file not registered";
    EXPECT_TRUE(saw_json) << "json file not registered";
}

TEST_F(UnifiedMemoryProcessorTest, FaultsOnlyEmitsOutput)
{
    processor->handle(make_kfd_page_fault_sample(/*agent=*/kGpu1, /*read=*/true));
    processor->handle(make_kfd_page_fault_sample(/*agent=*/kGpu1, /*read=*/false));
    processor->handle(make_kfd_page_fault_sample(/*agent=*/kGpu2, /*read=*/true));

    processor->finalize_processing();

    auto files = registered_files();
    EXPECT_EQ(files.size(), 2u);

    bool saw_txt  = false;
    bool saw_json = false;
    for(const auto& e : files)
    {
        EXPECT_TRUE(test_common::fs::exists(e.path)) << "missing file: " << e.path;
        if(e.format == output_format::text) saw_txt = true;
        if(e.format == output_format::json) saw_json = true;
    }
    EXPECT_TRUE(saw_txt);
    EXPECT_TRUE(saw_json);

    auto j_opt = read_json_output();
    ASSERT_TRUE(j_opt.has_value());
    auto const& j = *j_opt;
    EXPECT_EQ(j["summary"]["total_page_faults"], 3u);
    EXPECT_EQ(j["devices"].size(), 0u);
}

TEST_F(UnifiedMemoryProcessorTest, MalformedArgsStringSkipsEvent)
{
    // Co-feed one fault so finalize emits output and the malformed migrate is observable.
    processor->handle(make_kfd_page_fault_sample(/*agent=*/kGpu1, /*read=*/true));

    processor->handle(
        make_kfd_page_migrate_sample_raw_args("totally_wrong_format_no_markers_here"));

    processor->finalize_processing();

    auto j_opt = read_json_output();
    ASSERT_TRUE(j_opt.has_value());
    auto const& j = *j_opt;
    EXPECT_EQ(j["summary"]["total_page_faults"], 1u);
    EXPECT_EQ(j["devices"].size(), 0u)
        << "Malformed migrate event was incorrectly recorded as a device";
    auto const& triggers = j["summary"]["migration_triggers"];
    for(const auto& row : kTriggerTable)
    {
        EXPECT_EQ(triggers[row.json_key], 0u)
            << "Malformed migrate event incorrectly classified as trigger: "
            << row.json_key;
    }
}

TEST_F(UnifiedMemoryProcessorTest, TriggerTableCoversAllKfdNames)
{
    constexpr const char* kSentinelFedName = "SOMETHING_ELSE";
    for(const auto& row : kTriggerTable)
    {
        const char* name = (row.kfd_name != nullptr) ? row.kfd_name : kSentinelFedName;
        processor->handle(make_kfd_page_migrate_sample(kCpu0, kGpu1, /*size=*/1024,
                                                       /*duration=*/100, /*device_id=*/0,
                                                       agent_type::CPU, name));
    }

    processor->finalize_processing();

    auto j_opt = read_json_output();
    ASSERT_TRUE(j_opt.has_value());
    auto const& triggers = (*j_opt)["summary"]["migration_triggers"];
    for(const auto& row : kTriggerTable)
    {
        EXPECT_EQ(triggers[row.json_key], 1u)
            << "trigger key not incremented: " << row.json_key;
    }
}

TEST_F(UnifiedMemoryProcessorTest, JsonReportsXnackFlag)
{
    processor->handle(make_kfd_page_migrate_sample(kCpu0, kGpu1, 1024, 100, /*dev=*/0));
    processor->finalize_processing();

    auto j_opt = read_json_output();
    ASSERT_TRUE(j_opt.has_value());
    auto const& summary = (*j_opt)["summary"];
    ASSERT_TRUE(summary.contains("xnack_enabled"));
    ASSERT_TRUE(summary["xnack_enabled"].is_boolean());
    EXPECT_EQ(summary["xnack_enabled"].get<bool>(), expected_xnack_enabled);
}

TEST_F(UnifiedMemoryProcessorTest, FloatSanitizationProducesZeroSize)
{
    const std::vector<double> rejected_values = {
        std::numeric_limits<double>::quiet_NaN(),
        std::numeric_limits<double>::infinity(),
        -std::numeric_limits<double>::infinity(),
        -1.0,
        0.0,  // > 0 guard
        static_cast<double>(
            std::numeric_limits<std::uint64_t>::max()),  // = 2^64 (UB if cast)
    };

    for(double v : rejected_values)
        feed_h2d_migrate_with_value(v);

    processor->finalize_processing();

    auto j_opt = read_json_output();
    ASSERT_TRUE(j_opt.has_value());
    auto const& h2d = (*j_opt)["devices"][0]["migrations"]["host_to_device"];
    EXPECT_EQ(h2d["count"], rejected_values.size());
    EXPECT_EQ(h2d["total_size_bytes"], 0u)
        << "no rejected value should have contributed nonzero bytes";
    EXPECT_EQ(h2d["min_size_bytes"], 0u)
        << "sentinel not overwritten by add_migration(0)";
    EXPECT_EQ(h2d["max_size_bytes"], 0u);
}

TEST_F(UnifiedMemoryProcessorTest, FloatJustBelowBoundaryIsAccepted)
{
    static constexpr std::uint64_t kLargestExactDoubleUint =
        1ULL << std::numeric_limits<double>::digits;

    feed_h2d_migrate_with_value(static_cast<double>(kLargestExactDoubleUint));

    processor->finalize_processing();

    auto j_opt = read_json_output();
    ASSERT_TRUE(j_opt.has_value());
    auto const& h2d = (*j_opt)["devices"][0]["migrations"]["host_to_device"];
    EXPECT_EQ(h2d["count"], 1u);
    for(const char* k : { "total_size_bytes", "min_size_bytes", "max_size_bytes" })
    {
        ASSERT_TRUE(h2d[k].is_number_unsigned()) << "key=" << k;
        EXPECT_EQ(h2d[k].get<std::uint64_t>(), kLargestExactDoubleUint) << "key=" << k;
    }
}

TEST_F(UnifiedMemoryProcessorTest, NodeIdsExceedingUint32AreRejected)
{
    processor->handle(make_kfd_page_fault_sample(kGpu1, /*read=*/true));

    processor->handle(make_kfd_page_migrate_sample_raw_args(
        "0;;std::uint64_t;;start_address;;0x0;;"
        "1;;std::uint64_t;;end_address;;0x1000;;"
        "2;;string;;src_agent;;9999999999;;"  // > UINT32_MAX
        "3;;string;;dst_agent;;1;;"));

    processor->finalize_processing();

    auto j_opt = read_json_output();
    ASSERT_TRUE(j_opt.has_value());
    auto const& j = *j_opt;
    EXPECT_EQ(j["devices"].size(), 0u);
}

}  // namespace
