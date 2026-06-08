// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
#include "code_object_writer.h"
#include "gtest/gtest.h"
#include "mocks.h"
#include "nlohmann/json.hpp"
#include "pc_record_store.h"
#include "pc_sampling_feature.h"
#include "sdk_callbacks.h"

#include <rocprofiler-sdk/pc_sampling.h>
#include <rocprofiler-sdk/rocprofiler.h>

#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <system_error>
#include <vector>

using namespace rocprofiler_compute_tool;
namespace fs = std::filesystem;

// ---------------------------------------------------------------------------
// parse_pc_sampling_mode
// ---------------------------------------------------------------------------

TEST(parse_pc_sampling_mode_t, StochasticString)
{
    EXPECT_EQ(parse_pc_sampling_mode("stochastic"), PcSamplingMode::Stochastic);
}

TEST(parse_pc_sampling_mode_t, HostTrapString)
{
    EXPECT_EQ(parse_pc_sampling_mode("host_trap"), PcSamplingMode::HostTrap);
}

TEST(parse_pc_sampling_mode_t, EmptyStringIsDisabled)
{
    EXPECT_EQ(parse_pc_sampling_mode(""), PcSamplingMode::Disabled);
}

TEST(parse_pc_sampling_mode_t, UnknownStringIsDisabled)
{
    EXPECT_EQ(parse_pc_sampling_mode("garbage"), PcSamplingMode::Disabled);
}

// ---------------------------------------------------------------------------
// finalize() — JSON emission + source snapshot
// ---------------------------------------------------------------------------

namespace
{
// A self-contained collector that emits one code object with one instruction
// and reports one source path, so finalize() can be exercised without the real
// disassembler or the collector's mock translator.
class fake_collector_t : public pc_sampling_collector_t
{
public:
    explicit fake_collector_t(std::vector<std::string> source_paths)
        : m_source_paths(std::move(source_paths))
    {
    }

    void on_code_object_load(const rocprofiler_callback_tracing_code_object_load_data_t&) override
    {
    }

    void write(code_object_writer_t& writer) override
    {
        writer.start_code_obj(7);
        writer.start_symbol(symbol_t{"vecCopy", 0x10, 0x1000, 1});
        writer.write_instruction(instruction_t{"s_load_b64", "/src/k.hip:42", 0x1000, 0x10, 1});
        writer.end_symbol();
        writer.end_code_obj();
    }

    std::vector<std::string> collect_source_paths() override { return m_source_paths; }

private:
    std::vector<std::string> m_source_paths;
};

class test_pc_sampling_feature_t : public ::testing::Test
{
protected:
    void SetUp() override
    {
        std::error_code ec;
        const auto      base = fs::temp_directory_path(ec);
        ASSERT_FALSE(ec);
        const auto* info = ::testing::UnitTest::GetInstance()->current_test_info();
        m_tmp = base / (std::string("rpc_pcs_feature_") + info->test_case_name() + "_" + info->name());
        fs::remove_all(m_tmp, ec);
        ASSERT_TRUE(fs::create_directories(m_tmp, ec));
    }

    void TearDown() override
    {
        std::error_code ec;
        fs::remove_all(m_tmp, ec);
    }

    static void write_file(const fs::path& path, const std::string& content)
    {
        fs::create_directories(path.parent_path());
        std::ofstream out(path);
        out << content;
    }

    fs::path m_tmp;
};
}  // namespace

TEST_F(test_pc_sampling_feature_t, FinalizeWritesJsonAndSnapshotsSource)
{
    // A real source file the collector will report as referenced.
    const fs::path src = m_tmp / "src" / "k.hip";
    write_file(src, "int main() { return 0; }\n");

    const fs::path output  = m_tmp / "out" / "ps_file_code_obj_info.json";
    const fs::path ps_file = m_tmp / "out" / "ps_file_results.json";
    auto collector = std::make_shared<fake_collector_t>(std::vector<std::string>{src.string()});
    pc_sampling_feature_t feature(PcSamplingMode::Stochastic, /*interval=*/0, /*unit=*/"", output, ps_file, collector);

    feature.finalize();

    // (a) The code-object JSON is written and parseable.
    ASSERT_TRUE(fs::exists(output)) << "expected code-object JSON at " << output;
    std::ifstream  in(output);
    nlohmann::json json;
    ASSERT_NO_THROW(in >> json);
    ASSERT_EQ(json["code_objects"].size(), 1u);
    EXPECT_EQ(json["code_objects"][0]["id"], 7);

    // (b) The source file is snapshotted under <output parent>/code_obj_sources/.
    const fs::path snapshot = output.parent_path() / "code_obj_sources" / src.relative_path();
    EXPECT_TRUE(fs::exists(snapshot)) << "expected source snapshot at " << snapshot;
}

TEST_F(test_pc_sampling_feature_t, FinalizeResolvesRelativeOutputUnderCwd)
{
    // A relative output path resolves against the current directory; the
    // snapshot lands beside it under that same parent.
    const fs::path cwd_before = fs::current_path();
    fs::current_path(m_tmp);

    const fs::path src = m_tmp / "src" / "k.hip";
    write_file(src, "// src\n");

    auto collector = std::make_shared<fake_collector_t>(std::vector<std::string>{src.string()});
    pc_sampling_feature_t feature(PcSamplingMode::Stochastic,
                                  /*interval=*/0,
                                  /*unit=*/"",
                                  "out/ps_file_code_obj_info.json",
                                  "out/ps_file_results.json",
                                  collector);

    feature.finalize();

    EXPECT_TRUE(fs::exists(m_tmp / "out" / "ps_file_code_obj_info.json"));
    const fs::path snapshot = m_tmp / "out" / "code_obj_sources" / src.relative_path();
    EXPECT_TRUE(fs::exists(snapshot)) << "expected snapshot at " << snapshot;

    fs::current_path(cwd_before);
}

// ---------------------------------------------------------------------------
// configure() — SDK service wiring
// ---------------------------------------------------------------------------

namespace
{
// Build a PC sampling configuration the mock can advertise for an agent.
rocprofiler_pc_sampling_configuration_t make_config(rocprofiler_pc_sampling_method_t method,
                                                    rocprofiler_pc_sampling_unit_t   unit)
{
    rocprofiler_pc_sampling_configuration_t config{};
    config.size         = sizeof(rocprofiler_pc_sampling_configuration_t);
    config.method       = method;
    config.unit         = unit;
    config.min_interval = 1;
    config.max_interval = 1000;
    config.flags        = ROCPROFILER_PC_SAMPLING_CONFIGURATION_FLAGS_NONE;
    return config;
}
}  // namespace

TEST(pc_sampling_feature_configure_t, MatchingStochasticConfigConfiguresService)
{
    MockSdkWrapper sdk;
    sdk.set_pc_sampling_agent(/*agent_handle=*/5);
    sdk.add_pc_sampling_config(make_config(ROCPROFILER_PC_SAMPLING_METHOD_STOCHASTIC,
                                           ROCPROFILER_PC_SAMPLING_UNIT_CYCLES));

    pc_sampling_feature_t feature(PcSamplingMode::Stochastic,
                                  /*interval=*/256,
                                  /*unit=*/"",
                                  "code_obj.json",
                                  "ps_file.json");

    rocprofiler_context_id_t ctx{42};
    int                      user_data = 0;
    const bool               ok        = feature.configure(ctx, sdk, &user_data);

    EXPECT_TRUE(ok);

    // A delivery buffer is created with the free PC sampling trampoline + the
    // exact user_data pointer that was passed in.
    ASSERT_EQ(sdk.get_create_buffer_info().size(), 1u);
    EXPECT_EQ(sdk.get_create_buffer_info()[0].callback, &pc_sampling_buffer_callback);
    EXPECT_EQ(sdk.get_create_buffer_info()[0].callback_data, &user_data);

    // The service is configured for the stochastic method on the matched agent.
    ASSERT_EQ(sdk.get_configure_pc_sampling_service_info().size(), 1u);
    const auto& cfg = sdk.get_configure_pc_sampling_service_info()[0];
    EXPECT_EQ(cfg.method, ROCPROFILER_PC_SAMPLING_METHOD_STOCHASTIC);
    EXPECT_EQ(cfg.agent, 5u);
    EXPECT_EQ(cfg.context, 42u);
    EXPECT_EQ(cfg.interval, 256u);
}

TEST(pc_sampling_feature_configure_t, DisabledModeConfiguresNothing)
{
    MockSdkWrapper sdk;
    sdk.set_pc_sampling_agent(/*agent_handle=*/5);
    sdk.add_pc_sampling_config(make_config(ROCPROFILER_PC_SAMPLING_METHOD_STOCHASTIC,
                                           ROCPROFILER_PC_SAMPLING_UNIT_CYCLES));

    // Default-constructed feature is Disabled.
    pc_sampling_feature_t feature;

    rocprofiler_context_id_t ctx{1};
    const bool               ok = feature.configure(ctx, sdk, nullptr);

    EXPECT_FALSE(ok);
    EXPECT_TRUE(sdk.get_create_buffer_info().empty());
    EXPECT_TRUE(sdk.get_configure_pc_sampling_service_info().empty());
}

TEST(pc_sampling_feature_configure_t, NoMatchingMethodReturnsFalse)
{
    MockSdkWrapper sdk;
    sdk.set_pc_sampling_agent(/*agent_handle=*/5);
    // Agent only advertises host_trap, but stochastic is requested.
    sdk.add_pc_sampling_config(
        make_config(ROCPROFILER_PC_SAMPLING_METHOD_HOST_TRAP, ROCPROFILER_PC_SAMPLING_UNIT_TIME));

    pc_sampling_feature_t feature(PcSamplingMode::Stochastic,
                                  /*interval=*/256,
                                  /*unit=*/"",
                                  "code_obj.json",
                                  "ps_file.json");

    rocprofiler_context_id_t ctx{1};
    const bool               ok = feature.configure(ctx, sdk, nullptr);

    EXPECT_FALSE(ok);
    EXPECT_TRUE(sdk.get_create_buffer_info().empty());
    EXPECT_TRUE(sdk.get_configure_pc_sampling_service_info().empty());
}

TEST(pc_sampling_feature_configure_t, NoAgentReturnsFalse)
{
    MockSdkWrapper sdk;  // no agent injected.

    pc_sampling_feature_t feature(PcSamplingMode::HostTrap,
                                  /*interval=*/0,
                                  /*unit=*/"",
                                  "code_obj.json",
                                  "ps_file.json");

    rocprofiler_context_id_t ctx{1};
    EXPECT_FALSE(feature.configure(ctx, sdk, nullptr));
    EXPECT_TRUE(sdk.get_configure_pc_sampling_service_info().empty());
}

TEST(pc_sampling_feature_configure_t, SdkQueryFailureDegradesGracefully)
{
    // Simulate a runtime where PC sampling is unavailable: the SDK seam throws
    // (mirrors ROCPROFILER_STATUS_ERROR_NOT_IMPLEMENTED from
    // rocprofiler_query_pc_sampling_agent_configurations). configure() must NOT
    // propagate the exception (it runs from tool_init behind the SDK's C
    // callbacks); it must report false so the profiled application keeps running.
    MockSdkWrapper sdk;
    sdk.set_pc_sampling_agent(/*agent_handle=*/5);
    sdk.set_query_configs_should_throw(true);

    pc_sampling_feature_t feature(PcSamplingMode::Stochastic,
                                  /*interval=*/256,
                                  /*unit=*/"",
                                  "code_obj.json",
                                  "ps_file.json");

    rocprofiler_context_id_t ctx{1};
    bool                     ok = true;
    EXPECT_NO_THROW({ ok = feature.configure(ctx, sdk, nullptr); });
    EXPECT_FALSE(ok);
    EXPECT_TRUE(sdk.get_create_buffer_info().empty());
    EXPECT_TRUE(sdk.get_configure_pc_sampling_service_info().empty());
}

// ---------------------------------------------------------------------------
// flush() — drains the configured delivery buffer; no-op when none configured.
// ---------------------------------------------------------------------------

TEST(pc_sampling_feature_flush_t, FlushDrainsConfiguredBuffer)
{
    MockSdkWrapper sdk;
    sdk.set_pc_sampling_agent(/*agent_handle=*/5);
    sdk.add_pc_sampling_config(make_config(ROCPROFILER_PC_SAMPLING_METHOD_STOCHASTIC,
                                           ROCPROFILER_PC_SAMPLING_UNIT_CYCLES));

    pc_sampling_feature_t feature(PcSamplingMode::Stochastic,
                                  /*interval=*/256,
                                  /*unit=*/"",
                                  "code_obj.json",
                                  "ps_file.json");

    rocprofiler_context_id_t ctx{42};
    int                      user_data = 0;
    ASSERT_TRUE(feature.configure(ctx, sdk, &user_data));
    ASSERT_EQ(sdk.get_create_buffer_info().size(), 1u);

    feature.flush(sdk);

    // The exact buffer id the mock handed back from create_buffer is flushed.
    ASSERT_EQ(sdk.get_flushed_buffers().size(), 1u);
    EXPECT_EQ(sdk.get_flushed_buffers()[0], sdk.get_configure_pc_sampling_service_info()[0].buffer);
}

TEST(pc_sampling_feature_flush_t, FlushWithNoMatchingAgentIsNoOp)
{
    MockSdkWrapper sdk;
    sdk.set_pc_sampling_agent(/*agent_handle=*/5);
    // Agent advertises only host_trap, but stochastic is requested -> no match,
    // so configure() sets up no buffer.
    sdk.add_pc_sampling_config(
        make_config(ROCPROFILER_PC_SAMPLING_METHOD_HOST_TRAP, ROCPROFILER_PC_SAMPLING_UNIT_TIME));

    pc_sampling_feature_t feature(PcSamplingMode::Stochastic,
                                  /*interval=*/256,
                                  /*unit=*/"",
                                  "code_obj.json",
                                  "ps_file.json");

    rocprofiler_context_id_t ctx{1};
    ASSERT_FALSE(feature.configure(ctx, sdk, nullptr));

    feature.flush(sdk);

    EXPECT_TRUE(sdk.get_flushed_buffers().empty());
}

TEST(pc_sampling_feature_flush_t, FlushOnDisabledFeatureIsNoOp)
{
    MockSdkWrapper sdk;

    // Default-constructed feature is Disabled and was never configured.
    pc_sampling_feature_t feature;

    feature.flush(sdk);

    EXPECT_TRUE(sdk.get_flushed_buffers().empty());
}

// ---------------------------------------------------------------------------
// on_pc_sample_records() — record routing into the store (observed via the
// ps_file JSON written by finalize()).
// ---------------------------------------------------------------------------

TEST_F(test_pc_sampling_feature_t, OnRecordsRoutesStochasticAndHostTrap)
{
    const fs::path output    = m_tmp / "code_obj.json";
    const fs::path ps_file   = m_tmp / "ps_file_results.json";
    auto           collector = std::make_shared<fake_collector_t>(std::vector<std::string>{});
    pc_sampling_feature_t feature(PcSamplingMode::Stochastic, /*interval=*/0, /*unit=*/"", output, ps_file, collector);

    // One stochastic and one host_trap sample, each behind its own header.
    rocprofiler_pc_sampling_record_stochastic_v0_t stochastic{};
    stochastic.size                  = sizeof(stochastic);
    stochastic.pc.code_object_id     = 11;
    stochastic.pc.code_object_offset = 0x40;
    stochastic.wave_issued           = 1;
    stochastic.dispatch_id           = 7;

    rocprofiler_pc_sampling_record_host_trap_v0_t host_trap{};
    host_trap.size                  = sizeof(host_trap);
    host_trap.pc.code_object_id     = 22;
    host_trap.pc.code_object_offset = 0x80;
    host_trap.dispatch_id           = 9;

    rocprofiler_record_header_t stochastic_header{};
    stochastic_header.category = ROCPROFILER_BUFFER_CATEGORY_PC_SAMPLING;
    stochastic_header.kind     = ROCPROFILER_PC_SAMPLING_RECORD_STOCHASTIC_V0_SAMPLE;
    stochastic_header.payload  = &stochastic;

    rocprofiler_record_header_t host_trap_header{};
    host_trap_header.category = ROCPROFILER_BUFFER_CATEGORY_PC_SAMPLING;
    host_trap_header.kind     = ROCPROFILER_PC_SAMPLING_RECORD_HOST_TRAP_V0_SAMPLE;
    host_trap_header.payload  = &host_trap;

    rocprofiler_record_header_t* headers[2] = {&stochastic_header, &host_trap_header};
    feature.on_pc_sample_records(headers, 2);

    feature.finalize();

    ASSERT_TRUE(fs::exists(ps_file));
    std::ifstream  in(ps_file);
    nlohmann::json json;
    ASSERT_NO_THROW(in >> json);

    const auto& buffer_records = json["rocprofiler-sdk-tool"][0]["buffer_records"];
    const auto& stochastic_arr = buffer_records["pc_sample_stochastic"];
    const auto& host_trap_arr  = buffer_records["pc_sample_host_trap"];

    ASSERT_EQ(stochastic_arr.size(), 1u);
    ASSERT_EQ(host_trap_arr.size(), 1u);
    EXPECT_EQ(stochastic_arr[0]["record"]["pc"]["code_object_id"], 11u);
    EXPECT_EQ(stochastic_arr[0]["record"]["pc"]["code_object_offset"], 0x40u);
    EXPECT_EQ(stochastic_arr[0]["record"]["dispatch_id"], 7u);
    EXPECT_EQ(host_trap_arr[0]["record"]["pc"]["code_object_id"], 22u);
    EXPECT_EQ(host_trap_arr[0]["record"]["dispatch_id"], 9u);
}
