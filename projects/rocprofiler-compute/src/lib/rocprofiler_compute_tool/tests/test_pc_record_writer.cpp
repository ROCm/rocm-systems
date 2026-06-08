// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
#include "gtest/gtest.h"
#include "nlohmann/json.hpp"
#include "pc_record_store.h"
#include "pc_record_writer.h"

#include <filesystem>
#include <fstream>
#include <string>
#include <system_error>
#include <vector>

using namespace rocprofiler_compute_tool;
namespace fs = std::filesystem;

namespace
{
class test_pc_record_writer_t : public ::testing::Test
{
protected:
    void SetUp() override
    {
        std::error_code ec;
        const auto      base = fs::temp_directory_path(ec);
        ASSERT_FALSE(ec);
        const auto* info = ::testing::UnitTest::GetInstance()->current_test_info();
        m_tmp = base / (std::string("rpc_pcs_writer_") + info->test_case_name() + "_" + info->name());
        fs::remove_all(m_tmp, ec);
        ASSERT_TRUE(fs::create_directories(m_tmp, ec));
    }

    void TearDown() override
    {
        std::error_code ec;
        fs::remove_all(m_tmp, ec);
    }

    static nlohmann::json read_json(const fs::path& path)
    {
        std::ifstream  in(path);
        nlohmann::json json;
        in >> json;
        return json;
    }

    fs::path m_tmp;
};
}  // namespace

TEST_F(test_pc_record_writer_t, WritesStochasticRecordWithMetadata)
{
    pc_record_store_t  store;
    pc_sample_record_t record{};
    record.code_object_id     = 7;
    record.code_object_offset = 0x1000;
    record.stall_reason       = "NO_INSTRUCTION_AVAILABLE";
    record.wave_issued        = 1;
    record.inst_type          = "VALU";
    record.dispatch_id        = 3;
    store.add_stochastic(record);

    pc_sample_strings_t strings;
    strings.pc_sample_instructions       = {"s_load_b64"};
    strings.pc_sample_comments           = {"/src/k.hip:42"};
    strings.offset_to_index[{7, 0x1000}] = 0;

    std::vector<pc_kernel_symbol_t> kernel_symbols = {pc_kernel_symbol_t{7, "vecCopy"}};

    nlohmann::json::array_t code_objects = nlohmann::json::array();
    code_objects.push_back(nlohmann::json::object({{"id", 7}}));

    pc_record_writer_t writer;
    writer.write(store, strings, kernel_symbols, code_objects);

    const fs::path out = m_tmp / "ps_file_results.json";
    writer.flush(out);
    ASSERT_TRUE(fs::exists(out));

    const nlohmann::json json = read_json(out);
    ASSERT_TRUE(json.contains("rocprofiler-sdk-tool"));
    ASSERT_EQ(json["rocprofiler-sdk-tool"].size(), 1u);
    const auto& entry = json["rocprofiler-sdk-tool"][0];

    // strings
    ASSERT_EQ(entry["strings"]["pc_sample_instructions"].size(), 1u);
    EXPECT_EQ(entry["strings"]["pc_sample_instructions"][0], "s_load_b64");
    ASSERT_EQ(entry["strings"]["pc_sample_comments"].size(), 1u);
    EXPECT_EQ(entry["strings"]["pc_sample_comments"][0], "/src/k.hip:42");

    // kernel_symbols
    ASSERT_EQ(entry["kernel_symbols"].size(), 1u);
    EXPECT_EQ(entry["kernel_symbols"][0]["code_object_id"], 7u);
    EXPECT_EQ(entry["kernel_symbols"][0]["formatted_kernel_name"], "vecCopy");

    // code_objects
    ASSERT_EQ(entry["code_objects"].size(), 1u);
    EXPECT_EQ(entry["code_objects"][0]["id"], 7);

    // buffer_records.pc_sample_stochastic[0]
    const auto& stochastic = entry["buffer_records"]["pc_sample_stochastic"];
    ASSERT_EQ(stochastic.size(), 1u);
    EXPECT_EQ(stochastic[0]["record"]["pc"]["code_object_id"], 7u);
    EXPECT_EQ(stochastic[0]["record"]["pc"]["code_object_offset"], 0x1000u);
    EXPECT_EQ(stochastic[0]["record"]["snapshot"]["stall_reason"], "NO_INSTRUCTION_AVAILABLE");
    EXPECT_EQ(stochastic[0]["record"]["wave_issued"], 1);
    EXPECT_EQ(stochastic[0]["record"]["dispatch_id"], 3u);
    // inst_index resolved from offset_to_index sibling of "record".
    EXPECT_EQ(stochastic[0]["inst_index"], 0);

    // host_trap present and empty.
    ASSERT_TRUE(entry["buffer_records"].contains("pc_sample_host_trap"));
    EXPECT_TRUE(entry["buffer_records"]["pc_sample_host_trap"].is_array());
    EXPECT_TRUE(entry["buffer_records"]["pc_sample_host_trap"].empty());
}

TEST_F(test_pc_record_writer_t, EmptyStoreEmitsBothArraysEmpty)
{
    pc_record_store_t   store;  // empty
    pc_sample_strings_t strings;

    pc_record_writer_t writer;
    writer.write(store, strings, {}, nlohmann::json::array());

    const fs::path out = m_tmp / "ps_file_results.json";
    writer.flush(out);

    const nlohmann::json json  = read_json(out);
    const auto&          entry = json["rocprofiler-sdk-tool"][0];

    ASSERT_TRUE(entry["buffer_records"].contains("pc_sample_stochastic"));
    ASSERT_TRUE(entry["buffer_records"].contains("pc_sample_host_trap"));
    EXPECT_TRUE(entry["buffer_records"]["pc_sample_stochastic"].is_array());
    EXPECT_TRUE(entry["buffer_records"]["pc_sample_host_trap"].is_array());
    EXPECT_TRUE(entry["buffer_records"]["pc_sample_stochastic"].empty());
    EXPECT_TRUE(entry["buffer_records"]["pc_sample_host_trap"].empty());
}
