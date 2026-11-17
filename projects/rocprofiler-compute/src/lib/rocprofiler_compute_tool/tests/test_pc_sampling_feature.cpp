// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
#include "test_pc_sampling_feature.h"

#include <set>
#include <stdexcept>
#include <string>
#include <utility>

using namespace rocprofiler_compute_tool;

void TestPcSamplingFeature::SetUp()
{
    m_collector   = std::make_shared<MockPcSamplingCollector>();
    m_snapshotter = std::make_shared<MockSourceSnapshotter>();
    m_writer      = std::make_shared<MockCodeObjectWriter>();

    m_code_object_info_path = "code_obj_info.json";
    m_source_snapshot_path  = "src";
}

pc_sampling_feature_t TestPcSamplingFeature::create_feature()
{
    return pc_sampling_feature_t{PcSamplingMode::HostTrap,
                                 m_code_object_info_path,
                                 m_source_snapshot_path,
                                 m_collector,
                                 m_snapshotter,
                                 m_writer};
}

TEST_F(TestPcSamplingFeature, OnCodeObjectLoad_ForwardsToCollector)
{
    auto feature = create_feature();

    rocprofiler_callback_tracing_code_object_load_data_t payload = {};
    feature.on_code_object_load(payload);

    EXPECT_EQ(m_collector->load_count, 1);
}

TEST_F(TestPcSamplingFeature, Write_WritesCollectorAndSnapshotsCollectorSourcePaths)
{
    const std::set<std::filesystem::path> source_paths = {
        "/tmp/project/header.h",
        "kernel.cpp",
    };
    m_collector->set_source_paths(source_paths);
    m_collector->set_has_code_objects(true);
    auto feature = create_feature();

    feature.write(m_tool_data);

    EXPECT_EQ(m_collector->finalize_count, 1);
    const auto& flush_calls = m_writer->get_flush_calls();
    ASSERT_EQ(flush_calls.size(), 1);
    EXPECT_EQ(flush_calls[0], m_code_object_info_path);

    const auto& snapshot_calls = m_snapshotter->get_snapshot_calls();
    ASSERT_EQ(snapshot_calls.size(), 1);
    EXPECT_EQ(snapshot_calls[0].source_paths, source_paths);
    EXPECT_EQ(snapshot_calls[0].destination_root, m_source_snapshot_path);
}

TEST_F(TestPcSamplingFeature, Write_WithNoCodeObjects_DoesNotWriteFileOrSnapshotSources)
{
    auto feature = create_feature();

    feature.write(m_tool_data);

    EXPECT_EQ(m_collector->finalize_count, 1);
    EXPECT_TRUE(m_writer->get_flush_calls().empty());
    EXPECT_TRUE(m_snapshotter->get_snapshot_calls().empty());
}

TEST_F(TestPcSamplingFeature, DispatchRecords_FinalizesUniqueSymbolKeysRegardlessOfSelection)
{
    m_tool_data.kernel_symbols[7].code_object_id = 1;
    m_tool_data.kernel_symbols[7].mangled_name   = "_Z3fooi";
    m_tool_data.kernel_symbols[8].code_object_id = 2;
    m_tool_data.kernel_symbols[8].mangled_name   = "_Z3barv";
    m_tool_data.target_kernel_ids.insert(7);
    for (const auto kernel_id : {7, 8, 7})
    {
        dispatch_record_t dispatch = {};
        dispatch.kernel_id         = kernel_id;
        m_tool_data.dispatch_records.push_back(dispatch);
    }
    auto feature = create_feature();

    feature.write(m_tool_data);

    const std::set<std::pair<uint64_t, std::string>> expected = {{1, "_Z3fooi"}, {2, "_Z3barv"}};
    EXPECT_EQ(m_collector->last_dispatched_symbols, expected);
}

TEST_F(TestPcSamplingFeature, DispatchWithoutSymbolRecord_Throws)
{
    dispatch_record_t dispatch = {};
    dispatch.kernel_id         = 7;
    m_tool_data.dispatch_records.push_back(dispatch);
    auto feature = create_feature();

    EXPECT_THROW(feature.write(m_tool_data), std::runtime_error);
    EXPECT_EQ(m_collector->finalize_count, 0);
}
