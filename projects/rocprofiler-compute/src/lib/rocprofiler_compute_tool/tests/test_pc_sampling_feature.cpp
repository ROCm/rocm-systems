// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
#include "test_pc_sampling_feature.h"

#include <set>

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

TEST_F(TestPcSamplingFeature, Finalize_WritesCollectorAndSnapshotsCollectorSourcePaths)
{
    const std::set<std::filesystem::path> source_paths = {
        "/tmp/project/header.h",
        "kernel.cpp",
    };
    m_collector->set_source_paths(source_paths);
    m_collector->set_has_code_objects(true);
    auto feature = create_feature();

    feature.finalize();

    EXPECT_EQ(m_collector->finalize_count, 1);
    const auto& flush_calls = m_writer->get_flush_calls();
    ASSERT_EQ(flush_calls.size(), 1);
    EXPECT_EQ(flush_calls[0], m_code_object_info_path);

    const auto& snapshot_calls = m_snapshotter->get_snapshot_calls();
    ASSERT_EQ(snapshot_calls.size(), 1);
    EXPECT_EQ(snapshot_calls[0].source_paths, source_paths);
    EXPECT_EQ(snapshot_calls[0].destination_root, m_source_snapshot_path);
}

TEST_F(TestPcSamplingFeature, Finalize_WithNoCodeObjects_DoesNotWriteFileOrSnapshotSources)
{
    auto feature = create_feature();

    feature.finalize();

    EXPECT_EQ(m_collector->finalize_count, 1);
    EXPECT_TRUE(m_writer->get_flush_calls().empty());
    EXPECT_TRUE(m_snapshotter->get_snapshot_calls().empty());
}
