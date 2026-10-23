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

    EXPECT_EQ(m_collector->finalize_count, 0);
    EXPECT_TRUE(m_writer->get_flush_calls().empty());
    EXPECT_TRUE(m_snapshotter->get_snapshot_calls().empty());
}

TEST_F(TestPcSamplingFeature, KernelRegistration_ForwardsNormalizedSymbolKey)
{
    auto feature = create_feature();
    rocprofiler_callback_tracing_code_object_kernel_symbol_register_data_t payload = {};
    payload.kernel_id                                                              = 7;
    payload.code_object_id                                                         = 2;
    payload.kernel_name                                                            = "_Z3fooi.kd";
    feature.on_kernel_symbol_register(payload);
    payload.kernel_name = nullptr;
    feature.on_kernel_symbol_register(payload);

    ASSERT_EQ(m_collector->symbol_registrations.size(), 2);
    EXPECT_EQ(m_collector->symbol_registrations[0].kernel_id, 7);
    EXPECT_EQ(m_collector->symbol_registrations[0].code_object_id, 2);
    EXPECT_EQ(m_collector->symbol_registrations[0].symbol_name, "_Z3fooi");
    EXPECT_TRUE(m_collector->symbol_registrations[1].symbol_name.empty());
}

TEST_F(TestPcSamplingFeature, DispatchRecords_FinalizesAllUniqueIdsRegardlessOfSelection)
{
    m_collector->set_has_code_objects(true);
    m_tool_data.target_kernel_ids.insert(7);
    for (const auto kernel_id : {7, 8, 7})
    {
        dispatch_record_t dispatch = {};
        dispatch.kernel_id         = kernel_id;
        m_tool_data.dispatch_records.push_back(dispatch);
    }
    auto feature = create_feature();
    feature.write(m_tool_data);
    EXPECT_EQ(m_collector->last_dispatched_kernel_ids, (std::set<uint64_t>{7, 8}));
}

TEST_F(TestPcSamplingFeature, LoadedObjectsWithoutDispatches_FlushesSkeletonAndEmptySnapshot)
{
    m_collector->set_has_code_objects(true);
    auto feature = create_feature();
    feature.write(m_tool_data);
    EXPECT_EQ(m_collector->finalize_count, 1);
    EXPECT_TRUE(m_collector->last_dispatched_kernel_ids.empty());
    EXPECT_TRUE(m_writer->empty());
    ASSERT_EQ(m_writer->get_flush_calls().size(), 1);
    EXPECT_EQ(m_writer->get_flush_calls()[0], m_code_object_info_path);
    ASSERT_EQ(m_snapshotter->get_snapshot_calls().size(), 1);
    EXPECT_TRUE(m_snapshotter->get_snapshot_calls()[0].source_paths.empty());
}

TEST_F(TestPcSamplingFeature, DisabledFeature_WriteDoesNothing)
{
    m_collector->set_has_code_objects(true);
    pc_sampling_feature_t feature{PcSamplingMode::Disabled,
                                  m_code_object_info_path,
                                  m_source_snapshot_path,
                                  m_collector,
                                  m_snapshotter,
                                  m_writer};
    feature.write(m_tool_data);
    EXPECT_FALSE(feature.enabled());
    EXPECT_EQ(m_collector->finalize_count, 0);
    EXPECT_TRUE(m_writer->get_flush_calls().empty());
    EXPECT_TRUE(m_snapshotter->get_snapshot_calls().empty());
}
