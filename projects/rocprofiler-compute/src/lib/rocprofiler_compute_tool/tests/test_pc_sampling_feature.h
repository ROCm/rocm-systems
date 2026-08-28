// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
#pragma once

#include "gtest/gtest.h"
#include "mocks.h"
#include "pc_sampling_feature.h"

#include <filesystem>
#include <memory>

class TestPcSamplingFeature : public ::testing::Test
{
protected:
    void SetUp() override;

    rocprofiler_compute_tool::pc_sampling_feature_t create_feature();

    std::shared_ptr<MockPcSamplingCollector> m_collector;
    std::shared_ptr<MockSourceSnapshotter>   m_snapshotter;
    std::shared_ptr<MockCodeObjectWriter>    m_writer;
    std::filesystem::path                    m_code_object_info_path;
    std::filesystem::path                    m_source_snapshot_path;
};
