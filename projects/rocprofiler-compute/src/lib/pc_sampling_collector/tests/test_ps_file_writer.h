// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
#pragma once

#include "gtest/gtest.h"
#include "pc_sampling_record.h"
#include "ps_file_writer.h"

#include <filesystem>
#include <optional>

class test_ps_file_writer_t : public ::testing::Test
{
protected:
    rocprofiler_compute_tool::ps_file_writer_json_t m_writer;
};
