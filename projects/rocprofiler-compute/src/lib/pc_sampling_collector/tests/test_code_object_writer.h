// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
#pragma once

#include "code_object_writer.h"
#include "gtest/gtest.h"

class test_code_object_writer_t : public ::testing::Test
{
protected:
    void SetUp() override;

    rocm_compute::code_object_writer_json_t m_writer;
};
