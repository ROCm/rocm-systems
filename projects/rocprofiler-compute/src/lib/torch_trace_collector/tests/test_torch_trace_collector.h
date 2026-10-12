// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#pragma once

#include <gtest/gtest.h>

class TorchTraceCollectorTest : public ::testing::Test
{
protected:
    void SetUp() override;
    void TearDown() override;
};
