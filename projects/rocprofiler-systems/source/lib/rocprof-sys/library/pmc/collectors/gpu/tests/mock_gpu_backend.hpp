// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "backends/amd_smi/device_backend.hpp"

#include <gmock/gmock.h>

#include <cstdint>

namespace rocprofsys::backends::amd_smi::testing
{

struct mock_gpu_backend
{
    MOCK_METHOD(asic_info, get_gpu_asic_info, (), (const));
    MOCK_METHOD(metrics, get_gpu_metrics, (), (const));
    MOCK_METHOD(std::uint64_t, get_memory_usage, (), (const));

    MOCK_METHOD(std::uint64_t, get_raw_sdma_usage, (), (const));
    MOCK_METHOD(bool, is_sdma_supported, (), (const, noexcept));
};

}  // namespace rocprofsys::backends::amd_smi::testing
