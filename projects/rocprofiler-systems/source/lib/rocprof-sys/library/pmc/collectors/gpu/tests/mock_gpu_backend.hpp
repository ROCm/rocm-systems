// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "backends/amd_smi/backend.hpp"
#include "backends/amd_smi/gpu_types.hpp"

#include <gmock/gmock.h>

#include <cstdint>

namespace rocprofsys::backends::amd_smi::testing
{

using gpu::asic_info;
using gpu::metrics;

struct mock_gpu_backend
{
    MOCK_METHOD(asic_info, get_gpu_asic_info, (), (const));
    MOCK_METHOD(metrics, get_metrics, (), (const));
    MOCK_METHOD(std::uint64_t, get_memory_usage, (), (const));
    MOCK_METHOD(std::int64_t, get_hotspot_temperature, (), (const));
    MOCK_METHOD(std::int64_t, get_edge_temperature, (), (const));
    MOCK_METHOD(std::uint64_t, get_raw_sdma_usage, (), (const));
    MOCK_METHOD(bool, probe_sdma_gpu_support, (), (const, noexcept));
};

}  // namespace rocprofsys::backends::amd_smi::testing
