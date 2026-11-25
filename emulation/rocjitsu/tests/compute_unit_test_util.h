// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "util/diagnostic.h"

#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <utility>

namespace rocjitsu::test {

/// @brief Initialize an expected-valid CU, retaining its rejection diagnostic.
/// @details In SetUp, call with ASSERT_NO_FATAL_FAILURE to stop dependent setup.
/// Plain fixture constructors must return on a null CU, and their consumers must
/// check the CU/wave or HasFatalFailure() before using the fixture.
inline void initialize_compute_unit(std::unique_ptr<amdgpu::ComputeUnitCore> &compute_unit,
                                    std::string name, const amdgpu::ComputeUnitCore::Config &config,
                                    amdgpu::GpuMemory *memory, amdgpu::L2Cache *l2,
                                    simdojo::ExecMode mode = simdojo::ExecMode::FUNCTIONAL) {
  util::StringDiagnostic diagnostic;
  compute_unit = amdgpu::ComputeUnitCore::create(std::move(name), config, memory, l2, mode,
                                                 diagnostic.emitter());
  ASSERT_NE(compute_unit, nullptr) << diagnostic.message();
}

} // namespace rocjitsu::test
