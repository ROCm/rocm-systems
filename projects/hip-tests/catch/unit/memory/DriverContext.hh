/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once
#include <memory>
#include <hip_test_common.hh>

#include <hip_test_context.hh>

class DriverContext {
 private:
  hipCtx_t ctx;
  hipDevice_t device;

 public:
  DriverContext();
  ~DriverContext();

  // Rule of three
  DriverContext(const DriverContext& other) = delete;
  DriverContext(DriverContext&& other) noexcept = delete;
};
