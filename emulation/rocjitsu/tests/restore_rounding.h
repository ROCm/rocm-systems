// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <cfenv>

namespace rocjitsu::test {

class RestoreRounding {
public:
  RestoreRounding() : saved_(std::fegetround()) {}
  ~RestoreRounding() { std::fesetround(saved_); }
  int saved() const { return saved_; }

private:
  int saved_;
};

} // namespace rocjitsu::test
