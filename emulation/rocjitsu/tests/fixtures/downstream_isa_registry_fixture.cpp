// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file downstream_isa_registry_fixture.cpp
/// @brief Exported observations of a source-integrated downstream registry.

#include "rocjitsu/base/rj_compiler.h"
#include "rocjitsu/isa/target_registry.h"

#include <cstddef>

extern "C" RJ_API_EXPORT size_t rj_test_downstream_target_count() {
  return rocjitsu::default_isa_target_registry().targets().size();
}

extern "C" RJ_API_EXPORT bool rj_test_downstream_has_target(const char *id) {
  return id != nullptr && rocjitsu::default_isa_target_registry().find(id) != nullptr;
}
