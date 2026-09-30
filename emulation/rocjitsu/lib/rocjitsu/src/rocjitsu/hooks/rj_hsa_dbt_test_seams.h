// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file rj_hsa_dbt_test_seams.h
/// @brief Internal test seams of the DBT HSA hook.
///
/// @details The hook reads KFD topology from a fixed pair of sysfs roots, which
/// unit tests cannot populate. This declaration lets tests redirect that
/// lookup at a synthetic tree. It is defined in rj_hsa_dbt_hooks.cpp with
/// hidden visibility, so the production `librocjitsu_hooks.so` exports only its
/// HSA tool entry points (`OnLoad` and `OnUnload`). The C entry point that
/// drives the seam lives in rj_hsa_dbt_test_seams.cpp, which is linked into the
/// test-only `librocjitsu_hooks_testing.so` and nothing else.

#pragma once

namespace rocjitsu::hooks {

/// @brief Set or clear the synthetic KFD topology root used by hook unit tests.
///
/// @param root Directory holding `<node_id>/gpu_id` files, or nullptr/empty to
/// restore the real sysfs lookup.
void set_topology_nodes_root_for_test(const char *root);

} // namespace rocjitsu::hooks
