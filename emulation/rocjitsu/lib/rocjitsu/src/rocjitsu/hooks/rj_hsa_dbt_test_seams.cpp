// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file rj_hsa_dbt_test_seams.cpp
/// @brief Exported C entry points for the DBT HSA hook's test seams.
///
/// @details Linked only into `librocjitsu_hooks_testing.so`, the variant of the
/// hook that unit tests load. The production `librocjitsu_hooks.so` is built
/// from the same objects without this translation unit, so it exports only
/// `OnLoad` and `OnUnload`.

#include "rocjitsu/hooks/rj_hsa_dbt_test_seams.h"

#if defined(__GNUC__) || defined(__clang__)
#define RJ_HOOK_TEST_EXPORT __attribute__((visibility("default")))
#else
#define RJ_HOOK_TEST_EXPORT
#endif

/// @brief Set or clear the synthetic KFD topology root used by hook unit tests.
extern "C" RJ_HOOK_TEST_EXPORT void rj_hsa_dbt_set_topology_nodes_root_for_test(const char *root) {
  rocjitsu::hooks::set_topology_nodes_root_for_test(root);
}
