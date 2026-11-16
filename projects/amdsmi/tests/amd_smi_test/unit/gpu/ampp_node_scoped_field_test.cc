// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Regression coverage for the AMPP "node_"-prefix predicate that
// rsmi_dev_ampp_fields_get()/rsmi_dev_ampp_profile_configure() use to hide/
// reject node-wide fields on non-primary GPUs (rocm_smi_ampp.cc). Pure string
// comparison -- no sysfs, root, or mock-sysfs overlay required -- mirroring
// ampp_writable_slot_mask_parse_test.cc's coverage of the sibling pure helper
// in the same file.
#include <gtest/gtest.h>

#include <string>

#include "rocm_smi/rocm_smi_ampp.h"

namespace {

using amd::smi::is_node_scoped_field;

TEST(GpuUnit, AmppNodeScopedFieldPrefixMatchIsScoped) {
  EXPECT_TRUE(is_node_scoped_field("node_power_limit"));
}

TEST(GpuUnit, AmppNodeScopedFieldBareprefixIsScoped) {
  // The prefix itself, with nothing following, still matches -- rfind(..., 0)
  // only requires the prefix to occur at position 0, not a non-empty suffix.
  EXPECT_TRUE(is_node_scoped_field("node_"));
}

TEST(GpuUnit, AmppNodeScopedFieldNonNodeFieldIsNotScoped) {
  EXPECT_FALSE(is_node_scoped_field("PPT0_LIMIT"));
}

TEST(GpuUnit, AmppNodeScopedFieldMissingUnderscoreIsNotScoped) {
  EXPECT_FALSE(is_node_scoped_field("node power_limit"));
}

TEST(GpuUnit, AmppNodeScopedFieldTruncatedPrefixIsNotScoped) {
  EXPECT_FALSE(is_node_scoped_field("node"));
}

TEST(GpuUnit, AmppNodeScopedFieldEmptyNameIsNotScoped) { EXPECT_FALSE(is_node_scoped_field("")); }

TEST(GpuUnit, AmppNodeScopedFieldPrefixNotAtStartIsNotScoped) {
  // "node_" appearing mid-string must not match -- only a leading prefix
  // designates a node-wide field.
  EXPECT_FALSE(is_node_scoped_field("a_node_power_limit"));
}

TEST(GpuUnit, AmppNodeScopedFieldUppercaseIsNotScoped) {
  // Case-sensitive by design: the driver's real field names are observed
  // lowercase (e.g. "node_power_limit"); a differently-cased name is treated
  // as an ordinary (non-node-scoped) field name.
  EXPECT_FALSE(is_node_scoped_field("NODE_POWER_LIMIT"));
}

}  // namespace
