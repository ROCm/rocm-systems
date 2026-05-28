// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// P0 smoke test: the standalone lib builds, the strict JSON loader parses the
/// cdna4 sidecar, and an ArchModel constructs from it. Scheduler/memory behavior
/// is unimplemented (P1+) and not exercised here.

#include "cycle_model/ArchModel.h"
#include "cycle_model/UarchConfig.h"

#include <gtest/gtest.h>

#include <fstream>
#include <stdexcept>
#include <string>
#include <cstdio>

namespace {

TEST(CycleModelSmoke, LoadsCdna4Config) {
  auto cfg = cycle_model::load_uarch_config(CYCLE_MODEL_CDNA4_JSON);
  EXPECT_EQ(cfg.name, "cdna4");
  EXPECT_EQ(cfg.simds_per_cu, 4u);
  EXPECT_EQ(cfg.wave_size, 64u);
  EXPECT_TRUE(cfg.has_mfma);
  EXPECT_FALSE(cfg.has_wmma);
  EXPECT_EQ(cfg.hbm_bytes_per_channel_per_cycle, 396u);
  EXPECT_EQ(cfg.dep_rules.size(), 1u);          // parsed, not dropped
  EXPECT_FALSE(cfg.opcode_latency.empty());
}

TEST(CycleModelSmoke, BuildsArchModel) {
  auto cfg = cycle_model::load_uarch_config(CYCLE_MODEL_CDNA4_JSON);
  cycle_model::ArchModel model(cfg);
  EXPECT_EQ(model.num_simds(), 4u);
}

TEST(CycleModelSmoke, StrictLoaderThrowsOnMissingKey) {
  // The strict loader must fail loudly on a missing required key instead of
  // silently defaulting (the /code-review finding this phase fixes).
  const std::string path = std::string(testing::TempDir()) + "/cm_bad_uarch.json";
  { std::ofstream o(path); o << R"({ "name": "broken" })"; }
  EXPECT_THROW(cycle_model::load_uarch_config(path), std::runtime_error);
  std::remove(path.c_str());
}

}  // namespace
