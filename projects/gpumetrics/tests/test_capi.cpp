// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Tests the flat C API (capi.h) against the mock plugins, guarding the FFI
// contract the Rust bindings sit on.

#include <gtest/gtest.h>

#include <cstdlib>
#include <string>
#include <vector>

#include "gpumetrics/capi.h"

namespace {

gpum_collector* MakeCollector() {
  const char* dir = std::getenv("GPUM_TEST_PLUGIN_DIR");
  static std::string d = dir ? dir : ".";
  const char* paths[] = {d.c_str()};
  const char* plugins[] = {"mockA", "mockB"};
  const char* prio[] = {"mockA", "mockB"};
  gpum_collector_options o{};
  o.plugin_paths = paths;
  o.plugin_paths_count = 1;
  o.plugins = plugins;
  o.plugins_count = 2;
  o.provider_priority = prio;
  o.provider_priority_count = 2;
  gpum_status st;
  gpum_collector* c = gpum_collector_create(&o, &st);
  EXPECT_EQ(st, GPUM_OK);
  return c;
}

}  // namespace

TEST(CApi, CreateAndTopology) {
  gpum_collector* c = MakeCollector();
  ASSERT_NE(c, nullptr);
  EXPECT_EQ(gpum_collector_gpu_count(c), 2u);

  char name[GPUM_STRING_MAX];
  gpum_device_identity id{};
  uint32_t parts = 999;
  ASSERT_EQ(gpum_collector_gpu_info(c, 0, name, sizeof(name), &id, &parts), GPUM_OK);
  EXPECT_EQ(id.bdf, 0x6300u);
  EXPECT_EQ(parts, 0u);
  EXPECT_EQ(gpum_collector_gpu_info(c, 99, nullptr, 0, nullptr, nullptr), GPUM_ERR_NOT_FOUND);
  gpum_collector_destroy(c);
}

TEST(CApi, MetricEnumeration) {
  gpum_collector* c = MakeCollector();
  ASSERT_NE(c, nullptr);
  uint32_t n = gpum_collector_metric_count(c);
  EXPECT_GT(n, 0u);
  bool found_shared = false;
  for (uint32_t i = 0; i < n; ++i) {
    char key[GPUM_STRING_MAX], unit[32], prov[GPUM_STRING_MAX];
    gpum_value_type t;
    uint32_t scope;
    ASSERT_EQ(gpum_collector_metric_at(c, i, key, sizeof(key), unit, sizeof(unit), prov,
                                       sizeof(prov), &t, &scope),
              GPUM_OK);
    if (std::string(key) == "mock.shared") {
      found_shared = true;
      EXPECT_EQ(std::string(prov), "mockA");  // priority winner
    }
  }
  EXPECT_TRUE(found_shared);
  gpum_collector_destroy(c);
}

TEST(CApi, ResolveAndRead) {
  gpum_collector* c = MakeCollector();
  ASSERT_NE(c, nullptr);

  gpum_entity_id e{};
  ASSERT_EQ(gpum_collector_resolve(c, "gpu:1", &e), GPUM_OK);
  EXPECT_EQ(e.kind, GPUM_ENTITY_GPU);
  EXPECT_EQ(e.gpu, 1u);

  gpum_sample s{};
  ASSERT_EQ(gpum_collector_read(c, &e, "mock.temp", &s), GPUM_OK);
  EXPECT_EQ(s.status, GPUM_OK);
  EXPECT_EQ(s.type, GPUM_TYPE_F64);
  EXPECT_DOUBLE_EQ(s.value.f64, 41.0);  // 40 + local index 1

  EXPECT_EQ(gpum_collector_resolve(c, "nope", &e), GPUM_ERR_NOT_FOUND);
  gpum_collector_destroy(c);
}

TEST(CApi, BatchRead) {
  gpum_collector* c = MakeCollector();
  ASSERT_NE(c, nullptr);
  gpum_entity_id e{};
  ASSERT_EQ(gpum_collector_resolve(c, "gpu:0", &e), GPUM_OK);
  const char* keys[] = {"mock.temp", "mock.shared", "missing"};
  gpum_sample out[3]{};
  ASSERT_EQ(gpum_collector_read_batch(c, &e, keys, 3, out), GPUM_OK);
  EXPECT_EQ(out[0].status, GPUM_OK);
  EXPECT_EQ(out[1].status, GPUM_OK);
  EXPECT_EQ(out[2].status, GPUM_ERR_NOT_FOUND);
  gpum_collector_destroy(c);
}
