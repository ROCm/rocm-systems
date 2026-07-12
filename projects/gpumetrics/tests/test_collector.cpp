// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
// End-to-end Collector test using two mock plugin .so's (mockA, mockB) that
// report the SAME two physical GPUs. Exercises discovery/dlopen, cross-plugin
// correlation, metric-registry conflict resolution by priority, selector
// parsing, and batched routed reads. No GPU required.
//
// The mock .so directory is passed via GPUM_TEST_PLUGIN_DIR (set by CMake).

#include <gtest/gtest.h>

#include <cstdlib>
#include <string>

#include "gpumetrics/gpumetrics.h"

using namespace gpumetrics;

namespace {

CollectorOptions TestOpts() {
  CollectorOptions o;
  const char* dir = std::getenv("GPUM_TEST_PLUGIN_DIR");
  o.plugin_paths = {dir ? dir : "."};
  o.plugins = {"mockA", "mockB"};  // both mock instances
  o.provider_priority = {"mockA", "mockB"};
  return o;
}

}  // namespace

TEST(Collector, LoadsBothMockPlugins) {
  gpum_status st;
  auto c = Collector::Create(TestOpts(), &st);
  ASSERT_NE(c, nullptr) << gpum_status_string(st);
  auto plugins = c->loadedPlugins();
  ASSERT_EQ(plugins.size(), 2u);
}

TEST(Collector, CorrelatesToTwoCanonicalGpus) {
  auto c = Collector::Create(TestOpts(), nullptr);
  ASSERT_NE(c, nullptr);
  // Both mocks report the same 2 GPUs -> 2 canonical devices, each with both
  // providers.
  ASSERT_EQ(c->devices().size(), 2u);
  for (const auto& d : c->devices()) EXPECT_EQ(d.providers.size(), 2u);
}

TEST(Collector, MetricRegistryResolvesConflictByPriority) {
  auto c = Collector::Create(TestOpts(), nullptr);
  ASSERT_NE(c, nullptr);
  // mock.shared is offered by both; mockA has priority.
  auto d = c->describe("mock.shared");
  ASSERT_TRUE(d.has_value());
  EXPECT_EQ(d->provider, "mockA");
}

TEST(Collector, ResolvesSelectors) {
  auto c = Collector::Create(TestOpts(), nullptr);
  ASSERT_NE(c, nullptr);
  auto e = c->resolve("gpu:1");
  ASSERT_TRUE(e.has_value());
  EXPECT_EQ(e->id.kind, GPUM_ENTITY_GPU);
  EXPECT_EQ(e->id.gpu, 1u);

  // bdf selector: gpu1 has bdf 0x6400
  auto eb = c->resolve("bdf:0000:64:00.0");
  ASSERT_TRUE(eb.has_value());
  EXPECT_EQ(eb->id.gpu, 1u);

  EXPECT_FALSE(c->resolve("gpu:99").has_value());
  EXPECT_FALSE(c->resolve("garbage").has_value());
}

TEST(Collector, ReadRoutesToCorrectPluginLocalDevice) {
  auto c = Collector::Create(TestOpts(), nullptr);
  ASSERT_NE(c, nullptr);
  // mock.temp value = 40 + plugin_local_index. mockA uses local base 0, so
  // gpu0 -> local 0 -> 40, gpu1 -> local 1 -> 41.
  auto s0 = c->read({GPUM_ENTITY_GPU, 0, 0, -1}, "mock.temp");
  ASSERT_TRUE(s0.ok()) << gpum_status_string(s0.status);
  EXPECT_DOUBLE_EQ(s0.as_double(), 40.0);

  auto s1 = c->read({GPUM_ENTITY_GPU, 0, 1, -1}, "mock.temp");
  ASSERT_TRUE(s1.ok());
  EXPECT_DOUBLE_EQ(s1.as_double(), 41.0);
}

TEST(Collector, BatchReadFillsAllKeys) {
  auto c = Collector::Create(TestOpts(), nullptr);
  ASSERT_NE(c, nullptr);
  auto v = c->read({GPUM_ENTITY_GPU, 0, 0, -1},
                   std::vector<std::string>{"mock.temp", "mock.shared", "does.not.exist"});
  ASSERT_EQ(v.size(), 3u);
  EXPECT_TRUE(v[0].ok());
  EXPECT_TRUE(v[1].ok());
  EXPECT_EQ(v[2].status, GPUM_ERR_NOT_FOUND);
}

TEST(Collector, SharedMetricServedByPriorityPlugin) {
  auto c = Collector::Create(TestOpts(), nullptr);
  ASSERT_NE(c, nullptr);
  // mock.shared encodes first char of provider name; mockA -> 'A' (65).
  auto s = c->read({GPUM_ENTITY_GPU, 0, 0, -1}, "mock.shared");
  ASSERT_TRUE(s.ok());
  EXPECT_EQ(s.as_double(), 1000.0 + 'A');
}

TEST(Collector, EntitiesEnumeratesGpus) {
  auto c = Collector::Create(TestOpts(), nullptr);
  ASSERT_NE(c, nullptr);
  auto gpus = c->entities(GPUM_ENTITY_GPU);
  EXPECT_EQ(gpus.size(), 2u);
}
