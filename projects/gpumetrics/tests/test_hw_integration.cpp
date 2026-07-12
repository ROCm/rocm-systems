// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Hardware integration tests: exercise the Collector against the REAL backend
// plugins (amdsmi required, rocprofiler optional) on live GPUs. Every test
// GTEST_SKIPs gracefully when no GPU / plugin is available. Ranges are asserted
// only for OK metrics; UNSUPPORTED is logged and accepted.
//
// Real plugin dirs are passed via GPUM_HW_PLUGIN_DIRS (colon-separated, set by
// CMake).

#include <gtest/gtest.h>

#include <cmath>
#include <cstdlib>
#include <sstream>
#include <string>
#include <vector>

#include "gpumetrics/gpumetrics.h"

using namespace gpumetrics;

namespace {

// Split GPUM_HW_PLUGIN_DIRS into a directory list; falls back to the build
// layout if unset.
std::vector<std::string> PluginDirs() {
  std::vector<std::string> dirs;
  if (const char* env = std::getenv("GPUM_HW_PLUGIN_DIRS")) {
    std::stringstream ss(env);
    std::string tok;
    while (std::getline(ss, tok, ':'))
      if (!tok.empty()) dirs.push_back(tok);
  }
  if (dirs.empty()) {
    dirs = {"plugins/amdsmi", "plugins/rocprofiler"};
  }
  return dirs;
}

CollectorOptions HwOpts() {
  CollectorOptions o;
  o.plugin_paths = PluginDirs();
  // Empty `plugins` = load all discoverable plugins.
  o.provider_priority = {"amdsmi", "rocprofiler"};
  return o;
}

// Build a collector; nullptr only on hard failure (never for "no GPU").
std::unique_ptr<Collector> MakeCollector() {
  gpum_status st = GPUM_OK;
  auto c = Collector::Create(HwOpts(), &st);
  return c;
}

bool HasAmdsmi(const Collector& c) {
  for (const auto& n : c.loadedPlugins())
    if (n == "amdsmi") return true;
  return false;
}

bool HasRocprofiler(const Collector& c) {
  for (const auto& n : c.loadedPlugins())
    if (n == "rocprofiler") return true;
  return false;
}

bool IdentityHasStrongKey(const gpum_device_identity& id) {
  if (id.bdf != 0) return true;
  if (id.kfd_node_id != 0) return true;
  for (int i = 0; i < 16; ++i)
    if (id.uuid[i] != 0) return true;
  return false;
}

gpum_entity_id GpuId(uint32_t gpu) {
  gpum_entity_id id{};
  id.kind = GPUM_ENTITY_GPU;
  id.socket = 0;
  id.gpu = gpu;
  id.partition = -1;
  return id;
}

}  // namespace

// --- Discovery ------------------------------------------------------------

TEST(HwIntegration, LoadsAmdsmiAndDiscoversGpus) {
  auto c = MakeCollector();
  ASSERT_NE(c, nullptr);
  if (!HasAmdsmi(*c)) GTEST_SKIP() << "amdsmi plugin not loaded (no ROCm/HW?)";
  if (c->devices().empty()) GTEST_SKIP() << "no GPUs discovered";
  EXPECT_GE(c->devices().size(), 1u);
  std::fprintf(stderr, "[hw] discovered %zu GPU(s); plugins:", c->devices().size());
  for (const auto& n : c->loadedPlugins()) std::fprintf(stderr, " %s", n.c_str());
  std::fprintf(stderr, "\n");
}

TEST(HwIntegration, EveryGpuHasValidIdentityAndName) {
  auto c = MakeCollector();
  ASSERT_NE(c, nullptr);
  if (!HasAmdsmi(*c) || c->devices().empty()) GTEST_SKIP() << "no GPUs discovered";
  for (const auto& d : c->devices()) {
    EXPECT_TRUE(IdentityHasStrongKey(d.identity))
        << "GPU " << d.id.gpu << " has no BDF/KFD/UUID identity key";
    EXPECT_FALSE(d.name.empty()) << "GPU " << d.id.gpu << " has empty name";
    std::fprintf(stderr, "[hw] gpu:%u name='%s' bdf=0x%llx kfd=%u\n", d.id.gpu,
                 d.name.c_str(), static_cast<unsigned long long>(d.identity.bdf),
                 d.identity.kfd_node_id);
  }
}

// --- Core telemetry sanity ranges ----------------------------------------

TEST(HwIntegration, AmdsmiCoreMetricsInSaneRanges) {
  auto c = MakeCollector();
  ASSERT_NE(c, nullptr);
  if (!HasAmdsmi(*c) || c->devices().empty()) GTEST_SKIP() << "no GPUs discovered";

  // Assert a range only if the metric read OK; UNSUPPORTED is acceptable.
  auto check = [&](const gpum_entity_id& e, const std::string& key, double lo,
                   double hi) {
    Sample s = c->read(e, key);
    if (s.status == GPUM_ERR_UNSUPPORTED) {
      std::fprintf(stderr, "[hw]   %s: UNSUPPORTED (ok)\n", key.c_str());
      return;
    }
    if (s.status == GPUM_ERR_NOT_FOUND) {
      std::fprintf(stderr, "[hw]   %s: NOT_FOUND (not advertised)\n", key.c_str());
      return;
    }
    ASSERT_TRUE(s.ok()) << key << ": " << gpum_status_string(s.status);
    double v = s.as_double();
    EXPECT_TRUE(std::isfinite(v)) << key << " not finite";
    EXPECT_GE(v, lo) << key << " below range";
    EXPECT_LE(v, hi) << key << " above range";
    std::fprintf(stderr, "[hw]   %s = %.3f\n", key.c_str(), v);
  };

  for (const auto& d : c->devices()) {
    const gpum_entity_id e = d.id;
    std::fprintf(stderr, "[hw] ranges for gpu:%u\n", d.id.gpu);
    check(e, "temp.edge", 0.0, 150.0);
    check(e, "clock.gfx", 0.0, 10000.0);
    check(e, "power.average_socket", 0.0, 1000.0);
    check(e, "activity.gfx", 0.0, 100.0);

    // VRAM: total > 0, used <= total (only when both read OK).
    Sample total = c->read(e, "mem.vram.total");
    Sample used = c->read(e, "mem.vram.used");
    if (total.ok()) {
      EXPECT_GT(total.as_double(), 0.0) << "mem.vram.total must be > 0";
      std::fprintf(stderr, "[hw]   mem.vram.total = %.0f bytes\n", total.as_double());
    }
    if (total.ok() && used.ok()) {
      EXPECT_LE(used.as_double(), total.as_double())
          << "mem.vram.used exceeds total";
      std::fprintf(stderr, "[hw]   mem.vram.used = %.0f bytes\n", used.as_double());
    }
  }
}

// --- Error handling -------------------------------------------------------

TEST(HwIntegration, UnknownMetricReturnsNotFound) {
  auto c = MakeCollector();
  ASSERT_NE(c, nullptr);
  if (!HasAmdsmi(*c) || c->devices().empty()) GTEST_SKIP() << "no GPUs discovered";
  Sample s = c->read(c->devices()[0].id, "totally.bogus.metric.key");
  EXPECT_EQ(s.status, GPUM_ERR_NOT_FOUND);
}

TEST(HwIntegration, BogusEntityIsHandled) {
  auto c = MakeCollector();
  ASSERT_NE(c, nullptr);
  if (!HasAmdsmi(*c) || c->devices().empty()) GTEST_SKIP() << "no GPUs discovered";
  // A GPU ordinal far beyond what exists must not crash and must not report OK.
  // A known metric key is used so a non-OK status can only come from the entity
  // not resolving to a plugin-local device.
  Sample s = c->read(GpuId(9999u), "temp.edge");
  EXPECT_NE(s.status, GPUM_OK);
}

// --- rocprofiler (optional) ----------------------------------------------

TEST(HwIntegration, RocprofilerProfMetricReadsOkIfLoaded) {
  auto c = MakeCollector();
  ASSERT_NE(c, nullptr);
  if (!HasRocprofiler(*c)) GTEST_SKIP() << "rocprofiler plugin not loaded";
  if (c->devices().empty()) GTEST_SKIP() << "no GPUs discovered";

  // Collect advertised prof.* keys.
  std::vector<std::string> prof_keys;
  for (const auto& m : c->metrics())
    if (m.key.rfind("prof.", 0) == 0) prof_keys.push_back(m.key);
  if (prof_keys.empty()) GTEST_SKIP() << "no prof.* metrics advertised";
  EXPECT_GE(prof_keys.size(), 1u);

  // At least one prof.* key should read OK (value may be ~0 at idle); skip if
  // none can be sampled on this HW.
  const gpum_entity_id e = c->devices()[0].id;
  int ok_count = 0;
  for (const auto& k : prof_keys) {
    Sample s = c->read(e, k);
    if (s.ok()) {
      ++ok_count;
      EXPECT_TRUE(std::isfinite(s.as_double())) << k << " not finite";
      std::fprintf(stderr, "[hw]   %s = %.3f\n", k.c_str(), s.as_double());
    }
  }
  if (ok_count == 0)
    GTEST_SKIP() << "rocprofiler loaded but no prof.* counter sampled on this HW";
}

// --- Batched read ---------------------------------------------------------

TEST(HwIntegration, BatchedReadReturnsOneSamplePerKey) {
  auto c = MakeCollector();
  ASSERT_NE(c, nullptr);
  if (!HasAmdsmi(*c) || c->devices().empty()) GTEST_SKIP() << "no GPUs discovered";

  const std::vector<std::string> keys = {
      "temp.edge",     "temp.hotspot", "clock.gfx",  "clock.mem",
      "mem.vram.used", "mem.vram.total", "activity.gfx", "power.average_socket"};
  auto samples = c->read(c->devices()[0].id, keys);
  ASSERT_EQ(samples.size(), keys.size());
  for (size_t i = 0; i < keys.size(); ++i) {
    EXPECT_EQ(samples[i].key, keys[i]);
    // A batched entry must never be left at the internal default; the batch ran.
    EXPECT_NE(samples[i].status, GPUM_ERR_INTERNAL) << keys[i];
  }
}

// --- Determinism ----------------------------------------------------------

TEST(HwIntegration, DiscoveryIsDeterministic) {
  auto a = MakeCollector();
  auto b = MakeCollector();
  ASSERT_NE(a, nullptr);
  ASSERT_NE(b, nullptr);
  if (!HasAmdsmi(*a) || a->devices().empty()) GTEST_SKIP() << "no GPUs discovered";

  ASSERT_EQ(a->devices().size(), b->devices().size())
      << "GPU count changed between discoveries";
  for (size_t i = 0; i < a->devices().size(); ++i) {
    EXPECT_EQ(a->devices()[i].identity.bdf, b->devices()[i].identity.bdf)
        << "canonical ordering differs at ordinal " << i;
    EXPECT_EQ(a->devices()[i].id.gpu, b->devices()[i].id.gpu);
  }
  // BDF order must be non-decreasing (canonical socket-then-BDF ordering).
  for (size_t i = 1; i < a->devices().size(); ++i) {
    EXPECT_LE(a->devices()[i - 1].identity.bdf, a->devices()[i].identity.bdf)
        << "BDF ordering not canonical";
  }
}
