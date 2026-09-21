// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "embedded_schema.h"
#include "rocjitsu/config/config_common.h"
#include "rocjitsu/config/config_loader.h"
#include "rocjitsu/vm/amdgpu/matrix_coexecution.h"
#include "rocjitsu/vm/amdgpu/partitioning.h"
#include "rocjitsu/vm/rj_vm.h"
#include "rocjitsu/vm/rj_vm_impl.h"
#include "rocjitsu/vm/soc.h"

#include "simdojo/sim/component.h"
#include "simdojo/sim/simulation.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <set>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace rocjitsu;

const std::string CONFIG_PATH = std::string(CONFIG_DIR) + "/gfx950_mi355x.json";
const std::string CONFIG_KMD_PATH = std::string(CONFIG_DIR) + "/gfx950_mi355x_kmd.json";
const std::string CONFIG_2GPU_PATH = std::string(CONFIG_DIR) + "/gfx950_mi355x_kmd_2gpu.json";
const std::string CONFIG_1XCD_PATH = std::string(CONFIG_DIR) + "/gfx1100_w7900.json";

// Most shipped configs omit num_threads so they pick up the default; the 2-GPU
// KMD config pins it. Replace the pin when there is one, otherwise insert an
// override after the opening brace.
std::string config_json_with_num_threads(const std::string &path, uint32_t num_threads) {
  std::ifstream input(path);
  if (!input.is_open())
    throw std::runtime_error("Failed to open config: " + path);

  std::string json((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
  // Shipped configs indent top-level keys by two spaces; table entries are nested.
  const std::string field = "\n  \"num_threads\":";
  const std::string replacement = field + " " + std::to_string(num_threads);

  const size_t field_pos = json.find(field);
  if (field_pos != std::string::npos) {
    const size_t value_end = json.find_first_of(",}\n", field_pos + field.size());
    if (value_end == std::string::npos)
      throw std::runtime_error("Unterminated num_threads field: " + path);
    json.replace(field_pos, value_end - field_pos, replacement);
    return json;
  }

  const size_t brace_pos = json.find('{');
  if (brace_pos == std::string::npos)
    throw std::runtime_error("Config is not a JSON object: " + path);

  json.insert(brace_pos + 1, replacement + ",");
  return json;
}

struct PartitionedTopology {
  config::LoadedConfig loaded;
  SoC *soc = nullptr;
  simdojo::Component *memory = nullptr;
  std::unique_ptr<simdojo::SimulationEngine> engine;
  bool partitioned = false;
};

PartitionedTopology build_partitioned_topology(uint32_t num_threads) {
  auto loaded = config::load_config(CONFIG_PATH, rocjitsu::kEmbeddedSchema);
  auto *soc = loaded.soc();
  auto *memory = loaded.memory();
  loaded.engine_config.num_threads = num_threads;

  auto engine = std::make_unique<simdojo::SimulationEngine>(loaded.engine_config);
  engine->topology().set_root(loaded.take_root());
  loaded.wire_links(engine->topology());
  bool partitioned = amdgpu::partition_topology_by_xcds(engine->topology(), soc, num_threads);
  engine->create();

  return {std::move(loaded), soc, memory, std::move(engine), partitioned};
}

void expect_subtree_partition(simdojo::Component *component, simdojo::PartitionID expected) {
  ASSERT_NE(component, nullptr);
  EXPECT_EQ(component->partition_id(), expected) << component->full_path();

  auto *composite = dynamic_cast<simdojo::CompositeComponent *>(component);
  if (!composite)
    return;

  for (const auto &child : composite->children())
    expect_subtree_partition(child.get(), expected);
}

TEST(CpuDispatchBudgetTest, AutoUsesDetectedHostWidthBelowCap) {
  EXPECT_EQ(config::resolve_cpu_dispatch_thread_budgets(
                /*requested_threads=*/0, /*hardware_threads=*/6, /*soc_count=*/1),
            (std::vector<uint32_t>{6}));
}

TEST(CpuDispatchBudgetTest, AutoFallsBackToOneWhenHostWidthIsUnknown) {
  EXPECT_EQ(config::resolve_cpu_dispatch_thread_budgets(
                /*requested_threads=*/0, /*hardware_threads=*/0, /*soc_count=*/1),
            (std::vector<uint32_t>{1}));
}

TEST(CpuDispatchBudgetTest, AutoCapsHostWidthAtThirtyTwo) {
  EXPECT_EQ(config::resolve_cpu_dispatch_thread_budgets(
                /*requested_threads=*/0, /*hardware_threads=*/128, /*soc_count=*/1),
            (std::vector<uint32_t>{32}));
}

TEST(CpuDispatchBudgetTest, AutoUsesSuppliedCap) {
  EXPECT_EQ(config::resolve_cpu_dispatch_thread_budgets(
                /*requested_threads=*/0, /*hardware_threads=*/128, /*soc_count=*/4,
                /*automatic_thread_cap=*/12),
            (std::vector<uint32_t>{3, 3, 3, 3}));
}

TEST(CpuDispatchBudgetTest, ZeroSocCountReturnsNoBudgets) {
  EXPECT_EQ(config::resolve_cpu_dispatch_thread_budgets(
                /*requested_threads=*/0, /*hardware_threads=*/128, /*soc_count=*/0,
                /*automatic_thread_cap=*/12),
            (std::vector<uint32_t>{}));
}

TEST(CpuDispatchBudgetTest, AutoDividesHostBudgetFairlyAcrossSocs) {
  EXPECT_EQ(config::resolve_cpu_dispatch_thread_budgets(
                /*requested_threads=*/0, /*hardware_threads=*/10, /*soc_count=*/3),
            (std::vector<uint32_t>{4, 3, 3}));
  EXPECT_EQ(config::resolve_cpu_dispatch_thread_budgets(
                /*requested_threads=*/0, /*hardware_threads=*/128, /*soc_count=*/4),
            (std::vector<uint32_t>{8, 8, 8, 8}));
  EXPECT_EQ(config::resolve_cpu_dispatch_thread_budgets(
                /*requested_threads=*/0, /*hardware_threads=*/2, /*soc_count=*/4),
            (std::vector<uint32_t>{1, 1, 1, 1}));
}

TEST(CpuDispatchBudgetTest, ExplicitWidthRemainsPerSoc) {
  EXPECT_EQ(config::resolve_cpu_dispatch_thread_budgets(
                /*requested_threads=*/7, /*hardware_threads=*/2, /*soc_count=*/3,
                /*automatic_thread_cap=*/12),
            (std::vector<uint32_t>{7, 7, 7}));
}

TEST(CpuDispatchBudgetTest, SingleGpuDefaultsToAutomaticAndMultiGpuToSerialDispatch) {
  {
    auto loaded = config::load_config(CONFIG_KMD_PATH, rocjitsu::kEmbeddedSchema);
    EXPECT_EQ(loaded.cpu_dispatch_threads, 0u);
  }
  {
    auto loaded = config::load_config(CONFIG_2GPU_PATH, rocjitsu::kEmbeddedSchema);
    EXPECT_EQ(loaded.cpu_dispatch_threads, 1u);
  }
}

TEST(CpuDispatchBudgetTest, JsonDistinguishesOmittedZeroAndExplicitSchemaDefault) {
  struct Case {
    const char *json;
    bool present;
    uint32_t value;
  };
  for (const auto &c : {Case{"{}", false, 1}, Case{R"({"cpu_dispatch_threads":0})", true, 0},
                        Case{R"({"cpu_dispatch_threads":1})", true, 1}}) {
    SCOPED_TRACE(c.json);
    config::with_parsed_simulation_config_json(
        c.json, rocjitsu::kEmbeddedSchema, [&](const fb::SimulationConfig *parsed) {
          EXPECT_EQ(
              flatbuffers::IsFieldPresent(parsed, fb::SimulationConfig::VT_CPU_DISPATCH_THREADS),
              c.present);
          EXPECT_EQ(parsed->cpu_dispatch_threads(), c.value);
        });
  }
}

using Choice = config::ExecutionThreadChoice;
constexpr Choice kCdna4Choices[] = {{1, 1, 0},  {1, 2, 0},  {1, 4, 0},  {2, 7, 0},
                                    {8, 9, 0},  {8, 17, 0}, {8, 25, 0}, {8, 25, 2},
                                    {8, 25, 4}, {8, 25, 8}, {8, 25, 16}};

TEST(ExecutionThreadBudgetTest, SelectsLargestFittingGranuleAndCapsAutomaticBudget) {
  struct Case {
    uint32_t budget, engines, dispatch, helpers;
  };
  const Case cases[] = {{1, 1, 1, 0},    {2, 1, 2, 0},    {3, 1, 2, 0},   {4, 1, 4, 0},
                        {7, 1, 4, 0},    {8, 2, 7, 0},    {12, 2, 7, 0},  {16, 8, 9, 0},
                        {23, 8, 9, 0},   {24, 8, 17, 0},  {31, 8, 17, 0}, {32, 8, 25, 0},
                        {33, 8, 25, 0},  {34, 8, 25, 2},  {35, 8, 25, 2}, {36, 8, 25, 4},
                        {39, 8, 25, 4},  {40, 8, 25, 8},  {47, 8, 25, 8}, {48, 8, 25, 16},
                        {64, 8, 25, 16}, {128, 8, 25, 16}};
  const std::array capacities{36u};
  for (auto c : cases) {
    SCOPED_TRACE(c.budget);
    for (bool explicit_budget : {false, true}) {
      auto plan = config::resolve_execution_threads({.budget = explicit_budget ? c.budget : 0},
                                                    explicit_budget ? 128 : c.budget, 8, capacities,
                                                    true, kCdna4Choices);
      EXPECT_EQ(plan.engines, c.engines);
      EXPECT_EQ(plan.dispatch, (std::vector<uint32_t>{c.dispatch}));
      EXPECT_EQ(plan.helpers, c.helpers);
    }
  }
  auto unknown = config::resolve_execution_threads({}, 0, 8, capacities, true, kCdna4Choices);
  EXPECT_EQ(unknown.engines, 1u);
  EXPECT_EQ(unknown.dispatch, (std::vector<uint32_t>{1}));
  EXPECT_EQ(unknown.helpers, 0u);
  const Choice larger[] = {{8, 25, 0}, {8, 33, 8}};
  auto automatic = config::resolve_execution_threads({}, 64, 8, capacities, true, larger);
  EXPECT_EQ(automatic.dispatch, (std::vector<uint32_t>{25}));
  auto explicit_plan =
      config::resolve_execution_threads({.budget = 48}, 64, 8, capacities, true, larger);
  EXPECT_EQ(explicit_plan.dispatch, (std::vector<uint32_t>{33}));
  EXPECT_EQ(explicit_plan.helpers, 8u);
}

TEST(ExecutionThreadBudgetTest, ConfigCanStopAtAFewThreads) {
  const Choice choices[] = {{1, 1, 0}, {1, 2, 0}, {1, 4, 0}};
  const std::array capacities{64u};
  auto plan = config::resolve_execution_threads({}, 64, 1, capacities, false, choices);
  EXPECT_EQ(plan.engines, 1u);
  EXPECT_EQ(plan.dispatch, (std::vector<uint32_t>{4}));
  EXPECT_EQ(plan.helpers, 0u);
}

TEST(ExecutionThreadBudgetTest, AccountsForEveryGpuAndTopologyClamps) {
  const Choice choices[] = {{1, 1, 0}, {2, 5, 2}, {8, 17, 8}};
  const std::array capacities{36u, 36u};
  auto plan = config::resolve_execution_threads({}, 32, 16, capacities, true, choices);
  EXPECT_EQ(plan.engines, 2u);
  EXPECT_EQ(plan.dispatch, (std::vector<uint32_t>{5, 5}));
  EXPECT_EQ(plan.helpers, 2u); // 2 + 2*(5-1) + 2 = 12; the next entry costs 48.
  const std::array tiny{2u};
  plan = config::resolve_execution_threads({}, 32, 1, tiny, false, choices);
  EXPECT_EQ(plan.engines, 1u);
  EXPECT_EQ(plan.dispatch, (std::vector<uint32_t>{2}));
  EXPECT_EQ(plan.helpers, 0u);
}

TEST(ExecutionThreadBudgetTest, ExplicitOverridesAndSafeFallback) {
  const std::array capacities{64u};
  auto plan = config::resolve_execution_threads({.engines = 8, .dispatch = 33, .helpers = 8}, 4, 8,
                                                capacities, true, kCdna4Choices);
  EXPECT_EQ(plan.engines, 8u);
  EXPECT_EQ(plan.dispatch, (std::vector<uint32_t>{33}));
  EXPECT_EQ(plan.helpers, 8u);
  plan = config::resolve_execution_threads({.dispatch = 1, .helpers = 0}, 32, 8, capacities, true,
                                           kCdna4Choices);
  EXPECT_EQ(plan.engines, 8u);
  EXPECT_EQ(plan.dispatch, (std::vector<uint32_t>{1}));
  EXPECT_EQ(plan.helpers, 0u);
  plan = config::resolve_execution_threads({}, 64, 8, capacities, true, {});
  EXPECT_EQ(plan.engines, 1u);
  EXPECT_EQ(plan.dispatch, (std::vector<uint32_t>{1}));
  EXPECT_EQ(plan.helpers, 0u);
  plan = config::resolve_execution_threads({}, 4, 8, capacities, true, kCdna4Choices, true);
  EXPECT_EQ(plan.engines, 4u);
  EXPECT_EQ(plan.dispatch, (std::vector<uint32_t>{1}));
  EXPECT_EQ(plan.helpers, 0u);
}

TEST(ExecutionThreadBudgetTest, RejectsInvalidRequestsAndGranules) {
  const std::array capacities{36u};
  for (int helpers : {-2, 129})
    EXPECT_THROW(
        config::resolve_execution_threads({.helpers = helpers}, 64, 8, capacities, true, {}),
        std::invalid_argument);
  for (Choice choice : {Choice{0, 1, 0}, Choice{1, 0, 0}, Choice{1, 1, 129}})
    EXPECT_THROW(
        config::resolve_execution_threads({}, 64, 8, capacities, true, std::span(&choice, 1)),
        std::invalid_argument);
}

TEST(ExecutionThreadBudgetTest, AutomaticPlansStayWithinAffinityAndCapacity) {
  for (uint32_t host : {0u, 1u, 2u, 3u, 4u, 7u, 8u, 16u, 24u, 32u, 64u, 128u})
    for (uint32_t xcds : {1u, 2u, 3u, 4u, 7u, 8u, 16u, 32u})
      for (bool async : {false, true}) {
        const std::array capacities{2u, 36u};
        auto plan =
            config::resolve_execution_threads({}, host, xcds, capacities, async, kCdna4Choices);
        uint64_t used = plan.engines + plan.helpers;
        for (size_t i = 0; i < capacities.size(); ++i) {
          EXPECT_GE(plan.dispatch[i], 1u);
          EXPECT_LE(plan.dispatch[i], capacities[i]);
          used += plan.dispatch[i] - 1;
        }
        EXPECT_LE(used, std::max(host, 1u));
        EXPECT_LE(used - plan.helpers, 32u);
        EXPECT_LE(plan.engines, xcds);
      }
}

TEST(ExecutionThreadBudgetTest, MetadataMatchesBuiltTopology) {
  for (const auto &path : {CONFIG_PATH, CONFIG_2GPU_PATH, CONFIG_1XCD_PATH}) {
    auto settings = config::load_execution_thread_settings(path, rocjitsu::kEmbeddedSchema);
    auto loaded = config::load_config(path, rocjitsu::kEmbeddedSchema, 32);
    auto plan = settings.resolve(32);
    EXPECT_EQ(plan.engines, loaded.execution_threads.engines);
    EXPECT_EQ(plan.dispatch, loaded.execution_threads.dispatch);
    EXPECT_EQ(plan.helpers, loaded.execution_threads.helpers);
    EXPECT_EQ(settings.xcds, loaded.build_result.num_xcds * (1 + loaded.extra_gpu_builds.size()));
  }
}

TEST(ExecutionThreadBudgetTest, CountsOnlySocsThatTheLoaderConstructs) {
  const std::string json = R"({
  "cpu_thread_budget": 4,
  "thread_allocations": [
    {
      "num_threads": 1,
      "cpu_dispatch_threads": 1
    },
    {
      "num_threads": 2,
      "cpu_dispatch_threads": 3
    }
  ],
  "vm": {
    "arch": "cdna3",
    "gpu": {
      "num_gpus": 4
    }
  },
  "topology": {
    "root": {
      "name": "soc",
      "type": "soc",
      "children": [
        {
          "name": "vram",
          "type": "gpu_memory"
        },
        {
          "name": "xcd[0:2]",
          "type": "xcd",
          "children": [
            {
              "name": "cp",
              "type": "command_processor"
            },
            {
              "name": "se",
              "type": "shader_engine",
              "children": [
                {
                  "name": "cu[0:3]",
                  "type": "compute_unit"
                }
              ]
            }
          ]
        }
      ]
    }
  }
})";
  auto loaded = config::load_config_from_string(json, rocjitsu::kEmbeddedSchema, 4);
  ASSERT_TRUE(loaded.extra_gpu_builds.empty());
  ASSERT_EQ(loaded.build_result.num_xcds, 2u);
  EXPECT_EQ(loaded.execution_threads.engines, 2u);
  EXPECT_EQ(loaded.execution_threads.dispatch, (std::vector<uint32_t>{3}));
  loaded.apply_cpu_dispatch_threads();
  EXPECT_EQ(loaded.soc()->dispatch_threads(), 3u);
}

TEST(ExecutionThreadBudgetTest, ShippedServerChoicesAndExplicitSerial) {
  auto json = config_json_with_num_threads(CONFIG_PATH, 0);
  json.insert(json.find('{') + 1, R"("cpu_thread_budget":32,)");
  auto loaded = config::load_config_from_string(json, rocjitsu::kEmbeddedSchema, 64);
  EXPECT_EQ(loaded.execution_threads.engines, 8u);
  EXPECT_EQ(loaded.execution_threads.dispatch, (std::vector<uint32_t>{25}));
  EXPECT_EQ(loaded.execution_threads.helpers, 0u);
  EXPECT_EQ(loaded.async_resources->helpers(), 0u);
  json.insert(json.find('{') + 1, R"("cpu_dispatch_threads":1,"async_helper_threads":0,)");
  auto serial = config::load_config_from_string(json, rocjitsu::kEmbeddedSchema, 64);
  EXPECT_EQ(serial.cpu_dispatch_threads, 1u);
  EXPECT_EQ(serial.execution_threads.dispatch, (std::vector<uint32_t>{1}));
  EXPECT_EQ(serial.execution_threads.helpers, 0u);
}

TEST(ExecutionThreadBudgetTest, PresetsKeepSiblingTablesConsistent) {
  for (const auto &paths :
       {std::vector<std::string>{"gfx950_mi355x.json", "gfx950_mi355x_kmd.json"},
        std::vector<std::string>{"gfx942_cdna3.json", "gfx942_cdna3_kmd.json"}}) {
    const auto expected = config::load_execution_thread_settings(
                              std::string(CONFIG_DIR) + "/" + paths[0], rocjitsu::kEmbeddedSchema)
                              .choices;
    for (const auto &path : paths)
      EXPECT_EQ(config::load_execution_thread_settings(std::string(CONFIG_DIR) + "/" + path,
                                                       rocjitsu::kEmbeddedSchema)
                    .choices,
                expected)
          << path;
  }
}

TEST(ExecutionThreadBudgetTest, MultiGpuPresetsKeepSerialDefaultsAndAllowDispatchOverrides) {
  for (const auto &[name, gpus] : {std::pair{"gfx950_mi355x_kmd_2gpu.json", 2u},
                                   std::pair{"gfx1250_mi455x_kmd_4gpu.json", 4u}}) {
    auto settings = config::load_execution_thread_settings(std::string(CONFIG_DIR) + "/" + name,
                                                           rocjitsu::kEmbeddedSchema);
    EXPECT_EQ(settings.request.engines, 1u);
    EXPECT_EQ(settings.request.dispatch, 1u);
    for (auto choice : settings.choices)
      EXPECT_EQ(choice.engines, 1u) << name;
    for (uint32_t budget = 1; budget <= 64; ++budget) {
      settings.request.budget = budget;
      auto plan = settings.resolve(64);
      EXPECT_EQ(plan.engines, 1u) << name;
      ASSERT_EQ(plan.dispatch.size(), gpus);
      uint32_t total = plan.engines;
      for (uint32_t width : plan.dispatch) {
        EXPECT_EQ(width, 1u);
        total += width - 1;
      }
      EXPECT_LE(total, std::min(budget, 32u)) << name;
    }
    settings.request.budget = 32;
    settings.request.dispatch = 0;
    EXPECT_EQ(settings.resolve(64).dispatch, std::vector<uint32_t>(gpus, 32 / gpus));
    settings.request.dispatch = 4;
    EXPECT_EQ(settings.resolve(64).dispatch, std::vector<uint32_t>(gpus, 4));
  }
}

TEST(ExecutionThreadBudgetTest, OlderCdnaPresetsUseDispatchWorkers) {
  for (const auto *name : {"gfx90a_mi210_kmd.json", "gfx942_cdna3.json", "gfx942_cdna3_kmd.json"}) {
    auto settings = config::load_execution_thread_settings(std::string(CONFIG_DIR) + "/" + name,
                                                           rocjitsu::kEmbeddedSchema);
    settings.request.budget = 1;
    auto serial = settings.resolve(64);
    EXPECT_EQ(serial.engines, 1u);
    EXPECT_EQ(serial.dispatch, (std::vector<uint32_t>{1}));
    settings.request.budget = 32;
    auto parallel = settings.resolve(64);
    EXPECT_EQ(parallel.engines, settings.xcds);
    EXPECT_EQ(parallel.dispatch, (std::vector<uint32_t>{33 - parallel.engines}));
  }
}

TEST(ExecutionThreadBudgetTest, DesktopTablesStopAtTheMeasuredCeiling) {
  for (const auto *name : {"gfx1100_w7900", "gfx1151", "gfx1201_r9700"}) {
    auto settings = config::load_execution_thread_settings(
        std::string(CONFIG_DIR) + "/" + name + ".json", rocjitsu::kEmbeddedSchema);
    for (uint32_t budget : {32u, 64u}) {
      settings.request.budget = budget;
      auto plan = settings.resolve(128);
      EXPECT_EQ(plan.engines, 1u);
      EXPECT_EQ(plan.dispatch, (std::vector<uint32_t>{32}));
    }
  }
}

TEST(ExecutionThreadBudgetTest, DisablingHelpersKeepsSynchronousServerGranules) {
  struct Case {
    uint32_t budget, engines, dispatch;
  };
  for (const auto *name : {"gfx950_mi355x.json", "gfx1250_mi455x.json"}) {
    auto settings = config::load_execution_thread_settings(std::string(CONFIG_DIR) + "/" + name,
                                                           rocjitsu::kEmbeddedSchema);
    settings.request.helpers = 0;
    for (Case c : {Case{1, 1, 1}, Case{2, 1, 2}, Case{8, 2, 7}, Case{16, 8, 9}, Case{24, 8, 17},
                   Case{32, 8, 25}}) {
      SCOPED_TRACE(name);
      SCOPED_TRACE(c.budget);
      settings.request.budget = c.budget;
      auto plan = settings.resolve(64);
      EXPECT_EQ(plan.engines, c.engines);
      EXPECT_EQ(plan.dispatch, (std::vector<uint32_t>{c.dispatch}));
      EXPECT_EQ(plan.helpers, 0u);
    }
  }
}

TEST(ExecutionThreadBudgetTest, ServerPresetsAddHelpersWithoutDisplacingCpuWorkers) {
  for (const char *name : {"gfx950_mi355x.json", "gfx950_mi355x_kmd.json", "gfx1250_mi455x.json"}) {
    auto settings = config::load_execution_thread_settings(std::string(CONFIG_DIR) + "/" + name,
                                                           rocjitsu::kEmbeddedSchema);
    for (uint32_t affinity = 1; affinity <= 128; ++affinity) {
      const auto plan = settings.resolve(affinity);
      ASSERT_EQ(plan.dispatch.size(), 1u);
      const auto cpu = plan.engines + plan.dispatch[0] - 1;
      EXPECT_LE(cpu, std::min(affinity, 32u));
      EXPECT_LE(cpu + plan.helpers, std::min(affinity, 48u));
      const unsigned expected = affinity >= 48   ? 16
                                : affinity >= 40 ? 8
                                : affinity >= 36 ? 4
                                : affinity >= 34 ? 2
                                                 : 0;
      EXPECT_EQ(plan.helpers, expected) << name << " affinity=" << affinity;
      if (affinity >= 32) {
        EXPECT_EQ(plan.engines, 8u);
        EXPECT_EQ(plan.dispatch[0], 25u);
      }
    }
    settings.request.budget = 32;
    EXPECT_EQ(settings.resolve(128).helpers, 0u);
    settings.request.budget = 48;
    EXPECT_EQ(settings.resolve(128).helpers, 16u);
    settings.request.helpers = 0;
    const auto disabled = settings.resolve(128);
    EXPECT_EQ(disabled.engines, 8u);
    EXPECT_EQ(disabled.dispatch, (std::vector<uint32_t>{25}));
    EXPECT_EQ(disabled.helpers, 0u);
  }
}

TEST(ExecutionThreadBudgetTest, AsyncPoolsBelongToTheirVm) {
  namespace coexec = amdgpu::matrix_coexecution;
  auto first = std::make_shared<coexec::ExecutionResources>(1);
  auto second = std::make_shared<coexec::ExecutionResources>(2);
  EXPECT_NE(&first->pool(), &second->pool());
  auto shared = first;
  EXPECT_EQ(&first->pool(), &shared->pool());
  first.reset();
  EXPECT_TRUE(shared->pool().available());
  auto disabled = std::make_shared<coexec::ExecutionResources>(0);
  EXPECT_FALSE(disabled->pool().available());
}

TEST(XcdPartitioningTest, EightThreadsMapsEachCdna4XcdToItsOwnPartition) {
  auto topology = build_partitioned_topology(8);

  ASSERT_TRUE(topology.partitioned);
  ASSERT_EQ(topology.engine->topology().partitions().size(), 8u);
  ASSERT_EQ(topology.soc->num_xcds(), 8u);

  for (uint32_t i = 0; i < topology.soc->num_xcds(); ++i)
    expect_subtree_partition(topology.soc->xcd(i), i);
}

TEST(XcdPartitioningTest, FourThreadsDistributesCdna4XcdsRoundRobinWithoutSplits) {
  auto topology = build_partitioned_topology(4);

  ASSERT_TRUE(topology.partitioned);
  ASSERT_EQ(topology.engine->topology().partitions().size(), 4u);
  ASSERT_EQ(topology.soc->num_xcds(), 8u);

  for (uint32_t i = 0; i < topology.soc->num_xcds(); ++i)
    expect_subtree_partition(topology.soc->xcd(i), i % 4);
}

TEST(XcdPartitioningTest, ClampPartitionCountUsesAllNonNullSocs) {
  auto loaded = config::load_config(CONFIG_2GPU_PATH, rocjitsu::kEmbeddedSchema);
  ASSERT_EQ(loaded.extra_gpu_builds.size(), 1u);

  auto *soc0 = loaded.soc();
  auto *soc1 = dynamic_cast<SoC *>(loaded.extra_gpu_builds[0].root.get());
  ASSERT_NE(soc0, nullptr);
  ASSERT_NE(soc1, nullptr);
  ASSERT_EQ(soc0->num_xcds(), 8u);
  ASSERT_EQ(soc1->num_xcds(), 8u);

  std::array<SoC *, 3> socs = {soc0, nullptr, soc1};
  EXPECT_EQ(amdgpu::clamp_xcd_partition_count(std::span<SoC *>(socs), 0), 1u);
  EXPECT_EQ(amdgpu::clamp_xcd_partition_count(std::span<SoC *>(socs), 4), 4u);
  EXPECT_EQ(amdgpu::clamp_xcd_partition_count(std::span<SoC *>(socs), 64), 16u);
  EXPECT_EQ(amdgpu::clamp_xcd_partition_count(soc0, 64), 8u);
  EXPECT_EQ(amdgpu::clamp_xcd_partition_count(static_cast<SoC *>(nullptr), 64), 1u);
}

TEST(XcdPartitioningTest, ZeroPartitionsIsNoopWithoutManualPartitions) {
  auto loaded = config::load_config(CONFIG_PATH, rocjitsu::kEmbeddedSchema);
  auto *soc = loaded.soc();
  simdojo::Topology topology;
  topology.set_root(loaded.take_root());
  loaded.wire_links(topology);

  EXPECT_FALSE(amdgpu::partition_topology_by_xcds(topology, soc, 0));
  EXPECT_TRUE(topology.partitions().empty());
  EXPECT_EQ(soc->partition_id(), simdojo::INVALID_PARTITION_ID);
  for (uint32_t i = 0; i < soc->num_xcds(); ++i)
    EXPECT_EQ(soc->xcd(i)->partition_id(), simdojo::INVALID_PARTITION_ID);
}

TEST(XcdPartitioningTest, NoXcdsIsNoopWithoutManualPartitions) {
  simdojo::Topology topology;

  EXPECT_FALSE(amdgpu::partition_topology_by_xcds(topology, std::span<SoC *>{}, 1));
  EXPECT_FALSE(amdgpu::partition_topology_by_xcds(topology, static_cast<SoC *>(nullptr), 1));
  EXPECT_TRUE(topology.partitions().empty());
}

TEST(XcdPartitioningTest, SinglePartitionAssignsAllComponentsToZero) {
  auto loaded = config::load_config(CONFIG_PATH, rocjitsu::kEmbeddedSchema);
  auto *soc = loaded.soc();
  simdojo::Topology topology;
  topology.set_root(loaded.take_root());
  loaded.wire_links(topology);

  ASSERT_TRUE(amdgpu::partition_topology_by_xcds(topology, soc, 1));
  ASSERT_EQ(topology.partitions().size(), 1u);
  for (auto *component : topology.collect_all_components())
    EXPECT_EQ(component->partition_id(), 0u) << component->full_path();
}

TEST(XcdPartitioningTest, ThreeThreadsDistributesCdna4XcdsRoundRobinWithoutSplits) {
  auto topology = build_partitioned_topology(3);

  ASSERT_TRUE(topology.partitioned);
  ASSERT_EQ(topology.engine->topology().partitions().size(), 3u);
  ASSERT_EQ(topology.soc->num_xcds(), 8u);

  for (uint32_t i = 0; i < topology.soc->num_xcds(); ++i)
    expect_subtree_partition(topology.soc->xcd(i), i % 3);
}

TEST(XcdPartitioningTest, SpanOverMultipleSocsUsesGlobalXcdIndex) {
  auto loaded = config::load_config(CONFIG_2GPU_PATH, rocjitsu::kEmbeddedSchema);
  ASSERT_EQ(loaded.extra_gpu_builds.size(), 1u);

  auto root = std::make_unique<simdojo::CompositeComponent>("system");
  auto *soc0 = dynamic_cast<SoC *>(root->add_child(loaded.take_root()));
  auto *soc1 = dynamic_cast<SoC *>(root->add_child(std::move(loaded.extra_gpu_builds[0].root)));
  ASSERT_NE(soc0, nullptr);
  ASSERT_NE(soc1, nullptr);

  simdojo::Topology topology;
  topology.set_root(std::move(root));
  std::array<SoC *, 2> socs = {soc0, soc1};
  ASSERT_TRUE(amdgpu::partition_topology_by_xcds(topology, std::span<SoC *>(socs), 3));
  ASSERT_EQ(topology.partitions().size(), 3u);

  uint32_t global_xcd_index = 0;
  for (auto *soc : socs) {
    ASSERT_EQ(soc->num_xcds(), 8u);
    for (uint32_t i = 0; i < soc->num_xcds(); ++i, ++global_xcd_index)
      expect_subtree_partition(soc->xcd(i), global_xcd_index % 3);
  }
}

TEST(XcdPartitioningTest, RejectsSocOutsideTopologyWithoutChangingPartitionState) {
  auto loaded = config::load_config(CONFIG_2GPU_PATH, rocjitsu::kEmbeddedSchema);
  ASSERT_EQ(loaded.extra_gpu_builds.size(), 1u);

  auto *soc0 = loaded.soc();
  auto *soc1 = dynamic_cast<SoC *>(loaded.extra_gpu_builds[0].root.get());
  ASSERT_NE(soc0, nullptr);
  ASSERT_NE(soc1, nullptr);

  simdojo::Topology topology;
  topology.set_root(loaded.take_root());
  std::array<SoC *, 2> socs = {soc0, soc1};
  EXPECT_FALSE(amdgpu::partition_topology_by_xcds(topology, std::span<SoC *>(socs), 16));

  EXPECT_TRUE(topology.partitions().empty());
  for (SoC *soc : socs) {
    EXPECT_EQ(soc->partition_id(), simdojo::INVALID_PARTITION_ID);
    for (uint32_t i = 0; i < soc->num_xcds(); ++i)
      EXPECT_EQ(soc->xcd(i)->partition_id(), simdojo::INVALID_PARTITION_ID);
  }
}

TEST(XcdPartitioningTest, NonXcdComponentsStayOnPartitionZero) {
  auto topology = build_partitioned_topology(8);

  ASSERT_TRUE(topology.partitioned);
  EXPECT_EQ(topology.soc->partition_id(), 0u);
  EXPECT_EQ(topology.memory->partition_id(), 0u);

  for (uint32_t i = 0; i < topology.soc->num_iods(); ++i)
    expect_subtree_partition(topology.soc->iod(i), 0);
}

TEST(XcdPartitioningTest, VmStepRejectsMultipleEnginePartitions) {
  auto json = config_json_with_num_threads(CONFIG_PATH, 2);

  rj_vm_t *raw_vm = nullptr;
  ASSERT_EQ(rj_vm_create_from_string(json.c_str(), RJ_VM_MODE_DEFAULT, &raw_vm),
            ROCJITSU_STATUS_SUCCESS);
  ASSERT_NE(raw_vm, nullptr);
  std::unique_ptr<rj_vm_t, decltype(&rj_vm_destroy)> vm(raw_vm, &rj_vm_destroy);

  int active = 7;
  EXPECT_EQ(rj_vm_step(vm.get(), &active), ROCJITSU_STATUS_UNSUPPORTED);
  EXPECT_EQ(active, 7);

  uint64_t ticks_executed = 0;
  EXPECT_EQ(rj_vm_run(vm.get(), &ticks_executed), ROCJITSU_STATUS_SUCCESS);
}

TEST(XcdPartitioningTest, VmStepSucceedsWithSingleEnginePartition) {
  auto json = config_json_with_num_threads(CONFIG_1XCD_PATH, 1);

  rj_vm_t *raw_vm = nullptr;
  ASSERT_EQ(rj_vm_create_from_string(json.c_str(), RJ_VM_MODE_DEFAULT, &raw_vm),
            ROCJITSU_STATUS_SUCCESS);
  ASSERT_NE(raw_vm, nullptr);
  std::unique_ptr<rj_vm_t, decltype(&rj_vm_destroy)> vm(raw_vm, &rj_vm_destroy);

  int active = 7;
  EXPECT_EQ(rj_vm_step(vm.get(), &active), ROCJITSU_STATUS_SUCCESS);
  EXPECT_EQ(active, 0);
}

TEST(XcdPartitioningTest, CApiResolvesZeroThreadsToDefault) {
  auto json = config_json_with_num_threads(CONFIG_1XCD_PATH, 0);

  rj_vm_t *raw_vm = nullptr;
  ASSERT_EQ(rj_vm_create_from_string(json.c_str(), RJ_VM_MODE_DEFAULT, &raw_vm),
            ROCJITSU_STATUS_SUCCESS);
  ASSERT_NE(raw_vm, nullptr);
  std::unique_ptr<rj_vm_t, decltype(&rj_vm_destroy)> vm(raw_vm, &rj_vm_destroy);

  EXPECT_EQ(vm->engine_config.num_threads, 1u);
  EXPECT_EQ(rj_vm_run(vm.get(), nullptr), ROCJITSU_STATUS_SUCCESS);
}

TEST(XcdPartitioningTest, CApiClampsSingleGpuThreadsToXcdCount) {
  auto json = config_json_with_num_threads(CONFIG_PATH, 64);

  rj_vm_t *raw_vm = nullptr;
  testing::internal::CaptureStderr();
  const rj_status_t status = rj_vm_create_from_string(json.c_str(), RJ_VM_MODE_DEFAULT, &raw_vm);
  const std::string warning = testing::internal::GetCapturedStderr();
  ASSERT_EQ(status, ROCJITSU_STATUS_SUCCESS);
  ASSERT_NE(raw_vm, nullptr);
  std::unique_ptr<rj_vm_t, decltype(&rj_vm_destroy)> vm(raw_vm, &rj_vm_destroy);

  EXPECT_NE(warning.find("num_threads clamped: requested=64, effective=8"), std::string::npos);
  EXPECT_EQ(vm->engine_config.num_threads, 8u);
  EXPECT_EQ(rj_vm_run(vm.get(), nullptr), ROCJITSU_STATUS_SUCCESS);
}

TEST(XcdPartitioningTest, CApiDoesNotWarnWhenThreadCountIsUnchanged) {
  auto json = config_json_with_num_threads(CONFIG_1XCD_PATH, 1);

  rj_vm_t *raw_vm = nullptr;
  testing::internal::CaptureStderr();
  const rj_status_t status = rj_vm_create_from_string(json.c_str(), RJ_VM_MODE_DEFAULT, &raw_vm);
  const std::string warning = testing::internal::GetCapturedStderr();
  ASSERT_EQ(status, ROCJITSU_STATUS_SUCCESS);
  ASSERT_NE(raw_vm, nullptr);
  std::unique_ptr<rj_vm_t, decltype(&rj_vm_destroy)> vm(raw_vm, &rj_vm_destroy);

  EXPECT_EQ(warning.find("num_threads clamped"), std::string::npos);
}

TEST(XcdPartitioningTest, DefaultsRespectHostBudgetAndXcdTopology) {
  auto loaded = config::load_config(CONFIG_PATH, rocjitsu::kEmbeddedSchema);
  auto *soc = loaded.soc();
  ASSERT_NE(soc, nullptr);
  ASSERT_EQ(soc->num_xcds(), 8u);

  // Stated host widths, so the assertions describe the rule instead of
  // recomputing it. Deriving the expectation from available_host_threads() the
  // way the resolver does leaves the test green under any rule the two sides
  // share -- swapping affinity for hardware_concurrency(), say, would move both.
  EXPECT_EQ(amdgpu::default_xcd_partition_count(soc, 1u), 1u);
  EXPECT_EQ(amdgpu::default_xcd_partition_count(soc, 3u), 3u);
  EXPECT_EQ(amdgpu::default_xcd_partition_count(soc, 8u), 8u);
  EXPECT_EQ(amdgpu::default_xcd_partition_count(soc, 64u), 8u) << "must cap at the XCD count";
  EXPECT_EQ(amdgpu::default_xcd_partition_count(soc, 0u), 1u) << "indeterminate host floors at 1";

  // The measuring overload has to agree with the stated-width one for the width
  // this process actually has; that is the only part affinity is load-bearing for.
  const uint32_t host_threads = amdgpu::available_host_threads();
  const uint32_t expected = host_threads == 0 ? 1u : std::min(host_threads, 8u);
  EXPECT_EQ(amdgpu::default_xcd_partition_count(soc), expected);

  // The shipped config omits num_threads, so loading it resolves the default.
  EXPECT_EQ(loaded.engine_config.num_threads, loaded.execution_threads.engines);
  // Its four-thread granule uses two engines and two retained dispatch workers.
  const auto four_threads =
      config::load_config(CONFIG_PATH, rocjitsu::kEmbeddedSchema, /*host_threads=*/4);
  EXPECT_EQ(four_threads.engine_config.num_threads, 2u);
  EXPECT_EQ(four_threads.execution_threads.dispatch, (std::vector<uint32_t>{3}));
  EXPECT_EQ(four_threads.execution_threads.helpers, 0u);
}

TEST(XcdPartitioningTest, DefaultThreadCountAggregatesSocsAndFloorsAtOne) {
  auto loaded = config::load_config(CONFIG_PATH, rocjitsu::kEmbeddedSchema);
  auto *soc = loaded.soc();
  ASSERT_NE(soc, nullptr);

  // Two views of the same 8-XCD SoC stand in for a 16-XCD multi-GPU VM: the
  // default counts XCDs across every SoC it is given.
  std::array<SoC *, 2> pair = {soc, soc};

  // A stated width above one SoC's XCD count is what separates aggregation from
  // counting only the first SoC: at 16 the correct answer is 16, and a
  // first-SoC-only implementation says 8. A width measured on a narrow runner
  // saturates first and hides the difference.
  EXPECT_EQ(amdgpu::default_xcd_partition_count(std::span<SoC *>(pair), 16u), 16u);
  EXPECT_EQ(amdgpu::default_xcd_partition_count(std::span<SoC *>(pair), 64u), 16u);
  EXPECT_EQ(amdgpu::default_xcd_partition_count(std::span<SoC *>(pair), 12u), 12u);
  EXPECT_EQ(amdgpu::default_xcd_partition_count(std::span<SoC *>(pair), 0u), 1u);

  const uint32_t host_threads = amdgpu::available_host_threads();
  const uint32_t expected = host_threads == 0 ? 1u : std::min(host_threads, 16u);
  EXPECT_EQ(amdgpu::default_xcd_partition_count(std::span<SoC *>(pair)), expected);

  // No SoCs at all still yields a runnable engine.
  std::array<SoC *, 1> none = {nullptr};
  EXPECT_EQ(amdgpu::default_xcd_partition_count(std::span<SoC *>(none)), 1u);
}

// 0 must mean "use the default" through the loader's own multi-GPU path. The
// one-XCD C API case above cannot show this: it yields 1 under both the old
// 0 -> 1 clamp and the new default, so only an aggregate config distinguishes
// them, and only load_config() covers extra_gpu_builds.
TEST(XcdPartitioningTest, LoaderResolvesZeroThreadsAcrossExtraGpuBuilds) {
  const std::string json = config_json_with_num_threads(CONFIG_2GPU_PATH, 0);
  config::LoadedConfig loaded = config::load_config_from_string(json, rocjitsu::kEmbeddedSchema);

  SoC *soc = loaded.soc();
  ASSERT_NE(soc, nullptr);
  ASSERT_EQ(loaded.num_gpus, 2u);
  ASSERT_EQ(loaded.extra_gpu_builds.size(), 1u);

  SoC *extra_soc = dynamic_cast<SoC *>(loaded.extra_gpu_builds[0].root.get());
  ASSERT_NE(extra_soc, nullptr);
  ASSERT_EQ(soc->num_xcds(), 8u);
  ASSERT_EQ(extra_soc->num_xcds(), 8u);

  // Zero selects the multi-GPU table, which preserves one engine while
  // allowing the dispatch pools to share the host budget.
  EXPECT_EQ(config::load_config_from_string(json, rocjitsu::kEmbeddedSchema, 16)
                .engine_config.num_threads,
            1u);
  EXPECT_EQ(
      config::load_config_from_string(json, rocjitsu::kEmbeddedSchema, 9).engine_config.num_threads,
      1u);
  EXPECT_EQ(config::load_config_from_string(json, rocjitsu::kEmbeddedSchema, 64)
                .engine_config.num_threads,
            1u);
  EXPECT_EQ(loaded.engine_config.num_threads, loaded.execution_threads.engines);
}

// The shipped topology configs take the default by omitting num_threads. Only
// the multi-GPU KMD configs pin it, because any multi-partition setting there
// hangs RCCL collectives (see docs/configuration.md). Enforcing that here rather
// than in a CI grep keeps every config covered as new ones are added.
TEST(XcdPartitioningTest, ShippedConfigsOmitNumThreadsExceptTheDocumentedPin) {
  const std::set<std::string> kPinnedConfigs = {
      "gfx950_mi355x_kmd_2gpu.json",
      "gfx1250_mi455x_kmd_4gpu.json",
  };
  std::set<std::string> saw_pinned_configs;
  uint32_t configs_checked = 0;

  for (const std::filesystem::directory_entry &entry :
       std::filesystem::directory_iterator(CONFIG_DIR)) {
    if (!entry.is_regular_file() || entry.path().extension() != ".json")
      continue;

    std::ifstream input(entry.path());
    ASSERT_TRUE(input.is_open()) << entry.path();
    const std::string json((std::istreambuf_iterator<char>(input)),
                           std::istreambuf_iterator<char>());
    const bool pins_num_threads = json.find("\n  \"num_threads\"") != std::string::npos;
    ++configs_checked;

    if (kPinnedConfigs.count(entry.path().filename().string()) != 0) {
      saw_pinned_configs.insert(entry.path().filename().string());
      EXPECT_TRUE(pins_num_threads) << entry.path().filename() << " must keep its documented pin";
      // The pin's whole purpose is a single partition: docs/configuration.md
      // says any multi-partition value hangs RCCL collectives on these configs.
      // Checking only that the key is present would let 2 or 64 through, which
      // is the exact state the pin exists to prevent.
      //
      // The stated width has to exceed 1, or the assertion cannot see the
      // difference: resolving against the live affinity of a one-CPU runner
      // turns a pin of 0 -- which opts back into the default -- into 1 as
      // surely as a real pin does. At 64 the config has to carry the literal 1.
      const config::LoadedConfig pinned =
          config::load_config(entry.path().string(), rocjitsu::kEmbeddedSchema,
                              /*host_threads=*/64);
      EXPECT_EQ(pinned.engine_config.num_threads, 1u)
          << entry.path().filename() << " must pin num_threads to exactly 1";
    } else {
      EXPECT_FALSE(pins_num_threads)
          << entry.path().filename()
          << " pins num_threads, so it no longer exercises the default partition count";
    }
  }

  EXPECT_EQ(saw_pinned_configs, kPinnedConfigs)
      << "expected every documented pinned config in " << CONFIG_DIR;
  EXPECT_GT(configs_checked, 1u);
}

TEST(XcdPartitioningTest, CApiDefaultsToOnePartitionPerXcdAndRuns) {
  rj_vm_t *raw_vm = nullptr;
  testing::internal::CaptureStderr();
  const rj_status_t status = rj_vm_create(CONFIG_PATH.c_str(), RJ_VM_MODE_DEFAULT, &raw_vm);
  const std::string warning = testing::internal::GetCapturedStderr();
  ASSERT_EQ(status, ROCJITSU_STATUS_SUCCESS);
  ASSERT_NE(raw_vm, nullptr);
  std::unique_ptr<rj_vm_t, decltype(&rj_vm_destroy)> vm(raw_vm, &rj_vm_destroy);

  // Resolving the default is not a clamp, so it must not warn.
  EXPECT_EQ(warning.find("num_threads clamped"), std::string::npos);
  EXPECT_EQ(vm->engine_config.num_threads, amdgpu::default_xcd_partition_count(vm->soc));

  ASSERT_NE(vm->soc, nullptr);
  for (uint32_t i = 0; i < vm->soc->num_xcds(); ++i)
    expect_subtree_partition(vm->soc->xcd(i), i % vm->engine_config.num_threads);

  EXPECT_EQ(rj_vm_run(vm.get(), nullptr), ROCJITSU_STATUS_SUCCESS);
}

TEST(XcdPartitioningTest, CApiClampsMultiGpuThreadsToTotalXcdCountAndRuns) {
  auto json = config_json_with_num_threads(CONFIG_2GPU_PATH, 64);

  rj_vm_t *raw_vm = nullptr;
  ASSERT_EQ(rj_vm_create_from_string(json.c_str(), RJ_VM_MODE_DEFAULT, &raw_vm),
            ROCJITSU_STATUS_SUCCESS);
  ASSERT_NE(raw_vm, nullptr);
  std::unique_ptr<rj_vm_t, decltype(&rj_vm_destroy)> vm(raw_vm, &rj_vm_destroy);

  EXPECT_EQ(vm->engine_config.num_threads, 16u);
  ASSERT_NE(vm->vm, nullptr);
  ASSERT_EQ(vm->vm->num_socs(), 2u);

  uint32_t global_xcd_index = 0;
  for (uint32_t gpu = 0; gpu < vm->vm->num_socs(); ++gpu) {
    auto *soc = vm->vm->soc(gpu);
    ASSERT_NE(soc, nullptr);
    ASSERT_EQ(soc->num_xcds(), 8u);
    for (uint32_t i = 0; i < soc->num_xcds(); ++i, ++global_xcd_index)
      expect_subtree_partition(soc->xcd(i), global_xcd_index % vm->engine_config.num_threads);
  }

  EXPECT_EQ(rj_vm_run(vm.get(), nullptr), ROCJITSU_STATUS_SUCCESS);
}

} // namespace
