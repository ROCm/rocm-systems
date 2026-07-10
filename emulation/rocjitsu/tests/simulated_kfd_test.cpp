// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file simulated_kfd_test.cpp
/// @brief Tests for SimulatedKfd creation, open/close, and topology generation.

#include "rocjitsu/config/config_loader.h"
#include "rocjitsu/kmd/linux/simulated_kfd.h"
#include "rocjitsu/vm/virtual_machine.h"

#include "embedded_schema.h"
#include "simdojo/sim/simulation.h"

#include <gtest/gtest.h>

#include <cstdlib>
#include <filesystem>

namespace {

const std::string CONFIG_PATH = std::string(CONFIG_DIR) + "/gfx950_cdna4.json";

struct TestVM {
  rocjitsu::config::LoadedConfig loaded;
  std::unique_ptr<simdojo::SimulationEngine> engine;

  rocjitsu::SimulatedKfd *driver() {
    auto *vm = dynamic_cast<rocjitsu::VirtualMachine *>(engine->topology().root());
    return vm ? vm->driver() : nullptr;
  }
};

TestVM create_test_vm() {
  TestVM t;
  t.loaded = rocjitsu::config::load_config(CONFIG_PATH.c_str(), rocjitsu::kEmbeddedSchema);
  auto *soc = t.loaded.soc();

  t.loaded.engine_config.max_ticks = 0;
  t.loaded.engine_config.await_primaries = true;
  t.engine = std::make_unique<simdojo::SimulationEngine>(t.loaded.engine_config);

  auto root = t.loaded.take_root();
  root.release();
  auto vm = std::make_unique<rocjitsu::VirtualMachine>(std::unique_ptr<rocjitsu::SoC>(soc));
  vm->driver()->setup_topology(t.loaded.device, soc->num_xcds());

  t.engine->topology().set_root(std::move(vm));
  t.loaded.wire_links(t.engine->topology());
  soc->wire_backing(t.engine->topology());
  t.engine->build();
  t.engine->register_as_primary();

  return t;
}

class SimulatedKfdTest : public ::testing::Test {
protected:
  void SetUp() override { setenv("RJ_CONFIG", CONFIG_PATH.c_str(), 1); }
};

TEST_F(SimulatedKfdTest, CreateDefault) {
  auto t = create_test_vm();
  ASSERT_NE(t.driver(), nullptr);
}

TEST_F(SimulatedKfdTest, OpenAndClose) {
  auto t = create_test_vm();
  ASSERT_NE(t.driver(), nullptr);

  int fd = t.driver()->open();
  EXPECT_GE(fd, 0);

  int ret = t.driver()->close();
  EXPECT_EQ(ret, 0);
}

TEST_F(SimulatedKfdTest, TopologyDirectoryExists) {
  auto t = create_test_vm();
  ASSERT_NE(t.driver(), nullptr);

  const auto &path = t.driver()->topology().path();
  EXPECT_FALSE(path.empty());
  EXPECT_TRUE(std::filesystem::exists(path));
  EXPECT_TRUE(std::filesystem::exists(path + "/generation_id"));
  EXPECT_TRUE(std::filesystem::exists(path + "/nodes/0/properties"));
  EXPECT_TRUE(std::filesystem::exists(path + "/nodes/1/properties"));
}

} // namespace
