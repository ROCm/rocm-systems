// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "aql_queue.h"

#include "rocjitsu/config/checkpoint.h"
#include "rocjitsu/config/config_loader.h"
#include "rocjitsu/vm/rj_vm.h"
#include "rocjitsu/vm/soc.h"

#include "simdojo/sim/simulation.h"

#include "rocjitsu/base/rj_compiler.h"
RJ_DIAGNOSTIC_PUSH
RJ_DIAGNOSTIC_IGNORE_PEDANTIC
#include "hsa/AMDHSAKernelDescriptor.h"
RJ_DIAGNOSTIC_POP

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <string>

namespace {

const std::string SCHEMA_DIR_PATH = SCHEMA_DIR;
const std::string CONFIG_DIR_PATH = CONFIG_DIR;
const std::string SCHEMA_PATH = SCHEMA_DIR_PATH + "/simulation_config.fbs";

using namespace rocjitsu;

TEST(ConfigLoaderTest, LoadCdna4Config) {
  std::string json = CONFIG_DIR_PATH + "/amdgpu_cdna4.json";
  auto loaded = config::load_config(json, SCHEMA_PATH);
  auto *soc = loaded.soc();

  // CDNA4 config: 8 XCDs, 4 SEs per XCD, 8 CUs per SE, 2 IODs.
  EXPECT_EQ(soc->num_xcds(), 8u);
  EXPECT_EQ(soc->num_iods(), 2u);
  auto *xcd = soc->xcd(0);
  EXPECT_EQ(xcd->num_shader_engines(), 4u);
  EXPECT_EQ(xcd->shader_engine(0)->num_compute_units(), 8u);
}

TEST(ConfigLoaderTest, BuildFromJsonString) {
  const char *json = R"({
    "max_ticks": 5000,
    "num_threads": 1,
    "vm": { "arch": "cdna3" },
    "topology": {
      "root": {
        "name": "soc", "type": "soc",
        "children": [
          { "name": "vram", "type": "gpu_memory" },
          {
            "name": "xcd0", "type": "xcd",
            "children": [
              { "name": "l2", "type": "l2_cache" },
              { "name": "cp", "type": "command_processor" },
              {
                "name": "se0", "type": "shader_engine",
                "children": [{
                  "name": "cu[0:3]", "type": "compute_unit",
                  "config": [
                    { "key": "num_wf_slots", "value": "20" },
                    { "key": "sgprs_per_wf", "value": "104" },
                    { "key": "vgprs_per_wf", "value": "256" },
                    { "key": "lds_size_kb", "value": "64" }
                  ]
                }]
              },
              {
                "name": "se1", "type": "shader_engine",
                "children": [{
                  "name": "cu[0:3]", "type": "compute_unit",
                  "config": [
                    { "key": "num_wf_slots", "value": "20" },
                    { "key": "sgprs_per_wf", "value": "104" },
                    { "key": "vgprs_per_wf", "value": "256" },
                    { "key": "lds_size_kb", "value": "64" }
                  ]
                }]
              }
            ]
          }
        ]
      },
      "links": [
        { "src": "xcd0.cp.req_0", "dst": "xcd0.se0.cu0.cpl", "latency": 1, "weight": 2 },
        { "src": "xcd0.cp.req_1", "dst": "xcd0.se0.cu1.cpl", "latency": 1, "weight": 2 },
        { "src": "xcd0.cp.req_2", "dst": "xcd0.se0.cu2.cpl", "latency": 1, "weight": 2 },
        { "src": "xcd0.cp.req_3", "dst": "xcd0.se1.cu0.cpl", "latency": 1, "weight": 2 },
        { "src": "xcd0.cp.req_4", "dst": "xcd0.se1.cu1.cpl", "latency": 1, "weight": 2 },
        { "src": "xcd0.cp.req_5", "dst": "xcd0.se1.cu2.cpl", "latency": 1, "weight": 2 },
        { "src": "xcd0.se0.cu0.req", "dst": "xcd0.l2.cpl_0", "latency": 1, "weight": 10 },
        { "src": "xcd0.se0.cu1.req", "dst": "xcd0.l2.cpl_1", "latency": 1, "weight": 10 },
        { "src": "xcd0.se0.cu2.req", "dst": "xcd0.l2.cpl_2", "latency": 1, "weight": 10 },
        { "src": "xcd0.se1.cu0.req", "dst": "xcd0.l2.cpl_3", "latency": 1, "weight": 10 },
        { "src": "xcd0.se1.cu1.req", "dst": "xcd0.l2.cpl_4", "latency": 1, "weight": 10 },
        { "src": "xcd0.se1.cu2.req", "dst": "xcd0.l2.cpl_5", "latency": 1, "weight": 10 }
      ]
    }
  })";

  auto loaded = config::load_config_from_string(json, SCHEMA_PATH);
  auto *soc = loaded.soc();

  // 1 XCD, 2 SEs, each with 3 CUs.
  auto *xcd = soc->xcd(0);
  EXPECT_EQ(xcd->num_shader_engines(), 2u);
  EXPECT_EQ(xcd->shader_engine(0)->num_compute_units(), 3u);
  EXPECT_EQ(xcd->shader_engine(1)->num_compute_units(), 3u);
}

TEST(ConfigLoaderTest, DispatchDistributesAcrossCUs) {
  const char *json = R"({"max_ticks":10000,"num_threads":1,
    "vm":{"arch":"cdna3"},
    "topology":{
      "root":{
        "name":"soc","type":"soc",
        "children":[
          {"name":"vram","type":"gpu_memory"},
          {"name":"xcd0","type":"xcd","children":[
            {"name":"l2","type":"l2_cache"},
            {"name":"cp","type":"command_processor"},
            {"name":"se0","type":"shader_engine","children":[
              {"name":"cu[0:2]","type":"compute_unit","config":[
                {"key":"num_wf_slots","value":"10"},
                {"key":"sgprs_per_wf","value":"104"},
                {"key":"vgprs_per_wf","value":"256"},
                {"key":"lds_size_kb","value":"64"}
              ]}
            ]}
          ]}
        ]
      },
      "links":[
        {"src":"xcd0.cp.req_0","dst":"xcd0.se0.cu0.cpl","latency":1,"weight":2},
        {"src":"xcd0.cp.req_1","dst":"xcd0.se0.cu1.cpl","latency":1,"weight":2},
        {"src":"xcd0.se0.cu0.req","dst":"xcd0.l2.cpl_0","latency":1,"weight":10},
        {"src":"xcd0.se0.cu1.req","dst":"xcd0.l2.cpl_1","latency":1,"weight":10}
      ]
    }
  })";

  auto loaded = config::load_config_from_string(json, SCHEMA_PATH);
  auto *soc = loaded.soc();

  simdojo::SimulationEngine engine(loaded.engine_config);
  engine.topology().set_root(loaded.take_root());
  loaded.wire_links(engine.topology());
  engine.build();

  // Write a kernel descriptor + invalid instruction so wavefronts halt immediately.
  using namespace rocr::llvm::amdhsa;
  kernel_descriptor_t kd{};
  kd.kernel_code_entry_byte_offset = sizeof(kernel_descriptor_t);
  // CDNA3 (GFX940+) uses VGPR granularity 8 (not 4).
  AMDHSA_BITS_SET(kd.compute_pgm_rsrc1, COMPUTE_PGM_RSRC1_GRANULATED_WORKITEM_VGPR_COUNT,
                  ((256 / 8) - 1));
  AMDHSA_BITS_SET(kd.compute_pgm_rsrc1, COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT,
                  ((104 / 8) - 1));
  AMDHSA_BITS_SET(kd.compute_pgm_rsrc2, COMPUTE_PGM_RSRC2_USER_SGPR_COUNT, 2);

  constexpr uint64_t KD_ADDR = 0x1000;
  soc->memory()->load_image(reinterpret_cast<const uint8_t *>(&kd), sizeof(kd), KD_ADDR);
  soc->memory()->write32(KD_ADDR + sizeof(kernel_descriptor_t), 0xFFFFFFFF); // invalid instruction

  auto *xcd = soc->xcd(0);
  test::AqlQueue queue(soc->memory(), xcd->command_processor());
  queue.dispatch(KD_ADDR, 128, 64); // grid_size=128 = 2 workgroups of 64

  engine.step();

  // After one step, the doorbell event dispatched wavefronts to CUs
  // (allocated but not yet executed). Verify round-robin distribution.
  EXPECT_EQ(xcd->command_processor()->dispatched_count(), 1u);
  auto *se = soc->xcd(0)->shader_engine(0);
  EXPECT_EQ(se->compute_unit(0)->num_wfs(), 1u);
  EXPECT_EQ(se->compute_unit(1)->num_wfs(), 1u);
}

TEST(CheckpointTest, SaveAndRestoreMemory) {
  const char *json = R"({"max_ticks":10000,"num_threads":1,
    "vm":{"arch":"cdna3"},
    "topology":{
      "root":{
        "name":"soc","type":"soc",
        "children":[
          {"name":"vram","type":"gpu_memory"},
          {"name":"xcd0","type":"xcd","children":[
            {"name":"l2","type":"l2_cache"},
            {"name":"cp","type":"command_processor"},
            {"name":"se0","type":"shader_engine","children":[
              {"name":"cu[0:1]","type":"compute_unit","config":[
                {"key":"num_wf_slots","value":"10"},
                {"key":"sgprs_per_wf","value":"104"},
                {"key":"vgprs_per_wf","value":"256"},
                {"key":"lds_size_kb","value":"64"}
              ]}
            ]}
          ]}
        ]
      },
      "links":[
        {"src":"xcd0.cp.req_0","dst":"xcd0.se0.cu0.cpl","latency":1,"weight":2},
        {"src":"xcd0.se0.cu0.req","dst":"xcd0.l2.cpl_0","latency":1,"weight":10}
      ]
    }
  })";

  auto loaded = config::load_config_from_string(json, SCHEMA_PATH);
  auto *soc = loaded.soc();

  soc->memory()->write32(0x1000, 0xDEADBEEF);
  soc->memory()->write64(0x2000, 0x0123456789ABCDEFULL);

  const char *path = "/tmp/rocjitsu_test_checkpoint.bin";
  config::save_checkpoint(path, *soc, 42, loaded.engine_config);
  ASSERT_TRUE(std::filesystem::exists(path));

  auto restored = config::restore_checkpoint(path);
  EXPECT_EQ(restored.memory()->read32(0x1000), 0xDEADBEEFu);
  EXPECT_EQ(restored.memory()->read64(0x2000), 0x0123456789ABCDEFULL);

  std::filesystem::remove(path);
}

TEST(CApiTest, CreateAndDestroyFromString) {
  const char *json = R"({"max_ticks":10000,"num_threads":1,
    "vm":{"arch":"cdna3"},
    "topology":{
      "root":{
        "name":"soc","type":"soc",
        "children":[
          {"name":"vram","type":"gpu_memory"},
          {"name":"xcd0","type":"xcd","children":[
            {"name":"l2","type":"l2_cache"},
            {"name":"cp","type":"command_processor"},
            {"name":"se0","type":"shader_engine","children":[
              {"name":"cu[0:1]","type":"compute_unit","config":[
                {"key":"num_wf_slots","value":"10"},
                {"key":"sgprs_per_wf","value":"104"},
                {"key":"vgprs_per_wf","value":"256"},
                {"key":"lds_size_kb","value":"64"}
              ]}
            ]}
          ]}
        ]
      },
      "links":[
        {"src":"xcd0.cp.req_0","dst":"xcd0.se0.cu0.cpl","latency":1,"weight":2},
        {"src":"xcd0.se0.cu0.req","dst":"xcd0.l2.cpl_0","latency":1,"weight":10}
      ]
    }
  })";
  rj_vm_t *handle = nullptr;
  EXPECT_EQ(rj_vm_create_from_string(json, SCHEMA_PATH.c_str(), &handle), ROCJITSU_STATUS_SUCCESS);
  ASSERT_NE(handle, nullptr);
  rj_vm_destroy(handle);
}

TEST(CApiTest, InvalidArguments) {
  rj_vm_t *handle = nullptr;
  EXPECT_EQ(rj_vm_create_from_string(nullptr, SCHEMA_PATH.c_str(), &handle),
            ROCJITSU_STATUS_INVALID_ARGUMENT);
  EXPECT_EQ(rj_vm_step(nullptr, nullptr), ROCJITSU_STATUS_INVALID_ARGUMENT);
}

} // namespace
