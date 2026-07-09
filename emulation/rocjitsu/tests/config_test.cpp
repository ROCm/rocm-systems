// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "aql_queue.h"

#include "embedded_schema.h"
#include "rocjitsu/config/checkpoint.h"
#include "rocjitsu/config/config_loader.h"
#include "rocjitsu/isa/arch/amdgpu/cdna3/isa.h"
#include "rocjitsu/isa/arch/amdgpu/shared/accvgpr_layout.h"
#include "rocjitsu/kmd/linux/amdgpu_properties.h"
#include "rocjitsu/vm/rj_vm.h"
#include "rocjitsu/vm/soc.h"
#include "rocjitsu/vm/virtual_machine.h"

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

const std::string CONFIG_DIR_PATH = CONFIG_DIR;

// \NPI new GPU: add a config-load test for its configs/<gpu>.json here.
using namespace rocjitsu;

TEST(ConfigLoaderTest, LoadCdna4Config) {
  std::string json = CONFIG_DIR_PATH + "/gfx950_cdna4.json";
  auto loaded = config::load_config(json, rocjitsu::kEmbeddedSchema);
  auto *soc = loaded.soc();

  // CDNA4 config: 8 XCDs, 4 SEs per XCD, 8 CUs per SE, 2 IODs.
  EXPECT_EQ(soc->num_xcds(), 8u);
  EXPECT_EQ(soc->num_iods(), 2u);
  auto *xcd = soc->xcd(0);
  EXPECT_EQ(xcd->num_shader_engines(), 4u);
  EXPECT_EQ(xcd->shader_engine(0)->num_compute_units(), 8u);
}

TEST(ConfigLoaderTest, LoadRdnaKmdConfigs) {
  auto rdna4 =
      config::load_config(CONFIG_DIR_PATH + "/gfx1201_r9700.json", rocjitsu::kEmbeddedSchema);
  EXPECT_EQ(rdna4.soc()->arch(), ROCJITSU_CODE_ARCH_RDNA4);
  EXPECT_EQ(rdna4.device.gpu_id, 8716u);
  EXPECT_EQ(rdna4.device.device_id, 0x7551u);
  EXPECT_EQ(rdna4.device.family_id, 0x98u);
  EXPECT_EQ(rdna4.device.gfx_target_version, 120001u);
  EXPECT_EQ(rdna4.device.revision_id, 1u);
  EXPECT_EQ(rdna4.device.pci_revision_id, 192u);
  EXPECT_EQ(rdna4.device.simd_count, 128u);
  EXPECT_EQ(rdna4.device.num_shader_engines, 8u);
  EXPECT_EQ(rdna4.device.num_shader_arrays_per_engine, 2u);
  EXPECT_EQ(rdna4.device.num_cu_per_sh, 8u);
  EXPECT_EQ(rdna4.device.simd_per_cu, 2u);
  EXPECT_EQ(rdna4.device.vram_type, kmd::kAmdgpuVramTypeGddr6);
  EXPECT_EQ(rdna4.device.simd_count, rdna4.device.num_shader_engines * rdna4.device.num_cu_per_sh *
                                         rdna4.device.simd_per_cu);
  EXPECT_EQ(kmd::drm_shader_engine_count(rdna4.device.num_shader_engines,
                                         rdna4.device.num_shader_arrays_per_engine),
            4u);
  EXPECT_EQ(kmd::drm_cu_active_number(rdna4.device.num_shader_engines, rdna4.device.num_cu_per_sh),
            64u);
  EXPECT_EQ(kmd::external_rev_id_for_gfx_target_version(rdna4.device.gfx_target_version,
                                                        rdna4.device.revision_id),
            0x51u);
  EXPECT_EQ(kmd::gfx_target_name(rdna4.device.gfx_target_version), "gfx1201");
  EXPECT_EQ(kmd::gfx_target_name(90010), "gfx90a");
  EXPECT_EQ(kmd::gb_addr_config_for_arch(ROCJITSU_CODE_ARCH_RDNA3_5), 0u);
  EXPECT_EQ(kmd::gb_addr_config_for_gfx_target_version(110500), 0u);
  EXPECT_EQ(kmd::gb_addr_config_for_gfx_target_version(120500), 0u);
  EXPECT_EQ(kmd::drm_shader_engine_count(0, 2), 0u);
  EXPECT_EQ(kmd::drm_shader_engine_count(1, 2), 1u);
  EXPECT_EQ(kmd::drm_shader_engine_count(3, 2), 2u);
  EXPECT_EQ(kmd::drm_shader_engine_count(3, 0), 3u);
  EXPECT_EQ(kmd::num_hw_gfx_contexts_for_gfx_target_version(rdna4.device.gfx_target_version), 8u);
  EXPECT_EQ(rdna4.soc()->num_xcds(), 1u);
  EXPECT_EQ(rdna4.soc()->xcd(0)->num_shader_engines(), 4u);
  EXPECT_EQ(rdna4.soc()->xcd(0)->shader_engine(0)->num_compute_units(), 16u);
  EXPECT_FALSE(rdna4.soc()->xcd(0)->command_processor()->packed_tid());
  EXPECT_EQ(rdna4.soc()->xcd(0)->command_processor()->sdma_packet_dialect(),
            amdgpu::SdmaPacketDialect::Gfx11Plus);

  auto rdna3 =
      config::load_config(CONFIG_DIR_PATH + "/gfx1100_w7900.json", rocjitsu::kEmbeddedSchema);
  EXPECT_EQ(rdna3.soc()->arch(), ROCJITSU_CODE_ARCH_RDNA3);
  EXPECT_EQ(rdna3.device.gpu_id, 7019u);
  EXPECT_EQ(rdna3.device.device_id, 0x7448u);
  EXPECT_EQ(rdna3.device.family_id, 0x91u);
  EXPECT_EQ(rdna3.device.gfx_target_version, 110000u);
  EXPECT_EQ(rdna3.device.revision_id, 0u);
  EXPECT_EQ(rdna3.device.pci_revision_id, 0u);
  EXPECT_EQ(rdna3.device.simd_count, 192u);
  EXPECT_EQ(rdna3.device.num_shader_engines, 12u);
  EXPECT_EQ(rdna3.device.num_shader_arrays_per_engine, 2u);
  EXPECT_EQ(rdna3.device.num_cu_per_sh, 8u);
  EXPECT_EQ(rdna3.device.simd_per_cu, 2u);
  EXPECT_EQ(rdna3.device.vram_type, kmd::kAmdgpuVramTypeGddr6);
  EXPECT_EQ(rdna3.device.simd_count, rdna3.device.num_shader_engines * rdna3.device.num_cu_per_sh *
                                         rdna3.device.simd_per_cu);
  EXPECT_EQ(kmd::drm_shader_engine_count(rdna3.device.num_shader_engines,
                                         rdna3.device.num_shader_arrays_per_engine),
            6u);
  EXPECT_EQ(kmd::drm_cu_active_number(rdna3.device.num_shader_engines, rdna3.device.num_cu_per_sh),
            96u);
  EXPECT_EQ(kmd::external_rev_id_for_gfx_target_version(rdna3.device.gfx_target_version,
                                                        rdna3.device.revision_id),
            0x1u);
  EXPECT_EQ(kmd::gfx_target_name(rdna3.device.gfx_target_version), "gfx1100");
  EXPECT_EQ(kmd::num_hw_gfx_contexts_for_gfx_target_version(rdna3.device.gfx_target_version), 8u);
  EXPECT_EQ(rdna3.soc()->num_xcds(), 1u);
  EXPECT_EQ(rdna3.soc()->xcd(0)->num_shader_engines(), 6u);
  EXPECT_EQ(rdna3.soc()->xcd(0)->shader_engine(0)->num_compute_units(), 16u);
  EXPECT_FALSE(rdna3.soc()->xcd(0)->command_processor()->packed_tid());
  EXPECT_EQ(rdna3.soc()->xcd(0)->command_processor()->sdma_packet_dialect(),
            amdgpu::SdmaPacketDialect::Gfx11Plus);

  auto rdna35 = config::load_config(CONFIG_DIR_PATH + "/gfx1151.json", rocjitsu::kEmbeddedSchema);
  EXPECT_EQ(rdna35.soc()->arch(), ROCJITSU_CODE_ARCH_RDNA3_5);
  EXPECT_EQ(rdna35.device.gpu_id, 5510u);
  EXPECT_EQ(rdna35.device.device_id, 0x1586u);
  EXPECT_EQ(rdna35.device.family_id, 0x91u);
  EXPECT_EQ(rdna35.device.gfx_target_version, 110501u);
  EXPECT_EQ(rdna35.device.revision_id, 0u);
  EXPECT_EQ(rdna35.device.pci_revision_id, 0u);
  EXPECT_EQ(rdna35.device.simd_count, 64u);
  EXPECT_EQ(rdna35.device.num_shader_engines, 4u);
  EXPECT_EQ(rdna35.device.num_shader_arrays_per_engine, 2u);
  EXPECT_EQ(rdna35.device.num_cu_per_sh, 8u);
  EXPECT_EQ(rdna35.device.simd_per_cu, 2u);
  EXPECT_EQ(rdna35.device.vram_type, kmd::kAmdgpuVramTypeGddr6);
  EXPECT_EQ(rdna35.device.simd_count, rdna35.device.num_shader_engines *
                                          rdna35.device.num_cu_per_sh * rdna35.device.simd_per_cu);
  EXPECT_EQ(kmd::drm_shader_engine_count(rdna35.device.num_shader_engines,
                                         rdna35.device.num_shader_arrays_per_engine),
            2u);
  EXPECT_EQ(
      kmd::drm_cu_active_number(rdna35.device.num_shader_engines, rdna35.device.num_cu_per_sh),
      32u);
  EXPECT_EQ(kmd::external_rev_id_for_gfx_target_version(rdna35.device.gfx_target_version,
                                                        rdna35.device.revision_id),
            0xc1u);
  EXPECT_EQ(kmd::gfx_target_name(rdna35.device.gfx_target_version), "gfx1151");
  EXPECT_EQ(kmd::num_hw_gfx_contexts_for_gfx_target_version(rdna35.device.gfx_target_version), 8u);
  EXPECT_EQ(rdna35.soc()->num_xcds(), 1u);
  EXPECT_EQ(rdna35.soc()->xcd(0)->num_shader_engines(), 2u);
  EXPECT_EQ(rdna35.soc()->xcd(0)->shader_engine(0)->num_compute_units(), 16u);
  EXPECT_FALSE(rdna35.soc()->xcd(0)->command_processor()->packed_tid());
  EXPECT_EQ(rdna35.soc()->xcd(0)->command_processor()->sdma_packet_dialect(),
            amdgpu::SdmaPacketDialect::Gfx11Plus);
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

  auto loaded = config::load_config_from_string(json, rocjitsu::kEmbeddedSchema);
  auto *soc = loaded.soc();

  // 1 XCD, 2 SEs, each with 3 CUs.
  auto *xcd = soc->xcd(0);
  EXPECT_EQ(xcd->num_shader_engines(), 2u);
  EXPECT_EQ(xcd->shader_engine(0)->num_compute_units(), 3u);
  EXPECT_EQ(xcd->shader_engine(1)->num_compute_units(), 3u);
}

TEST(ConfigLoaderTest, DeviceCapabilityFieldsDefaultToAutoCompute) {
  const char *json = R"({
    "max_ticks": 5000,
    "num_threads": 1,
    "vm": {
      "arch": "cdna3",
      "gpu": { "device": { "gfx_target_version": 90500 } }
    },
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
              { "name": "se0", "type": "shader_engine",
                "children": [{ "name": "cu0", "type": "compute_unit" }] }
            ]
          }
        ]
      },
      "links": []
    }
  })";

  auto loaded = config::load_config_from_string(json, rocjitsu::kEmbeddedSchema);

  // Not specified in JSON: 0 means "auto-compute" (see
  // rocjitsu::default_non_debug_capability()/debug_topology_for()).
  EXPECT_EQ(loaded.device.capability, 0u);
  EXPECT_EQ(loaded.device.capability2, 0u);
  EXPECT_EQ(loaded.device.debug_prop, 0u);
}

TEST(ConfigLoaderTest, DeviceCapabilityFieldsRoundTripFromJson) {
  const char *json = R"({
    "max_ticks": 5000,
    "num_threads": 1,
    "vm": {
      "arch": "cdna3",
      "gpu": { "device": {
        "gfx_target_version": 90500,
        "capability": 268468354,
        "capability2": 3,
        "debug_prop": 3119
      } }
    },
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
              { "name": "se0", "type": "shader_engine",
                "children": [{ "name": "cu0", "type": "compute_unit" }] }
            ]
          }
        ]
      },
      "links": []
    }
  })";

  auto loaded = config::load_config_from_string(json, rocjitsu::kEmbeddedSchema);

  EXPECT_EQ(loaded.device.capability, 268468354u);
  EXPECT_EQ(loaded.device.capability2, 3u);
  EXPECT_EQ(loaded.device.debug_prop, 3119u);
}

TEST(ConfigLoaderTest, Gfx1250ComputeUnitDefaultsCoverTtmpAndHighVgprs) {
  const char *json = R"({"max_ticks":1000,"num_threads":1,
    "vm":{"arch":"gfx1250"},
    "topology":{"root":{"name":"soc","type":"soc","children":[
      {"name":"vram","type":"gpu_memory"},
      {"name":"xcd0","type":"xcd","children":[
        {"name":"l2","type":"l2_cache"},
        {"name":"cp","type":"command_processor"},
        {"name":"se0","type":"shader_engine","children":[
          {"name":"cu[0:1]","type":"compute_unit","config":[
            {"key":"num_wf_slots","value":"1"},
            {"key":"lds_size_kb","value":"64"}
          ]}
        ]}
      ]}
    ]},"links":[
      {"src":"xcd0.cp.req_0","dst":"xcd0.se0.cu0.cpl","latency":1,"weight":2},
      {"src":"xcd0.se0.cu0.req","dst":"xcd0.l2.cpl_0","latency":1,"weight":10}
    ]}})";

  auto loaded = config::load_config_from_string(json, rocjitsu::kEmbeddedSchema);
  auto *cu = loaded.soc()->xcd(0)->shader_engine(0)->compute_unit(0);
  ASSERT_NE(cu, nullptr);
  EXPECT_EQ(cu->config().sgprs_per_wf, 128u);
  EXPECT_EQ(cu->config().vgprs_per_wf, 1024u);
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

  auto loaded = config::load_config_from_string(json, rocjitsu::kEmbeddedSchema);
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

  auto loaded = config::load_config_from_string(json, rocjitsu::kEmbeddedSchema);
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

TEST(CheckpointTest, SaveAndRestoreAccVgprs) {
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
                {"key":"num_wf_slots","value":"1"},
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

  auto loaded = config::load_config_from_string(json, rocjitsu::kEmbeddedSchema);
  auto *cu = loaded.soc()->xcd(0)->shader_engine(0)->compute_unit(0);
  ASSERT_NE(cu, nullptr);

  auto *wf = cu->dispatch_wf(0, 0, cu->config().sgprs_per_wf, cu->config().vgprs_per_wf);
  ASSERT_NE(wf, nullptr);
  const uint32_t acc0 = wf->vgpr_alloc().base + amdgpu::ACC_VGPR_OFFSET;
  const uint32_t acc_last = acc0 + cdna3::Isa::MAX_ACC_VGPRS_PER_WF - 1;
  cu->write_vgpr(acc0, 0, 0xA55A0001u);
  cu->write_vgpr(acc_last, 0, 0xDEADBEEFu);

  const char *path = "/tmp/rocjitsu_test_checkpoint_accvgpr.bin";
  config::save_checkpoint(path, *loaded.soc(), 42, loaded.engine_config);
  ASSERT_TRUE(std::filesystem::exists(path));

  auto restored = config::restore_checkpoint(path);
  auto *restored_vm = dynamic_cast<VirtualMachine *>(restored.build_result.root.get());
  ASSERT_NE(restored_vm, nullptr);
  auto *restored_cu = restored_vm->soc()->xcd(0)->shader_engine(0)->compute_unit(0);
  ASSERT_NE(restored_cu, nullptr);
  auto *restored_wf = restored_cu->wf(0);
  ASSERT_NE(restored_wf, nullptr);
  EXPECT_EQ(restored_cu->read_vgpr(restored_wf->vgpr_alloc().base + amdgpu::ACC_VGPR_OFFSET, 0),
            0xA55A0001u);
  EXPECT_EQ(restored_cu->read_vgpr(restored_wf->vgpr_alloc().base + amdgpu::ACC_VGPR_OFFSET +
                                       cdna3::Isa::MAX_ACC_VGPRS_PER_WF - 1,
                                   0),
            0xDEADBEEFu);

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
  EXPECT_EQ(rj_vm_create_from_string(json, RJ_VM_MODE_DEFAULT, &handle), ROCJITSU_STATUS_SUCCESS);
  ASSERT_NE(handle, nullptr);
  rj_vm_destroy(handle);
}

TEST(CApiTest, InvalidArguments) {
  rj_vm_t *handle = nullptr;
  EXPECT_EQ(rj_vm_create_from_string(nullptr, RJ_VM_MODE_DEFAULT, &handle),
            ROCJITSU_STATUS_INVALID_ARGUMENT);
  EXPECT_EQ(rj_vm_step(nullptr, nullptr), ROCJITSU_STATUS_INVALID_ARGUMENT);
}

} // namespace
