// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "aql_queue.h"
#include "halt_snapshot_plugin.h"
#include "long_path_handoff.h"
#include "scoped_temp.h"

#include "checkpoint_generated.h"
#include "embedded_schema.h"
#include "rocjitsu/config/checkpoint.h"
#include "rocjitsu/config/config_loader.h"
#include "rocjitsu/config/dbt_guest_config.h"
#include "rocjitsu/config/pci_device_config.h"
#include "rocjitsu/isa/arch/amdgpu/cdna3/isa.h"
#include "rocjitsu/isa/arch/amdgpu/shared/accvgpr_layout.h"
#include "rocjitsu/kmd/linux/amdgpu_properties.h"
#include "rocjitsu/kmd/linux/rpc.h"
#include "rocjitsu/vm/rj_vm.h"
#include "rocjitsu/vm/rj_vm_impl.h"
#include "rocjitsu/vm/soc.h"

#include "simdojo/sim/simulation.h"

#include "rocjitsu/base/rj_compiler.h"
RJ_DIAGNOSTIC_PUSH
RJ_DIAGNOSTIC_IGNORE_PEDANTIC
#include "hsa/AMDHSAKernelDescriptor.h"
RJ_DIAGNOSTIC_POP

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace rocjitsu::test {

class SoCTestAccess {
public:
  static uint32_t dispatch_pool_threads(const SoC &soc) {
    return soc.dispatch_pool_ ? soc.dispatch_pool_->thread_count() : 0;
  }
};

} // namespace rocjitsu::test

namespace {

const std::string CONFIG_DIR_PATH = CONFIG_DIR;

// \NPI new GPU: add a config-load test for its configs/<gpu>.json here.
using namespace rocjitsu;

test::ScopedTempFile write_temp_config(std::string_view json) {
  test::ScopedTempFile file("rocjitsu-config-");
  file.write(json);
  return file;
}

std::vector<uint8_t> read_binary_file(const std::string &path) {
  std::ifstream stream(path, std::ios::binary);
  return {std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>()};
}

TEST(ConfigLoaderTest, LoadCdna2Config) {
  auto loaded =
      config::load_config(CONFIG_DIR_PATH + "/gfx90a_mi210_kmd.json", rocjitsu::kEmbeddedSchema);
  auto *soc = loaded.soc();

  EXPECT_EQ(soc->arch(), ROCJITSU_CODE_ARCH_CDNA2);
  EXPECT_EQ(loaded.device.gpu_id, 50149u);
  EXPECT_EQ(loaded.device.device_id, 0x740fu);
  EXPECT_EQ(loaded.device.family_id, 0x8du);
  EXPECT_EQ(loaded.device.gfx_target_version, 90010u);
  EXPECT_EQ(loaded.device.revision_id, 1u);
  EXPECT_EQ(loaded.device.pci_revision_id, 2u);
  EXPECT_EQ(loaded.device.wave_front_size, 64u);
  EXPECT_EQ(loaded.device.max_waves_per_simd, 8u);
  EXPECT_EQ(loaded.device.lds_size_kb, 64u);

  // Aldebaran is a single-die part: one XCD of 8 SEs, 14 CUs per SE. MI210
  // exposes 104 active CUs through simd_count but has capacity of 112
  EXPECT_EQ(soc->num_xcds(), 1u);
  EXPECT_EQ(loaded.device.num_shader_engines, 8u);
  EXPECT_EQ(loaded.device.num_shader_arrays_per_engine, 1u);
  EXPECT_EQ(loaded.device.num_cu_per_sh, 14u);
  EXPECT_EQ(loaded.device.simd_per_cu, 4u);
  EXPECT_EQ(loaded.device.simd_count, 416u);
  EXPECT_EQ(soc->xcd(0)->num_shader_engines(), 8u);
  EXPECT_EQ(soc->xcd(0)->shader_engine(0)->num_compute_units(), 14u);
  EXPECT_EQ(kmd::drm_cu_active_number(loaded.device.simd_count, loaded.device.simd_per_cu), 104u);
  EXPECT_LE(loaded.device.simd_count, loaded.device.num_shader_engines *
                                          loaded.device.num_shader_arrays_per_engine *
                                          loaded.device.num_cu_per_sh * loaded.device.simd_per_cu);
}

std::pair<uint32_t, uint32_t> run_two_spi_dispatch() {
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
              {"name":"cu0","type":"compute_unit","config":[
                {"key":"num_wf_slots","value":"10"},
                {"key":"sgprs_per_wf","value":"104"},
                {"key":"vgprs_per_wf","value":"256"},
                {"key":"lds_size_kb","value":"64"}
              ]}
            ]},
            {"name":"se1","type":"shader_engine","children":[
              {"name":"cu0","type":"compute_unit","config":[
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
        {"src":"xcd0.cp.req_1","dst":"xcd0.se1.cu0.cpl","latency":1,"weight":2},
        {"src":"xcd0.se0.cu0.req","dst":"xcd0.l2.cpl_0","latency":1,"weight":10},
        {"src":"xcd0.se1.cu0.req","dst":"xcd0.l2.cpl_1","latency":1,"weight":10}
      ]
    }
  })";

  auto loaded = config::load_config_from_string(json, rocjitsu::kEmbeddedSchema);
  auto *soc = loaded.soc();

  simdojo::SimulationEngine engine(loaded.engine_config);
  engine.topology().set_root(loaded.take_root());
  loaded.wire_links(engine.topology());
  engine.create();

  rocjitsu::test::DispatchCountPlugin *dispatch_count = nullptr;
  soc->set_plugin_group(rocjitsu::test::make_dispatch_count_group(&dispatch_count));

  using namespace rocr::llvm::amdhsa;
  kernel_descriptor_t kd{};
  kd.kernel_code_entry_byte_offset = sizeof(kernel_descriptor_t);
  AMDHSA_BITS_SET(kd.compute_pgm_rsrc1, COMPUTE_PGM_RSRC1_GRANULATED_WORKITEM_VGPR_COUNT,
                  ((256 / 8) - 1));
  AMDHSA_BITS_SET(kd.compute_pgm_rsrc1, COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT,
                  ((104 / 8) - 1));
  AMDHSA_BITS_SET(kd.compute_pgm_rsrc2, COMPUTE_PGM_RSRC2_USER_SGPR_COUNT, 2);

  constexpr uint64_t KD_ADDR = 0x1000;
  soc->memory()->load_image(reinterpret_cast<const uint8_t *>(&kd), sizeof(kd), KD_ADDR);
  soc->memory()->write32(KD_ADDR + sizeof(kernel_descriptor_t), 0xFFFFFFFF);

  auto *xcd = soc->xcd(0);
  auto *cp = xcd->command_processor();
  cp->set_dispatch_threads(2);
  assert(cp->dispatch_threads() == 2);

  test::AqlQueue queue(soc->memory(), cp);
  queue.dispatch(KD_ADDR, 128, 64);

  engine.step();

  return {dispatch_count->for_cu(xcd->shader_engine(0)->compute_unit(0)),
          dispatch_count->for_cu(xcd->shader_engine(1)->compute_unit(0))};
}

TEST(ConfigLoaderTest, LoadCdna4Config) {
  std::string json = CONFIG_DIR_PATH + "/gfx950_mi355x.json";
  auto loaded = config::load_config(json, rocjitsu::kEmbeddedSchema);
  auto *soc = loaded.soc();

  // MI350X physical geometry: 8 XCDs, 4 SEs per XCD, 9 CUs per SE, 2 IODs.
  // The part exposes 256 active CUs through simd_count but has capacity for 288.
  EXPECT_EQ(soc->num_xcds(), 8u);
  EXPECT_EQ(soc->num_iods(), 2u);
  EXPECT_EQ(loaded.device.num_sdma_queues_per_engine, 8u);
  auto *xcd = soc->xcd(0);
  EXPECT_EQ(xcd->num_shader_engines(), 4u);
  EXPECT_EQ(xcd->shader_engine(0)->num_compute_units(), 9u);
  EXPECT_EQ(kmd::drm_cu_active_number(loaded.device.simd_count, loaded.device.simd_per_cu), 256u);
  EXPECT_EQ(soc->assign_queue_owner_cp(0), soc->xcd(0)->command_processor());
  EXPECT_EQ(soc->assign_queue_owner_cp(1), soc->xcd(1)->command_processor());
  EXPECT_EQ(soc->assign_queue_owner_cp(soc->num_xcds()), soc->xcd(0)->command_processor());
}

// The VMM reads the bus shape through these two entry points, and nothing else
// in the suite does: the device tests build a PciDeviceConfig by hand, so a
// field dropped in the loader, or a wrong default for an omitted section, would
// leave every test green while a guest saw the wrong device.
TEST(ConfigLoaderTest, LoadsThePciSectionThroughBothEntryPoints) {
  const auto identity = config::load_device_identity(CONFIG_DIR_PATH + "/gfx1250_mi455x.json",
                                                     rocjitsu::kEmbeddedSchema);
  EXPECT_EQ(identity.pci.vram_aperture_bytes, 268435456u);
  EXPECT_EQ(identity.device.vendor_id, 0x1002u)
      << "the identity must come from the same file as the bus shape";

  const auto loaded =
      config::load_config(CONFIG_DIR_PATH + "/gfx1250_mi455x.json", rocjitsu::kEmbeddedSchema);
  EXPECT_EQ(loaded.pci.vram_aperture_bytes, identity.pci.vram_aperture_bytes)
      << "the two entry points disagree about the same file";
}

// A config with no bus section still has to yield a usable one, because most
// parts do not describe a bus at all.
TEST(ConfigLoaderTest, DefaultsThePciSectionWhenTheFileOmitsIt) {
  const auto identity =
      config::load_device_identity(CONFIG_DIR_PATH + "/gfx1151.json", rocjitsu::kEmbeddedSchema);
  const config::PciDeviceConfig fallback;
  EXPECT_EQ(identity.pci.class_code, fallback.class_code);
  EXPECT_EQ(identity.pci.doorbell_aperture_bytes, fallback.doorbell_aperture_bytes);
  EXPECT_EQ(identity.pci.register_aperture_bytes, fallback.register_aperture_bytes);
  EXPECT_EQ(identity.pci.vram_aperture_bytes, fallback.vram_aperture_bytes);
}

TEST(ConfigLoaderTest, LoadFourGpuMi455xKmdConfig) {
  auto loaded = config::load_config(CONFIG_DIR_PATH + "/gfx1250_mi455x_kmd_4gpu.json",
                                    rocjitsu::kEmbeddedSchema);
  auto standalone =
      config::load_config(CONFIG_DIR_PATH + "/gfx1250_mi455x.json", rocjitsu::kEmbeddedSchema);

  EXPECT_EQ(loaded.num_gpus, 4u);
  ASSERT_EQ(loaded.devices.size(), 4u);
  ASSERT_EQ(loaded.extra_gpu_builds.size(), 3u);

  EXPECT_EQ(loaded.device.revision_id, standalone.device.revision_id);
  EXPECT_EQ(loaded.device.simd_count, standalone.device.simd_count);
  EXPECT_EQ(loaded.device.num_shader_engines, standalone.device.num_shader_engines);
  EXPECT_EQ(loaded.device.num_shader_arrays_per_engine,
            standalone.device.num_shader_arrays_per_engine);
  EXPECT_EQ(loaded.device.num_cu_per_sh, standalone.device.num_cu_per_sh);
  EXPECT_EQ(loaded.device.simd_per_cu, standalone.device.simd_per_cu);
  EXPECT_EQ(loaded.device.l1_size_kb, standalone.device.l1_size_kb);
  EXPECT_EQ(loaded.device.l1_line_size, standalone.device.l1_line_size);
  EXPECT_EQ(loaded.device.l1_assoc, standalone.device.l1_assoc);
  EXPECT_EQ(loaded.device.l2_size_kb, standalone.device.l2_size_kb);
  EXPECT_EQ(loaded.device.l2_line_size, standalone.device.l2_line_size);
  EXPECT_EQ(loaded.device.l2_assoc, standalone.device.l2_assoc);

  for (uint32_t i = 0; i < loaded.num_gpus; ++i) {
    SCOPED_TRACE(i);
    EXPECT_EQ(loaded.devices[i].gpu_id, 1250u + i);
    EXPECT_EQ(loaded.devices[i].location_id, 0x0300u + (i << 8));
    EXPECT_EQ(loaded.devices[i].drm_render_minor, 128u + i);
    EXPECT_EQ(loaded.devices[i].unique_id, 1250u + i);
    EXPECT_EQ(loaded.devices[i].revision_id, 1u);
  }

  for (const auto &build : loaded.extra_gpu_builds)
    EXPECT_NE(dynamic_cast<SoC *>(build.root.get()), nullptr);
}

TEST(ConfigLoaderTest, DispatchPoolBudgetIsSharedAcrossProductionTopology) {
  auto loaded =
      config::load_config(CONFIG_DIR_PATH + "/gfx950_cdna4.json", rocjitsu::kEmbeddedSchema);
  auto *soc = loaded.soc();

  soc->set_dispatch_threads(8);
  EXPECT_EQ(soc->dispatch_threads(), 8u);
  EXPECT_EQ(test::SoCTestAccess::dispatch_pool_threads(*soc), 8u);
  soc->for_each_cp([](auto *cp) { EXPECT_EQ(cp->dispatch_threads(), 8u); });

  soc->set_dispatch_threads(1);
  EXPECT_EQ(test::SoCTestAccess::dispatch_pool_threads(*soc), 0u);
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
  EXPECT_EQ(rdna4.device.num_shader_engines, 4u); // R9700: 4 SEs of 2 arrays
  EXPECT_EQ(rdna4.device.num_shader_arrays_per_engine, 2u);
  EXPECT_EQ(rdna4.device.num_cu_per_sh, 8u);
  EXPECT_EQ(rdna4.device.simd_per_cu, 2u);
  EXPECT_EQ(rdna4.device.num_sdma_queues_per_engine, 6u);
  EXPECT_EQ(rdna4.device.vram_type, kmd::kAmdgpuVramTypeGddr6);
  EXPECT_EQ(rdna4.device.simd_count, rdna4.device.num_shader_engines *
                                         rdna4.device.num_shader_arrays_per_engine *
                                         rdna4.device.num_cu_per_sh * rdna4.device.simd_per_cu);
  EXPECT_EQ(kmd::drm_shader_engine_count(rdna4.device.num_shader_engines *
                                             rdna4.device.num_shader_arrays_per_engine,
                                         rdna4.device.num_shader_arrays_per_engine),
            4u);
  EXPECT_EQ(kmd::drm_cu_active_number(rdna4.device.simd_count, rdna4.device.simd_per_cu), 64u);
  EXPECT_EQ(kmd::external_rev_id_for_gfx_target_version(rdna4.device.gfx_target_version,
                                                        rdna4.device.revision_id),
            0x51u);
  EXPECT_EQ(kmd::gfx_target_name(rdna4.device.gfx_target_version), "gfx1201");
  EXPECT_EQ(kmd::gfx_target_version_from_name("gfx1201"), rdna4.device.gfx_target_version);
  EXPECT_EQ(kmd::gfx_target_name(90010), "gfx90a");
  EXPECT_EQ(kmd::gfx_target_version_from_name("gfx90a"), 90010u);
  EXPECT_EQ(kmd::gfx_target_name(120501u), "gfx1251");
  EXPECT_EQ(kmd::gfx_target_version_from_name("gfx1251"), 120501u);
  EXPECT_FALSE(kmd::gfx_target_version_from_name("cdna4"));
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
  EXPECT_TRUE(rdna4.soc()->xcd(0)->command_processor()->packed_tid());
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
  EXPECT_EQ(rdna3.device.num_shader_engines, 6u); // W7900: 6 SEs of 2 arrays
  EXPECT_EQ(rdna3.device.num_shader_arrays_per_engine, 2u);
  EXPECT_EQ(rdna3.device.num_cu_per_sh, 8u);
  EXPECT_EQ(rdna3.device.simd_per_cu, 2u);
  EXPECT_EQ(rdna3.device.num_sdma_queues_per_engine, 6u);
  EXPECT_EQ(rdna3.device.vram_type, kmd::kAmdgpuVramTypeGddr6);
  EXPECT_EQ(rdna3.device.simd_count, rdna3.device.num_shader_engines *
                                         rdna3.device.num_shader_arrays_per_engine *
                                         rdna3.device.num_cu_per_sh * rdna3.device.simd_per_cu);
  EXPECT_EQ(kmd::drm_shader_engine_count(rdna3.device.num_shader_engines *
                                             rdna3.device.num_shader_arrays_per_engine,
                                         rdna3.device.num_shader_arrays_per_engine),
            6u);
  EXPECT_EQ(kmd::drm_cu_active_number(rdna3.device.simd_count, rdna3.device.simd_per_cu), 96u);
  EXPECT_EQ(kmd::external_rev_id_for_gfx_target_version(rdna3.device.gfx_target_version,
                                                        rdna3.device.revision_id),
            0x1u);
  EXPECT_EQ(kmd::gfx_target_name(rdna3.device.gfx_target_version), "gfx1100");
  EXPECT_EQ(kmd::num_hw_gfx_contexts_for_gfx_target_version(rdna3.device.gfx_target_version), 8u);
  EXPECT_EQ(rdna3.soc()->num_xcds(), 1u);
  EXPECT_EQ(rdna3.soc()->xcd(0)->num_shader_engines(), 6u);
  EXPECT_EQ(rdna3.soc()->xcd(0)->shader_engine(0)->num_compute_units(), 16u);
  EXPECT_TRUE(rdna3.soc()->xcd(0)->command_processor()->packed_tid());
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
  EXPECT_EQ(rdna35.device.num_shader_engines, 2u); // 2 SEs of 2 arrays
  EXPECT_EQ(rdna35.device.num_shader_arrays_per_engine, 2u);
  EXPECT_EQ(rdna35.device.num_cu_per_sh, 8u);
  EXPECT_EQ(rdna35.device.simd_per_cu, 2u);
  EXPECT_EQ(rdna35.device.num_sdma_queues_per_engine, 2u);
  EXPECT_EQ(rdna35.device.vram_type, kmd::kAmdgpuVramTypeGddr6);
  EXPECT_EQ(rdna35.device.simd_count, rdna35.device.num_shader_engines *
                                          rdna35.device.num_shader_arrays_per_engine *
                                          rdna35.device.num_cu_per_sh * rdna35.device.simd_per_cu);
  EXPECT_EQ(kmd::drm_shader_engine_count(rdna35.device.num_shader_engines *
                                             rdna35.device.num_shader_arrays_per_engine,
                                         rdna35.device.num_shader_arrays_per_engine),
            2u);
  EXPECT_EQ(kmd::drm_cu_active_number(rdna35.device.simd_count, rdna35.device.simd_per_cu), 32u);
  EXPECT_EQ(kmd::external_rev_id_for_gfx_target_version(rdna35.device.gfx_target_version,
                                                        rdna35.device.revision_id),
            0xc1u);
  EXPECT_EQ(kmd::gfx_target_name(rdna35.device.gfx_target_version), "gfx1151");
  EXPECT_EQ(kmd::num_hw_gfx_contexts_for_gfx_target_version(rdna35.device.gfx_target_version), 8u);
  EXPECT_EQ(rdna35.soc()->num_xcds(), 1u);
  EXPECT_EQ(rdna35.soc()->xcd(0)->num_shader_engines(), 2u);
  EXPECT_EQ(rdna35.soc()->xcd(0)->shader_engine(0)->num_compute_units(), 16u);
  EXPECT_TRUE(rdna35.soc()->xcd(0)->command_processor()->packed_tid());
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
      "gpu": { "device": {
        "gfx_target_version": 90500,
        "num_sdma_queues_per_engine": 8
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
        "num_sdma_queues_per_engine": 8,
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

TEST(ConfigLoaderTest, LoadsDbtOnlyConfigWithoutVmOrTopology) {
  const auto file = write_temp_config(R"({
      "dbt_guest": {
        "enabled": true,
        "guest_isa": "gfx950",
        "host_isa": "gfx1201",
        "host_gpu_id": 8716,
        "log_level": 2,
        "signal_backtrace": true,
        "guest_device": {
          "gpu_id": 38144,
          "gfx_target_version": 90500,
          "vendor_id": 4098,
          "device_id": 30112,
          "family_id": 160,
          "unique_id": 5929628898254127105,
          "marketing_name": "AMD Instinct MI350X",
          "drm_render_minor": 191,
          "simd_count": 64,
          "num_shader_engines": 2,
          "num_shader_arrays_per_engine": 2,
          "num_cu_per_sh": 4,
          "local_mem_size": 309237645312,
          "num_sdma_queues_per_engine": 8
        }
      }
    })");

  auto dbt = config::load_dbt_guest_config_from_file(file.path());

  EXPECT_TRUE(dbt.enabled);
  EXPECT_EQ(dbt.guest_isa, "gfx950");
  EXPECT_EQ(dbt.host.isa, "gfx1201");
  EXPECT_EQ(dbt.host.gpu_id, 8716u);
  EXPECT_EQ(dbt.host.backend, config::DbtExecutionBackend::Hardware);
  EXPECT_EQ(dbt.log_level, 2);
  EXPECT_TRUE(dbt.signal_backtrace);
  ASSERT_TRUE(dbt.guest_device.present);
  EXPECT_EQ(dbt.guest_device.gpu_id, 38144u);
  EXPECT_EQ(dbt.guest_device.gfx_target_version, 90500u);
  EXPECT_EQ(dbt.guest_device.marketing_name, "AMD Instinct MI350X");
  EXPECT_EQ(dbt.guest_device.drm_render_minor, 191u);
  EXPECT_EQ(dbt.guest_device.simd_count,
            dbt.guest_device.num_shader_engines * dbt.guest_device.num_shader_arrays_per_engine *
                dbt.guest_device.num_cu_per_sh * dbt.guest_device.simd_per_cu);
  EXPECT_EQ(dbt.guest_device.num_shader_arrays_per_engine, 2u);
  EXPECT_EQ(dbt.guest_device.local_mem_size, 309237645312ULL);
  EXPECT_EQ(dbt.guest_device.num_sdma_queues_per_engine, 8u);
  // Revisions default to Unspecified when the config omits them.
  EXPECT_EQ(dbt.guest_revision, config::DbtSiliconRevision::Unspecified);
  EXPECT_EQ(dbt.host_revision, config::DbtSiliconRevision::Unspecified);
}

TEST(ConfigLoaderTest, RejectsMissingOrZeroSdmaQueuesWhenRegularEnginesArePresent) {
  const auto missing = write_temp_config(R"({
      "dbt_guest": {
        "guest_device": {
          "num_sdma_engines": 1
        }
      }
    })");
  const auto zero = write_temp_config(R"({
      "dbt_guest": {
        "guest_device": {
          "num_sdma_engines": 1,
          "num_sdma_queues_per_engine": 0
        }
      }
    })");

  for (const auto &path : {missing.path(), zero.path()})
    EXPECT_THAT([&] { (void)config::load_dbt_guest_config_from_file(path); },
                testing::ThrowsMessage<std::runtime_error>(
                    testing::AllOf(testing::HasSubstr("dbt_guest.guest_device."),
                                   testing::HasSubstr("num_sdma_queues_per_engine"))));
}

TEST(ConfigLoaderTest, SdmaQueueValidationIdentifiesVmDevicePath) {
  const char *json = R"({
    "max_ticks": 1,
    "num_threads": 1,
    "vm": {
      "arch": "cdna3",
      "gpu": { "device": { "num_sdma_engines": 1 } }
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

  EXPECT_THAT(
      [&] { (void)config::load_config_from_string(json, rocjitsu::kEmbeddedSchema); },
      testing::ThrowsMessage<std::runtime_error>(testing::AllOf(
          testing::HasSubstr("vm.gpu.device."), testing::HasSubstr("num_sdma_queues_per_engine"))));
}

TEST(ConfigLoaderTest, AllowsZeroSdmaQueuesWithoutRegularEngines) {
  const auto file = write_temp_config(R"({
      "dbt_guest": {
        "guest_device": {
          "num_sdma_engines": 0,
          "num_sdma_queues_per_engine": 0
        }
      }
    })");

  auto dbt = config::load_dbt_guest_config_from_file(file.path());

  ASSERT_TRUE(dbt.guest_device.present);
  EXPECT_EQ(dbt.guest_device.num_sdma_engines, 0u);
  EXPECT_EQ(dbt.guest_device.num_sdma_queues_per_engine, 0u);
}

TEST(ConfigLoaderTest, DefaultKfdDeviceHasNoRegularSdmaEngines) {
  const config::KfdDeviceConfig device;

  EXPECT_EQ(device.num_sdma_engines, 0u);
  EXPECT_EQ(device.num_sdma_queues_per_engine, 0u);
}

TEST(ConfigLoaderTest, ShippedDevicesDeclareSdmaQueues) {
  unsigned checked = 0;
  for (const auto &entry : std::filesystem::directory_iterator(CONFIG_DIR_PATH)) {
    if (entry.path().extension() != ".json")
      continue;

    const std::string name = entry.path().filename().string();
    SCOPED_TRACE(name);
    if (name.rfind("guest_", 0) == 0) {
      auto dbt = config::load_dbt_guest_config_from_file(entry.path().string());
      ASSERT_TRUE(dbt.guest_device.present);
      ASSERT_NE(dbt.guest_device.num_sdma_engines, 0u);
      EXPECT_NE(dbt.guest_device.num_sdma_queues_per_engine, 0u);
    } else {
      auto loaded = config::load_config(entry.path().string(), rocjitsu::kEmbeddedSchema);
      ASSERT_TRUE(loaded.device.present);
      ASSERT_NE(loaded.device.num_sdma_engines, 0u);
      EXPECT_NE(loaded.device.num_sdma_queues_per_engine, 0u);
    }
    ++checked;
  }

  EXPECT_GE(checked, 12u) << "expected every shipped device config to be discovered";
}

#if defined(RJ_DAEMON_LOGGING_CONFIG_PATH) && defined(RJ_LOGGING_CONFIG_PATH) &&                   \
    defined(RJ_INSTALLED_LOGGING_CONFIG_PATH)
TEST(ConfigLoaderTest, GeneratedDeviceConfigsDeclareSdmaQueues) {
  const std::string paths[] = {
      RJ_DAEMON_LOGGING_CONFIG_PATH,
      RJ_LOGGING_CONFIG_PATH,
      RJ_INSTALLED_LOGGING_CONFIG_PATH,
  };

  for (const std::string &path : paths) {
    SCOPED_TRACE(path);
    auto loaded = config::load_config(path, rocjitsu::kEmbeddedSchema);
    ASSERT_TRUE(loaded.device.present);
    ASSERT_NE(loaded.device.num_sdma_engines, 0u);
    EXPECT_NE(loaded.device.num_sdma_queues_per_engine, 0u);
  }
}
#endif

TEST(ConfigLoaderTest, ShippedGfx950GuestsUseCapturedSdmaQueueCount) {
  unsigned checked = 0;
  for (const auto &entry : std::filesystem::directory_iterator(CONFIG_DIR_PATH)) {
    if (entry.path().extension() != ".json" ||
        entry.path().filename().string().rfind("guest_", 0) != 0)
      continue;

    auto dbt = config::load_dbt_guest_config_from_file(entry.path().string());
    if (dbt.guest_isa != "gfx950")
      continue;

    SCOPED_TRACE(entry.path().filename().string());
    ASSERT_TRUE(dbt.guest_device.present);
    EXPECT_EQ(dbt.guest_device.device_id, 30112u);
    EXPECT_EQ(dbt.guest_device.num_sdma_queues_per_engine, 8u);
    ++checked;
  }

  EXPECT_GE(checked, 3u) << "expected every shipped gfx950 guest config to be discovered";
}

TEST(ConfigLoaderTest, LoadsDbtGuestSiliconRevisions) {
  // gfx1250 A0 and B0 share an ELF machine ID, so the configured revisions
  // select the B0-to-A0 translation profile.
  //
  // This also pins guest_isa == host_isa as a legal configuration. The hook
  // layer resolves the resulting agent-role overlap by matching the host first
  // (only the host carries the node-id constraint) rather than by rejecting the
  // config here, which would foreclose this profile.
  const auto file = write_temp_config(R"({
      "dbt_guest": {
        "enabled": true,
        "guest_isa": "gfx1250",
        "host_isa": "gfx1250",
        "guest_revision": "gfx1250_b0",
        "host_revision": "gfx1250_a0"
      }
    })");

  auto dbt = config::load_dbt_guest_config_from_file(file.path());

  EXPECT_EQ(dbt.guest_revision, config::DbtSiliconRevision::Gfx1250B0);
  EXPECT_EQ(dbt.host_revision, config::DbtSiliconRevision::Gfx1250A0);
}

TEST(ConfigLoaderTest, RejectsDbtGuestDeviceWithInconsistentSimdCount) {
  const auto file = write_temp_config(R"({
      "dbt_guest": {
        "enabled": true,
        "guest_isa": "gfx950",
        "host_isa": "gfx1201",
        "guest_device": {
          "gpu_id": 38144,
          "gfx_target_version": 90500,
          "simd_count": 1024,
          "num_shader_engines": 4,
          "num_cu_per_sh": 4,
          "simd_per_cu": 4,
          "num_sdma_queues_per_engine": 8
        }
      }
    })");

  EXPECT_THROW(config::load_dbt_guest_config_from_file(file.path()), std::runtime_error);
}

TEST(ConfigLoaderTest, LoadsDbtGuestThroughFullConfigLoader) {
  std::ifstream base(CONFIG_DIR_PATH + "/gfx1201_r9700.json");
  ASSERT_TRUE(base.is_open());
  std::string json((std::istreambuf_iterator<char>(base)), std::istreambuf_iterator<char>());
  const size_t insert_pos = json.find('{');
  ASSERT_NE(insert_pos, std::string::npos);
  json.insert(insert_pos + 1, R"(
    "dbt_guest": {
      "enabled": true,
      "guest_isa": "gfx950",
      "host_isa": "gfx1201",
      "host_gpu_id": 8716,
      "log_level": 2,
      "signal_backtrace": true,
      "guest_device": {
        "gpu_id": 38144,
        "gfx_target_version": 90500,
        "vendor_id": 4098,
        "device_id": 30112,
        "family_id": 160,
        "unique_id": 5929628898254127105,
        "marketing_name": "AMD Instinct MI350X",
        "drm_render_minor": 191,
        "simd_count": 64,
        "num_shader_engines": 2,
        "num_shader_arrays_per_engine": 2,
        "num_cu_per_sh": 4,
        "local_mem_size": 309237645312,
        "num_sdma_queues_per_engine": 8
      }
    },
  )");

  const auto file = write_temp_config(json);
  auto loaded = config::load_config(file.path(), rocjitsu::kEmbeddedSchema);

  EXPECT_TRUE(loaded.dbt_guest.enabled);
  EXPECT_EQ(loaded.dbt_guest.guest_isa, "gfx950");
  EXPECT_EQ(loaded.dbt_guest.host.isa, "gfx1201");
  EXPECT_EQ(loaded.dbt_guest.host.gpu_id, 8716u);
  EXPECT_EQ(loaded.dbt_guest.host.backend, config::DbtExecutionBackend::Hardware);
  EXPECT_EQ(loaded.dbt_guest.log_level, 2);
  EXPECT_TRUE(loaded.dbt_guest.signal_backtrace);
  ASSERT_TRUE(loaded.dbt_guest.guest_device.present);
  EXPECT_EQ(loaded.dbt_guest.guest_device.gpu_id, 38144u);
  EXPECT_EQ(loaded.dbt_guest.guest_device.gfx_target_version, 90500u);
  EXPECT_EQ(loaded.dbt_guest.guest_device.marketing_name, "AMD Instinct MI350X");
}

TEST(ConfigLoaderTest, MissingDbtGuestConfigReturnsDefaults) {
  const auto file = write_temp_config("{}");

  auto dbt = config::load_dbt_guest_config_from_file(file.path());

  EXPECT_FALSE(dbt.enabled);
  EXPECT_TRUE(dbt.guest_isa.empty());
  EXPECT_TRUE(dbt.host.isa.empty());
  EXPECT_EQ(dbt.host.gpu_id, 0u);
  EXPECT_EQ(dbt.host.backend, config::DbtExecutionBackend::Hardware);
  EXPECT_EQ(dbt.log_level, 0);
  EXPECT_FALSE(dbt.signal_backtrace);
  EXPECT_FALSE(dbt.guest_device.present);
}

TEST(ConfigLoaderTest, MissingDbtGuestDeviceLeavesDeviceAbsent) {
  const auto file = write_temp_config(R"({
        "dbt_guest": {
          "enabled": true,
          "guest_isa": "gfx950",
          "host_isa": "gfx1201"
        }
      })");

  auto dbt = config::load_dbt_guest_config_from_file(file.path());

  EXPECT_TRUE(dbt.enabled);
  EXPECT_EQ(dbt.guest_isa, "gfx950");
  EXPECT_EQ(dbt.host.isa, "gfx1201");
  EXPECT_FALSE(dbt.guest_device.present);
}

TEST(ConfigLoaderTest, MalformedDbtGuestConfigThrows) {
  const auto file = write_temp_config(R"({ "dbt_guest": )");

  EXPECT_THROW(config::load_dbt_guest_config_from_file(file.path()), std::runtime_error);
}

TEST(ConfigLoaderTest, LoadsSimulatorDbtBackendConfig) {
  const auto external_file = write_temp_config(R"({
        "dbt_guest": {
          "enabled": true,
          "guest_isa": "gfx950",
          "host_isa": "gfx942",
          "execution_backend": "simulator",
          "simulator_config": "gfx942_cdna3_kmd.json"
        }
      })");
  const auto self_contained_file = write_temp_config(R"({
        "dbt_guest": {
          "enabled": true,
          "guest_isa": "gfx950",
          "host_isa": "gfx942",
          "execution_backend": "simulator"
        }
      })");

  auto external = config::load_dbt_guest_config_from_file(external_file.path());
  auto self_contained = config::load_dbt_guest_config_from_file(self_contained_file.path());

  EXPECT_EQ(external.host.isa, "gfx942");
  EXPECT_EQ(external.host.backend, config::DbtExecutionBackend::Simulator);
  EXPECT_EQ(external.host.simulator_config_path, "gfx942_cdna3_kmd.json");
  EXPECT_EQ(self_contained.host.backend, config::DbtExecutionBackend::Simulator);
  EXPECT_TRUE(self_contained.host.simulator_config_path.empty());
}

TEST(ConfigLoaderTest, LoadsExplicitHardwareDbtBackendConfig) {
  const auto file = write_temp_config(R"({
        "dbt_guest": {
          "enabled": true,
          "execution_backend": "hardware"
        }
      })");

  auto dbt = config::load_dbt_guest_config_from_file(file.path());

  EXPECT_EQ(dbt.host.backend, config::DbtExecutionBackend::Hardware);
}

TEST(ConfigLoaderTest, AppliesResolvedDbtHostGpuId) {
  config::DbtGuestConfig automatic;
  automatic.enabled = true;
  automatic.host.backend = config::DbtExecutionBackend::Hardware;
  config::DbtGuestConfig explicit_id = automatic;
  explicit_id.host.gpu_id = 8716;
  config::DbtGuestConfig simulator = automatic;
  simulator.host.backend = config::DbtExecutionBackend::Simulator;

  config::apply_resolved_dbt_host_gpu_id(automatic, "28851");
  config::apply_resolved_dbt_host_gpu_id(explicit_id, "28851");
  config::apply_resolved_dbt_host_gpu_id(simulator, "28851");

  EXPECT_EQ(automatic.host.gpu_id, 28851u);
  EXPECT_EQ(explicit_id.host.gpu_id, 8716u);
  EXPECT_EQ(simulator.host.gpu_id, 28851u);
}

TEST(ConfigLoaderTest, RejectsInvalidResolvedDbtHostGpuId) {
  const std::array<std::string_view, 5> invalid_values = {"", "0", "not-a-number", "28851 trailing",
                                                          "4294967296"};
  for (std::string_view value : invalid_values) {
    config::DbtGuestConfig dbt;
    dbt.enabled = true;
    EXPECT_THROW(config::apply_resolved_dbt_host_gpu_id(dbt, value), std::runtime_error) << value;
  }
}

TEST(ConfigLoaderTest, ExplicitDbtHostGpuIdOverridesResolvedHandoff) {
  config::DbtGuestConfig dbt;
  dbt.enabled = true;
  dbt.host.gpu_id = 8716;

  EXPECT_NO_THROW(config::apply_resolved_dbt_host_gpu_id(dbt, "invalid-but-ignored"));
  EXPECT_EQ(dbt.host.gpu_id, 8716u);
}

TEST(ConfigLoaderTest, ParsesRuntimeConfigHandoff) {
  const auto dbt = config::parse_dbt_runtime_config_handoff("/tmp/config.json\r\n28851\r\n");
  ASSERT_TRUE(dbt);
  EXPECT_EQ(dbt->config_path, "/tmp/config.json");
  ASSERT_TRUE(dbt->resolved_gpu_id);
  EXPECT_EQ(*dbt->resolved_gpu_id, "28851");

  const auto non_dbt = config::parse_dbt_runtime_config_handoff("/tmp/config.json");
  ASSERT_TRUE(non_dbt);
  EXPECT_EQ(non_dbt->config_path, "/tmp/config.json");
  EXPECT_FALSE(non_dbt->resolved_gpu_id);

  const auto newline_terminated = config::parse_dbt_runtime_config_handoff("/tmp/config.json\n");
  ASSERT_TRUE(newline_terminated);
  EXPECT_FALSE(newline_terminated->resolved_gpu_id);
  EXPECT_FALSE(config::parse_dbt_runtime_config_handoff("\n28851\n"));
}

TEST(ConfigLoaderTest, RoundTripsRuntimeConfigHandoff) {
  const test::ScopedTempDirectory runtime("rocjitsu-runtime-config-round-trip-");
  test::ScopedEnvironmentVariable runtime_dir("ROCJITSU_RUNTIME_DIR", runtime.path());
  config::DbtGuestConfig dbt;
  dbt.enabled = true;
  dbt.host.gpu_id = 28851;

  ASSERT_TRUE(config::write_dbt_runtime_config_handoff("/tmp/config.json", dbt, getpid()));
  std::ifstream handoff(rocjitsu::rpc_invocation_config_file_path(getpid()));
  const std::string contents((std::istreambuf_iterator<char>(handoff)),
                             std::istreambuf_iterator<char>());
  const auto parsed = config::parse_dbt_runtime_config_handoff(contents);

  ASSERT_TRUE(parsed);
  EXPECT_EQ(parsed->config_path, "/tmp/config.json");
  ASSERT_TRUE(parsed->resolved_gpu_id);
  EXPECT_EQ(*parsed->resolved_gpu_id, "28851");
}

TEST(ConfigLoaderTest, RejectsUnresolvedAutomaticDbtHandoffWrite) {
  const test::ScopedTempDirectory runtime("rocjitsu-runtime-config-unresolved-");
  test::ScopedEnvironmentVariable runtime_dir("ROCJITSU_RUNTIME_DIR", runtime.path());
  config::DbtGuestConfig dbt;
  dbt.enabled = true;

  EXPECT_FALSE(config::write_dbt_runtime_config_handoff("/tmp/config.json", dbt, getpid()));
  EXPECT_FALSE(std::filesystem::exists(rocjitsu::rpc_invocation_config_file_path(getpid())));
}

TEST(ConfigLoaderTest, RuntimeConfigHandoffReportsDirectoryCreationFailure) {
  const test::ScopedTempDirectory runtime("rocjitsu-runtime-config-write-failure-");
  const std::filesystem::path blocked_root = std::filesystem::path(runtime.path()) / "blocked";
  std::ofstream(blocked_root) << "not a directory";
  test::ScopedEnvironmentVariable runtime_dir("ROCJITSU_RUNTIME_DIR", blocked_root.string());
  config::DbtGuestConfig dbt;
  dbt.enabled = true;
  dbt.host.gpu_id = 28851;

  EXPECT_FALSE(config::write_dbt_runtime_config_handoff("/tmp/config.json", dbt, getpid()));
}

TEST(ConfigLoaderTest, RejectsEmptyResolvedGpuIdLineForEnabledDbt) {
  const auto handoff = config::parse_dbt_runtime_config_handoff("/tmp/config.json\n\n");
  ASSERT_TRUE(handoff);
  ASSERT_TRUE(handoff->resolved_gpu_id);

  config::DbtGuestConfig dbt;
  dbt.enabled = true;
  EXPECT_THROW(config::apply_resolved_dbt_host_gpu_id(dbt, *handoff->resolved_gpu_id),
               std::runtime_error);
}

TEST(ConfigLoaderTest, LoadsDbtRuntimeConfigHandoffFromInvocationDirectory) {
  const test::ScopedTempDirectory runtime("rocjitsu-runtime-config-handoff-");
  const auto config_file = write_temp_config(R"({
        "dbt_guest": {
          "enabled": true,
          "guest_isa": "gfx950",
          "host_isa": "gfx942"
        }
      })");
  {
    std::ofstream handoff(std::filesystem::path(runtime.path()) / "config_path");
    handoff << config_file.path() << "\n28851\n";
  }
  test::ScopedEnvironmentVariable invocation_dir(rocjitsu::kRpcInvocationDirEnv, runtime.path());

  const std::optional<config::DbtGuestConfig> loaded =
      config::load_dbt_guest_config_from_runtime_config();

  ASSERT_TRUE(loaded);
  EXPECT_TRUE(loaded->enabled);
  EXPECT_EQ(loaded->host.gpu_id, 28851u);
}

// The HSA-hook half of a pair. GuestKfdConfigTest.ReadsRuntimeHandoffLargerThan4095Bytes drives
// the same oversized handoff through the other consumer -- the KFD interposer's raw read loop,
// which is where a fixed 4096-byte read once truncated it. This reader has always been an
// unbounded std::ifstream, so the case is coverage rather than a fix; what it locks down is that
// the two independent readers agree. Both are built by install_oversized_handoff() and both
// assert test::kOversizedHandoffHostGpuId, so a reader that starts resolving a different host
// GPU from identical bytes fails here or there instead of silently splitting the two layers
// onto different GPUs on a multi-GPU host.
TEST(ConfigLoaderTest, ReadsRuntimeHandoffLargerThan4095Bytes) {
  const test::ScopedTempDirectory runtime("rocjitsu-runtime-config-oversized-");

  // Same treatment as the KFD-side test: a temp directory already deeper than the path being
  // built is a limit of where the test runs, not a defect in the reader, so it must skip.
  const test::LongPathHandoff handoff = test::install_oversized_handoff(runtime.path(), R"({
        "dbt_guest": {
          "enabled": true,
          "guest_isa": "gfx950",
          "host_isa": "gfx942"
        }
      })");
  if (handoff.status() == test::LongPathHandoff::Status::kSkip)
    GTEST_SKIP() << "cannot build the oversized handoff here: " << handoff.reason();
  ASSERT_TRUE(handoff.status() == test::LongPathHandoff::Status::kOk) << handoff.reason();

  const test::ScopedEnvironmentVariable invocation_dir(rocjitsu::kRpcInvocationDirEnv,
                                                       runtime.path());
  const std::optional<config::DbtGuestConfig> loaded =
      config::load_dbt_guest_config_from_runtime_config();

  ASSERT_TRUE(loaded);
  EXPECT_TRUE(loaded->enabled);
  EXPECT_EQ(loaded->host.gpu_id, test::kOversizedHandoffHostGpuId);
}

TEST(ConfigLoaderTest, RejectsPathOnlyHandoffForAutomaticDbtHost) {
  const test::ScopedTempDirectory runtime("rocjitsu-runtime-config-automatic-");
  const auto config_file = write_temp_config(R"({
        "dbt_guest": {
          "enabled": true,
          "guest_isa": "gfx950",
          "host_isa": "gfx942"
        }
      })");
  {
    std::ofstream handoff(std::filesystem::path(runtime.path()) / "config_path");
    handoff << config_file.path() << '\n';
  }
  test::ScopedEnvironmentVariable invocation_dir(rocjitsu::kRpcInvocationDirEnv, runtime.path());

  EXPECT_THROW(config::load_dbt_guest_config_from_runtime_config(), std::runtime_error);
}

TEST(ConfigLoaderTest, AllowsPathOnlyHandoffWithoutAutomaticDbtHost) {
  const test::ScopedTempDirectory runtime("rocjitsu-runtime-config-path-only-");
  test::ScopedEnvironmentVariable invocation_dir(rocjitsu::kRpcInvocationDirEnv, runtime.path());

  for (const std::string_view dbt_guest : {
           R"("dbt_guest": {"enabled": true, "host_gpu_id": 28851})",
           R"("dbt_guest": {"enabled": false})",
       }) {
    const auto config_file = write_temp_config("{" + std::string(dbt_guest) + "}");
    {
      std::ofstream handoff(std::filesystem::path(runtime.path()) / "config_path");
      handoff << config_file.path() << '\n';
    }

    const auto loaded = config::load_dbt_guest_config_from_runtime_config();
    ASSERT_TRUE(loaded);
    EXPECT_EQ(loaded->host.gpu_id, dbt_guest.find("28851") == std::string_view::npos ? 0u : 28851u);
  }
}

TEST(ConfigLoaderTest, RejectsEmptyDbtExecutionBackend) {
  const auto file = write_temp_config(R"({
        "dbt_guest": {
          "enabled": true,
          "execution_backend": ""
        }
      })");

  EXPECT_THROW(config::load_dbt_guest_config_from_file(file.path()), std::runtime_error);
}

TEST(ConfigLoaderTest, RejectsMisspelledDbtExecutionBackend) {
  const auto file = write_temp_config(R"({
        "dbt_guest": {
          "enabled": true,
          "execution_backed": "simulator"
        }
      })");

  EXPECT_THROW(config::load_dbt_guest_config_from_file(file.path()), std::runtime_error);
}

TEST(ConfigLoaderTest, ValidatesSimulatorDbtGuestDeviceLimits) {
  config::DbtGuestConfig guest;
  guest.enabled = true;
  guest.host.backend = config::DbtExecutionBackend::Simulator;
  guest.guest_device.present = true;
  guest.guest_device.lds_size_kb = 64;
  guest.guest_device.max_slots_scratch_cu = 32;
  guest.guest_device.max_waves_per_simd = 8;
  guest.guest_device.wave_front_size = 64;

  config::KfdDeviceConfig simulator;
  simulator.present = true;
  simulator.lds_size_kb = 64;
  simulator.max_slots_scratch_cu = 32;
  simulator.max_waves_per_simd = 8;
  simulator.wave_front_size = 64;

  EXPECT_NO_THROW(config::validate_dbt_simulator_device_limits(guest, simulator));
  guest.guest_device.lds_size_kb = 65;
  EXPECT_THROW(config::validate_dbt_simulator_device_limits(guest, simulator), std::runtime_error);
  guest.guest_device.lds_size_kb = 64;
  guest.guest_device.max_slots_scratch_cu = 33;
  EXPECT_THROW(config::validate_dbt_simulator_device_limits(guest, simulator), std::runtime_error);
}

TEST(ConfigLoaderTest, DisabledDbtBackendSkipsBackendSpecificValidation) {
  const auto simulator_file = write_temp_config(R"({
        "dbt_guest": {
          "enabled": false,
          "execution_backend": "simulator"
        }
      })");
  const auto hardware_file = write_temp_config(R"({
        "dbt_guest": {
          "enabled": false,
          "execution_backend": "hardware",
          "simulator_config": "ignored.json"
        }
      })");

  EXPECT_NO_THROW(config::load_dbt_guest_config_from_file(simulator_file.path()));
  EXPECT_NO_THROW(config::load_dbt_guest_config_from_file(hardware_file.path()));
}

TEST(ConfigLoaderTest, ResolvesDbtHostConfigPath) {
  EXPECT_EQ(config::resolve_dbt_host_config_path("/a/b/dbt.json", ""), "/a/b/dbt.json");
  EXPECT_EQ(config::resolve_dbt_host_config_path("/a/b/dbt.json", "sim.json"), "/a/b/sim.json");
  EXPECT_EQ(config::resolve_dbt_host_config_path("/a/b/dbt.json", "/abs/sim.json"),
            "/abs/sim.json");
  EXPECT_EQ(config::resolve_dbt_host_config_path("/a/b/dbt.json", "../c/./sim.json"),
            "/a/c/sim.json");
}

TEST(ConfigLoaderTest, RejectsUnknownDbtExecutionBackend) {
  const auto file = write_temp_config(R"({
        "dbt_guest": {
          "enabled": true,
          "execution_backend": "magic"
        }
      })");

  EXPECT_THROW(config::load_dbt_guest_config_from_file(file.path()), std::runtime_error);
}

TEST(ConfigLoaderTest, RejectsSimulatorConfigForHardwareDbtBackend) {
  const auto file = write_temp_config(R"({
        "dbt_guest": {
          "enabled": true,
          "execution_backend": "hardware",
          "simulator_config": "gfx942_cdna3_kmd.json"
        }
      })");

  EXPECT_THROW(config::load_dbt_guest_config_from_file(file.path()), std::runtime_error);
}

TEST(ConfigLoaderTest, Gfx1250ComputeUnitDefaultsCoverTtmpAndHighVgprs) {
  const char *json = R"({"max_ticks":1000,"num_threads":1,
    "vm":{"arch":"cdna5"},
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
  ASSERT_EQ(cu->vgpr_storage_lane_count(), 32u);
  EXPECT_EQ(cu->config().sgprs_per_wf, 128u);
  EXPECT_EQ(cu->config().vgprs_per_wf, 1024u);
}

TEST(ConfigLoaderTest, RejectsTargetFromDifferentArchitecture) {
  const char *json = R"({"vm":{"arch":"cdna4","target":"gfx1250"}})";
  EXPECT_THROW(config::load_config_from_string(json, rocjitsu::kEmbeddedSchema),
               std::runtime_error);
}

TEST(ConfigLoaderTest, RejectsTargetVersionMismatch) {
  const char *json = R"({"vm":{"arch":"cdna5","target":"gfx1250","gpu":{
    "device":{"gfx_target_version":120501}}}})";
  EXPECT_THROW(config::load_config_from_string(json, rocjitsu::kEmbeddedSchema),
               std::runtime_error);
}

TEST(ConfigLoaderTest, RejectsGfx1251SimulationUntilExecutionIsImplemented) {
  const char *json = R"({"vm":{"arch":"cdna5","target":"gfx1251"}})";
  EXPECT_THROW(config::load_config_from_string(json, rocjitsu::kEmbeddedSchema),
               std::runtime_error);
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
  engine.create();

  rocjitsu::test::DispatchCountPlugin *dispatch_count = nullptr;
  auto plugin_group = rocjitsu::test::make_dispatch_count_group(&dispatch_count);
  soc->set_plugin_group(plugin_group);

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

  // After one step, the doorbell event dispatched wavefronts to CUs. Count them
  // via the dispatch hook (fired at placement) so the check is independent of when
  // waves execute and free themselves. Verify round-robin distribution.
  EXPECT_EQ(xcd->command_processor()->dispatched_count(), 1u);
  auto *se = soc->xcd(0)->shader_engine(0);
  EXPECT_EQ(dispatch_count->for_cu(se->compute_unit(0)), 1u);
  EXPECT_EQ(dispatch_count->for_cu(se->compute_unit(1)), 1u);
}

TEST(ConfigLoaderTest, DispatchPlacementDoesNotFollowHostThreadCount) {
  auto [se0_wfs, se1_wfs] = run_two_spi_dispatch();

  EXPECT_EQ(se0_wfs, 2u);
  EXPECT_EQ(se1_wfs, 0u);
}

TEST(CheckpointTest, SaveAndRestoreMemory) {
  const char *json = R"({"max_ticks":10000,"num_threads":1,"exec_mode":"clocked",
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
  auto *source_cp = soc->xcd(0)->command_processor();
  ASSERT_NE(source_cp, nullptr);
  EXPECT_TRUE(source_cp->packed_tid());
  EXPECT_EQ(soc->exec_mode(), simdojo::ExecMode::CLOCKED);

  soc->memory()->write32(0x1000, 0xDEADBEEF);
  soc->memory()->write64(0x2000, 0x0123456789ABCDEFULL);

  test::ScopedTempFile checkpoint("rocjitsu-checkpoint-");
  config::save_checkpoint(checkpoint.path(), *soc, 42, loaded.engine_config);
  ASSERT_TRUE(std::filesystem::exists(checkpoint.path()));

  auto restored = config::restore_checkpoint(checkpoint.path());
  EXPECT_EQ(restored.memory()->read32(0x1000), 0xDEADBEEFu);
  EXPECT_EQ(restored.memory()->read64(0x2000), 0x0123456789ABCDEFULL);
  EXPECT_EQ(restored.exec_mode, simdojo::ExecMode::CLOCKED);
  EXPECT_EQ(restored.soc()->exec_mode(), simdojo::ExecMode::CLOCKED);
  EXPECT_TRUE(restored.soc()->xcd(0)->command_processor()->packed_tid());
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
                {"key":"num_wf_slots","value":"4"},
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

  auto *lower_wf = cu->dispatch_wf(0, 0x1000, cu->config().sgprs_per_wf, cu->config().vgprs_per_wf);
  ASSERT_NE(lower_wf, nullptr);
  auto *wf = cu->dispatch_wf(1, 0x2000, cu->config().sgprs_per_wf, cu->config().vgprs_per_wf);
  ASSERT_NE(wf, nullptr);
  ASSERT_EQ(wf->wf_id(), 1u);
  lower_wf->halt();
  const uint32_t acc0 = wf->vgpr_alloc().base + amdgpu::ACC_VGPR_OFFSET;
  const uint32_t acc_last = acc0 + cdna3::Isa::MAX_ACC_VGPRS_PER_WF - 1;
  const uint32_t acc_quarter = acc0 + cdna3::Isa::MAX_ACC_VGPRS_PER_WF / 4;
  const uint32_t acc_midpoint = acc0 + cdna3::Isa::MAX_ACC_VGPRS_PER_WF / 2;
  cu->write_vgpr(acc0, 0, 0xA55A0001u);
  cu->write_vgpr(acc_quarter, 63, 0xA55A003Fu);
  cu->write_vgpr(acc_midpoint, 63, 0xA55A103Fu);
  cu->write_vgpr(acc_last, 0, 0xDEADBEEFu);
  cu->write_vgpr(acc_last, 63, 0xFEEDFACEu);

  test::ScopedTempFile checkpoint("rocjitsu-checkpoint-");
  config::save_checkpoint(checkpoint.path(), *loaded.soc(), 42, loaded.engine_config);
  ASSERT_TRUE(std::filesystem::exists(checkpoint.path()));

  const auto checkpoint_bytes = read_binary_file(checkpoint.path());
  flatbuffers::Verifier verifier(checkpoint_bytes.data(), checkpoint_bytes.size());
  ASSERT_TRUE(fb::VerifySimulationCheckpointBuffer(verifier));
  const auto *saved_checkpoint = fb::GetSimulationCheckpoint(checkpoint_bytes.data());
  ASSERT_NE(saved_checkpoint->compute_units(), nullptr);
  ASSERT_EQ(saved_checkpoint->compute_units()->size(), 1u);
  const auto *saved_wavefronts = saved_checkpoint->compute_units()->Get(0)->wavefronts();
  ASSERT_NE(saved_wavefronts, nullptr);
  ASSERT_EQ(saved_wavefronts->size(), 1u);
  const auto *saved_wavefront = saved_wavefronts->Get(0);
  ASSERT_NE(saved_wavefront->vgprs(), nullptr);
  EXPECT_EQ(saved_wavefront->kernel_wave_size(), 64u);
  EXPECT_EQ(saved_wavefront->vgpr_lane_count(), 64u);
  EXPECT_EQ(saved_wavefront->vgprs()->size(),
            cu->vgpr_allocation_block_size() * 64u * sizeof(uint32_t));

  auto restored = config::restore_checkpoint(checkpoint.path());
  auto *restored_soc = restored.soc();
  ASSERT_NE(restored_soc, nullptr);
  auto *restored_cu = restored_soc->xcd(0)->shader_engine(0)->compute_unit(0);
  ASSERT_NE(restored_cu, nullptr);
  EXPECT_TRUE(restored_cu->wf(0)->is_halted());
  auto *restored_wf = restored_cu->wf(1);
  ASSERT_NE(restored_wf, nullptr);
  EXPECT_FALSE(restored_wf->is_halted());
  EXPECT_EQ(restored_wf->wf_id(), 1u);
  EXPECT_EQ(restored_wf->wg_id(), 1u);
  EXPECT_EQ(restored_wf->pc, 0x2000u);
  EXPECT_EQ(restored_cu->read_vgpr(restored_wf->vgpr_alloc().base + amdgpu::ACC_VGPR_OFFSET, 0),
            0xA55A0001u);
  EXPECT_EQ(restored_cu->read_vgpr(restored_wf->vgpr_alloc().base + amdgpu::ACC_VGPR_OFFSET +
                                       cdna3::Isa::MAX_ACC_VGPRS_PER_WF / 4,
                                   63),
            0xA55A003Fu);
  EXPECT_EQ(restored_cu->read_vgpr(restored_wf->vgpr_alloc().base + amdgpu::ACC_VGPR_OFFSET +
                                       cdna3::Isa::MAX_ACC_VGPRS_PER_WF / 2,
                                   63),
            0xA55A103Fu);
  EXPECT_EQ(restored_cu->read_vgpr(restored_wf->vgpr_alloc().base + amdgpu::ACC_VGPR_OFFSET +
                                       cdna3::Isa::MAX_ACC_VGPRS_PER_WF - 1,
                                   0),
            0xDEADBEEFu);
  EXPECT_EQ(restored_cu->read_vgpr(restored_wf->vgpr_alloc().base + amdgpu::ACC_VGPR_OFFSET +
                                       cdna3::Isa::MAX_ACC_VGPRS_PER_WF - 1,
                                   63),
            0xFEEDFACEu);
}

TEST(CheckpointTest, SaveAndRestoreRdnaWave64State) {
  const char *json = R"({"max_ticks":10000,"num_threads":1,
    "vm":{"arch":"rdna4"},
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
                {"key":"sgprs_per_wf","value":"128"},
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

  auto *wf = cu->dispatch_wf(0, 0, cu->config().sgprs_per_wf, cu->config().vgprs_per_wf, 64);
  ASSERT_NE(wf, nullptr);
  ASSERT_EQ(wf->wf_size(), 64u);
  wf->set_exec_raw(0xDEADBEEF0000000FULL);
  const uint32_t vgpr_base = wf->vgpr_alloc().base;
  const uint32_t vgpr_last = vgpr_base + cu->vgpr_allocation_block_size() - 1;
  cu->write_vgpr(vgpr_base + 1, 31, 0x1234001Fu);
  cu->write_vgpr(vgpr_base + 1, 43, 0x1234002Bu);
  cu->write_vgpr(vgpr_base + cu->vgpr_allocation_block_size() / 2, 31, 0x5678001Fu);
  cu->write_vgpr(vgpr_last, 31, 0x9ABC001Fu);

  test::ScopedTempFile checkpoint("rocjitsu-checkpoint-");
  config::save_checkpoint(checkpoint.path(), *loaded.soc(), 42, loaded.engine_config);
  ASSERT_TRUE(std::filesystem::exists(checkpoint.path()));

  const auto checkpoint_bytes = read_binary_file(checkpoint.path());
  flatbuffers::Verifier verifier(checkpoint_bytes.data(), checkpoint_bytes.size());
  ASSERT_TRUE(fb::VerifySimulationCheckpointBuffer(verifier));
  const auto *saved_checkpoint = fb::GetSimulationCheckpoint(checkpoint_bytes.data());
  ASSERT_NE(saved_checkpoint->compute_units(), nullptr);
  ASSERT_EQ(saved_checkpoint->compute_units()->size(), 1u);
  const auto *saved_wavefronts = saved_checkpoint->compute_units()->Get(0)->wavefronts();
  ASSERT_NE(saved_wavefronts, nullptr);
  ASSERT_EQ(saved_wavefronts->size(), 1u);
  const auto *saved_wavefront = saved_wavefronts->Get(0);
  ASSERT_NE(saved_wavefront->vgprs(), nullptr);
  EXPECT_EQ(saved_wavefront->kernel_wave_size(), 64u);
  EXPECT_EQ(saved_wavefront->vgpr_lane_count(), 64u);
  EXPECT_EQ(saved_wavefront->vgprs()->size(),
            cu->vgpr_allocation_block_size() * 64u * sizeof(uint32_t));

  auto restored = config::restore_checkpoint(checkpoint.path());
  auto *restored_soc = restored.soc();
  ASSERT_NE(restored_soc, nullptr);
  auto *restored_cp = restored_soc->xcd(0)->command_processor();
  ASSERT_NE(restored_cp, nullptr);
  EXPECT_EQ(restored_cp->sdma_packet_dialect(), amdgpu::SdmaPacketDialect::Gfx11Plus);
  auto *restored_cu = restored_soc->xcd(0)->shader_engine(0)->compute_unit(0);
  ASSERT_NE(restored_cu, nullptr);
  auto *restored_wf = restored_cu->wf(0);
  ASSERT_NE(restored_wf, nullptr);
  EXPECT_EQ(restored_wf->exec(), 0xDEADBEEF0000000FULL);
  EXPECT_EQ(restored_wf->exec_raw(), 0xDEADBEEF0000000FULL);
  EXPECT_EQ(restored_wf->kernel_wave_size(), 64u);
  EXPECT_EQ(restored_cu->read_vgpr(restored_wf->vgpr_alloc().base + 1, 31), 0x1234001Fu);
  EXPECT_EQ(restored_cu->read_vgpr(restored_wf->vgpr_alloc().base + 1, 43), 0x1234002Bu);
  EXPECT_EQ(restored_cu->read_vgpr(
                restored_wf->vgpr_alloc().base + restored_cu->vgpr_allocation_block_size() / 2, 31),
            0x5678001Fu);
  EXPECT_EQ(restored_cu->read_vgpr(
                restored_wf->vgpr_alloc().base + restored_cu->vgpr_allocation_block_size() - 1, 31),
            0x9ABC001Fu);
}

TEST(CheckpointTest, SaveAndRestoreHwregState) {
  const char *json = R"({"max_ticks":10000,"num_threads":1,
    "vm":{"arch":"cdna5"},
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
  constexpr uint32_t kStatus = 0xA5A55A5Au;
  constexpr uint32_t kWaveSchedMode = 0x5A5AA5A5u;
  wf->set_status_raw(kStatus);
  wf->set_mode_raw(amdgpu::Wavefront::FP16_OVFL_BIT);
  wf->set_wave_sched_mode_raw(kWaveSchedMode);
  ASSERT_TRUE(wf->fp16_ovfl());

  test::ScopedTempFile checkpoint("rocjitsu-checkpoint-");
  config::save_checkpoint(checkpoint.path(), *loaded.soc(), 42, loaded.engine_config);
  ASSERT_TRUE(std::filesystem::exists(checkpoint.path()));

  const auto checkpoint_bytes = read_binary_file(checkpoint.path());
  flatbuffers::Verifier verifier(checkpoint_bytes.data(), checkpoint_bytes.size());
  ASSERT_TRUE(fb::VerifySimulationCheckpointBuffer(verifier));
  const auto *saved_checkpoint = fb::GetSimulationCheckpoint(checkpoint_bytes.data());
  ASSERT_NE(saved_checkpoint->compute_units(), nullptr);
  ASSERT_EQ(saved_checkpoint->compute_units()->size(), 1u);
  const auto *saved_wavefronts = saved_checkpoint->compute_units()->Get(0)->wavefronts();
  ASSERT_NE(saved_wavefronts, nullptr);
  ASSERT_EQ(saved_wavefronts->size(), 1u);
  const auto *saved_wavefront = saved_wavefronts->Get(0);
  ASSERT_NE(saved_wavefront->vgprs(), nullptr);
  EXPECT_EQ(saved_wavefront->kernel_wave_size(), 32u);
  EXPECT_EQ(saved_wavefront->vgpr_lane_count(), 32u);
  EXPECT_EQ(saved_wavefront->vgprs()->size(),
            cu->vgpr_allocation_block_size() * 32u * sizeof(uint32_t));

  auto restored = config::restore_checkpoint(checkpoint.path());
  auto *restored_soc = restored.soc();
  ASSERT_NE(restored_soc, nullptr);
  auto *restored_wf = restored_soc->xcd(0)->shader_engine(0)->compute_unit(0)->wf(0);
  ASSERT_NE(restored_wf, nullptr);
  EXPECT_EQ(restored_wf->status_raw(), kStatus);
  EXPECT_EQ(restored_wf->mode_raw(), amdgpu::Wavefront::FP16_OVFL_BIT);
  EXPECT_EQ(restored_wf->wave_sched_mode_raw(), kWaveSchedMode);
  EXPECT_TRUE(restored_wf->fp16_ovfl());
}

// The checkpoint record carries the architectural registers and the TTMPs but
// none of the trap/debug state around them, so a wave captured mid-handler
// would restore without the EXEC restore or the privileged STATUS write that
// leaving the handler performs. Refusing beats resuming the application with
// the trap handler's state installed.
TEST(CheckpointTest, RefusesToSaveTrappedOrDebuggerStoppedWaves) {
  const char *json = R"({"max_ticks":10000,"num_threads":1,
    "vm":{"arch":"cdna5"},
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

  test::ScopedTempFile checkpoint("rocjitsu-checkpoint-");
  // A plain running wave still checkpoints.
  EXPECT_NO_THROW(
      config::save_checkpoint(checkpoint.path(), *loaded.soc(), 1, loaded.engine_config));

  wf->set_in_trap_handler(true);
  EXPECT_THROW(config::save_checkpoint(checkpoint.path(), *loaded.soc(), 2, loaded.engine_config),
               std::runtime_error);
  wf->set_in_trap_handler(false);

  wf->set_debug_halted(true);
  EXPECT_THROW(config::save_checkpoint(checkpoint.path(), *loaded.soc(), 3, loaded.engine_config),
               std::runtime_error);
  wf->set_debug_halted(false);

  wf->set_debug_suspended(true);
  EXPECT_THROW(config::save_checkpoint(checkpoint.path(), *loaded.soc(), 4, loaded.engine_config),
               std::runtime_error);
  wf->set_debug_suspended(false);

  // The runtime's own pause is not a debugger stop. A queue throttled to
  // queue_percentage 0 carries none of the trap or debugger state the refusals
  // above exist to protect, so it must stay checkpointable -- this is the one
  // assertion that tells debug_stopped() apart from debug_paused().
  wf->set_runtime_suspended(true);
  EXPECT_NO_THROW(
      config::save_checkpoint(checkpoint.path(), *loaded.soc(), 5, loaded.engine_config));
  wf->set_runtime_suspended(false);
}

// wg_coord is dispatch identity, and the flat wg_id cannot stand in for it: the
// grid dimensions needed to unflatten one into the other live in the dispatch
// packet, which is not part of a checkpoint. A restored wave that lost the
// coordinate publishes the wrong workgroup in TTMP8/9/10 at trap entry and can
// no longer be matched to its own CWSR record.
TEST(CheckpointTest, RoundTripsWorkgroupCoordinates) {
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
  // A flat id that is not any of the coordinates, so restoring wg_id into them
  // would not pass either.
  auto *wf = cu->dispatch_wf(/*wg_id=*/9, /*pc=*/0x1000, cu->config().sgprs_per_wf,
                             cu->config().vgprs_per_wf);
  ASSERT_NE(wf, nullptr);
  wf->set_wg_coord(3, 5, 7);

  test::ScopedTempFile checkpoint("rocjitsu-wg-coord-checkpoint-");
  ASSERT_NO_THROW(
      config::save_checkpoint(checkpoint.path(), *loaded.soc(), 1, loaded.engine_config));

  auto restored = config::restore_checkpoint(checkpoint.path());
  auto *restored_cu = restored.soc()->xcd(0)->shader_engine(0)->compute_unit(0);
  ASSERT_NE(restored_cu, nullptr);
  ASSERT_EQ(restored_cu->num_wfs(), 1u);
  const auto *restored_wf = restored_cu->wf(0);
  ASSERT_NE(restored_wf, nullptr);
  EXPECT_EQ(restored_wf->wg_id(), 9u);
  EXPECT_EQ(restored_wf->wg_coord(), (std::array<uint32_t, 3>{3, 5, 7}));
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

TEST(CApiTest, ClockedDispatchStaysEventDriven) {
  const char *json = R"({"max_ticks":10000,"num_threads":1,
    "exec_mode":"clocked",
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
  rj_vm_t *raw = nullptr;
  ASSERT_EQ(rj_vm_create_from_string(json, RJ_VM_MODE_DEFAULT, &raw), ROCJITSU_STATUS_SUCCESS);
  ASSERT_NE(raw, nullptr);
  std::unique_ptr<rj_vm_t, decltype(&rj_vm_destroy)> handle(raw, &rj_vm_destroy);
  auto *cp = handle->soc->xcd(0)->command_processor();
  cp->set_dispatch_threads(8);
  EXPECT_EQ(cp->dispatch_threads(), 1u);

  using namespace rocr::llvm::amdhsa;
  kernel_descriptor_t kd{};
  kd.kernel_code_entry_byte_offset = sizeof(kernel_descriptor_t);
  AMDHSA_BITS_SET(kd.compute_pgm_rsrc1, COMPUTE_PGM_RSRC1_GRANULATED_WORKITEM_VGPR_COUNT,
                  ((256 / 8) - 1));
  AMDHSA_BITS_SET(kd.compute_pgm_rsrc1, COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT,
                  ((104 / 8) - 1));
  AMDHSA_BITS_SET(kd.compute_pgm_rsrc2, COMPUTE_PGM_RSRC2_USER_SGPR_COUNT, 2);

  constexpr uint64_t kKernelAddress = 0x1000;
  constexpr uint32_t kCode[] = {0xBF800000u, 0xBF810000u}; // s_nop; s_endpgm
  handle->soc->memory()->load_image(reinterpret_cast<const uint8_t *>(&kd), sizeof(kd),
                                    kKernelAddress);
  handle->soc->memory()->load_image(reinterpret_cast<const uint8_t *>(kCode), sizeof(kCode),
                                    kKernelAddress + sizeof(kd));

  test::AqlQueue queue(handle->soc->memory(), cp);
  queue.dispatch(kKernelAddress, /*grid_size=*/64, /*workgroup_size=*/64);

  int active = 0;
  const auto tick_before_doorbell = handle->engine->global_time();
  EXPECT_EQ(rj_vm_step(handle.get(), &active), ROCJITSU_STATUS_SUCCESS);
  EXPECT_EQ(handle->engine->global_time(), tick_before_doorbell);
  auto *cu = handle->soc->xcd(0)->shader_engine(0)->compute_unit(0);
  ASSERT_NE(cu->wf(0), nullptr);
  EXPECT_EQ(cu->wf(0)->trace_inst_count_, 0u);
}

TEST(CApiTest, CheckpointRoundTrip) {
  rj_vm_t *raw_source = nullptr;
  ASSERT_EQ(rj_vm_create((CONFIG_DIR_PATH + "/gfx942_cdna3.json").c_str(), RJ_VM_MODE_DEFAULT,
                         &raw_source),
            ROCJITSU_STATUS_SUCCESS);
  ASSERT_NE(raw_source, nullptr);
  std::unique_ptr<rj_vm_t, decltype(&rj_vm_destroy)> source(raw_source, &rj_vm_destroy);

  constexpr uint64_t kCodeAddress = 0x1000;
  constexpr uint32_t kSEndpgm = 0xBF810000u;
  source->soc->memory()->write32(kCodeAddress, kSEndpgm);
  auto *source_cu = source->soc->xcd(0)->shader_engine(0)->compute_unit(0);
  ASSERT_NE(source_cu, nullptr);
  auto *source_wf = source_cu->dispatch_wf(0, kCodeAddress, source_cu->config().sgprs_per_wf,
                                           source_cu->config().vgprs_per_wf);
  ASSERT_NE(source_wf, nullptr);

  test::ScopedTempFile checkpoint("rocjitsu-c-api-checkpoint-");
  ASSERT_EQ(rj_vm_save_checkpoint(source.get(), checkpoint.path().c_str(), 42),
            ROCJITSU_STATUS_SUCCESS);

  rj_vm_t *raw_restored = nullptr;
  ASSERT_EQ(rj_vm_restore_checkpoint(checkpoint.path().c_str(), &raw_restored),
            ROCJITSU_STATUS_SUCCESS);
  ASSERT_NE(raw_restored, nullptr);
  std::unique_ptr<rj_vm_t, decltype(&rj_vm_destroy)> restored(raw_restored, &rj_vm_destroy);

  auto *restored_cu = restored->soc->xcd(0)->shader_engine(0)->compute_unit(0);
  ASSERT_NE(restored_cu, nullptr);
  ASSERT_EQ(restored_cu->num_wfs(), 1u);

  int active = 1;
  EXPECT_EQ(rj_vm_step(restored.get(), &active), ROCJITSU_STATUS_SUCCESS);
  EXPECT_EQ(restored_cu->num_wfs(), 0u);
}

TEST(CApiTest, RejectsMalformedCheckpoints) {
  test::ScopedTempFile junk("rocjitsu-junk-checkpoint-");
  junk.write(std::string(512, static_cast<char>(0xA5)));

  rj_vm_t *restored = nullptr;
  EXPECT_EQ(rj_vm_restore_checkpoint(junk.path().c_str(), &restored), ROCJITSU_STATUS_INVALID_FILE);
  EXPECT_EQ(restored, nullptr);

  rj_vm_t *raw_source = nullptr;
  ASSERT_EQ(rj_vm_create((CONFIG_DIR_PATH + "/gfx942_cdna3.json").c_str(), RJ_VM_MODE_DEFAULT,
                         &raw_source),
            ROCJITSU_STATUS_SUCCESS);
  ASSERT_NE(raw_source, nullptr);
  std::unique_ptr<rj_vm_t, decltype(&rj_vm_destroy)> source(raw_source, &rj_vm_destroy);

  test::ScopedTempFile valid("rocjitsu-valid-checkpoint-");
  ASSERT_EQ(rj_vm_save_checkpoint(source.get(), valid.path().c_str(), 42), ROCJITSU_STATUS_SUCCESS);
  std::ifstream input(valid.path(), std::ios::binary);
  std::string bytes{std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
  ASSERT_GT(bytes.size(), 8u);
  bytes.resize(bytes.size() / 2);

  test::ScopedTempFile truncated("rocjitsu-truncated-checkpoint-");
  truncated.write(bytes);
  EXPECT_EQ(rj_vm_restore_checkpoint(truncated.path().c_str(), &restored),
            ROCJITSU_STATUS_INVALID_FILE);
  EXPECT_EQ(restored, nullptr);
}

TEST(CApiTest, InvalidArguments) {
  rj_vm_t *handle = nullptr;
  EXPECT_EQ(rj_vm_create_from_string(nullptr, RJ_VM_MODE_DEFAULT, &handle),
            ROCJITSU_STATUS_INVALID_ARGUMENT);
  EXPECT_EQ(rj_vm_step(nullptr, nullptr), ROCJITSU_STATUS_INVALID_ARGUMENT);
}

} // namespace
