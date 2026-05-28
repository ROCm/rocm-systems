// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file execution_plugin_test.cpp
/// @brief Tests for the ExecutionPlugin infrastructure.

#include "aql_queue.h"

#include "embedded_schema.h"
#include "rocjitsu/config/config_loader.h"
#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/vm/risc_v/hart.h"
#include "rocjitsu/vm/soc.h"

#include "simdojo/sim/simulation.h"

#include "rocjitsu/base/rj_compiler.h"
RJ_DIAGNOSTIC_PUSH
RJ_DIAGNOSTIC_IGNORE_PEDANTIC
#include "hsa/AMDHSAKernelDescriptor.h"
RJ_DIAGNOSTIC_POP

#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <string>

namespace {

using namespace rocjitsu;

// SOPP encoding: bits[31:23]=0x17F, bits[22:16]=op, bits[15:0]=simm16.
constexpr uint32_t sopp(uint32_t op, uint16_t simm16 = 0) {
  return 0xBF800000u | (op << 16) | simm16;
}
constexpr uint32_t S_NOP = sopp(0);
constexpr uint32_t S_ENDPGM = sopp(1);
constexpr uint32_t S_BARRIER = sopp(10);

/// A plugin that counts how many times each hook is called.
class CountingPlugin : public ExecutionPlugin {
public:
  int kernel_dispatches = 0;
  int workgroup_dispatches = 0;
  int instructions = 0;
  int barriers = 0;
  int barriers_resolved = 0;
  int memory_insts = 0;
  int riscv_instructions = 0;

  void onAmdgpuKernelDispatch(uint64_t, uint64_t) override { ++kernel_dispatches; }
  void onAmdgpuDispatchWorkgroup(uint32_t, uint32_t, uint32_t,
                                 std::span<amdgpu::Wavefront *>) override {
    ++workgroup_dispatches;
  }
  void onAmdgpuExecuteInstruction(uint64_t, const Instruction &inst) override {
    ++instructions;
    if (inst.mnemonic() == "s_barrier")
      ++barriers;
  }
  void onAmdgpuBarrierResolved(uint32_t) override { ++barriers_resolved; }
  void onAmdgpuRouteMemoryInstruction(const Instruction &) override { ++memory_insts; }
  void onRiscvExecuteInstruction(uint64_t, const Instruction &) override { ++riscv_instructions; }
};

/// Minimal SoC fixture: 1 XCD, 1 SE, 1 CU.
struct PluginFixture {
  std::unique_ptr<simdojo::SimulationEngine> engine;
  SoC *soc = nullptr;
  amdgpu::GpuMemory *mem = nullptr;

  PluginFixture() {
    std::string json = R"({
      "max_ticks":10000,"num_threads":1,"exec_mode":"functional",
      "vm":{"arch":"cdna4","gpu":{"device":{"wave_front_size":64}}},
      "topology":{"root":{"name":"soc","type":"soc","children":[
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
      ]},"links":[
        {"src":"xcd0.cp.req_0","dst":"xcd0.se0.cu0.cpl","latency":1,"weight":2},
        {"src":"xcd0.se0.cu0.req","dst":"xcd0.l2.cpl_0","latency":1,"weight":10}
      ]}}
    )";
    auto loaded = config::load_config_from_string(json, rocjitsu::kEmbeddedSchema);
    soc = loaded.soc();
    mem = loaded.memory();
    engine = std::make_unique<simdojo::SimulationEngine>(loaded.engine_config);
    engine->topology().set_root(loaded.take_root());
    loaded.wire_links(engine->topology());
    engine->build();
  }

  amdgpu::ComputeUnitCore *cu() { return soc->xcd(0)->shader_engine(0)->compute_unit(0); }
  amdgpu::CommandProcessor *cp() { return soc->xcd(0)->command_processor(); }

  uint64_t write_kernel(uint64_t addr, const uint32_t *code, size_t num_words) {
    using namespace rocr::llvm::amdhsa;
    kernel_descriptor_t kd{};
    kd.kernel_code_entry_byte_offset = sizeof(kernel_descriptor_t);
    AMDHSA_BITS_SET(kd.compute_pgm_rsrc1, COMPUTE_PGM_RSRC1_GRANULATED_WORKITEM_VGPR_COUNT, 31);
    AMDHSA_BITS_SET(kd.compute_pgm_rsrc1, COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT, 12);
    AMDHSA_BITS_SET(kd.compute_pgm_rsrc2, COMPUTE_PGM_RSRC2_USER_SGPR_COUNT, 2);
    mem->load_image(reinterpret_cast<const uint8_t *>(&kd), sizeof(kd), addr);
    mem->load_image(reinterpret_cast<const uint8_t *>(code), num_words * 4,
                    addr + sizeof(kernel_descriptor_t));
    return addr;
  }

  /// Attach a CountingPlugin and return a raw pointer to it.
  CountingPlugin *attach_plugin() {
    auto group = std::make_shared<ExecutionPluginGroup>();
    auto plugin = std::make_unique<CountingPlugin>();
    auto *p = plugin.get();
    group->add(std::move(plugin));
    soc->set_plugin_group(group);
    return p;
  }

  void run_kernel(const uint32_t *code, size_t num_words, uint32_t grid = 64,
                  uint32_t workgroup = 64) {
    uint64_t ko = write_kernel(0x1000, code, num_words);
    test::AqlQueue queue(mem, cp());
    queue.dispatch(ko, grid, workgroup);
    for (uint32_t i = 0; i < 10000 && engine->step(); ++i) {
      if (cu()->num_wfs() > 0) {
        bool all_halted = true;
        for (uint32_t w = 0; w < cu()->num_wf_slots(); ++w) {
          auto *wf = cu()->wf(w);
          if (wf && wf->sgpr_alloc().count > 0 && !wf->is_halted()) {
            all_halted = false;
            break;
          }
        }
        if (all_halted)
          break;
      }
    }
  }
};

TEST(ExecutionPluginTest, SimpleKernelFiresHooks) {
  PluginFixture f;
  auto *p = f.attach_plugin();

  const uint32_t code[] = {S_NOP, S_ENDPGM};
  f.run_kernel(code, 2);

  EXPECT_EQ(p->kernel_dispatches, 1);
  EXPECT_EQ(p->workgroup_dispatches, 1);
  EXPECT_GE(p->instructions, 2);
}

TEST(ExecutionPluginTest, BarrierFiresHooks) {
  PluginFixture f;
  auto *p = f.attach_plugin();

  const uint32_t code[] = {S_BARRIER, S_ENDPGM};
  f.run_kernel(code, 2, /*grid=*/128, /*workgroup=*/128);

  EXPECT_EQ(p->kernel_dispatches, 1);
  EXPECT_EQ(p->workgroup_dispatches, 1);
  EXPECT_EQ(p->barriers, 2);          // one per wavefront
  EXPECT_EQ(p->barriers_resolved, 1); // one resolution for the workgroup
}

TEST(ExecutionPluginTest, NoPluginNoCrash) {
  PluginFixture f;
  const uint32_t code[] = {S_NOP, S_ENDPGM};
  f.run_kernel(code, 2);
}

TEST(ExecutionPluginTest, RiscvInstructionHook) {
  simdojo::SimulationEngine::Config config{};
  config.max_ticks = 1000;
  config.num_threads = 1;
  auto engine = std::make_unique<simdojo::SimulationEngine>(config);
  auto *clk = engine->topology().add_clock_domain("clk", simdojo::TICKS_PER_SECOND);

  auto root = std::make_unique<simdojo::CompositeComponent>("soc");
  auto hart_ptr = std::make_unique<rocjitsu::risc_v::Hart>("hart0", *clk);
  auto *hart = static_cast<rocjitsu::risc_v::Hart *>(root->add_child(std::move(hart_ptr)));

  auto group = std::make_shared<ExecutionPluginGroup>();
  auto plugin = std::make_unique<CountingPlugin>();
  auto *p = plugin.get();
  group->add(std::move(plugin));
  hart->set_plugin_group(group);

  engine->topology().set_root(std::move(root));
  engine->build();

  // addi x1, x0, 42 then ebreak (halt).
  const uint32_t program[] = {0x02A00093u, 0x00100073u};
  hart->memory().load_image(reinterpret_cast<const uint8_t *>(program), sizeof(program), 0);
  hart->state().pc = 0;
  engine->run();

  EXPECT_EQ(p->riscv_instructions, 2);
  EXPECT_EQ(hart->state().read_xreg(1), 42);
}

} // namespace
