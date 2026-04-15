// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/config/config_loader.h"
#include "rocjitsu/vm/rj_vm.h"
#include "rocjitsu/vm/soc.h"

#include "simdojo/sim/simulation.h"

#include <gtest/gtest.h>

#include <bit>
#include <cstdint>
#include <string>
#include <vector>

namespace {

const std::string kSchemaPath = std::string(SCHEMA_DIR) + "/simulation_config.fbs";

// SOPP encoding: bits[31:23] = 0x17F (SOPP prefix), bits[22:16] = op.
constexpr uint32_t SOPP_S_NOP = 0xBF800000;
constexpr uint32_t SOPP_S_ENDPGM = 0xBF810000;

using namespace rocjitsu;

// Helper: create a VM from inline JSON via the C++ config loader.
struct VmFixture {
  std::unique_ptr<simdojo::SimulationEngine> engine;
  SoC *soc_ptr = nullptr;
  amdgpu::GpuMemory *gpu_mem = nullptr;

  VmFixture(const std::string &arch = "cdna3", uint32_t num_cus = 1, uint32_t num_wf_slots = 10) {
    // Build CU children with range expansion.
    std::string cu_range = "cu[0:" + std::to_string(num_cus) + "]";
    // Build link specs for dispatch and L2.
    std::string links;
    for (uint32_t i = 0; i < num_cus; ++i) {
      if (i > 0)
        links += ",";
      links += R"({"src":"xcd0.cp.req_)" + std::to_string(i) + R"(","dst":"xcd0.se0.cu)" +
               std::to_string(i) + R"(.cpl","latency":1,"weight":2})";
      links += R"(,{"src":"xcd0.se0.cu)" + std::to_string(i) + R"(.req","dst":"xcd0.l2.cpl_)" +
               std::to_string(i) + R"(","latency":1,"weight":10})";
    }

    std::string json = R"({"max_ticks":10000,"num_threads":1,"vm":{"arch":")" + arch +
                       R"("},)"
                       R"("topology":{"root":{"name":"soc","type":"soc","children":[)"
                       R"({"name":"vram","type":"gpu_memory"},)"
                       R"({"name":"xcd0","type":"xcd","children":[)"
                       R"({"name":"l2","type":"l2_cache"},)"
                       R"({"name":"cp","type":"command_processor"},)"
                       R"({"name":"se0","type":"shader_engine","children":[)"
                       R"({"name":")" +
                       cu_range +
                       R"(","type":"compute_unit","config":[)"
                       R"({"key":"num_wf_slots","value":")" +
                       std::to_string(num_wf_slots) +
                       R"("},)"
                       R"({"key":"sgprs_per_wf","value":"104"},)"
                       R"({"key":"vgprs_per_wf","value":"256"},)"
                       R"({"key":"lds_size_kb","value":"64"})"
                       R"(]}]}]}]},"links":[)" +
                       links + R"(]}})";
    auto loaded = config::load_config_from_string(json, kSchemaPath);
    soc_ptr = loaded.soc();
    gpu_mem = loaded.memory();
    engine = std::make_unique<simdojo::SimulationEngine>(loaded.engine_config);
    engine->topology().set_root(loaded.take_root());
    loaded.wire_links(engine->topology());
    engine->build();
  }

  amdgpu::Xcd *xcd(uint32_t idx = 0) { return soc_ptr->xcd(idx); }
  amdgpu::ShaderEngine *se(uint32_t idx = 0) { return soc_ptr->xcd(0)->shader_engine(idx); }
  amdgpu::GpuMemory *mem() { return gpu_mem; }
  amdgpu::ComputeUnitCore *cu(uint32_t idx = 0) { return se()->compute_unit(idx); }
  amdgpu::CommandProcessor *cp(uint32_t idx = 0) { return xcd(idx)->command_processor(); }
};

amdgpu::DispatchPacket make_pkt(uint64_t pc = 0, uint32_t wg_count = 1, uint32_t wfs = 1) {
  amdgpu::DispatchPacket pkt;
  pkt.kernel_entry_pc = pc;
  pkt.workgroup_count = wg_count;
  pkt.wfs_per_workgroup = wfs;
  pkt.sgprs_per_wf = 104;
  pkt.vgprs_per_wf = 256;
  return pkt;
}

TEST(GpuMemoryTest, ReadWriteRoundTrip) {
  VmFixture f;
  auto *mem = f.mem();

  mem->write32(0x1000, 0xDEADBEEF);
  EXPECT_EQ(mem->read32(0x1000), 0xDEADBEEF);

  mem->write64(0x2000, 0x0123456789ABCDEFULL);
  EXPECT_EQ(mem->read64(0x2000), 0x0123456789ABCDEFULL);
}

TEST(GpuMemoryTest, LoadImage) {
  VmFixture f;
  auto *mem = f.mem();

  const uint32_t program[] = {SOPP_S_NOP, SOPP_S_ENDPGM};
  mem->load_image(reinterpret_cast<const uint8_t *>(program), sizeof(program), 0x0);

  EXPECT_EQ(mem->read32(0x0), SOPP_S_NOP);
  EXPECT_EQ(mem->read32(0x4), SOPP_S_ENDPGM);
}

TEST(GpuMemoryTest, SparsePages) {
  VmFixture f;
  auto *mem = f.mem();

  mem->write32(0x0, 42);
  mem->write32(0x100000, 99);
  EXPECT_EQ(mem->read32(0x0), 42u);
  EXPECT_EQ(mem->read32(0x100000), 99u);
  EXPECT_EQ(mem->read32(0x50000), 0u);
}

TEST(VmLifecycleTest, CreateAndDestroy) {
  std::string json = R"({"max_ticks":10000,"num_threads":1,
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
              {"name":"cu[0:3]","type":"compute_unit","config":[
                {"key":"num_wf_slots","value":"10"},
                {"key":"sgprs_per_wf","value":"104"},
                {"key":"vgprs_per_wf","value":"256"},
                {"key":"lds_size_kb","value":"64"}
              ]}
            ]},
            {"name":"se1","type":"shader_engine","children":[
              {"name":"cu[0:3]","type":"compute_unit","config":[
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
        {"src":"xcd0.cp.req_2","dst":"xcd0.se0.cu2.cpl","latency":1,"weight":2},
        {"src":"xcd0.cp.req_3","dst":"xcd0.se1.cu0.cpl","latency":1,"weight":2},
        {"src":"xcd0.cp.req_4","dst":"xcd0.se1.cu1.cpl","latency":1,"weight":2},
        {"src":"xcd0.cp.req_5","dst":"xcd0.se1.cu2.cpl","latency":1,"weight":2},
        {"src":"xcd0.se0.cu0.req","dst":"xcd0.l2.cpl_0","latency":1,"weight":10},
        {"src":"xcd0.se0.cu1.req","dst":"xcd0.l2.cpl_1","latency":1,"weight":10},
        {"src":"xcd0.se0.cu2.req","dst":"xcd0.l2.cpl_2","latency":1,"weight":10},
        {"src":"xcd0.se1.cu0.req","dst":"xcd0.l2.cpl_3","latency":1,"weight":10},
        {"src":"xcd0.se1.cu1.req","dst":"xcd0.l2.cpl_4","latency":1,"weight":10},
        {"src":"xcd0.se1.cu2.req","dst":"xcd0.l2.cpl_5","latency":1,"weight":10}
      ]
    }
  })";
  auto loaded = config::load_config_from_string(json, kSchemaPath);
  auto *soc = loaded.soc();

  auto *xcd = soc->xcd(0);
  EXPECT_EQ(xcd->num_shader_engines(), 2u);
  EXPECT_EQ(xcd->shader_engine(0)->num_compute_units(), 3u);
  EXPECT_EQ(xcd->shader_engine(1)->num_compute_units(), 3u);
}

TEST(VmLifecycleTest, MissingArchFails) {
  const char *json = R"({"vm":{"gpu":{"num_shader_engines":1}}})";
  rj_vm_t *handle = nullptr;
  EXPECT_NE(rj_vm_create_from_string(json, kSchemaPath.c_str(), &handle), ROCJITSU_STATUS_SUCCESS);
}

TEST(VmLifecycleTest, InvalidArchFails) {
  const char *json = R"({"vm":{"arch":"bogus"}})";
  rj_vm_t *handle = nullptr;
  EXPECT_NE(rj_vm_create_from_string(json, kSchemaPath.c_str(), &handle), ROCJITSU_STATUS_SUCCESS);
}

class IsaTest : public ::testing::TestWithParam<std::string> {
protected:
  std::string arch() const { return GetParam(); }
};

TEST_P(IsaTest, RegisterAccess) {
  VmFixture f(arch());

  f.cp()->enqueue(make_pkt());
  f.cp()->step();

  auto *cu = f.cu();
  ASSERT_EQ(cu->num_wfs(), 1u);
  auto *w = cu->wf(0);

  EXPECT_EQ(w->exec(), ~0ULL);
  EXPECT_FALSE(w->is_halted());
  EXPECT_EQ(w->wf_size(), 64u);
  EXPECT_EQ(w->num_sgprs(), 104u);
  EXPECT_EQ(w->num_vgprs(), 256u);

  uint32_t sb = w->sgpr_alloc().base;
  uint32_t vb = w->vgpr_alloc().base;
  cu->write_sgpr(sb + 0, 42);
  cu->write_sgpr(sb + 103, 0xFFFFFFFF);
  EXPECT_EQ(cu->read_sgpr(sb + 0), 42u);
  EXPECT_EQ(cu->read_sgpr(sb + 103), 0xFFFFFFFF);
  EXPECT_EQ(cu->read_sgpr(sb + 50), 0u);

  cu->write_vgpr(vb + 1, 0, 100);
  cu->write_vgpr(vb + 1, 63, 200);
  EXPECT_EQ(cu->read_vgpr(vb + 1, 0), 100u);
  EXPECT_EQ(cu->read_vgpr(vb + 1, 63), 200u);
  EXPECT_EQ(cu->read_vgpr(vb + 1, 1), 0u); // v1 unaffected by writes to other lanes
}

TEST_P(IsaTest, RegisterFileIsolation) {
  VmFixture f(arch(), 1, 2);

  f.cp()->enqueue(make_pkt(0, 1, 2));
  f.cp()->step();

  auto *cu = f.cu();
  ASSERT_EQ(cu->num_wfs(), 2u);
  auto *w0 = cu->wf(0);
  auto *w1 = cu->wf(1);

  cu->write_sgpr(w0->sgpr_alloc().base + 0, 42);
  cu->write_sgpr(w1->sgpr_alloc().base + 0, 99);
  EXPECT_EQ(cu->read_sgpr(w0->sgpr_alloc().base + 0), 42u);
  EXPECT_EQ(cu->read_sgpr(w1->sgpr_alloc().base + 0), 99u);

  cu->write_vgpr(w0->vgpr_alloc().base + 0, 0, 100);
  cu->write_vgpr(w1->vgpr_alloc().base + 0, 0, 200);
  EXPECT_EQ(cu->read_vgpr(w0->vgpr_alloc().base + 0, 0), 100u);
  EXPECT_EQ(cu->read_vgpr(w1->vgpr_alloc().base + 0, 0), 200u);
}

TEST_P(IsaTest, DispatchAndCapacity) {
  VmFixture f(arch(), 1, 2);

  f.cp()->enqueue(make_pkt(0, 1, 3)); // 3 waves, only 2 fit.
  f.cp()->step();

  EXPECT_EQ(f.cu()->num_wfs(), 2u);
}

TEST_P(IsaTest, DispatchCreatesWavefronts) {
  VmFixture f(arch(), 2);

  f.cp()->enqueue(make_pkt(0x0, 2, 1));
  f.cp()->step();

  EXPECT_EQ(f.se()->compute_unit(0)->num_wfs(), 1u);
  EXPECT_EQ(f.se()->compute_unit(1)->num_wfs(), 1u);
}

TEST_P(IsaTest, MultipleWavesPerWorkgroup) {
  VmFixture f(arch());

  f.cp()->enqueue(make_pkt(0x200, 1, 3));
  f.cp()->step();

  EXPECT_EQ(f.cu()->num_wfs(), 3u);
}

TEST_P(IsaTest, RunToCompletion) {
  std::string json = R"({"max_ticks":10000,"num_threads":1,"vm":{"arch":")" + arch() +
                     R"("},)"
                     R"("topology":{"root":{"name":"soc","type":"soc","children":[)"
                     R"({"name":"vram","type":"gpu_memory"},)"
                     R"({"name":"xcd0","type":"xcd","children":[)"
                     R"({"name":"l2","type":"l2_cache"},)"
                     R"({"name":"cp","type":"command_processor"},)"
                     R"({"name":"se0","type":"shader_engine","children":[)"
                     R"({"name":"cu[0:1]","type":"compute_unit","config":[)"
                     R"({"key":"num_wf_slots","value":"10"},)"
                     R"({"key":"sgprs_per_wf","value":"104"},)"
                     R"({"key":"vgprs_per_wf","value":"256"},)"
                     R"({"key":"lds_size_kb","value":"64"})"
                     R"(]}]}]}]},"links":[)"
                     R"({"src":"xcd0.cp.req_0","dst":"xcd0.se0.cu0.cpl","latency":1,"weight":2},)"
                     R"({"src":"xcd0.se0.cu0.req","dst":"xcd0.l2.cpl_0","latency":1,"weight":10})"
                     R"(]}})";

  rj_vm_t *handle = nullptr;
  ASSERT_EQ(rj_vm_create_from_string(json.c_str(), kSchemaPath.c_str(), &handle),
            ROCJITSU_STATUS_SUCCESS);

  uint64_t ticks = 0;
  EXPECT_EQ(rj_vm_run(handle, &ticks), ROCJITSU_STATUS_SUCCESS);

  rj_vm_destroy(handle);
}

namespace enc {

constexpr uint32_t SGPR(uint32_t idx) { return idx; }
constexpr uint32_t VGPR_SRC(uint32_t idx) { return 256 + idx; }
constexpr uint32_t INLINE_CONST(uint32_t val) { return 128 + val; }

// SOPP: encoding[31:23]=0x17F, op[22:16], simm16[15:0]
constexpr uint32_t sopp(uint32_t op, uint16_t simm16 = 0) {
  return (0x17Fu << 23) | (op << 16) | simm16;
}

constexpr uint32_t s_branch(int16_t off) { return sopp(2, static_cast<uint16_t>(off)); }
constexpr uint32_t s_cbranch_scc0(int16_t off) { return sopp(4, static_cast<uint16_t>(off)); }
constexpr uint32_t s_cbranch_scc1(int16_t off) { return sopp(5, static_cast<uint16_t>(off)); }

// SOP1: encoding[31:23]=0x17D, sdst[22:16], op[15:8], ssrc0[7:0]
constexpr uint32_t sop1(uint32_t op, uint32_t sdst, uint32_t ssrc0) {
  return (0x17Du << 23) | (sdst << 16) | (op << 8) | ssrc0;
}
constexpr uint32_t s_mov_b32(uint32_t sdst, uint32_t ssrc0) { return sop1(0, sdst, ssrc0); }

// SOP2: encoding[31:30]=0x2, op[29:23], sdst[22:16], ssrc1[15:8], ssrc0[7:0]
constexpr uint32_t sop2(uint32_t op, uint32_t sdst, uint32_t ssrc0, uint32_t ssrc1) {
  return (0x2u << 30) | (op << 23) | (sdst << 16) | (ssrc1 << 8) | ssrc0;
}
constexpr uint32_t s_add_u32(uint32_t sdst, uint32_t s0, uint32_t s1) {
  return sop2(0, sdst, s0, s1);
}
constexpr uint32_t s_add_i32(uint32_t sdst, uint32_t s0, uint32_t s1) {
  return sop2(2, sdst, s0, s1);
}
// SOPC: encoding[31:23]=0x17E, op[22:16], ssrc1[15:8], ssrc0[7:0]
constexpr uint32_t sopc(uint32_t op, uint32_t ssrc0, uint32_t ssrc1) {
  return (0x17Eu << 23) | (op << 16) | (ssrc1 << 8) | ssrc0;
}
constexpr uint32_t s_cmp_eq_i32(uint32_t s0, uint32_t s1) { return sopc(0, s0, s1); }
constexpr uint32_t s_cmp_gt_i32(uint32_t s0, uint32_t s1) { return sopc(2, s0, s1); }
// VOP1: encoding[31:25]=0x3F, vdst[24:17], op[16:9], src0[8:0]
constexpr uint32_t vop1(uint32_t op, uint32_t vdst, uint32_t src0) {
  return (0x3Fu << 25) | (vdst << 17) | (op << 9) | src0;
}
constexpr uint32_t v_mov_b32(uint32_t vdst, uint32_t src0) { return vop1(1, vdst, src0); }

// VOP2: encoding[31]=0, op[30:25], vdst[24:17], vsrc1[16:9], src0[8:0]
constexpr uint32_t vop2(uint32_t op, uint32_t vdst, uint32_t src0, uint32_t vsrc1) {
  return (op << 25) | (vdst << 17) | (vsrc1 << 9) | src0;
}
constexpr uint32_t v_add_f32(uint32_t vdst, uint32_t s0, uint32_t vs1) {
  return vop2(1, vdst, s0, vs1);
}
constexpr uint32_t v_mul_f32(uint32_t vdst, uint32_t s0, uint32_t vs1) {
  return vop2(5, vdst, s0, vs1);
}
constexpr uint32_t v_add_u32(uint32_t vdst, uint32_t s0, uint32_t vs1) {
  return vop2(52, vdst, s0, vs1);
}
constexpr uint32_t v_cndmask_b32(uint32_t vdst, uint32_t s0, uint32_t vs1) {
  return vop2(0, vdst, s0, vs1);
}

// VOPC: encoding[31:25]=0x3E, op[24:17], vsrc1[16:9], src0[8:0]
constexpr uint32_t vopc(uint32_t op, uint32_t src0, uint32_t vsrc1) {
  return (0x3Eu << 25) | (op << 17) | (vsrc1 << 9) | src0;
}
constexpr uint32_t v_cmp_eq_f32(uint32_t s0, uint32_t vs1) { return vopc(66, s0, vs1); }

} // namespace enc

// Helper to set up a single wavefront and run instruction programs.
struct ExecFixture {
  VmFixture f;
  std::string arch_;

  explicit ExecFixture(const std::string &arch) : f(arch), arch_(arch) {}

  bool is_cdna4() const { return arch_ == "cdna4"; }
  uint32_t sopp_bytes() const { return 4u; }

  // Emit a SOPP instruction as 1 dword (32-bit encoding on all archs).
  std::vector<uint32_t> sopp(uint32_t word) const { return {word}; }

  // Flatten multiple instruction fragments into a single program.
  static std::vector<uint32_t> cat(std::initializer_list<std::vector<uint32_t>> parts) {
    std::vector<uint32_t> result;
    for (const auto &p : parts)
      result.insert(result.end(), p.begin(), p.end());
    return result;
  }

  void load_program(const std::vector<uint32_t> &words, uint64_t pc = 0x1000) {
    f.mem()->load_image(reinterpret_cast<const uint8_t *>(words.data()),
                        words.size() * sizeof(uint32_t), pc);
    f.cp()->enqueue(make_pkt(pc));
    f.cp()->step();
  }

  amdgpu::Wavefront *wf() { return f.cu()->wf(0); }
  amdgpu::ComputeUnitCore *cu() { return f.cu(); }
  bool step() { return f.cu()->step(); }

  uint32_t read_sgpr(uint32_t idx) { return cu()->read_sgpr(wf()->sgpr_alloc().base + idx); }
  void write_sgpr(uint32_t idx, uint32_t val) {
    cu()->write_sgpr(wf()->sgpr_alloc().base + idx, val);
  }
  uint32_t read_vgpr(uint32_t reg, uint32_t lane) {
    return cu()->read_vgpr(wf()->vgpr_alloc().base + reg, lane);
  }
  void write_vgpr(uint32_t reg, uint32_t lane, uint32_t val) {
    cu()->write_vgpr(wf()->vgpr_alloc().base + reg, lane, val);
  }
};

TEST_P(IsaTest, StepExecutesAndHalts) {
  ExecFixture fx(arch());
  uint32_t ss = fx.sopp_bytes();

  auto prog = ExecFixture::cat({fx.sopp(SOPP_S_NOP), fx.sopp(SOPP_S_NOP), fx.sopp(SOPP_S_ENDPGM)});
  fx.load_program(prog, 0x0);

  auto *cu = fx.cu();
  EXPECT_TRUE(cu->has_active_wfs());

  EXPECT_TRUE(cu->step());
  EXPECT_EQ(cu->wf(0)->pc, ss);

  EXPECT_TRUE(cu->step());
  EXPECT_EQ(cu->wf(0)->pc, 2 * ss);

  EXPECT_FALSE(cu->step());
  EXPECT_TRUE(cu->wf(0)->is_halted());
}

TEST_P(IsaTest, RoundRobinScheduling) {
  ExecFixture fx(arch());
  auto *mem = fx.f.mem();

  auto prog_a =
      ExecFixture::cat({fx.sopp(SOPP_S_NOP), fx.sopp(SOPP_S_NOP), fx.sopp(SOPP_S_ENDPGM)});
  auto prog_b = ExecFixture::cat({fx.sopp(SOPP_S_NOP), fx.sopp(SOPP_S_ENDPGM)});
  mem->load_image(reinterpret_cast<const uint8_t *>(prog_a.data()),
                  prog_a.size() * sizeof(uint32_t), 0x0);
  mem->load_image(reinterpret_cast<const uint8_t *>(prog_b.data()),
                  prog_b.size() * sizeof(uint32_t), 0x100);

  fx.f.cp()->enqueue(make_pkt(0x0));
  fx.f.cp()->enqueue(make_pkt(0x100));
  fx.f.cp()->step();
  fx.f.cp()->step();

  auto *cu = fx.cu();
  while (cu->has_active_wfs())
    cu->step();

  EXPECT_TRUE(cu->wf(0)->is_halted());
  EXPECT_TRUE(cu->wf(1)->is_halted());
}

TEST_P(IsaTest, EngineRunsToCompletion) {
  ExecFixture fx(arch());

  auto prog = ExecFixture::cat({fx.sopp(SOPP_S_NOP), fx.sopp(SOPP_S_ENDPGM)});
  fx.f.mem()->load_image(reinterpret_cast<const uint8_t *>(prog.data()),
                         prog.size() * sizeof(uint32_t), 0x0);
  fx.f.cp()->enqueue(make_pkt(0x0));

  fx.f.engine->run();

  EXPECT_TRUE(fx.cu()->wf(0)->is_halted());
}

TEST_P(IsaTest, SMovB32_InlineConst) {
  ExecFixture fx(arch());
  fx.load_program({enc::s_mov_b32(0, enc::INLINE_CONST(42)), SOPP_S_ENDPGM});
  fx.step();
  EXPECT_EQ(fx.read_sgpr(0), 42u);
}

TEST_P(IsaTest, SMovB32_SgprToSgpr) {
  ExecFixture fx(arch());
  fx.load_program({enc::s_mov_b32(1, enc::SGPR(0)), SOPP_S_ENDPGM});
  fx.write_sgpr(0, 0xDEADBEEF);
  fx.step();
  EXPECT_EQ(fx.read_sgpr(1), 0xDEADBEEFu);
}

TEST_P(IsaTest, SAddI32_NoOverflow) {
  ExecFixture fx(arch());
  fx.load_program({enc::s_add_i32(2, enc::SGPR(0), enc::SGPR(1)), SOPP_S_ENDPGM});
  fx.write_sgpr(0, 10);
  fx.write_sgpr(1, 20);
  fx.step();
  EXPECT_EQ(fx.read_sgpr(2), 30u);
  EXPECT_EQ(fx.wf()->status_raw() & 1u, 0u); // SCC=0 (no overflow)
}

TEST_P(IsaTest, SAddI32_Overflow) {
  ExecFixture fx(arch());
  fx.load_program({enc::s_add_i32(2, enc::SGPR(0), enc::SGPR(1)), SOPP_S_ENDPGM});
  fx.write_sgpr(0, 0x7FFFFFFF); // INT32_MAX
  fx.write_sgpr(1, 1);
  fx.step();
  EXPECT_EQ(fx.read_sgpr(2), 0x80000000u);   // wraps to INT32_MIN
  EXPECT_EQ(fx.wf()->status_raw() & 1u, 1u); // SCC=1 (overflow)
}

TEST_P(IsaTest, SAddU32_Carry) {
  ExecFixture fx(arch());
  fx.load_program({enc::s_add_u32(2, enc::SGPR(0), enc::SGPR(1)), SOPP_S_ENDPGM});
  fx.write_sgpr(0, 0xFFFFFFFF);
  fx.write_sgpr(1, 1);
  fx.step();
  EXPECT_EQ(fx.read_sgpr(2), 0u);            // wraps
  EXPECT_EQ(fx.wf()->status_raw() & 1u, 1u); // SCC=1 (carry)
}

TEST_P(IsaTest, SAddU32_NoCarry) {
  ExecFixture fx(arch());
  fx.load_program({enc::s_add_u32(2, enc::SGPR(0), enc::SGPR(1)), SOPP_S_ENDPGM});
  fx.write_sgpr(0, 100);
  fx.write_sgpr(1, 200);
  fx.step();
  EXPECT_EQ(fx.read_sgpr(2), 300u);
  EXPECT_EQ(fx.wf()->status_raw() & 1u, 0u); // SCC=0 (no carry)
}

TEST_P(IsaTest, SCmpEqI32_Equal) {
  ExecFixture fx(arch());
  fx.load_program({enc::s_cmp_eq_i32(enc::SGPR(0), enc::SGPR(1)), SOPP_S_ENDPGM});
  fx.write_sgpr(0, 42);
  fx.write_sgpr(1, 42);
  fx.step();
  EXPECT_EQ(fx.wf()->status_raw() & 1u, 1u); // SCC=1
}

TEST_P(IsaTest, SCmpEqI32_NotEqual) {
  ExecFixture fx(arch());
  fx.load_program({enc::s_cmp_eq_i32(enc::SGPR(0), enc::SGPR(1)), SOPP_S_ENDPGM});
  fx.write_sgpr(0, 42);
  fx.write_sgpr(1, 99);
  fx.step();
  EXPECT_EQ(fx.wf()->status_raw() & 1u, 0u); // SCC=0
}

TEST_P(IsaTest, SCmpGtI32) {
  ExecFixture fx(arch());
  fx.load_program({enc::s_cmp_gt_i32(enc::SGPR(0), enc::SGPR(1)), SOPP_S_ENDPGM});
  fx.write_sgpr(0, static_cast<uint32_t>(-5));  // -5 as signed
  fx.write_sgpr(1, static_cast<uint32_t>(-10)); // -10 as signed
  fx.step();
  EXPECT_EQ(fx.wf()->status_raw() & 1u, 1u); // -5 > -10, SCC=1
}

TEST_P(IsaTest, SBranch_Forward) {
  ExecFixture fx(arch());
  uint32_t ss = fx.sopp_bytes();
  // Branch skips one s_nop, lands on s_endpgm.
  // target = PC + 4 + offset*4; PC = 0x1000; target = 0x1000 + 2*ss
  int16_t off = static_cast<int16_t>((2 * ss - 4) / 4);
  fx.load_program(
      ExecFixture::cat({fx.sopp(enc::s_branch(off)), fx.sopp(SOPP_S_NOP), fx.sopp(SOPP_S_ENDPGM)}));
  fx.step();
  EXPECT_EQ(fx.wf()->pc, 0x1000u + 2 * ss);
}

TEST_P(IsaTest, SCbranchScc0_Taken) {
  ExecFixture fx(arch());
  uint32_t ss = fx.sopp_bytes();
  int16_t off = static_cast<int16_t>((2 * ss - 4) / 4);
  fx.load_program(ExecFixture::cat(
      {fx.sopp(enc::s_cbranch_scc0(off)), fx.sopp(SOPP_S_NOP), fx.sopp(SOPP_S_ENDPGM)}));
  fx.wf()->set_status_raw(fx.wf()->status_raw() & ~1u); // SCC=0
  fx.step();
  EXPECT_EQ(fx.wf()->pc, 0x1000u + 2 * ss); // branch taken, skipped NOP
}

TEST_P(IsaTest, SCbranchScc0_NotTaken) {
  ExecFixture fx(arch());
  uint32_t ss = fx.sopp_bytes();
  int16_t off = static_cast<int16_t>((2 * ss - 4) / 4);
  fx.load_program(ExecFixture::cat(
      {fx.sopp(enc::s_cbranch_scc0(off)), fx.sopp(SOPP_S_NOP), fx.sopp(SOPP_S_ENDPGM)}));
  fx.wf()->set_status_raw(fx.wf()->status_raw() | 1u); // SCC=1
  fx.step();
  EXPECT_EQ(fx.wf()->pc, 0x1000u + ss); // falls through
}

TEST_P(IsaTest, SCbranchScc1_Taken) {
  ExecFixture fx(arch());
  uint32_t ss = fx.sopp_bytes();
  int16_t off = static_cast<int16_t>((2 * ss - 4) / 4);
  fx.load_program(ExecFixture::cat(
      {fx.sopp(enc::s_cbranch_scc1(off)), fx.sopp(SOPP_S_NOP), fx.sopp(SOPP_S_ENDPGM)}));
  fx.wf()->set_status_raw(fx.wf()->status_raw() | 1u); // SCC=1
  fx.step();
  EXPECT_EQ(fx.wf()->pc, 0x1000u + 2 * ss); // branch taken
}

TEST_P(IsaTest, SEndpgm_Halts) {
  ExecFixture fx(arch());
  fx.load_program({SOPP_S_ENDPGM});
  EXPECT_FALSE(fx.wf()->is_halted());
  fx.step();
  EXPECT_TRUE(fx.wf()->is_halted());
}

TEST_P(IsaTest, VMovB32_PerLane) {
  ExecFixture fx(arch());
  // V_MOV_B32 v0, v1
  fx.load_program({enc::v_mov_b32(0, enc::VGPR_SRC(1)), SOPP_S_ENDPGM});
  fx.write_vgpr(1, 0, 100);
  fx.write_vgpr(1, 1, 200);
  fx.write_vgpr(1, 63, 999);
  fx.step();
  EXPECT_EQ(fx.read_vgpr(0, 0), 100u);
  EXPECT_EQ(fx.read_vgpr(0, 1), 200u);
  EXPECT_EQ(fx.read_vgpr(0, 63), 999u);
}

TEST_P(IsaTest, VAddF32_PerLane) {
  ExecFixture fx(arch());
  // V_ADD_F32 v2, v0, v1
  fx.load_program({enc::v_add_f32(2, enc::VGPR_SRC(0), 1), SOPP_S_ENDPGM});
  fx.write_vgpr(0, 0, std::bit_cast<uint32_t>(1.5f));
  fx.write_vgpr(1, 0, std::bit_cast<uint32_t>(2.5f));
  fx.write_vgpr(0, 1, std::bit_cast<uint32_t>(10.0f));
  fx.write_vgpr(1, 1, std::bit_cast<uint32_t>(20.0f));
  fx.step();
  EXPECT_EQ(std::bit_cast<float>(fx.read_vgpr(2, 0)), 4.0f);
  EXPECT_EQ(std::bit_cast<float>(fx.read_vgpr(2, 1)), 30.0f);
}

TEST_P(IsaTest, VMulF32_PerLane) {
  ExecFixture fx(arch());
  // V_MUL_F32 v2, v0, v1
  fx.load_program({enc::v_mul_f32(2, enc::VGPR_SRC(0), 1), SOPP_S_ENDPGM});
  fx.write_vgpr(0, 0, std::bit_cast<uint32_t>(3.0f));
  fx.write_vgpr(1, 0, std::bit_cast<uint32_t>(4.0f));
  fx.step();
  EXPECT_EQ(std::bit_cast<float>(fx.read_vgpr(2, 0)), 12.0f);
}

TEST_P(IsaTest, VAddU32_PerLane) {
  ExecFixture fx(arch());
  // V_ADD_U32 v2, v0, v1
  fx.load_program({enc::v_add_u32(2, enc::VGPR_SRC(0), 1), SOPP_S_ENDPGM});
  fx.write_vgpr(0, 0, 100);
  fx.write_vgpr(1, 0, 200);
  fx.write_vgpr(0, 3, 0xFFFFFFFF);
  fx.write_vgpr(1, 3, 1);
  fx.step();
  EXPECT_EQ(fx.read_vgpr(2, 0), 300u);
  EXPECT_EQ(fx.read_vgpr(2, 3), 0u); // wraps
}

TEST_P(IsaTest, VCmpEqF32_SetsVCC) {
  ExecFixture fx(arch());
  // V_CMP_EQ_F32 v0, v1
  fx.load_program({enc::v_cmp_eq_f32(enc::VGPR_SRC(0), 1), SOPP_S_ENDPGM});
  float pi = 3.14f;
  uint32_t pi_bits = std::bit_cast<uint32_t>(pi);
  // Lane 0: equal
  fx.write_vgpr(0, 0, pi_bits);
  fx.write_vgpr(1, 0, pi_bits);
  // Lane 1: not equal
  fx.write_vgpr(0, 1, std::bit_cast<uint32_t>(1.0f));
  fx.write_vgpr(1, 1, std::bit_cast<uint32_t>(2.0f));
  // Lane 2: equal (both zero)
  fx.write_vgpr(0, 2, 0);
  fx.write_vgpr(1, 2, 0);
  fx.wf()->set_vcc(0);
  fx.step();
  uint64_t vcc = fx.wf()->vcc();
  EXPECT_TRUE(vcc & (1ULL << 0));  // lane 0: equal
  EXPECT_FALSE(vcc & (1ULL << 1)); // lane 1: not equal
  EXPECT_TRUE(vcc & (1ULL << 2));  // lane 2: equal
}

TEST_P(IsaTest, VCndmaskB32) {
  ExecFixture fx(arch());
  // V_CNDMASK_B32 v2, v0, v1 - selects v1 where VCC bit is set, v0 otherwise
  fx.load_program({enc::v_cndmask_b32(2, enc::VGPR_SRC(0), 1), SOPP_S_ENDPGM});
  fx.write_vgpr(0, 0, 0xAAAA); // false source
  fx.write_vgpr(1, 0, 0xBBBB); // true source
  fx.write_vgpr(0, 1, 0xCCCC);
  fx.write_vgpr(1, 1, 0xDDDD);
  fx.wf()->set_vcc(0x2); // only lane 1 set
  fx.step();
  EXPECT_EQ(fx.read_vgpr(2, 0), 0xAAAAu); // VCC[0]=0 → src0
  EXPECT_EQ(fx.read_vgpr(2, 1), 0xDDDDu); // VCC[1]=1 → vsrc1
}

TEST_P(IsaTest, ExecMask_PreservesInactiveLanes) {
  ExecFixture fx(arch());
  // V_MOV_B32 v0, v1  with EXEC = only lane 0 active
  fx.load_program({enc::v_mov_b32(0, enc::VGPR_SRC(1)), SOPP_S_ENDPGM});
  fx.write_vgpr(0, 0, 0xAAAA);
  fx.write_vgpr(0, 1, 0xBBBB);
  fx.write_vgpr(1, 0, 100);
  fx.write_vgpr(1, 1, 200);
  fx.wf()->set_exec(1ULL); // only lane 0
  fx.step();
  EXPECT_EQ(fx.read_vgpr(0, 0), 100u);    // active lane: overwritten
  EXPECT_EQ(fx.read_vgpr(0, 1), 0xBBBBu); // inactive: preserved
}

TEST_P(IsaTest, MultiInstructionProgram) {
  ExecFixture fx(arch());
  // s_mov_b32 s0, 10
  // s_mov_b32 s1, 20
  // s_add_i32 s2, s0, s1
  // s_endpgm
  fx.load_program({
      enc::s_mov_b32(0, enc::INLINE_CONST(10)),
      enc::s_mov_b32(1, enc::INLINE_CONST(20)),
      enc::s_add_i32(2, enc::SGPR(0), enc::SGPR(1)),
      SOPP_S_ENDPGM,
  });
  fx.step(); // s_mov_b32 s0, 10
  EXPECT_EQ(fx.read_sgpr(0), 10u);
  fx.step(); // s_mov_b32 s1, 20
  EXPECT_EQ(fx.read_sgpr(1), 20u);
  fx.step(); // s_add_i32 s2, s0, s1
  EXPECT_EQ(fx.read_sgpr(2), 30u);
  fx.step(); // s_endpgm
  EXPECT_TRUE(fx.wf()->is_halted());
}

TEST_P(IsaTest, BranchLoop) {
  ExecFixture fx(arch());
  uint32_t ss = fx.sopp_bytes();
  // Scalar loop: s0 starts at 3, each iteration subtracts 1, loop back if s0 > 0.
  // 0x1000: s_add_i32 s0, s0, -1  (4B)
  // 0x1004: s_cmp_gt_i32 s0, 0    (4B)
  // 0x1008: s_cbranch_scc1 -3     (ss bytes - back to 0x1000)
  // 0x1008+ss: s_endpgm            (ss bytes)
  // Branch offset: target = 0x1008 + 4 + off*4 = 0x1000 → off = -3 (same for both arches)
  constexpr uint32_t NEG_1 = 193; // inline constant -1
  fx.load_program(ExecFixture::cat({
      {enc::s_add_i32(0, enc::SGPR(0), NEG_1)},
      {enc::s_cmp_gt_i32(enc::SGPR(0), enc::INLINE_CONST(0))},
      fx.sopp(enc::s_cbranch_scc1(-3)),
      fx.sopp(SOPP_S_ENDPGM),
  }));
  fx.write_sgpr(0, 3);

  // Iteration 1: s0 = 3-1 = 2, s0>0 → SCC=1, branch taken
  fx.step();
  fx.step();
  fx.step();
  EXPECT_EQ(fx.read_sgpr(0), 2u);
  EXPECT_EQ(fx.wf()->pc, 0x1000u); // back to loop start

  // Iteration 2: s0 = 2-1 = 1, s0>0 → SCC=1, branch taken
  fx.step();
  fx.step();
  fx.step();
  EXPECT_EQ(fx.read_sgpr(0), 1u);
  EXPECT_EQ(fx.wf()->pc, 0x1000u);

  // Iteration 3: s0 = 1-1 = 0, s0>0 → SCC=0, branch NOT taken
  fx.step();
  fx.step();
  fx.step();
  EXPECT_EQ(fx.read_sgpr(0), 0u);
  EXPECT_EQ(fx.wf()->pc, 0x1008u + ss); // falls through to endpgm

  fx.step(); // s_endpgm
  EXPECT_TRUE(fx.wf()->is_halted());
}

INSTANTIATE_TEST_SUITE_P(Cdna, IsaTest, ::testing::Values("cdna3", "cdna4"),
                         [](const auto &info) { return info.param; });

} // namespace
