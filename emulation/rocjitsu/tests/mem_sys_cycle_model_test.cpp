// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// Engine-driven test for the shared MemSysCycleModel component: a stub "CU"
/// component wired to the MemSys over simdojo Ports/Links sends a MemReqMsg, the
/// MemSys services it via SharedMemModel::service and replies a MemCompletionMsg on
/// the paired outbound port. Verifies the round-trip + the cold-miss completion cycle.

#include "plugins/mem_messages.h"
#include "plugins/mem_sys_cycle_model.h"

#include "cycle_model/SharedMemModel.h"   // baseline (single-CU) reference completion

#include "simdojo/sim/clocked.h"
#include "simdojo/sim/component.h"
#include "simdojo/sim/simulation.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <iostream>
#include <optional>
#include <vector>

using namespace rocjitsu::amdgpu;

// A cache-configured uarch (mirrors cdna4's cache geometry) so a cold-miss request
// walks the full L1->L2->HBM hierarchy in the shared model. (Mirrors the cfg in
// cu_cycle_model_test.cpp; the L1 hit_latency is baked into each SharedReq's
// arrive_cyc by the producer — here the test supplies arrive_cyc directly.)
static cycle_model::UarchConfig cache_cfg() {
  cycle_model::UarchConfig c;
  c.name = "t"; c.wave_size = 64; c.simds_per_cu = 4;
  c.wave_slots_per_simd = 8; c.front_end_issue_per_simd = 1;
  c.valu = {1, 4}; c.salu = {1, 1}; c.smem = {1, 20}; c.vmem = {1, 200}; c.lds_pipe = {1, 24};
  c.l1v = {/*size_kb=*/32, /*ways=*/4, /*line_bytes=*/128, /*hit_latency=*/6, /*miss_to_next_level=*/200};
  c.l1s = {16, 4, 64, 4, 200};
  c.l2  = {4096, 16, 128, 80, 400};
  c.mshrs_per_l1v = 16;
  c.hbm_channels = 8;
  c.l2_bytes_per_cycle = 1024;
  c.hbm_access_latency = 300;
  c.hbm_bytes_per_channel_per_cycle = 396;
  return c;
}

// Cold-miss latency for a request that arrives at the shared hierarchy at cycle 0:
// L2 hit_latency (80) + HBM access latency (300) = 380. The R2.2b convention adds the
// L1 hit_latency (6) into arrive_cyc at the producer; here the stub supplies
// arrive_cyc=6 to reproduce the full VMEM_COLD = 6 + 80 + 300 = 386.
constexpr uint64_t L1_HIT = 6;
constexpr uint64_t VMEM_COLD = L1_HIT + 80 + 300;   // 386

// A clocked stub "CU": holds a paired OUT(req)/IN(cpl) port. On its first clock edge
// it sends its queued MemReqMsg(s), then idles. The cpl handler captures the reply.
class StubCu : public simdojo::Clocked<simdojo::Component> {
public:
  StubCu(std::string name, const simdojo::ClockDomain &domain)
      : simdojo::Clocked<simdojo::Component>(std::move(name), domain) {
    req_out_ = add_port(std::make_unique<simdojo::Port>(
        this->name() + ".req_out", 0, this, simdojo::PortDirection::OUT,
        simdojo::PortProtocol::MEMORY));
    cpl_in_ = add_port(std::make_unique<simdojo::Port>(
        this->name() + ".cpl_in", 1, this, simdojo::PortDirection::IN,
        simdojo::PortProtocol::MEMORY));
    cpl_in_->set_handler([this](simdojo::Tick, simdojo::Message *msg) {
      auto *cpl = static_cast<MemCompletionMsg *>(msg);
      last_completion_cyc = cpl->completion_cyc;
      last_rid = cpl->rid;
      ++completions;
    });
  }

  simdojo::Port *req_out() { return req_out_; }
  simdojo::Port *cpl_in() { return cpl_in_; }

  // Queue a request to send on the first clock edge.
  void enqueue(cycle_model::MemReqId rid, std::vector<cycle_model::SharedReq> shared) {
    pending_.push_back(std::make_unique<MemReqMsg>(rid, std::move(shared), id()));
  }

  bool advance(simdojo::Tick /*now*/) override {
    for (auto &m : pending_) req_out_->send(std::move(m));
    pending_.clear();
    return false;   // one-shot: send then idle
  }

  std::optional<uint64_t> last_completion_cyc;
  cycle_model::MemReqId last_rid = 0;
  uint32_t completions = 0;

private:
  simdojo::Port *req_out_ = nullptr;
  simdojo::Port *cpl_in_ = nullptr;
  std::vector<std::unique_ptr<MemReqMsg>> pending_;
};

// Build a tiny engine: one MemSysCycleModel + one StubCu, wired both directions.
// Returns ownership of the engine so the caller can inspect the stub post-run.
namespace {

struct Harness {
  simdojo::SimulationEngine engine;
  StubCu *stub = nullptr;
  MemSysCycleModel *memsys = nullptr;

  explicit Harness(const cycle_model::UarchConfig &cfg)
      : engine([] {
          simdojo::SimulationEngine::Config c{};
          c.max_ticks = 1'000'000;
          c.num_threads = 1;
          return c;
        }()) {
    auto *clk = engine.topology().add_clock_domain("cu_clk", /*frequency_hz=*/1'000'000'000ull);
    auto root = std::make_unique<simdojo::CompositeComponent>("root");
    stub = static_cast<StubCu *>(root->add_child(std::make_unique<StubCu>("stub_cu", *clk)));
    memsys = static_cast<MemSysCycleModel *>(
        root->add_child(std::make_unique<MemSysCycleModel>("mem_sys")));
    // Allocate the MemSys's per-CU port pair and wire both directions (pre-build).
    auto ports = memsys->add_cu_ports();
    engine.topology().add_link(stub->req_out(), ports.req_in, /*latency=*/1);
    engine.topology().add_link(ports.cpl_out, stub->cpl_in(), /*latency=*/1);
    // Configure with the caller-owned (long-lived) config — SharedMemModel holds a
    // const ref, so the config must outlive engine.run().
    memsys->configure(cfg);
    engine.topology().set_root(std::move(root));
    engine.build();
  }
};

// Cross-CU harness: one shared MemSysCycleModel + TWO stub CUs, each wired with its
// own dedicated port pair (add_cu_ports() called twice). Both stubs send on the same
// clock edge, so their requests reach the shared model in the same cycle window and
// contend on the shared L2/HBM BwQueues.
struct Harness2 {
  simdojo::SimulationEngine engine;
  StubCu *stub_a = nullptr;
  StubCu *stub_b = nullptr;
  MemSysCycleModel *memsys = nullptr;

  explicit Harness2(const cycle_model::UarchConfig &cfg)
      : engine([] {
          simdojo::SimulationEngine::Config c{};
          c.max_ticks = 1'000'000;
          c.num_threads = 1;
          return c;
        }()) {
    auto *clk = engine.topology().add_clock_domain("cu_clk", /*frequency_hz=*/1'000'000'000ull);
    auto root = std::make_unique<simdojo::CompositeComponent>("root");
    stub_a = static_cast<StubCu *>(root->add_child(std::make_unique<StubCu>("stub_cu_a", *clk)));
    stub_b = static_cast<StubCu *>(root->add_child(std::make_unique<StubCu>("stub_cu_b", *clk)));
    memsys = static_cast<MemSysCycleModel *>(
        root->add_child(std::make_unique<MemSysCycleModel>("mem_sys")));
    // One dedicated port pair per stub CU.
    auto ports_a = memsys->add_cu_ports();
    auto ports_b = memsys->add_cu_ports();
    engine.topology().add_link(stub_a->req_out(), ports_a.req_in, /*latency=*/1);
    engine.topology().add_link(ports_a.cpl_out, stub_a->cpl_in(), /*latency=*/1);
    engine.topology().add_link(stub_b->req_out(), ports_b.req_in, /*latency=*/1);
    engine.topology().add_link(ports_b.cpl_out, stub_b->cpl_in(), /*latency=*/1);
    memsys->configure(cfg);
    engine.topology().set_root(std::move(root));
    engine.build();
  }
};

// Build M cold-miss lines that all map to the SAME HBM channel: channel =
// (l2base / line_bytes) % hbm_channels; with line_bytes=128, hbm_channels=8 a stride of
// 128*8 = 1024 bytes keeps every line on one channel while remaining distinct L2 lines.
constexpr uint32_t HBM_CHANNELS = 8;
constexpr uint32_t L2_LINE = 128;
static std::vector<cycle_model::SharedReq> same_channel_lines(uint64_t start_base,
                                                              uint32_t count) {
  std::vector<cycle_model::SharedReq> v;
  v.reserve(count);
  const uint64_t stride = static_cast<uint64_t>(L2_LINE) * HBM_CHANNELS;  // 1024 -> same channel
  for (uint32_t i = 0; i < count; ++i)
    v.push_back({/*line_base=*/start_base + i * stride, /*arrive_cyc=*/L1_HIT, /*skip_l2=*/false});
  return v;
}

// Service one request alone through a freshly-reset shared model to get the baseline
// (no cross-CU contention) completion cycle for that request.
static uint64_t baseline_completion(const cycle_model::UarchConfig &cfg,
                                    const std::vector<cycle_model::SharedReq> &reqs) {
  cycle_model::SharedMemModel shared(cfg);
  return shared.service(reqs);
}

}  // namespace

// A single cold-miss request returns the full L1->L2->HBM completion cycle (386).
TEST(MemSysCycleModel, ServicesColdMissOverPorts) {
  static const cycle_model::UarchConfig cfg_owned = cache_cfg();
  Harness h(cfg_owned);

  h.stub->enqueue(/*rid=*/0, {{/*line_base=*/0x1000, /*arrive_cyc=*/L1_HIT, /*skip_l2=*/false}});
  h.engine.run();

  EXPECT_EQ(h.stub->completions, 1u);
  ASSERT_TRUE(h.stub->last_completion_cyc.has_value());
  EXPECT_EQ(*h.stub->last_completion_cyc, VMEM_COLD);
  EXPECT_EQ(h.stub->last_rid, 0u);
}

// Two SharedReqs to DIFFERENT L2 lines in one request: both are distinct cold lines,
// so service() walks each through the shared L2/HBM hierarchy and dedups by physical
// L2 line. The completion reflects the max line completion; with these sub-line-rate
// transactions a couple of lines add no extra BW cycles, so assert a sane >= cold-miss.
TEST(MemSysCycleModel, ServicesTwoDistinctLines) {
  static const cycle_model::UarchConfig cfg_owned = cache_cfg();
  Harness h(cfg_owned);

  // Two addresses in different L2 lines (l2 line_bytes = 128).
  std::vector<cycle_model::SharedReq> reqs = {
      {/*line_base=*/0x2000, /*arrive_cyc=*/L1_HIT, /*skip_l2=*/false},
      {/*line_base=*/0x2080, /*arrive_cyc=*/L1_HIT, /*skip_l2=*/false},
  };
  h.stub->enqueue(/*rid=*/7, std::move(reqs));
  h.engine.run();

  EXPECT_EQ(h.stub->completions, 1u);
  ASSERT_TRUE(h.stub->last_completion_cyc.has_value());
  EXPECT_GE(*h.stub->last_completion_cyc, VMEM_COLD);   // two distinct cold lines serialize >= one
  EXPECT_EQ(h.stub->last_rid, 7u);
}

// Cross-CU contention: two stub CUs each send (same arrive_cyc) a request whose lines
// all land on the SAME HBM channel. Serviced alone, that request completes at the
// baseline cycle C1. With both CUs contending in the same cycle window, the shared
// per-channel BwQueue watermark advanced by the first-serviced CU pushes the
// second-serviced CU's completion strictly LATER than C1 — i.e. the shared MemSys
// serializes bandwidth ACROSS CUs (the whole point of R2). This is the key property
// that the in-process R2.2b loopback could not exhibit.
TEST(MemSysCycleModel, CrossCuBandwidthSerializes) {
  static const cycle_model::UarchConfig cfg_owned = cache_cfg();

  // Each CU drives 8 distinct cold lines on HBM channel 0. 8 lines * 128 B = 1024 B
  // booked on the channel per CU; the per-channel rate is 396 B/cyc, so the aggregate
  // 16 lines (2048 B) across both CUs in one cycle window exceeds the rate and forces
  // serialization. (Distinct start bases keep CU A and CU B lines non-overlapping.)
  constexpr uint32_t kLines = 8;
  auto lines_a = same_channel_lines(/*start_base=*/0x10000, kLines);
  auto lines_b = same_channel_lines(/*start_base=*/0x20000, kLines);

  // Baseline: one CU's request serviced alone (fresh shared model).
  const uint64_t C1 = baseline_completion(cfg_owned, lines_a);

  // Contended: both CUs send concurrently through the one shared MemSysCycleModel.
  Harness2 h(cfg_owned);
  h.stub_a->enqueue(/*rid=*/11, lines_a);
  h.stub_b->enqueue(/*rid=*/22, lines_b);
  h.engine.run();

  ASSERT_EQ(h.stub_a->completions, 1u);
  ASSERT_EQ(h.stub_b->completions, 1u);
  ASSERT_TRUE(h.stub_a->last_completion_cyc.has_value());
  ASSERT_TRUE(h.stub_b->last_completion_cyc.has_value());
  EXPECT_EQ(h.stub_a->last_rid, 11u);
  EXPECT_EQ(h.stub_b->last_rid, 22u);

  const uint64_t ca = *h.stub_a->last_completion_cyc;
  const uint64_t cb = *h.stub_b->last_completion_cyc;
  const uint64_t later = std::max(ca, cb);
  const uint64_t earlier = std::min(ca, cb);

  // Report the numbers (visible with --output-on-failure / on failure).
  std::cout << "[cross-CU] baseline C1=" << C1 << "  cu_a=" << ca << "  cu_b=" << cb
            << "  earlier=" << earlier << "  later(C2)=" << later
            << "  serialization=" << (later - C1) << std::endl;

  // The first-serviced CU sees an uncontended channel -> equals the baseline.
  EXPECT_EQ(earlier, C1);
  // The second-serviced CU is pushed strictly later by the shared BwQueue: cross-CU
  // bandwidth serialization is observed.
  EXPECT_GT(later, C1);
}
