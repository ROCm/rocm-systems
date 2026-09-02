// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file trace_source_test.cpp
/// @brief Tests for the timing model's ingress: stream shape, dispatch
/// finalisation, replay, and the identity a fact carries.

#include "aql_queue.h"
#include "embedded_schema.h"
#include "rocjitsu/config/config_loader.h"
#include "rocjitsu/vm/amdgpu/command_processor.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "rocjitsu/vm/amdgpu/l2_cache.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"
#include "rocjitsu/vm/plugins/execution_plugin_group.h"
#include "rocjitsu/vm/soc.h"
#include "rocjitsu/vm/timing/trace_plugin.h"
#include "rocjitsu/vm/timing/trace_source.h"

#include "rocjitsu/base/rj_compiler.h"
RJ_DIAGNOSTIC_PUSH
RJ_DIAGNOSTIC_IGNORE_PEDANTIC
#include "hsa/AMDHSAKernelDescriptor.h"
RJ_DIAGNOSTIC_POP

#include "simdojo/sim/simulation.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <format>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

using namespace rocjitsu;
using namespace rocjitsu::timing;

namespace {

/// @brief A modelled consumer standing in for the compute unit the source will
///        eventually feed.
class TraceSink final : public simdojo::Component {
public:
  TraceSink() : simdojo::Component("trace_sink") {}

  void initialize() override {
    if (in_ != nullptr)
      return;
    in_ = add_port(std::make_unique<simdojo::Port>("trace_in", /*port_id=*/0, this,
                                                   simdojo::PortDirection::IN));
    in_->set_handler([this](simdojo::Tick now, simdojo::Message *message) {
      const auto *trace = dynamic_cast<const TraceMessage *>(message);
      if (trace == nullptr) {
        ADD_FAILURE() << "a message arrived that was not a TraceMessage";
        return;
      }
      received.push_back(trace->event());
      arrival_ticks.push_back(now);
      header_sequences.push_back(message->header().sequence_num);
    });
  }

  simdojo::Port *in() { return in_; }

  std::vector<TraceEvent> received;
  std::vector<simdojo::Tick> arrival_ticks;
  std::vector<uint64_t> header_sequences;

private:
  simdojo::Port *in_ = nullptr;
};

/// @brief A source wired to a sink over a clocked link.
class TraceRig {
public:
  explicit TraceRig(simdojo::Tick latency = 0) {
    auto root = std::make_unique<simdojo::CompositeComponent>("timing");
    source =
        static_cast<TimingTraceSource *>(root->add_child(std::make_unique<TimingTraceSource>()));
    sink = static_cast<TraceSink *>(root->add_child(std::make_unique<TraceSink>()));
    engine.topology().set_root(std::move(root));
    // create() runs initialize(), which is where both components build their
    // ports; the link can only be added once those exist.
    engine.create();
    engine.topology().add_link(source->output_port(), sink->in(), latency);
    engine.shutdown();
    engine.create();
  }

  simdojo::SimulationEngine engine{{}};
  TimingTraceSource *source = nullptr;
  TraceSink *sink = nullptr;
};

/// @brief The source's ended-dispatch window, mirrored for the tests that have
///        to roll past it. Kept in step with TimingTraceSource::kEndedWindow.
constexpr uint32_t kEndedWindowForTest = 1024;

/// @brief One event of @p kind for @p dispatch.
TraceEvent make_event(TraceEventKind kind, uint32_t dispatch, uint32_t wavefront = 0) {
  TraceEvent event;
  event.kind = kind;
  event.identity.dispatch_id = dispatch;
  event.identity.wavefront_id = wavefront;
  return event;
}

} // namespace

TEST(TimingTraceSourceTest, StampsAConsecutiveGaplessStream) {
  TraceRig rig;

  EXPECT_EQ(rig.source->next_sequence(), 1u);
  EXPECT_TRUE(rig.source->submit(make_event(TraceEventKind::DISPATCH_BEGIN, 1)));
  EXPECT_TRUE(rig.source->submit(make_event(TraceEventKind::WAVE_BEGIN, 1)));
  EXPECT_TRUE(rig.source->submit(make_event(TraceEventKind::INSTRUCTION, 1)));
  rig.engine.run_until_idle();

  std::vector<uint64_t> sequences;
  for (const TraceEvent &event : rig.sink->received)
    sequences.push_back(event.sequence);
  // Gapless from one, so a consumer can tell a dropped event from a reordered
  // one rather than guessing.
  EXPECT_EQ(sequences, (std::vector<uint64_t>{1, 2, 3}));
  // And carried in the header too, so a consumer can correlate without looking
  // inside the payload.
  EXPECT_EQ(rig.sink->header_sequences, sequences);
  EXPECT_EQ(rig.source->accepted(), 3u);
  EXPECT_EQ(rig.source->next_sequence(), 4u);
}

TEST(TimingTraceSourceTest, RefusesEventsAfterTheirDispatchHasEnded) {
  // A dispatch's terms are closed when it ends. An event arriving afterwards
  // would either be dropped downstream or quietly change a number already
  // reported, and neither is visible.
  TraceRig rig;

  ASSERT_TRUE(rig.source->submit(make_event(TraceEventKind::DISPATCH_BEGIN, 7)));
  ASSERT_TRUE(rig.source->submit(make_event(TraceEventKind::INSTRUCTION, 7)));
  ASSERT_TRUE(rig.source->submit(make_event(TraceEventKind::DISPATCH_END, 7)));

  EXPECT_FALSE(rig.source->submit(make_event(TraceEventKind::INSTRUCTION, 7)));
  EXPECT_FALSE(rig.source->submit(make_event(TraceEventKind::WAVE_END, 7)));
  EXPECT_EQ(rig.source->rejected(), 2u);

  // Another dispatch is unaffected.
  EXPECT_TRUE(rig.source->submit(make_event(TraceEventKind::INSTRUCTION, 8)));

  rig.engine.run_until_idle();
  ASSERT_EQ(rig.sink->received.size(), 4u);
  // A refused event takes no sequence number, so the stream stays gapless.
  EXPECT_EQ(rig.sink->received.back().sequence, 4u);
}

TEST(TimingTraceSourceTest, ADispatchIdReusedByANewDispatchIsAcceptedAgain) {
  // Dispatch ids are reused. Refusing the reuse would silently drop every
  // kernel after the first wrap, and remembering every id that ever ended
  // would grow without bound.
  TraceRig rig;
  ASSERT_TRUE(rig.source->submit(make_event(TraceEventKind::DISPATCH_BEGIN, 3)));
  ASSERT_TRUE(rig.source->submit(make_event(TraceEventKind::DISPATCH_END, 3)));
  ASSERT_FALSE(rig.source->submit(make_event(TraceEventKind::INSTRUCTION, 3)));

  EXPECT_TRUE(rig.source->submit(make_event(TraceEventKind::DISPATCH_BEGIN, 3)));
  EXPECT_TRUE(rig.source->submit(make_event(TraceEventKind::INSTRUCTION, 3)));
}

TEST(TimingTraceSourceTest, ItKeepsNoClockOfItsOwn) {
  // The source decides what happened, never when. The tick a message departs
  // at is the timing engine's, and the engine belongs to whoever advances it.
  TraceRig rig(/*latency=*/250);

  rig.source->submit(make_event(TraceEventKind::INSTRUCTION, 1));
  // Nothing has been scheduled by the source itself: the only event in the
  // queue is the message's arrival.
  EXPECT_EQ(rig.engine.run_until_idle(), 250u);
  ASSERT_EQ(rig.sink->arrival_ticks.size(), 1u);
  EXPECT_EQ(rig.sink->arrival_ticks[0], 250u);
  EXPECT_EQ(rig.engine.events_processed(), 1u);

  // A second fact submitted from the same tick departs from that tick, not
  // from a clock the source advanced on its own.
  rig.source->submit(make_event(TraceEventKind::INSTRUCTION, 1));
  EXPECT_EQ(rig.engine.run_until_idle(), 500u);
  ASSERT_EQ(rig.sink->arrival_ticks.size(), 2u);
  EXPECT_EQ(rig.sink->arrival_ticks[1], 500u);
}

TEST(TimingTraceSourceTest, ARecordedStreamReplaysToTheSameMessages) {
  TraceRig live;
  live.source->submit(make_event(TraceEventKind::DISPATCH_BEGIN, 2));
  live.source->submit(make_event(TraceEventKind::INSTRUCTION, 2, /*wavefront=*/5));
  live.source->submit(make_event(TraceEventKind::DISPATCH_END, 2));
  live.engine.run_until_idle();
  ASSERT_EQ(live.sink->received.size(), 3u);

  TraceRig replayed;
  replayed.source->replay(live.sink->received);
  replayed.engine.run_until_idle();

  ASSERT_EQ(replayed.sink->received.size(), live.sink->received.size());
  for (size_t i = 0; i < live.sink->received.size(); ++i) {
    EXPECT_EQ(replayed.sink->received[i].sequence, live.sink->received[i].sequence);
    EXPECT_EQ(replayed.sink->received[i].kind, live.sink->received[i].kind);
    EXPECT_TRUE(replayed.sink->received[i].identity == live.sink->received[i].identity);
  }
  // The replayed source carries on from where the recording left off, so a
  // recording can be a prefix of a longer run.
  EXPECT_EQ(replayed.source->next_sequence(), 4u);
  EXPECT_FALSE(replayed.source->submit(make_event(TraceEventKind::INSTRUCTION, 2)))
      << "replay did not carry the recording's dispatch finalisation forward";
}

TEST(TimingTraceSourceTest, AMalformedRecordingIsRefusedWhole) {
  TraceRig rig;
  std::vector<TraceEvent> recording;
  recording.push_back(make_event(TraceEventKind::DISPATCH_BEGIN, 1));
  recording.back().sequence = 1;
  recording.push_back(make_event(TraceEventKind::INSTRUCTION, 1));
  recording.back().sequence = 3; // a gap

  EXPECT_THROW(rig.source->replay(recording), std::invalid_argument);
  rig.engine.run_until_idle();
  EXPECT_TRUE(rig.sink->received.empty()) << "half of a rejected recording was sent";

  recording[1].sequence = 2;
  recording.push_back(make_event(TraceEventKind::DISPATCH_END, 1));
  recording.back().sequence = 3;
  recording.push_back(make_event(TraceEventKind::INSTRUCTION, 1));
  recording.back().sequence = 4; // after its dispatch ended

  EXPECT_THROW(rig.source->replay(recording), std::invalid_argument);
  rig.engine.run_until_idle();
  EXPECT_TRUE(rig.sink->received.empty());
}

TEST(TimingTraceSourceTest, AnUnsupportedFactIsCarriedRatherThanDropped) {
  // A model that dropped what it could not represent would report a kernel it
  // had modelled completely. The honest answer is that it modelled everything
  // except this.
  TraceRig rig;
  TraceEvent event = make_event(TraceEventKind::MEMORY, 4);
  event.supported = false;
  event.unsupported_reason = "memory access was routed to no pipeline";
  rig.source->submit(std::move(event));
  rig.engine.run_until_idle();

  ASSERT_EQ(rig.sink->received.size(), 1u);
  EXPECT_FALSE(rig.sink->received[0].supported);
  EXPECT_EQ(rig.sink->received[0].unsupported_reason, "memory access was routed to no pipeline");
}

TEST(TimingTraceSourceTest, ResetRestartsTheStream) {
  TraceRig rig;
  rig.source->submit(make_event(TraceEventKind::DISPATCH_BEGIN, 1));
  rig.source->submit(make_event(TraceEventKind::DISPATCH_END, 1));
  ASSERT_FALSE(rig.source->submit(make_event(TraceEventKind::INSTRUCTION, 1)));

  rig.source->reset();
  EXPECT_EQ(rig.source->next_sequence(), 1u);
  EXPECT_EQ(rig.source->accepted(), 0u);
  EXPECT_EQ(rig.source->rejected(), 0u);
  EXPECT_TRUE(rig.source->submit(make_event(TraceEventKind::INSTRUCTION, 1)));
}

// ============================================================================
// End to end: a real dispatch, through the plugin, into the stream
// ============================================================================

namespace {

/// @brief A functional GPU running a real dispatch, with the timing plugin
///        attached and the trace source living in a second engine of its own.
///
/// @details Two engines, deliberately. The functional side has one; the timing
/// plane has another that nothing but modelled components run on. That
/// separation is the architecture this series is built around, and it is why
/// the trace source is a SimDojo component rather than a callback.
class LiveTraceRig {
public:
  explicit LiveTraceRig(uint32_t wf_slots = 4) {
    const std::string json = std::format(R"({{
      "max_ticks":2000000,"num_threads":1,"exec_mode":"functional",
      "vm":{{"arch":"cdna4","gpu":{{"device":{{"wave_front_size":64,"num_sdma_engines":0}}}}}},
      "topology":{{"root":{{"name":"soc","type":"soc","children":[
        {{"name":"vram","type":"gpu_memory"}},
        {{"name":"xcd0","type":"xcd","children":[
          {{"name":"l2","type":"l2_cache"}},
          {{"name":"cp","type":"command_processor"}},
          {{"name":"se0","type":"shader_engine","children":[
            {{"name":"cu[0:1]","type":"compute_unit","config":[
              {{"key":"num_wf_slots","value":"{}"}},
              {{"key":"sgprs_per_wf","value":"104"}},
              {{"key":"vgprs_per_wf","value":"256"}},
              {{"key":"lds_size_kb","value":"64"}}
            ]}}
          ]}}
        ]}}
      ]}},"links":[
        {{"src":"xcd0.cp.req_0","dst":"xcd0.se0.cu0.cpl","latency":1,"weight":2}},
        {{"src":"xcd0.se0.cu0.req","dst":"xcd0.l2.cpl_0","latency":1,"weight":10}}
      ]}}}}
    )",
                                         wf_slots);
    auto loaded = config::load_config_from_string(json, rocjitsu::kEmbeddedSchema);
    soc = loaded.soc();
    memory = loaded.memory();
    functional = std::make_unique<simdojo::SimulationEngine>(loaded.engine_config);
    functional->topology().set_root(loaded.take_root());
    loaded.wire_links(functional->topology());
    functional->create();

    auto root = std::make_unique<simdojo::CompositeComponent>("timing");
    source =
        static_cast<TimingTraceSource *>(root->add_child(std::make_unique<TimingTraceSource>()));
    sink = static_cast<TraceSink *>(root->add_child(std::make_unique<TraceSink>()));
    timing.topology().set_root(std::move(root));
    timing.create();
    timing.topology().add_link(source->output_port(), sink->in(), /*latency=*/0);
    timing.shutdown();
    timing.create();

    group = std::make_shared<ExecutionPluginGroup>(PluginSinkConfig{});
    auto owned = std::make_unique<TimingTracePlugin>(*source);
    plugin = owned.get();
    group->add(std::move(owned));
    soc->set_plugin_group(group);
    group->onInit();
  }

  ~LiveTraceRig() { group->onShutdown(); }

  amdgpu::CommandProcessor *cp() { return soc->xcd(0)->command_processor(); }

  /// @brief Write @p code as a kernel object at @p addr.
  uint64_t write_kernel(uint64_t addr, std::span<const uint32_t> code) {
    using namespace rocr::llvm::amdhsa;
    kernel_descriptor_t kd{};
    kd.kernel_code_entry_byte_offset = sizeof(kernel_descriptor_t);
    AMDHSA_BITS_SET(kd.compute_pgm_rsrc1, COMPUTE_PGM_RSRC1_GRANULATED_WORKITEM_VGPR_COUNT, 31);
    AMDHSA_BITS_SET(kd.compute_pgm_rsrc1, COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT, 12);
    AMDHSA_BITS_SET(kd.compute_pgm_rsrc2, COMPUTE_PGM_RSRC2_USER_SGPR_COUNT, 2);
    memory->load_image(reinterpret_cast<const uint8_t *>(&kd), sizeof(kd), addr);
    memory->load_image(reinterpret_cast<const uint8_t *>(code.data()), code.size() * 4,
                       addr + sizeof(kernel_descriptor_t));
    return addr;
  }

  /// @brief Run @p code as a one-workgroup dispatch and drain both engines.
  ///
  /// @details The queue is created once and reused: the command processor
  /// keys fan-out on (queue, process), so a second queue with the same id is
  /// refused.
  void run_kernel(std::span<const uint32_t> code, uint32_t grid = 64) {
    const uint64_t kernel_object = write_kernel(0x1000, code);
    if (!queue)
      queue = std::make_unique<test::AqlQueue>(memory, cp());
    queue->dispatch(kernel_object, grid, /*workgroup_size_x=*/64);
    for (uint32_t i = 0; i < 200000 && functional->step(); ++i) {
    }
    timing.run_until_idle();
  }

  // Declaration order is teardown order reversed, and it is load-bearing here.
  // The SoC holds its own reference to the plugin group, so the plugin -- which
  // holds a reference to the trace source -- outlives this rig's `group` member
  // and dies with `functional`. `functional` therefore has to be destroyed
  // before `timing`, which owns the source; and `queue` before `functional`,
  // whose command processor it points at.
  SoC *soc = nullptr;
  amdgpu::GpuMemory *memory = nullptr;
  simdojo::SimulationEngine timing{{}};
  TimingTraceSource *source = nullptr;
  TraceSink *sink = nullptr;
  std::shared_ptr<ExecutionPluginGroup> group;
  TimingTracePlugin *plugin = nullptr;
  std::unique_ptr<simdojo::SimulationEngine> functional;
  std::unique_ptr<test::AqlQueue> queue;
};

/// @brief Events of one kind, in order.
std::vector<const TraceEvent *> of_kind(const std::vector<TraceEvent> &events,
                                        TraceEventKind kind) {
  std::vector<const TraceEvent *> result;
  for (const TraceEvent &event : events) {
    if (event.kind == kind)
      result.push_back(&event);
  }
  return result;
}

} // namespace

TEST(TimingTracePluginTest, ARealDispatchProducesAReplayableStream) {
  LiveTraceRig rig;
  // s_mov_b32 s0, 1 ; s_mov_b32 s1, 2 ; s_endpgm
  const uint32_t code[] = {0xBE800081u, 0xBE810082u, 0xBF810000u};
  rig.run_kernel(code);

  const std::vector<TraceEvent> &stream = rig.sink->received;
  ASSERT_FALSE(stream.empty());

  // Gapless from one, in submission order: a consumer can tell a dropped event
  // from a reordered one.
  for (size_t i = 0; i < stream.size(); ++i)
    EXPECT_EQ(stream[i].sequence, i + 1);

  // The dispatch brackets everything, and it is announced with its shape.
  const auto begins = of_kind(stream, TraceEventKind::DISPATCH_BEGIN);
  const auto ends = of_kind(stream, TraceEventKind::DISPATCH_END);
  ASSERT_EQ(begins.size(), 1u);
  ASSERT_EQ(ends.size(), 1u);
  EXPECT_EQ(stream.front().kind, TraceEventKind::DISPATCH_BEGIN);
  EXPECT_EQ(stream.back().kind, TraceEventKind::DISPATCH_END);
  ASSERT_NE(begins.front()->dispatch, nullptr);
  EXPECT_EQ(begins.front()->dispatch->grid_size_x, 64u);
  EXPECT_EQ(begins.front()->dispatch->workgroup_size_x, 64u);
  EXPECT_TRUE(begins.front()->supported);

  // One wavefront, beginning and ending once, with the instructions it retired
  // in between.
  const auto wave_begins = of_kind(stream, TraceEventKind::WAVE_BEGIN);
  const auto wave_ends = of_kind(stream, TraceEventKind::WAVE_END);
  ASSERT_EQ(wave_begins.size(), 1u);
  ASSERT_EQ(wave_ends.size(), 1u);
  const auto instructions = of_kind(stream, TraceEventKind::INSTRUCTION);
  ASSERT_EQ(instructions.size(), std::size(code));

  // The identity functional execution used, not a modelled one.
  const TraceIdentity &identity = instructions.front()->identity;
  EXPECT_EQ(identity.compute_unit_id, rig.soc->xcd(0)->shader_engine(0)->compute_unit(0)->id());
  EXPECT_EQ(identity.dispatch_id, begins.front()->identity.dispatch_id);
  EXPECT_EQ(identity.workgroup_id, wave_begins.front()->identity.workgroup_id);
  EXPECT_EQ(identity.wavefront_id, wave_begins.front()->identity.wavefront_id);

  // Ordinals count this wavefront's own instructions, from one, in issue
  // order, and each event names the PC it retired at.
  for (size_t i = 0; i < instructions.size(); ++i) {
    EXPECT_EQ(instructions[i]->instruction_ordinal, i + 1);
    EXPECT_FALSE(instructions[i]->mnemonic.empty());
    EXPECT_TRUE(instructions[i]->identity == identity);
  }
  EXPECT_EQ(wave_ends.front()->instruction_ordinal, instructions.size());
  EXPECT_EQ(rig.plugin->unsupported(), 0u);

  // And the recording is a stream in its own right: replayed into a fresh
  // source it produces the same messages, which is what lets a model be
  // re-run against a recording rather than against the emulator.
  TraceRig replayed;
  replayed.source->replay(stream);
  replayed.engine.run_until_idle();
  ASSERT_EQ(replayed.sink->received.size(), stream.size());
  for (size_t i = 0; i < stream.size(); ++i) {
    EXPECT_EQ(replayed.sink->received[i].sequence, stream[i].sequence);
    EXPECT_EQ(replayed.sink->received[i].kind, stream[i].kind);
    EXPECT_EQ(replayed.sink->received[i].pc, stream[i].pc);
    EXPECT_TRUE(replayed.sink->received[i].identity == stream[i].identity);
  }
}

TEST(TimingTracePluginTest, AMemoryInstructionCarriesTheRoutedFacts) {
  LiveTraceRig rig;
  // s_load_dword s4, s[0:1], 0x0 ; s_waitcnt lgkmcnt(0) ; s_endpgm
  const uint32_t code[] = {0xC0020100u, 0x00000000u, 0xBF8CC07Fu, 0xBF810000u};
  rig.run_kernel(code);

  const auto memory_events = of_kind(rig.sink->received, TraceEventKind::MEMORY);
  ASSERT_EQ(memory_events.size(), 1u);
  const TraceEvent &event = *memory_events.front();
  ASSERT_NE(event.memory, nullptr);
  // The route the memory system was actually given, with the addresses copied
  // out of the borrowed spans before the pipeline reuses them.
  EXPECT_EQ(event.memory->route, amdgpu::MemoryRoute::SCALAR);
  ASSERT_EQ(event.memory->addresses.size(), 1u);
  EXPECT_EQ(event.memory->wait_counter, amdgpu::WaitCounterType::LGKMCNT);
  EXPECT_TRUE(event.memory->is_load);
  EXPECT_TRUE(event.supported);

  // The whole identity, compared the way a model keying on it would: a single
  // field left at its default is enough to make every lookup miss.
  const auto instructions = of_kind(rig.sink->received, TraceEventKind::INSTRUCTION);
  ASSERT_FALSE(instructions.empty());
  EXPECT_TRUE(event.identity == instructions.front()->identity)
      << "a memory event cannot be matched to the wavefront that made it";
  // And it names the instruction, not just the PC: inside a loop the PC
  // repeats and cannot say which access this is.
  EXPECT_GT(event.instruction_ordinal, 0u);
  EXPECT_LE(event.instruction_ordinal, instructions.size());
}

TEST(TimingTraceSourceTest, ASourceWithNowhereToSendRefusesRatherThanCrashes) {
  // submit() is called from inside functional execution, from a plugin hook
  // with a lock held, on a path with no way to handle a failure. Every way the
  // timing plane can be unable to take an event has to come back as a refusal.
  auto root = std::make_unique<simdojo::CompositeComponent>("timing");
  auto *source =
      static_cast<TimingTraceSource *>(root->add_child(std::make_unique<TimingTraceSource>()));
  simdojo::SimulationEngine engine({});
  engine.topology().set_root(std::move(root));

  // No engine yet.
  EXPECT_FALSE(source->submit(make_event(TraceEventKind::INSTRUCTION, 1)));
  engine.create();
  // An engine, but nothing wired to the output port.
  EXPECT_FALSE(source->submit(make_event(TraceEventKind::INSTRUCTION, 1)));
  EXPECT_EQ(source->rejected(), 2u);
  // A refusal takes no sequence number, so the stream stays gapless.
  EXPECT_EQ(source->next_sequence(), 1u);
}

TEST(TimingTraceSourceTest, ASourceOutlivingItsEngineRefusesRatherThanThrows) {
  TraceRig rig;
  ASSERT_TRUE(rig.source->submit(make_event(TraceEventKind::INSTRUCTION, 1)));
  rig.engine.run_until_idle();

  // The timing plane torn down while functional execution is still running --
  // which is the ordinary shutdown order, not an exotic one.
  rig.engine.shutdown();
  EXPECT_FALSE(rig.source->submit(make_event(TraceEventKind::INSTRUCTION, 1)));
  EXPECT_EQ(rig.source->rejected(), 1u);
  EXPECT_EQ(rig.sink->received.size(), 1u);
}

TEST(TimingTraceSourceTest, ReplayIsRefusedOntoASourceThatHasAlreadySent) {
  // Replaying onto a live source would hand out sequence numbers it had used
  // and wind the counter backwards, which is the one property a consumer uses
  // to tell a dropped event from a reordered one.
  TraceRig rig;
  ASSERT_TRUE(rig.source->submit(make_event(TraceEventKind::INSTRUCTION, 1)));
  ASSERT_TRUE(rig.source->submit(make_event(TraceEventKind::INSTRUCTION, 1)));

  std::vector<TraceEvent> recording;
  recording.push_back(make_event(TraceEventKind::INSTRUCTION, 2));
  recording.back().sequence = 1;
  EXPECT_THROW(rig.source->replay(recording), std::logic_error);

  rig.engine.run_until_idle();
  ASSERT_EQ(rig.sink->received.size(), 2u);
  EXPECT_EQ(rig.sink->received[0].sequence, 1u);
  EXPECT_EQ(rig.sink->received[1].sequence, 2u);
  EXPECT_EQ(rig.source->next_sequence(), 3u);
}

TEST(TimingTraceSourceTest, AnEmptyReplayChangesNothing) {
  // The degenerate case of the same bug: an empty recording used to clear the
  // ended-dispatch set and rewind the counter, and report success doing it.
  TraceRig rig;
  ASSERT_TRUE(rig.source->submit(make_event(TraceEventKind::DISPATCH_BEGIN, 1)));
  ASSERT_TRUE(rig.source->submit(make_event(TraceEventKind::DISPATCH_END, 1)));

  rig.source->replay({});

  EXPECT_EQ(rig.source->next_sequence(), 3u);
  EXPECT_FALSE(rig.source->submit(make_event(TraceEventKind::INSTRUCTION, 1)))
      << "an empty replay resurrected a dispatch that had ended";
}

TEST(TimingTraceSourceTest, TheEndedDispatchSetIsBounded) {
  // Dispatch ids climb monotonically and only wrap after hundreds of millions,
  // so remembering every one that ended would grow for the life of the run. A
  // late event arrives just after its dispatch ended, which is what the window
  // has to cover.
  TraceRig rig;
  ASSERT_TRUE(rig.source->submit(make_event(TraceEventKind::DISPATCH_END, /*dispatch=*/0)));
  for (uint32_t id = 1; id <= 4; ++id)
    ASSERT_TRUE(rig.source->submit(make_event(TraceEventKind::DISPATCH_END, id)));

  // Just-ended dispatches are refused, which is the guarantee.
  EXPECT_FALSE(rig.source->submit(make_event(TraceEventKind::INSTRUCTION, 4)));
  EXPECT_FALSE(rig.source->submit(make_event(TraceEventKind::INSTRUCTION, 1)));

  // Past the window, the oldest is forgotten rather than remembered forever.
  for (uint32_t id = 5; id < 5 + 2048; ++id)
    ASSERT_TRUE(rig.source->submit(make_event(TraceEventKind::DISPATCH_END, id)));
  EXPECT_TRUE(rig.source->submit(make_event(TraceEventKind::INSTRUCTION, /*dispatch=*/0)));
  EXPECT_FALSE(rig.source->submit(make_event(TraceEventKind::INSTRUCTION, /*dispatch=*/5 + 2047)));
}

TEST(TimingTraceSourceTest, AReusedDispatchIdKeepsItsLatestFinalisation) {
  // The window is a deque of ends and a set of live finalisations, and a
  // re-announcement can only reach the set. Evicting the deque entry a
  // re-announcement orphaned must not take the *later* end of the same id with
  // it, or a dispatch that has ended starts accepting events again.
  TraceRig rig;
  ASSERT_TRUE(rig.source->submit(make_event(TraceEventKind::DISPATCH_BEGIN, 5)));
  ASSERT_TRUE(rig.source->submit(make_event(TraceEventKind::DISPATCH_END, 5)));
  // Reused, and ended again. Two entries in the window now stand for one id,
  // and only the second one is in force.
  ASSERT_TRUE(rig.source->submit(make_event(TraceEventKind::DISPATCH_BEGIN, 5)));
  ASSERT_TRUE(rig.source->submit(make_event(TraceEventKind::DISPATCH_END, 5)));

  // Roll the window just far enough to evict the orphaned entry and no
  // further: the two ends above plus this many is one past the window, which
  // drops exactly the oldest.
  for (uint32_t id = 100; id < 100 + kEndedWindowForTest - 1; ++id)
    ASSERT_TRUE(rig.source->submit(make_event(TraceEventKind::DISPATCH_END, id)));

  EXPECT_FALSE(rig.source->submit(make_event(TraceEventKind::INSTRUCTION, 5)))
      << "evicting a spent window entry reopened a dispatch that had ended";
}

TEST(TimingTraceSourceTest, ARefusedEventChangesNothing) {
  // A refusal has to leave the source exactly as it found it. A DISPATCH_BEGIN
  // refused because the timing plane was down must not un-end the dispatch that
  // previously held its id: the events of that first dispatch would then be
  // accepted into the stream as though they belonged to the second.
  TraceRig rig;
  ASSERT_TRUE(rig.source->submit(make_event(TraceEventKind::DISPATCH_BEGIN, 9)));
  ASSERT_TRUE(rig.source->submit(make_event(TraceEventKind::DISPATCH_END, 9)));
  rig.engine.run_until_idle();

  rig.engine.shutdown();
  EXPECT_FALSE(rig.source->submit(make_event(TraceEventKind::DISPATCH_BEGIN, 9)));
  rig.engine.create();

  EXPECT_FALSE(rig.source->submit(make_event(TraceEventKind::INSTRUCTION, 9)))
      << "a refused dispatch-begin resurrected a dispatch that had ended";
}

TEST(TimingTracePluginTest, InstructionOrdinalsRestartInAReusedSlot) {
  // Ordinals count a wavefront's own instructions. One compute-unit slot and
  // two workgroups, so the second wavefront gets the slot the first has just
  // released -- and must start again from one, or it looks like a continuation
  // of its predecessor.
  LiveTraceRig rig(/*wf_slots=*/1);
  const uint32_t code[] = {0xBE800081u, 0xBF810000u};
  rig.run_kernel(code, /*grid=*/128);

  const std::vector<TraceEvent> &stream = rig.sink->received;
  ASSERT_EQ(of_kind(stream, TraceEventKind::WAVE_BEGIN).size(), 2u);
  const auto instructions = of_kind(stream, TraceEventKind::INSTRUCTION);
  ASSERT_EQ(instructions.size(), 2 * std::size(code));

  // Both wavefronts landed in the same slot, one after the other.
  const auto waves = of_kind(stream, TraceEventKind::WAVE_BEGIN);
  ASSERT_EQ(waves[0]->identity.wavefront_id, waves[1]->identity.wavefront_id);
  ASSERT_NE(waves[0]->identity.workgroup_id, waves[1]->identity.workgroup_id);

  std::vector<uint64_t> ordinals;
  for (const TraceEvent *event : instructions)
    ordinals.push_back(event->instruction_ordinal);
  EXPECT_EQ(ordinals, (std::vector<uint64_t>{1, 2, 1, 2}))
      << "the reused slot carried the previous wavefront's count forward";
}
