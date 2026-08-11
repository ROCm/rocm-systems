# hazard_core

The data hazard engine, with no dependency on rocjitsu or any other simulator.
It consumes abstract execution events — dispatches, waves, instructions,
resource accesses, waits and barriers — and reports RAW, WAR, WAW and cross-wave
race findings.

rocjitsu's `data_hazard` plugin is one frontend for it; see
[docs/data-hazard.md](../../../../../../../../docs/data-hazard.md) for what the
engine detects and how it works. This note covers driving it from your own
simulator.

## Linking

The engine is built twice: statically into the rocjitsu plugin, and as a
standalone `libdata_hazard_core.so` with default visibility for out-of-tree
consumers. Disable the standalone library with `-DRJ_BUILD_DATA_HAZARD_CORE=OFF`.

After `cmake --install`, headers land in
`<prefix>/include/rocjitsu/data_hazard` and the library in `<prefix>/lib`:

```cmake
find_path(HAZARD_CORE_INCLUDE_DIR data_hazard_engine.h
          PATH_SUFFIXES rocjitsu/data_hazard)
find_library(HAZARD_CORE_LIBRARY data_hazard_core)

target_include_directories(my_simulator PRIVATE ${HAZARD_CORE_INCLUDE_DIR})
target_link_libraries(my_simulator PRIVATE ${HAZARD_CORE_LIBRARY})
```

The engine requires C++17.

## Public headers

Headers include each other by bare name, so the install directory itself has to
be on the include path.

| Header | Contents |
|---|---|
| `data_hazard_engine.h` | `DataHazardEngine` — the entry point |
| `simulator_api.h` | `DataHazardSimulatorApi` (the event interface the engine implements) and `SimulatorInstructionFormatter` |
| `hazard_events.h` | Event and finding types: `InstructionEvent`, `ResourceAccessEvent`, `BarrierEvent`, `WaitAction`, `HazardFinding` |
| `types.h` | `WaitCntType`, `ExecutionKey`, and the per-wave state types |
| `data_hazard_state.h` | `EngineWarning`, `HazardWarningSink`, and the engine's state structures |
| `wait_suggestion.h` | `make_wait_suggestion` and the wording of the advice attached to a hazard |
| `hash_utils.h`, `spin_lock.h` | Small utilities the state types expose in their interfaces |

Everything under `detail/` is an implementation header. It is not installed, and
its contents can change without notice.

## Driving the engine

Implement two interfaces, then feed events in the order your simulator executes
them.

```c++
#include "data_hazard_engine.h"
#include "data_hazard_state.h"
#include "simulator_api.h"

using namespace hazard_core;

// 1. Turn an instruction into text your users will recognise.
class MyFormatter final : public SimulatorInstructionFormatter {
public:
  std::string format_instruction(const InstructionDescriptor &inst) const override {
    return my_disassemble(inst.pc);
  }
};

// 2. Receive findings as they are made.
class MySink final : public HazardWarningSink {
public:
  void emit_warning(const EngineWarning &warning) const override {
    std::fprintf(stderr, "%s | %s\n", warning.message.c_str(), warning.suggestion.c_str());
  }
};

MyFormatter formatter;
MySink sink;

DataHazardEngine engine;
engine.set_instruction_formatter(&formatter);
engine.set_warning_sink(&sink);
engine.set_diagnostic_handler([](const std::string &m) { log_error(m); });
```

Then, per wave:

```c++
engine.on_dispatch_begin(dispatch_id);
engine.on_workgroup_begin(dispatch_id, cluster_id, workgroup_id);
engine.on_wave_begin(wave_key);

for (each executed instruction) {
  // One InstructionEvent first: it carries the wait this instruction performs
  // and which counters guard its asynchronous destinations.
  engine.on_instruction(instruction_event);

  // Then one ResourceAccessEvent per register or memory range it touches.
  for (each access)
    engine.on_resource_access(access_event);

  if (is_barrier)
    engine.on_barrier(barrier_event);
}

engine.on_wave_end(wave_key);
engine.on_workgroup_end(dispatch_id, cluster_id, workgroup_id);
engine.on_dispatch_end(dispatch_id);
engine.on_shutdown();
```

Ordering matters in one place: the `InstructionEvent` must arrive before the
resource accesses belonging to it, because it establishes which wait counter
guards the instruction's asynchronous destinations. The engine does not infer
that from access ordering.

## Behaviour worth knowing

**It never throws on the event path.** Malformed events — an implausible access
size, an LDS address past the end of the block, a wait on a counter that cannot
be drained — are counted and passed to the diagnostic handler, and the event is
dropped. Check `rejected_event_count()` if you want to fail a run on them.
An adapter that silently produces bad events otherwise looks like a clean run.

**It is safe to call from multiple threads.** Per-wave state is held behind a
shared mutex and per-wave locks, so simulators that execute waves concurrently
can call in from any thread. The sink and formatter you install are called from
those threads, so they must be thread-safe too.

**Findings are deduplicated by default.** Logically identical hazards are folded
together by the frontend, not the engine — the engine emits every finding. If
you want folding, do it in your sink.

**Reset between runs.** `reset()` drops all wave, workgroup and epoch state
without disturbing the installed formatter, sink or diagnostic handler.

## Inspecting state

`wave_snapshot()`, `wave_snapshots()` and `wave_count()` return copies of the
per-wave pending-operation state, which is what the engine's own tests assert
against. `warning_snapshot()` returns every finding made so far, for frontends
that want to write a report at the end rather than stream.
