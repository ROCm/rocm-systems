# Data Hazard Detector

Detects hazards caused by missing or insufficient `s_wait_*` instructions in AMD
GPU kernels. The plugin tracks vector loads and stores, scalar loads, LDS accesses,
and tensor DMA transfers, and reports accesses that happen before the wait counter
covering them has drained.

Where the [race detector](race-detector.md) answers "can two waves see different
values here", this plugin answers "did the wave wait long enough for its own and
its workgroup's asynchronous traffic". The two overlap on LDS but are separate
tools.

## Quick start

This section is a standalone guide. For general rocjitsu build and usage, see
the [README](../README.md); for the plugin system and sink configuration, see
[plugins.md](plugins.md).

Build rocjitsu (see [building.md](building.md)). No physical GPU is needed — the
emulator runs entirely on the CPU.

```bash
cd emulation/rocjitsu
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

### Example: reading a register before the load lands

```c++
// hazard_example.hip
#include <hip/hip_runtime.h>
#include <cstdio>

__global__ void double_values(const int *in, int *out) {
  int i = threadIdx.x;
  out[i] = in[i] * 2;
}

int main() {
  int *d_in, *d_out;
  hipMalloc(&d_in, 128 * sizeof(int));
  hipMalloc(&d_out, 128 * sizeof(int));
  double_values<<<1, 128>>>(d_in, d_out);
  hipDeviceSynchronize();
  hipFree(d_in);
  hipFree(d_out);
  printf("done\n");
}
```

Compiled normally this kernel has no hazard: the compiler emits
`s_wait_loadcnt 0` between the `global_load_dword` and the multiply. The
detector becomes interesting once that wait is gone, which is what the mutation
harness in `tests/data-hazard/` does automatically — see
[shaders/README.md](../tests/data-hazard/shaders/README.md).

Compile with an `--offload-arch` matching the config you run under (e.g.
`gfx950_cdna4.json` emulates gfx950). With `amdclang++`, pass `-O1` or higher;
the emulator does not support unoptimized GPU code objects.

```bash
hipcc -o /tmp/hazard_example hazard_example.hip --offload-arch=gfx950
```

Enable the plugin in your rocjitsu config file (`my_config.json`):

```json
{ "plugins": { "data_hazard": {} } }
```

Run it:

```bash
$BUILD_DIR/tools/rocjitsu/rocjitsu --config my_config.json -- /tmp/hazard_example
```

Each hazard is streamed to the sink as one line the moment it is found:

```
[data_hazard] pc=0x1a4 wave=0: RAW hazard: VGPR v3 read before async load completes | Add s_wait_loadcnt 0 before reading this register
```

At the end of the run a summary line reports the total, repeating only those
hazards whose duplicate count was still unknown when they were first streamed.

### Configuration

| Config key | Default | Description |
|---|---|---|
| `report_path` | `""` | File to write the JSON hazard report to. Empty means output goes only to the sink. |
| `verbose` | `false` | Report every occurrence instead of folding logically identical hazards into one entry with a `suppressed_occurrences` count. |

```json
{
  "plugins": { "data_hazard": { "report_path": "/tmp/hazards.json" } },
  "sinks": { "types": ["file"], "dir": "/tmp/output" }
}
```

With that config the streamed lines go to `/tmp/output/data_hazard.log` and the
structured report to `/tmp/hazards.json`.

The JSON report is an array with one object per hazard. Each entry carries the
message and suggestion, the `consumer` (the instruction that hit the hazard) and
the `producer` (the asynchronous operation it raced, or the writing wave of a
cross-wave race), both identified by dispatch, workgroup, wave, PC, disassembly
text and raw ISA words.

The report is written when the run ends, which for a launched application means
process exit. While a run is still going the sink is the only output.

## What this plugin detects

- **RAW**: a register or memory location is read before the asynchronous
  operation writing it has drained. Covers VMEM loads (`loadcnt`), scalar loads
  (`kmcnt`), LDS (`dscnt`), direct-to-LDS loads, and tensor DMA into LDS
  (`tensorcnt`).
- **WAW**: a write lands on a register or LDS address that a pending
  asynchronous write still targets.
- **WAR**: an LDS address is overwritten while an asynchronous read of it is
  still pending. See the limitation on register-level WAR below.
- **Cross-wave LDS races**: two waves in a workgroup touch the same LDS address
  within one barrier epoch and at least one of them writes.
- **Cross-workgroup global races**: two workgroups touch overlapping global
  bytes and at least one writes. Sub-dword accesses that fall in the same
  four-byte shadow entry without sharing a byte do not race. Two atomics to one
  address are ordered by the hardware and do not race, but an ordinary access
  racing an atomic is reported whichever of the two ran first.

Every report names the wait that would have prevented it, and the suggestion
names the same resource the message names — a register hazard suggests waiting
before reading that register, an LDS hazard before reading that address.

## How it works

The engine keeps per-wave state holding the FIFOs of pending asynchronous
operations, one per wait counter. Instruction callbacks feed it three kinds of
event:

1. **Resource access** — a read or write of a register, LDS address or global
   address. The engine checks the pending FIFOs for an entry covering the
   accessed range; an overlap that the wave has not waited for is a hazard.
2. **Wait** — an `s_wait_*` instruction. The engine retires entries from the
   named counter's FIFO down to the immediate's target depth.
3. **Barrier** — closes the current LDS epoch for the workgroup. Accesses
   recorded in the closed epoch are checked pairwise for cross-wave races and
   then discarded. The epoch closes when the simulator resolves the barrier,
   not when a wave reaches it: a wave that arrives early shares its epoch with
   the waves still issuing pre-barrier accesses, so those races are compared.

Pending operations are tracked at the granularity they are issued: registers as
spans of consecutive registers, LDS and global memory as byte ranges. A
multi-register load is one pending entry spanning every register it writes, so a
read of any register in the span is caught.

A tensor DMA into LDS names no address in its encoding: the LDS base, element
size, tile dimensions, iteration stride and row padding all come from a
descriptor the transfer reads out of scalar registers. The plugin reads that same
descriptor as the instruction issues and tracks the stretches of LDS the transfer
will write, one pending entry each, so a read anywhere in the tile before
`s_wait_tensorcnt` is caught. A descriptor that disables its transfer, or one the
executor will reject, writes no LDS and leaves nothing pending.

Wait counts are read from the disassembled operand — `vmcnt(1) expcnt(0)
lgkmcnt(3)` — and carried to the adapter as the counts themselves rather than
repacked into an immediate. The s_waitcnt field widths differ by family, four
bits of lgkmcnt on CDNA against six on the RDNA families, so a single layout
would truncate the counts of the families it was not written for and drain more
operations than the wait named. A frontend handed a real hardware immediate
instead decodes it with the field layout of the ISA it is executing
(`hazard_core/include/detail/waitcnt_decode.h`).

## Directory layout

Source files are under `lib/rocjitsu/src/rocjitsu/vm/plugins/data_hazard/`:

```
data_hazard/
├── plugin.h/.cpp        rocjitsu plugin: hooks, config, report writing
├── adapter.h/.cpp       translates rocjitsu instruction views into engine events
├── report.h/.cpp        warning collection, JSON report, sink line formatting
└── hazard_core/         simulator-neutral engine (no rocjitsu dependency)
    ├── include/         public API — see hazard_core/README.md
    │   └── detail/      implementation headers, not installed
    └── src/
```

## Tests

Unit tests are part of the rocjitsu test suite (`emulation/rocjitsu/tests/`):

- `data-hazard/hazard_engine_tests.cpp` — drives `DataHazardEngine` directly.
- `data-hazard/hazard_helpers_tests.cpp` — FIFO operations, wait decoding,
  suggestion wording.
- `data-hazard/data_hazard_adapter_tests.cpp` — instruction view to engine event
  translation.
- `data-hazard/tensor_lds_range_tests.cpp` — the LDS a tensor DMA writes, derived
  from its descriptor.

```bash
ctest --test-dir build -R "DataHazard|Hazard"
```

End-to-end coverage comes from the mutation harness in `tests/data-hazard/`,
which compiles each shader in `tests/data-hazard/shaders/`, removes one
`s_wait_*` at a time, and reports which removals the plugin catches:

```bash
cd tests/data-hazard
python -m mutate_and_test --hazard-detection --arch gfx950 shaders/*.hip
```

## Limitations

- **Performance**: rocjitsu emulates GPU execution on the CPU, and hazard
  detection adds overhead on top of that. This is a correctness tool — use
  targeted test cases, not production workloads.

- **No register-level WAR**: the engine can represent a WAR on a VGPR still
  being read by a pending store, and the previous `vm/plugins/hazard`
  integration reported them, but nothing in the current rocjitsu store path
  feeds `track_vector_read`, so no register WAR is reported today. LDS WAR is
  reported. This is a known coverage gap rather than a design decision.

- **Wait immediates only**: hazards are resolved by `s_wait_*` and `s_barrier`.
  Software synchronization the hardware does not express through a wait counter
  is invisible to the engine.

- **Padded tensor tiles are tracked as one span**: a tensor DMA that pads its
  rows apart leaves gaps of untransferred bytes between them, and the tile is
  tracked as the span the padded rows occupy rather than as one range per row.
  An access to a gap byte while the transfer is still outstanding is therefore
  reported, and the alternative leaves every later LDS access scanning a pending
  entry per row of the tile.

- **XCNT is not a data counter**: `s_wait_xcnt` tracks address translation
  replay rather than data completion, so removing it produces no hazard. The
  mutation harness excludes it by default.

- **Two workgroups retained per global address**: a four-byte shadow entry keeps
  two writers, two readers and two atomics, each pair on distinct workgroups,
  which is enough for any pair that overlaps to be reported. Three or more
  workgroups writing mutually disjoint bytes of one dword exhaust the slots, and
  the displaced workgroup is no longer available as a conflict partner.

- **Kernel name resolution**: names come from the code object; unresolved
  symbols appear as `?`.
