# gpumetrics

A focused library + CLI for reading AMD GPU metrics per **GPU**, per **partition**,
and per **socket**. Nothing else: no gRPC, no daemon, no job tracking, no diagnostics.
Just metrics, from pluggable backends, behind a small stable API.

It reads telemetry from **amdsmi** (temperatures, power, clocks, utilization, VRAM, ECC,
PCIe, fan) and hardware performance counters from **rocprofiler-sdk** (occupancy, cache/
memory bandwidth, activity, FLOPs) and correlates the two views of the *same* physical GPU,
so you address a device once and get metrics from whichever backend serves them.

See [DESIGN.md](DESIGN.md) for the architecture and the RDC lessons that motivated it.

## Quick start

```bash
cmake -B build -GNinja -DCMAKE_CXX_COMPILER=g++
ninja -C build

# discover topology (sockets / GPUs / partitions / which plugins serve each)
GPUMETRICS_PLUGIN_PATH=build/plugins/amdsmi:build/plugins/rocprofiler \
LD_LIBRARY_PATH=build/src \
  build/cli/gpumetrics discover

# read specific metrics for GPU 0
... gpumetrics read -e gpu:0 -m temp.edge,clock.gfx,power.average_socket,prof.gpu_util_pct

# watch loop, CSV, twice a second, 10 samples
... gpumetrics dmon -e gpu:0 -m temp.edge,activity.gfx -i 500 -c 10 --format csv
```

Once installed (plugins under `<libdir>/gpumetrics/`), the env vars aren't needed.

## CLI

| Command                    | Purpose                                              |
| -------------------------- | ---------------------------------------------------- |
| `discover`                 | List sockets, GPUs, partitions, identities, providers |
| `list-metrics [--scope]`   | List available metric keys (unit / provider / desc)  |
| `read -e <sel> -m <keys>`  | One-shot read                                         |
| `dmon [-e][-m][-i][-c]`    | Watch loop (Ctrl-C to stop)                           |

**Entity selectors:** `gpu:N`, `g<N>.<P>` (partition P of GPU N), `socket:N`,
`bdf:DDDD:BB:DD.F`, `uuid:HEX`, bare `N`, or `all`.
**Formats:** `--format table|csv|json`.

## Metric keys

Metrics use dotted string keys, e.g. `temp.edge`, `power.average_socket`, `clock.gfx`,
`activity.gfx`, `mem.vram.used`, `ecc.total.correctable`, `pcie.speed`, `fan.rpm`,
`prof.active_cycles`, `prof.occupancy_pct`, `prof.mem_read_bytes`. Run `list-metrics` to
see everything the loaded plugins support on your hardware. A metric available from more
than one backend is served by the higher-priority provider (default `amdsmi` before
`rocprofiler`); overridable via the collector's provider priority.

## Library

C++ (`include/gpumetrics/gpumetrics.h`):

```cpp
#include <gpumetrics/gpumetrics.h>
using namespace gpumetrics;

gpum_status st;
auto c = Collector::Create({}, &st);
for (const auto& d : c->devices())
  std::cout << EntityLabel(d.id) << " " << d.name << "\n";
auto s = c->read(c->resolve("gpu:0")->id, "temp.edge");
if (s.ok()) std::cout << "edge temp: " << s.to_string() << " C\n";
```

A flat **C API** (`include/gpumetrics/capi.h`) provides the same functionality for FFI,
and a **Rust** binding lives under [`rust/`](rust/).

## Plugins

A plugin is a shared library named `libgpumetrics_<name>.so` that exports a single symbol
`gpum_plugin_entry_v1` returning a versioned C vtable
(`include/gpumetrics/plugin_abi.h`). It declares the devices it serves (each with
correlation keys: BDF, oam_id, KFD node id, UUID, render minor) and a declarative table of
metric descriptors, and implements a batched `read`. The core loads plugins with
`RTLD_GLOBAL` (required so rocprofiler-sdk can find the rocprofiler plugin's
`rocprofiler_configure` symbol), correlates their devices into canonical GPUs, and routes
each metric read to the owning plugin. New backends are just new `.so`s, no core changes.

Correlation groups a physical GPU by its masked BDF (PCIe function stripped) and oam_id,
which stay stable across partitioning; on a partitioned GPU (e.g. MI350X CPX) each partition
is a distinct PCIe function and becomes a partition entity under the one physical GPU. A
plugin that aborts on init against an unsupported config (some rocprofiler-sdk versions do on
large partition layouts) is isolated by a fork-probe and skipped rather than crashing the
tool; set `GPUMETRICS_NO_PROBE=1` to disable the probe.

## Tests

```bash
ctest --test-dir build --output-on-failure
```

Covers device correlation and routing (with hardware-free mock plugins), the C API, the
end-to-end Collector, live-hardware sanity ranges (skipped when no GPU), and a
side-by-side comparison harness that validates values against `amd-smi`.
