---
name: discovering-profiling-options
description: Answers questions about what rocprofiler-systems configuration options exist and how to use them — e.g. "what hardware counters are available on my system", "what ROCm APIs can I trace", "what settings control sampling/output", "how do I persist my configuration" — by mapping the question to the right `rocprof-sys-avail` or `rocprof-sys-run --help=<topic>` command. Use when the user wants to discover or explore ROCPROFSYS_* settings, Timemory components, hardware counters, ROCm API tracing domains, or GPU metrics, rather than apply them. Do NOT use this for picking a starting --preset for a workload — that's `choosing-a-profiling-preset`.
---

# Discovering Profiling Options

Maps a "what's available" or "how do I configure X" question to the exact
`rocprof-sys-avail` / `rocprof-sys-run --help=<topic>` command that answers it.
`rocprof-sys-avail` is self-updating and authoritative — prefer running it over
reciting static documentation.

## Prerequisites

A working `rocprof-sys-avail` binary, built/installed alongside
`rocprof-sys-run`/`rocprof-sys-sample`. If it is not on `PATH`, run
`source <install-prefix>/share/rocprofiler-systems/setup-env.sh` first.

## Discovery command reference

| User question | Command | Notes |
| --- | --- | --- |
| What settings/env vars exist, with descriptions? | `rocprof-sys-avail -S -bd` | `-b` suppresses the current-value/availability column, `-d` adds the description column |
| What hardware counters are available on my system? | `rocprof-sys-avail -H -bd -A` | `-A`/`--available` restricts to counters this machine can actually use; add `-c CPU` or `-c GPU` to split by device |
| What ROCm APIs can be traced? | `rocprof-sys-avail -bd -r ROCM_DOMAINS` | Regex-filters settings to the `ROCPROFSYS_ROCM_DOMAINS` row, listing every valid domain (`hip_runtime_api`, `hsa_api`, `kernel_dispatch`, `rccl_api`, `kfd_events`, ...) |
| What operations/API calls exist within one ROCm domain? | `rocprof-sys-avail --list-domains` then `rocprof-sys-avail --list-operations <domain>` | e.g. `--list-operations hip_runtime_api` lists individual HIP calls. Only domains printed by `--list-domains` are accepted: composites must be expanded (`hip_api` → `hip_runtime_api`, `hip_compiler_api`; `hsa_api` → `hsa_core_api`, `hsa_amd_ext_api`, `hsa_image_ext_api`, `hsa_finalize_ext_api`; `kfd_events` → the `kfd_*` domains; `roctx` → `marker_api`), and `kernel_dispatch` has no operation list |
| What Timemory components can I collect? | `rocprof-sys-avail -C -bd` | add `-A` to restrict to components usable on this build |
| What settings exist for one topic (tracing, sampling, output, ...)? | `rocprof-sys-avail --list-categories` then `rocprof-sys-avail -S -bd -c settings::<category>` | category names don't always match plain English — it's `settings::trace`, not `settings::tracing` — so always check `--list-categories` first |
| What GPU power/temp/utilization metrics can I sample? | `rocprof-sys-avail -bd -r AMD_SMI_METRICS` | surfaces `ROCPROFSYS_AMD_SMI_METRICS` choices (`busy, temp, power, mem_usage, ...`) |
| What CLI flags does `rocprof-sys-run` have for topic X? | `rocprof-sys-run --help=<topic>` | topics: `preset, general, tracing, profiling, output, sampling, process, counters, backend, execution, debug, misc, gpu, cpu, rocm, parallel` |
| How do I persist chosen settings across runs? | `rocprof-sys-avail -G ~/.rocprof-sys.cfg [--all]` then `export ROCPROFSYS_CONFIG_FILE=~/.rocprof-sys.cfg` | `--all` includes descriptions/categories in the generated file; formats are txt (default), json, xml via `-F` |

## How to answer a discovery question

1. Identify which axis the question is about: settings/env-vars, Timemory
   components, hardware counters, ROCm API domains, GPU metrics, or CLI flags.
2. Pick the matching row from the table above.
3. Run the command (or hand it to the user) and report the relevant rows —
   don't dump the entire table if the question was narrow; combine with
   `-r <regex>` / `-c <category>` / `-A` to narrow first.
4. If the result is still broad, suggest adding `-A`/`--available` (installed/
   supported only) or a category/regex filter.

## Persisting configuration

Once the right settings are identified, freeze them into a config file instead
of re-typing environment variables every run:

```bash
rocprof-sys-avail -G ~/.rocprof-sys.cfg --all   # --all adds descriptions/categories
export ROCPROFSYS_CONFIG_FILE=~/.rocprof-sys.cfg
```

Select the format with `-F txt|json|xml`; they load differently:

- `txt` (default) and `xml`: load with `ROCPROFSYS_CONFIG_FILE`.
- `json`: this is a hierarchical preset file. It does **not** load through
  `ROCPROFSYS_CONFIG_FILE` (the run fails with "missing the expected
  'rocprofiler-systems' root object"); load it with
  `rocprof-sys-run --preset=<file>.json -- <app>`.

An explicit environment variable always overrides the same setting in a config
file (e.g. `ROCPROFSYS_SAMPLING_FREQ=77` beats `ROCPROFSYS_SAMPLING_FREQ = 50`
in the file).

## Using what you discover

Names found with the commands above are applied through these settings:

- **CPU counters** (`-H -c CPU`): set `ROCPROFSYS_PAPI_EVENTS` together with
  `ROCPROFSYS_PROFILE=ON`; results are written to `papi_array*.txt/json`. All
  events must come from one namespace — mixing `PAPI_*` presets with `perf::*`
  events only prints a "Failure to add named event" warning and drops the
  counters, the run still exits 0. Reading most counters needs
  `/proc/sys/kernel/perf_event_paranoid` <= 2.
- **CPU overflow sampling**: sample on a hardware event instead of a timer.
  Set `ROCPROFSYS_USE_SAMPLING=ON`, `ROCPROFSYS_SAMPLING_OVERFLOW=ON`,
  `ROCPROFSYS_SAMPLING_OVERFLOW_EVENT=<event>` (list events with
  `rocprof-sys-avail -H -c CPU -r overflow`) **and** a large
  `ROCPROFSYS_SAMPLING_OVERFLOW_FREQ` (events per sample, e.g. `10000000`).
  Setting only the event does nothing (it falls back to CPU-time sampling), and
  leaving the interval at its default of 300 events aborted the run when tested.
- **GPU counters** (`-H -c GPU`): names are listed per device, e.g.
  `FETCH_SIZE:device=0`. Set `ROCPROFSYS_ROCM_EVENTS` for kernel-dispatch
  counters (writes `rocprof-device-<N>-<counter>.txt/json`), or
  `ROCPROFSYS_GPU_PERF_COUNTERS` for polled (PMC) sampling. A name without the
  `:device=N` suffix is collected on every device.
- **GPU metrics** (power, temperature, utilization): collected by default via
  AMD SMI. `ROCPROFSYS_AMD_SMI_METRICS` selects which (default
  `busy,temp,power,mem_usage`; `none` collects nothing) and
  `ROCPROFSYS_USE_AMD_SMI=OFF` disables them entirely.
- **ROCm API tracing**: set `ROCPROFSYS_ROCM_DOMAINS` to a comma-separated list,
  e.g. `hip_runtime_api,kernel_dispatch,memory_copy`. Copy names exactly from
  `rocprof-sys-avail -bd -r ROCM_DOMAINS`: one unknown name silently disables
  the whole list (no warning, exit 0, no kernel dispatches recorded when tested).
  `kfd_events` also needs `HSA_XNACK=1` and an XNACK-capable GPU; check with
  `rocminfo | grep -i xnack` (a `xnack-`/`xnack+` suffix on the GPU target means
  capable). On a GPU without XNACK it records nothing and prints nothing.

## Common Mistakes

| Mistake | Fix |
| --- | --- |
| Guessing a category name for `-c` (e.g. `settings::tracing`) | Run `rocprof-sys-avail --list-categories` first — category names don't always match the obvious English word (it's `settings::trace`) |
| Dumping the full `-S`/`-C`/`-H` table for a narrow question | Combine with `-r <regex>` or `-c <category>`/`-A` to filter before reporting |
| Confusing this skill with picking a profiling preset | Preset selection is `choosing-a-profiling-preset`; this skill is for exploring/fine-tuning individual options, including on top of a chosen preset |
| Typing a ROCm domain name from memory | An unknown `ROCPROFSYS_ROCM_DOMAINS` name silently disables the whole list; copy names from `rocprof-sys-avail -bd -r ROCM_DOMAINS` |
| Forgetting `-A`/`--available` | Without it, `-H`/`-C` list counters/components the current build or hardware doesn't actually support, which can overwhelm the answer |

`rocprof-sys-avail` is the source of truth for names and descriptions. For
topics beyond option discovery (unified memory profiling, causal profiling), see
the online docs:
<https://rocm.docs.amd.com/projects/rocprofiler-systems/en/develop/how-to/configuring-runtime-options.html>
