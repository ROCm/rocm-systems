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
`rocprof-sys-run`/`rocprof-sys-sample`.

## Discovery command reference

| User question | Command | Notes |
| --- | --- | --- |
| What settings/env vars exist, with descriptions? | `rocprof-sys-avail -S -bd` | `-b` suppresses the current-value/availability column, `-d` adds the description column |
| What hardware counters are available on my system? | `rocprof-sys-avail -H -bd -A` | `-A`/`--available` restricts to counters this machine can actually use; add `-c CPU` or `-c GPU` to split by device |
| What ROCm APIs can be traced? | `rocprof-sys-avail -bd -r ROCM_DOMAINS` | Regex-filters settings to the `ROCPROFSYS_ROCM_DOMAINS` row, listing every valid domain (`hip_runtime_api`, `hsa_api`, `kernel_dispatch`, `rccl_api`, `kfd_events`, ...) |
| What operations/API calls exist within one ROCm domain? | `rocprof-sys-avail --list-domains` then `rocprof-sys-avail --list-operations <domain>` | e.g. `--list-operations hip_runtime_api` lists individual HIP calls |
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

Supported formats: plain text (default), JSON, and XML — select with
`-F txt|json|xml`. An explicit environment variable always overrides the same
setting defined in the config file.

## Common Mistakes

| Mistake | Fix |
| --- | --- |
| Guessing a category name for `-c` (e.g. `settings::tracing`) | Run `rocprof-sys-avail --list-categories` first — category names don't always match the obvious English word (it's `settings::trace`) |
| Dumping the full `-S`/`-C`/`-H` table for a narrow question | Combine with `-r <regex>` or `-c <category>`/`-A` to filter before reporting |
| Confusing this skill with picking a profiling preset | Preset selection is `choosing-a-profiling-preset`; this skill is for exploring/fine-tuning individual options, including on top of a chosen preset |
| Forgetting `-A`/`--available` | Without it, `-H`/`-C` list counters/components the current build or hardware doesn't actually support, which can overwhelm the answer |

For anything deeper than option discovery (PAPI `perf_event_paranoid` setup,
KFD/XNACK prerequisites for `kfd_events`, unified memory profiling details,
causal profiling), see `docs/how-to/configuring-runtime-options.rst` and
`docs/how-to/configuring-validating-environment.rst`.
