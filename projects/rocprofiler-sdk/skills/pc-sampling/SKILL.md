---
name: pc-sampling
description: Profiles a target application using rocprofv3 with program counter sampling enabled and then analyzes the results. Use when the user has a program that runs on an AMD GPU and asks to perform PC sampling, to determine the runtime performance characteristics of their application, to determine stall reasons, or to determine hotspots in the code.
---

# PC Sampling

## Prerequisites

- The system must have a ROCm installation and an accessible AMD GPU; that is, `amd-smi` should return at least one valid device.
- The GPU must support PC sampling. Available PC sampling configurations can be determined via `rocprofv3-avail info --pc-sampling` or `rocprofv3 -L`. Your sampling method and interval must align with the outputs of this command. Also pay attention to any flags reported for a given configuration; at time of writing, stochastic sampling intervals must be powers of two.
- The application should be built with debug symbols AND optimization flags (e.g. `-g -O3`) in order to get source line attribution and a representative execution profile.

## How to use PC Sampling

To run rocprofv3 with PC sampling:

- explicitly enable pc sampling with the flag `--pc-sampling-beta-enabled`.
- specify the type of pc sampling method with `--pc-sampling-method`: `host_trap` or `stochastic`
- specify the interval unit for the sampling with `--pc-sampling-unit`: `time` for host_trap and `cycles` for stochastic
- specify the sampling interval with `--pc-sampling-interval`: an integer representing microseconds if you specified `time` or cycles if you specified `cycles`.
- add the `--kernel-trace` flag to track kernel launches for attribution
- specify either the `json` or `csv` output format with `--output-format`; the default output format, `rocpd`, does not currently support PC sampling output.

A typical command will look like the following:

```bash
rocprofv3 \
  --pc-sampling-beta-enabled \
  --pc-sampling-method stochastic \
  --pc-sampling-unit cycles \
  --pc-sampling-interval 1048576 \
  --kernel-trace \
  --output-format json \
  -- <target app>
```

### PC Sampling Outputs

Output-related options:

- `-d <dir>`: output location (default: `%cwd%`)
- `-o <prefix>`: name prefix (default: `%hostname%/%pid%`)
- `--output-format csv,json`: emit both formats

Default path pattern: `<output_dir>/<output_prefix>_<name>.<ext>` where prefix is `%hostname%/%pid%` unless overridden.

#### Possible output files

| File | Format | Contents |
|-----|-----|-----|
| `*_pc_sampling_stochastic.csv` or `*_pc_sampling_host_trap.csv` | CSV | Flat sample records; one row per sample |
| `*_agent_info.csv` | CSV | GPU/CPU agent metadata (**not** samples) |
| `*_kernel_trace.csv` | CSV | Kernel dispatch records (with `--kernel-trace`) |
| `*_results.json` | JSON | All data in one file |

### PC Sampling Guidance

There are two types of PC sampling: host-trap and stochastic. Stochastic requires hardware support and is available on fewer GPUs, but provides more information.

The default output format, rocpd, does not currently support pc sampling, so you must explicitly specify json (preferred) or csv. json outputs contain more information.

- Prefer stochastic sampling unless the sampling interval needs to be larger than what stochastic can support.
- Prefer the json output format since it will carry richer information when using stochastic sampling.
- Prefer larger stochastic sampling intervals (2^18^-2^20^; often the maximum) unless the workload is short enough that this does not produce sufficient samples. Shorter intervals will quickly produce huge amounts of data.
- Prefer host trap intervals on the order of 1000 microseconds, decreasing if needed.

## Analyzing PC Sampling Results

Inside this skill's folder is an analysis script: scripts/analyze_pc_sampling.py. When passed the output sample file (.json or .csv) it will report the top sample locations and top stall reasons. You may also filter results to a particular kernel -with `--kernel <kernel name>`.

Run the included python script to perform basic analysis of the output files. Use its output to analyze the user's code, explain its performance, and help give specific and relevant suggestions for improvements. Inform the user of hot spots, the causes of the hotspots (if available), and possible improvements if any. If there are no notable potential improvements, briefly explain why. For example, transpose operations are heavily memory bound and a profile of a well-written transpose kernel will exhibit large amounts of WAITCNT stalls. This is inherent from the task itself and is not *necessarily* a failing of the kernel. However, a naive transpose kernel may exhibit scattered reads or writes, causing far more memory traffic and thus WAITCNT stalls than is necessary.

## Procedure Checklist

1. ensure the prerequisites defined above are met
2. if there is not an up-to-date build of the target application with debug information, build it
3. profile the application with rocprofv3 using pc sampling settings as directed above
4. run this skill's included analysis script on the profiling outputs
5. check that the run was successful and produced a non-trivial number of samples
6. use the output of the script to provide analysis of their code's behavior and specific guidance on possible improvements

## Misc. Tips

Tiny workloads may produce too few samples. Ensure the workload is large enough to get a representative sample count. If source code is available and the workload is too small, consider modifying the code to invoke the kernel multiple times, and then restoring it to its original state after the profiling run.

Host-trap sampling can have an instruction "skid" of one or two instructions; if sample attribution does not make sense, look at the surrounding instructions and branch targets.

A brief overview of stall types and their causes is available in this skill's resources/stall-reasons.md.

The rocm-systems repository has several PC sampling documentation files in this folder: projects/rocprofiler-sdk/source/docs/how-to/. You may also access them here: https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/develop/how-to/cdna3-cdna4-pc-sampling.html

Only one PC sampling configurations is supported at a time, so if exactly one valid configuration is reported then PC sampling is probably being actively used by another process.
