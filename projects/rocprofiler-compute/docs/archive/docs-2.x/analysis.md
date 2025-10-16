# Analyze Mode

```eval_rst
.. toctree::
   :glob:
   :maxdepth: 4
```
Omniperf offers several ways to interact with the metrics it generates from profiling. The option you choose will likely be influenced by your familiarity with the profiled application, computing environment, and experience with Omniperf.

While analyzing with the CLI offers quick and straightforward access to Omniperf metrics from terminal, the GUI adds an extra layer of styling and interactiveness some users may prefer.

See sections below for more information on each.

```{note}
Profiling results from the [aforementioned vcopy workload](profiling.md#workload-compilation) will be used in the following sections to demonstrate the use of Omniperf in MI GPU performance analysis. Unless otherwise noted, the performance analysis is done on the MI200 platform.
```

## CLI Analysis

### Features

- __Derived metrics__: All of Omniperf's built-in metrics.
- __Baseline comparison__: Compare multiple runs in a side-by-side manner.
- __Metric customization__: Isolate a subset of built-in metrics or build your own profiling configuration.
- __Filtering__: Hone in on a particular kernel, gpu-id, and/or dispatch-id via post-process filtering.

Run `omniperf analyze -h` for more details.

### Demo

1) To begin, generate a high-level analysis report utilizing Omniperf's `-b` (a.k.a. `--block`) flag.
```shell-session
$ omniperf analyze -p workloads/vcopy/MI200/ -b 2


  ___                  _                  __
 / _ \ _ __ ___  _ __ (_)_ __   ___ _ __ / _|
| | | | '_ ` _ \| '_ \| | '_ \ / _ \ '__| |_
| |_| | | | | | | | | | | |_) |  __/ |  |  _|
 \___/|_| |_| |_|_| |_|_| .__/ \___|_|  |_|
                        |_|

Analysis mode = cli
[analysis] deriving Omniperf metrics...

--------------------------------------------------------------------------------
0. Top Stats
0.1 Top Kernels
╒════╤══════════════════════════════════════════╤═════════╤═══════════╤════════════╤══════════════╤════════╕
│    │ Kernel_Name                              │   Count │   Sum(ns) │   Mean(ns) │   Median(ns) │    Pct │
╞════╪══════════════════════════════════════════╪═════════╪═══════════╪════════════╪══════════════╪════════╡
│  0 │ vecCopy(double*, double*, double*, int,  │    1.00 │  20160.00 │   20160.00 │     20160.00 │ 100.00 │
│    │ int) [clone .kd]                         │         │           │            │              │        │
╘════╧══════════════════════════════════════════╧═════════╧═══════════╧════════════╧══════════════╧════════╛
0.2 Dispatch List
╒════╤═══════════════╤══════════════════════════════════════════════════════════╤══════════╕
│    │   Dispatch_ID │ Kernel_Name                                              │   GPU_ID │
╞════╪═══════════════╪══════════════════════════════════════════════════════════╪══════════╡
│  0 │             0 │ vecCopy(double*, double*, double*, int, int) [clone .kd] │        0 │
╘════╧═══════════════╧══════════════════════════════════════════════════════════╧══════════╛


--------------------------------------------------------------------------------
2. System Speed-of-Light
2.1 Speed-of-Light
╒═════════════╤═══════════════════════════╤═════════╤══════════════════╤══════════╤═══════════════╕
│ Metric_ID   │ Metric                    │ Avg     │ Unit             │ Peak     │ Pct of Peak   │
╞═════════════╪═══════════════════════════╪═════════╪══════════════════╪══════════╪═══════════════╡
│ 2.1.0       │ VALU FLOPs                │ 0.0     │ Gflop            │ 22630.4  │ 0.0           │
├─────────────┼───────────────────────────┼─────────┼──────────────────┼──────────┼───────────────┤
│ 2.1.1       │ VALU IOPs                 │ 364.09  │ Giop             │ 22630.4  │ 1.61          │
├─────────────┼───────────────────────────┼─────────┼──────────────────┼──────────┼───────────────┤
│ 2.1.2       │ MFMA FLOPs (BF16)         │ 0.0     │ Gflop            │ 181043.2 │ 0.0           │
├─────────────┼───────────────────────────┼─────────┼──────────────────┼──────────┼───────────────┤
│ 2.1.3       │ MFMA FLOPs (F16)          │ 0.0     │ Gflop            │ 181043.2 │ 0.0           │
├─────────────┼───────────────────────────┼─────────┼──────────────────┼──────────┼───────────────┤
│ 2.1.4       │ MFMA FLOPs (F32)          │ 0.0     │ Gflop            │ 45260.8  │ 0.0           │
├─────────────┼───────────────────────────┼─────────┼──────────────────┼──────────┼───────────────┤
│ 2.1.5       │ MFMA FLOPs (F64)          │ 0.0     │ Gflop            │ 45260.8  │ 0.0           │
├─────────────┼───────────────────────────┼─────────┼──────────────────┼──────────┼───────────────┤
│ 2.1.6       │ MFMA IOPs (Int8)          │ 0.0     │ Giop             │ 181043.2 │ 0.0           │
├─────────────┼───────────────────────────┼─────────┼──────────────────┼──────────┼───────────────┤
│ 2.1.7       │ Active CUs                │ 70.0    │ Cus              │ 104.0    │ 67.31         │
├─────────────┼───────────────────────────┼─────────┼──────────────────┼──────────┼───────────────┤
│ 2.1.8       │ SALU Utilization          │ 3.78    │ Pct              │ 100.0    │ 3.78          │
├─────────────┼───────────────────────────┼─────────┼──────────────────┼──────────┼───────────────┤
│ 2.1.9       │ VALU Utilization          │ 5.4     │ Pct              │ 100.0    │ 5.4           │
├─────────────┼───────────────────────────┼─────────┼──────────────────┼──────────┼───────────────┤
│ 2.1.10      │ MFMA Utilization          │ 0.0     │ Pct              │ 100.0    │ 0.0           │
├─────────────┼───────────────────────────┼─────────┼──────────────────┼──────────┼───────────────┤
│ 2.1.11      │ VMEM Utilization          │ 1.08    │ Pct              │ 100.0    │ 1.08          │
├─────────────┼───────────────────────────┼─────────┼──────────────────┼──────────┼───────────────┤
│ 2.1.12      │ Branch Utilization        │ 1.08    │ Pct              │ 100.0    │ 1.08          │
├─────────────┼───────────────────────────┼─────────┼──────────────────┼──────────┼───────────────┤
│ 2.1.13      │ VALU Active Threads       │ 64.0    │ Threads          │ 64.0     │ 100.0         │
├─────────────┼───────────────────────────┼─────────┼──────────────────┼──────────┼───────────────┤
│ 2.1.14      │ IPC                       │ 0.21    │ Instr/cycle      │ 5.0      │ 4.13          │
├─────────────┼───────────────────────────┼─────────┼──────────────────┼──────────┼───────────────┤
│ 2.1.15      │ Wavefront Occupancy       │ 2488.86 │ Wavefronts       │ 3328.0   │ 74.79         │
├─────────────┼───────────────────────────┼─────────┼──────────────────┼──────────┼───────────────┤
│ 2.1.16      │ Theoretical LDS Bandwidth │ 0.0     │ Gb/s             │ 22630.4  │ 0.0           │
├─────────────┼───────────────────────────┼─────────┼──────────────────┼──────────┼───────────────┤
│ 2.1.17      │ LDS Bank Conflicts/Access │         │ Conflicts/access │ 32.0     │               │
├─────────────┼───────────────────────────┼─────────┼──────────────────┼──────────┼───────────────┤
│ 2.1.18      │ vL1D Cache Hit Rate       │ 50.0    │ Pct              │ 100.0    │ 50.0          │
├─────────────┼───────────────────────────┼─────────┼──────────────────┼──────────┼───────────────┤
│ 2.1.19      │ vL1D Cache BW             │ 1664.41 │ Gb/s             │ 11315.2  │ 14.71         │
├─────────────┼───────────────────────────┼─────────┼──────────────────┼──────────┼───────────────┤
│ 2.1.20      │ L2 Cache Hit Rate         │ 35.74   │ Pct              │ 100.0    │ 35.74         │
├─────────────┼───────────────────────────┼─────────┼──────────────────┼──────────┼───────────────┤
│ 2.1.21      │ L2 Cache BW               │ 1296.31 │ Gb/s             │ 3481.6   │ 37.23         │
├─────────────┼───────────────────────────┼─────────┼──────────────────┼──────────┼───────────────┤
│ 2.1.22      │ L2-Fabric Read BW         │ 416.52  │ Gb/s             │ 1638.4   │ 25.42         │
├─────────────┼───────────────────────────┼─────────┼──────────────────┼──────────┼───────────────┤
│ 2.1.23      │ L2-Fabric Write BW        │ 292.3   │ Gb/s             │ 1638.4   │ 17.84         │
├─────────────┼───────────────────────────┼─────────┼──────────────────┼──────────┼───────────────┤
│ 2.1.24      │ L2-Fabric Read Latency    │ 262.85  │ Cycles           │          │               │
├─────────────┼───────────────────────────┼─────────┼──────────────────┼──────────┼───────────────┤
│ 2.1.25      │ L2-Fabric Write Latency   │ 307.4   │ Cycles           │          │               │
├─────────────┼───────────────────────────┼─────────┼──────────────────┼──────────┼───────────────┤
│ 2.1.26      │ sL1D Cache Hit Rate       │ 99.82   │ Pct              │ 100.0    │ 99.82         │
├─────────────┼───────────────────────────┼─────────┼──────────────────┼──────────┼───────────────┤
│ 2.1.27      │ sL1D Cache BW             │ 208.05  │ Gb/s             │ 6092.8   │ 3.41          │
├─────────────┼───────────────────────────┼─────────┼──────────────────┼──────────┼───────────────┤
│ 2.1.28      │ L1I Hit Rate              │ 99.91   │ Pct              │ 100.0    │ 99.91         │
├─────────────┼───────────────────────────┼─────────┼──────────────────┼──────────┼───────────────┤
│ 2.1.29      │ L1I BW                    │ 208.05  │ Gb/s             │ 6092.8   │ 3.41          │
├─────────────┼───────────────────────────┼─────────┼──────────────────┼──────────┼───────────────┤
│ 2.1.30      │ L1I Fetch Latency         │ 20.86   │ Cycles           │          │               │
╘═════════════╧═══════════════════════════╧═════════╧══════════════════╧══════════╧═══════════════╛

...
```
2. Use `--list-metrics` to generate a list of available metrics for inspection
```shell-session
$ omniperf analyze -p workloads/vcopy/MI200/ --list-metrics gfx90a

  ___                  _                  __
/ _ \ _ __ ___  _ __ (_)_ __   ___ _ __ / _|
| | | | '_ ` _ \| '_ \| | '_ \ / _ \ '__| |_
| |_| | | | | | | | | | | |_) |  __/ |  |  _|
\___/|_| |_| |_|_| |_|_| .__/ \___|_|  |_|
                        |_|

Analysis mode = cli
[analysis] deriving Omniperf metrics...
0 -> Top Stats
1 -> System Info
2 -> System Speed-of-Light
        2.1 -> Speed-of-Light
                2.1.0 -> VALU FLOPs
                2.1.1 -> VALU IOPs
                2.1.2 -> MFMA FLOPs (BF16)
                2.1.3 -> MFMA FLOPs (F16)
                2.1.4 -> MFMA FLOPs (F32)
                2.1.5 -> MFMA FLOPs (F64)
                2.1.6 -> MFMA IOPs (Int8)
                2.1.7 -> Active CUs
                2.1.8 -> SALU Utilization
                2.1.9 -> VALU Utilization
                2.1.10 -> MFMA Utilization
                2.1.11 -> VMEM Utilization
                2.1.12 -> Branch Utilization
                2.1.13 -> VALU Active Threads
                2.1.14 -> IPC
                2.1.15 -> Wavefront Occupancy
                2.1.16 -> Theoretical LDS Bandwidth
                2.1.17 -> LDS Bank Conflicts/Access
                2.1.18 -> vL1D Cache Hit Rate
                2.1.19 -> vL1D Cache BW
                2.1.20 -> L2 Cache Hit Rate
                2.1.21 -> L2 Cache BW
                2.1.22 -> L2-Fabric Read BW
                2.1.23 -> L2-Fabric Write BW
                2.1.24 -> L2-Fabric Read Latency
                2.1.25 -> L2-Fabric Write Latency
                2.1.26 -> sL1D Cache Hit Rate
                2.1.27 -> sL1D Cache BW
                2.1.28 -> L1I Hit Rate
                2.1.29 -> L1I BW
                2.1.30 -> L1I Fetch Latency
...
```
3. Choose your own customized subset of metrics with `-b` (a.k.a. `--block`), or build your own config following [config_template](https://github.com/ROCm/omniperf/blob/amd-mainline/src/rocprof_compute_analyze/configs/panel_config_template.yaml). Below shows how to generate a report containing only metric 2 (a.k.a. System Speed-of-Light).
```shell-session
$ omniperf analyze -p workloads/vcopy/MI200/ -b 2
--------
Analyze
--------

--------------------------------------------------------------------------------
0. Top Stat
╒════╤══════════════════════════════════════════╤═════════╤═══════════╤════════════╤══════════════╤════════╕
│    │ KernelName                               │   Count │   Sum(ns) │   Mean(ns) │   Median(ns) │    Pct │
╞════╪══════════════════════════════════════════╪═════════╪═══════════╪════════════╪══════════════╪════════╡
│  0 │ vecCopy(double*, double*, double*, int,  │       1 │  20000.00 │   20000.00 │     20000.00 │ 100.00 │
│    │ int) [clone .kd]                         │         │           │            │              │        │
╘════╧══════════════════════════════════════════╧═════════╧═══════════╧════════════╧══════════════╧════════╛


--------------------------------------------------------------------------------
2. System Speed-of-Light
╒═════════╤═══════════════════════════╤═══════════════════════╤══════════════════╤════════════════════╤════════════════════════╕
│ Index   │ Metric                    │ Value                 │ Unit             │ Peak               │ PoP                    │
╞═════════╪═══════════════════════════╪═══════════════════════╪══════════════════╪════════════════════╪════════════════════════╡
│ 2.1.0   │ VALU FLOPs                │ 0.0                   │ Gflop            │ 22630.4            │ 0.0                    │
├─────────┼───────────────────────────┼───────────────────────┼──────────────────┼────────────────────┼────────────────────────┤
│ 2.1.1   │ VALU IOPs                 │ 367.0016              │ Giop             │ 22630.4            │ 1.6217194570135745     │
├─────────┼───────────────────────────┼───────────────────────┼──────────────────┼────────────────────┼────────────────────────┤
│ 2.1.2   │ MFMA FLOPs (BF16)         │ 0.0                   │ Gflop            │ 90521.6            │ 0.0                    │
├─────────┼───────────────────────────┼───────────────────────┼──────────────────┼────────────────────┼────────────────────────┤
│ 2.1.3   │ MFMA FLOPs (F16)          │ 0.0                   │ Gflop            │ 181043.2           │ 0.0                    │
├─────────┼───────────────────────────┼───────────────────────┼──────────────────┼────────────────────┼────────────────────────┤
│ 2.1.4   │ MFMA FLOPs (F32)          │ 0.0                   │ Gflop            │ 45260.8            │ 0.0                    │
├─────────┼───────────────────────────┼───────────────────────┼──────────────────┼────────────────────┼────────────────────────┤
│ 2.1.5   │ MFMA FLOPs (F64)          │ 0.0                   │ Gflop            │ 45260.8            │ 0.0                    │
├─────────┼───────────────────────────┼───────────────────────┼──────────────────┼────────────────────┼────────────────────────┤
│ 2.1.6   │ MFMA IOPs (Int8)          │ 0.0                   │ Giop             │ 181043.2           │ 0.0                    │
├─────────┼───────────────────────────┼───────────────────────┼──────────────────┼────────────────────┼────────────────────────┤
│ 2.1.7   │ Active CUs                │ 74                    │ Cus              │ 104                │ 71.15384615384616      │
├─────────┼───────────────────────────┼───────────────────────┼──────────────────┼────────────────────┼────────────────────────┤
│ 2.1.8   │ SALU Util                 │ 4.016057506716307     │ Pct              │ 100                │ 4.016057506716307      │
├─────────┼───────────────────────────┼───────────────────────┼──────────────────┼────────────────────┼────────────────────────┤
│ 2.1.9   │ VALU Util                 │ 5.737225009594725     │ Pct              │ 100                │ 5.737225009594725      │
├─────────┼───────────────────────────┼───────────────────────┼──────────────────┼────────────────────┼────────────────────────┤
│ 2.1.10  │ MFMA Util                 │ 0.0                   │ Pct              │ 100                │ 0.0                    │
├─────────┼───────────────────────────┼───────────────────────┼──────────────────┼────────────────────┼────────────────────────┤
│ 2.1.11  │ VALU Active Threads/Wave  │ 64.0                  │ Threads          │ 64                 │ 100.0                  │
├─────────┼───────────────────────────┼───────────────────────┼──────────────────┼────────────────────┼────────────────────────┤
│ 2.1.12  │ IPC - Issue               │ 1.0                   │ Instr/cycle      │ 5                  │ 20.0                   │
├─────────┼───────────────────────────┼───────────────────────┼──────────────────┼────────────────────┼────────────────────────┤
│ 2.1.13  │ LDS BW                    │ 0.0                   │ Gb/sec           │ 22630.4            │ 0.0                    │
├─────────┼───────────────────────────┼───────────────────────┼──────────────────┼────────────────────┼────────────────────────┤
│ 2.1.14  │ LDS Bank Conflict         │                       │ Conflicts/access │ 32                 │                        │
├─────────┼───────────────────────────┼───────────────────────┼──────────────────┼────────────────────┼────────────────────────┤
│ 2.1.15  │ Instr Cache Hit Rate      │ 99.91306912556854     │ Pct              │ 100                │ 99.91306912556854      │
├─────────┼───────────────────────────┼───────────────────────┼──────────────────┼────────────────────┼────────────────────────┤
│ 2.1.16  │ Instr Cache BW            │ 209.7152              │ Gb/s             │ 6092.8             │ 3.442016806722689      │
├─────────┼───────────────────────────┼───────────────────────┼──────────────────┼────────────────────┼────────────────────────┤
│ 2.1.17  │ Scalar L1D Cache Hit Rate │ 99.81986908342313     │ Pct              │ 100                │ 99.81986908342313      │
├─────────┼───────────────────────────┼───────────────────────┼──────────────────┼────────────────────┼────────────────────────┤
│ 2.1.18  │ Scalar L1D Cache BW       │ 209.7152              │ Gb/s             │ 6092.8             │ 3.442016806722689      │
├─────────┼───────────────────────────┼───────────────────────┼──────────────────┼────────────────────┼────────────────────────┤
│ 2.1.19  │ Vector L1D Cache Hit Rate │ 50.0                  │ Pct              │ 100                │ 50.0                   │
├─────────┼───────────────────────────┼───────────────────────┼──────────────────┼────────────────────┼────────────────────────┤
│ 2.1.20  │ Vector L1D Cache BW       │ 1677.7216             │ Gb/s             │ 11315.199999999999 │ 14.82714932126697      │
├─────────┼───────────────────────────┼───────────────────────┼──────────────────┼────────────────────┼────────────────────────┤
│ 2.1.21  │ L2 Cache Hit Rate         │ 35.55067615693325     │ Pct              │ 100                │ 35.55067615693325      │
├─────────┼───────────────────────────┼───────────────────────┼──────────────────┼────────────────────┼────────────────────────┤
│ 2.1.22  │ L2-Fabric Read BW         │ 419.8496              │ Gb/s             │ 1638.4             │ 25.6255859375          │
├─────────┼───────────────────────────┼───────────────────────┼──────────────────┼────────────────────┼────────────────────────┤
│ 2.1.23  │ L2-Fabric Write BW        │ 293.9456              │ Gb/s             │ 1638.4             │ 17.941015625           │
├─────────┼───────────────────────────┼───────────────────────┼──────────────────┼────────────────────┼────────────────────────┤
│ 2.1.24  │ L2-Fabric Read Latency    │ 256.6482321288385     │ Cycles           │                    │                        │
├─────────┼───────────────────────────┼───────────────────────┼──────────────────┼────────────────────┼────────────────────────┤
│ 2.1.25  │ L2-Fabric Write Latency   │ 317.2264255699014     │ Cycles           │                    │                        │
├─────────┼───────────────────────────┼───────────────────────┼──────────────────┼────────────────────┼────────────────────────┤
│ 2.1.26  │ Wave Occupancy            │ 1821.723057333852     │ Wavefronts       │ 3328               │ 54.73927455931046      │
├─────────┼───────────────────────────┼───────────────────────┼──────────────────┼────────────────────┼────────────────────────┤
│ 2.1.27  │ Instr Fetch BW            │ 4.174722306564298e-08 │ Gb/s             │ 3046.4             │ 1.3703789084047721e-09 │
├─────────┼───────────────────────────┼───────────────────────┼──────────────────┼────────────────────┼────────────────────────┤
│ 2.1.28  │ Instr Fetch Latency       │ 21.729248046875       │ Cycles           │                    │                        │
╘═════════╧═══════════════════════════╧═══════════════════════╧══════════════════╧════════════════════╧════════════════════════╛
```

```{note}
Some cells may be blank indicating a missing/unavailable hardware counter or NULL value
```

3. Optimize application, iterate, and re-profile to inspect performance changes.
4. Redo a comprehensive analysis with Omniperf CLI at any milestone or at the end.

### More options

- __Single run__
  ```shell
  $ omniperf analyze -p workloads/vcopy/MI200/
  ```

- __List top kernels and dispatches__
  ```shell
  $ omniperf analyze -p workloads/vcopy/MI200/  --list-stats
  ```

- __List metrics__

  ```shell
  $ omniperf analyze -p workloads/vcopy/MI200/  --list-metrics gfx90a
  ```

- __Show "System Speed-of-Light" and "CS_Busy" blocks only__

  ```shell
  $ omniperf analyze -p workloads/vcopy/MI200/  -b 2  5.1.0
  ```

  ```{note}
  Users can filter single metric or the whole hardware component by its id. In this case, 1 is the id for "system speed of light" and 5.1.0 the id for metric "GPU Busy Cycles".
  ```

- __Filter kernels__

  First, list the top kernels in your application using `--list-stats`.
  ```shell-session
  $ omniperf analyze -p workloads/vcopy/MI200/ --list-stats

  Analysis mode = cli
  [analysis] deriving Omniperf metrics...

  --------------------------------------------------------------------------------
  Detected Kernels (sorted descending by duration)
  ╒════╤══════════════════════════════════════════════╕
  │    │ Kernel_Name                                  │
  ╞════╪══════════════════════════════════════════════╡
  │  0 │ vecCopy(double*, double*, double*, int, int) │
  ╘════╧══════════════════════════════════════════════╛

  --------------------------------------------------------------------------------
  Dispatch list
  ╒════╤═══════════════╤══════════════════════════════════════════════╤══════════╕
  │    │   Dispatch_ID │ Kernel_Name                                  │   GPU_ID │
  ╞════╪═══════════════╪══════════════════════════════════════════════╪══════════╡
  │  0 │             0 │ vecCopy(double*, double*, double*, int, int) │        0 │
  ╘════╧═══════════════╧══════════════════════════════════════════════╧══════════╛

  ```

  Second, select the index of the kernel you would like to filter (i.e. __vecCopy(double*, double*, double*, int, int) [clone .kd]__ at index __0__). Then, use this index to apply the filter via `-k/--kernels`.

  ```shell-session
  $ omniperf analyze -p workloads/vcopy/MI200/ -k 0

  Analysis mode = cli
  [analysis] deriving Omniperf metrics...

  --------------------------------------------------------------------------------
  0. Top Stats
  0.1 Top Kernels
  ╒════╤══════════════════════════════════════════╤═════════╤═══════════╤════════════╤══════════════╤════════╤═════╕
  │    │ Kernel_Name                              │   Count │   Sum(ns) │   Mean(ns) │   Median(ns) │    Pct │ S   │
  ╞════╪══════════════════════════════════════════╪═════════╪═══════════╪════════════╪══════════════╪════════╪═════╡
  │  0 │ vecCopy(double*, double*, double*, int,  │    1.00 │  18560.00 │   18560.00 │     18560.00 │ 100.00 │ *   │
  │    │ int)                                     │         │           │            │              │        │     │
  ╘════╧══════════════════════════════════════════╧═════════╧═══════════╧════════════╧══════════════╧════════╧═════╛
  ... ...
  ```

  ```{note}
  You will see your filtered kernel(s) indicated by an asterisk in the Top Stats table
  ```


- __Baseline comparison__

  ```shell
  omniperf analyze -p workload1/path/  -p workload2/path/
  ```
  OR
  ```shell
  omniperf analyze -p workload1/path/ -k 0  -p workload2/path/ -k 1
  ```


## GUI Analysis

### Web-based GUI

#### Features

Omniperf's standalone GUI analyzer is a lightweight web page that can
be generated directly from the command-line. This option is provided
as an alternative for users wanting to explore profiling results
graphically, but without the additional setup requirements or
server-side overhead of Omniperf's detailed [Grafana
interface](analysis.md#grafana-based-gui)
option.  The standalone GUI analyzer is provided as simple
[Flask](https://flask.palletsprojects.com/en/2.2.x/) application
allowing users to view results from within a web browser.

```{admonition} Port forwarding

Note that the standalone GUI analyzer publishes a web interface on port 8050 by default.
On production HPC systems where profiling jobs run
under the auspices of a resource manager, additional SSH tunneling
between the desired web browser host (e.g. login node or remote workstation) and compute host may be
required. Alternatively, users may find it more convenient to download
profiled workloads to perform analysis on their local system.

See [FAQ](faq.md) for more details on SSH tunneling.
```

#### Usage

To launch the standalone GUI, include the `--gui` flag with your desired analysis command. For example:

```shell-session
$ omniperf analyze -p workloads/vcopy/MI200/ --gui

  ___                  _                  __
 / _ \ _ __ ___  _ __ (_)_ __   ___ _ __ / _|
| | | | '_ ` _ \| '_ \| | '_ \ / _ \ '__| |_
| |_| | | | | | | | | | | |_) |  __/ |  |  _|
 \___/|_| |_| |_|_| |_|_| .__/ \___|_|  |_|
                        |_|

Analysis mode = web_ui
[analysis] deriving Omniperf metrics...
Dash is running on http://0.0.0.0:8050/

 * Serving Flask app 'rocprof_compute_analyze.analysis_webui' (lazy loading)
 * Environment: production
   WARNING: This is a development server. Do not use it in a production deployment.
   Use a production WSGI server instead.
 * Debug mode: off
 * Running on all addresses (0.0.0.0)
   WARNING: This is a development server. Do not use it in a production deployment.
 * Running on http://127.0.0.1:8050
 * Running on http://10.228.33.172:8050 (Press CTRL+C to quit)
```

At this point, users can then launch their web browser of choice and
go to http://localhost:8050/ to see an analysis page.



![Standalone GUI Homepage](images/standalone_gui.png)

```{tip}
To launch the web application on a port other than 8050, include an optional port argument:
`--gui <desired port>`
```

When no filters are applied, users will see five basic sections derived from their application's profiling data:

1. Memory Chart Analysis
2. Empirical Roofline Analysis
3. Top Stats (Top Kernel Statistics)
4. System Info
5. System Speed-of-Light

To dive deeper, use the top drop down menus to isolate particular
kernel(s) or dispatch(s). You will then see the web page update with
metrics specific to the filter you have applied.

Once you have applied a filter, you will also see several additional
sections become available with detailed metrics specific to that area
of AMD hardware. These detailed sections mirror the data displayed in
Omniperf's [Grafana
interface](analysis.md#grafana-based-gui).

### Grafana-based GUI

#### Features
The Omniperf Grafana GUI Analyzer supports the following features to facilitate MI GPU performance profiling and analysis:

- System and Hardware Component (Hardware Block) Speed-of-Light (SOL)
- Multiple normalization options, including per-cycle, per-wave, per-kernel and per-second.
- Baseline comparisons
- Regex based Dispatch ID filtering
- Roofline Analysis
- Detailed performance counters and metrics per hardware component, e.g.,
  - Command Processor - Fetch (CPF) / Command Processor - Controller (CPC)
  - Workgroup Manager (SPI)
  - Shader Sequencer (SQ)
  - Shader Sequencer Controller (SQC)
  - L1 Address Processing Unit, a.k.a. Texture Addresser (TA) / L1 Backend Data Processing Unit, a.k.a. Texture Data (TD)
  - L1 Cache (TCP)
  - L2 Cache (TCC) (both aggregated and per-channel perf info)

##### Speed-of-Light
Speed-of-light panels are provided at both the system and per hardware component level to help diagnosis performance bottlenecks. The performance numbers of the workload under testing are compared to the theoretical maximum, (e.g. floating point operations, bandwidth, cache hit rate, etc.), to indicate the available room to further utilize the hardware capability.

##### Multi Normalization

Multiple performance number normalizations are provided to allow performance inspection within both HW and SW context. The following normalizations are permitted:
- per cycle
- per wave
- per kernel
- per second

##### Baseline Comparison
Omniperf enables baseline comparison to allow checking A/B effect. Currently baseline comparison is limited to the same SoC. Cross comparison between SoCs is in development.

For both the Current Workload and the Baseline Workload, one can independently setup the following filters to allow fine grained comparisons:
- Workload Name
- GPU ID filtering (multi-selection)
- Kernel Name filtering (multi-selection)
- Dispatch ID filtering (Regex filtering)
- Omniperf Panels (multi-selection)

##### Regex based Dispatch ID filtering
Omniperf enables Regular Expression (regex), a standard Linux string matching syntax, based dispatch ID filtering to flexibly choose the kernel invocations. One may refer to [Regex Numeric Range Generator](https://3widgets.com/), to generate typical number ranges.

For example, if one wants to inspect Dispatch Range from 17 to 48, inclusive, the corresponding regex is : **(1[7-9]|[23]\d|4[0-8])**. The generated expression can be copied over for filtering.

##### Incremental Profiling
Omniperf supports incremental profiling to significantly speed up performance analysis.

> Refer to [*Hardware Component Filtering*](profiling.md#hardware-component-filtering) section for this command.

By default, the entire application is profiled to collect performance counters for all hardware blocks, giving a complete view of where the workload stands in terms of performance optimization opportunities and bottlenecks.

After that one may focus on only a few hardware components, (e.g., L1 Cache or LDS) to closely check the effect of software optimizations, without performing application replay for all other hardware components. This saves lots of compute time. In addition, the prior profiling results for other hardware components are not overwritten. Instead, they can be merged during the import to piece together the system view.

##### Color Coding
The uniform color coding is applied to most visualizations (bars, table, diagrams etc). Typically, Yellow color means over 50%, while Red color mean over 90% percent, for easy inspection.

##### Global Variables and Configurations

![Grafana GUI Global Variables](images/global_variables.png)

#### Grafana GUI Import
The omniperf database `--import` option imports the raw profiling data to Grafana's backend MongoDB database. This step is only required for Grafana GUI based performance analysis.

Default username and password for MongoDB (to be used in database mode) are as follows:

 - Username: **temp**
 - Password: **temp123**

Each workload is imported to a separate database with the following naming convention:

    omniperf_<team>_<database>_<soc>

e.g., omniperf_asw_vcopy_mi200.

When using database mode, be sure to tailor the connection options to the machine hosting your [sever-side instance](./installation.md). Below is the sample command to import the *vcopy* profiling data, lets assuming our host machine is called "dummybox".

```shell-session
$ omniperf database --help
usage:

omniperf database <interaction type> [connection options]



-------------------------------------------------------------------------------

Examples:

        omniperf database --import -H pavii1 -u temp -t asw -w workloads/vcopy/mi200/

        omniperf database --remove -H pavii1 -u temp -w omniperf_asw_sample_mi200

-------------------------------------------------------------------------------



Help:
  -h, --help         show this help message and exit

General Options:
  -v, --version      show program's version number and exit
  -V, --verbose      Increase output verbosity (use multiple times for higher levels)
  -s, --specs        Print system specs.

Interaction Type:
  -i, --import                                  Import workload to Omniperf DB
  -r, --remove                                  Remove a workload from Omniperf DB

Connection Options:
  -H , --host                                   Name or IP address of the server host.
  -P , --port                                   TCP/IP Port. (DEFAULT: 27018)
  -u , --username                               Username for authentication.
  -p , --password                               The user's password. (will be requested later if it's not set)
  -t , --team                                   Specify Team prefix.
  -w , --workload                               Specify name of workload (to remove) or path to workload (to import)
  --kernel-verbose              Specify Kernel Name verbose level 1-5. Lower the level, shorter the kernel name. (DEFAULT: 5) (DISABLE: 5)
```

**omniperf import for vcopy:**
```shell-session
$ omniperf database --import -H dummybox -u temp -t asw -w workloads/vcopy/mi200/

  ___                  _                  __
 / _ \ _ __ ___  _ __ (_)_ __   ___ _ __ / _|
| | | | '_ ` _ \| '_ \| | '_ \ / _ \ '__| |_
| |_| | | | | | | | | | | |_) |  __/ |  |  _|
 \___/|_| |_| |_|_| |_|_| .__/ \___|_|  |_|
                        |_|


Pulling data from  /home/auser/repos/omniperf/sample/workloads/vcopy/MI200
The directory exists
Found sysinfo file
KernelName shortening enabled
Kernel name verbose level: 2
Password:
Password received
-- Conversion & Upload in Progress --
  0%|                                                                                                                                                                                                             | 0/11 [00:00<?, ?it/s]/home/auser/repos/omniperf/sample/workloads/vcopy/MI200/SQ_IFETCH_LEVEL.csv
  9%|█████████████████▉                                                                                                                                                                                   | 1/11 [00:00<00:01,  8.53it/s]/home/auser/repos/omniperf/sample/workloads/vcopy/MI200/pmc_perf.csv
 18%|███████████████████████████████████▊                                                                                                                                                                 | 2/11 [00:00<00:01,  6.99it/s]/home/auser/repos/omniperf/sample/workloads/vcopy/MI200/SQ_INST_LEVEL_SMEM.csv
 27%|█████████████████████████████████████████████████████▋                                                                                                                                               | 3/11 [00:00<00:01,  7.90it/s]/home/auser/repos/omniperf/sample/workloads/vcopy/MI200/SQ_LEVEL_WAVES.csv
 36%|███████████████████████████████████████████████████████████████████████▋                                                                                                                             | 4/11 [00:00<00:00,  8.56it/s]/home/auser/repos/omniperf/sample/workloads/vcopy/MI200/SQ_INST_LEVEL_LDS.csv
 45%|█████████████████████████████████████████████████████████████████████████████████████████▌                                                                                                           | 5/11 [00:00<00:00,  9.00it/s]/home/auser/repos/omniperf/sample/workloads/vcopy/MI200/SQ_INST_LEVEL_VMEM.csv
 55%|███████████████████████████████████████████████████████████████████████████████████████████████████████████▍                                                                                         | 6/11 [00:00<00:00,  9.24it/s]/home/auser/repos/omniperf/sample/workloads/vcopy/MI200/sysinfo.csv
 64%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▎                                                                       | 7/11 [00:00<00:00,  9.37it/s]/home/auser/repos/omniperf/sample/workloads/vcopy/MI200/roofline.csv
 82%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▏                                   | 9/11 [00:00<00:00, 12.60it/s]/home/auser/repos/omniperf/sample/workloads/vcopy/MI200/timestamps.csv
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 11/11 [00:00<00:00, 11.05it/s]
9 collections added.
Workload name uploaded
-- Complete! --
```

#### Omniperf Panels

##### Overview

There are currently 18 main panel categories available for analyzing the compute workload performance. Each category contains several panels for close inspection of the system performance.

- Kernel Statistics
  - Kernel time histogram
  - Top Ten bottleneck kernels
- System Speed-of-Light
  - Speed-of-Light
  - System Info table
- Memory Chart Analysis
- Roofline Analysis
  - FP32/FP64
  - FP16/INT8
- Command Processor
  - Command Processor - Fetch (CPF)
  - Command Processor - Controller (CPC)
- Workgroup Manager or Shader Processor Input (SPI)
  - SPI Stats
  - SPI Resource Allocations
- Wavefront Launch
  - Wavefront Launch Stats
  - Wavefront runtime stats
  - per-SE Wavefront Scheduling performance
- Wavefront Lifetime
  - Wavefront lifetime breakdown
  - per-SE wavefront life (average)
  - per-SE wavefront life (histogram)
- Wavefront Occupancy
  - per-SE wavefront occupancy
  - per-CU wavefront occupancy
- Compute Unit - Instruction Mix
  - per-wave Instruction mix
  - per-wave VALU Arithmetic instruction mix
  - per-wave MFMA Arithmetic instruction mix
- Compute Unit - Compute Pipeline
  - Speed-of-Light: Compute Pipeline
  - Arithmetic OPs count
  - Compute pipeline stats
  - Memory latencies
- Local Data Share (LDS)
  - Speed-of-Light: LDS
  - LDS stats
- Instruction Cache
  - Speed-of-Light: Instruction Cache
  - Instruction Cache Accesses
- Constant Cache
  - Speed-of-Light: Constant Cache
  - Constant Cache Accesses
  - Constant Cache - L2 Interface stats
- Texture Address and Texture Data
  - Texture Address (TA)
  - Texture Data (TD)
- L1 Cache
  - Speed-of-Light: L1 Cache
  - L1 Cache Accesses
  - L1 Cache Stalls
  - L1 - L2 Transactions
  - L1 - UTCL1 Interface stats
- L2 Cache
  - Speed-of-Light: L2 Cache
  - L2 Cache Accesses
  - L2 - EA Transactions
  - L2 - EA Stalls
- L2 Cache Per Channel Performance
  - Per-channel L2 Hit rate
  - Per-channel L1-L2 Read requests
  - Per-channel L1-L2 Write Requests
  - Per-channel L1-L2 Atomic Requests
  - Per-channel L2-EA Read requests
  - Per-channel L2-EA Write requests
  - Per-channel L2-EA Atomic requests
  - Per-channel L2-EA Read latency
  - Per-channel L2-EA Write latency
  - Per-channel L2-EA Atomic latency
  - Per-channel L2-EA Read stall (I/O, GMI, HBM)
  - Per-channel L2-EA Write stall (I/O, GMI, HBM, Starve)

Most panels are designed around a specific hardware component block to thoroughly understand its behavior. Additional panels, including custom panels, could also be added to aid the performance analysis.

##### System Info Panel
``` {figure} images/system-info_panel.png
:alt: System Info
:figclass: figure
:align: center

System details logged from host machine.
```

##### Kernel Statistics

###### Kernel Time Histogram
``` {figure} images/Kernel_time_histogram.png
:alt: Kernel Time Histogram
:figclass: figure
:align: center

Mapping application kernel launches to execution duration.
```
###### Top Bottleneck Kernels
``` {figure} images/top-stat_panel.png
:alt: Top Bottleneck Kernels
:figclass: figure
:align: center

Top N kernels and relevant statistics. Sorted by total duration.
```
###### Top Bottleneck Dispatches
``` {figure} images/Top_bottleneck_dispatches.png
:alt: Top Bottleneck Dispatches
:figclass: figure
:align: center

Top N kernel dispatches and relevant statistics. Sorted by total duration.
```
###### Current and Baseline Dispatch IDs (Filtered)
``` {figure} images/Current_and_baseline_dispatch_ids.png
:alt: Current and Baseline Dispatch IDs
:figclass: figure
:align: center

List of all kernel dispatches.
```

##### System Speed-of-Light
``` {figure} images/sol_panel.png
:alt: System Speed-of-Light
:figclass: figure
:align: center

Key metrics from various sections of Omniperf’s profiling report.
```

##### Memory Chart Analysis
> Note: The Memory Chart Analysis support multiple normalizations. Due to the space limit, all transactions, when normalized to per-sec, default to unit of Billion transactions per second.

``` {figure} images/memory-chart_panel.png
:alt: Memory Chart Analysis
:figclass: figure
:align: center

A graphical representation of performance data for memory blocks on the GPU.
```

##### Empirical Roofline Analysis
``` {figure} images/roofline_panel.png
:alt: Roofline Analysis
:figclass: figure
:align: center

Visualize achieved performance relative to a benchmarked peak performance.
```

##### Command Processor
###### Command Processor Fetcher
``` {figure} images/cpc_panel.png
:alt: Command Processor Fetcher
:figclass: figure
:align: center

Fetches commands out of memory to hand them over to the Command Processor Fetcher (CPC) for processing
```
###### Command Processor Compute
``` {figure} images/cpf_panel.png
:alt: Command Processor Compute
:figclass: figure
:align: center

The micro-controller running the command processing firmware that decodes the fetched commands, and (for kernels) passes them to the Workgroup Managers (SPIs) for scheduling.
```

##### Shader Processor Input (SPI)
###### SPI Stats
``` {figure} images/spi-stats_panel.png
:alt: SPI Stats
:figclass: figure
:align: center

TODO: Add caption after merge
```
###### SPI Resource Allocation
``` {figure} images/spi-resource-allocation_panel.png
:alt: SPI Resource Allocation
:figclass: figure
:align: center

TODO: Add caption after merge
```

##### Wavefront
###### Wavefront Launch Stats
``` {figure} images/wavefront-launch-stats_panel.png
:alt: Wavefront Launch Stats
:figclass: figure
:align: center

General information about the kernel launch.
```
###### Wavefront Runtime Stats
``` {figure} images/wavefront-runtime-stats_panel.png
:alt: Wavefront Runtime Stats
:figclass: figure
:align: center

High-level overview of the execution of wavefronts in a kernel.
```

##### Compute Unit - Instruction Mix
###### Instruction Mix
``` {figure} images/cu-inst-mix_panel.png
:alt: Instruction Mix
:figclass: figure
:align: center

Breakdown of the various types of instructions executed by the user’s kernel, and which pipelines on the Compute Unit (CU) they were executed on.
```
###### VALU Arithmetic Instruction Mix
``` {figure} images/cu-value-arith-instr-mix_panel.png
:alt: VALU Arithmetic Instruction Mix
:figclass: figure
:align: center

The various types of vector instructions that were issued to the vector arithmetic logic unit (VALU).
```
###### MFMA Arithmetic Instruction Mix
``` {figure} images/cu-mafma-arith-instr-mix_panel.png
:alt: MFMA Arithmetic Instruction Mix
:figclass: figure
:align: center

The types of Matrix Fused Multiply-Add (MFMA) instructions that were issued.
```
###### VMEM Arithmetic Instruction Mix
``` {figure} images/cu-vmem-instr-mix_panel.png
:alt: VMEM Arithmetic Instruction Mix
:figclass: figure
:align: center

The types of vector memory (VMEM) instructions that were issued.
```

##### Compute Unit - Compute Pipeline
###### Speed-of-Light
``` {figure} images/cu-sol_panel.png
:alt: Speed-of-Light
:figclass: figure
:align: center

The number of floating-point and integer operations executed on the vector arithmetic logic unit (VALU) and Matrix Fused Multiply-Add (MFMA) units in various precisions.
```
###### Pipeline Stats
``` {figure} images/cu-pipeline-stats_panel.png
:alt: Pipeline Stats
:figclass: figure
:align: center

More detailed metrics to analyze the several independent pipelines found in the Compute Unit (CU).
```
###### Arithmetic Operations
``` {figure} images/cu-arith-ops_panel.png
:alt: Arithmetic Operations
:figclass: figure
:align: center

The total number of floating-point and integer operations executed in various precisions.
```

##### Local Data Share (LDS)
###### Speed-of-Light
``` {figure} images/lds-sol_panel.png
:alt: Speed-of-Light
:figclass: figure
:align: center

Key metrics for the Local Data Share (LDS) as a comparison with the peak achievable values of those metrics.
```
###### LDS Stats
``` {figure} images/lds-stats_panel.png
:alt: LDS Stats
:figclass: figure
:align: center

More detailed view of the Local Data Share (LDS) performance.
```

##### Instruction Cache
###### Speed-of-Light
``` {figure} images/instr-cache-sol_panel.png
:alt: Speed-of-Light
:figclass: figure
:align: center

Key metrics of the L1 Instruction (L1I) cache as a comparison with the peak achievable values of those metrics.
```
###### Instruction Cache Stats
``` {figure} images/instr-cache-accesses_panel.png
:alt: Instruction Cache Stats
:figclass: figure
:align: center

More detail on the hit/miss statistics of the L1 Instruction (L1I) cache.
```

##### Scalar L1D Cache
###### Speed-of-Light
``` {figure} images/sl1d-sol_panel.png
:alt: Speed-of-Light
:figclass: figure
:align: center

Key metrics of the Scalar L1 Data (sL1D) cache as a comparison with the peak achievable values of those metrics.
```
###### Scalar L1D Cache Accesses
``` {figure} images/sl1d-cache-accesses_panel.png
:alt: Scalar L1D Cache Accesses
:figclass: figure
:align: center

More detail on the types of accesses made to the Scalar L1 Data (sL1D) cache, and the hit/miss statistics.
```
###### Scalar L1D Cache - L2 Interface
``` {figure} images/sl1d-l12-interface_panel.png
:alt: Scalar L1D Cache - L2 Interface
:figclass: figure
:align: center

More detail on the data requested across the Scalar L1 Data (sL1D) cache <-> L2 interface.
```

##### Texture Address and Texture Data
###### Texture Addresser
``` {figure} images/ta_panel.png
:alt: Texture Addresser
:figclass: figure
:align: center

Metric specific to texture addresser (TA) which receives commands (e.g., instructions) and write/atomic data from the Compute Unit (CU), and coalesces them into fewer requests for the cache to process.
```
###### Texture Data
``` {figure} images/td_panel.png
:alt: Texture Data
:figclass: figure
:align: center

Metrics specific to texture data (TD) which routes data back to the requesting Compute Unit (CU).
```

##### Vector L1 Data Cache
###### Speed-of-Light
``` {figure} images/vl1d-sol_panel.png
:alt: Speed-of-Light
:figclass: figure
:align: center

Key metrics of the vector L1 data (vL1D) cache as a comparison with the peak achievable values of those metrics.
```
###### L1D Cache Stalls
``` {figure} images/vl1d-cache-stalls_panel.png
:alt: L1D Cache Stalls
:figclass: figure
:align: center

More detail on where vector L1 data (vL1D) cache is stalled in the pipeline, which may indicate performance limiters of the cache.
```
###### L1D Cache Accesses
``` {figure} images/vl1d-cache-accesses_panel.png
:alt: L1D Cache Accesses
:figclass: figure
:align: center

The type of requests incoming from the cache frontend, the number of requests that were serviced by the vector L1 data (vL1D) cache, and the number & type of outgoing requests to the L2 cache.
```
###### L1D - L2 Transactions
``` {figure} images/vl1d-l2-transactions_panel.png
:alt: L1D - L2 Transactions
:figclass: figure
:align: center

A more granular look at the types of requests made to the L2 cache.
```
###### L1D Addr Translation
``` {figure} images/vl1d-addr-translation_panel.png
:alt: L1D Addr Translation
:figclass: figure
:align: center

After a vector memory instruction has been processed/coalesced by the address processing unit of the vector L1 data (vL1D) cache, it must be translated from a virtual to physical address. These metrics provide more details on the L1 Translation Lookaside Buffer (TLB) which handles this process.
```

##### L2 Cache
###### Speed-of-Light
``` {figure} images/l2-sol_panel.png
:alt: Speed-of-Light
:figclass: figure
:align: center

Key metrics about the performance of the L2 cache, aggregated over all the L2 channels, as a comparison with the peak achievable values of those metrics.
```
###### L2 Cache Accesses
``` {figure} images/l2-accesses_panel.png
:alt: L2 Cache Accesses
:figclass: figure
:align: center

Incoming requests to the L2 cache from the vector L1 data (vL1D) cache and other clients (e.g., the sL1D and L1I caches).
```
###### L2 - Fabric Transactions
``` {figure} images/l2-fabric-transactions_panel.png
:alt: L2 - Fabric Transactions
:figclass: figure
:align: center

More detail on the flow of requests through Infinity Fabric™.
```
###### L2 - Fabric Interface Stalls
``` {figure} images/l2-fabric-interface-stalls_panel.png
:alt: L2 - Fabric Interface Stalls
:figclass: figure
:align: center

A breakdown of what types of requests in a kernel caused a stall (e.g., read vs write), and to which locations (e.g., to the accelerator’s local memory, or to remote accelerators/CPUs).
```

##### L2 Cache Per Channel
###### Aggregate Stats
``` {figure} images/l2-per-channel-agg-stats_panel.png
:alt: Aggregate Stats
:figclass: figure
:align: center

L2 Cache per channel performance at a glance. Metrics are aggregated over all available channels.
```
