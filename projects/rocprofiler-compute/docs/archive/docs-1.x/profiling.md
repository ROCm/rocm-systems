# Profile Mode

```eval_rst
.. toctree::
   :glob:
   :maxdepth: 5
```

The [Omniperf](https://github.com/ROCm/omniperf) repository
includes source code for a sample GPU compute workload,
__vcopy.cpp__. A copy of this file is available in the `share/sample`
subdirectory after a normal Omniperf installation, or via the
`$OMNIPERF_SHARE/sample` directory when using the supplied modulefile.

A compiled version of this workload is used throughout the following
sections to demonstrate the use of Omniperf in MI GPU performance
analysis. Unless otherwise noted, the performance analysis is done on
the MI200 platform.

## Workload Compilation
**vcopy compilation:**
```shell-session
$ hipcc vcopy.cpp -o vcopy
$ ls
vcopy   vcopy.cpp
$ ./vcopy 1048576 256
Finished allocating vectors on the CPU
Finished allocating vectors on the GPU
Finished copying vectors to the GPU
sw thinks it moved 1.000000 KB per wave
Total threads: 1048576, Grid Size: 4096 block Size:256, Wavefronts:16384:
Launching the  kernel on the GPU
Finished executing kernel
Finished copying the output vector from the GPU to the CPU
Releasing GPU memory
Releasing CPU memory
```

## Omniperf Profiling
The *omniperf* script, availible through the [Omniperf](https://github.com/ROCm/omniperf) repository, is used to aquire all necessary perfmon data through analysis of compute workloads.

**omniperf help:**
```shell-session
$ omniperf profile --help
ROC Profiler:  /usr/bin/rocprof

usage:

omniperf profile --name <workload_name> [profile options] [roofline options] -- <profile_cmd>



-------------------------------------------------------------------------------

Examples:

        omniperf profile -n vcopy_all -- ./vcopy 1048576 256

        omniperf profile -n vcopy_SPI_TCC -b SQ TCC -- ./vcopy 1048576 256

        omniperf profile -n vcopy_kernel -k vecCopy -- ./vcopy 1048576 256

        omniperf profile -n vcopy_disp -d 0 -- ./vcopy 1048576 256

        omniperf profile -n vcopy_roof --roof-only -- ./vcopy 1048576 256

-------------------------------------------------------------------------------



Help:
  -h, --help                      show this help message and exit

General Options:
  -v, --version                   show program's version number and exit
  -V, --verbose                   Increase output verbosity

Profile Options:
  -n , --name                                           Assign a name to workload.
  -p , --path                                           Specify path to save workload.
                                                        (DEFAULT: /home/colramos/GitHub/omniperf/workloads/<name>)
  -k  [ ...], --kernel  [ ...]                          Kernel filtering.
  -b  [ ...], --ipblocks  [ ...]                        IP block filtering:
                                                           SQ
                                                           SQC
                                                           TA
                                                           TD
                                                           TCP
                                                           TCC
                                                           SPI
                                                           CPC
                                                           CPF
  -d  [ ...], --dispatch  [ ...]                        Dispatch ID filtering.
  --no-roof                                             Profile without collecting roofline data.
  -- [ ...]                                             Provide command for profiling after double dash.

Standalone Roofline Options:
  --roof-only                                           Profile roofline data only.
  --sort                                                Overlay top kernels or top dispatches: (DEFAULT: kernels)
                                                           kernels
                                                           dispatches
  -m , --mem-level                                      Filter by memory level: (DEFAULT: ALL)
                                                           HBM
                                                           L2
                                                           vL1D
                                                           LDS
  --device                                              GPU device ID. (DEFAULT: ALL)
  --kernel-names                                        Include kernel names in roofline plot.
```

The following sample command profiles the *vcopy* workload.

**vcopy profiling:**
```shell-session
$ omniperf profile --name vcopy -- ./vcopy 1048576 256
Resolving rocprof
ROC Profiler:  /usr/bin/rocprof


-------------
Profile only
-------------

omniperf ver:  1.0.8-PR1
Path:  /home/colramos/GitHub/omniperf-pub/workloads
Target:  mi200
Command:  /home/colramos/vcopy 1048576 256
Kernel Selection:  None
Dispatch Selection:  None
IP Blocks: All
Log:  /home/colramos/GitHub/omniperf-pub/workloads/vcopy/mi200/log.txt

/home/colramos/GitHub/omniperf-pub/workloads/vcopy/mi200/perfmon/SQ_INST_LEVEL_SMEM.txt
RPL: on '230411_165021' from '/opt/rocm-5.2.1' in '/home/colramos/GitHub/omniperf-pub'
RPL: profiling '""/home/colramos/vcopy 1048576 256""'
RPL: input file '/home/colramos/GitHub/omniperf-pub/workloads/vcopy/mi200/perfmon/SQ_INST_LEVEL_SMEM.txt'
RPL: output dir '/tmp/rpl_data_230411_165021_26406'
RPL: result dir '/tmp/rpl_data_230411_165021_26406/input0_results_230411_165021'
Finished allocating vectors on the CPU
ROCProfiler: input from "/tmp/rpl_data_230411_165021_26406/input0.xml"
  gpu_index =
  kernel =
  range =
  3 metrics
    SQ_INSTS_SMEM, SQ_INST_LEVEL_SMEM, SQ_ACCUM_PREV_HIRES
Finished allocating vectors on the GPU
Finished copying vectors to the GPU
sw thinks it moved 1.000000 KB per wave
Total threads: 1048576, Grid Size: 4096 block Size:256, Wavefronts:16384:
Launching the  kernel on the GPU
Finished executing kernel
Finished copying the output vector from the GPU to the CPU
Releasing GPU memory
Releasing CPU memory

... ...
ROCPRofiler: 1 contexts collected, output directory /tmp/rpl_data_220527_130317_1787038/input_results_220527_130317
File 'workloads/vcopy/mi200/timestamps.csv' is generating
Total detected GPU devices: 2
GPU Device 0: Profiling...
 99% [||||||||||||||||||||||||||||||||||||||||||||||||||||||||||| ]
HBM BW, GPU ID: 0, workgroupSize:256, workgroups:2097152, experiments:100, traffic:8589934592 bytes, duration:6.2 ms, mean:1382.7 GB/sec, stdev=2.4 GB/sec
 99% [||||||||||||||||||||||||||||||||||||||||||||||||||||||||||| ]
L2 BW, GPU ID: 0, workgroupSize:256, workgroups:8192, experiments:100, traffic:687194767360 bytes, duration:157.9 ms, mean:4358.7 GB/sec, stdev=4.7 GB/sec
 99% [||||||||||||||||||||||||||||||||||||||||||||||||||||||||||| ]
L1 BW, GPU ID: 0, workgroupSize:256, workgroups:16384, experiments:100, traffic:26843545600 bytes, duration:3.3 ms, mean:8247.1 GB/sec, stdev=5.1 GB/sec
 99% [||||||||||||||||||||||||||||||||||||||||||||||||||||||||||| ]
LDS BW, GPU ID: 0, workgroupSize:256, workgroups:16384, experiments:100, traffic:33554432000 bytes, duration:2.4 ms, mean:14246.3 GB/sec, stdev=29.5 GB/sec
 99% [||||||||||||||||||||||||||||||||||||||||||||||||||||||||||| ]
Peak FLOPs (FP32), GPU ID: 0, workgroupSize:256, workgroups:16384, experiments:100, FLOP:274877906944, duration:14.507 ms, mean:18949.6 GFLOPS, stdev=4.5 GFLOPS
 99% [||||||||||||||||||||||||||||||||||||||||||||||||||||||||||| ]
Peak FLOPs (FP64), GPU ID: 0, workgroupSize:256, workgroups:16384, experiments:100, FLOP:137438953472, duration:7.5 ms, mean:18308.197266.1 GFLOPS, stdev=3.6 GFLOPS
 99% [||||||||||||||||||||||||||||||||||||||||||||||||||||||||||| ]
Peak MFMA FLOPs (BF16), GPU ID: 0, workgroupSize:256, workgroups:16384, experiments:100, FLOP:2147483648000, duration:14.0 ms, mean:153574.8 GFLOPS, stdev=79.9 GFLOPS
 99% [||||||||||||||||||||||||||||||||||||||||||||||||||||||||||| ]
Peak MFMA FLOPs (F16), GPU ID: 0, workgroupSize:256, workgroups:16384, experiments:100, FLOP:2147483648000, duration:14.5 ms, mean:147680.1 GFLOPS, stdev=34.7 GFLOPS
 99% [||||||||||||||||||||||||||||||||||||||||||||||||||||||||||| ]
Peak MFMA FLOPs (F32), GPU ID: 0, workgroupSize:256, workgroups:16384, experiments:100, FLOP:536870912000, duration:14.5 ms, mean:37142.1 GFLOPS, stdev=8.4 GFLOPS
 99% [||||||||||||||||||||||||||||||||||||||||||||||||||||||||||| ]
Peak MFMA FLOPs (F64), GPU ID: 0, workgroupSize:256, workgroups:16384, experiments:100, FLOP:268435456000, duration:7.3 ms, mean:36919.5 GFLOPS, stdev=14.1 GFLOPS
 99% [||||||||||||||||||||||||||||||||||||||||||||||||||||||||||| ]
Peak MFMA IOPs (I8), GPU ID: 0, workgroupSize:256, workgroups:16384, experiments:100, IOP:2147483648000, duration:14.4 ms, mean:149570.6 GOPS, stdev=41.7 GOPS
GPU Device 1: Profiling...
 99% [||||||||||||||||||||||||||||||||||||||||||||||||||||||||||| ]
HBM BW, GPU ID: 1, workgroupSize:256, workgroups:2097152, experiments:100, traffic:8589934592 bytes, duration:6.2 ms, mean:1382.7 GB/sec, stdev=2.9 GB/sec
 99% [||||||||||||||||||||||||||||||||||||||||||||||||||||||||||| ]
L2 BW, GPU ID: 1, workgroupSize:256, workgroups:8192, experiments:100, traffic:687194767360 bytes, duration:157.6 ms, mean:4371.0 GB/sec, stdev=4.1 GB/sec
 99% [||||||||||||||||||||||||||||||||||||||||||||||||||||||||||| ]
L1 BW, GPU ID: 1, workgroupSize:256, workgroups:16384, experiments:100, traffic:26843545600 bytes, duration:3.2 ms, mean:8297.4 GB/sec, stdev=11.6 GB/sec
 99% [||||||||||||||||||||||||||||||||||||||||||||||||||||||||||| ]
LDS BW, GPU ID: 1, workgroupSize:256, workgroups:16384, experiments:100, traffic:33554432000 bytes, duration:1.8 ms, mean:18839.2 GB/sec, stdev=44.5 GB/sec
 99% [||||||||||||||||||||||||||||||||||||||||||||||||||||||||||| ]
Peak FLOPs (FP32), GPU ID: 1, workgroupSize:256, workgroups:16384, experiments:100, FLOP:274877906944, duration:14.441 ms, mean:19037.6 GFLOPS, stdev=2.7 GFLOPS
 99% [||||||||||||||||||||||||||||||||||||||||||||||||||||||||||| ]
Peak FLOPs (FP64), GPU ID: 1, workgroupSize:256, workgroups:16384, experiments:100, FLOP:137438953472, duration:7.5 ms, mean:18402.255859.1 GFLOPS, stdev=20.1 GFLOPS
 99% [||||||||||||||||||||||||||||||||||||||||||||||||||||||||||| ]
Peak MFMA FLOPs (BF16), GPU ID: 1, workgroupSize:256, workgroups:16384, experiments:100, FLOP:2147483648000, duration:13.9 ms, mean:154240.3 GFLOPS, stdev=119.3 GFLOPS
 99% [||||||||||||||||||||||||||||||||||||||||||||||||||||||||||| ]
Peak MFMA FLOPs (F16), GPU ID: 1, workgroupSize:256, workgroups:16384, experiments:100, FLOP:2147483648000, duration:14.5 ms, mean:148450.1 GFLOPS, stdev=112.6 GFLOPS
 99% [||||||||||||||||||||||||||||||||||||||||||||||||||||||||||| ]
Peak MFMA FLOPs (F32), GPU ID: 1, workgroupSize:256, workgroups:16384, experiments:100, FLOP:536870912000, duration:14.4 ms, mean:37335.2 GFLOPS, stdev=43.1 GFLOPS
 99% [||||||||||||||||||||||||||||||||||||||||||||||||||||||||||| ]
Peak MFMA FLOPs (F64), GPU ID: 1, workgroupSize:256, workgroups:16384, experiments:100, FLOP:268435456000, duration:7.2 ms, mean:37105.3 GFLOPS, stdev=39.5 GFLOPS
 99% [||||||||||||||||||||||||||||||||||||||||||||||||||||||||||| ]
Peak MFMA IOPs (I8), GPU ID: 1, workgroupSize:256, workgroups:16384, experiments:100, IOP:2147483648000, duration:14.3 ms, mean:150317.8 GOPS, stdev=203.5 GOPS
```
You'll notice two stages in *default* Omniperf profiling. The first stage collects all the counters needed for Omniperf analysis (omitting any filters you've provided). The second stage collects data for the roofline analysis (this stage can be disabled using `--no-roof`)

At the end of the profiling, all resulting csv files should be located in a SOC specific target directory, e.g.:
  - "mi200" for the AMD Instinct (tm) MI-200 family of accelerators
  - "mi100" for the AMD Instinct (tm) MI-100 family of accelerators
etc.  The SOC names are generated as a part of Omniperf, and do not necessarily distinguish between different accelerators in the same family (e.g., an AMD Instinct (tm) MI-210 vs an MI-250)

> Note: Additionally, you'll notice a few extra files. An SoC parameters file, *sysinfo.csv*, is created to reflect the target device settings. All profiling output is stored in *log.txt*. Roofline specific benchmark results are stored in *roofline.csv*.

```shell
$ ls workloads/vcopy/mi200/
total 112
drwxrwxr-x 3 colramos colramos  4096 Apr 11 16:42 .
drwxrwxr-x 3 colramos colramos  4096 Apr 11 16:42 ..
-rw-rw-r-- 1 colramos colramos 40750 Apr 11 16:44 log.txt
drwxrwxr-x 2 colramos colramos  4096 Apr 11 16:42 perfmon
-rw-rw-r-- 1 colramos colramos 25877 Apr 11 16:42 pmc_perf.csv
-rw-rw-r-- 1 colramos colramos  1716 Apr 11 16:44 roofline.csv
-rw-rw-r-- 1 colramos colramos   429 Apr 11 16:42 SQ_IFETCH_LEVEL.csv
-rw-rw-r-- 1 colramos colramos   366 Apr 11 16:42 SQ_INST_LEVEL_LDS.csv
-rw-rw-r-- 1 colramos colramos   391 Apr 11 16:42 SQ_INST_LEVEL_SMEM.csv
-rw-rw-r-- 1 colramos colramos   384 Apr 11 16:42 SQ_INST_LEVEL_VMEM.csv
-rw-rw-r-- 1 colramos colramos   509 Apr 11 16:42 SQ_LEVEL_WAVES.csv
-rw-rw-r-- 1 colramos colramos   498 Apr 11 16:42 sysinfo.csv
-rw-rw-r-- 1 colramos colramos   309 Apr 11 16:42 timestamps.csv
```

### Filtering
To reduce profiling time and the counters collected one may use profiling filters. Profiling filters and their functionality depend on the underlying profiler being used. While Omniperf is profiler agnostic, we've provided a detailed description of profiling filters available when using Omniperf with [rocProfiler](https://rocm.docs.amd.com/projects/rocprofiler/en/latest/rocprof.html) below.



Filtering Options:

- The `-k` \<kernel> flag allows for kernel filtering. Useage is equivalent with the current rocprof utility ([see details below](#kernel-filtering)).

- The `-d` \<dispatch> flag allows for dispatch ID filtering. Useage is equivalent with the current rocprof utility ([see details below](#dispatch-filtering)).

- The `-b` \<ipblocks> allows system profiling on one or more selected IP blocks to speed up the profiling process. One can gradually incorporate more IP blocks, without overwriting performance data acquired on other IP blocks.

```{note}
Be cautious while combining different profiling filters in the same call. Conflicting filters may result in error.

i.e. filtering dispatch X, but dispatch X does not match your kernel name filter
```

#### IP Block Filtering
One can profile a selected IP Block to speed up the profiling process. All profiling results are accumulated in the same target directory, without overwriting those for other IP blocks, hence enabling the incremental profiling and analysis.

The following example only gathers hardware counters for SQ and TCC, skipping all other IP Blocks:
```shell
$ omniperf profile --name vcopy -b SQ TCC -- ./sample/vcopy 1048576 256
Resolving rocprof
ROC Profiler:  /usr/bin/rocprof


-------------
Profile only
-------------

omniperf ver:  1.0.8-PR1
Path:  /home/colramos/GitHub/omniperf-pub/workloads
Target:  mi200
Command:  /home/colramos/vcopy 1048576 256
Kernel Selection:  None
Dispatch Selection:  None
IP Blocks:  ['SQ', 'TCC']
fname: pmc_sq_perf2: Added
fname: pmc_td_perf: Skipped
fname: pmc_tcc2_perf: Skipped
fname: pmc_tcp_perf: Skipped
fname: pmc_spi_perf: Skipped
fname: pmc_sq_perf4: Added
fname: pmc_sqc_perf1: Skipped
fname: pmc_tcc_perf: Added
fname: pmc_cpf_perf: Skipped
fname: pmc_sq_perf8: Added
fname: pmc_cpc_perf: Skipped
fname: pmc_sq_perf1: Added
fname: pmc_ta_perf: Skipped
fname: pmc_sq_perf3: Added
fname: pmc_sq_perf6: Added
Log:  /home/colramos/GitHub/omniperf-pub/workloads/vcopy/mi200/log.txt
...
```

#### Kernel Filtering
Kernel filtering is based on the name of the kernel(s) you'd like to isolate. Use a kernel name substring list to isolate desired kernels.

The following example demonstrates profiling isolating the kernel matching substring "vecCopy":
```shell
$ omniperf profile --name vcopy -k vecCopy -- ./vcopy 1048576 256
Resolving rocprof
ROC Profiler:  /usr/bin/rocprof


-------------
Profile only
-------------

omniperf ver:  1.0.8-PR1
Path:  /home/colramos/GitHub/omniperf-pub/workloads
Target:  mi200
Command:  /home/colramos/vcopy 1048576 256
Kernel Selection:  ['vecCopy']
Dispatch Selection:  None
IP Blocks: All
Log:  /home/colramos/GitHub/omniperf-pub/workloads/vcopy/mi200/log.txt

/home/colramos/GitHub/omniperf-pub/workloads/vcopy/mi200/perfmon/SQ_INST_LEVEL_SMEM.txt
RPL: on '230411_170300' from '/opt/rocm-5.2.1' in '/home/colramos/GitHub/omniperf-pub'
RPL: profiling '""/home/colramos/vcopy 1048576 256""'
RPL: input file '/home/colramos/GitHub/omniperf-pub/workloads/vcopy/mi200/perfmon/SQ_INST_LEVEL_SMEM.txt'
RPL: output dir '/tmp/rpl_data_230411_170300_29696'
RPL: result dir '/tmp/rpl_data_230411_170300_29696/input0_results_230411_170300'
Finished allocating vectors on the CPU
ROCProfiler: input from "/tmp/rpl_data_230411_170300_29696/input0.xml"
  gpu_index =
  kernel = vecCopy

... ...
```

#### Dispatch Filtering
Dispatch filtering is based on the *global* dispatch index of kernels in a run.

The following example profiles only the 0th dispatched kernel in execution of the application:
```shell-session
$ omniperf profile --name vcopy -d 0 -- ./vcopy 1048576 256
Resolving rocprof
ROC Profiler:  /usr/bin/rocprof


-------------
Profile only
-------------

omniperf ver:  1.0.8-PR1
Path:  /home/colramos/GitHub/omniperf-pub/workloads
Target:  mi200
Command:  /home/colramos/vcopy 1048576 256
Kernel Selection:  None
Dispatch Selection:  ['0']
IP Blocks: All
Log:  /home/colramos/GitHub/omniperf-pub/workloads/vcopy/mi200/log.txt

/home/colramos/GitHub/omniperf-pub/workloads/vcopy/mi200/perfmon/SQ_INST_LEVEL_SMEM.txt
RPL: on '230411_170356' from '/opt/rocm-5.2.1' in '/home/colramos/GitHub/omniperf-pub'
RPL: profiling '""/home/colramos/vcopy 1048576 256""'
RPL: input file '/home/colramos/GitHub/omniperf-pub/workloads/vcopy/mi200/perfmon/SQ_INST_LEVEL_SMEM.txt'
RPL: output dir '/tmp/rpl_data_230411_170356_30314'
RPL: result dir '/tmp/rpl_data_230411_170356_30314/input0_results_230411_170356'
Finished allocating vectors on the CPU
ROCProfiler: input from "/tmp/rpl_data_230411_170356_30314/input0.xml"
  gpu_index =
  kernel =
  range = 0
...
```



### Standalone Roofline
If you're only interested in generating roofline analysis data try using `--roof-only`. This will only collect counters relevent to roofline, as well as generate a standalone .pdf output of your roofline plot.

Standalone Roofline Options:

- The `--sort` \<desired_sort> allows you to specify whether you'd like to overlay top kernel or top dispatch data in your roofline plot.

- The `-m` \<cache_level> allows you to specify specific level(s) of cache you'd like to include in your roofline plot.

- The `--device` \<gpu_id> allows you to specify a device id to collect performace data from when running our roofline benchmark on your system.

- If you'd like to distinguish different kernels in your .pdf roofline plot use `--kernel-names`. This will give each kernel a unique marker identifiable from the plot's key.


#### Roofline Only
The following example demonstrates profiling roofline data only:
```shell-session
$ omniperf profile --name vcopy --roof-only -- ./vcopy 1048576 256
Resolving rocprof
ROC Profiler:  /usr/bin/rocprof


--------
Roofline only
--------

Checking for roofline.csv in  /home/colramos/GitHub/omniperf-pub/workloads/vcopy/mi200
No roofline data found. Generating...
Empirical Roofline Calculation
Copyright © 2022  Advanced Micro Devices, Inc. All rights reserved.
Total detected GPU devices: 4
GPU Device 0: Profiling...
 99% [||||||||||||||||||||||||||||||||||||||||||||||||||||||||||| ]
 ... ...
Checking for roofline.csv in  /home/colramos/GitHub/omniperf-pub/workloads/mix/mi200
Checking for sysinfo.csv in  /home/colramos/GitHub/omniperf-pub/workloads/mix/mi200
Checking for pmc_perf.csv in  /home/colramos/GitHub/omniperf-pub/workloads/mix/mi200
Empirical Roofline PDFs saved!
```
An inspection of our workload output folder shows .pdf plots were generated successfully
```shell-session
$ ls workloads/vcopy/mi200/
total 176
drwxrwxr-x 3 colramos colramos  4096 Apr 11 17:18 .
drwxrwxr-x 3 colramos colramos  4096 Apr 11 17:15 ..
-rw-rw-r-- 1 colramos colramos 13271 Apr 11 17:18 empirRoof_gpu-ALL_fp32.pdf
-rw-rw-r-- 1 colramos colramos 13175 Apr 11 17:18 empirRoof_gpu-ALL_int8_fp16.pdf
-rw-rw-r-- 1 colramos colramos 26560 Apr 11 17:16 log.txt
drwxrwxr-x 2 colramos colramos  4096 Apr 11 17:16 perfmon
-rw-rw-r-- 1 colramos colramos 54031 Apr 11 17:16 pmc_perf.csv
-rw-rw-r-- 1 colramos colramos  1714 Apr 11 17:16 roofline.csv
-rw-rw-r-- 1 colramos colramos   457 Apr 11 17:16 sysinfo.csv
-rw-rw-r-- 1 colramos colramos 37521 Apr 11 17:16 timestamps.csv
```
A sample *empirRoof_gpu-ALL_fp32.pdf* looks something like this:

![Sample Standalone Roof Plot](images/sample-roof-plot.png)
