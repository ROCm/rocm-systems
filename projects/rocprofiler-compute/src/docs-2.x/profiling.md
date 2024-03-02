# Profile Mode

```eval_rst
.. toctree::
   :glob:
   :maxdepth: 5
```

The [Omniperf](https://github.com/AMDResearch/omniperf) repository
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
$ ./vcopy -n 1048576 -b 256
vcopy testing on GCD 0
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
The *omniperf* script, available through the Omniperf repository, is used to aquire all necessary performance monitoring data through analysis of compute workloads.

**omniperf help:**
```shell-session
$ omniperf profile --help
usage: 
            
omniperf profile --name <workload_name> [profile options] [roofline options] -- <profile_cmd>

-------------------------------------------------------------------------------
            
Examples:
            
	omniperf profile -n vcopy_all -- ./vcopy -n 1048576 -b 256
	omniperf profile -n vcopy_SPI_TCC -b SQ TCC -- ./vcopy -n 1048576 -b 256
	omniperf profile -n vcopy_kernel -k vecCopy -- ./vcopy -n 1048576 -b 256
	omniperf profile -n vcopy_disp -d 0 -- ./vcopy -n 1048576 -b 256
	omniperf profile -n vcopy_roof --roof-only -- ./vcopy -n 1048576 -b 256
            
-------------------------------------------------------------------------------


Help:
  -h, --help                       show this help message and exit

General Options:
  -v, --version                    show program's version number and exit
  -V, --verbose                    Increase output verbosity

Profile Options:
  -n , --name                      			Assign a name to workload.
  -p , --path                      			Specify path to save workload.

  -k  [ ...], --kernel  [ ...]     			Kernel filtering.
  -d  [ ...], --dispatch  [ ...]   			Dispatch ID filtering.
  -b  [ ...], --ipblocks  [ ...]   			IP block filtering:
                                   			   SQ
                                   			   SQC
                                   			   TA
                                   			   TD
                                   			   TCP
                                   			   TCC
                                   			   SPI
                                   			   CPC
                                   			   CPF
  --join-type                      			Choose how to join rocprof runs: (DEFAULT: grid)
                                   			   kernel (i.e. By unique kernel name dispatches)
                                   			   grid (i.e. By unique kernel name + grid size dispatches)
  --no-roof                        			Profile without collecting roofline data.
  -- [ ...]                        			Provide command for profiling after double dash.
  --kernel-verbose                 			Specify Kernel Name verbose level 1-5. Lower the level, shorter the kernel name. (DEFAULT: 2) (DISABLE: 5)

Standalone Roofline Options:
  --roof-only                      			Profile roofline data only.
  --sort                           			Overlay top kernels or top dispatches: (DEFAULT: kernels)
                                   			   kernels
                                   			   dispatches
  -m  [ ...], --mem-level  [ ...]  			Filter by memory level: (DEFAULT: ALL)
                                   			   HBM
                                   			   L2
                                   			   vL1D
                                   			   LDS
  --device                         			GPU device ID. (DEFAULT: ALL)
  --kernel-names                   			Include kernel names in roofline plot.

```

- The `-k` \<kernel> flag allows for kernel filtering, which is compatible with the current rocProf utility.

- The `-d` \<dispatch> flag allows for dispatch ID filtering,  which is compatible with the current rocProf utility.

- The `-b` \<ipblocks> allows system profiling on one or more selected hardware components to speed up the profiling process. One can gradually include more hardware components, without overwriting performance data acquired on other hardware components.


The following sample command profiles the *vcopy* workload.

**vcopy profiling:**
```shell-session
$ omniperf profile --name vcopy -- ./vcopy -n 1048576 -b 256
ROC Profiler: /opt/rocm-5.7.1/bin/rocprof
Execution mode = profile

  ___                  _                  __ 
 / _ \ _ __ ___  _ __ (_)_ __   ___ _ __ / _|
| | | | '_ ` _ \| '_ \| | '_ \ / _ \ '__| |_ 
| |_| | | | | | | | | | | |_) |  __/ |  |  _|
 \___/|_| |_| |_|_| |_|_| .__/ \___|_|  |_|  
                        |_|                  

SoC = {'MI200'}
Profiler choice = rocprofv1
omniperf ver: 1.0.10
Path: /home/auser/repos/omniperf/sample/workloads/vcopy/MI200
Target: MI200
Command: ./vcopy -n 1048576 -b 256
Kernel Selection: None
Dispatch Selection: None
IP Blocks: All
KernelName verbose: 2

Current input file: /home/auser/repos/omniperf/sample/workloads/vcopy/MI200/perfmon/pmc_perf_11.txt
RPL: on '240301_151506' from '/opt/rocm-5.7.1' in '/home/auser/repos/omniperf/sample'
RPL: profiling '""./vcopy -n 1048576 -b 256""'
RPL: input file '/home/auser/repos/omniperf/sample/workloads/vcopy/MI200/perfmon/pmc_perf_11.txt'
RPL: output dir '/tmp/rpl_data_240301_151506_553019'
RPL: result dir '/tmp/rpl_data_240301_151506_553019/input0_results_240301_151506'
ROCProfiler: input from "/tmp/rpl_data_240301_151506_553019/input0.xml"
  gpu_index = 
  kernel = 
  range = 
  8 metrics
    SQ_INSTS_VALU_MFMA_F16, SQ_INSTS_VALU_MFMA_BF16, SQ_INSTS_VALU_MFMA_F32, SQ_INSTS_VALU_MFMA_F64, SQ_VALU_MFMA_BUSY_CYCLES, SQ_INSTS_FLAT_LDS_ONLY, SQ_INSTS_VALU_MFMA_MOPS_I8, SQ_INSTS_VALU_MFMA_MOPS_F16
vcopy testing on GCD 0
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

ROCPRofiler: 1 contexts collected, output directory /tmp/rpl_data_240301_151506_553019/input0_results_240301_151506
File '/home/auser/repos/omniperf/sample/workloads/vcopy/MI200/pmc_perf_11.csv' is generating
... 
[profiling] Kernel_Name shortening complete.

[roofline] Checking for roofline.csv in /home/auser/repos/omniperf/sample/workloads/vcopy/MI200
[roofline] No roofline data found. Generating...
Empirical Roofline Calculation
Copyright © 2022  Advanced Micro Devices, Inc. All rights reserved.
Total detected GPU devices: 4
GPU Device 0: Profiling...
 99% [||||||||||||||||||||||||||||||||||||||||||||||||||||||||||| ]
HBM BW, GPU ID: 0, workgroupSize:256, workgroups:2097152, experiments:100, traffic:8589934592 bytes, duration:6.2 ms, mean:1388.0 GB/sec, stdev=3.1 GB/sec
 99% [||||||||||||||||||||||||||||||||||||||||||||||||||||||||||| ]
L2 BW, GPU ID: 0, workgroupSize:256, workgroups:8192, experiments:100, traffic:687194767360 bytes, duration:136.5 ms, mean:5020.8 GB/sec, stdev=16.5 GB/sec
 99% [||||||||||||||||||||||||||||||||||||||||||||||||||||||||||| ]
L1 BW, GPU ID: 0, workgroupSize:256, workgroups:16384, experiments:100, traffic:26843545600 bytes, duration:2.9 ms, mean:9229.5 GB/sec, stdev=2.9 GB/sec
 99% [||||||||||||||||||||||||||||||||||||||||||||||||||||||||||| ]
LDS BW, GPU ID: 0, workgroupSize:256, workgroups:16384, experiments:100, traffic:33554432000 bytes, duration:1.9 ms, mean:17645.6 GB/sec, stdev=20.1 GB/sec
 99% [||||||||||||||||||||||||||||||||||||||||||||||||||||||||||| ]
Peak FLOPs (FP32), GPU ID: 0, workgroupSize:256, workgroups:16384, experiments:100, FLOP:274877906944, duration:13.078 ms, mean:20986.9 GFLOPS, stdev=310.8 GFLOPS
 99% [||||||||||||||||||||||||||||||||||||||||||||||||||||||||||| ]
Peak FLOPs (FP64), GPU ID: 0, workgroupSize:256, workgroups:16384, experiments:100, FLOP:137438953472, duration:6.7 ms, mean:20408.029297.1 GFLOPS, stdev=2.7 GFLOPS
 99% [||||||||||||||||||||||||||||||||||||||||||||||||||||||||||| ]
Peak MFMA FLOPs (BF16), GPU ID: 0, workgroupSize:256, workgroups:16384, experiments:100, FLOP:2147483648000, duration:12.6 ms, mean:170280.0 GFLOPS, stdev=22.3 GFLOPS
 99% [||||||||||||||||||||||||||||||||||||||||||||||||||||||||||| ]
Peak MFMA FLOPs (F16), GPU ID: 0, workgroupSize:256, workgroups:16384, experiments:100, FLOP:2147483648000, duration:13.0 ms, mean:164733.6 GFLOPS, stdev=24.3 GFLOPS
 99% [||||||||||||||||||||||||||||||||||||||||||||||||||||||||||| ]
Peak MFMA FLOPs (F32), GPU ID: 0, workgroupSize:256, workgroups:16384, experiments:100, FLOP:536870912000, duration:13.0 ms, mean:41399.6 GFLOPS, stdev=4.1 GFLOPS
 99% [||||||||||||||||||||||||||||||||||||||||||||||||||||||||||| ]
Peak MFMA FLOPs (F64), GPU ID: 0, workgroupSize:256, workgroups:16384, experiments:100, FLOP:268435456000, duration:6.5 ms, mean:41379.2 GFLOPS, stdev=4.4 GFLOPS
 99% [||||||||||||||||||||||||||||||||||||||||||||||||||||||||||| ]
Peak MFMA IOPs (I8), GPU ID: 0, workgroupSize:256, workgroups:16384, experiments:100, IOP:2147483648000, duration:12.9 ms, mean:166281.9 GOPS, stdev=2495.9 GOPS
GPU Device 1: Profiling...
...
GPU Device 2: Profiling...
...
GPU Device 3: Profiling...
...
Peak MFMA IOPs (I8), GPU ID: 3, workgroupSize:256, workgroups:16384, experiments:100, IOP:2147483648000, duration:12.9 ms, mean:166686.0 GOPS, stdev=11.2 GOPS
```
You will notice two main stages in *default* Omniperf profiling. The first stage collects all the counters needed for Omniperf analysis (omitting any filters you have provided). The second stage collects data for the roofline analysis (this stage can be disabled using `--no-roof`)

In this document, we use the term System on Chip (SoC) to refer to a particular family of accelerators. At the end of profiling, all resulting csv files should be located in a SoC specific target directory, e.g.:
  - "mi200" for the AMD Instinct (tm) MI200 family of accelerators
  - "mi100" for the AMD Instinct (tm) MI100 family of accelerators
etc.  The SoC names are generated as a part of Omniperf, and do not necessarily distinguish between different accelerators in the same family (e.g., an AMD Instinct (tm) MI210 vs an MI250)

> Note: Additionally, you will notice a few extra files. An SoC parameters file, *sysinfo.csv*, is created to reflect the target device settings. All profiling output is stored in *log.txt*. Roofline specific benchmark results are stored in *roofline.csv*.

```shell-session
$ ls workloads/vcopy/MI200/
total 112
total 60
drwxr-xr-x 1 auser agroup     0 Mar  1 15:15 perfmon
-rw-r--r-- 1 auser agroup 26175 Mar  1 15:15 pmc_perf.csv
-rw-r--r-- 1 auser agroup  1708 Mar  1 15:17 roofline.csv
-rw-r--r-- 1 auser agroup   519 Mar  1 15:15 SQ_IFETCH_LEVEL.csv
-rw-r--r-- 1 auser agroup   456 Mar  1 15:15 SQ_INST_LEVEL_LDS.csv
-rw-r--r-- 1 auser agroup   474 Mar  1 15:15 SQ_INST_LEVEL_SMEM.csv
-rw-r--r-- 1 auser agroup   474 Mar  1 15:15 SQ_INST_LEVEL_VMEM.csv
-rw-r--r-- 1 auser agroup   599 Mar  1 15:15 SQ_LEVEL_WAVES.csv
-rw-r--r-- 1 auser agroup   650 Mar  1 15:15 sysinfo.csv
-rw-r--r-- 1 auser agroup   399 Mar  1 15:15 timestamps.csv
```

### Filtering
To reduce profiling time and the counters collected one may use profiling filters. Profiling filters and their functionality depend on the underlying profiler being used. While Omniperf is profiler agnostic, we have provided a detailed description of profiling filters available when using Omniperf with [rocProf](https://rocm.docs.amd.com/projects/rocprofiler/en/latest/rocprof.html) below.



Filtering Options:

- The `-k` \<kernel> flag allows for kernel filtering. Useage is equivalent with the current rocProf utility ([see details below](#kernel-filtering)).

- The `-d` \<dispatch> flag allows for dispatch ID filtering. Useage is equivalent with the current rocProf utility ([see details below](#dispatch-filtering)).

- The `-b` \<ipblocks> allows system profiling on one or more selected hardware components to speed up the profiling process. One can gradually include more hardware components, without overwriting performance data acquired on other hardware components.

```{note}
Be cautious while combining different profiling filters in the same call. Conflicting filters may result in error.

i.e. filtering dispatch X, but dispatch X does not match your kernel name filter
```

#### Hardware Component Filtering
One can profile specific hardware components to speed up the profiling process. In Omniperf, we use the term IP block to refer to a hardware component or a group of hardware components. All profiling results are accumulated in the same target directory, without overwriting those for other hardware components, hence enabling the incremental profiling and analysis.

The following example only gathers hardware counters for the Shader Sequencer (SQ) and L2 Cache (TCC) components, skipping all other hardware components:
```shell-session
$ omniperf profile --name vcopy -b SQ TCC -- ./vcopy -n 1048576 -b 256
ROC Profiler: /opt/rocm-5.7.1/bin/rocprof
Execution mode = profile
               

SoC = {'MI200'}
Profiler choice = rocprofv1
fname: pmc_sq_perf8: Added
fname: pmc_spi_perf: Skipped
fname: pmc_sq_perf4: Added
fname: pmc_sq_perf6: Added
fname: pmc_cpf_perf: Skipped
fname: pmc_sqc_perf1: Skipped
fname: pmc_tcc_perf: Added
fname: pmc_tcc2_perf: Skipped
fname: pmc_sq_perf2: Added
fname: pmc_cpc_perf: Skipped
fname: pmc_td_perf: Skipped
fname: pmc_tcp_perf: Skipped
fname: pmc_sq_perf1: Added
fname: pmc_sq_perf3: Added
fname: pmc_ta_perf: Skipped
omniperf ver: 1.0.10
Path: /home/auser/repos/omniperf/sample/workloads/vcopy/MI200
Target: MI200
Command: ./vcopy -n 1048576 -b 256
Kernel Selection: None
Dispatch Selection: None
IP Blocks: ['sq', 'tcc']
KernelName verbose: 2
...
```

#### Kernel Filtering
Kernel filtering is based on the name of the kernel(s) you would like to isolate. Use a kernel name substring list to isolate desired kernels.

The following example demonstrates profiling isolating the kernel matching substring "vecCopy":
```shell-session
$ omniperf profile --name vcopy -k vecCopy -- ./vcopy -n 1048576 -b 256
ROC Profiler: /opt/rocm-5.7.1/bin/rocprof
Execution mode = profile
                

SoC = {'MI200'}
Profiler choice = rocprofv1
omniperf ver: 1.0.10
Path: /home/auser/repos/omniperf/sample/workloads/vcopy/MI200
Target: MI200
Command: ./vcopy -n 1048576 -b 256
Kernel Selection: ['vecCopy']
Dispatch Selection: None
IP Blocks: All
KernelName verbose: 2

Current input file: /home/auser/repos/omniperf/sample/workloads/vcopy/MI200/perfmon/pmc_perf_12.txt
RPL: on '240301_152305' from '/opt/rocm-5.7.1' in '/home/auser/repos/omniperf/sample'
RPL: profiling '""./vcopy -n 1048576 -b 256""'
RPL: input file '/home/auser/repos/omniperf/sample/workloads/vcopy/MI200/perfmon/pmc_perf_12.txt'
RPL: output dir '/tmp/rpl_data_240301_152305_562565'
RPL: result dir '/tmp/rpl_data_240301_152305_562565/input0_results_240301_152305'
ROCProfiler: input from "/tmp/rpl_data_240301_152305_562565/input0.xml"
  gpu_index = 
  kernel = vecCopy
...
```

#### Dispatch Filtering
Dispatch filtering is based on the *global* dispatch index of kernels in a run. 

The following example profiles only the 0th dispatched kernel in execution of the application:
```shell-session
$ omniperf profile --name vcopy -d 0 -- ./vcopy -n 1048576 -b 256
ROC Profiler: /opt/rocm-5.7.1/bin/rocprof
Execution mode = profile
               

SoC = {'MI200'}
Profiler choice = rocprofv1
omniperf ver: 1.0.10
Path: /home/auser/repos/omniperf/sample/workloads/vcopy/MI200
Target: MI200
Command: ./vcopy -n 1048576 -b 256
Kernel Selection: None
Dispatch Selection: ['0']
IP Blocks: All
KernelName verbose: 2

Current input file: /home/auser/repos/omniperf/sample/workloads/vcopy/MI200/perfmon/timestamps.txt
RPL: on '240301_152445' from '/opt/rocm-5.7.1' in '/home/auser/repos/omniperf/sample'
RPL: profiling '""./vcopy -n 1048576 -b 256""'
RPL: input file '/home/auser/repos/omniperf/sample/workloads/vcopy/MI200/perfmon/timestamps.txt'
RPL: output dir '/tmp/rpl_data_240301_152445_563349'
RPL: result dir '/tmp/rpl_data_240301_152445_563349/input0_results_240301_152445'
ROCProfiler: input from "/tmp/rpl_data_240301_152445_563349/input0.xml"
  gpu_index = 
  kernel = 
  range = 0

...
```


### Standalone Roofline
If you are only interested in generating roofline analysis data try using `--roof-only`. This will only collect counters relevant to roofline, as well as generate a standalone .pdf output of your roofline plot. 

Standalone Roofline Options:

- The `--sort` \<desired_sort> allows you to specify whether you would like to overlay top kernel or top dispatch data in your roofline plot.

- The `-m` \<cache_level> allows you to specify specific level(s) of cache you would like to include in your roofline plot.

- The `--device` \<gpu_id> allows you to specify a device id to collect performace data from when running our roofline benchmark on your system.

- If you would like to distinguish different kernels in your .pdf roofline plot use `--kernel-names`. This will give each kernel a unique marker identifiable from the plot's key.


#### Roofline Only
The following example demonstrates profiling roofline data only:
```shell-session
$ omniperf profile --name vcopy --roof-only -- ./vcopy -n 1048576 -b 256

...
[roofline] Checking for roofline.csv in /home/auser/repos/omniperf/sample/workloads/vcopy/MI200
[roofline] No roofline data found. Generating...
Checking for roofline.csv in /home/auser/repos/omniperf/sample/workloads/vcopy/MI200
Empirical Roofline Calculation
Copyright © 2022  Advanced Micro Devices, Inc. All rights reserved.
Total detected GPU devices: 4
GPU Device 0: Profiling...
 99% [||||||||||||||||||||||||||||||||||||||||||||||||||||||||||| ]
 ...
Empirical Roofline PDFs saved!
```
An inspection of our workload output folder shows .pdf plots were generated successfully
```shell-session
$ ls workloads/vcopy/MI200/
total 48
-rw-r--r-- 1 auser agroup 13331 Mar  1 16:05 empirRoof_gpu-0_fp32.pdf
-rw-r--r-- 1 auser agroup 13136 Mar  1 16:05 empirRoof_gpu-0_int8_fp16.pdf
drwxr-xr-x 1 auser agroup     0 Mar  1 16:03 perfmon
-rw-r--r-- 1 auser agroup  1101 Mar  1 16:03 pmc_perf.csv
-rw-r--r-- 1 auser agroup  1715 Mar  1 16:05 roofline.csv
-rw-r--r-- 1 auser agroup   650 Mar  1 16:03 sysinfo.csv
-rw-r--r-- 1 auser agroup   399 Mar  1 16:03 timestamps.csv
```
A sample *empirRoof_gpu-ALL_fp32.pdf* looks something like this:

![Sample Standalone Roof Plot](images/sample-roof-plot.png)
