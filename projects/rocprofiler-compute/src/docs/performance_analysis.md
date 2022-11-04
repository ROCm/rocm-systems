# Omniperf Performance Analysis

```eval_rst
.. toctree::
   :glob:
   :maxdepth: 4
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
```shell
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
The *omniperf* script, availible through the [Omniperf](https://github.com/AMDResearch/omniperf) repository, is used to aquire all necessary perfmon data through analysis of compute workloads.

**omniperf help:**
```shell
$ omniperf profile --help
ROC Profiler:  /usr/bin/rocprof

usage:

omniperf profile --name <workload_name> [profile options] [roofline options] -- <profile_cmd>



-------------------------------------------------------------------------------

Examples:

        omniperf profile -n vcopy_all -- ./vcopy 1048576 256

        omniperf profile -n vcopy_SPI_TD -b SQ TCC -- ./vcopy 1048576 256

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
  --axes  [ ...]                                        Desired axis values for graph. As follows:
                                                           xmin xmax ymin ymax
  --device                                              GPU device ID. (DEFAULT: ALL)
```

- The `-k` \<kernel> flag allows for kernel filtering, which is compatible with the current rocprof utility.

- The `-d` \<dispatch> flag allows for dispatch ID filtering,  which is compatible with the current rocprof utility. 

- The `-b` \<ipblocks> allows system profiling on one or more selected IP blocks to speed up the profiling process. One can gradually incorporate more IP blocks, without overwriting performance data acquired on other IP blocks.

The following sample command profiles the *vcopy* workload.

**vcopy profiling:**
```shell
$ omniperf profile --name vcopy -- ./vcopy 1048576 256
ROC Profiler:  /usr/bin/rocprof
 
--------
Profile only
--------
 
omniperf ver:  v1.0.3
Path:  workloads
Target:  mi200
Command:  ./vcopy 1048576 256
Kernel Selection:  None
Dispatch Selection:  None
IP Blocks: All
RPL: on '220527_130247' from '/opt/rocm-5.2.0-9768/rocprofiler' in '/home/amd/xlu/test'
RPL: profiling '""./vcopy 1048576 256""'
RPL: input file 'workloads/vcopy/mi200/perfmon/SQ_IFETCH_LEVEL.txt'
RPL: output dir '/tmp/rpl_data_220527_130247_1781699'
RPL: result dir '/tmp/rpl_data_220527_130247_1781699/input0_results_220527_130247'
Finished allocating vectors on the CPU
ROCProfiler: input from "/tmp/rpl_data_220527_130247_1781699/input0.xml"
  gpu_index =
  kernel =
  range =
  6 metrics
    GRBM_COUNT, GRBM_GUI_ACTIVE, SQ_WAVES, SQ_IFETCH, SQ_IFETCH_LEVEL, SQ_ACCUM_PREV_HIRES
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

At the end of the profiling, all resulting csv files should be located in the SOC specific target directory, e.g., mi200.

> Note: An SoC parameters file, *sysinfo.csv*, is also created to reflect the target device settings. 
```shell
$ ls workloads/vcopy/mi200/
total 116
-rw-rw-r-- 1 amd amd   400 May 27 13:03 SQC_DCACHE_INFLIGHT_LEVEL.csv
-rw-rw-r-- 1 amd amd   452 May 27 13:03 SQC_DCACHE_TC_INFLIGHT_LEVEL.csv
-rw-rw-r-- 1 amd amd   451 May 27 13:03 SQC_DCACHE_UTCL1_INFLIGHT_LEVEL.csv
-rw-rw-r-- 1 amd amd   445 May 27 13:03 SQC_DCACHE_UTCL2_INFLIGHT_LEVEL.csv
-rw-rw-r-- 1 amd amd   396 May 27 13:03 SQC_ICACHE_INFLIGHT_LEVEL.csv
-rw-rw-r-- 1 amd amd   396 May 27 13:03 SQC_ICACHE_TC_INFLIGHT_LEVEL.csv
-rw-rw-r-- 1 amd amd   445 May 27 13:03 SQC_ICACHE_UTCL1_INFLIGHT_LEVEL.csv
-rw-rw-r-- 1 amd amd   442 May 27 13:03 SQC_ICACHE_UTCL2_INFLIGHT_LEVEL.csv
-rw-rw-r-- 1 amd amd   423 May 27 13:03 SQC_TC_INFLIGHT_LEVEL.csv
-rw-rw-r-- 1 amd amd   437 May 27 13:02 SQ_IFETCH_LEVEL.csv
-rw-rw-r-- 1 amd amd   374 May 27 13:03 SQ_INST_LEVEL_EXP.csv
-rw-rw-r-- 1 amd amd   374 May 27 13:03 SQ_INST_LEVEL_GDS.csv
-rw-rw-r-- 1 amd amd   374 May 27 13:02 SQ_INST_LEVEL_LDS.csv
-rw-rw-r-- 1 amd amd   392 May 27 13:03 SQ_INST_LEVEL_SMEM.csv
-rw-rw-r-- 1 amd amd   392 May 27 13:03 SQ_INST_LEVEL_VMEM.csv
-rw-rw-r-- 1 amd amd   516 May 27 13:03 SQ_LEVEL_WAVES.csv
drwxrwxr-x 2 amd amd  4096 May 27 13:02 perfmon
-rw-rw-r-- 1 amd amd 32797 May 27 13:03 pmc_perf.csv
-rw-rw-r-- 1 amd amd   958 May 27 13:04 roofline.csv
-rw-rw-r-- 1 amd amd   469 May 27 13:03 sysinfo.csv
-rw-rw-r-- 1 amd amd   317 May 27 13:03 timestamps.csv
```

### IP Block Profiling
One can profile a selected IP Block to speed up the profiling process. All profiling results are accumulated in the same target directory, without overwriting those for other IP blocks, hence enabling the incremental profiling and analysis.

The following example only profiles SQ and TCC, skipping all other IP Blocks.
```shell
$ omniperf profile --name vcopy -b SQ TCC -- ./sample/vcopy 1048576 256
ROC Profiler:  /usr/bin/rocprof

--------
Profile only
--------

omniperf ver:  v1.0.3
Path:  workloads
Target:  mi200
Command:  ./vcopy 1048576 256
Kernel Selection:  None
Dispatch Selection:  None
IP Blocks:  ['SQ', 'TCC']
fname: pmc_ta_perf: Skipped
fname: pmc_sq_perf3: Added
fname: pmc_sqc_icache_perf2: Skipped
fname: pmc_sqc_dcache_perf2: Skipped
fname: pmc_sqc_dcache_perf3: Skipped
fname: pmc_sqc_icache_perf4: Skipped
fname: pmc_sqc_dcache_perf5: Skipped
fname: pmc_sqc_dcache_perf4: Skipped
fname: pmc_cpc_perf: Skipped
fname: pmc_sqc_icache_perf1: Skipped
fname: pmc_sq_perf4: Added
fname: pmc_sqc_icache_perf5: Skipped
fname: pmc_sq_perf5: Added
fname: pmc_grbm_perf: Skipped
fname: pmc_sq_perf8: Added
fname: pmc_sq_perf2: Added
fname: pmc_sq_perf6: Added
fname: pmc_sqc_icache_perf3: Skipped
fname: pmc_sqc_dcache_perf1: Skipped
fname: pmc_sq_perf7: Added
fname: pmc_cpf_perf: Skipped
fname: pmc_sqc_dcache_perf6: Skipped
fname: pmc_tcp_perf: Skipped
fname: pmc_spi_perf: Skipped
fname: pmc_td_perf: Skipped
fname: pmc_tcc_perf: Added
fname: pmc_tcc2_perf: Skipped
fname: pmc_sq_perf1: Added
RPL: on '220527_130730' from '/opt/rocm-5.2.0-9768/rocprofiler' in '/home/amd/xlu/test'
RPL: profiling '""./vcopy 1048576 256""'
RPL: input file 'workloads/vcopy/mi200/perfmon/SQ_IFETCH_LEVEL.txt'
RPL: output dir '/tmp/rpl_data_220527_130730_1788165'
RPL: result dir '/tmp/rpl_data_220527_130730_1788165/input0_results_220527_130730'
Finished allocating vectors on the CPU
ROCProfiler: input from "/tmp/rpl_data_220527_130730_1788165/input0.xml"
 
... ...
ROCPRofiler: 1 contexts collected, output directory /tmp/rpl_data_220527_130751_1791421/input_results_220527_130751
File 'workloads/vcopy/mi200/timestamps.csv' is generating
Total detected GPU devices: 2
GPU Device 0: Profiling...
... ...
```

### Kernel Filtering
The following example demonstrates profiling on selected kernels:
```shell
$ omniperf profile --name vcopy -k vecCopy -- ./vcopy 1048576 256
ROC Profiler:  /usr/bin/rocprof
 
--------
Profile only
--------
 
omniperf ver:  v1.0.3
Path:  workloads
Target:  mi200
Command:  ./vcopy 1048576 256
Kernel Selection:  ['vecCopy']
Dispatch Selection:  None
IP Blocks: All
RPL: on '220527_164748' from '/opt/rocm-5.2.0-9768/rocprofiler' in '/home/amd/xlu/test'
RPL: profiling '""./vcopy 1048576 256""'
RPL: input file 'workloads/vcopy/mi200/perfmon/SQ_IFETCH_LEVEL.txt'
RPL: output dir '/tmp/rpl_data_220527_164748_1795414'
RPL: result dir '/tmp/rpl_data_220527_164748_1795414/input0_results_220527_164748'
Finished allocating vectors on the CPU
ROCProfiler: input from "/tmp/rpl_data_220527_164748_1795414/input0.xml"
  gpu_index =
  kernel = vecCopy
 
... ...
```

### Dispatch Filtering
The following example demonstrates profiling on selected dispatches:
```shell
$ omniperf profile --name vcopy -d 0 -- ./vcopy 1048576 256
ROC Profiler:  /usr/bin/rocprof
 
--------
Profile only
--------
 
omniperf ver:  v1.0
Path:  workloads
Target:  mi200
Command:  ./vcopy 1048576 256
Kernel Selection:  None
Dispatch Selection:  ['0']
IP Blocks: All
... ...
```

## Omniperf Grafana GUI Import
The omniperf database `--import` option imports the raw profiling data to Grafana's backend MongoDB database. This step is only required for Grafana GUI based performance analysis. 

Each workload is imported to a separate database with the following naming convention:

    omniperf_<team>_<database>_<soc>

e.g., omniperf_asw_vcopy_mi200.

Below is the sample command to import the *vcopy* profiling data.

```shell
$ omniperf database --help
ROC Profiler:  /usr/bin/rocprof

usage: 
                                        
omniperf database <interaction type> [connection options]

                                        

-------------------------------------------------------------------------------
                                        
Examples:
                                        
        omniperf database --import -H pavii1 -u amd -t asw -w workloads/vcopy/mi200/
                                        
        omniperf database --remove -H pavii1 -u amd -w omniperf_asw_sample_mi200
                                        
-------------------------------------------------------------------------------

                                        

Help:
  -h, --help             show this help message and exit

General Options:
  -v, --version          show program's version number and exit
  -V, --verbose          Increase output verbosity

Interaction Type:
  -i, --import                                          Import workload to Omniperf DB
  -r, --remove                                          Remove a workload from Omniperf DB

Connection Options:
  -H , --host                                           Name or IP address of the server host.
  -P , --port                                           TCP/IP Port. (DEFAULT: 27018)
  -u , --username                                       Username for authentication.
  -p , --password                                       The user's password. (will be requested later if it's not set)
  -t , --team                                           Specify Team prefix.
  -w , --workload                                       Specify name of workload (to remove) or path to workload (to import)
  -k , --kernelVerbose                                  Specify Kernel Name verbose level 1-5. 
                                                        Lower the level, shorter the kernel name. (DEFAULT: 2) (DISABLE: 5)
```

**omniperf import for vcopy:**
```shell
$ omniperf database --import -H pavii1 -u amd -t asw -w workloads/vcopy/mi200/
ROC Profiler:  /usr/bin/rocprof
 
--------
Import Profiling Results
--------
 
Pulling data from  /home/amd/xlu/test/workloads/vcopy/mi200
The directory exists
Found sysinfo file
KernelName shortening enabled
Kernel name verbose level: 2
Password:
Password recieved
-- Conversion & Upload in Progress --
  0%|                                                                                                                                                                                                             | 0/11 [00:00<?, ?it/s]/home/amd/xlu/test/workloads/vcopy/mi200/SQ_IFETCH_LEVEL.csv
  9%|█████████████████▉                                                                                                                                                                                   | 1/11 [00:00<00:01,  8.53it/s]/home/amd/xlu/test/workloads/vcopy/mi200/pmc_perf.csv
 18%|███████████████████████████████████▊                                                                                                                                                                 | 2/11 [00:00<00:01,  6.99it/s]/home/amd/xlu/test/workloads/vcopy/mi200/SQ_INST_LEVEL_SMEM.csv
 27%|█████████████████████████████████████████████████████▋                                                                                                                                               | 3/11 [00:00<00:01,  7.90it/s]/home/amd/xlu/test/workloads/vcopy/mi200/SQ_LEVEL_WAVES.csv
 36%|███████████████████████████████████████████████████████████████████████▋                                                                                                                             | 4/11 [00:00<00:00,  8.56it/s]/home/amd/xlu/test/workloads/vcopy/mi200/SQ_INST_LEVEL_LDS.csv
 45%|█████████████████████████████████████████████████████████████████████████████████████████▌                                                                                                           | 5/11 [00:00<00:00,  9.00it/s]/home/amd/xlu/test/workloads/vcopy/mi200/SQ_INST_LEVEL_VMEM.csv
 55%|███████████████████████████████████████████████████████████████████████████████████████████████████████████▍                                                                                         | 6/11 [00:00<00:00,  9.24it/s]/home/amd/xlu/test/workloads/vcopy/mi200/sysinfo.csv
 64%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▎                                                                       | 7/11 [00:00<00:00,  9.37it/s]/home/amd/xlu/test/workloads/vcopy/mi200/roofline.csv
 82%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▏                                   | 9/11 [00:00<00:00, 12.60it/s]/home/amd/xlu/test/workloads/vcopy/mi200/timestamps.csv
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 11/11 [00:00<00:00, 11.05it/s]
9 collections added.
Workload name uploaded
-- Complete! --
```
