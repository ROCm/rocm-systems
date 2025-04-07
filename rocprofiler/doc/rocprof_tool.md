# rocprof
## 1. Overview
The rocProf is a command line tool implemented on the top of  rocProfiler and rocTracer APIs. Source code for rocProf may be found here:
GitHub: https://github.com/ROCm-Developer-Tools/rocprofiler/blob/amd-master/bin/rocprof
This command line tool is implemented as a script which is setting up the environment for attaching the profiler and then run the provided application command line. The tool uses two profiling plugins loaded by ROC runtime and based on rocProfiler and rocTracer for collecting metrics/counters, HW traces and runtime API/activity traces. The tool consumes an input XML or text file with counters list or trace parameters and provides output profiling data and statistics in various formats as text, CSV and JSON traces. Google Chrome tracing can be used to visualize the JSON traces with runtime API/activity timelines and per kernel counters data.
## 2. Profiling Modes
‘rocprof’ can be used for GPU profiling using HW counters and application tracing
### 2.1. GPU profiling
GPU profiling is controlled with input file which defines a list of metrics/counters and a profiling scope. An input file is provided using option ‘-i [input file]’. Output CSV file with a line per submitted kernel is generated. Each line has kernel name, kernel parameters and counter values. By option ‘—stats’ the kernel execution stats can be generated in CSV format. Currently profiling has limitation of serializing submitted kernels.
An example of input file:
```
   # Perf counters group 1
   pmc : Wavefronts VALUInsts SALUInsts SFetchInsts
   # Perf counters group 2
   pmc : TCC_HIT[0], TCC_MISS[0]
   # Filter by dispatches range, GPU index and kernel names
   # supported range formats: "3:9", "3:", "3"
   range: 1 : 4
   gpu: 0 1 2 3
   kernel: simple Pass1 simpleConvolutionPass2
```
An example of profiling command line for ‘MatrixTranspose’ application
```
$ rocprof -i input.txt MatrixTranspose
RPL: on '191018_011134' from '/…./rocprofiler_pkg' in '/…./MatrixTranspose'
RPL: profiling '"./MatrixTranspose"'
RPL: input file 'input.txt'
RPL: output dir '/tmp/rpl_data_191018_011134_9695'
RPL: result dir '/tmp/rpl_data_191018_011134_9695/input0_results_191018_011134'
ROCProfiler: rc-file '/…./rpl_rc.xml'
ROCProfiler: input from "/tmp/rpl_data_191018_011134_9695/input0.xml"
  gpu_index =
  kernel =
  range =
  4 metrics
    L2CacheHit, VFetchInsts, VWriteInsts, MemUnitStalled
  0 traces
Device name Ellesmere [Radeon RX 470/480/570/570X/580/580X]
PASSED!

ROCprofiler: 1 contexts collected, output directory /tmp/rpl_data_191018_011134_9695/input0_results_191018_011134
RPL: '/…./MatrixTranspose/input.csv' is generated
```
#### 2.1.1.	Counters and metrics
There are two profiling features, metrics and traces. Hardware performance counters are treated as the basic metrics and the formulas can be defined for derived metrics.
Counters and metrics can be dynamically configured using XML configuration files with counters and metrics tables:
 - Counters table entry, basic metric: counter name, block name, event id
 - Derived metrics table entry: metric name, an expression for calculation the metric from the counters

Metrics XML File Example:
```
<gfx8>
	<metric name=L1_CYCLES_COUNTER block=L1 event=0 descr=”L1 cache cycles”></metric>
	<metric name=L1_MISS_COUNTER block=L1 event=33 descr=”L1 cache misses”></metric>
	. . .
</gfx8>

<gfx9>
	. . .
</gfx9>

<global>
  <metric
    name=L1_MISS_RATIO
    expr=L1_CYCLES_COUNT/L1_MISS_COUNTER
    descry=”L1 miss rate metric”
  ></metric>
</global>
```
##### 2.1.1.1. Metrics query
Available counters and metrics can be queried by options ‘—list-basic’ for counters and ‘—list-derived’ for derived metrics. The output for counters indicates number of block instances and number of block counter registers. The output for derived metrics prints the metrics expressions.
Examples:
```
$ rocprof --list-basic
RPL: on '191018_014450' from '/opt/rocm/rocprofiler' in '/…./MatrixTranspose'
ROCProfiler: rc-file '/…./rpl_rc.xml'
Basic HW counters:
  gpu-agent0 : GRBM_COUNT : Tie High - Count Number of Clocks
      block GRBM has 2 counters
  gpu-agent0 : GRBM_GUI_ACTIVE : The GUI is Active
      block GRBM has 2 counters
	  . . .
  gpu-agent0 : TCC_HIT[0-15] : Number of cache hits.
      block TCC has 4 counters
  gpu-agent0 : TCC_MISS[0-15] : Number of cache misses. UC reads count as misses.
      block TCC has 4 counters
	  . . .

$ rocprof --list-derived
RPL: on '191018_015911' from '/opt/rocm/rocprofiler' in '/home/evgeny/work/BUILD/0_MatrixTranspose'
ROCProfiler: rc-file '/home/evgeny/rpl_rc.xml'
Derived metrics:
  gpu-agent0 : TCC_HIT_sum : Number of cache hits. Sum over TCC instances.
      TCC_HIT_sum = sum(TCC_HIT,16)
  gpu-agent0 : TCC_MISS_sum : Number of cache misses. Sum over TCC instances.
      TCC_MISS_sum = sum(TCC_MISS,16)
  gpu-agent0 : TCC_MC_RDREQ_sum : Number of 32-byte reads. Sum over TCC instances.
      TCC_MC_RDREQ_sum = sum(TCC_MC_RDREQ,16)
	. . .
```
##### 2.1.1.2.	Metrics collecting
Counters and metrics accumulated per kernel can be collected using input file with a list of metrics, see an example in 2.1.
Currently profiling has limitation of serializing submitted kernels.
The number of counters which can be dumped by one run is limited by GPU HW by number of counter registers per block. The number of counters can be different for different blocks and can be queried, see 2.1.1.1.
###### 2.1.1.2.1.	Blocks instancing
GPU blocks are implemented as several identical instances. To dump counters of specific instance square brackets can be used, see an example in 2.1.
The number of block instances can be queried, see 2.1.1.1.
###### 2.1.1.2.2.	HW limitations
The number of counters which can be dumped by one run is limited by GPU HW by number of counter registers per block. The number of counters can be different for different blocks and can be queried, see 2.1.1.1.
 - Metrics groups

To dump a list of metrics exceeding HW limitations the metrics list can be split on groups.
The tool supports automatic splitting on optimal metric groups:
```
$ rocprof -i input.txt ./MatrixTranspose
RPL: on '191018_032645' from '/opt/rocm/rocprofiler' in '/…./MatrixTranspose'
RPL: profiling './MatrixTranspose'
RPL: input file 'input.txt'
RPL: output dir '/tmp/rpl_data_191018_032645_12106'
RPL: result dir '/tmp/rpl_data_191018_032645_12106/input0_results_191018_032645'
ROCProfiler: rc-file '/…./rpl_rc.xml'
ROCProfiler: input from "/tmp/rpl_data_191018_032645_12106/input0.xml"
  gpu_index =
  kernel =
  range =
  20 metrics
    Wavefronts, VALUInsts, SALUInsts, SFetchInsts, FlatVMemInsts, LDSInsts, FlatLDSInsts, GDSInsts, VALUUtilization, FetchSize, WriteSize, L2CacheHit, VWriteInsts, GPUBusy, VALUBusy, SALUBusy, MemUnitStalled, WriteUnitStalled, LDSBankConflict, MemUnitBusy
  0 traces
Device name Ellesmere [Radeon RX 470/480/570/570X/580/580X]

Input metrics out of HW limit. Proposed metrics group set:
 group1: L2CacheHit VWriteInsts MemUnitStalled WriteUnitStalled MemUnitBusy FetchSize FlatVMemInsts LDSInsts VALUInsts SALUInsts SFetchInsts FlatLDSInsts GPUBusy Wavefronts
 group2: WriteSize GDSInsts VALUUtilization VALUBusy SALUBusy LDSBankConflict

ERROR: rocprofiler_open(), Construct(), Metrics list exceeds HW limits

Aborted (core dumped)
Error found, profiling aborted.
```
 - Collecting with multiple runs

To collect several metric groups a full application replay is used by defining several ‘pmc:’ lines in the input file, see 2.1.

### 2.2.	Application tracing
Supported application tracing includes runtime API and GPU activity tracing’
Supported runtimes are: ROCr (HSA API) and HIP
Supported GPU activity: kernel execution, async memory copy, barrier packets.
The trace is generated in JSON format compatible with Chrome tracing.
The trace consists of several sections with timelines for API trace per thread and GPU activity. The timelines events show event name and parameters.
Supported options: ‘—hsa-trace’, ‘—hip-trace’, ‘—sys-trace’, where ‘sys trace’ is for HIP and HSA combined trace.
#### 2.2.1.	HIP runtime trace
The trace is generated by option ‘—hip-trace’ and includes HIP API timelines and GPU activity at the runtime level.
#### 2.2.2.	ROCr runtime trace
The trace is generated by option ‘—hsa-trace’ and includes ROCr API timelines and GPU activity at AQL queue level. Also, can provide counters per kernel.
#### 2.2.3.	KFD driver trace
The trace is generated by option ‘—kfd-trace’ and includes KFD Thunk API timeline.
It is planned to add memory allocations/migration tracing.
#### 2.2.4.	Code annotation
Support for application code annotation.
Start/stop API is supported to programmatically control the profiling.
A ‘roctx’ library provides annotation API. Annotation is visualized in JSON trace as a separate "Markers and Ranges" timeline section.
##### 2.2.4.1.	Start/stop API
```
// Tracing start API
void roctracer_start();

// Tracing stop API
void roctracer_stop();
```
##### 2.2.4.2.	rocTX basic markers API
```
// A marker created by given ASCII massage
void roctxMark(const char* message);

// Returns the 0 based level of a nested range being started by given message associated to this range.
// A negative value is returned on the error.
int roctxRangePush(const char* message);

// Marks the end of a nested range.
// Returns the 0 based level the range.
// A negative value is returned on the error.
int roctxRangePop();
```
### 2.3.	Multiple GPUs profiling
The profiler supports multiple GPU’s profiling and provide GPI id for counters and kernels data in CSV output file. Also, GPU id is indicating for respective GPU activity timeline in JSON trace.
## 3.	Profiling control
Profiling can be controlled by specifying a profiling scope, by filtering trace events and specifying interesting time intervals.
### 3.1.	Profiling scope
Counters profiling scope can be specified by GPU id list, kernel name substrings list and dispatch range.
Supported range formats examples: "3:9", "3:", "3". You can see an example of input file in 2.1.
#### 3.2.	Tracing control
Tracing can be filtered by events names using profiler input file and by enabling interesting time intervals by command line option.
#### 3.2.1.	Filtering traced APIs
A list of traced API names can be specified in profiler input file.
An example of input file line for ROCr runtime trace (HAS API):
```
hsa: hsa_queue_create hsa_amd_memory_pool_allocate
```
#### 3.2.2.	Tracing time period
Trace can be dumped periodically with initial delay, dumping period length and rate:
```
--trace-period <delay:length:rate>
```
### 3.3.	Concurrent kernels
Currently concurrent kernels profiling is not supported which is a planned feature. Kernels are serialized.
### 3.4.	Multi-processes profiling
Multi-processes profiling is not currently supported.
### 3.5.	Errors logging
Profiler errors are logged to global logs:
```
/tmp/aql_profile_log.txt
/tmp/rocprofiler_log.txt
/tmp/roctracer_log.txt
```
## 4.	3rd party visualization tools
‘rocprof’ is producing JSON trace compatible with Chrome Tracing, which is an internal trace visualization tool in Google Chrome.
### 4.1.	Chrome tracing
Good review can be found by the link: https://aras-p.info/blog/2017/01/23/Chrome-Tracing-as-Profiler-Frontend/
## 5.	Command line options
The command line options can be printed with option ‘-h’:
```
$ rocprof -h
RPL: on '191018_023018' from '/opt/rocm/rocprofiler' in '/…./MatrixTranspose'
ROCm Profiling Library (RPL) run script, a part of ROCprofiler library package.
Full path: /opt/rocm/rocprofiler/bin/rocprof
Metrics definition: /opt/rocm/rocprofiler/lib/metrics.xml

Usage:
  rocprof [-h] [--list-basic] [--list-derived] [-i <input .txt/.xml file>] [-o <output CSV file>] <app command line>

Options:
  -h - this help
  --verbose - verbose mode, dumping all base counters used in the input metrics
  --list-basic - to print the list of basic HW counters
  --list-derived - to print the list of derived metrics with formulas
  --cmd-qts <on|off> - quoting profiled cmd-line [on]

  -i <.txt|.xml file> - input file
      Input file .txt format, automatically rerun application for every pmc line:

        # Perf counters group 1
        pmc : Wavefronts VALUInsts SALUInsts SFetchInsts FlatVMemInsts LDSInsts FlatLDSInsts GDSInsts VALUUtilization FetchSize
        # Perf counters group 2
        pmc : WriteSize L2CacheHit
        # Filter by dispatches range, GPU index and kernel names
        # supported range formats: "3:9", "3:", "3"
        range: 1 : 4
        gpu: 0 1 2 3
        kernel: simple Pass1 simpleConvolutionPass2

      Input file .xml format, for single profiling run:

        # Metrics list definition, also the form "<block-name>:<event-id>" can be used
        # All defined metrics can be found in the 'metrics.xml'
        # There are basic metrics for raw HW counters and high-level metrics for derived counters
        <metric name=SQ:4,SQ_WAVES,VFetchInsts
        ></metric>

        # Filter by dispatches range, GPU index and kernel names
        <metric
          # range formats: "3:9", "3:", "3"
          range=""
          # list of gpu indexes "0,1,2,3"
          gpu_index=""
          # list of matched sub-strings "Simple1,Conv1,SimpleConvolution"
          kernel=""
        ></metric>

  -o <output file> - output CSV file [<input file base>.csv]
  -d <data directory> - directory where profiler store profiling data including traces [/tmp]
      The data directory is removed automatically if the directory is matching the temporary one, which is the default.
  -t <temporary directory> - to change the temporary directory [/tmp]
      By changing the temporary directory you can prevent removing the profiling data from /tmp or enable removing from not '/tmp' directory.

  --basenames <on|off> - to turn on/off truncating of the kernel full function names till the base ones [off]
  --timestamp <on|off> - to turn on/off the kernel dispatches timestamps, dispatch/begin/end/complete [off]
  --ctx-wait <on|off> - to wait for outstanding contexts on profiler exit [on]
  --ctx-limit <max number> - maximum number of outstanding contexts [0 - unlimited]
  --heartbeat <rate sec> - to print progress heartbeats [0 - disabled]
  --obj-tracking <on|off> - to turn on/off kernels code objects tracking [off]

  --stats - generating kernel execution stats, file <output name>.stats.csv

  --roctx-trace - to enable rocTX application code annotation trace, "Markers and Ranges" JSON trace section.
  --sys-trace - to trace HIP/HSA APIs and GPU activity, generates stats and JSON trace chrome-tracing compatible
  --hip-trace - to trace HIP, generates API execution stats and JSON file chrome-tracing compatible
  --hsa-trace - to trace HSA, generates API execution stats and JSON file chrome-tracing compatible
  --kfd-trace - to trace KFD, generates API execution stats and JSON file chrome-tracing compatible
    Generated files: <output name>.<domain>_stats.txt <output name>.json
    Traced API list can be set by input .txt or .xml files.
    Input .txt:
      hsa: hsa_queue_create hsa_amd_memory_pool_allocate
    Input .xml:
      <trace name="HSA">
        <parameters list="hsa_queue_create, hsa_amd_memory_pool_allocate">
        </parameters>
      </trace>

  --trace-start <on|off> - to enable tracing on start [on]
  --trace-period <delay:length:rate> - to enable trace with initial delay, with periodic sample length and rate
    Supported time formats: <number(m|s|ms|us)>

Configuration file:
  You can set your parameters defaults preferences in the configuration file 'rpl_rc.xml'. The search path sequence: .:/home/user:<package path>
  First the configuration file is looking in the current directory, then in your home, and then in the package directory.
  Configurable options: 'basenames', 'timestamp', 'ctx-limit', 'heartbeat', 'obj-tracking'.
  An example of 'rpl_rc.xml':
    <defaults
      basenames=off
      timestamp=off
      ctx-limit=0
      heartbeat=0
      obj-tracking=off
    ></defaults>
```
## 6.	Publicly available counters and metrics
The following counters are publicly available for commercially available VEGA10/20 GPUs.

Counters:
```
•	GRBM_COUNT : Tie High - Count Number of Clocks
•	GRBM_GUI_ACTIVE : The GUI is Active
•	SQ_WAVES : Count number of waves sent to SQs. (per-simd, emulated, global)
•	SQ_INSTS_VALU : Number of VALU instructions issued. (per-simd, emulated)
•	SQ_INSTS_VMEM_WR : Number of VMEM write instructions issued (including FLAT). (per-simd, emulated)
•	SQ_INSTS_VMEM_RD : Number of VMEM read instructions issued (including FLAT). (per-simd, emulated)
•	SQ_INSTS_SALU : Number of SALU instructions issued. (per-simd, emulated)
•	SQ_INSTS_SMEM : Number of SMEM instructions issued. (per-simd, emulated)
•	SQ_INSTS_FLAT : Number of FLAT instructions issued. (per-simd, emulated)
•	SQ_INSTS_FLAT_LDS_ONLY : Number of FLAT instructions issued that read/wrote only from/to LDS (only works if EARLY_TA_DONE is enabled). (per-simd, emulated)
•	SQ_INSTS_LDS : Number of LDS instructions issued (including FLAT). (per-simd, emulated)
•	SQ_INSTS_GDS : Number of GDS instructions issued. (per-simd, emulated)
•	SQ_WAIT_INST_LDS : Number of wave-cycles spent waiting for LDS instruction issue. In units of 4 cycles. (per-simd, nondeterministic)
•	SQ_ACTIVE_INST_VALU : regspec 71? Number of cycles the SQ instruction arbiter is working on a VALU instruction. (per-simd, nondeterministic)
•	SQ_INST_CYCLES_SALU : Number of cycles needed to execute non-memory read scalar operations. (per-simd, emulated)
•	SQ_THREAD_CYCLES_VALU : Number of thread-cycles used to execute VALU operations (similar to INST_CYCLES_VALU but multiplied by # of active threads). (per-simd)
•	SQ_LDS_BANK_CONFLICT : Number of cycles LDS is stalled by bank conflicts. (emulated)
•	TA_TA_BUSY[0-15] : TA block is busy. Perf_Windowing not supported for this counter.
•	TA_FLAT_READ_WAVEFRONTS[0-15] : Number of flat opcode reads processed by the TA.
•	TA_FLAT_WRITE_WAVEFRONTS[0-15] : Number of flat opcode writes processed by the TA.
•	TCC_HIT[0-15] : Number of cache hits.
•	TCC_MISS[0-15] : Number of cache misses. UC reads count as misses.
•	TCC_EA_WRREQ[0-15] : Number of transactions (either 32-byte or 64-byte) going over the TC_EA_wrreq interface. Atomics may travel over the same interface and are generally classified as write requests. This does not include probe commands.
•	TCC_EA_WRREQ_64B[0-15] : Number of 64-byte transactions going (64-byte write or CMPSWAP) over the TC_EA_wrreq interface.
•	TCC_EA_WRREQ_STALL[0-15] : Number of cycles a write request was stalled.
•	TCC_EA_RDREQ[0-15] : Number of TCC/EA read requests (either 32-byte or 64-byte)
•	TCC_EA_RDREQ_32B[0-15] : Number of 32-byte TCC/EA read requests
•	TCP_TCP_TA_DATA_STALL_CYCLES[0-15] : TCP stalls TA data interface. Now Windowed.
```

The following derived metrics have been defined and the profiler metrics XML specification can be found at: https://github.com/ROCm-Developer-Tools/rocprofiler/blob/amd-master/test/tool/metrics.xml.

Metrics:
```
•	TA_BUSY_avr : TA block is busy. Average over TA instances.
•	TA_BUSY_max : TA block is busy. Max over TA instances.
•	TA_BUSY_min : TA block is busy. Min over TA instances.
•	TA_FLAT_READ_WAVEFRONTS_sum : Number of flat opcode reads processed by the TA. Sum over TA instances.
•	TA_FLAT_WRITE_WAVEFRONTS_sum : Number of flat opcode writes processed by the TA. Sum over TA instances.
•	TCC_HIT_sum : Number of cache hits. Sum over TCC instances.
•	TCC_MISS_sum : Number of cache misses. Sum over TCC instances.
•	TCC_EA_RDREQ_32B_sum : Number of 32-byte TCC/EA read requests. Sum over TCC instances.
•	TCC_EA_RDREQ_sum : Number of TCC/EA read requests (either 32-byte or 64-byte). Sum over TCC instances.
•	TCC_EA_WRREQ_sum : Number of transactions (either 32-byte or 64-byte) going over the TC_EA_wrreq interface. Sum over TCC instances.
•	TCC_EA_WRREQ_64B_sum : Number of 64-byte transactions going (64-byte write or CMPSWAP) over the TC_EA_wrreq interface. Sum over TCC instances.
•	TCC_WRREQ_STALL_max : Number of cycles a write request was stalled. Max over TCC instances.
•	TCC_MC_WRREQ_sum : Number of 32-byte effective writes. Sum over TCC instaces.
•	FETCH_SIZE : The total kilobytes fetched from the video memory. This is measured with all extra fetches and any cache or memory effects taken into account.
•	WRITE_SIZE : The total kilobytes written to the video memory. This is measured with all extra fetches and any cache or memory effects taken into account.
•	GPUBusy : The percentage of time GPU was busy.
•	Wavefronts : Total wavefronts.
•	VALUInsts : The average number of vector ALU instructions executed per work-item (affected by flow control).
•	SALUInsts : The average number of scalar ALU instructions executed per work-item (affected by flow control).
•	VFetchInsts : The average number of vector fetch instructions from the video memory executed per work-item (affected by flow control). Excludes FLAT instructions that fetch from video memory.
•	SFetchInsts : The average number of scalar fetch instructions from the video memory executed per work-item (affected by flow control).
•	VWriteInsts : The average number of vector write instructions to the video memory executed per work-item (affected by flow control). Excludes FLAT instructions that write to video memory.
•	FlatVMemInsts : The average number of FLAT instructions that read from or write to the video memory executed per work item (affected by flow control). Includes FLAT instructions that read from or write to scratch.
•	LDSInsts : The average number of LDS read or LDS write instructions executed per work item (affected by flow control).  Excludes FLAT instructions that read from or write to LDS.
•	FlatLDSInsts : The average number of FLAT instructions that read or write to LDS executed per work item (affected by flow control).
•	GDSInsts : The average number of GDS read or GDS write instructions executed per work item (affected by flow control).
•	VALUUtilization : The percentage of active vector ALU threads in a wave. A lower number can mean either more thread divergence in a wave or that the work-group size is not a multiple of 64. Value range: 0% (bad), 100% (ideal - no thread divergence).
•	VALUBusy : The percentage of GPUTime vector ALU instructions are processed. Value range: 0% (bad) to 100% (optimal).
•	SALUBusy : The percentage of GPUTime scalar ALU instructions are processed. Value range: 0% (bad) to 100% (optimal).
•	Mem32Bwrites :
•	FetchSize : The total kilobytes fetched from the video memory. This is measured with all extra fetches and any cache or memory effects taken into account.
•	WriteSize : The total kilobytes written to the video memory. This is measured with all extra fetches and any cache or memory effects taken into account.
•	L2CacheHit : The percentage of fetch, write, atomic, and other instructions that hit the data in L2 cache. Value range: 0% (no hit) to 100% (optimal).
•	MemUnitBusy : The percentage of GPUTime the memory unit is active. The result includes the stall time (MemUnitStalled). This is measured with all extra fetches and writes and any cache or memory effects taken into account. Value range: 0% to 100% (fetch-bound).
•	MemUnitStalled : The percentage of GPUTime the memory unit is stalled. Try reducing the number or size of fetches and writes if possible. Value range: 0% (optimal) to 100% (bad).
•	WriteUnitStalled : The percentage of GPUTime the Write unit is stalled. Value range: 0% to 100% (bad).
•	ALUStalledByLDS : The percentage of GPUTime ALU units are stalled by the LDS input queue being full or the output queue being not ready. If there are LDS bank conflicts, reduce them. Otherwise, try reducing the number of LDS accesses if possible. Value range: 0% (optimal) to 100% (bad).
•	LDSBankConflict : The percentage of GPUTime LDS is stalled by bank conflicts. Value range: 0% (optimal) to 100% (bad).
```
