# ROCProfiler PC sampling example code

The ROCProfiler library includes an API to enable periodic sampling of the GPU
program counter during kernel execution.  This program demonstrates the PC
sampling API, with additional code to illustrate a typical non-trivial use case:
correlation of sampled PC addresses with their disassembled machine code, as
well as source code and symbolic debugging information if available.

## Building the demo program

If your ROCm installation already includes ROCProfiler, the only requirements to
build the demo program are:

* GNU `make`
* libdw (**not** libdwarf)
* libelf

If ROCm is installed in the standard location (`/opt/rocm`), running `make` in
the same directory as this README should work; otherwise, set `ROCM_PATH` to the
location of the ROCm installation in your environment and `ROCPROFILER_PATH` to
the location of the ROCProfiler source repo before running `make`.

If your ROCm installation does **not** include ROCProfiler, you will need to build
it yourself.  This demo program will be built as part of that process.  See the
main ROCProfiler README for additional requirements and directions.

## Running the demo program

The demo program simply fills a vector with random 64-bit unsigned integers and
tallies the count of those greater than the mandatory `MIN` argument:

```
usage: code_printing_sample [OPTION]... MIN [SEED]
  -d DEV        HIP device number
  -n LEN        Length of random integer array
  -D            Print kernel disassembly
  -P            Print source and disassembly of sampled PC locations
where
  DEV : i32
  MIN : u64
  LEN : u64
  SEED : u64
```

### Defaults and troubleshooting

* `-d`: use HIP device 0
* `-n`: 4194304 (1024 * 1024 * 4)
* `-D`: false
* `-P`: false
* `SEED`: random seed; taken from the system's monotonic clock

The program contains two trivial GPU kernels: an implementation of `memset`, and
the parallel counting procedure.  Because the actual point is to demonstrate the
PC sampling functionality, it is recommended to use the `-n` option with an
argument such that the allocated vector fits in the smaller of available host as
well as device memory, but sufficiently large argument such that the kernels run
long enough for ROCProfiler to actually collect some samples.

In order for the `-P` option to display source, the demo program must have been
built with debug symbols (at least `-gdwarf-4`).  Any optimization level is
fine, but if the kernels run too quickly for ROCProfiler to collect any samples
even when a very large vector is given with the `-n` option, try rebuilding the
demo program without optimizations by adding `-O0` to the `hipcc` compilation
flags.

## Files

* `main.cpp`: initializes ROCProfiler and PC sampling and runs the GPU kernels
* `code_printing.cpp`: inspects the ELF and DWARF info for the GPU programs
  embedded in the host binary and uses amd-dbgapi to print disassembly and
  source
* `disassembly.cpp`: wrapper for `code_printing.cpp`

## PC sampling API

Adding PC sampling to a program already using the ROCProfiler API requires only
two changes:

1. Call `rocprofiler_create_filter` to create a `ROCPROFILER_PC_SAMPLING_COLLECTION`
   filter, then `rocprofiler_set_filter_buffer` to add the filter to the desired
   buffer (see functions `main` and `run_kernel` in `main.cpp`)

2. Handle records of kind `ROCPROFILER_PC_SAMPLING_RECORD` in the buffer callback
   function.  These should be cast to `rocprofiler_record_pc_sample_t *` (see
   function `callback_flush_fn` in `main.cpp`)

Like all ROCProfiler records, PC sample records contain a standard header followed
by one or more payloads:

```c
/**
 * PC sample record: contains the program counter/instruction pointer observed
 * during periodic sampling of a kernel
 */
typedef struct {
  /**
   * ROCMtool General Record base header to identify the id and kind of every
   * record
   */
  rocprofiler_record_header_t header;
  /**
   * PC sample data
   */
  rocprofiler_pc_sample_t pc_sample;
} rocprofiler_record_pc_sample_t;
```

PC samples are delivered via the normal ROCProfiler buffer callback mechanism,
along with some additional information allowing each sample to be associated
with a unique, individual kernel execution:

```c
/**
 * An individual PC sample
 */
typedef struct {
  /**
   * Kernel dispatch ID.  This is used by PC sampling to associate samples with
   * individual dispatches and is unrelated to any user-supplied correlation ID
   */
  rocprofiler_kernel_dispatch_id_t dispatch_id;
  union {
    /**
     * Host timestamp
     */
    rocprofiler_timestamp_t timestamp;
    /**
     * GPU clock counter (not currently used)
     */
    uint64_t cycle;
  };
  /**
   * Sampled program counter
   */
  uint64_t pc;
  /**
   * Sampled shader element
   */
  uint32_t se;
  /**
   * Sampled GPU agent
   */
  rocprofiler_agent_id_t gpu_id;
} rocprofiler_pc_sample_t;
```

PC sampling is started and stopped with `rocprofiler_start_session` and
`rocprofiler_terminate_session`, just like other profiling activities.
