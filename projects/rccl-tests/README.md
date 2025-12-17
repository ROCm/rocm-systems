# RCCL Tests

These tests check both the performance and the correctness of RCCL operations. They can be compiled against [RCCL](https://github.com/ROCm/rccl).

## Build

To build the tests, just type `make` or `make -j`

If HIP is not installed in `/opt/rocm`, you may specify `HIP_HOME`. Similarly, if RCCL (`librccl.so`) is not installed in `/opt/rocm/lib/`, you may specify `NCCL_HOME` and `CUSTOM_RCCL_LIB`.

```shell
$ make HIP_HOME=/path/to/hip NCCL_HOME=/path/to/rccl
```

RCCL Tests rely on MPI to work on multiple processes, hence multiple nodes.

> [!TIP]
> To compile RCCL tests with MPI support, you need to set `MPI=1` and set `MPI_HOME` to the path where MPI is installed.

```shell
$ make MPI=1 MPI_HOME=/path/to/mpi HIP_HOME=/path/to/hip NCCL_HOME=/path/to/rccl
```

RCCL Tests can also be built using cmake. A typical sequence will be:

```shell
$ mkdir build
$ cd build
$ cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=/path/to/rocm ..
$ make
```

When using the cmake build procedure for building RCCL-Tests with custom/user-built `librccl.so`, please make sure that RCCL has been installed (i.e. using `make install`) and not pointing to the RCCL `build` directory, since cmake will check for cmake target and config files. This is not necessary as one can modify `LD_LIBRARY_PATH` to point to the custom/user-built `librccl.so` when running RCCL Tests.

Using the cmake method also has the advantage that it automatically checks for MPI installation during the build. The tests can be compiled with MPI support by adding the `-DUSE_MPI=ON` flag to the cmake command line.

> [!TIP]
> Users can choose to link against a particular MPI library by using one of these options:
> * setting the environment variable `MPI_HOME`.
> * by adding the path to the MPI library to the cmake prefix path with `-DCMAKE_PREFIX_PATH`.
> * including the paths to MPI `bin` and `lib` in the `PATH` and `LD_LIBRARY_PATH` environment variables, respectively.

e.g.,
```shell
$ mkdir build
$ cd build
$ cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="/path/to/mpi;/path/to/rocm" -DUSE_MPI=ON ..
$ make
```

By default, for both Makefile and `cmake` based builds, RCCL Tests will link against all supported GPU targets (defined in `src/Makefile` and as `DEFAULT_GPUS` in `CMakeLists.txt`).

To target specific GPU(s), and potentially reduce build time, use:
* `GPU_TARGETS` as a `,` separated string listing GPU(s) to target for Makefile based build.
e.g. build RCCL-Tests using Makefile only for `gfx942` and `gfx950`. e.g.,
    ```shell
    $ GPU_TARGETS="gfx942,gfx950" make MPI=1 MPI_HOME=/path/to/mpi NCCL_HOME=/opt/rocm
    ```
* `-DGPU_TARGETS` as a `;` separated string listing GPU(s) to target for `cmake` based build.
e.g. build RCCL-Tests using CMake for `gfx90a`, `gfx942` and `gfx1200`. e.g.,
    ```shell
    $ cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="/path/to/mpi;/path/to/rocm" -DUSE_MPI=ON -DGPU_TARGETS="gfx90a;gfx942;gfx1200;" ..
    ```
* For CMake builds, we also have another flag `DBUILD_LOCAL_GPU_TARGET_ONLY` that queries and builds for the local GPU target only (similar to RCCL).
    ```shell
    $ cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="/path/to/mpi;/path/to/rocm" -DUSE_MPI=ON -DBUILD_LOCAL_GPU_TARGET_ONLY=ON ..
    ```

`-DBUILD_LOCAL_GPU_TARGET_ONLY` will not work with `docker build`-based setups, as the docker build engine is unable to query the local GPU architecture. Please use `-DGPU_TARGETS` for CMake-based builds or `GPU_TARGETS` for Makefile-based builds when building RCCL-Tests using a Dockerfile and `docker build`.

## Usage

RCCL Tests can run on multiple processes, multiple threads, and multiple HIP devices per thread. The number of process is managed by MPI and is therefore not passed to the tests as argument. The total number of ranks (=HIP devices) will be equal to (number of processes)\*(number of threads)\*(number of GPUs per thread).

### Quick examples

Run on single node with 8 GPUs (`-g 8`), scanning from 8 Bytes to 128MBytes :

```shell
$ ./build/all_reduce_perf -b 8 -e 128M -f 2 -g 8
```

Run 64 MPI processes on nodes with 8 GPUs each, for a total of 64 GPUs spread across 8 nodes :
(NB: The rccl-tests binaries must be compiled with `MPI=1` for this case)

```shell
$ mpirun -np 64 -N 8 ./build/all_reduce_perf -b 8 -e 8G -f 2 -g 1
```

> [!TIP]
> For performance-oriented runs, on both single-node and multi-node, we suggest using 1 MPI process per GPU and `-g 1`. So, a run on 8 GPUs looks like :
> ```shell
> $ mpirun -np 8 --bind-to numa ./build/all_reduce_perf -b 8 -e 128M -f 2 -g 1
> ```
> Running with 1 MPI process per GPU ensures a 1:1 mapping for CPUs and GPUs, which can be beneficial for smaller message sizes and better represents the real-world use of RCCL in Deep Learning frameworks like Pytorch and TensorFlow.

### Performance

See the [Performance](doc/PERFORMANCE.md) page for explanation about numbers, and in particular the "busbw" column.

#### Environment variables
On some earlier versions of ROCm (before ROCm 6.4.0), setting `HSA_NO_SCRATCH_RECLAIM=1` as part of the environment is necessary to achieve better performance on MI300 GPUs. When running without MPI, a command similar to the following one should be sufficient:
```shell
HSA_NO_SCRATCH_RECLAIM=1 ./build/all_reduce_perf -b 8 -e 128M -f 2 -g 8
```

For MPI (using MPICH), you need to use a command similar to the following:
```shell
mpirun.mpich -np 8 -env NCCL_DEBUG=VERSION -env HSA_NO_SCRATCH_RECLAIM=1 ./build/all_reduce_perf -b 8M -e 128M -i 8388608 -g 1 -d bfloat16
```

### Arguments

All tests support the same set of arguments :

* Number of GPUs
  * `-t,--nthreads <num threads>` number of threads per process. Default : 1.
  * `-g,--ngpus <GPUs per thread>` number of gpus per thread. Default : 1.
* Sizes to scan
  * `-b,--minbytes <min size in bytes>` minimum size to start with. Default : 32M.
  * `-e,--maxbytes <max size in bytes>` maximum size to end at. Default : 32M.
  * Increments can be either fixed or a multiplication factor. Only one of those should be used
    * `-i,--stepbytes <increment size>` fixed increment between sizes. Default : 1M.
    * `-f,--stepfactor <increment factor>` multiplication factor between sizes. Default : disabled.
* RCCL operations arguments
  * `-o,--op <sum/prod/min/max/avg/all>` Specify which reduction operation to perform. Only relevant for reduction operations like Allreduce, Reduce or ReduceScatter. Default : Sum.
  * `-d,--datatype <nccltype/all>` Specify which datatype to use. Default : Float.
  * `-r,--root <root/all>` Specify which root to use. Only for operations with a root like broadcast or reduce. Default : 0.
  * `-y,--memory_type <coarse/fine/host/managed>` Default: Coarse
  * `-u,--cumask <d0,d1,d2,d3>` Default: None
* Performance
  * `-n,--iters <iteration count>` number of iterations. Default : 20.
  * `-w,--warmup_iters <warmup iteration count>` number of warmup iterations (not timed). Default : 5.
  * `-m,--agg_iters <aggregation count>` number of operations to aggregate together in each iteration. Default : 1.
  * `-N,--run_cycles <cycle count>` run & print each cycle. Default : 1; 0=infinite.
  * `-a,--average <0/1/2/3>` Report performance as an average across all ranks (MPI=1 only). <0=Rank0,1=Avg,2=Min,3=Max>. Default : 1.
* Test operation
  * `-p,--parallel_init <0/1>` use threads to initialize NCCL in parallel. Default : 0.
  * `-c,--check <check iteration count>` perform count iterations, checking correctness of results on each iteration. This can be quite slow on large numbers of GPUs. Default : 1.
  * `-z,--blocking <0/1>` Make RCCL collective blocking, i.e. have CPUs wait and sync after each collective. Default : 0.
  * `-G,--hipgraph <num graph launches>` Capture iterations as a HIP graph and then replay specified number of times. Default : 0.
  * `-C,--report_cputime <0/1>]` Report CPU time instead of latency. Default : 0.
  * `-R,--local_register <0/1/2>` enable local (1) or symmetric (2) buffer registration on send/recv buffers. Default : 0.
  * `-T,--timeout <time in seconds>` timeout each test after specified number of seconds. Default : disabled.
  * `-F,--cache_flush <cache flush after every -F iteration>` Enable cache flush after every -F iteration. Default : 0 (No cache flush).
  * `-O,--out_of_place <0=in-place only, 1=out-of-place only>`. Default: both.
  * `-q,--delay <delay>` Delay between out-of-place and in-place runs (in microseconds). Default: 10.
* Parsing RCCL-Tests output
  * `-Z,--output_format <csv|json>` Parse RCCL-Tests output as a CSV or JSON. Default : disabled.
  * `-x,--output_file <output file name>` RCCL-Tests output file name. Default : disabled.
  * `-M,--output_algo_proto_channels <0/1>` Report Algorithm/Protocol/Channels for each message size. Default : 0.

### Running multiple operations in parallel

RCCL Tests allow to partition the set of GPUs into smaller sets, each executing the same operation in parallel. 
To split the GPUs, RCCL will compute a "color" for each rank, based on the `NCCL_TESTS_SPLIT` environment variable, then all ranks
with the same color will end up in the same group. The resulting group is printed next to each GPU at the beginning of the test.

`NCCL_TESTS_SPLIT` takes the following syntax: `<operation><value>`. Operation can be `AND`, `OR`, `MOD` or `DIV`. The `&`, `|`, `%`, and `/` symbols are also supported. The value can be either decimal, hexadecimal (prefixed by `0x`) or binary (prefixed by `0b`).

`NCCL_TESTS_SPLIT_MASK="<value>"` is equivalent to `NCCL_TESTS_SPLIT="&<value>"`.

Here are a few examples:

 - `NCCL_TESTS_SPLIT="AND 0x7"` or `NCCL_TESTS_SPLIT="MOD 8"`: On systems with 8 GPUs, run 8 parallel operations, each with 1 GPU per node (purely communicating over the inter-node network)

- `NCCL_TESTS_SPLIT="OR 0x7"` or `NCCL_TESTS_SPLIT="DIV 8"`: On systems with 8 GPUs, run one operation per node, purely intra-node.

- `NCCL_TESTS_SPLIT="AND 0x1"` or `NCCL_TESTS_SPLIT="MOD 2"`: Run two operations, each operation using every other rank.

Note that the reported bandwidth is per group, hence to get the total bandwidth used by all groups, one must multiply by the number of groups.

## Unit tests

Unit tests for rccl-tests are implemented with pytest (python3 is also required). Several notes for the unit tests:

1. The `LD_LIBRARY_PATH` environment variable will need to be set to include `/path/to/rccl-install/lib/` in order to run the unit tests.
2. The `HSA_FORCE_FINE_GRAIN_PCIE` environment variable will need to be set to 1 in order to run the unit tests which use fine-grained memory type.

The unit tests can be invoked within the rccl-tests root, or in the test subfolder. An example call to the unit tests:
```shell
$ LD_LIBRARY_PATH=/path/to/rccl-install/lib/ HSA_FORCE_FINE_GRAIN_PCIE=1 python3 -m pytest
```

## Copyright

NCCL tests are provided under the BSD license. All source code and accompanying documentation is copyright (c) 2016-2025, NVIDIA CORPORATION. All rights reserved.

All modifications are copyright (c) 2019-2025 Advanced Micro Devices, Inc. All rights reserved.
