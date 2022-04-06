# Instrumenting with Omnitrace

```eval_rst
.. toctree::
   :glob:
   :maxdepth: 4
```

## omnitrace Executable

Instrumentation is performed with the `omnitrace` executable. View the help menu with the `-h` / `--help` option:

```shell
$ omnitrace --help
[omnitrace] Usage: omnitrace   [ --help (count: 0, dtype: bool)
                                 --debug (max: 1, dtype: bool)
                                 --verbose (max: 1, dtype: bool)
                                 --error (max: 1, dtype: boolean)
                                 --simulate (max: 1, dtype: bool)
                                 --print-format (min: 1, dtype: string)
                                 --print-dir (count: 1, dtype: string)
                                 --print-available (count: 1)
                                 --print-instrumented (count: 1)
                                 --print-excluded (count: 1)
                                 --print-overlapping (count: 1)
                                 --output (count: 1)
                                 --pid (count: 1, dtype: int)
                                 --mode (count: 1)
                                 --command (count: 1)
                                 --prefer (count: 1)
                                 --library (count: unlimited)
                                 --main-function (count: 1)
                                 --driver (max: 1, dtype: boolean)
                                 --load (count: unlimited, dtype: string)
                                 --load-instr (count: unlimited, dtype: filepath)
                                 --init-functions (count: unlimited, dtype: string)
                                 --fini-functions (count: unlimited, dtype: string)
                                 --function-include (count: unlimited)
                                 --function-exclude (count: unlimited)
                                 --module-include (count: unlimited)
                                 --module-exclude (count: unlimited)
                                 --label (count: unlimited, dtype: string)
                                 --default-components (count: unlimited, dtype: string)
                                 --env (count: unlimited)
                                 --mpi (max: 1, dtype: bool)
                                 --instrument-loops (max: 1, dtype: boolean)
                                 --min-address-range (count: 1, dtype: int)
                                 --min-address-range-loop (count: 1, dtype: int)
                                 --dynamic-callsites (max: 1, dtype: boolean)
                                 --traps (max: 1, dtype: bool)
                                 --loop-traps (max: 1, dtype: bool)
                                 --allow-overlapping (count: 0, dtype: bool)
                                 --batch-size (count: 1, dtype: int)
                                 --dyninst-options (count: unlimited)
                               ] -- <CMD> <ARGS>

Options:
    -h, -?, --help                 Shows this page

    [DEBUG OPTIONS]

    --debug                        Debug output
    -v, --verbose                  Verbose output
    -e, --error                    All warnings produce runtime errors
    --simulate                     Exit after outputting diagnostic {available,instrumented,excluded,overlapping} module
                                   function lists, e.g. available-instr.txt
    --print-format [ json | txt | xml ]
                                   Output format for diagnostic {available,instrumented,excluded,overlapping} module
                                   function lists, e.g. {print-dir}/available-instr.txt
    --print-dir                    Output directory for diagnostic {available,instrumented,excluded,overlapping} module
                                   function lists, e.g. {print-dir}/available-instr.txt
    --print-available [ functions | functions+ | modules | pair | pair+ ]
                                   Print the available entities for instrumentation (functions, modules, or module-function
                                   pair) to stdout applying regular expressions and exit
    --print-instrumented [ functions | functions+ | modules | pair | pair+ ]
                                   Print the instrumented entities (functions, modules, or module-function pair) to stdout
                                   after applying regular expressions and exit
    --print-excluded [ functions | functions+ | modules | pair | pair+ ]
                                   Print the entities for instrumentation (functions, modules, or module-function pair)
                                   which are excluded from the instrumentation to stdout after applying regular expressions
                                   and exit
    --print-overlapping [ functions | functions+ | modules | pair | pair+ ]
                                   Print the entities for instrumentation (functions, modules, or module-function pair)
                                   which overlap other function calls or have multiple entry points to stdout applying
                                   regular expressions and exit

    [MODE OPTIONS]

    -o, --output                   Enable generation of a new executable (binary-rewrite)
    -p, --pid                      Connect to running process
    -M, --mode [ sampling | trace ]
                                   Instrumentation mode. 'trace' mode instruments the selected functions, 'sampling' mode
                                   only instruments the main function to start and stop the sampler.
    -c, --command                  Input executable and arguments (if '-- <CMD>' not provided)

    [LIBRARY OPTIONS]

    --prefer [ shared | static ]   Prefer this library types when available
    -L, --library                  Libraries with instrumentation routines (default: "libomnitrace")
    -m, --main-function            The primary function to instrument around, e.g. 'main'
    --driver                       Force main or _init/_fini instrumentation
    --load                         Supplemental instrumentation library names w/o extension (e.g. 'libinstr' for
                                   'libinstr.so' or 'libinstr.a')
    --load-instr                   Load {available,instrumented,excluded,overlapping}-instr JSON or XML file(s) and override
                                   what is read from the binary
    --init-functions               Initialization function(s) for supplemental instrumentation libraries (see '--load'
                                   option)
    --fini-functions               Finalization function(s) for supplemental instrumentation libraries (see '--load' option)

    [SYMBOL SELECTION OPTIONS]

    -I, -R, --function-include     Regex for selecting functions
    -E, --function-exclude         Regex for excluding functions
    -MI, -MR, --module-include     Regex for selecting modules/files/libraries
    -ME, --module-exclude          Regex for excluding modules/files/libraries

    [RUNTIME OPTIONS]

    --label [ args | file | line | return ]
                                   Labeling info for functions. By default, just the function name is recorded. Use these
                                   options to gain more information about the function signature or location of the
                                   functions
    -d, --default-components       Default components to instrument (only useful when timemory is enabled in omnitrace
                                   library)
    --env                          Environment variables to add to the runtime in form VARIABLE=VALUE. E.g. use '--env
                                   OMNITRACE_USE_TIMEMORY=ON' to default to using timemory instead of perfetto
    --mpi                          Enable MPI support (requires omnitrace built w/ MPI and GOTCHA support). NOTE: this will
                                   automatically be activated if MPI_Init/MPI_Init_thread and MPI_Finalize are found in the
                                   symbol table of target

    [GRANULARITY OPTIONS]

    -l, --instrument-loops         Instrument at the loop level
    -r, --min-address-range        If the address range of a function is less than this value, exclude it from
                                   instrumentation
    --min-address-range-loop       If the address range of a function containing a loop is less than this value, exclude it
                                   from instrumentation
    --dynamic-callsites            Force instrumentation if a function has dynamic callsites (e.g. function pointers)
    --traps                        Instrument points which require using a trap. On the x86 architecture, because
                                   instructions are of variable size, the instruction at a point may be too small for
                                   Dyninst to replace it with the normal code sequence used to call instrumentation. Also,
                                   when instrumentation is placed at points other than subroutine entry, exit, or call
                                   points, traps may be used to ensure the instrumentation fits. In this case, Dyninst
                                   replaces the instruction with a single-byte instruction that generates a trap.
    --loop-traps                   Instrument points within a loop which require using a trap (only relevant when
                                   --instrument-loops is enabled).
    --allow-overlapping            Allow dyninst to instrument either multiple functions which overlap (share part of same
                                   function body) or single functions with multiple entry points. For more info, see Section
                                   2 of the DyninstAPI documentation.

    [DYNINST OPTIONS]

    -b, --batch-size               Dyninst supports batch insertion of multiple points during runtime instrumentation. If
                                   one large batch insertion fails, this value will be used to create smaller batches.
                                   Larger batches generally decrease the instrumentation time
    --dyninst-options [ BaseTrampDeletion | DebugParsing | DelayedParsing | InstrStackFrames | MergeTramp | SaveFPR | TrampRecursive | TypeChecking ]
                                   Advanced dyninst options: BPatch::set<OPTION>(bool), e.g. bpatch->setTrampRecursive(true)
```

There are three ways to perform instrumentation:

1. Running the application via the omnitrace executable (analagous to `gdb --args <program> <args>`)
   - This mode is the default if neither the `-p` nor `-o` comand-line options are used
   - Runtime instrumentation supports instrumenting not only the target executable but also the
     the shared libraries loaded by the target executable. Consequently, this mode consumes more memory,
     takes longer to perform the instrumentation, and tends to have a more significant overhead on the
     runtime of the application
   - This mode is recommended if you want to analyze not only the performance of your executable and/or
     libraries but also the performance of the library dependencies
2. Attaching to a process that is currently running (analagous to `gdb -p <PID>`)
   - This mode is activate via `-p <PID>`
   - Same caveats as 1. with respect to memory and overhead
3. Generating a new executable or library with the instrumentation built-in (binary rewrite)
   - This mode is activated via the `-o <output-file>` option
   - Binary rewriting is limited to the text section of the target executable or library: it will not instrument
     the dynamically-linked libraries. Consequently, this mode performs the instrumentation significantly faster
     and has a much lower overhead when running the instrumentated executable and/or libraries
   - Binary rewriting is the recommended mode when the target executable uses process-level parallelism (e.g. MPI)
   - If your target executable has a minimal main which and the bulk of your application is in one specific dynamic library,
     see [Binary Rewriting a Library](#binary-rewriting-a-library) for help


> ***Attaching to a running process is an alpha feature and support for detaching from the target process***
> ***without ending the target process is not currently supported.***

The general syntax for separating omnitrace command line arguments from the application arguments follows the
is consistent with the LLVM style of using a standalone double-hyphen (`--`). All arguments preceding the double-hyphen
are interpreted as belonging to omnitrace and all arguments following the double-hyphen are interpreted as the
application and it's arguments. In binary rewrite mode, all application arguments after the first argument
are ignored, i.e. `./omnitrace -o ls.inst -- ls -l` interprets `ls` as the target to instrument (ignores the `-l` argument)
and generates a `ls.inst` executable that you can subsequently run `ls.inst -l` with.

## Runtime Instrumentation

```shell
omnitrace <omnitrace-options> -- <exe> [<exe-options>...]
```

## Attaching to Running Process

```shell
omnitrace <omnitrace-options> -p <PID> -- <exe-name>
```

## Binary Rewrite

```shell
omnitrace <omnitrace-options> -o <name-of-new-exe-or-library> -- <exe-or-library>
```

### Binary Rewriting a Library

Many applications bundle the bulk of their functionality into one or more dynamic libraries and have a relatively simple main
which links to these libraries and simply serves as the "driver" for setting up the workflow. If you binary rewrite your
executable and find there is insufficient info because of this, you can either switch to runtime instrumentation or
binary rewrite the libraries of interest.

Support for standalone binary rewriting of a dynamic library without binary rewriting the executable is a beta feature.
In general, it is supported as long as the library contains the `_init` and `_fini` symbols but these symbols are not
standardized to the extent of `main` in an executable.
The recommended workflow is as follows:

1. Determine the names of the dynamically linked libraries of interest via `ldd`
2. Generate a binary rewrite of the executable
3. Generate a binary rewrite of the desired libraries with the same base name as the original library, e.g. `libfoo.so.2` instead of `libfoo.so`
   - Output the instrumented library into a different folder than the original library
4. Prefix the `LD_LIBRARY_PATH` executable with the output folder from 3
5. Verify via `ldd` that the instrumented executable resolves the location of the instrumented library

### Binary Rewriting a Library Example

`foo` executable is dynamically linked to `libfoo.so.2`:

```shell
$ pwd
/home/user
$ which foo
/usr/local/bin/foo
$ ldd /usr/local/bin/foo
        ...
        libfoo.so.2 => /usr/local/lib/libfoo.so.2 (...)
        ...
```

Generate binary rewrites of `foo` and `libfoo.so.2`:

```shell
omnitrace -o ./foo.inst -- foo
omnitrace -o ./libfoo.so.2 -- /usr/local/lib/libfoo.so.2
```

At this point, the instrumented `foo.inst` executable will still dynamically load the original `libfoo.so.2` in `/usr/local/lib`:

```shell
$ ldd ./foo.inst
        ...
        libfoo.so.2 => /usr/local/lib/libfoo.so.2 (...)
        ...
```

Prefix the `LD_LIBRARY_PATH` environment variable with the folder containing the instrumented `libfoo.so.2`:

```shell
export LD_LIBRARY_PATH=/home/user:${LD_LIBRARY_PATH}
```

When `foo.inst` is executed, it will now load the instrumented library:

```shell
$ ldd ./foo.inst
        ...
        libfoo.so.2 => /home/user/libfoo.so.2 (...)
        ...
```

## Selective Instrumentation

The default behavior of omnitrace does not instrument every symbol in the binary. These default rules are:

- Skip instrumenting dynamic call-sites (i.e. function pointers)
    - Option `--dynamic-callsites` will force instrumentation for all dynamic call-sites
- The cost of a function can be loosely approximated by the size of the function in the binary so by default, omnitrace only instruments functions which span an address range of 256 bytes.
    - Option `--min-address-range` will modify this heuristic for all functions which do not contain loops
    - Option `--min-address-range-loop` will modify this heuristic for functions which contain loops
        - This separate loop option is provided because functions with loops can be compact in the binary while also being costly
- Skip instrumentation points which require using a trap
    - See the description for the `--traps` and `--loop-traps` options for more information
- Skip instrumenting loops within the body of a function
    - Option `--instrument-loops` will enable this behavior
- Skip instrumenting functions with overlapping function bodies and single functions with multiple entry point
    - These arise from various optimizations and instrumenting these functions can be enabled via the `--allow-overlapping` option

### Viewing the Available, Instrumented, Excluded, and Overlapping Functions

Whenever omnitrace is executed with a verbosity of zero or higher, it emits files which detail which functions (and which module they were defined in)
were available for instrumentation, which functions were instrumented, which functions were excluded, and which functions contained overlapping function bodies.
The default output path of these files will be in a `omnitrace-<NAME>-output` folder where `<NAME>` is the basename of the targeted binary or
(in the case of binary rewrite, the basename of the resulting executable), e.g.
`omnitrace -- ls` will output it's files to `omnitrace-ls-output` whereas `omnitrace -o ls.inst -- ls` will output to `omnitrace-ls.inst-output`.

If you would like to generate these files without executing or generating an executable, use the `--simulate` option:

```shell
omnitrace --simulate -- foo
omnitrace --simulate -o foo.inst -- foo
```

### Excluding and Including Modules and Functions

[Omnitrace](https://github.com/AMDResearch/omnitrace) has a set of 6 command-line options which each accept one or more regular expressions for customizing the scope of which module and/or functions are
instrumented. Multiple regexes per option are treated as an OR operation, e.g. `--module-include libfoo libbar` is effectively that same as `--module-include 'libfoo|libbar'`.

If you would like to force the inclusion of certain modules and/or function without changing any of the heuristics, use the `--module-include` and/or `--function-include` options.
Note that these options will not exclude modules and/or functions which do not satisfy their regular expression.

If you would like to narrow the scope of the instrumentation to a specific set of libraries and/or functions, use the `--module-restrict` and `--function-restrict` options.
Applying these options allow you to exclusively select the union one or more regular expressions, regardless of whether or not the functions satisfy the
aforementioned default heuristics. Any function or module that is not within the union of these regular expressions will be excluded from instrumentation.

If you would like to avoid instrumenting a set of modules and/or functions, use the `--module-exclude` and `--function-exclude` options.
These options are always applied regardless of whether the module or function satisfied the "restrict" or "include" regular expression.

#### Example Available Module and Function Info Output

> ***`omnitrace -o lulesh.inst --label file line args --simulate -- lulesh`***

```console
  AddressRange  Module                                    Function                                                                                 FunctionSignature
          9165  ../examples/lulesh/lulesh-comm.cc         CommMonoQ                                                                                CommMonoQ(domain) [lulesh-comm.cc:1891]
          3396  ../examples/lulesh/lulesh-comm.cc         CommRecv                                                                                 CommRecv(domain, int, Index_t, Index_t, Index_t, Index_t, bool, bool) [lulesh...
          8666  ../examples/lulesh/lulesh-comm.cc         CommSBN                                                                                  CommSBN(domain, int, Domain_member *) [lulesh-comm.cc:926]
         10212  ../examples/lulesh/lulesh-comm.cc         CommSend                                                                                 CommSend(domain, int, Index_t, Domain_member *, Index_t, Index_t, Index_t, bo...
          6823  ../examples/lulesh/lulesh-comm.cc         CommSyncPosVel                                                                           CommSyncPosVel(domain) [lulesh-comm.cc:1404]
           126  ../examples/lulesh/lulesh-comm.cc         _GLOBAL__sub_I_lulesh_comm.cc                                                            _GLOBAL__sub_I_lulesh_comm.cc() [lulesh-comm.cc]
           308  ../examples/lulesh/lulesh-init.cc         .omp_outlined..26                                                                        .omp_outlined..26(const , const , const ParallelFor<Kokkos::Impl::ViewCopy<Ko...
           628  ../examples/lulesh/lulesh-init.cc         .omp_outlined..34                                                                        .omp_outlined..34(const , const , const ParallelFor<Kokkos::Impl::ViewCopy<Ko...
           656  ../examples/lulesh/lulesh-init.cc         .omp_outlined..41                                                                        .omp_outlined..41(const , const , const ParallelFor<Kokkos::Impl::ViewCopy<Ko...
           662  ../examples/lulesh/lulesh-init.cc         .omp_outlined..45                                                                        .omp_outlined..45(const , const , const ParallelFor<Kokkos::Impl::ViewCopy<Ko...
           550  ../examples/lulesh/lulesh-init.cc         .omp_outlined..55                                                                        .omp_outlined..55(const , const , const ParallelFor<Kokkos::Impl::ViewFill<Ko...
           556  ../examples/lulesh/lulesh-init.cc         .omp_outlined..57                                                                        .omp_outlined..57(const , const , const ParallelFor<Kokkos::Impl::ViewFill<Ko...
           550  ../examples/lulesh/lulesh-init.cc         .omp_outlined..78                                                                        .omp_outlined..78(const , const , const ParallelFor<Kokkos::Impl::ViewFill<Ko...
           640  ../examples/lulesh/lulesh-init.cc         .omp_outlined..84                                                                        .omp_outlined..84(const , const , const ParallelFor<Kokkos::Impl::ViewCopy<Ko...
           646  ../examples/lulesh/lulesh-init.cc         .omp_outlined..88                                                                        .omp_outlined..88(const , const , const ParallelFor<Kokkos::Impl::ViewCopy<Ko...
          1840  ../examples/lulesh/lulesh-init.cc         Domain::AllocateElemPersistent                                                           Domain::AllocateElemPersistent(Domain *, Int_t) [lulesh-init.cc:94]
          1384  ../examples/lulesh/lulesh-init.cc         Domain::AllocateNodePersistent                                                           Domain::AllocateNodePersistent(Domain *, Int_t) [lulesh-init.cc:94]
          1264  ../examples/lulesh/lulesh-init.cc         Domain::BuildMesh                                                                        Domain::BuildMesh(Domain *, Int_t, Int_t, Int_t) [lulesh-init.cc:308]
          2312  ../examples/lulesh/lulesh-init.cc         Domain::CreateRegionIndexSets                                                            Domain::CreateRegionIndexSets(Domain *, Int_t, Int_t) [lulesh-init.cc:409]
          7109  ../examples/lulesh/lulesh-init.cc         Domain::Domain                                                                           Domain::Domain(Domain *, Int_t, Index_t, Index_t, Index_t, Index_t, int, int,...
          2458  ../examples/lulesh/lulesh-init.cc         Domain::SetupBoundaryConditions                                                          Domain::SetupBoundaryConditions(Domain *, Int_t) [lulesh-init.cc:409]
           956  ../examples/lulesh/lulesh-init.cc         Domain::SetupCommBuffers                                                                 Domain::SetupCommBuffers(Domain *, Int_t) [lulesh-init.cc]
          1456  ../examples/lulesh/lulesh-init.cc         Domain::SetupElementConnectivities                                                       Domain::SetupElementConnectivities(Domain *, Int_t) [lulesh-init.cc:409]
           721  ../examples/lulesh/lulesh-init.cc         Domain::SetupSymmetryPlanes                                                              Domain::SetupSymmetryPlanes(Domain *, Int_t) [lulesh-init.cc:409]
          1591  ../examples/lulesh/lulesh-init.cc         Domain::SetupThreadSupportStructures                                                     Domain::SetupThreadSupportStructures(Domain *) [lulesh-init.cc:376]
          1644  ../examples/lulesh/lulesh-init.cc         Domain::~Domain                                                                          Domain::~Domain(Domain *) [lulesh-init.cc:286]
           218  ../examples/lulesh/lulesh-init.cc         InitMeshDecomp                                                                           InitMeshDecomp(Int_t, Int_t, Int_t *, Int_t *, Int_t *, Int_t *) [lulesh-init...
           260  ../examples/lulesh/lulesh-init.cc         Kokkos::Impl::CommonSubview<Kokkos::View<int* [8], Kokkos::LayoutRight>, Kokk...         Kokkos::Impl::CommonSubview<Kokkos::View<int* [8], Kokkos::LayoutRight>, Kokk...
          1786  ../examples/lulesh/lulesh-init.cc         Kokkos::Impl::HostIterateTile<Kokkos::MDRangePolicy<Kokkos::OpenMP, Kokkos::R...         Kokkos::Impl::HostIterateTile<Kokkos::MDRangePolicy<Kokkos::OpenMP, Kokkos::R...
           330  ../examples/lulesh/lulesh-init.cc         Kokkos::Impl::ParallelConstructName<Kokkos::Impl::ViewCopy<Kokkos::View<int**...         Kokkos::Impl::ParallelConstructName<Kokkos::Impl::ViewCopy<Kokkos::View<int**...
           330  ../examples/lulesh/lulesh-init.cc         Kokkos::Impl::ParallelConstructName<Kokkos::Impl::ViewCopy<Kokkos::View<int**...         Kokkos::Impl::ParallelConstructName<Kokkos::Impl::ViewCopy<Kokkos::View<int**...
           330  ../examples/lulesh/lulesh-init.cc         Kokkos::Impl::ParallelConstructName<Kokkos::Impl::ViewCopy<Kokkos::View<int*,...         Kokkos::Impl::ParallelConstructName<Kokkos::Impl::ViewCopy<Kokkos::View<int*,...
           330  ../examples/lulesh/lulesh-init.cc         Kokkos::Impl::ParallelConstructName<Kokkos::Impl::ViewCopy<Kokkos::View<int*,...         Kokkos::Impl::ParallelConstructName<Kokkos::Impl::ViewCopy<Kokkos::View<int*,...
           330  ../examples/lulesh/lulesh-init.cc         Kokkos::Impl::ParallelConstructName<Kokkos::Impl::ViewFill<Kokkos::View<doubl...         Kokkos::Impl::ParallelConstructName<Kokkos::Impl::ViewFill<Kokkos::View<doubl...
           330  ../examples/lulesh/lulesh-init.cc         Kokkos::Impl::ParallelConstructName<Kokkos::Impl::ViewFill<Kokkos::View<doubl...         Kokkos::Impl::ParallelConstructName<Kokkos::Impl::ViewFill<Kokkos::View<doubl...
           330  ../examples/lulesh/lulesh-init.cc         Kokkos::Impl::ParallelConstructName<Kokkos::Impl::ViewFill<Kokkos::View<doubl...         Kokkos::Impl::ParallelConstructName<Kokkos::Impl::ViewFill<Kokkos::View<doubl...
           522  ../examples/lulesh/lulesh-init.cc         Kokkos::Impl::ParallelFor<Kokkos::Impl::ViewCopy<Kokkos::View<int**, Kokkos::...         Kokkos::Impl::ParallelFor<Kokkos::Impl::ViewCopy<Kokkos::View<int**, Kokkos::...
           232  ../examples/lulesh/lulesh-init.cc         Kokkos::Impl::ParallelFor<Kokkos::Impl::ViewCopy<Kokkos::View<int**, Kokkos::...         Kokkos::Impl::ParallelFor<Kokkos::Impl::ViewCopy<Kokkos::View<int**, Kokkos::...
            49  ../examples/lulesh/lulesh-init.cc         Kokkos::Impl::SharedAllocationRecord<Kokkos::HostSpace, Kokkos::Impl::ViewVal...         Kokkos::Impl::SharedAllocationRecord<Kokkos::HostSpace, Kokkos::Impl::ViewVal...
          1476  ../examples/lulesh/lulesh-init.cc         Kokkos::Impl::Tile_Loop_Type<2, false, int, void, void>::apply<Kokkos::Impl::...         Kokkos::Impl::Tile_Loop_Type<2, false, int, void, void>::apply<Kokkos::Impl::...
           555  ../examples/lulesh/lulesh-init.cc         Kokkos::Impl::ViewCopy<Kokkos::View<int**, Kokkos::LayoutRight, Kokkos::Devic...         Kokkos::Impl::ViewCopy<Kokkos::View<int**, Kokkos::LayoutRight, Kokkos::Devic...
           613  ../examples/lulesh/lulesh-init.cc         Kokkos::Impl::ViewCopy<Kokkos::View<int**, Kokkos::LayoutRight, Kokkos::Devic...         Kokkos::Impl::ViewCopy<Kokkos::View<int**, Kokkos::LayoutRight, Kokkos::Devic...
           603  ../examples/lulesh/lulesh-init.cc         Kokkos::Impl::ViewCopy<Kokkos::View<int*, Kokkos::LayoutLeft, Kokkos::Device<...         Kokkos::Impl::ViewCopy<Kokkos::View<int*, Kokkos::LayoutLeft, Kokkos::Device<...
           604  ../examples/lulesh/lulesh-init.cc         Kokkos::Impl::ViewCopy<Kokkos::View<int*, Kokkos::LayoutLeft, Kokkos::Device<...         Kokkos::Impl::ViewCopy<Kokkos::View<int*, Kokkos::LayoutLeft, Kokkos::Device<...
           281  ../examples/lulesh/lulesh-init.cc         Kokkos::Impl::ViewCtorProp<std::__cxx11::basic_string<char, std::char_traits<...         Kokkos::Impl::ViewCtorProp<std::__cxx11::basic_string<char, std::char_traits<...
           281  ../examples/lulesh/lulesh-init.cc         Kokkos::Impl::ViewCtorProp<std::__cxx11::basic_string<char, std::char_traits<...         Kokkos::Impl::ViewCtorProp<std::__cxx11::basic_string<char, std::char_traits<...
           281  ../examples/lulesh/lulesh-init.cc         Kokkos::Impl::ViewCtorProp<std::__cxx11::basic_string<char, std::char_traits<...         Kokkos::Impl::ViewCtorProp<std::__cxx11::basic_string<char, std::char_traits<...
           281  ../examples/lulesh/lulesh-init.cc         Kokkos::Impl::ViewCtorProp<std::__cxx11::basic_string<char, std::char_traits<...         Kokkos::Impl::ViewCtorProp<std::__cxx11::basic_string<char, std::char_traits<...
           281  ../examples/lulesh/lulesh-init.cc         Kokkos::Impl::ViewCtorProp<std::__cxx11::basic_string<char, std::char_traits<...         Kokkos::Impl::ViewCtorProp<std::__cxx11::basic_string<char, std::char_traits<...
           524  ../examples/lulesh/lulesh-init.cc         Kokkos::Impl::ViewFill<Kokkos::View<double*, Kokkos::LayoutRight, Kokkos::Dev...         Kokkos::Impl::ViewFill<Kokkos::View<double*, Kokkos::LayoutRight, Kokkos::Dev...
           525  ../examples/lulesh/lulesh-init.cc         Kokkos::Impl::ViewFill<Kokkos::View<double*, Kokkos::LayoutRight, Kokkos::Dev...         Kokkos::Impl::ViewFill<Kokkos::View<double*, Kokkos::LayoutRight, Kokkos::Dev...
           524  ../examples/lulesh/lulesh-init.cc         Kokkos::Impl::ViewFill<Kokkos::View<double*, Kokkos::LayoutRight, Kokkos::Dev...         Kokkos::Impl::ViewFill<Kokkos::View<double*, Kokkos::LayoutRight, Kokkos::Dev...
           583  ../examples/lulesh/lulesh-init.cc         Kokkos::Impl::ViewMapping<Kokkos::ViewTraits<int* [8], Kokkos::LayoutRight>, ...         SharedAllocationRecord<void, void> * Kokkos::Impl::ViewMapping<Kokkos::ViewTr...
           529  ../examples/lulesh/lulesh-init.cc         Kokkos::Impl::ViewMapping<Kokkos::ViewTraits<int*, Kokkos::HostSpace>, void>:...         SharedAllocationRecord<void, void> * Kokkos::Impl::ViewMapping<Kokkos::ViewTr...
           529  ../examples/lulesh/lulesh-init.cc         Kokkos::Impl::ViewMapping<Kokkos::ViewTraits<int*>, void>::allocate_shared<st...         SharedAllocationRecord<void, void> * Kokkos::Impl::ViewMapping<Kokkos::ViewTr...
           203  ../examples/lulesh/lulesh-init.cc         Kokkos::Impl::ViewRemap<Kokkos::View<int* [8], Kokkos::LayoutRight>, Kokkos::...         Kokkos::Impl::ViewRemap<Kokkos::View<int* [8], Kokkos::LayoutRight>, Kokkos::...
           331  ../examples/lulesh/lulesh-init.cc         Kokkos::Impl::ViewRemap<Kokkos::View<int*>, Kokkos::View<int*>, Kokkos::OpenM...         Kokkos::Impl::ViewRemap<Kokkos::View<int*>, Kokkos::View<int*>, Kokkos::OpenM...
           461  ../examples/lulesh/lulesh-init.cc         Kokkos::Impl::ViewValueFunctor<Kokkos::Device<Kokkos::OpenMP, Kokkos::HostSpa...         enable_if_t<std::is_trivial<int>::value && std::is_trivially_copy_assignable<...
           353  ../examples/lulesh/lulesh-init.cc         Kokkos::Impl::contiguous_fill<Kokkos::OpenMP, double*>                                   Kokkos::Impl::contiguous_fill<Kokkos::OpenMP, double*>(exec_space, dst, value...
           139  ../examples/lulesh/lulesh-init.cc         Kokkos::Impl::contiguous_fill<Kokkos::OpenMP, double, Kokkos::LayoutRight, Ko...         Kokkos::Impl::contiguous_fill<Kokkos::OpenMP, double, Kokkos::LayoutRight, Ko...
           824  ../examples/lulesh/lulesh-init.cc         Kokkos::Impl::view_copy<Kokkos::View<int* [8], Kokkos::LayoutRight, Kokkos::D...         Kokkos::Impl::view_copy<Kokkos::View<int* [8], Kokkos::LayoutRight, Kokkos::D...
           824  ../examples/lulesh/lulesh-init.cc         Kokkos::Impl::view_copy<Kokkos::View<int* [8], Kokkos::LayoutRight, Kokkos::D...         Kokkos::Impl::view_copy<Kokkos::View<int* [8], Kokkos::LayoutRight, Kokkos::D...
           824  ../examples/lulesh/lulesh-init.cc         Kokkos::Impl::view_copy<Kokkos::View<int* [8], Kokkos::LayoutRight>, Kokkos::...         Kokkos::Impl::view_copy<Kokkos::View<int* [8], Kokkos::LayoutRight>, Kokkos::...
           824  ../examples/lulesh/lulesh-init.cc         Kokkos::Impl::view_copy<Kokkos::View<int* [8], Kokkos::LayoutRight>, Kokkos::...         Kokkos::Impl::view_copy<Kokkos::View<int* [8], Kokkos::LayoutRight>, Kokkos::...
           697  ../examples/lulesh/lulesh-init.cc         Kokkos::Impl::view_copy<Kokkos::View<int*, Kokkos::LayoutRight, Kokkos::Devic...         Kokkos::Impl::view_copy<Kokkos::View<int*, Kokkos::LayoutRight, Kokkos::Devic...
           697  ../examples/lulesh/lulesh-init.cc         Kokkos::Impl::view_copy<Kokkos::View<int*>, Kokkos::View<int*> >                         Kokkos::Impl::view_copy<Kokkos::View<int*>, Kokkos::View<int*> >(dst, src) [l...
          2036  ../examples/lulesh/lulesh-init.cc         Kokkos::RangePolicy<Kokkos::OpenMP, Kokkos::Schedule<Kokkos::Static>, int>::R...         Kokkos::RangePolicy<Kokkos::OpenMP, Kokkos::Schedule<Kokkos::Static>, int>::R...
          2506  ../examples/lulesh/lulesh-init.cc         Kokkos::RangePolicy<Kokkos::OpenMP, Kokkos::Schedule<Kokkos::Static>, long>::...         Kokkos::RangePolicy<Kokkos::OpenMP, Kokkos::Schedule<Kokkos::Static>, long>::...
           271  ../examples/lulesh/lulesh-init.cc         Kokkos::StaticCrsGraph<int, Kokkos::LayoutLeft, Kokkos::OpenMP, Kokkos::Memor...         Kokkos::StaticCrsGraph<int, Kokkos::LayoutLeft, Kokkos::OpenMP, Kokkos::Memor...
           470  ../examples/lulesh/lulesh-init.cc         Kokkos::View<int* [8], Kokkos::LayoutRight>::View<std::__cxx11::basic_string<...         Kokkos::View<int* [8], Kokkos::LayoutRight>::View<std::__cxx11::basic_string<...
           323  ../examples/lulesh/lulesh-init.cc         Kokkos::View<int* [8], Kokkos::LayoutRight>::View<std::__cxx11::basic_string<...         Kokkos::View<int* [8], Kokkos::LayoutRight>::View<std::__cxx11::basic_string<...
           410  ../examples/lulesh/lulesh-init.cc         Kokkos::View<int*, Kokkos::HostSpace>::View<char [10]>                                   Kokkos::View<int*, Kokkos::HostSpace>::View<char [10]>(View<int *, Kokkos::Ho...
           410  ../examples/lulesh/lulesh-init.cc         Kokkos::View<int*, Kokkos::HostSpace>::View<char [14]>                                   Kokkos::View<int*, Kokkos::HostSpace>::View<char [14]>(View<int *, Kokkos::Ho...
           462  ../examples/lulesh/lulesh-init.cc         Kokkos::View<int*, Kokkos::HostSpace>::View<std::__cxx11::basic_string<char, ...         Kokkos::View<int*, Kokkos::HostSpace>::View<std::__cxx11::basic_string<char, ...
           410  ../examples/lulesh/lulesh-init.cc         Kokkos::View<int*>::View<char [16]>                                                      Kokkos::View<int*>::View<char [16]>(View<int *> *, arg_label, type, const siz...
           410  ../examples/lulesh/lulesh-init.cc         Kokkos::View<int*>::View<char [19]>                                                      Kokkos::View<int*>::View<char [19]>(View<int *> *, arg_label, type, const siz...
           410  ../examples/lulesh/lulesh-init.cc         Kokkos::View<int*>::View<char [21]>                                                      Kokkos::View<int*>::View<char [21]>(View<int *> *, arg_label, type, const siz...
           462  ../examples/lulesh/lulesh-init.cc         Kokkos::View<int*>::View<std::__cxx11::basic_string<char, std::char_traits<ch...         Kokkos::View<int*>::View<std::__cxx11::basic_string<char, std::char_traits<ch...
           323  ../examples/lulesh/lulesh-init.cc         Kokkos::View<int*>::View<std::__cxx11::basic_string<char, std::char_traits<ch...         Kokkos::View<int*>::View<std::__cxx11::basic_string<char, std::char_traits<ch...
          6589  ../examples/lulesh/lulesh-init.cc         Kokkos::deep_copy<double*, , double*, Kokkos::LayoutRight, Kokkos::Device<Kok...         Kokkos::deep_copy<double*, , double*, Kokkos::LayoutRight, Kokkos::Device<Kok...
          1052  ../examples/lulesh/lulesh-init.cc         Kokkos::deep_copy<double*>                                                               Kokkos::deep_copy<double*>(dst, value) [lulesh-init.cc]
          1050  ../examples/lulesh/lulesh-init.cc         Kokkos::deep_copy<double, Kokkos::LayoutRight, Kokkos::Device<Kokkos::OpenMP,...         Kokkos::deep_copy<double, Kokkos::LayoutRight, Kokkos::Device<Kokkos::OpenMP,...
          7686  ../examples/lulesh/lulesh-init.cc         Kokkos::deep_copy<int* [8], Kokkos::LayoutRight, Kokkos::Device<Kokkos::OpenM...         Kokkos::deep_copy<int* [8], Kokkos::LayoutRight, Kokkos::Device<Kokkos::OpenM...
          7686  ../examples/lulesh/lulesh-init.cc         Kokkos::deep_copy<int* [8], Kokkos::LayoutRight, int* [8], Kokkos::LayoutRigh...         Kokkos::deep_copy<int* [8], Kokkos::LayoutRight, int* [8], Kokkos::LayoutRigh...
          6589  ../examples/lulesh/lulesh-init.cc         Kokkos::deep_copy<int*, , int*, Kokkos::LayoutRight, Kokkos::Device<Kokkos::O...         Kokkos::deep_copy<int*, , int*, Kokkos::LayoutRight, Kokkos::Device<Kokkos::O...
          6589  ../examples/lulesh/lulesh-init.cc         Kokkos::deep_copy<int*, Kokkos::LayoutLeft, Kokkos::Device<Kokkos::OpenMP, Ko...         Kokkos::deep_copy<int*, Kokkos::LayoutLeft, Kokkos::Device<Kokkos::OpenMP, Ko...
          6589  ../examples/lulesh/lulesh-init.cc         Kokkos::deep_copy<int*, Kokkos::LayoutRight, Kokkos::Device<Kokkos::OpenMP, K...         Kokkos::deep_copy<int*, Kokkos::LayoutRight, Kokkos::Device<Kokkos::OpenMP, K...
           863  ../examples/lulesh/lulesh-init.cc         Kokkos::impl_resize<, int* [8], Kokkos::LayoutRight>                                     type Kokkos::impl_resize<, int* [8], Kokkos::LayoutRight>(v, const size_t, co...
           854  ../examples/lulesh/lulesh-init.cc         Kokkos::impl_resize<, int*>                                                              type Kokkos::impl_resize<, int*>(v, const size_t, const size_t, const size_t,...
           697  ../examples/lulesh/lulesh-init.cc         Kokkos::parallel_for<Kokkos::MDRangePolicy<Kokkos::OpenMP, Kokkos::Rank<2u, (...         Kokkos::parallel_for<Kokkos::MDRangePolicy<Kokkos::OpenMP, Kokkos::Rank<2u, (...
           706  ../examples/lulesh/lulesh-init.cc         Kokkos::parallel_for<Kokkos::MDRangePolicy<Kokkos::OpenMP, Kokkos::Rank<2u, (...         Kokkos::parallel_for<Kokkos::MDRangePolicy<Kokkos::OpenMP, Kokkos::Rank<2u, (...
           912  ../examples/lulesh/lulesh-init.cc         Kokkos::parallel_for<Kokkos::RangePolicy<Kokkos::OpenMP, Kokkos::IndexType<in...         Kokkos::parallel_for<Kokkos::RangePolicy<Kokkos::OpenMP, Kokkos::IndexType<in...
           791  ../examples/lulesh/lulesh-init.cc         Kokkos::parallel_for<Kokkos::RangePolicy<Kokkos::OpenMP, Kokkos::IndexType<in...         Kokkos::parallel_for<Kokkos::RangePolicy<Kokkos::OpenMP, Kokkos::IndexType<in...
           791  ../examples/lulesh/lulesh-init.cc         Kokkos::parallel_for<Kokkos::RangePolicy<Kokkos::OpenMP, Kokkos::IndexType<in...         Kokkos::parallel_for<Kokkos::RangePolicy<Kokkos::OpenMP, Kokkos::IndexType<in...
           944  ../examples/lulesh/lulesh-init.cc         Kokkos::parallel_for<Kokkos::RangePolicy<Kokkos::OpenMP, Kokkos::IndexType<lo...         Kokkos::parallel_for<Kokkos::RangePolicy<Kokkos::OpenMP, Kokkos::IndexType<lo...
           839  ../examples/lulesh/lulesh-init.cc         Kokkos::parallel_for<Kokkos::RangePolicy<Kokkos::OpenMP, Kokkos::IndexType<lo...         Kokkos::parallel_for<Kokkos::RangePolicy<Kokkos::OpenMP, Kokkos::IndexType<lo...
           126  ../examples/lulesh/lulesh-init.cc         _GLOBAL__sub_I_lulesh_init.cc                                                            _GLOBAL__sub_I_lulesh_init.cc() [lulesh-init.cc]
          6589  ../examples/lulesh/lulesh-util.cc         Kokkos::deep_copy<double*, Kokkos::LayoutRight, Kokkos::Device<Kokkos::OpenMP...         Kokkos::deep_copy<double*, Kokkos::LayoutRight, Kokkos::Device<Kokkos::OpenMP...
          1345  ../examples/lulesh/lulesh-util.cc         ParseCommandLineOptions                                                                  ParseCommandLineOptions(int, char * *, int, cmdLineOpts *) [lulesh-util.cc:67]
           171  ../examples/lulesh/lulesh-util.cc         PrintCommandLineOptions                                                                  PrintCommandLineOptions(char *, int) [lulesh-util.cc:31]
            67  ../examples/lulesh/lulesh-util.cc         StrToInt                                                                                 int StrToInt(const char *, int *) [lulesh-util.cc:13]
           706  ../examples/lulesh/lulesh-util.cc         VerifyAndWriteFinalOutput                                                                VerifyAndWriteFinalOutput(Real_t, locDom, Int_t, Int_t) [lulesh-util.cc:222]
           126  ../examples/lulesh/lulesh-util.cc         _GLOBAL__sub_I_lulesh_util.cc                                                            _GLOBAL__sub_I_lulesh_util.cc() [lulesh-util.cc]
            17  ../examples/lulesh/lulesh-viz.cc          DumpToVisit                                                                              DumpToVisit(domain, int, int, int) [lulesh-viz.cc:415]
           126  ../examples/lulesh/lulesh-viz.cc          _GLOBAL__sub_I_lulesh_viz.cc                                                             _GLOBAL__sub_I_lulesh_viz.cc() [lulesh-viz.cc]
           451  ../examples/lulesh/lulesh.cc              .omp_outlined..103                                                                       .omp_outlined..103(const , const , const ParallelReduce<(lambda at ../example...
           796  ../examples/lulesh/lulesh.cc              .omp_outlined..109                                                                       .omp_outlined..109(const , const , const ParallelFor<(lambda at ../examples/l...
           394  ../examples/lulesh/lulesh.cc              .omp_outlined..111                                                                       .omp_outlined..111(const , const , const ParallelFor<(lambda at ../examples/l...
           402  ../examples/lulesh/lulesh.cc              .omp_outlined..113                                                                       .omp_outlined..113(const , const , const ParallelFor<(lambda at ../examples/l...
           427  ../examples/lulesh/lulesh.cc              .omp_outlined..115                                                                       .omp_outlined..115(const , const , const ParallelReduce<(lambda at ../example...
           859  ../examples/lulesh/lulesh.cc              .omp_outlined..119                                                                       .omp_outlined..119(const , const , const ParallelFor<(lambda at ../examples/l...
           243  ../examples/lulesh/lulesh.cc              .omp_outlined..122                                                                       .omp_outlined..122(const , const , const ParallelFor<(lambda at ../examples/l...
           426  ../examples/lulesh/lulesh.cc              .omp_outlined..124                                                                       .omp_outlined..124(const , const , const ParallelFor<(lambda at ../examples/l...
           529  ../examples/lulesh/lulesh.cc              .omp_outlined..127                                                                       .omp_outlined..127(const , const , const ParallelFor<(lambda at ../examples/l...
           865  ../examples/lulesh/lulesh.cc              .omp_outlined..130                                                                       .omp_outlined..130(const , const , const ParallelFor<(lambda at ../examples/l...
           539  ../examples/lulesh/lulesh.cc              .omp_outlined..132                                                                       .omp_outlined..132(const , const , const ParallelReduce<(lambda at ../example...
           456  ../examples/lulesh/lulesh.cc              .omp_outlined..134                                                                       .omp_outlined..134(const , const , const ParallelReduce<(lambda at ../example...
           252  ../examples/lulesh/lulesh.cc              .omp_outlined..20                                                                        .omp_outlined..20(const , const , const ParallelFor<(lambda at ../examples/lu...
           870  ../examples/lulesh/lulesh.cc              .omp_outlined..35                                                                        .omp_outlined..35(const , const , const ParallelFor<(lambda at ../examples/lu...
           473  ../examples/lulesh/lulesh.cc              .omp_outlined..42                                                                        .omp_outlined..42(const , const , const ParallelFor<(lambda at ../examples/lu...
           252  ../examples/lulesh/lulesh.cc              .omp_outlined..46                                                                        .omp_outlined..46(const , const , const ParallelFor<(lambda at ../examples/lu...
          1101  ../examples/lulesh/lulesh.cc              .omp_outlined..48                                                                        .omp_outlined..48(const , const , const ParallelFor<(lambda at ../examples/lu...
           427  ../examples/lulesh/lulesh.cc              .omp_outlined..55                                                                        .omp_outlined..55(const , const , const ParallelReduce<(lambda at ../examples...
          1326  ../examples/lulesh/lulesh.cc              .omp_outlined..57                                                                        .omp_outlined..57(const , const , const ParallelReduce<(lambda at ../examples...
           243  ../examples/lulesh/lulesh.cc              .omp_outlined..61                                                                        .omp_outlined..61(const , const , const ParallelFor<(lambda at ../examples/lu...
          1101  ../examples/lulesh/lulesh.cc              .omp_outlined..63                                                                        .omp_outlined..63(const , const , const ParallelFor<(lambda at ../examples/lu...
           372  ../examples/lulesh/lulesh.cc              .omp_outlined..66                                                                        .omp_outlined..66(const , const , const ParallelFor<(lambda at ../examples/lu...
           499  ../examples/lulesh/lulesh.cc              .omp_outlined..71                                                                        .omp_outlined..71(const , const , const ParallelFor<(lambda at ../examples/lu...
           499  ../examples/lulesh/lulesh.cc              .omp_outlined..73                                                                        .omp_outlined..73(const , const , const ParallelFor<(lambda at ../examples/lu...
           499  ../examples/lulesh/lulesh.cc              .omp_outlined..75                                                                        .omp_outlined..75(const , const , const ParallelFor<(lambda at ../examples/lu...
           465  ../examples/lulesh/lulesh.cc              .omp_outlined..78                                                                        .omp_outlined..78(const , const , const ParallelFor<(lambda at ../examples/lu...
           396  ../examples/lulesh/lulesh.cc              .omp_outlined..81                                                                        .omp_outlined..81(const , const , const ParallelFor<(lambda at ../examples/lu...
           656  ../examples/lulesh/lulesh.cc              .omp_outlined..85                                                                        .omp_outlined..85(const , const , const ParallelFor<Kokkos::Impl::ViewCopy<Ko...
           662  ../examples/lulesh/lulesh.cc              .omp_outlined..89                                                                        .omp_outlined..89(const , const , const ParallelFor<Kokkos::Impl::ViewCopy<Ko...
           443  ../examples/lulesh/lulesh.cc              .omp_outlined..93                                                                        .omp_outlined..93(const , const , const ParallelReduce<(lambda at ../examples...
           243  ../examples/lulesh/lulesh.cc              .omp_outlined..96                                                                        .omp_outlined..96(const , const , const ParallelFor<(lambda at ../examples/lu...
           243  ../examples/lulesh/lulesh.cc              .omp_outlined..99                                                                        .omp_outlined..99(const , const , const ParallelFor<(lambda at ../examples/lu...
         13367  ../examples/lulesh/lulesh.cc              ApplyMaterialPropertiesForElems                                                          ApplyMaterialPropertiesForElems(domain) [lulesh.cc:409]
          1530  ../examples/lulesh/lulesh.cc              CalcElemCharacteristicLength                                                             Real_t CalcElemCharacteristicLength(const Real_t *, const Real_t *, const Rea...
           982  ../examples/lulesh/lulesh.cc              CalcElemFBHourglassForce                                                                 CalcElemFBHourglassForce(const Real_t *, const Real_t[] *, coefficient, Real_...
          2428  ../examples/lulesh/lulesh.cc              CalcElemNodeNormals                                                                      CalcElemNodeNormals(Real_t *, Real_t *, Real_t *, const Real_t *, const Real_...
           853  ../examples/lulesh/lulesh.cc              CalcElemShapeFunctionDerivatives                                                         CalcElemShapeFunctionDerivatives(const Real_t *, const Real_t *, const Real_t...
          1097  ../examples/lulesh/lulesh.cc              CalcElemVolumeDerivative                                                                 CalcElemVolumeDerivative(i, dvdx, dvdy, dvdz, const Real_t *, const Real_t *,...
          1054  ../examples/lulesh/lulesh.cc              CalcKinematicsForElems                                                                   CalcKinematicsForElems(domain, Real_t, Index_t) [lulesh.cc]
         14160  ../examples/lulesh/lulesh.cc              CalcVolumeForceForElems                                                                  CalcVolumeForceForElems(domain) [lulesh.cc:409]
           366  ../examples/lulesh/lulesh.cc              Domain::AllocateGradients                                                                Domain::AllocateGradients(Domain *, Int_t, Int_t) [lulesh.cc:214]
           475  ../examples/lulesh/lulesh.cc              Domain::DeallocateGradients                                                              Domain::DeallocateGradients(Domain *) [lulesh.cc:105]
           250  ../examples/lulesh/lulesh.cc              Domain::DeallocateStrains                                                                Domain::DeallocateStrains(Domain *) [lulesh.cc:105]
          4356  ../examples/lulesh/lulesh.cc              Domain::Domain                                                                           Domain::Domain(Domain *) [lulesh.cc:78]
            15  ../examples/lulesh/lulesh.cc              Domain::delv_eta                                                                         Domain::delv_eta(const Domain *, const Index_t) [lulesh.cc:371]
            15  ../examples/lulesh/lulesh.cc              Domain::delv_xi                                                                          Domain::delv_xi(const Domain *, const Index_t) [lulesh.cc:368]
            15  ../examples/lulesh/lulesh.cc              Domain::delv_zeta                                                                        Domain::delv_zeta(const Domain *, const Index_t) [lulesh.cc:374]
            15  ../examples/lulesh/lulesh.cc              Domain::fx                                                                               Domain::fx(const Domain *, const Index_t) [lulesh.cc:303]
            15  ../examples/lulesh/lulesh.cc              Domain::fy                                                                               Domain::fy(const Domain *, const Index_t) [lulesh.cc:306]
            15  ../examples/lulesh/lulesh.cc              Domain::fz                                                                               Domain::fz(const Domain *, const Index_t) [lulesh.cc:309]
            15  ../examples/lulesh/lulesh.cc              Domain::nodalMass                                                                        Domain::nodalMass(const Domain *, const Index_t) [lulesh.cc:314]
            15  ../examples/lulesh/lulesh.cc              Domain::x                                                                                Domain::x(const Domain *, const Index_t) [lulesh.cc:257]
            15  ../examples/lulesh/lulesh.cc              Domain::xd                                                                               Domain::xd(const Domain *, const Index_t) [lulesh.cc:272]
            15  ../examples/lulesh/lulesh.cc              Domain::y                                                                                Domain::y(const Domain *, const Index_t) [lulesh.cc:258]
            15  ../examples/lulesh/lulesh.cc              Domain::yd                                                                               Domain::yd(const Domain *, const Index_t) [lulesh.cc:275]
            15  ../examples/lulesh/lulesh.cc              Domain::z                                                                                Domain::z(const Domain *, const Index_t) [lulesh.cc:259]
            15  ../examples/lulesh/lulesh.cc              Domain::zd                                                                               Domain::zd(const Domain *, const Index_t) [lulesh.cc:278]
           330  ../examples/lulesh/lulesh.cc              Kokkos::Impl::ParallelConstructName<Kokkos::Impl::ViewCopy<Kokkos::View<doubl...         Kokkos::Impl::ParallelConstructName<Kokkos::Impl::ViewCopy<Kokkos::View<doubl...
           330  ../examples/lulesh/lulesh.cc              Kokkos::Impl::ParallelConstructName<Kokkos::Impl::ViewCopy<Kokkos::View<doubl...         Kokkos::Impl::ParallelConstructName<Kokkos::Impl::ViewCopy<Kokkos::View<doubl...
          1508  ../examples/lulesh/lulesh.cc              Kokkos::Impl::ParallelFor<CalcEnergyForElems(double*, double*, double*, doubl...         type Kokkos::Impl::ParallelFor<CalcEnergyForElems(double*, double*, double*, ...
          3606  ../examples/lulesh/lulesh.cc              Kokkos::Impl::ParallelFor<CalcFBHourglassForceForElems(Domain&, double*, Kokk...         type Kokkos::Impl::ParallelFor<CalcFBHourglassForceForElems(Domain&, double*,...
          2917  ../examples/lulesh/lulesh.cc              Kokkos::Impl::ParallelFor<CalcKinematicsForElems(Domain&, double, int)::$_0, ...         type Kokkos::Impl::ParallelFor<CalcKinematicsForElems(Domain&, double, int)::...
          3119  ../examples/lulesh/lulesh.cc              Kokkos::Impl::ParallelFor<CalcMonotonicQGradientsForElems(Domain&)::{lambda(i...         type Kokkos::Impl::ParallelFor<CalcMonotonicQGradientsForElems(Domain&)::{lam...
          1969  ../examples/lulesh/lulesh.cc              Kokkos::Impl::ParallelFor<CalcMonotonicQRegionForElems(Domain&, int, double):...         type Kokkos::Impl::ParallelFor<CalcMonotonicQRegionForElems(Domain&, int, dou...
          1265  ../examples/lulesh/lulesh.cc              Kokkos::Impl::ParallelFor<IntegrateStressForElems(Domain&, double*, double*, ...         type Kokkos::Impl::ParallelFor<IntegrateStressForElems(Domain&, double*, doub...
            49  ../examples/lulesh/lulesh.cc              Kokkos::Impl::SharedAllocationRecord<Kokkos::HostSpace, Kokkos::Impl::ViewVal...         Kokkos::Impl::SharedAllocationRecord<Kokkos::HostSpace, Kokkos::Impl::ViewVal...
          1497  ../examples/lulesh/lulesh.cc              Kokkos::Impl::TeamPolicyInternal<Kokkos::OpenMP>::TeamPolicyInternal                     Kokkos::Impl::TeamPolicyInternal<Kokkos::OpenMP>::TeamPolicyInternal(TeamPoli...
           603  ../examples/lulesh/lulesh.cc              Kokkos::Impl::ViewCopy<Kokkos::View<double*, Kokkos::LayoutLeft, Kokkos::Devi...         Kokkos::Impl::ViewCopy<Kokkos::View<double*, Kokkos::LayoutLeft, Kokkos::Devi...
           604  ../examples/lulesh/lulesh.cc              Kokkos::Impl::ViewCopy<Kokkos::View<double*, Kokkos::LayoutLeft, Kokkos::Devi...         Kokkos::Impl::ViewCopy<Kokkos::View<double*, Kokkos::LayoutLeft, Kokkos::Devi...
           281  ../examples/lulesh/lulesh.cc              Kokkos::Impl::ViewCtorProp<std::__cxx11::basic_string<char, std::char_traits<...         Kokkos::Impl::ViewCtorProp<std::__cxx11::basic_string<char, std::char_traits<...
           281  ../examples/lulesh/lulesh.cc              Kokkos::Impl::ViewCtorProp<std::__cxx11::basic_string<char, std::char_traits<...         Kokkos::Impl::ViewCtorProp<std::__cxx11::basic_string<char, std::char_traits<...
           521  ../examples/lulesh/lulesh.cc              Kokkos::Impl::ViewMapping<Kokkos::ViewTraits<double*>, void>::allocate_shared...         SharedAllocationRecord<void, void> * Kokkos::Impl::ViewMapping<Kokkos::ViewTr...
           331  ../examples/lulesh/lulesh.cc              Kokkos::Impl::ViewRemap<Kokkos::View<double*>, Kokkos::View<double*>, Kokkos:...         Kokkos::Impl::ViewRemap<Kokkos::View<double*>, Kokkos::View<double*>, Kokkos:...
           461  ../examples/lulesh/lulesh.cc              Kokkos::Impl::ViewValueFunctor<Kokkos::Device<Kokkos::OpenMP, Kokkos::HostSpa...         enable_if_t<std::is_trivial<double>::value && std::is_trivially_copy_assignab...
          1609  ../examples/lulesh/lulesh.cc              Kokkos::Impl::runtime_check_rank_host                                                    Kokkos::Impl::runtime_check_rank_host(const size_t, const bool, const size_t,...
           697  ../examples/lulesh/lulesh.cc              Kokkos::Impl::view_copy<Kokkos::View<double*, Kokkos::LayoutRight, Kokkos::De...         Kokkos::Impl::view_copy<Kokkos::View<double*, Kokkos::LayoutRight, Kokkos::De...
           697  ../examples/lulesh/lulesh.cc              Kokkos::Impl::view_copy<Kokkos::View<double*>, Kokkos::View<double*> >                   Kokkos::Impl::view_copy<Kokkos::View<double*>, Kokkos::View<double*> >(dst, s...
          2250  ../examples/lulesh/lulesh.cc              Kokkos::RangePolicy<Kokkos::OpenMP>::RangePolicy                                         Kokkos::RangePolicy<Kokkos::OpenMP>::RangePolicy(RangePolicy<Kokkos::OpenMP> ...
           213  ../examples/lulesh/lulesh.cc              Kokkos::StaticCrsGraph<int, Kokkos::LayoutLeft, Kokkos::OpenMP, Kokkos::Memor...         Kokkos::StaticCrsGraph<int, Kokkos::LayoutLeft, Kokkos::OpenMP, Kokkos::Memor...
           410  ../examples/lulesh/lulesh.cc              Kokkos::View<double*>::View<char [6]>                                                    Kokkos::View<double*>::View<char [6]>(View<double *> *, arg_label, type, cons...
           410  ../examples/lulesh/lulesh.cc              Kokkos::View<double*>::View<char [7]>                                                    Kokkos::View<double*>::View<char [7]>(View<double *> *, arg_label, type, cons...
           462  ../examples/lulesh/lulesh.cc              Kokkos::View<double*>::View<std::__cxx11::basic_string<char, std::char_traits...         Kokkos::View<double*>::View<std::__cxx11::basic_string<char, std::char_traits...
           323  ../examples/lulesh/lulesh.cc              Kokkos::View<double*>::View<std::__cxx11::basic_string<char, std::char_traits...         Kokkos::View<double*>::View<std::__cxx11::basic_string<char, std::char_traits...
            25  ../examples/lulesh/lulesh.cc              Kokkos::View<double*>::~View                                                             Kokkos::View<double*>::~View(View<double *> *) [lulesh.cc:409]
           840  ../examples/lulesh/lulesh.cc              Kokkos::abort                                                                            Kokkos::abort(const const char *, const const char *) [lulesh.cc:202]
           854  ../examples/lulesh/lulesh.cc              Kokkos::impl_resize<, double*>                                                           type Kokkos::impl_resize<, double*>(v, const size_t, const size_t, const size...
           928  ../examples/lulesh/lulesh.cc              Kokkos::parallel_for<Kokkos::RangePolicy<Kokkos::OpenMP, Kokkos::IndexType<in...         Kokkos::parallel_for<Kokkos::RangePolicy<Kokkos::OpenMP, Kokkos::IndexType<in...
           960  ../examples/lulesh/lulesh.cc              Kokkos::parallel_for<Kokkos::RangePolicy<Kokkos::OpenMP, Kokkos::IndexType<lo...         Kokkos::parallel_for<Kokkos::RangePolicy<Kokkos::OpenMP, Kokkos::IndexType<lo...
         21470  ../examples/lulesh/lulesh.cc              LagrangeLeapFrog                                                                         LagrangeLeapFrog(domain) [lulesh.cc]
           226  ../examples/lulesh/lulesh.cc              ResizeBuffer                                                                             ResizeBuffer(const size_t) [lulesh.cc:23]
           169  ../examples/lulesh/lulesh.cc              _GLOBAL__sub_I_lulesh.cc                                                                 _GLOBAL__sub_I_lulesh.cc() [lulesh.cc]
          1836  ../examples/lulesh/lulesh.cc              main                                                                                     int main(int, char * *) [lulesh.cc]
            63  ../examples/lulesh/lulesh.cc              std::_Rb_tree<std::__cxx11::basic_string<char, std::char_traits<char>, std::a...         std::_Rb_tree<std::__cxx11::basic_string<char, std::char_traits<char>, std::a...
            20  ../examples/lulesh/lulesh.cc              std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::alloca...         std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::alloca...
           160  ../examples/lulesh/lulesh.cc              std::operator+<char, std::char_traits<char>, std::allocator<char> >                      basic_string<char, std::char_traits<char>, std::allocator<char> > std::operat...
           187  ../examples/lulesh/lulesh.cc              std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::alloc...         std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::alloc...
            11  lulesh                                    __clang_call_terminate                                                                   __clang_call_terminate() [lulesh]
            33  lulesh                                    __do_global_dtors_aux                                                                    __do_global_dtors_aux() [lulesh]
             5  lulesh                                    __libc_csu_fini                                                                          __libc_csu_fini() [lulesh]
           101  lulesh                                    __libc_csu_init                                                                          __libc_csu_init() [lulesh]
             5  lulesh                                    _dl_relocate_static_pie                                                                  _dl_relocate_static_pie() [lulesh]
            13  lulesh                                    _fini                                                                                    _fini() [lulesh]
            27  lulesh                                    _init                                                                                    _init() [lulesh]
            47  lulesh                                    _start                                                                                   _start() [lulesh]
             6  lulesh                                    frame_dummy                                                                              frame_dummy() [lulesh]
```

#### Example Instrumented Module and Function Info Output

> ***`omnitrace -o lulesh.inst --label file line args --simulate -- lulesh`***

After the heuristics are applied in [Example Available Module and Function Info Output](#example-available-module-and-function-info-output),
the selected module/functions are:

```console
  AddressRange  Module                                    Function                                                                                 FunctionSignature
          9165  ../examples/lulesh/lulesh-comm.cc         CommMonoQ                                                                                CommMonoQ(domain) [lulesh-comm.cc:1891]
          3396  ../examples/lulesh/lulesh-comm.cc         CommRecv                                                                                 CommRecv(domain, int, Index_t, Index_t, Index_t, Index_t, bool, bool) [lulesh...
          8666  ../examples/lulesh/lulesh-comm.cc         CommSBN                                                                                  CommSBN(domain, int, Domain_member *) [lulesh-comm.cc:926]
         10212  ../examples/lulesh/lulesh-comm.cc         CommSend                                                                                 CommSend(domain, int, Index_t, Domain_member *, Index_t, Index_t, Index_t, bo...
          6823  ../examples/lulesh/lulesh-comm.cc         CommSyncPosVel                                                                           CommSyncPosVel(domain) [lulesh-comm.cc:1404]
          1840  ../examples/lulesh/lulesh-init.cc         Domain::AllocateElemPersistent                                                           Domain::AllocateElemPersistent(Domain *, Int_t) [lulesh-init.cc:94]
          1384  ../examples/lulesh/lulesh-init.cc         Domain::AllocateNodePersistent                                                           Domain::AllocateNodePersistent(Domain *, Int_t) [lulesh-init.cc:94]
          1264  ../examples/lulesh/lulesh-init.cc         Domain::BuildMesh                                                                        Domain::BuildMesh(Domain *, Int_t, Int_t, Int_t) [lulesh-init.cc:308]
          2312  ../examples/lulesh/lulesh-init.cc         Domain::CreateRegionIndexSets                                                            Domain::CreateRegionIndexSets(Domain *, Int_t, Int_t) [lulesh-init.cc:409]
          7109  ../examples/lulesh/lulesh-init.cc         Domain::Domain                                                                           Domain::Domain(Domain *, Int_t, Index_t, Index_t, Index_t, Index_t, int, int,...
          2458  ../examples/lulesh/lulesh-init.cc         Domain::SetupBoundaryConditions                                                          Domain::SetupBoundaryConditions(Domain *, Int_t) [lulesh-init.cc:409]
           956  ../examples/lulesh/lulesh-init.cc         Domain::SetupCommBuffers                                                                 Domain::SetupCommBuffers(Domain *, Int_t) [lulesh-init.cc]
          1456  ../examples/lulesh/lulesh-init.cc         Domain::SetupElementConnectivities                                                       Domain::SetupElementConnectivities(Domain *, Int_t) [lulesh-init.cc:409]
           721  ../examples/lulesh/lulesh-init.cc         Domain::SetupSymmetryPlanes                                                              Domain::SetupSymmetryPlanes(Domain *, Int_t) [lulesh-init.cc:409]
          1591  ../examples/lulesh/lulesh-init.cc         Domain::SetupThreadSupportStructures                                                     Domain::SetupThreadSupportStructures(Domain *) [lulesh-init.cc:376]
          1644  ../examples/lulesh/lulesh-init.cc         Domain::~Domain                                                                          Domain::~Domain(Domain *) [lulesh-init.cc:286]
           271  ../examples/lulesh/lulesh-init.cc         Kokkos::StaticCrsGraph<int, Kokkos::LayoutLeft, Kokkos::OpenMP, Kokkos::Memor...         Kokkos::StaticCrsGraph<int, Kokkos::LayoutLeft, Kokkos::OpenMP, Kokkos::Memor...
           410  ../examples/lulesh/lulesh-init.cc         Kokkos::View<int*, Kokkos::HostSpace>::View<char [10]>                                   Kokkos::View<int*, Kokkos::HostSpace>::View<char [10]>(View<int *, Kokkos::Ho...
           410  ../examples/lulesh/lulesh-init.cc         Kokkos::View<int*, Kokkos::HostSpace>::View<char [14]>                                   Kokkos::View<int*, Kokkos::HostSpace>::View<char [14]>(View<int *, Kokkos::Ho...
           410  ../examples/lulesh/lulesh-init.cc         Kokkos::View<int*>::View<char [16]>                                                      Kokkos::View<int*>::View<char [16]>(View<int *> *, arg_label, type, const siz...
           410  ../examples/lulesh/lulesh-init.cc         Kokkos::View<int*>::View<char [19]>                                                      Kokkos::View<int*>::View<char [19]>(View<int *> *, arg_label, type, const siz...
           410  ../examples/lulesh/lulesh-init.cc         Kokkos::View<int*>::View<char [21]>                                                      Kokkos::View<int*>::View<char [21]>(View<int *> *, arg_label, type, const siz...
          6589  ../examples/lulesh/lulesh-init.cc         Kokkos::deep_copy<double*, , double*, Kokkos::LayoutRight, Kokkos::Device<Kok...         Kokkos::deep_copy<double*, , double*, Kokkos::LayoutRight, Kokkos::Device<Kok...
          1052  ../examples/lulesh/lulesh-init.cc         Kokkos::deep_copy<double*>                                                               Kokkos::deep_copy<double*>(dst, value) [lulesh-init.cc]
          1050  ../examples/lulesh/lulesh-init.cc         Kokkos::deep_copy<double, Kokkos::LayoutRight, Kokkos::Device<Kokkos::OpenMP,...         Kokkos::deep_copy<double, Kokkos::LayoutRight, Kokkos::Device<Kokkos::OpenMP,...
          7686  ../examples/lulesh/lulesh-init.cc         Kokkos::deep_copy<int* [8], Kokkos::LayoutRight, Kokkos::Device<Kokkos::OpenM...         Kokkos::deep_copy<int* [8], Kokkos::LayoutRight, Kokkos::Device<Kokkos::OpenM...
          7686  ../examples/lulesh/lulesh-init.cc         Kokkos::deep_copy<int* [8], Kokkos::LayoutRight, int* [8], Kokkos::LayoutRigh...         Kokkos::deep_copy<int* [8], Kokkos::LayoutRight, int* [8], Kokkos::LayoutRigh...
          6589  ../examples/lulesh/lulesh-init.cc         Kokkos::deep_copy<int*, , int*, Kokkos::LayoutRight, Kokkos::Device<Kokkos::O...         Kokkos::deep_copy<int*, , int*, Kokkos::LayoutRight, Kokkos::Device<Kokkos::O...
          6589  ../examples/lulesh/lulesh-init.cc         Kokkos::deep_copy<int*, Kokkos::LayoutLeft, Kokkos::Device<Kokkos::OpenMP, Ko...         Kokkos::deep_copy<int*, Kokkos::LayoutLeft, Kokkos::Device<Kokkos::OpenMP, Ko...
          6589  ../examples/lulesh/lulesh-init.cc         Kokkos::deep_copy<int*, Kokkos::LayoutRight, Kokkos::Device<Kokkos::OpenMP, K...         Kokkos::deep_copy<int*, Kokkos::LayoutRight, Kokkos::Device<Kokkos::OpenMP, K...
           697  ../examples/lulesh/lulesh-init.cc         Kokkos::parallel_for<Kokkos::MDRangePolicy<Kokkos::OpenMP, Kokkos::Rank<2u, (...         Kokkos::parallel_for<Kokkos::MDRangePolicy<Kokkos::OpenMP, Kokkos::Rank<2u, (...
           706  ../examples/lulesh/lulesh-init.cc         Kokkos::parallel_for<Kokkos::MDRangePolicy<Kokkos::OpenMP, Kokkos::Rank<2u, (...         Kokkos::parallel_for<Kokkos::MDRangePolicy<Kokkos::OpenMP, Kokkos::Rank<2u, (...
           912  ../examples/lulesh/lulesh-init.cc         Kokkos::parallel_for<Kokkos::RangePolicy<Kokkos::OpenMP, Kokkos::IndexType<in...         Kokkos::parallel_for<Kokkos::RangePolicy<Kokkos::OpenMP, Kokkos::IndexType<in...
           791  ../examples/lulesh/lulesh-init.cc         Kokkos::parallel_for<Kokkos::RangePolicy<Kokkos::OpenMP, Kokkos::IndexType<in...         Kokkos::parallel_for<Kokkos::RangePolicy<Kokkos::OpenMP, Kokkos::IndexType<in...
           791  ../examples/lulesh/lulesh-init.cc         Kokkos::parallel_for<Kokkos::RangePolicy<Kokkos::OpenMP, Kokkos::IndexType<in...         Kokkos::parallel_for<Kokkos::RangePolicy<Kokkos::OpenMP, Kokkos::IndexType<in...
           944  ../examples/lulesh/lulesh-init.cc         Kokkos::parallel_for<Kokkos::RangePolicy<Kokkos::OpenMP, Kokkos::IndexType<lo...         Kokkos::parallel_for<Kokkos::RangePolicy<Kokkos::OpenMP, Kokkos::IndexType<lo...
           839  ../examples/lulesh/lulesh-init.cc         Kokkos::parallel_for<Kokkos::RangePolicy<Kokkos::OpenMP, Kokkos::IndexType<lo...         Kokkos::parallel_for<Kokkos::RangePolicy<Kokkos::OpenMP, Kokkos::IndexType<lo...
          6589  ../examples/lulesh/lulesh-util.cc         Kokkos::deep_copy<double*, Kokkos::LayoutRight, Kokkos::Device<Kokkos::OpenMP...         Kokkos::deep_copy<double*, Kokkos::LayoutRight, Kokkos::Device<Kokkos::OpenMP...
          1345  ../examples/lulesh/lulesh-util.cc         ParseCommandLineOptions                                                                  ParseCommandLineOptions(int, char * *, int, cmdLineOpts *) [lulesh-util.cc:67]
           706  ../examples/lulesh/lulesh-util.cc         VerifyAndWriteFinalOutput                                                                VerifyAndWriteFinalOutput(Real_t, locDom, Int_t, Int_t) [lulesh-util.cc:222]
         13367  ../examples/lulesh/lulesh.cc              ApplyMaterialPropertiesForElems                                                          ApplyMaterialPropertiesForElems(domain) [lulesh.cc:409]
           982  ../examples/lulesh/lulesh.cc              CalcElemFBHourglassForce                                                                 CalcElemFBHourglassForce(const Real_t *, const Real_t[] *, coefficient, Real_...
          2428  ../examples/lulesh/lulesh.cc              CalcElemNodeNormals                                                                      CalcElemNodeNormals(Real_t *, Real_t *, Real_t *, const Real_t *, const Real_...
           853  ../examples/lulesh/lulesh.cc              CalcElemShapeFunctionDerivatives                                                         CalcElemShapeFunctionDerivatives(const Real_t *, const Real_t *, const Real_t...
          1054  ../examples/lulesh/lulesh.cc              CalcKinematicsForElems                                                                   CalcKinematicsForElems(domain, Real_t, Index_t) [lulesh.cc]
         14160  ../examples/lulesh/lulesh.cc              CalcVolumeForceForElems                                                                  CalcVolumeForceForElems(domain) [lulesh.cc:409]
           366  ../examples/lulesh/lulesh.cc              Domain::AllocateGradients                                                                Domain::AllocateGradients(Domain *, Int_t, Int_t) [lulesh.cc:214]
           475  ../examples/lulesh/lulesh.cc              Domain::DeallocateGradients                                                              Domain::DeallocateGradients(Domain *) [lulesh.cc:105]
          4356  ../examples/lulesh/lulesh.cc              Domain::Domain                                                                           Domain::Domain(Domain *) [lulesh.cc:78]
           410  ../examples/lulesh/lulesh.cc              Kokkos::View<double*>::View<char [6]>                                                    Kokkos::View<double*>::View<char [6]>(View<double *> *, arg_label, type, cons...
           410  ../examples/lulesh/lulesh.cc              Kokkos::View<double*>::View<char [7]>                                                    Kokkos::View<double*>::View<char [7]>(View<double *> *, arg_label, type, cons...
           928  ../examples/lulesh/lulesh.cc              Kokkos::parallel_for<Kokkos::RangePolicy<Kokkos::OpenMP, Kokkos::IndexType<in...         Kokkos::parallel_for<Kokkos::RangePolicy<Kokkos::OpenMP, Kokkos::IndexType<in...
           960  ../examples/lulesh/lulesh.cc              Kokkos::parallel_for<Kokkos::RangePolicy<Kokkos::OpenMP, Kokkos::IndexType<lo...         Kokkos::parallel_for<Kokkos::RangePolicy<Kokkos::OpenMP, Kokkos::IndexType<lo...
         21470  ../examples/lulesh/lulesh.cc              LagrangeLeapFrog                                                                         LagrangeLeapFrog(domain) [lulesh.cc]
          1836  ../examples/lulesh/lulesh.cc              main                                                                                     int main(int, char * *) [lulesh.cc]
```

## Sampling

By default, omnitrace uses `--mode trace` for instrumentation. The `--mode sampling` option
will only instrument `main` in an executable and will activate both CPU call-stack sampling and
background system-level thread sampling by default.
Tracing capabilities which do not rely on instrumentation, such as the HIP API and kernel tracing
(which is collected via roctracer), will still be available.

[Omnitrace](https://github.com/AMDResearch/omnitrace)'s sampling capabilities are always available, even in trace mode, but is deactivated by default.
In order to activate sampling in trace mode, simply set `OMNITRACE_USE_SAMPLING=ON` in the environment
or in an omnitrace configuration file.

## Embedding a Default Configuration

Using the `--env` option, a default configuration can be embedded into the target. Although this option
works for runtime instrumentation, it is most useful when generating new binaries since the generated
binary may be used later in a different login sessions when the environment may have changed.

For example, if the following sequence of commands are run:

```shell
omnitrace -o ./foo.inst -- ./foo
export OMNITRACE_USE_SAMPLING=ON
export OMNITRACE_SAMPLING_FREQ=5
./foo.inst
```

These configuration settings will not be preserved in another session, whereas:

```shell
omnitrace -o ./foo.samp --env OMNITRACE_USE_SAMPLING=ON OMNITRACE_SAMPLING_FREQ=5 -- ./foo
```

will preserve those environment variables:

```shell
# will sample 5x per second
./foo.samp
```

while still allowing the subsequent session to override those defaults:

```shell
# will sample 100x per second
export OMNITRACE_SAMPLING_FREQ=100
./foo.samp
```

### Troubleshooting

#### Checking for RPATH

If `ldd ./foo.inst` from the [Binary Rewriting a Library Example](#binary-rewriting-a-library-example) section still returned `/usr/local/lib/libfoo.so.2`, your executable may have an rpath encoded in the binary.
This ELF entry will result in the dynamic linker to ignore `LD_LIBRARY_PATH` if it finds a `libfoo.so.2` in the rpath.
You can use the `objdump` tool to perform this query:

```shell
objdump -p <exe-or-library> | egrep 'RPATH|RUNPATH'
```

If this produces output, e.g.:

```shell
  RUNPATH              $ORIGIN:$ORIGIN/../lib
```

You will have to remove or modify the rpath in order to get `foo.inst` to resolve to the instrumented `libfoo.so.2`

#### Modifying RPATH

> ***Requires `patchelf` package***

```shell
patchelf --remove-rpath <exe-or-library>
patchelf --set-rpath '/home/user' <exe-or-library>
```
