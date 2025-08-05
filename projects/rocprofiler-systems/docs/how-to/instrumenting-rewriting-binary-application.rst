.. meta::
   :description: ROCm Systems Profiler binary instrumentation and rewrite documentation and reference
   :keywords: rocprof-sys, rocprofiler-systems, Omnitrace, ROCm, binary instrumentation, binary rewrite, profiler, tracking, visualization, tool, Instinct, accelerator, AMD

****************************************************
Instrumenting and rewriting a binary application
****************************************************

There are three ways to perform instrumentation with the ``rocprof-sys-instrument`` executable:

* Runtime instrumentation
* Attaching to an already running process
* Binary rewrite

Here is a comparison of the three modes:

* Runtime instrumentation of the application using the ``rocprof-sys-instrument`` executable
  (analogous to ``gdb --args <program> <args>``)

  * This mode is the default if neither the ``-p`` nor ``-o`` command-line options are used
  * Runtime instrumentation supports instrumenting not only the target executable but also
    the shared libraries loaded by the target executable. Consequently, this mode consumes more memory,
    takes longer to perform the instrumentation, and tends to add more significant overhead to the
    runtime of the application.
  * This mode is recommended if you want to analyze not only the performance of your executable and/or
    libraries but also the performance of the library dependencies

* Attaching to a process that is currently running (analogous to ``gdb -p <PID>``)

  * This mode is activated using ``-p <PID>``
  * The same caveats from the first example apply with respect to memory and overhead

  .. note::

     Attaching to a running process is an alpha feature and detaching from the target process
     without ending the target process is not currently supported.

* Binary rewrite to generate a new executable or library with the instrumentation built-in

  * This mode is activated through the ``-o <output-file>`` option
  * Binary rewriting is limited to the text section of the target executable or library. It does not instrument
    the dynamically-linked libraries. Consequently, this mode performs the
    instrumentation significantly faster
    and has a much lower overhead when running the instrumented executable and libraries.
  * Binary rewriting is the recommended mode when the target executable uses
    process-level parallelism (for example, MPI)
  * If the target executable has a minimal ``main`` routine and the bulk of your
    application is in one specific dynamic library,
    see :ref:`binary-rewriting-library-label` for help

The rocprof-sys-instrument executable
========================================

Instrumentation is performed with the ``rocprof-sys-instrument`` executable. For more details, use the ``-h`` or ``--help`` option to
view the help menu.

.. code-block:: shell

   $ rocprof-sys-instrument --help
   [rocprof-sys-instrument] Usage: rocprof-sys-instrument [ --help (count: 0, dtype: bool)
                                                      --version (count: 0, dtype: bool)
                                                      --verbose (max: 1, dtype: bool)
                                                      --error (max: 1, dtype: boolean)
                                                      --debug (max: 1, dtype: bool)
                                                      --log (count: 1)
                                                      --log-file (count: 1)
                                                      --simulate (max: 1, dtype: boolean)
                                                      --print-format (min: 1, dtype: string)
                                                      --print-dir (count: 1, dtype: string)
                                                      --print-available (count: 1)
                                                      --print-instrumented (count: 1)
                                                      --print-coverage (count: 1)
                                                      --print-excluded (count: 1)
                                                      --print-overlapping (count: 1)
                                                      --print-instructions (max: 1, dtype: bool)
                                                      --output (min: 0, dtype: string)
                                                      --pid (count: 1, dtype: int)
                                                      --mode (count: 1)
                                                      --force (max: 1, dtype: bool)
                                                      --command (count: 1)
                                                      --prefer (count: 1)
                                                      --library (count: unlimited)
                                                      --main-function (count: 1)
                                                      --load (count: unlimited, dtype: string)
                                                      --load-instr (count: unlimited, dtype: filepath)
                                                      --init-functions (count: unlimited, dtype: string)
                                                      --fini-functions (count: unlimited, dtype: string)
                                                      --all-functions (max: 1, dtype: boolean)
                                                      --function-include (count: unlimited)
                                                      --function-exclude (count: unlimited)
                                                      --function-restrict (count: unlimited)
                                                      --caller-include (count: unlimited)
                                                      --module-include (count: unlimited)
                                                      --module-exclude (count: unlimited)
                                                      --module-restrict (count: unlimited)
                                                      --internal-function-include (count: unlimited)
                                                      --internal-module-include (count: unlimited)
                                                      --instruction-exclude (count: unlimited)
                                                      --internal-library-deps (min: 0, dtype: boolean)
                                                      --internal-library-append (count: unlimited)
                                                      --internal-library-remove (count: unlimited)
                                                      --linkage (min: 1)
                                                      --visibility (min: 1)
                                                      --label (count: unlimited, dtype: string)
                                                      --config (min: 1, dtype: string)
                                                      --default-components (count: unlimited, dtype: string)
                                                      --env (count: unlimited)
                                                      --mpi (max: 1, dtype: bool)
                                                      --instrument-loops (max: 1, dtype: boolean)
                                                      --min-instructions (count: 1, dtype: int)
                                                      --min-address-range (count: 1, dtype: int)
                                                      --min-instructions-loop (count: 1, dtype: int)
                                                      --min-address-range-loop (count: 1, dtype: int)
                                                      --coverage (max: 1, dtype: bool)
                                                      --dynamic-callsites (max: 1, dtype: boolean)
                                                      --traps (max: 1, dtype: boolean)
                                                      --loop-traps (max: 1, dtype: boolean)
                                                      --allow-overlapping (max: 1, dtype: bool)
                                                      --parse-all-modules (max: 1, dtype: bool)
                                                      --batch-size (count: 1, dtype: int)
                                                      --dyninst-rt (min: 1, dtype: filepath)
                                                      --dyninst-options (count: unlimited)
                                                      ] -- <CMD> <ARGS>

   Options:
      -h, -?, --help                 Shows this page
      --version                      Prints the version and exit

      [DEBUG OPTIONS]

      -v, --verbose                  Verbose output
      -e, --error                    All warnings produce runtime errors
      --debug                        Debug output
      --log                          Number of log entries to display after an error. Any value < 0 will emit the entire log
      --log-file                     Write the log out the specified file during the run
      --simulate                     Exit after outputting diagnostic {available,instrumented,excluded,overlapping} module
                                    function lists, e.g. available.txt
      --print-format [ json | txt | xml ]
                                    Output format for diagnostic {available,instrumented,excluded,overlapping} module
                                    function lists, e.g. {print-dir}/available.txt
      --print-dir                    Output directory for diagnostic {available,instrumented,excluded,overlapping} module
                                    function lists, e.g. {print-dir}/available.txt
      --print-available [ functions | functions+ | modules | pair | pair+ ]
                                    Print the available entities for instrumentation (functions, modules, or module-function
                                    pair) to stdout after applying regular expressions
      --print-instrumented [ functions | functions+ | modules | pair | pair+ ]
                                    Print the instrumented entities (functions, modules, or module-function pair) to stdout
                                    after applying regular expressions
      --print-coverage [ functions | functions+ | modules | pair | pair+ ]
                                    Print the instrumented coverage entities (functions, modules, or module-function pair) to
                                    stdout after applying regular expressions
      --print-excluded [ functions | functions+ | modules | pair | pair+ ]
                                    Print the entities for instrumentation (functions, modules, or module-function pair)
                                    which are excluded from the instrumentation to stdout after applying regular expressions
      --print-overlapping [ functions | functions+ | modules | pair | pair+ ]
                                    Print the entities for instrumentation (functions, modules, or module-function pair)
                                    which overlap other function calls or have multiple entry points to stdout after applying
                                    regular expressions
      --print-instructions           Print the instructions for each basic-block in the JSON/XML outputs

      [MODE OPTIONS]

      -o, --output                   Enable generation of a new executable (binary-rewrite). If a filename is not provided,
                                    rocprof-sys will use the basename and output to the cwd, unless the target binary is in the
                                    cwd. In the latter case, rocprof-sys will either use ${PWD}/<basename>.inst (non-libraries)
                                    or ${PWD}/instrumented/<basename> (libraries)
      -p, --pid                      Connect to running process
      -M, --mode [ coverage | sampling | trace ]
                                    Instrumentation mode. \'trace\' mode instruments the selected functions, \'sampling\' mode
                                    only instruments the main function to start and stop the sampler.
      -f, --force                    Force the command-line argument configuration, i.e. don't get cute. Useful for forcing
                                    runtime instrumentation of an executable that [A] Dyninst thinks is a library after
                                    reading ELF and [B] whose name makes it look like a library (e.g. starts with 'lib'
                                    and/or ends in \'.so\', \'.so.*\', or \'.a\')
      -c, --command                  Input executable and arguments (if \'-- <CMD>\' not provided)

      [LIBRARY OPTIONS]

      --prefer [ shared | static ]   Prefer this library types when available
      -L, --library                  Libraries with instrumentation routines (default: "librocprof-sys-dl")
      -m, --main-function            The primary function to instrument around, e.g. \'main\'
      --load                         Supplemental instrumentation library names w/o extension (e.g. \'libinstr\' for
                                    \'libinstr.so\' or \'libinstr.a\')
      --load-instr                   Load {available,instrumented,excluded,overlapping}-instr JSON or XML file(s) and override
                                    what is read from the binary
      --init-functions               Initialization function(s) for supplemental instrumentation libraries (see \'--load\'
                                    option)
      --fini-functions               Finalization function(s) for supplemental instrumentation libraries (see \'--load\' option)
      --all-functions                When finding functions, include the functions which are not instrumentable. This is
                                    purely diagnostic for the available/excluded functions output

      [SYMBOL SELECTION OPTIONS]

      -I, --function-include         Regex(es) for including functions (despite heuristics)
      -E, --function-exclude         Regex(es) for excluding functions (always applied)
      -R, --function-restrict        Regex(es) for restricting functions only to those that match the provided
                                    regular-expressions
      --caller-include               Regex(es) for including functions that call the listed functions (despite heuristics)
      -MI, --module-include          Regex(es) for selecting modules/files/libraries (despite heuristics)
      -ME, --module-exclude          Regex(es) for excluding modules/files/libraries (always applied)
      -MR, --module-restrict         Regex(es) for restricting modules/files/libraries only to those that match the provided
                                    regular-expressions
      --internal-function-include    Regex(es) for including functions which are (likely) utilized by rocprof-sys itself. Use
                                    this option with care.
      --internal-module-include      Regex(es) for including modules/libraries which are (likely) utilized by rocprof-sys
                                    itself. Use this option with care.
      --instruction-exclude          Regex(es) for excluding functions containing certain instructions
      --internal-library-deps        Treat the libraries linked to the internal libraries as internal libraries. This increase
                                    the internal library processing time and consume more memory (so use with care) but may
                                    be useful when the application uses Boost libraries and Dyninst is dynamically linked
                                    against the same boost libraries
      --internal-library-append      Append to the list of libraries which rocprof-sys treats as being used internally, e.g.
                                    ROCm Systems Profiler will find all the symbols in this library and prevent them from being
                                    instrumented.
      --internal-library-remove [ ld-linux-x86-64.so.2
                                 libBrokenLocale.so.1
                                 libanl.so.1
                                 libbfd.so
                                 libbz2.so
                                 libc.so.6
                                 libcaliper.so
                                 libcommon.so
                                 libcrypt.so.1
                                 libdl.so.2
                                 libdw.so
                                 libdwarf.so
                                 libdyninstAPI_RT.so
                                 libelf.so
                                 libgcc_s.so.1
                                 libgotcha.so
                                 liblikwid.so
                                 liblzma.so
                                 libnsl.so.1
                                 libnss_compat.so.2
                                 libnss_db.so.2
                                 libnss_dns.so.2
                                 libnss_files.so.2
                                 libnss_hesiod.so.2
                                 libnss_ldap.so.2
                                 libnss_nis.so.2
                                 libnss_nisplus.so.2
                                 libnss_test1.so.2
                                 libnss_test2.so.2
                                 libpapi.so
                                 libpfm.so
                                 libprofiler.so
                                 libpthread.so.0
                                 libresolv.so.2
                                 libamd_smi64.so
                                 librocprofiler-sdk.so
                                 librt.so.1
                                 libstdc++.so.6
                                 libtbb.so
                                 libtbbmalloc.so
                                 libtbbmalloc_proxy.so
                                 libtcmalloc.so
                                 libtcmalloc_and_profiler.so
                                 libtcmalloc_debug.so
                                 libtcmalloc_minimal.so
                                 libtcmalloc_minimal_debug.so
                                 libthread_db.so.1
                                 libunwind-coredump.so
                                 libunwind-generic.so
                                 libunwind-ptrace.so
                                 libunwind-setjmp.so
                                 libunwind-x86_64.so
                                 libunwind.so
                                 libutil.so.1
                                 libz.so
                                 libzstd.so ]
                                    Remove the specified libraries from being treated as being used internally, e.g.
                                    ROCm System Profiler will permit all the symbols in these libraries to be eligible for
                                    instrumentation.
      --linkage [ global | local | unique | unknown | weak ]
                                    Only instrument functions with specified linkage (default: global, local, unique)
      --visibility [ default | hidden | internal | protected | unknown ]
                                    Only instrument functions with specified visibility (default: default, internal, hidden,
                                    protected)

      [RUNTIME OPTIONS]

      --label [ args | file | line | return ]
                                    Labeling info for functions. By default, just the function name is recorded. Use these
                                    options to gain more information about the function signature or location of the
                                    functions
      -C, --config                   Read in a configuration file and encode these values as the defaults in the executable
      -d, --default-components       Default components to instrument (only useful when timemory is enabled in rocprof-sys
                                    library)
      --env                          Environment variables to add to the runtime in form VARIABLE=VALUE. E.g. use \'--env
                                    ROCPROFSYS_PROFILE=ON\' to default to using timemory instead of perfetto
      --mpi                          Enable MPI support (requires rocprof-sys built w/ full or partial MPI support). NOTE: this
                                    will automatically be activated if MPI_Init, MPI_Init_thread, MPI_Finalize,
                                    MPI_Comm_rank, or MPI_Comm_size are found in the symbol table of target

      [GRANULARITY OPTIONS]

      -l, --instrument-loops         Instrument at the loop level
      -i, --min-instructions         If the number of instructions in a function is less than this value, exclude it from
                                    instrumentation
      -r, --min-address-range        If the address range of a function is less than this value, exclude it from
                                    instrumentation
      --min-instructions-loop        If the number of instructions in a function containing a loop is less than this value,
                                    exclude it from instrumentation
      --min-address-range-loop       If the address range of a function containing a loop is less than this value, exclude it
                                    from instrumentation
      --coverage [ basic_block | function | none ]
                                    Enable recording the code coverage. If instrumenting in coverage mode (\'-M converage\'),
                                    this simply specifies the granularity. If instrumenting in trace or sampling mode, this
                                    enables recording code-coverage in addition to the instrumentation of that mode (if any).
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
      --parse-all-modules            By default, rocprof-sys simply requests Dyninst to provide all the procedures in the
                                    application image. If this option is enabled, rocprof-sys will iterate over all the modules
                                    and extract the functions. Theoretically, it should be the same but the data is slightly
                                    different, possibly due to weak binding scopes. In general, enabling option will probably
                                    have no visible effect

      [DYNINST OPTIONS]

      -b, --batch-size               Dyninst supports batch insertion of multiple points during runtime instrumentation. If
                                    one large batch insertion fails, this value will be used to create smaller batches.
                                    Larger batches generally decrease the instrumentation time
      --dyninst-rt                   Path(s) to the dyninstAPI_RT library
      --dyninst-options [ BaseTrampDeletion
                           DebugParsing
                           DelayedParsing
                           InstrStackFrames
                           MergeTramp
                           SaveFPR
                           TrampRecursive
                           TypeChecking ]
      Advanced dyninst options: BPatch::set<OPTION>(bool), e.g. bpatch->setTrampRecursive(true)

``rocprof-sys-instrument`` uses a similar syntax as LLVM to separate command-line arguments from the
application's arguments. It uses a standalone
double-hyphen (``--``) as a separator.
All arguments preceding the double-hyphen
are interpreted as belonging to ROCm Systems Profiler and all arguments following the
double-hyphen are interpreted as being part of the
application and its arguments. In binary rewrite mode, all application arguments after the first argument
are ignored. As an example, ``./rocprof-sys-instrument -o ls.inst -- ls -l`` interprets ``ls`` as
the target to instrument, ignoring the ``-l`` argument,
and generates a ``ls.inst`` executable that you can subsequently run using the
``rocprof-sys-run -- ls.inst -l`` command.

Runtime instrumentation example
========================================

The following example shows how to enable runtime instrumentation.

.. code-block:: shell

   rocprof-sys-instrument <rocprof-sys-options> -- <exe> [<exe-options>...]

Attaching to a running process
========================================

Use the following command to attach to an active process.

.. code-block:: shell

   rocprof-sys-instrument <rocprof-sys-options> -p <PID> -- <exe-name>

Binary rewrite
========================================

This example demonstrates how to rewrite a binary.

.. code-block:: shell

   rocprof-sys-instrument <rocprof-sys-options> -o <name-of-new-exe-or-library> -- <exe-or-library>

.. _binary-rewriting-library-label:

Binary rewrite of a library
-----------------------------------

Many applications bundle the bulk of their functionality into one or more
dynamic libraries and have a relatively simple ``main``
which links to these libraries and serves as the "driver" for
setting up the workflow. If you perform a binary rewrite of an
executable like this and find there is insufficient information, you
can either switch to runtime instrumentation or perform a
binary rewrite on the relevant libraries.

Support for stand-alone binary rewriting of a dynamic library without a binary rewrite of
the executable is a beta feature.
In general, it is supported as long as the library contains the ``_init`` and
``_fini`` symbols but these symbols are not
standardized to the extent of ``main`` in an executable.

Here is the recommended workflow for the binary rewrite of a library:

#. Determine the names of the dynamically linked libraries of interest using ``ldd``
#. Generate a binary rewrite of the executable
#. Generate a binary rewrite of the desired libraries with the same base name as the
   original library, for example, ``libfoo.so.2`` instead of ``libfoo.so``,  and output the instrumented
   library into a different folder than the original library.

#. Prefix the ``LD_LIBRARY_PATH`` executable with the output folder from the previous step
#. Use ``ldd`` to verify that the instrumented executable can resolve the location of the instrumented library

Binary rewrite of a library example
-----------------------------------

The ``foo`` executable is dynamically linked to ``libfoo.so.2``:

.. code-block:: shell

   $ pwd
   /home/user
   $ which foo
   /usr/local/bin/foo
   $ ldd /usr/local/bin/foo
         ...
         libfoo.so.2 => /usr/local/lib/libfoo.so.2 (...)
         ...

Generate binary rewrites of ``foo`` and ``libfoo.so.2``:

.. code-block:: shell

   rocprof-sys-instrument -o ./foo.inst -- foo
   rocprof-sys-instrument -o ./libfoo.so.2 -- /usr/local/lib/libfoo.so.2

At this point, the instrumented ``foo.inst`` executable still dynamically loads the
original ``libfoo.so.2`` in ``/usr/local/lib``:

.. code-block:: shell

   $ ldd ./foo.inst
         ...
         libfoo.so.2 => /usr/local/lib/libfoo.so.2 (...)
         ...

Prefix the ``LD_LIBRARY_PATH`` environment variable with the folder containing
the instrumented ``libfoo.so.2``:

.. code-block:: shell

   export LD_LIBRARY_PATH=/home/user:${LD_LIBRARY_PATH}

``foo.inst`` now loads the instrumented library when it runs:

.. code-block:: shell

   $ ldd ./foo.inst
         ...
         libfoo.so.2 => /home/user/libfoo.so.2 (...)
         ...

Selective instrumentation
========================================

The default behavior of ``rocprof-sys-instrument`` does not instrument every symbol in the binary.
The default rules are:

* Skip instrumenting dynamic call-sites (such as function pointers)

  * The ``--dynamic-callsites`` option forces instrumentation for all dynamic call-sites

* The cost of a function can be loosely approximated by the number of
  instructions. By default, ``rocprof-sys-instrument`` only instruments functions
  with at least 1024 instructions

  * The  ``--min-instructions`` option modifies this heuristic for all functions which do not contain loops
  * The ``--min-instructions-loop`` option modifies this heuristic for functions which contain loops.

* The cost of a function can be also be loosely approximated by the size of the function
  in the binary so this heuristic can be used in lieu of or in addition to the
  minimum number of instructions

  * The ``--min-address-range`` option modifies this heuristic for all functions which do not contain loops
  * The ``--min-address-range-loop`` option modifies this heuristic for functions which contain loops

* Skip instrumentation points which require using a trap

  * See the description for the ``--traps`` and ``--loop-traps`` options for more information

* Skip instrumenting loops within the body of a function

  * The ``--instrument-loops`` option enables this behavior

* Skip instrumenting functions with overlapping function bodies and single
  functions with multiple entry point

  * These behaviors arise from various optimizations. Enable instrumenting for these functions
    by using the ``--allow-overlapping`` option

.. note::

   The separate loop options ``--min-instructions-loop`` and ``--min-address-range-loop``
   are provided because functions with loops can be compact in the binary while also being costly

Viewing the available, instrumented, excluded, and overlapping functions
-------------------------------------------------------------------------

Whenever ``rocprof-sys-instrument`` runs with a verbosity of zero or higher,
it generates files that detail which functions
were available for instrumentation (along with the module they were defined in), actually instrumented,
excluded, and which contained overlapping function bodies.
By default, these files are saved to the ``rocprof-sys-<NAME>-output`` folder
where ``<NAME>`` is the base name of the targeted binary (or
the base name of the resulting executable in the case of binary rewrite). For example,
``rocprof-sys-instrument -- ls`` outputs these files to ``rocprof-sys-ls-output``
whereas ``rocprof-sys-instrument -o ls.inst -- ls`` places them in ``rocprof-sys-ls.inst-output``.

To generate these files without running or generating an
executable, use the ``--simulate`` option:

.. code-block:: shell

   rocprof-sys-instrument --simulate -- foo
   rocprof-sys-instrument --simulate -o foo.inst -- foo

Excluding and including modules and functions
----------------------------------------------

ROCm Systems Profiler has a set of six command-line options which each accept one or more
regular expressions for customizing the scope of which module and/or functions are
instrumented. Multiple regex patterns per option are treated as an OR operation,
for example, ``--module-include libfoo libbar`` is effectively the same as ``--module-include 'libfoo|libbar'``.

To force the inclusion of certain modules and/or function
without changing any of the heuristics, use the ``--module-include`` and/or ``--function-include`` options.
These options do not exclude modules or functions which do
not satisfy their regular expression.

To narrow the scope of the instrumentation to a specific set
of libraries and/or functions, use the ``--module-restrict`` and ``--function-restrict`` options.
These options let you exclusively select the union of one or more
regular expressions, regardless of whether or not the functions satisfy the
previously-mentioned default heuristics. Any function or module that is not within
the union of these regular expressions is excluded from instrumentation.

To avoid instrumenting a set of modules and/or functions,
use the ``--module-exclude`` and ``--function-exclude`` options.
These options are always applied, even if the module or function
satisfies the "restrict" or "include" regular expression.

.. _available-module-function-output:

An example of the available module and function info output
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: shell

   rocprof-sys-instrument -o lulesh.inst --label file line args --simulate -- lulesh

.. code-block:: shell

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

An example of instrumented module and function info output
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: shell

   rocprof-sys-instrument -o lulesh.inst --label file line args --simulate -- lulesh

After the heuristics are applied based on the pattern in :ref:`available-module-function-output`,
the selected module and functions are:

.. code-block:: shell

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

Sampling
========================================

.. note::

   This capability has been deprecated in favor of :doc:`Call stack sampling <./sampling-call-stack>`.

By default, ``rocprof-sys-instrument`` uses ``--mode trace`` for instrumentation. The ``--mode sampling`` option
only instruments ``main`` in an executable. It activates both CPU call-stack sampling and
background system-level thread sampling by default.
Tracing capabilities which do not rely on instrumentation, such as the HIP API and kernel tracing
are still available.

The ROCm Systems Profiler sampling capabilities are always available, even in trace mode, but are deactivated by default.
To activate sampling in trace mode, set ``ROCPROFSYS_USE_SAMPLING=ON`` in the environment
or in an ROCm Systems Profiler configuration file.

Embedding a default configuration
========================================

Use the ``--env`` option to embed a default configuration into the target. Although this option
works for runtime instrumentation, it is most useful when generating new binaries because the generated
binary can be used later on in a different login session when the environment might have changed.

For example, if the following commands are run,
the configuration settings are not be preserved for subsequent sessions:

.. code-block:: shell

   rocprof-sys-instrument -o ./foo.inst -- ./foo
   export ROCPROFSYS_USE_SAMPLING=ON
   export ROCPROFSYS_SAMPLING_FREQ=5
   rocprof-sys-run -- ./foo.inst

Whereas the following command preserves those environment variables:

.. code-block:: shell

   rocprof-sys-instrument -o ./foo.samp --env ROCPROFSYS_USE_SAMPLING=ON ROCPROFSYS_SAMPLING_FREQ=5 -- ./foo

They can now be used in future sessions.

.. code-block:: shell

   # will sample 5x per second
   rocprof-sys-run -- ./foo.samp

Even though the environment variables are preserved, subsequent sessions can still override those defaults:

.. code-block:: shell

   # will sample 100x per second
   export ROCPROFSYS_SAMPLING_FREQ=100
   rocprof-sys-run -- ./foo.samp

.. _rpath-troubleshooting:

Troubleshooting
----------------------------------------------

Checking for RPATH
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If ``ldd ./foo.inst`` from the :ref:`binary-rewriting-library-label`
section still returns ``/usr/local/lib/libfoo.so.2``, the executable could have
an rpath encoded in the binary.
This ELF entry results in the dynamic linker ignoring ``LD_LIBRARY_PATH`` if
it finds ``libfoo.so.2`` in the rpath.
Using the ``objdump`` tool, perform the following query:

.. code-block:: shell

   objdump -p <exe-or-library> | egrep 'RPATH|RUNPATH'

If this produces output that appears similar to this output.:

.. code-block:: shell

   RUNPATH              $ORIGIN:$ORIGIN/../lib

Remove or modify the rpath to get ``foo.inst`` to resolve
to the instrumented ``libfoo.so.2`` as explained in the next section.

Modifying an RPATH
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This code snippet uses the ``patchelf`` tool to modify the rpath of the given executable
or library to ``/home/user``, which is where the instrumented libraries are located.

.. note::

   This functionality requires the ``patchelf`` package.

.. code-block:: shell

   patchelf --remove-rpath <exe-or-library>
   patchelf --set-rpath '/home/user' <exe-or-library>
