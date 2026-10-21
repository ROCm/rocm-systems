# AGENTS.md

Guidance for AI coding agents working on hipFile. hipFile is AMD's Infinity
Storage (AIS) library for direct-to-GPU file I/O on ROCm, with C and Python
APIs for synchronous, asynchronous, and batch I/O; see [README.md](README.md)
for scope and
[docs/install/build-from-source.rst](docs/install/build-from-source.rst) for
building.

hipFile lives in the [rocm-systems](https://github.com/ROCm/rocm-systems)
monorepo, under `projects/hipfile`. Unless a section says otherwise, run
commands from `projects/hipfile`; paths are relative to it. The CI workflows
are at the top of the monorepo, in
[`.github/workflows/hipfile-*.yml`](../../.github/workflows).

Ignore `nvidia_detail` directories (`src/nvidia_detail`,
`test/nvidia_detail`) and the `CMAKE_HIP_PLATFORM=nvidia` build. That code
isn't production code, isn't well maintained, and CI no longer builds it.

## Current practice and where it's heading

hipFile is older than its sister project hipObject (see
[Sister projects](#sister-projects)), and some of the rules below describe
where hipFile is heading rather than what all of its code does today. Those
rules are marked **Advisory**: existing code doesn't pass them yet, so no
build or CI job can enforce them. Follow them in new and changed code, and
don't make the existing count of findings worse, but don't rewrite unrelated
code to satisfy them unless you're asked to. An advisory rule becomes
mandatory once the code passes it. Rules without the mark apply to
everything.

## Project direction

hipFile is built by [TheRock](https://github.com/ROCm/TheRock), ROCm's build
system, as part of a much larger superproject build. Make changes that keep
working there:

- **Ask before changing compile or link options, dependency handling, or any
  other non-trivial CMake.** A superproject build has requirements that a
  standalone build doesn't show, so a change that works here can break there.
  For example, the GoogleTest programs are compiled and linked with
  `-pthread`, even though it looks redundant, because TheRock's sanitizer
  builds need it.
  Never remove it, and add it to any new GoogleTest program.
- **Link only against libraries TheRock already provides.** Getting a new
  link-time C dependency into TheRock is extremely difficult, and TheRock is
  moving to hermetic builds, where every dependency has to be declared and
  supplied by the build itself. A library from ROCm or one that TheRock
  vendors in its
  [`third-party`](https://github.com/ROCm/TheRock/tree/main/third-party)
  directory (system libraries are under `third-party/sysdeps`) is fine to
  use. Don't add any other link-time dependency without asking, and only
  after very careful consideration.
- **Don't assume hipFile is the top-level CMake project.** Use
  `PROJECT_SOURCE_DIR`/`PROJECT_BINARY_DIR`,
  `CMAKE_CURRENT_SOURCE_DIR`/`CMAKE_CURRENT_BINARY_DIR`, or the
  `HIPFILE_*_PATH` variables the top-level `CMakeLists.txt` sets, not
  `CMAKE_SOURCE_DIR`/`CMAKE_BINARY_DIR`. Prefix new cache variables with
  `HIPFILE_` and new CMake functions with `hipfile_`, and set flags per
  target, as `hipfile_set_compiler_flags()`
  ([`cmake/HipFileCompilerOptions.cmake`](cmake/HipFileCompilerOptions.cmake))
  does, rather than globally.
- **The `AIS_*` CMake options are deprecated** aliases for the `HIPFILE_*`
  ones ([`cmake/HipFileDeprecatedAISOptions.cmake`](cmake/HipFileDeprecatedAISOptions.cmake)).
  Don't add new `AIS_*` options or new uses of the old names. The
  `AIS_TESTING` compile definition is separate, and is still the name the
  code uses.
- **rocprofiler-sdk depends on hipFile's API dispatch table.** Its
  `hipfile/abi.cpp` checks the table's layout and version with its own
  `static_assert`s, so a change to the table can break the profiler's build
  after hipFile's PR merges. See [Public API](#public-api).

### Sister projects

The same team writes two sister projects:

- **hipObject**, RDMA-accelerated S3 object I/O for AMD GPUs, in its own
  repository for now ([ROCm/hipObject](https://github.com/ROCm/hipObject)),
  and expected to move into rocm-systems. It started from hipFile's
  conventions (its `.clang-format` and warning-suppression header came from
  here), but it's newer and its code and checks are stricter: dependencies
  passed in rather than looked up globally, Clang thread safety annotations,
  `-Wunsafe-buffer-usage` on, a coverage target, and more linters. It's the
  model for most of the **Advisory** rules here.
- **rocm-ernic**, in its own repository
  ([ROCm/rocm-ernic](https://github.com/ROCm/rocm-ernic)).

Borrow ideas from them freely. Ideas can flow the other way too. If you find
something in hipFile that one of them should adopt, or a bug or improvement in
them, ask whether to create a to-do file for it; don't create one or edit the
other project without asking.

## Toolchain and paths

- **ROCm**: If `ROCM_PATH` isn't set as a CMake or environment variable, it
  defaults to `/opt/rocm/core` when that holds a ROCm installation (the ROCm
  7.11+ layout) and to `/opt/rocm` otherwise; setting `ROCM_VERSION` changes
  this (see the top-level [`CMakeLists.txt`](CMakeLists.txt)). Put
  `/opt/rocm/bin` on `PATH` when running binaries you build.
- **Compiler**: hipFile is C++20 plus one HIP kernel
  (`src/amd_detail/backend/memcpy-kernel.hip`), so the project enables the C,
  CXX, and HIP languages, and configuring needs ROCm's HIP compiler.
  `amdclang++` is the usual C++ compiler, and the one TheRock uses, but the
  code has to build with upstream Clang and GCC too (see
  [C and C++](#c-and-c)). Set `CMAKE_HIP_ARCHITECTURES` to limit the GPU
  targets the kernel is built for; CI's list is in
  [hipfile-build.yml](../../.github/workflows/hipfile-build.yml).

## Configure and build (default agent flow)

```bash
cmake -S . -B <build-dir> -DCMAKE_CXX_COMPILER=amdclang++
cmake --build <build-dir> --parallel
```

The library is `src/amd_detail/libhipfile.so` in the build directory.

**Delete your build directories when you're done with them**, unless you're
asked to keep them. Leftover builds fill `/tmp` and break other work on the
machine.

Notes:

- The default build type is `RelWithDebInfo`, and `BUILD_TESTING` is on by
  default.
- `HIPFILE_CXX_STANDARD` selects C++20 (the default), 23, or 26.
- CI's build workflow
  ([hipfile-build.yml](../../.github/workflows/hipfile-build.yml), started
  by [hipfile-ci-toplevel.yml](../../.github/workflows/hipfile-ci-toplevel.yml))
  builds with `amdclang++`, `clang++`, and `g++` across several Linux
  distributions and ROCm versions (the matrix is in
  `hipfile-ci-toplevel.yml`). Each job builds `Debug` and
  `Release`, both with `-DCMAKE_CXX_FLAGS=-Werror`, runs the unit tests in
  each, then packages the Debug build with `cpack` and installs the package.
  The CI images are built from
  [`util/docker`](util/docker).
- One fixed image also runs the sanitizer builds
  ([hipfile-sanitizers.yml](../../.github/workflows/hipfile-sanitizers.yml)),
  clang-tidy
  ([hipfile-clang-tidy.yml](../../.github/workflows/hipfile-clang-tidy.yml)),
  CodeQL
  ([hipfile-codeql.yml](../../.github/workflows/hipfile-codeql.yml)), and the
  profiler ABI check
  ([hipfile-profiler-abi.yml](../../.github/workflows/hipfile-profiler-abi.yml)).

## Tests

- After a successful build, from the build directory:
  `ctest --output-on-failure -L unit`. The ctest labels are:
  - `unit`: the GoogleTest suites that don't need a GPU (`internal_tests`,
    which tests the library's internals with mocks, plus the CMake test of the
    deprecated `AIS_*` options). CI runs these in a plain container.
  - `system`: `hipfile_system_tests`, which needs a GPU and does real I/O in
    `HIPFILE_CAPABLE_DIR` (default `/tmp`). The fastpath tests need storage
    that supports it (check with `tools/ais-check/ais-check`); they fail
    without it unless configured with
    `-DHIPFILE_ALLOW_SKIP_FASTPATH_TESTS=ON`.
  - `stress`: `state_mt` and `batch_mt`, short concurrency tests run under
    `gdb`. They can pass for a while after a bug lands, so a failure may not
    come from the change you're looking at.

  System and stress tests may not work on every machine. CI runs them on
  self-hosted AIS runners
  ([hipfile-ci-system.yml](../../.github/workflows/hipfile-ci-system.yml))
  on several kinds of storage, along with
  [`util/ci-aiscp-test.sh`](util/ci-aiscp-test.sh), which copies files with
  the `aiscp` example on both I/O paths.
- **Every change to library behavior needs tests**
  ([STYLEGUIDE.md](STYLEGUIDE.md)): new functionality gets tests, including
  its error paths, and a bug fix gets a test that fails before the fix and
  passes after. Tests must run unattended.
- **Unit tests mock the outside world through `Context<T>`**
  ([`src/amd_detail/context.h`](src/amd_detail/context.h)). Library code
  reaches the system and the HIP runtime through `Context<Sys>::get()` and
  `Context<Hip>::get()` (and the library's own singletons, such as
  `Context<DriverState>`, the same way). In builds with `BUILD_TESTING` on
  (`AIS_TESTING` defined), a test replaces one with a Google Mock object for
  its lifetime: the mocks in `test/amd_detail/m*.h`, such as `MSys` in
  [`msys.h`](test/amd_detail/msys.h), hold a `ContextOverride<T>` that does
  it. Some tests share process-wide state through these singletons, so each
  test case runs in its own process (`gtest_discover_tests()`); don't rely on
  running a whole test binary at once.
- **Code coverage** isn't held to a target yet, but don't let it drop: cover
  the lines a change adds. To measure it, configure with Clang and
  `-DHIPFILE_USE_CODE_COVERAGE=ON`, run the unit tests, then run
  [`util/llvm-coverage.sh`](util/llvm-coverage.sh) `-b <build dir>` with
  ROCm's `llvm-profdata` and `llvm-cov` on `PATH`
  (`/opt/rocm/llvm/bin`). It writes `coverage-report.txt` (per file) and
  `coverage-lines.txt` (per line) in the build directory. CI's `amdclang++`
  C++20 jobs on Ubuntu put the report on the run's summary page.
- **Python**: the bindings' tests ([python/tests](python/tests/README.md))
  and the `ais-check` tests ([tools/ais-check/tests](tools/ais-check/tests/README.md))
  are hermetic pytest suites that need no GPU or ROCm: `python3 -m pytest` in
  either directory. `python/tests/main.py` is a separate end-to-end check
  that needs a GPU and AIS-capable storage. See [Python](#python) for
  building the bindings.

## Public API

- **Don't change the API in the public headers,
  [`include/hipfile.h`](include/hipfile.h) and
  [`include/hipfile-api-trace.h`](include/hipfile-api-trace.h), without
  explicit permission.** The API is everything that a consumer's code
  compiles or links against: the functions and their signatures, the types
  and struct layouts, the enumerators and their values, and the macros. Don't
  rename, move, reorder, or restyle declarations as a side effect of other
  work. If a change seems to need an API change, ask first.
- **The headers' documentation comments can change without asking**, and
  should, whenever a change makes them wrong or incomplete, such as a new
  requirement on callers or a new error a function can return. Change only
  the comments, not the declarations they document. Every public symbol needs
  Doxygen markup, in the right `@defgroup` group, since the API reference
  pages (`docs/reference/api-*.rst`) pull in whole groups. Check that the
  documentation still builds (see
  [Spelling and documentation](#spelling-and-documentation)).
- **The public headers are C**, and C11 and C++11 consumers both have to be
  able to include them. Keep them free of C++-isms and of anything newer than
  C11 or C++11, and keep each one self-contained. Every build with
  `BUILD_TESTING=ON`, including CI's, compiles `hipfile.h` as C11 and as
  C++11 with `-Wall -Wextra -Werror -pedantic`
  ([`test/header`](test/header)), so a violation breaks the build. A macro is
  only checked where it's expanded, so when you add a macro to `hipfile.h`,
  use it in [`test/header/header-check.c`](test/header/header-check.c).
  The test doesn't include `hipfile-api-trace.h`, so check that one by hand
  when you change it.
- **hipFile mirrors NVIDIA's cuFile API**, and HIPIFY maps cuFile calls onto
  it. Keep the two compatible, and update
  [docs/reference/hipFile-cuFile-compatibility.rst](docs/reference/hipFile-cuFile-compatibility.rst)
  when they differ.
- **Keep the ABI stable within a major version.** The shared library's
  `SOVERSION` is the major version, so minor and patch releases must never
  break the ABI, including by reordering struct fields.
- **A new public function also needs a dispatch table entry.** Every public
  call goes through `hipFileDispatchTable` so that profilers can intercept
  it; [`src/amd_detail/api_trace/README.md`](src/amd_detail/api_trace/README.md)
  has the steps and the versioning rules. Never reorder or change existing
  members. A table change also needs a matching change to rocprofiler-sdk's
  `abi.cpp`, which [`util/check-profiler-abi.sh`](util/check-profiler-abi.sh)
  checks (CI:
  [hipfile-profiler-abi.yml](../../.github/workflows/hipfile-profiler-abi.yml)).
- **The Python bindings wrap the C API**
  ([`python/hipfile/_chipfile.pxd`](python/hipfile/_chipfile.pxd) and
  [`_hipfile.pyx`](python/hipfile/_hipfile.pyx)), so an API change usually
  needs a change there too, with an entry in
  [python/CHANGELOG.md](python/CHANGELOG.md).
- **API changes are user-facing, and so are documentation changes that
  change what callers can rely on or have to do**, so both need a
  [changelog](#changelog) entry. Fixing a typo or rewording without changing
  the meaning doesn't.
- **The version is set in three places**: `HIPFILE_LIBRARY_MAJOR`,
  `_MINOR`, and `_PATCH` in the top-level `CMakeLists.txt` (which
  `docs/conf.py` reads, so keep each a plain `set()` of a number);
  `HIPFILE_VERSION_MAJOR`, `_MINOR`, and `_PATCH` in `hipfile.h` (which the
  Python bindings report as `hipfile.__version__`); and `version` in
  [`python/pyproject.toml`](python/pyproject.toml). Keep them in step.
- **The examples in [`examples/`](examples/) show how consumers use the
  API.** The build compiles them as C++20 with the full warning set, but
  they're installed with `CMakeLists.install.cmake` files that build them as
  C++17 against the installed hipFile, so keep them to C++17. Add a new
  example directory to `HIPFILE_EXAMPLE_DIRS` in
  [`cmake/HipFileInstall.cmake`](cmake/HipFileInstall.cmake) and to the
  top-level `CMakeLists.txt`.

## Thread safety

Every public function can be called from any thread, at the same time as any
other; hipFile has no API-wide lock. Shared state is guarded by the locks of
the objects that own it, such as `DriverState::state_mutex`
([`src/amd_detail/state.h`](src/amd_detail/state.h)) for the registered
files, buffers, streams, and batch contexts, and the batch, stream, and
async-monitor locks. Batch operations run on an internal thread pool
(Taskflow, in [`third-party/taskflow`](third-party/taskflow)).

- **Lookups return a `std::shared_ptr` copy**, which keeps a file, buffer, or
  stream alive for as long as the call holds it, even if another thread
  deregisters it. Don't keep a raw pointer or reference into a locked
  container after its lock is released.
- **Advisory: hold a lock only while you find or change what it guards**,
  and don't hold one across a HIP call, a blocking system call, or a wait on
  another thread. Existing code hasn't been audited for this; if you find a
  lock held across one, say so rather than changing it as a side effect.
- **Document what each lock guards**, and which locks a function expects its
  caller to hold, in comments, as `batch.h` and `file.h` do.
- **Advisory: annotate locks for Clang's thread safety analysis.**
  The Clang build turns on `-Wthread-safety`, but the code uses the standard
  library's locks, which carry no annotations, so the analysis checks
  nothing. hipObject wraps its locks in annotated classes and marks guarded
  data with `GUARDED_BY`; bring that in when a change gives you the chance,
  but ask first, since it touches many files.
- **Test concurrency changes with ThreadSanitizer** (see
  [C and C++](#c-and-c)) and the `stress` tests, and add concurrency tests
  for new shared state to `state_mt` or `batch_mt` where they fit.

## C++ code

- **The library, tests, and tools are C++20**, without GNU extensions
  (`HIPFILE_CXX_STANDARD`, which also accepts 23 and 26).
- **Follow [STYLEGUIDE.md](STYLEGUIDE.md).** In short: hyphens, not
  underscores, in file names; `#pragma once`; quoted local includes before
  angle-bracket system ones, each block in alphabetical order; Doxygen markup
  on public symbols, and `@internal` on private ones.
- **Write modern C++.** hipFile is a C++ library with a C API, and only the
  public headers have to look like C.
  - Use RAII to own resources: `std::unique_ptr` (or `std::shared_ptr` when
    ownership really is shared), not raw owning pointers, `new`/`delete`, or
    `malloc`/`free`. Wrap file descriptors and other handles from C APIs in a
    small RAII class or a `std::unique_ptr` with a custom deleter, as
    [`file-descriptor.h`](src/amd_detail/file-descriptor.h) does, so they're
    released on every path.
  - Use raw pointers only for non-owning access, such as the pointers the C
    API passes in and out.
  - Prefer the standard library and C++ language features to C idioms:
    `std::string` and `std::string_view` rather than `char` buffers;
    `std::array`, `std::vector`, and `std::span` rather than C arrays and
    pointer-and-length pairs; `std::optional` rather than sentinel values;
    `enum class`; `constexpr` rather than macros; and C++ casts rather than C
    casts.
  - Include the C++ versions of C standard library headers (`<cstdlib>`,
    `<cstring>`, `<cstdint>`, not `<stdlib.h>`, `<string.h>`, `<stdint.h>`).
    Only headers that C code includes, such as the public headers, use the C
    names. POSIX and system headers (`<unistd.h>`, `<sys/stat.h>`, and so on)
    have no C++ versions.
- **Don't let exceptions cross the C API.** The public functions in
  [`src/amd_detail/hipfile.cpp`](src/amd_detail/hipfile.cpp) and
  [`src/common/hipfile-common.cpp`](src/common/hipfile-common.cpp) are
  function-`try` blocks that catch everything and turn it into a
  `hipFileError_t`, mostly through `handle_exception()`; keep it that way.
  Inside the library, report errors by throwing.
- **Go through the wrappers for the outside world.** Call system functions
  through `Sys` ([`src/amd_detail/sys.h`](src/amd_detail/sys.h)), whose
  methods throw `std::system_error` on failure, and the HIP runtime through
  `Hip` ([`src/amd_detail/hip.h`](src/amd_detail/hip.h)), both reached
  through `Context<T>`, so that tests can mock them (see [Tests](#tests)).
  Add a wrapper method, and its mock, for a new system or HIP call rather
  than calling it directly. Log through `Sys::syslog()`.
- **Advisory: pass dependencies in, rather than looking them up.**
  `Context<T>` is a set of global accessors. hipObject instead gives each
  object what it uses through its constructor or parameters, and replaces
  the outside world only at the top, so tests run the same code as
  production. Don't add new `Context<T>` lookups where passing the
  dependency in is practical, and don't start converting existing code
  without asking.
- **Use `PassKey<T>`** ([`src/amd_detail/passkey.h`](src/amd_detail/passkey.h))
  to limit who can construct a class, as the registered file and buffer
  classes do, rather than making a whole class a friend.
- **Advisory: don't do arithmetic on raw pointers.** Clang's
  `-Wunsafe-buffer-usage` flags pointer arithmetic, indexing through a raw
  pointer, the C library's memory and string functions, and building a
  `std::span` from a pointer and a separate length. The
  `HIPFILE_WARN_UNSAFE_BUFFER_OPS` option turns it on; it's off by default
  until the existing warnings are fixed. Build with it on and keep the files
  you touch from adding to the count. HIP kernel code is exempt.
- **Advisory: avoid `reinterpret_cast`.** If a conversion is
  unavoidable, such as a pointer to an integer address, put it in one small,
  named helper rather than repeating it.
- **Don't convert numbers with the C library's functions** (`atoi()` and the
  rest of the `ato*` family, the `strto*` family, or `sscanf()`). They
  report errors badly or not at all. Use `std::from_chars`, and check both
  the error code and that it consumed all the text.
- **Advisory: don't print with `printf()` or its relatives.** Their
  arguments arrive without their types. Format with `std::format()`, whose
  format string is checked against the arguments when it compiles. The
  examples are the exception, since they're C++17.

## Portability

hipFile only builds on Linux today, and some of the public header already
anticipates Windows, so don't add assumptions that only hold on Linux's data
model.

- **Don't use `long` or `unsigned long` for sizes, offsets, or anything that
  has to be 64 bits.** `long` is 64 bits on Linux but 32 bits on Windows. Use
  `size_t` for sizes, `ptrdiff_t` for pointer differences, the fixed-width
  types from `<cstdint>` for everything with a defined width, and `hoff_t`
  for file offsets at the API.
- **Watch the APIs that are tied to `long`**: `printf()`'s `%ld`/`%lu` (use
  the `<cinttypes>` macros, or `%zu` for `size_t`) and integer literal
  suffixes (`L`, `UL`).
- **Keep Linux-specific code behind the wrappers.** System calls go through
  `Sys`, and mount information through `MountInfo`
  ([`src/amd_detail/mountinfo.h`](src/amd_detail/mountinfo.h)). Don't
  scatter direct system calls, `/proc` or `/sys` parsing, or `libmount`
  calls through the library.

## Security

**Security is critical.** hipFile moves data between files and GPU memory,
sometimes by direct DMA, on behalf of whatever process calls it. A bug here
can corrupt files or memory, or read or write memory the caller didn't ask
for, so security takes priority over speed and convenience.

- **Treat everything the library reads from outside as untrusted.** That
  includes the arguments to public functions, environment variables, the
  configuration, and what the library reads from `/proc` and `/sys`. Check
  pointers, lengths, offsets, flags, and enum values before using them, and
  reject bad input with an error rather than truncating, clamping, or
  guessing. A C caller can pass an enum value that isn't one of its
  enumerators; read it as its underlying integer type first, as
  `hipFileHandleRegister()` does.
- **Integer arithmetic must be safe.**
  - Check for wrap-around before it happens, not after. `offset + size`,
    size multiplications, and buffer-end calculations need an explicit
    overflow check (`__builtin_add_overflow`, `__builtin_mul_overflow`, or
    comparing against the limit first).
  - Check ranges before narrowing, and use `std::in_range` and the
    `std::cmp_*` functions from `<utility>` for signed/unsigned comparisons.
    Remember that `hoff_t` is signed, and that I/O sizes are `size_t` but
    their results are `ssize_t`.
  - Avoid undefined behavior: signed overflow, shifts by a negative amount or
    by the type's width or more, and casts of out-of-range values.
  - UBSan catches signed overflow, but unsigned wrap-around is defined
    behavior, so no sanitizer reports it. That part is on you.
- **Bound every buffer.** No `strcpy`, `sprintf`, or unbounded `strcat`.
  Where the library fills a caller's buffer, respect its length and always
  NUL-terminate strings.
- **Keep I/O inside the registered region.** An I/O with a registered buffer
  must stay within the range that was registered.
- **Fully initialize structures handed to the kernel or the HIP runtime**
  (`T x{}`), and don't let uninitialized memory reach a file or a caller.
- **Keep the hardening flags.** Don't remove or weaken `_FORTIFY_SOURCE`, the
  stack protector, `-fstack-clash-protection`,
  `-ftrivial-auto-var-init=pattern`, or the `-z noexecstack` link option
  ([`cmake/`](cmake/)). Ask before changing any compile or link options (see
  [Project direction](#project-direction)).
- **Fix security scan findings.** CodeQL
  ([hipfile-codeql.yml](../../.github/workflows/hipfile-codeql.yml), plus
  the monorepo's own CodeQL for Python and Actions) runs on PRs, and its
  findings get fixed, not dismissed.

## Data path performance

**Keep the overhead on the I/O path low.** On the fastpath, data moves by DMA
between storage and GPU memory, so what an I/O costs beyond the device is
mostly what hipFile adds. The hot path is everything that runs once per I/O:
`hipFileRead()` and `hipFileWrite()`, the async and batch submissions, backend
selection, and stats collection. Opening the driver, registering files,
buffers, and streams, and shutdown are setup, and can afford more.

- **Do expensive work once, at setup**: querying pointer attributes, reading
  mount information, checking whether a file can use the fastpath, and
  allocating bounce buffers. Don't move any of it onto the hot path.
- **Keep the hot path lean.** Avoid heap allocations, copying the payload
  (the fallback path is the only one that copies it), string formatting and
  logging on success, unneeded system calls, and HIP calls that synchronize.
  Don't add locks without a reason, and hold them briefly (see
  [Thread safety](#thread-safety)).
- **Modern C++ is compatible with this.** `std::unique_ptr`, `std::span`, and
  `std::string_view` cost nothing over their C equivalents. Watch for the
  constructs that do cost something, such as copying a `std::shared_ptr` (an
  atomic reference count update), a `std::function` that allocates, or a
  `std::string` built per I/O.
- **Don't give up correctness or safety for speed.** If a change has to add
  work to the hot path, say so and explain why.

## Linters, warnings, and sanitizers (mandatory)

**Every change must pass every check below that covers the files it touches.
These checks are not optional or advisory.** A change with an outstanding
finding isn't finished, so don't hand it back. CI doesn't run all of these
checks, and several CI jobs only run when files matching a path filter change,
so a green PR doesn't prove a change is clean. Run the checks yourself. Where a
check says existing code doesn't pass it yet, hold the files you touch to it
and leave the rest alone.

Fix the code rather than suppressing a finding (`// NOLINT`,
`# shellcheck disable=`, `# pylint: disable=`, `-Wno-…`, and so on). Only
suppress a finding when it's unavoidable, for example because of how a
framework such as Google Test or pytest works, and comment why. Compiler
warnings have their own suppression mechanism, described under
[C and C++](#c-and-c).

### C and C++

- **clang-format 18**, with [`.clang-format`](.clang-format). Reformat with
  `util/format-source.sh 18`, or check without changing anything:

  ```bash
  git ls-files -z '*.c' '*.h' '*.cpp' ':!third-party' \
    | xargs -0 clang-format-18 --dry-run -Werror
  ```

  CI:
  [hipfile-clang-format.yml](../../.github/workflows/hipfile-clang-format.yml).
- **No compiler warnings, with both GCC and Clang.** They warn about
  different things, so build with each (`-DCMAKE_CXX_COMPILER=g++`, and
  `amdclang++` or `clang++`) in separate build directories. The build turns
  on a long list of warnings
  ([`HipFileGNUCompilerOptions.cmake`](cmake/HipFileGNUCompilerOptions.cmake),
  [`HipFileClangCompilerOptions.cmake`](cmake/HipFileClangCompilerOptions.cmake))
  but doesn't make them errors, so configure with
  `-DCMAKE_CXX_FLAGS=-Werror`, as CI does, or read the build output. Build
  both `Debug` and an optimized type: some warnings (and `_FORTIFY_SOURCE`)
  only appear with optimization.
- **Suppress a warning only when it's unavoidable**, and then only with the
  macros in [`shared/hipfile-warnings.h`](shared/hipfile-warnings.h), never
  with raw `#pragma`s, `-Wno-…` flags, or changes to the warning lists. Add an
  `HIPFILE_WARN_<THING>_OFF`/`_ON` pair for the specific warning there,
  following the instructions at the top of the file, add both to
  `StatementMacros` in [`.clang-format`](.clang-format), wrap the smallest
  region you can, and comment why the warning can't be fixed. The per-file
  GCC options on `thread-pool.cpp` in
  [`src/amd_detail/CMakeLists.txt`](src/amd_detail/CMakeLists.txt), for
  warnings that come from Taskflow, are an existing exception; don't add
  others without asking. Third-party code (`third-party/`, and GoogleTest
  fetched with `FetchContent`) is included as `SYSTEM`, so its warnings
  don't count.
- **clang-tidy**: configure with `-DHIPFILE_USE_CLANG_TIDY=ON` and Clang as
  the C++ compiler, and build; it must report nothing (every finding is an
  error). [`.clang-tidy`](.clang-tidy) enables only the `clang-analyzer-*`
  checks, which ROCm's clang-tidy doesn't have, so point
  `HIPFILE_CLANG_TIDY_EXECUTABLE` at an upstream clang-tidy whose LLVM major
  version matches ROCm's (the CI workflow sets the version it uses). If it
  reports "no checks enabled," it's the wrong clang-tidy; don't work around
  it. CI:
  [hipfile-clang-tidy.yml](../../.github/workflows/hipfile-clang-tidy.yml).
  **Advisory:** hipObject runs a much broader set of checks
  (`bugprone-*`, `cppcoreguidelines-*`, `modernize-*`, `readability-*`, and
  more); keep new code clean of what those would report where you can.
- **include-what-you-use (IWYU)**: configure with `-DHIPFILE_USE_IWYU=ON`
  (Clang only) in a separate build directory, since it leaves out the
  warning flags, and build; IWYU prints the `#include` lines it suggests
  adding or removing for each file. Unlike the other checks, it doesn't have
  to report nothing, since some of its suggestions are wrong. Make the
  suggestions that are right for the files you touch, and ignore the ones
  that are wrong. Fix a suggestion that names a library's internal header,
  rather than its public one, with a mapping in a `cmake/iwyu-*.imp` file,
  as [`cmake/iwyu-libc.imp`](cmake/iwyu-libc.imp) does. Nothing in CI runs
  IWYU.
- **Sanitizer-clean**: the unit tests must run without a report from any
  sanitizer, and any change that involves pointers, memory, or concurrency
  must be tested this way. The sanitizers need Clang (`amdclang++` or
  `clang++`). Build in two build directories, one configured with
  `-DHIPFILE_USE_SANITIZERS=ON` (AddressSanitizer, LeakSanitizer,
  UndefinedBehaviorSanitizer, and the other checks listed in
  [`HipFileSanitizers.cmake`](cmake/HipFileSanitizers.cmake)) and one with
  `-DHIPFILE_USE_THREAD_SANITIZER=ON`, then run `ctest -L unit` in each.
  Every report is fatal, so a test with one fails. A sanitizer build always
  fetches GoogleTest to instrument it, so it needs network access the first
  time it's configured. If ThreadSanitizer aborts at startup with an
  "unexpected memory mapping" error, lower the address-space randomization
  with `sudo sysctl vm.mmap_rnd_bits=28`. The LeakSanitizer suppressions in
  [`cmake/lsan-suppressions.supp`](cmake/lsan-suppressions.supp) cover only
  the ROCm runtime; don't add suppressions for hipFile's own code. Don't add
  sanitizer flags anywhere but `HipFileSanitizers.cmake`. CI:
  [hipfile-sanitizers.yml](../../.github/workflows/hipfile-sanitizers.yml),
  with `amdclang++`.
- **Advisory: valgrind-clean.** Valgrind's memcheck finds reads of
  uninitialized memory, which the sanitizers don't look for. Run it on the
  unit tests in a GCC build without the sanitizers (valgrind can't run a
  program built with them), and check that a change adds no reports:

  ```bash
  ctest --test-dir <gcc-build-dir> -L unit -T memcheck --output-on-failure \
    --overwrite "MemoryCheckCommand=$(command -v valgrind)" \
    --overwrite "MemoryCheckCommandOptions=--leak-check=full --track-origins=yes --num-callers=40 --error-exitcode=1"
  ```

  The reports go to `Testing/Temporary/MemoryChecker.*.log` in the build
  directory, not to ctest's output. Don't run valgrind on an optimized Clang
  build: with `-ftrivial-auto-var-init=pattern`, Clang at `-O1` and above can
  emit x87 instructions that valgrind emulates with too few bits, which
  produces bogus results. Nothing in CI runs valgrind.
- **Lifetime safety, when a change could leave something dangling.** Clang's
  `-Wlifetime-safety` looks for use-after-free, use-after-scope, and
  dangling references, but reports too many false positives to leave on.
  When a change creates or moves a pointer, reference, iterator, or view
  (`std::string_view`, `std::span`) into memory that something else owns,
  stores one in an object or lambda that outlives the call, or destroys,
  moves, or resizes an object that other code may still point into, build
  once with it on and read what it reports for the files you touched:

  ```bash
  cmake -S . -B <build-dir> -DCMAKE_CXX_COMPILER=amdclang++ \
    -DCMAKE_CXX_FLAGS=-Wlifetime-safety
  cmake --build <build-dir> --parallel
  ```

  Fix the reports that are real, and leave the code alone for the false
  positives; don't rewrite code just to quiet the check, and don't add
  suppressions for it. Most false positives come from its
  `-Wlifetime-safety-invalidation` subgroup, which treats changing a
  container as invalidating references to the container itself, not just
  pointers into it. Nothing in CI runs it.

### Shell

Shell scripts are bash, not POSIX sh: start them with `#!/usr/bin/env bash`
(see [STYLEGUIDE.md](STYLEGUIDE.md)).

- **shellcheck** on every shell script:

  ```bash
  git ls-files -z '*.sh' | xargs -0 shellcheck
  ```

  CI: [hipfile-shellcheck.yml](../../.github/workflows/hipfile-shellcheck.yml),
  when a `.sh` file or the workflow file changes.
- **The shell embedded in the hipFile workflows must be shellcheck-clean
  too.** Every `run:` block in `.github/workflows/hipfile-*.yml` is a bash
  script, and the shellcheck workflow doesn't look at them. Check them with
  [actionlint](https://github.com/rhysd/actionlint), which runs shellcheck on
  each `run:` block, from the top of the monorepo:

  ```bash
  actionlint .github/workflows/hipfile-*.yml
  ```

  **Advisory** for existing workflows, which have many findings
  (mostly unquoted variables); a `run:` block you add or change must be
  clean. actionlint also flags the self-hosted `AIS` runner label, which is
  spurious. Neither tool sees the scripts passed to `bash -c` inside
  `docker exec`, so hold those to the same standard by hand. Nothing in CI
  runs actionlint.

### CMake

- **cmakelint**, with [`.cmakelintrc`](.cmakelintrc):

  ```bash
  git ls-files -z 'CMakeLists.txt' '*/CMakeLists.txt' '*.cmake' \
    | xargs -0 cmakelint
  ```

  CI: [hipfile-cmakelint.yml](../../.github/workflows/hipfile-cmakelint.yml),
  when a CMake file, `.cmakelintrc`, or the workflow file changes.
- Use modern, target-based CMake, and keep to the minimum CMake version in
  the top-level `CMakeLists.txt` (3.21); hipFile hasn't moved to CMake 4.

### Python

Python code must run on **Python 3.10**. Don't use anything newer, such as
`tomllib`, `except*`, or `typing.Self` (all 3.11).

- **black** and **pylint**, over every Python file, including the
  extensionless `tools/ais-check/ais-check`:

  ```bash
  git ls-files -z '*.py' tools/ais-check/ais-check \
    | xargs -0 black --check --target-version py310
  git ls-files -z '*.py' tools/ais-check/ais-check \
    | xargs -0 pylint --py-version=3.10
  ```

  CI: [hipfile-pylint.yml](../../.github/workflows/hipfile-pylint.yml), when
  a Python file or the workflow file changes. Neither tool reads the Cython
  sources (`.pyx`, `.pxd`), so keep them consistent with the Python code by
  hand.
- **vermin**, since black and pylint don't catch most uses of newer language
  features or modules. It prints `Target versions not met` if the code needs
  a newer Python:

  ```bash
  git ls-files -z '*.py' tools/ais-check/ais-check \
    | xargs -0 vermin -t=3.10- --violations --no-tips
  ```

- **Building the bindings**: they're built by scikit-build-core from
  [`python/`](python/), and need an installed hipFile or the paths to a
  built one. Install them as an editable package in a virtual environment
  (`.venv`, if one exists):

  ```bash
  pip install -e python
  ```

  If that can't find hipFile, give the paths explicitly, with
  `HIPFILE_LIBRARY` naming the library file in your build directory:

  ```bash
  pip install -e python \
    -Ccmake.define.HIPFILE_INCLUDE_DIR="$PWD/include" \
    -Ccmake.define.HIPFILE_LIBRARY=<build-dir>/src/amd_detail/libhipfile.so \
    -Ccmake.define.HIP_INCLUDE_DIR=/opt/rocm/include
  ```

  CI builds an sdist, installs it, and runs the tests
  ([hipfile-python-bindings.yml](../../.github/workflows/hipfile-python-bindings.yml)).
  The `ais-check` tests run separately
  ([hipfile-ais-check.yml](../../.github/workflows/hipfile-ais-check.yml)),
  on Python 3.10 and the latest release.

### Spelling and documentation

- **codespell**, with [`.codespellrc`](.codespellrc), over the tracked files,
  Markdown included:

  ```bash
  git ls-files -z ':!third-party' | xargs -0 codespell
  ```

  CI: [hipfile-codespell.yml](../../.github/workflows/hipfile-codespell.yml),
  which uses [`.codespellignore`](.codespellignore) as its word list; keep
  that file and `ignore-words-list` in `.codespellrc` in step.
- **The documentation must build.** Read the Docs builds it
  ([`.readthedocs.yaml`](.readthedocs.yaml)) by running doxygen in
  `docs/doxygen` and then Sphinx on `docs/conf.py`. To build it locally
  (with the packages in `docs/sphinx/requirements.txt` installed):

  ```bash
  (cd docs/doxygen && doxygen Doxyfile)
  python3 -m sphinx -b html docs docs/_build/html
  ```

  **Advisory:** doxygen shouldn't warn about the public headers; it
  currently warns about undocumented macros. `-DHIPFILE_BUILD_DOCS=ON`
  adds a `doc` target that runs only doxygen, from
  [`docs/doxygen/Doxyfile.in`](docs/doxygen/Doxyfile.in); keep it and
  `Doxyfile` in step. CI's Debug builds configure with
  `HIPFILE_BUILD_DOCS=ON` but don't build the `doc` target, so nothing in
  hipFile's CI builds the documentation.
- **Document user-visible behavior.** A new environment variable or
  configuration setting goes in
  [docs/reference/hipFile-environment-variables.rst](docs/reference/hipFile-environment-variables.rst),
  and a new limitation in
  [docs/troubleshooting](docs/troubleshooting).

## Changelog

**Every user-facing change needs an entry in [CHANGELOG.md](CHANGELOG.md),
in the same change.** User-facing means anything someone building,
installing, or calling hipFile could notice: the public API, what its
documentation promises or requires (see [Public API](#public-api)), library
behavior, environment variables, CMake options and variables, the installed
files and packages, the tools (`ais-check`, `ais-stats`), dependencies and
supported platforms, and the examples. CI-only, test-only, and internal
refactoring changes don't need an entry. Changes to the Python bindings go in
[python/CHANGELOG.md](python/CHANGELOG.md) instead.

Add the entry to the topmost (unreleased) version, under the matching
heading: `Added`, `Changed`, `Resolved issues`, `Removed`, or `Known issues`.
Write it for users, like the existing entries: say what changed, and what
users have to do differently, if anything.

## Commits and pull requests

rocm-systems uses [Conventional Commits](https://www.conventionalcommits.org/)
for commit subjects and PR titles, with the project as the scope:
`type(hipfile): summary`, as in `fix(hipfile): Attribute IO stats to the
buffer's GPU`. The type is one of `feat`, `fix`, `docs`, `test`,
`refactor`, `perf`, `build`, `ci`, `chore`, or `revert`. Write the summary
in the imperative, without a trailing period. PRs are squash-merged, so the
PR title becomes the commit subject.

Fill in the sections of the monorepo's
[pull request template](../../.github/pull_request_template.md), including
a JIRA ID or issue reference; the Systems PR Bot
([`tools/systems_pr_bot`](../../tools/systems_pr_bot)) checks the
description, and that a PR changing code also changes a test, and fails
the PR until they're fixed.
