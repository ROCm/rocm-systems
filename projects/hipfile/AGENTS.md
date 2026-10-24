# AGENTS.md

Guidance for AI coding agents working on hipFile, AMD's Infinity Storage (AIS)
library for direct-to-GPU file I/O on ROCm, with C and Python APIs for
synchronous, asynchronous, and batch I/O. See [README.md](README.md) for scope
and [docs/install/build-from-source.rst](docs/install/build-from-source.rst)
for building.

hipFile lives in the [rocm-systems](https://github.com/ROCm/rocm-systems)
monorepo, under `projects/hipfile`. Unless a section says otherwise, run
commands from there; paths are relative to it. The CI workflows are at the top
of the monorepo, in
[`.github/workflows/hipfile-*.yml`](../../.github/workflows).

Ignore `nvidia_detail` directories (`src/nvidia_detail`,
`test/nvidia_detail`) and the `CMAKE_HIP_PLATFORM=nvidia` build. That code
isn't production code, isn't well maintained, and CI no longer builds it.

Rules marked **Advisory** describe where hipFile is heading, mostly following
its sister project hipObject. Existing code doesn't pass them yet, so nothing
enforces them. Follow them in new and changed code, don't make the count of
findings worse, and don't rewrite unrelated code to satisfy them unless
you're asked to. An advisory rule becomes mandatory once the code passes it.
Rules without the mark apply to everything.

## Project direction

hipFile is built by [TheRock](https://github.com/ROCm/TheRock), ROCm's build
system, as part of a much larger superproject. Make changes that keep working
there:

- **Ask before changing compile or link options, dependency handling, or any
  other non-trivial CMake.** A superproject has requirements that a
  standalone build doesn't show. For example, the GoogleTest programs are
  compiled and linked with `-pthread` because TheRock's sanitizer builds need
  it, even though it looks redundant. Never remove it, and add it to any new
  GoogleTest program.
- **Link only against libraries TheRock already provides**: ROCm's, or ones
  TheRock vendors in its
  [`third-party`](https://github.com/ROCm/TheRock/tree/main/third-party)
  directory (system libraries are under `third-party/sysdeps`). TheRock is
  moving to hermetic builds, and getting a new link-time dependency into it
  is extremely difficult, so don't add one without asking.
- **Don't assume hipFile is the top-level CMake project.** Use
  `PROJECT_SOURCE_DIR`/`PROJECT_BINARY_DIR`,
  `CMAKE_CURRENT_SOURCE_DIR`/`CMAKE_CURRENT_BINARY_DIR`, or the
  `HIPFILE_*_PATH` variables from the top-level `CMakeLists.txt`, not
  `CMAKE_SOURCE_DIR`/`CMAKE_BINARY_DIR`. Prefix new cache variables with
  `HIPFILE_` and new CMake functions with `hipfile_`, and set flags per
  target, as `hipfile_set_compiler_flags()`
  ([`cmake/HipFileCompilerOptions.cmake`](cmake/HipFileCompilerOptions.cmake))
  does, not globally.
- **The `AIS_*` CMake options are deprecated** aliases for the `HIPFILE_*`
  ones ([`cmake/HipFileDeprecatedAISOptions.cmake`](cmake/HipFileDeprecatedAISOptions.cmake)).
  Don't add new ones or new uses of them. The `AIS_TESTING` compile
  definition is separate, and is still the name the code uses.

The same team writes two sister projects. **hipObject**
([ROCm/hipObject](https://github.com/ROCm/hipObject)), RDMA-accelerated S3
object I/O for AMD GPUs, is expected to move into rocm-systems. It started
from hipFile's conventions but is stricter: dependencies passed in rather
than looked up globally, Clang thread safety annotations,
`-Wunsafe-buffer-usage` on, a coverage target, and more linters. It's the
model for most of the advisory rules. **rocm-ernic**
([ROCm/rocm-ernic](https://github.com/ROCm/rocm-ernic)) is in its own
repository. Borrow ideas from both freely. If you find something one of them
should adopt, or a bug in one, ask whether to create a to-do file for it;
don't create one or edit the other project without asking.

## Configure and build

```bash
cmake -S . -B <build-dir> -DCMAKE_CXX_COMPILER=amdclang++
cmake --build <build-dir> --parallel
```

- The library is `src/amd_detail/libhipfile.so` in the build directory. The
  default build type is `RelWithDebInfo`, `BUILD_TESTING` is on, and
  `HIPFILE_CXX_STANDARD` selects C++20 (the default), 23, or 26 (which
  needs CMake 3.30 or later).
- **Delete your build directories when you're done with them**, unless
  you're asked to keep them. Leftover builds fill `/tmp` and break other work
  on the machine.
- **ROCm**: without `ROCM_PATH` (a CMake or environment variable), the build
  uses `/opt/rocm/core` if that holds a ROCm installation (the 7.11+ layout),
  and `/opt/rocm` otherwise; `ROCM_VERSION` changes this. Put
  `/opt/rocm/bin` on `PATH` to run what you build.
- **Compiler**: the one HIP kernel
  (`src/amd_detail/backend/memcpy-kernel.hip`) needs ROCm's HIP compiler,
  and `CMAKE_HIP_ARCHITECTURES` limits its GPU targets. `amdclang++` is the
  usual C++ compiler, and TheRock's, but the code has to build with upstream
  Clang and GCC too.
- **CI** ([hipfile-build.yml](../../.github/workflows/hipfile-build.yml);
  the matrix is in
  [hipfile-ci-toplevel.yml](../../.github/workflows/hipfile-ci-toplevel.yml))
  builds `Debug` and `Release` with `amdclang++`, `clang++`, and `g++` on
  several distributions and ROCm versions, with `-DCMAKE_CXX_FLAGS=-Werror`,
  runs the unit tests, and packages and installs the Debug build. Its images
  come from [`util/docker`](util/docker).

## Tests

- After a successful build, from the build directory:
  `ctest --output-on-failure -L unit`. The ctest labels are:
  - `unit`: the GoogleTest suites that don't need a GPU (`internal_tests`,
    which tests the internals with mocks, plus the CMake test of the
    deprecated `AIS_*` options). CI runs these in a plain container.
  - `system`: `hipfile_system_tests`, which needs a GPU and does real I/O in
    `HIPFILE_CAPABLE_DIR` (default `/tmp`). The fastpath tests need storage
    that supports it (check with `tools/ais-check/ais-check`), and fail
    without it unless configured with
    `-DHIPFILE_ALLOW_SKIP_FASTPATH_TESTS=ON`.
  - `stress`: `state_mt` and `batch_mt`, short concurrency tests run under
    `gdb`. They can pass for a while after a bug lands, so a failure may not
    come from the change you're looking at.

  System and stress tests may not work on every machine. CI runs them on
  self-hosted AIS runners
  ([hipfile-ci-system.yml](../../.github/workflows/hipfile-ci-system.yml)) on
  several kinds of storage, along with
  [`util/ci-aiscp-test.sh`](util/ci-aiscp-test.sh), which copies files with
  the `aiscp` example on both I/O paths.
- **Every change to library behavior needs tests**
  ([STYLEGUIDE.md](STYLEGUIDE.md)): new functionality gets tests, including
  its error paths, and a bug fix gets a test that fails before the fix and
  passes after. Tests must run unattended.
- **Unit tests mock the outside world through `Context<T>`**
  ([`src/amd_detail/context.h`](src/amd_detail/context.h)). Library code
  reaches the system, the HIP runtime, and its own singletons through
  `Context<Sys>::get()`, `Context<Hip>::get()`, `Context<DriverState>::get()`,
  and so on. When `BUILD_TESTING` is on (`AIS_TESTING` defined), the mocks in
  `test/amd_detail/m*.h`, such as `MSys` in
  [`msys.h`](test/amd_detail/msys.h), replace one for their lifetime. Tests
  share process-wide state through these singletons, so each test case runs
  in its own process (`gtest_discover_tests()`); don't rely on running a
  whole test binary at once.
- **Don't let code coverage drop**; cover the lines a change adds. Configure
  with Clang and `-DHIPFILE_USE_CODE_COVERAGE=ON`, run the unit tests, then
  run [`util/llvm-coverage.sh`](util/llvm-coverage.sh) `-b <build-dir>` with
  `/opt/rocm/llvm/bin` on `PATH`, which writes `coverage-report.txt` and
  `coverage-lines.txt` in the build directory.
- **Python**: the bindings' tests ([python/tests](python/tests/README.md))
  and the `ais-check` tests
  ([tools/ais-check/tests](tools/ais-check/tests/README.md)) are hermetic
  pytest suites that need no GPU or ROCm: run `python3 -m pytest` in either
  directory. `python/tests/main.py` is a separate end-to-end check that needs
  a GPU and AIS-capable storage.

## Public API

- **Don't change the API in [`include/hipfile.h`](include/hipfile.h) or
  [`include/hipfile-api-trace.h`](include/hipfile-api-trace.h) without
  explicit permission.** The API is everything a consumer compiles or links
  against: functions and their signatures, types and struct layouts,
  enumerators and their values, and macros. Don't rename, move, reorder, or
  restyle declarations as a side effect of other work. If a change seems to
  need an API change, ask first.
- **The headers' documentation comments can change without asking**, and
  should whenever a change makes them wrong or incomplete, such as a new
  requirement on callers or a new error. Every public symbol needs Doxygen
  markup in the right `@defgroup`, since the API reference pages
  (`docs/reference/api-*.rst`) pull in whole groups. Check that the
  documentation still builds (see
  [Spelling and documentation](#spelling-and-documentation)).
- **API changes, and documentation changes that change what callers can rely
  on or have to do, need a [changelog](#changelog) entry.** Typo fixes and
  rewording don't.
- **The headers are C, for C11 and C++11 consumers.** Keep them free of
  C++-isms and of anything newer than C11 or C++11, and keep each one
  self-contained. Builds with `BUILD_TESTING=ON` compile `hipfile.h` both
  ways with `-Wall -Wextra -Werror -pedantic` ([`test/header`](test/header)).
  A macro is only checked where it's expanded, so use a new macro in
  [`header-check.c`](test/header/header-check.c). The test doesn't include
  `hipfile-api-trace.h`, so check that one by hand.
- **Keep the ABI stable within a major version.** The library's `SOVERSION`
  is the major version, so minor and patch releases must never break the
  ABI, including by reordering struct fields.
- **A new public function needs a dispatch table entry**, so that profilers
  can intercept it;
  [`src/amd_detail/api_trace/README.md`](src/amd_detail/api_trace/README.md)
  has the steps and the versioning rules. Never reorder or change existing
  members. rocprofiler-sdk's `hipfile/abi.cpp` checks the table's layout and
  version with `static_assert`s, so a table change needs a matching change
  there, or the profiler's build breaks after hipFile's PR merges;
  [`util/check-profiler-abi.sh`](util/check-profiler-abi.sh) checks it (CI:
  [hipfile-profiler-abi.yml](../../.github/workflows/hipfile-profiler-abi.yml)).
- **hipFile mirrors NVIDIA's cuFile API**, and HIPIFY maps cuFile calls onto
  it. Keep the two compatible, and update
  [docs/reference/hipFile-cuFile-compatibility.rst](docs/reference/hipFile-cuFile-compatibility.rst)
  where they differ.
- **The Python bindings wrap the C API**
  ([`python/hipfile/_chipfile.pxd`](python/hipfile/_chipfile.pxd) and
  [`_hipfile.pyx`](python/hipfile/_hipfile.pyx)), so an API change usually
  needs a change there too, with an entry in
  [python/CHANGELOG.md](python/CHANGELOG.md).
- **The version is set in three places; keep them in step**:
  `HIPFILE_LIBRARY_MAJOR`, `_MINOR`, and `_PATCH` in the top-level
  `CMakeLists.txt` (which `docs/conf.py` reads, so keep each a plain `set()`
  of a number); `HIPFILE_VERSION_MAJOR`, `_MINOR`, and `_PATCH` in
  `hipfile.h` (which the bindings report as `hipfile.__version__`); and
  `version` in [`python/pyproject.toml`](python/pyproject.toml).
- **The examples in [`examples/`](examples/) show how consumers use the
  API.** The build compiles them as C++20, but they're installed to build as
  C++17 (C++20 for `batch`; see their `CMakeLists.install.cmake` files), so
  keep each to its installed standard. Add a new example directory to
  `HIPFILE_EXAMPLE_DIRS` in
  [`cmake/HipFileInstall.cmake`](cmake/HipFileInstall.cmake) and to the
  top-level `CMakeLists.txt`.

## Thread safety

Every public function can be called from any thread, at the same time as any
other; there's no API-wide lock. Shared state is guarded by the locks of the
objects that own it, such as `DriverState::state_mutex`
([`src/amd_detail/state.h`](src/amd_detail/state.h)) for the registered
files, buffers, streams, and batch contexts, and the batch, stream, and
async-monitor locks. Batch operations run on an internal Taskflow thread pool
([`third-party/taskflow`](third-party/taskflow)).

- **Lookups return a `std::shared_ptr` copy**, which keeps a file, buffer, or
  stream alive while the call holds it, even if another thread deregisters
  it. Don't keep a raw pointer or reference into a locked container after
  its lock is released.
- **Advisory: hold a lock only while you find or change what it guards**, and
  not across a HIP call, a blocking system call, or a wait on another
  thread. Existing code hasn't been audited for this; if you find a lock held
  across one, say so rather than changing it as a side effect.
- **Document what each lock guards**, and which locks a function expects its
  caller to hold, in comments, as `batch.h` and `file.h` do.
- **Advisory: annotate locks for Clang's thread safety analysis.** The Clang
  build turns on `-Wthread-safety`, but the standard library's locks carry
  no annotations, so it checks nothing. hipObject wraps its locks in
  annotated classes and marks guarded data with `GUARDED_BY`; bring that in
  when a change gives you the chance, but ask first, since it touches many
  files.
- **Test concurrency changes with ThreadSanitizer** (see
  [C and C++](#c-and-c)) and the `stress` tests, and add tests for new shared
  state to `state_mt` or `batch_mt` where they fit.

## C++ code

- **The library, tests, and tools are C++20**, without GNU extensions.
  `hipfile_set_cxx_standard()`
  ([`cmake/HipFileCompilerOptions.cmake`](cmake/HipFileCompilerOptions.cmake))
  builds a target with exactly the standard `HIPFILE_CXX_STANDARD` selects,
  and the `hipfile_add_*()` helpers call it. Use them for new targets, rather
  than `target_compile_features()`, whose `cxx_std_*` sets only a minimum.
- **Follow [STYLEGUIDE.md](STYLEGUIDE.md).** In short: hyphens, not
  underscores, in file names; `#pragma once`; quoted local includes before
  angle-bracket system ones, each block in alphabetical order; Doxygen markup
  on public symbols, and `@internal` on private ones.
- **Write modern C++.** Only the public headers have to look like C.
  - Own resources with RAII: `std::unique_ptr` (`std::shared_ptr` only when
    ownership really is shared), not raw owning pointers, `new`/`delete`, or
    `malloc`/`free`. Wrap file descriptors and other C handles in a small
    RAII class or a `std::unique_ptr` with a custom deleter, as
    [`file-descriptor.h`](src/amd_detail/file-descriptor.h) does. Use raw
    pointers only for non-owning access, such as the pointers the C API
    passes in and out.
  - Prefer the standard library to C idioms: `std::string` and
    `std::string_view`, not `char` buffers; `std::array`, `std::vector`, and
    `std::span`, not C arrays and pointer-and-length pairs; `std::optional`,
    not sentinel values; `enum class`; `constexpr`, not macros; and C++
    casts, not C casts.
  - Include `<cstdlib>`, `<cstring>`, `<cstdint>`, and so on, not
    `<stdlib.h>`, `<string.h>`, `<stdint.h>`, except in headers that C code
    includes. POSIX and system headers (`<unistd.h>`, `<sys/stat.h>`) have
    no C++ versions.
- **Don't let exceptions cross the C API.** The public functions in
  [`src/amd_detail/hipfile.cpp`](src/amd_detail/hipfile.cpp) and
  [`src/common/hipfile-common.cpp`](src/common/hipfile-common.cpp) are
  function-`try` blocks that catch everything and turn it into a
  `hipFileError_t`, mostly through `handle_exception()`; keep it that way.
  Inside the library, report errors by throwing.
- **Go through the wrappers for the outside world**: system calls through
  `Sys` ([`src/amd_detail/sys.h`](src/amd_detail/sys.h)), whose methods throw
  `std::system_error` on failure, and the HIP runtime through `Hip`
  ([`src/amd_detail/hip.h`](src/amd_detail/hip.h)), both reached through
  `Context<T>` so tests can mock them. Add a wrapper method, and its mock,
  for a new call rather than calling it directly. Log through
  `Sys::syslog()`.
- **Advisory: pass dependencies in, rather than looking them up.** hipObject
  gives each object what it uses through its constructor or parameters, and
  replaces the outside world only at the top, so tests run the same code as
  production. Don't add `Context<T>` lookups where passing the dependency in
  is practical, and don't convert existing code without asking.
- **Use `PassKey<T>`** ([`src/amd_detail/passkey.h`](src/amd_detail/passkey.h))
  to limit who can construct a class, as the registered file and buffer
  classes do, rather than making a whole class a friend.
- **Advisory: don't do arithmetic on raw pointers.** Clang's
  `-Wunsafe-buffer-usage` flags pointer arithmetic, indexing through a raw
  pointer, the C library's memory and string functions, and building a
  `std::span` from a pointer and a separate length.
  `HIPFILE_WARN_UNSAFE_BUFFER_OPS` turns it on; it's off by default until
  the existing warnings are fixed. Build with it on and keep the files you
  touch from adding to the count. HIP kernel code is exempt.
- **Advisory: avoid `reinterpret_cast`.** If one is unavoidable, such as a
  pointer to an integer address, put it in one small, named helper.
- **Don't convert numbers with the C library** (the `ato*` and `strto*`
  families, or `sscanf()`), which reports errors badly or not at all. Use
  `std::from_chars`, and check both the error code and that it consumed all
  the text.
- **Advisory: format with `std::format()`, not `printf()` or its
  relatives**, whose arguments arrive without their types. The examples,
  being C++17, are the exception.

## Portability

hipFile only builds on Linux today, but the public header already
anticipates Windows, so don't add assumptions that hold only on Linux's data
model.

- **Don't use `long` or `unsigned long` for sizes, offsets, or anything that
  has to be 64 bits**; `long` is 32 bits on Windows. Use `size_t` for sizes,
  `ptrdiff_t` for pointer differences, the fixed-width types from
  `<cstdint>` where the width matters, and `hoff_t` for file offsets at the
  API. Watch the APIs tied to `long`: `printf()`'s `%ld`/`%lu` (use the
  `<cinttypes>` macros, or `%zu` for `size_t`) and the `L` and `UL` literal
  suffixes.
- **Keep Linux-specific code behind the wrappers**: system calls in `Sys`,
  and mount information in `MountInfo`
  ([`src/amd_detail/mountinfo.h`](src/amd_detail/mountinfo.h)). Don't
  scatter direct system calls, `/proc` or `/sys` parsing, or `libmount`
  calls through the library.

## Security

**Security is critical.** hipFile moves data between files and GPU memory,
sometimes by direct DMA, on behalf of whatever process calls it. A bug can
corrupt files or memory, or read or write memory the caller didn't ask for,
so security takes priority over speed and convenience.

- **Treat everything the library reads from outside as untrusted**: the
  arguments to public functions, environment variables, the configuration,
  and `/proc` and `/sys`. Check pointers, lengths, offsets, flags, and enum
  values before using them, and reject bad input with an error rather than
  truncating, clamping, or guessing. A C caller can pass an enum value that
  isn't one of its enumerators, so read it as its underlying integer type
  first, as `hipFileHandleRegister()` does.
- **Integer arithmetic must be safe.**
  - Check for wrap-around before it happens. `offset + size`, size
    multiplications, and buffer ends need an explicit check
    (`__builtin_add_overflow`, `__builtin_mul_overflow`, or a comparison
    against the limit first). UBSan catches signed overflow, but unsigned
    wrap-around is defined behavior, so no sanitizer reports it.
  - Check ranges before narrowing, and use `std::in_range` and the
    `std::cmp_*` functions from `<utility>` for signed/unsigned comparisons.
    `hoff_t` is signed, and I/O sizes are `size_t` but their results are
    `ssize_t`.
  - Avoid undefined behavior: signed overflow, shifts by a negative amount or
    by the type's width or more, and casts of out-of-range values.
- **Bound every buffer.** No `strcpy`, `sprintf`, or unbounded `strcat`.
  Where the library fills a caller's buffer, respect its length and always
  NUL-terminate strings.
- **Keep I/O with a registered buffer inside the registered range.**
- **Fully initialize structures handed to the kernel or the HIP runtime**
  (`T x{}`), and don't let uninitialized memory reach a file or a caller.
- **Keep the hardening flags** ([`cmake/`](cmake/)): `_FORTIFY_SOURCE`, the
  stack protector, `-fstack-clash-protection`,
  `-ftrivial-auto-var-init=pattern`, and the `-z noexecstack` link option.
  Don't remove or weaken them.
- **Fix security scan findings** rather than dismissing them. CodeQL runs on
  PRs ([hipfile-codeql.yml](../../.github/workflows/hipfile-codeql.yml), plus
  the monorepo's own CodeQL for Python and Actions).

## Data path performance

On the fastpath, data moves by DMA between storage and GPU memory, so what
an I/O costs beyond the device is mostly what hipFile adds. **Keep the
overhead on the hot path low.** The hot path is everything that runs once per
I/O: `hipFileRead()` and `hipFileWrite()`, the async and batch submissions,
backend selection, and stats collection. Opening the driver, registering
files, buffers, and streams, and shutdown are setup, and can afford more.

- **Do expensive work once, at setup**: querying pointer attributes, reading
  mount information, checking whether a file can use the fastpath, and
  allocating bounce buffers.
- **On the hot path, avoid** heap allocations, copying the payload (only the
  fallback path copies it), string formatting and logging on success,
  unneeded system calls, HIP calls that synchronize, and needless or
  long-held locks. `std::unique_ptr`, `std::span`, and `std::string_view`
  cost nothing over their C equivalents, but copying a `std::shared_ptr` (an
  atomic update), a `std::function` that allocates, or a `std::string` built
  per I/O does.
- **Don't give up correctness or safety for speed.** If a change has to add
  work to the hot path, say so and explain why.

## Linters, warnings, and sanitizers (mandatory)

**Every change must pass every check below that covers the files it touches.
These checks are not optional.** A change with an outstanding finding isn't
finished, so don't hand it back. CI doesn't run all of them, and several jobs
run only when files matching a path filter change, so a green PR doesn't
prove a change is clean; run them yourself. Where existing code doesn't pass
a check yet, hold the files you touch to it and leave the rest alone.

Fix the code rather than suppressing a finding (`// NOLINT`,
`# shellcheck disable=`, `# pylint: disable=`, `-Wno-…`). Suppress a finding
only when it's unavoidable, for example because of how Google Test or pytest
works, and comment why. Compiler warnings have their own mechanism, below.

### C and C++

- **clang-format 18**, with [`.clang-format`](.clang-format). Reformat with
  `util/format-source.sh 18`, or check without changing anything:

  ```bash
  git ls-files -z '*.c' '*.h' '*.cpp' ':!third-party' \
    | xargs -0 clang-format-18 --dry-run -Werror
  ```

  CI:
  [hipfile-clang-format.yml](../../.github/workflows/hipfile-clang-format.yml).
- **No compiler warnings, with GCC or Clang.** They warn about different
  things, so build with each (`-DCMAKE_CXX_COMPILER=g++`, and `amdclang++`
  or `clang++`) in separate build directories, in both `Debug` and an
  optimized type, since some warnings (and `_FORTIFY_SOURCE`) only appear
  with optimization. The warning lists
  ([`HipFileGNUCompilerOptions.cmake`](cmake/HipFileGNUCompilerOptions.cmake),
  [`HipFileClangCompilerOptions.cmake`](cmake/HipFileClangCompilerOptions.cmake))
  aren't errors by default, so configure with `-DCMAKE_CXX_FLAGS=-Werror`, as
  CI does.
- **Suppress a warning only when it's unavoidable**, and then only with the
  macros in [`shared/hipfile-warnings.h`](shared/hipfile-warnings.h), never
  with raw `#pragma`s, `-Wno-…` flags, or changes to the warning lists. Add
  an `HIPFILE_WARN_<THING>_OFF`/`_ON` pair there, following the instructions
  at the top of the file, add both to `StatementMacros` in
  [`.clang-format`](.clang-format), wrap the smallest region you can, and
  comment why the warning can't be fixed. The per-file GCC options on
  `thread-pool.cpp` in
  [`src/amd_detail/CMakeLists.txt`](src/amd_detail/CMakeLists.txt), for
  Taskflow's warnings, are an existing exception; don't add others without
  asking. Third-party code (`third-party/`, and GoogleTest from
  `FetchContent`) is included as `SYSTEM`, so its warnings don't count.
- **clang-tidy**: configure with Clang and `-DHIPFILE_USE_CLANG_TIDY=ON`, and
  build; it must report nothing. [`.clang-tidy`](.clang-tidy) enables only
  the `clang-analyzer-*` checks, which ROCm's clang-tidy lacks, so point
  `HIPFILE_CLANG_TIDY_EXECUTABLE` at an upstream clang-tidy whose LLVM major
  version matches ROCm's (the CI workflow sets the version it uses). If it
  reports "no checks enabled," it's the wrong clang-tidy; don't work around
  it. CI:
  [hipfile-clang-tidy.yml](../../.github/workflows/hipfile-clang-tidy.yml).
  **Advisory:** keep new code clean of what hipObject's much broader set
  (`bugprone-*`, `cppcoreguidelines-*`, `modernize-*`, `readability-*`, and
  more) would report.
- **include-what-you-use**: configure with Clang and `-DHIPFILE_USE_IWYU=ON`
  in a separate build directory (it leaves out the warning flags), and
  build. Some of its suggestions are wrong; make the right ones for the files
  you touch. Fix one that names a library's internal header with a mapping in
  a `cmake/iwyu-*.imp` file, like [`iwyu-libc.imp`](cmake/iwyu-libc.imp).
  Nothing in CI runs it.
- **Sanitizer-clean**: test any change that involves pointers, memory, or
  concurrency with the sanitizers. With Clang, build in two directories, one
  with `-DHIPFILE_USE_SANITIZERS=ON` (ASan, LSan, UBSan, and the rest in
  [`HipFileSanitizers.cmake`](cmake/HipFileSanitizers.cmake)) and one with
  `-DHIPFILE_USE_THREAD_SANITIZER=ON`, and run `ctest -L unit` in each; any
  report fails its test. The first configure needs network access, to fetch
  GoogleTest. If ThreadSanitizer aborts with "unexpected memory mapping," run
  `sudo sysctl vm.mmap_rnd_bits=28`. The LeakSanitizer suppressions
  ([`cmake/lsan-suppressions.supp`](cmake/lsan-suppressions.supp)) cover only
  the ROCm runtime; don't add any for hipFile's code, or sanitizer flags
  anywhere but `HipFileSanitizers.cmake`. CI:
  [hipfile-sanitizers.yml](../../.github/workflows/hipfile-sanitizers.yml).
- **Advisory: valgrind-clean.** Memcheck finds reads of uninitialized
  memory, which the sanitizers don't. Check that a change adds no reports in
  a GCC build without the sanitizers:

  ```bash
  ctest --test-dir <gcc-build-dir> -L unit -T memcheck --output-on-failure \
    --overwrite "MemoryCheckCommand=$(command -v valgrind)" \
    --overwrite "MemoryCheckCommandOptions=--leak-check=full --track-origins=yes --num-callers=40 --error-exitcode=1"
  ```

  The reports go to `Testing/Temporary/MemoryChecker.*.log`, not to ctest's
  output. Don't use an optimized Clang build: with
  `-ftrivial-auto-var-init=pattern`, it can emit x87 instructions that
  valgrind emulates with too few bits. Nothing in CI runs valgrind.
- **Lifetime safety, when a change could leave something dangling**, such as
  a pointer, reference, iterator, or view into memory something else owns, or
  one stored in an object or lambda that outlives the call. Build once with
  Clang and `-DCMAKE_CXX_FLAGS=-Wlifetime-safety` and read what it reports
  for the files you touched. Fix the real reports; leave the false positives
  alone, without suppressions. Most come from its
  `-Wlifetime-safety-invalidation` subgroup, which treats changing a
  container as invalidating references to the container itself. Nothing in
  CI runs it.

### Shell

Shell scripts are bash, not POSIX sh: start them with `#!/usr/bin/env bash`
(see [STYLEGUIDE.md](STYLEGUIDE.md)).

- **shellcheck** every shell script:
  `git ls-files -z '*.sh' | xargs -0 shellcheck`. CI:
  [hipfile-shellcheck.yml](../../.github/workflows/hipfile-shellcheck.yml),
  when a `.sh` file or the workflow changes.
- **The `run:` blocks in `.github/workflows/hipfile-*.yml` are bash scripts
  too**, and the shellcheck workflow doesn't see them. Check them with
  [actionlint](https://github.com/rhysd/actionlint), which runs shellcheck
  on each, from the top of the monorepo:
  `actionlint .github/workflows/hipfile-*.yml`. **Advisory** for existing
  workflows, which have many findings, but a block you add or change must be
  clean. actionlint's complaint about the self-hosted `AIS` runner label is
  spurious. Neither tool sees scripts passed to `bash -c` inside
  `docker exec`, so check those by hand. Nothing in CI runs actionlint.

### CMake

- **cmakelint**, with [`.cmakelintrc`](.cmakelintrc):

  ```bash
  git ls-files -z 'CMakeLists.txt' '*/CMakeLists.txt' '*.cmake' \
    | xargs -0 cmakelint
  ```

  CI: [hipfile-cmakelint.yml](../../.github/workflows/hipfile-cmakelint.yml),
  when a CMake file, `.cmakelintrc`, or the workflow changes.
- Use modern, target-based CMake, and keep to the minimum CMake version in
  the top-level `CMakeLists.txt` (3.21); hipFile hasn't moved to CMake 4.

### Python

Python code must run on **Python 3.10**, so nothing newer, such as `tomllib`,
`except*`, or `typing.Self` (all 3.11).

- **black**, **pylint**, and **vermin**, over every Python file, including
  the extensionless `tools/ais-check/ais-check`. vermin catches the newer
  language features and modules the others miss, and prints
  `Target versions not met` if it finds one:

  ```bash
  git ls-files -z '*.py' tools/ais-check/ais-check \
    | xargs -0 black --check --target-version py310
  git ls-files -z '*.py' tools/ais-check/ais-check \
    | xargs -0 pylint --py-version=3.10
  git ls-files -z '*.py' tools/ais-check/ais-check \
    | xargs -0 vermin -t=3.10- --violations --no-tips
  ```

  CI ([hipfile-pylint.yml](../../.github/workflows/hipfile-pylint.yml)) runs
  black and pylint when a Python file or the workflow changes. None of these
  tools read the Cython sources (`.pyx`, `.pxd`), so keep them consistent
  with the Python code by hand.
- **Building the bindings**: install them as an editable package in a
  virtual environment (`.venv`, if one exists) with `pip install -e python`.
  If that can't find hipFile, give the paths to a built one:

  ```bash
  pip install -e python \
    -Ccmake.define.HIPFILE_INCLUDE_DIR="$PWD/include" \
    -Ccmake.define.HIPFILE_LIBRARY=<build-dir>/src/amd_detail/libhipfile.so \
    -Ccmake.define.HIP_INCLUDE_DIR=/opt/rocm/include
  ```

  CI tests the bindings from an sdist
  ([hipfile-python-bindings.yml](../../.github/workflows/hipfile-python-bindings.yml)),
  and `ais-check` on Python 3.10 and the latest release
  ([hipfile-ais-check.yml](../../.github/workflows/hipfile-ais-check.yml)).

### Spelling and documentation

- **codespell**, with [`.codespellrc`](.codespellrc), over the tracked files,
  Markdown included: `git ls-files -z ':!third-party' | xargs -0 codespell`.
  CI ([hipfile-codespell.yml](../../.github/workflows/hipfile-codespell.yml))
  uses [`.codespellignore`](.codespellignore) as its word list; keep that
  file and `ignore-words-list` in `.codespellrc` in step.
- **The documentation must build.** Read the Docs
  ([`.readthedocs.yaml`](.readthedocs.yaml)) runs doxygen in `docs/doxygen`
  and then Sphinx on `docs/conf.py`. To build it locally, with the packages
  in `docs/sphinx/requirements.txt` installed:

  ```bash
  (cd docs/doxygen && doxygen Doxyfile)
  python3 -m sphinx -b html docs docs/_build/html
  ```

  Nothing in hipFile's CI builds the documentation. **Advisory:** doxygen
  shouldn't warn about the public headers; it currently warns about
  undocumented macros. `-DHIPFILE_BUILD_DOCS=ON` adds a `doc` target that
  runs only doxygen, from
  [`docs/doxygen/Doxyfile.in`](docs/doxygen/Doxyfile.in); keep it and
  `Doxyfile` in step.
- **Document user-visible behavior.** A new environment variable or
  configuration setting goes in
  [docs/reference/hipFile-environment-variables.rst](docs/reference/hipFile-environment-variables.rst),
  and a new limitation in [docs/troubleshooting](docs/troubleshooting). A new
  documentation page also needs an entry in
  [`docs/sphinx/_toc.yml.in`](docs/sphinx/_toc.yml.in).

## Changelog

**Every user-facing change needs an entry in [CHANGELOG.md](CHANGELOG.md), in
the same change.** User-facing means anything someone building, installing,
or calling hipFile could notice: the public API, what its documentation
promises or requires (see [Public API](#public-api)), library behavior,
environment variables, CMake options and variables, the installed files and
packages, the tools (`ais-check`, `ais-stats`), dependencies and supported
platforms, and the examples. CI-only, test-only, and internal refactoring
changes don't need one. Changes to the Python bindings go in
[python/CHANGELOG.md](python/CHANGELOG.md) instead.

Add the entry to the topmost (unreleased) version, under `Added`, `Changed`,
`Fixed`, `Removed`, or `Known issues`. Write it for users, like the
existing entries: say what changed, and what users have to do differently, if
anything.

## Commits and pull requests

rocm-systems uses [Conventional Commits](https://www.conventionalcommits.org/)
for commit subjects and PR titles, with the project as the scope:
`type(hipfile): summary`, as in `fix(hipfile): Attribute IO stats to the
buffer's GPU`. The type is one of `feat`, `fix`, `docs`, `test`,
`refactor`, `perf`, `build`, `ci`, `chore`, or `revert`. Write the summary in
the imperative, without a trailing period. PRs are squash-merged, so the PR
title becomes the commit subject.

Fill in the sections of the monorepo's
[pull request template](../../.github/pull_request_template.md), including a
JIRA ID or issue reference. The Systems PR Bot
([`tools/systems_pr_bot`](../../tools/systems_pr_bot)) fails a PR whose
description doesn't, and warns, without failing, about a PR that changes
code but no test.

## Review your change before you finish

When you've modified the repository, review the whole change as a reviewer
would. Read all of `git diff`, and every new file `git status` lists, from
start to finish. Every PR gets a Copilot review, and these are what it and
hipFile's reviewers ask for most often:

- **An incomplete change.** Find every other place that has to change with
  this one, and change it or say why not. A new backend, for example, goes
  in every list of backends, including the stats report and
  `hipFileGetStatsL1()` and `hipFileGetStatsL3()`; and since `ais-stats` maps
  the stats from another process, a layout change needs a new version.
- **A missing or wrong changelog entry or document**, including the header
  comments and the environment variable reference. Neither may claim more,
  or less, than the diff does.
- **A new file left out of the change.** Name each new file when you hand the
  change back, so it gets committed.
- **A test that passes without the fix.** Revert the fix in a scratch copy
  and check. Match the arguments that matter, such as the device or the
  `error_code`, not just call counts or an exception's type, and use a
  buffer on a GPU other than GPU 0 and the current device. Have the mock
  fail the call to test new error paths.
- **An exception from `noexcept` code.** The `Hip` and `Sys` wrappers throw,
  so a failing `hipFree()` or `hipSetDevice()` in a destructor or deleter
  calls `std::terminate`. Catch and log there, with a `try` for each release
  so one failure doesn't skip the rest.
- **Cleanup that's wrong on some path.** Follow each failure branch, and ask
  what happens if an allocation or insertion throws partway: a constructor
  that throws leaks what it created unless a local RAII owner holds it, and
  a map entry inserted before its initialization fails is left half-built.
- **A race with teardown.** A call releases `state_mutex` before it uses
  what it found, so `hipFileDriverClose()`, a deregistration, or
  `hipFileBatchIODestroy()` can run in between. Check each point, test the
  interleaving, and don't hand out a raw handle that another thread can
  destroy.
- **The wrong HIP device or thread.** HIP calls use the calling thread's
  current device, which on a pool thread needn't be the buffer's or the
  stream's, so set it (and restore it). Don't call HIP from a
  `hipLaunchHostFunc()` callback, and remember that `hipFree()` synchronizes
  the device, so it can deadlock a callback or a pool thread.
- **An unhandled edge case**, such as a short read or write, an output file
  that's the input, a timeout that overflows when converted, or
  `std::thread::hardware_concurrency()` returning 0.
- **Words that don't match the code.** Comments, documentation, changelog
  entries, and the PR description must claim only what the diff does.

Fix what you find, then run the
[checks](#linters-warnings-and-sanitizers-mandatory) again.
