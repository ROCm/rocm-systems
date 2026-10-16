---
myst:
  html_meta:
    "description lang=en": "How to build AMD SMI from source."
    "keywords": "system, management, interface, contribute, contributing, ROCm, develop, testing"
---

# Build AMD SMI from source

To build AMD SMI as part of the ROCm Core SDK, see [TheRock build
instructions](https://github.com/ROCm/TheRock/blob/main/docs/development/README.md).
TheRock is the recommended way to build ROCm components from source.

Alternatively, you can build AMD SMI standalone using the following
instructions.

(build_reqs)=
## Required software

To build the AMD SMI library, the following components are required. Note that
the software versions specified were used during development; earlier
versions are not guaranteed to work.

* CMake (v3.20 or later) -- `python3 -m pip install cmake`
* amdclang++ from ROCm, or clang++ -- see [Compiler requirements](#build_compilers)
* g++ (v5.4.0 or later) -- Clang also uses its C++ standard library
* libdrm-dev (for Ubuntu and Debian)
* libssl-dev (for Ubuntu and Debian)
* libdrm-devel (for RPM-based distributions)
* openssl-devel (for RPM-based distributions)

In order to build the AMD SMI Python package, the following components are
required:

* Python (3.6.8 or later)
* virtualenv -- `python3 -m pip install virtualenv`

Users that wish to also build the AMD SMI Rust interface will also need the following components:

* Rust (1.56 or later)

(build_compilers)=
## Compiler requirements

AMD SMI selects a Clang compiler in the following priority order:

1. **amdclang++** (recommended) - AMD's optimized Clang compiler from ROCm
2. **clang++** - ROCm's clang++ if ROCm is installed, otherwise system clang++

**GCC is supported but not auto-selected.** To build with GCC, use a toolchain file (see below).

### How compilers are discovered

The build uses **`hipconfig --path`** for tool-based ROCm discovery.

**If ROCm is installed** (hipconfig found), only ROCm's LLVM directories,
`<rocm-path>/lib/llvm/bin` and then `<rocm-path>/llvm/bin`, are searched:

1. `amdclang++` and `amdclang`
2. `clang++` and `clang`

There is no fallback to the system clang++, so ROCm and system toolchains are
not mixed.

**If ROCm is not installed** (hipconfig not found), PATH is searched for
`clang++` and `clang` (for example `/usr/bin/clang++`).

Both compilers of a pair must run (`--version`): a name on PATH can be a ccache
link with no compiler behind it.

**Note:** The search never assumes a fixed path such as `/opt/rocm`; it uses
the path `hipconfig --path` reports, so several ROCm installations work.

## Build steps

1. Clone the rocm-systems repository to your local Linux machine
   and sparse-checkout the AMD SMI project.

   ```shell
   git clone --filter=blob:none --sparse https://github.com/ROCm/rocm-systems.git
   git -C rocm-systems sparse-checkout set projects/amdsmi
   cd rocm-systems/projects/amdsmi
   ```

2. The default installation location for the library and headers is `/opt/rocm`.
   Before installation, any old ROCm directories should be deleted:

   * `/opt/rocm`
   * `/opt/rocm-<version_number>`

3. Build the library by following the typical CMake build sequence (run as root
   user or use `sudo` before `make install` command); for instance:

   ```bash
   mkdir -p build
   cd build
   cmake ..          # Picks amdclang++, else clang++
   make -j $(nproc)
   make install
   ```

   The built library is located in the  `build/` directory. To build the `rpm`
   and `deb` packages use the following command:

   ```bash
   make package
   ```

   To build with AddressSanitizer instrumentation, see [Build with AddressSanitizer](#build_asan).

## Alternative compiler options

### Building with GCC

To build with GCC instead of amdclang++, use the provided toolchain file:

```bash
mkdir -p build
cd build
cmake .. -DCMAKE_TOOLCHAIN_FILE=../cmake/toolchains/amdsmi-gcc-toolchain.cmake
make -j $(nproc)
```

To use a specific GCC version, edit `cmake/toolchains/amdsmi-gcc-toolchain.cmake` or create your own toolchain file.

Setting `CC` or `CXX` also skips the Clang auto-detection. Set both, in a new
build directory, so C and C++ use the same toolchain: `CC=gcc CXX=g++ cmake ..`.

### Building with clang++

To use clang++ instead of amdclang++, explicitly specify the compiler with full path.

**Using ROCm's clang++:**

```bash
mkdir -p build
cd build
cmake .. -DCMAKE_CXX_COMPILER=/opt/rocm-<version_number>/lib/llvm/bin/clang++ -DCMAKE_C_COMPILER=/opt/rocm-<version_number>/lib/llvm/bin/clang
make -j $(nproc)
```

**Using system clang++:**

Requires clang to be installed (`sudo apt install clang` on Ubuntu/Debian).

```bash
mkdir -p build
cd build
cmake .. -DCMAKE_CXX_COMPILER=/usr/bin/clang++ -DCMAKE_C_COMPILER=/usr/bin/clang
make -j $(nproc)
```

### Troubleshooting compiler detection

**If cmake fails with "No suitable C++ compiler found (amdclang++ or clang++)":**

```bash
# ROCm installed: put hipconfig on PATH; ROCm's LLVM must be installed too
export PATH=$PATH:/opt/rocm/bin
"$(hipconfig --path)/lib/llvm/bin/amdclang++" --version

# No ROCm: install clang
sudo apt install clang     # Ubuntu/Debian
sudo dnf install clang     # RHEL/AlmaLinux/Fedora
```

**To check what compiler will be used:**

```bash
# A compiler counts only if it runs; a name on PATH can be a ccache link
amdclang++ --version
clang++ --version

# A new build directory's configure output names the compiler it picked
cmake .. | grep 'Clang compiler'

# An existing build directory keeps its compiler in the cache
grep '^CMAKE_CXX_COMPILER:' CMakeCache.txt
```

(rebuild_py_wrapper)=
## Rebuild the Python wrapper

The Python wrapper for the AMD SMI library is found in the [auto-generated
file](#py_lib_fs) `py-interface/amdsmi_wrapper.py`. It is essential to
regenerate this wrapper whenever there are changes to the C++ API. It is not
regenerated automatically.

To regenerate the wrapper, use the following command.

```shell
./update_wrapper.sh
```

After this command, the file in `py-interface/amdsmi_wrapper.py` will be updated
on compile.

```{note}
You need Docker installed on your system to regenerate the Python wrapper.
```

(build_tests)=
## Build the tests

To verify the build and capabilities of AMD SMI on your system, as well as to
see practical examples of its usage, you can build and run the available [tests
in the repository](https://github.com/ROCm/rocm-systems/tree/develop/projects/amdsmi/tests).
Follow these steps to build the tests:

```bash
mkdir -p build
cd build
cmake -DBUILD_TESTS=ON ..
make -j $(nproc)
```

(run_tests)=
### Run the tests

Once the tests are [built](#build_tests), you can run them by executing the
`amdsmitst` program. The executable can be found at `build/tests/amd_smi_test/`.

(build_asan)=
## Build with AddressSanitizer

`-DADDRESS_SANITIZER=ON` instruments the library and the C/C++ tests.

```bash
cmake -DADDRESS_SANITIZER=ON ..
make -j $(nproc)
```

With Clang, instrumented executables such as `amdsmitst` load the shared ASAN
runtime, which is not on the default library path. Add its directory before
running them, or configure with the GCC toolchain file instead:

```bash
# From the build directory
AMDSMI_CXX=$(sed -n 's/^CMAKE_CXX_COMPILER:[A-Z]*=//p' CMakeCache.txt)
ASAN_RT=$("$AMDSMI_CXX" -print-file-name="libclang_rt.asan-$(uname -m).so")
if [ -f "$ASAN_RT" ]; then
  export LD_LIBRARY_PATH="$(dirname "$ASAN_RT")${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
else
  echo "No shared Clang ASAN runtime for $AMDSMI_CXX"
fi
```

`-DENABLE_ASAN_PACKAGING=ON` produces a library-only `amd-smi-lib-asan` package.
The CLI and the Python bindings are excluded from it because neither can load an
instrumented `libamd_smi.so` unaided; see below.

(asan_python)=
### Run the CLI or the Python bindings against an instrumented library

`amd-smi` is a Python script, and the bindings are `ctypes`, so both reach the
library through `dlopen`. AddressSanitizer has to be initialized before the
process allocates any memory, which is long before `dlopen` runs, so an
uninstrumented Python aborts with:

```text
==1234==ASan runtime does not come first in initial library list; you should
either link runtime to your application or manually preload it with LD_PRELOAD.
```

Preload the runtime the library was linked against. Leak detection has to be off
because CPython does not free everything at shutdown.

```bash
ASAN_RT=$(ldd build/lib/libamd_smi.so | awk '/libclang_rt\.asan|libasan/ && $3 ~ /^\// {print $3}')
[ -n "$ASAN_RT" ] || echo "No ASAN runtime resolved; rebuild with -DADDRESS_SANITIZER=ON."
LD_PRELOAD=$ASAN_RT ASAN_OPTIONS=detect_leaks=0 amd-smi static
```

An empty `ASAN_RT` means the library is either uninstrumented or instrumented
against a runtime that is not on the search path, which `ldd` reports as
`=> not found`.

This applies to any Python process that imports `amdsmi`, not just the CLI. C and
C++ consumers linked with `-fsanitize=address` need no preload, because the
runtime is already in their initial library list.

(build_docs)=
## Build the docs

The [C/C++ API reference](../reference/amdsmi-cpp-api/index.md) is generated
with [Doxygen 1.15.0](https://www.doxygen.nl/manual/changelog.html#log_1_15_0),
which must be installed separately and available on your PATH.

1. Create a Python virtual environment and install documentation dependencies.

   ```bash
   # From rocm-systems/projects/amdsmi
   python3.12 -m venv docs/.venv
   source docs/.venv/bin/activate
   pip install -r docs/sphinx/requirements.txt
   ```

2. Use the following command to build the documentation using
   [Sphinx](https://www.sphinx-doc.org/en/master/).

   ```bash
   python -m sphinx docs docs/_build -j auto -E -v
   ```

3. Open `docs/_build/index.html` in your web browser to view the
   documentation. To serve the site locally instead, run:

   ```bash
   python -m http.server -d docs/_build
   # Go to http://localhost:8000 in your browser
   ```

For related information, see [Building
documentation](https://rocm.docs.amd.com/en/latest/contribute/building.html).
