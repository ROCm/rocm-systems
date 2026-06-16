# `api`

Minimal examples of the non-I/O parts of the hipFile API — calls that query or
configure the library rather than move data through the GPU. These do **not**
require an `O_DIRECT`-capable filesystem or even file arguments.

## The examples

| Program | What it shows | Args |
| --- | --- | --- |
| `get-version` | Read the hipFile version both ways: the `HIPFILE_VERSION_*` header macros (compile-time) and `hipFileGetVersion()` (runtime). | none |

## Building

### In-tree

Built by the parent hipFile project when `AIS_INSTALL_EXAMPLES=ON` (the
default). Unlike `basics/` and `async/`, these use the `ais_add_executable`
macro and link the `hipfile` target directly:

```bash
cd rocm-systems/projects/hipfile
cmake -DCMAKE_CXX_COMPILER=amdclang++ -DCMAKE_HIP_PLATFORM=amd \
      -DAIS_INSTALL_EXAMPLES=ON -B build
cmake --build build --parallel
```

### Standalone (against an installed hipFile)

Copy `CMakeLists.install.in` to `CMakeLists.txt` in a scratch copy of this
directory — it uses `find_package(hipfile)` instead of the in-tree macro:

```bash
mkdir -p /tmp/api-example
cp CMakeLists.install.in /tmp/api-example/CMakeLists.txt
cp get-version.cpp /tmp/api-example/
cmake -DCMAKE_PREFIX_PATH="/opt/rocm;/path/to/hipfile" -S /tmp/api-example -B /tmp/api-example/build
cmake --build /tmp/api-example/build
```

## Running

```bash
./get-version
```

Prints the version from the header symbols and from the runtime call. No file
or GPU memory is touched.
