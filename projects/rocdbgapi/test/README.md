# ROCdbgapi test suite

This directory holds the GoogleTest-based test suite for the
`amd-dbgapi` library.  Currently, there is only infrastructure
for unit tests that do not require a GPU.
The directory structure is as follows:

```
test/
  unit/      OS-agnostic, no kernel driver, no GPU.  Drives internal
             seams (the `amd-dbgapi-internal` target) and the public
             API against in-process mocks.  Always runnable.
```

## Building

The suite is off by default.  Unit tests are enabled with the CMake
option AMD_DBGAPI_BUILD_UNIT_TESTS.

Typical configure + build from the repo root:

```shell
cmake -S projects/rocdbgapi -B build-rocdbgapi \
      -AMD_DBGAPI_BUILD_UNIT_TESTS=ON
cmake --build build-rocdbgapi -j
```

GoogleTest + GMock are resolved in the following order:
1. If a parent project (e.g., TheRock superbuild) provides all four
   `GTest::*` targets (`gtest`, `gtest_main`, `gmock`, `gmock_main`),
   those are used.
2. Otherwise, `find_package(GTest CONFIG)` searches for a system
   installation.

If GoogleTest is not found or incomplete, CMake will fail with
instructions on how to install it for your platform.

## Running

From the build tree:

```shell
cd build-rocdbgapi
ctest --output-on-failure              # everything
ctest -L unit                          # OS-agnostic tier only
```

Individual binaries accept the standard GoogleTest flags
(`--gtest_filter`, `--gtest_list_tests`, `--gtest_repeat`, etc.):

### CTest labels

Labels are applied automatically by the `add_dbgapi_*_test()` helpers.
Conventions used across the suite:

| Label                    | Meaning                                              |
| ----------------------   | ---------------------------------------------------- |
| `dbgapi`                 | Any rocdbgapi test (applied to every target).        |
| `unit`                   | Unit-tier test.                                      |
| `os-linux`, `os-windows` | OS-specific tests.                                   |

## Adding a new unit test

```cmake
# test/unit/CMakeLists.txt
add_dbgapi_unit_test(my_new_unit_test
  SOURCES
    my_new_unit_test.cpp
  LIBS amd-dbgapi)
```

Unit tests link `GTest::gtest_main` automatically.

Optional knobs:

* `LABELS extra-label ...` — extra CTest labels.
* `LIBS extra::link ...` — extra link dependencies.

## Layout reference

```
test/
├── CMakeLists.txt              Top-level makefile.
├── README.md                   You are here.
├── unit/
│   ├── CMakeLists.txt          One add_dbgapi_unit_test() per binary.
│   ├── *_test.cpp              One file per subsystem under test.
│   └── support/                Mocks (MockOsDriver), shared fixtures.
```

