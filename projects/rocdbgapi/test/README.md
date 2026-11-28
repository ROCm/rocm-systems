# ROCdbgapi test suite

This directory holds the GoogleTest-based test suite for the
`amd-dbgapi` library, split in tiers.
The directory structure is as follows:

```
test/
  unit/      OS-agnostic, no kernel driver, no GPU.  Drives internal
             seams (the `amd-dbgapi-internal` target) and the public
             API against in-process mocks.  Always runnable.
  feature/   Exercises the public `amd-dbgapi` library.  May need the
             kernel driver (KFD on Linux, KMD on Windows) and a GPU.
             Each test skips itself at runtime when its prerequisites
             are absent, so the binaries also run on hosts without
             hardware.
```

## Building

The suite is off by default.  `ENABLE_TESTS` is the master switch that
enables every tier.  `ENABLE_UNIT_TESTS` and `ENABLE_FEATURE_TESTS` enable
just one tier each and can be set independently of the master switch, e.g.
`-DENABLE_TESTS=OFF -DENABLE_FEATURE_TESTS=ON` builds only the feature
tier.  Future tiers will only turn on with `ENABLE_TESTS` or their own
dedicated option.

Typical configure + build from the repo root:

```shell
cmake -S projects/rocdbgapi -B build-rocdbgapi \
      -DENABLE_TESTS=ON
cmake --build build-rocdbgapi -j
```

See the main [README.md](../README.md) for GoogleTest/GMock prerequisites
and per-distro install instructions. If GoogleTest is not found or
incomplete, CMake will fail with instructions on how to install it for
your platform.

`make install` places built test binaries under
`tests/rocdbgapi/` relative to `CMAKE_INSTALL_PREFIX`.

## Running

From the build tree:

```shell
cd build-rocdbgapi
ctest --output-on-failure              # everything
ctest -L unit                          # OS-agnostic tier only
ctest -L feature                       # feature tier only
ctest -L feature-gpu                   # tests that need a GPU
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
| `feature`                | Feature-tier test.                                   |
| `feature-driver`         | Feature test that needs the kernel driver.           |
| `feature-kfd`            | ... on Linux (KFD).                                  |
| `feature-kmd`            | ... on Windows (KMD).                                |
| `feature-gpu`            | Feature test that needs a GPU.                       |
| `feature-gpu-<arch>`     | ... a GPU of a given architecture (e.g. `gfx942`).   |
| `os-linux`, `os-windows` | OS-specific tests.                                   |

The `feature-*` labels route tests to suitable CI runners.  They describe
what a test needs; they do not stop it from running elsewhere, which is why
tests must also skip at runtime.

## Adding a new unit test

```cmake
# test/unit/CMakeLists.txt
add_dbgapi_unit_test(my_new_unit_test
  SOURCES
    my_new_unit_test.cpp
  LIBS amd-dbgapi)
```

Unit tests link `GTest::gtest_main` automatically.  Use `amd-dbgapi-internal`
when you need to reach internal symbols; otherwise link `amd-dbgapi`.

A test must link exactly one of `amd-dbgapi` and `amd-dbgapi-internal`, never
both.  `amd-dbgapi-internal` is a static archive that publicly defines
`AMD_DBGAPI=AMD_DBGAPI_CALL`, so its API declarations carry no
`dllimport`/`dllexport`, whereas `amd-dbgapi` uses the import decorations.
Mixing them gives inconsistent decorations (and duplicate definitions).

`amd-dbgapi-internal` is built differently from the public library: RTTI is
on (GMock matchers need it) and `-Werror` is off, so a warning in a library
source does not break test builds.  The public library is still built with
`-fno-rtti` and `-Werror`.

Optional knobs:

* `LABELS extra-label ...`: extra CTest labels.
* `LIBS extra::link ...`: extra link dependencies.

## Adding a new feature test

```cmake
# test/feature/CMakeLists.txt
add_dbgapi_feature_test(my_new_feature_test
  SOURCES
    my_new_feature_test.cpp
    support/capabilities.cpp
  REQUIRES_DRIVER
  REQUIRES_GPU gfx942
  LIBS amd-dbgapi)
```

Feature tests link the public `amd-dbgapi` library.  They accept the same
`LABELS` and `LIBS` knobs as unit tests, plus:

* `REQUIRES_DRIVER`: the test needs the kernel driver at runtime.  Adds the
  `feature-driver` label and `feature-kfd` or `feature-kmd`.
* `REQUIRES_GPU [arch ...]`: the test needs a GPU.  Adds `feature-gpu`, and
  `feature-gpu-<arch>` for each architecture named.

The labels only describe the requirements.  Derive the test fixture from the
matching class in `support/dbgapi_fixture.h` (`dbgapi_fixture_t`,
`driver_fixture_t`, `gpu_fixture_t`) so that the test skips itself, using the
probes in `support/capabilities.h`, when the driver or the GPU is missing.

## Layout reference

```
test/
├── CMakeLists.txt              Top-level makefile.
├── README.md                   You are here.
├── unit/
│   ├── CMakeLists.txt          One add_dbgapi_unit_test() per binary.
│   ├── *_test.cpp              One file per subsystem under test.
│   └── support/                Mocks (MockOsDriver), shared fixtures.
└── feature/
    ├── CMakeLists.txt          One add_dbgapi_feature_test() per binary.
    ├── *_test.cpp              One file per feature under test.
    └── support/                Capability probes, fixtures, callbacks.
```

