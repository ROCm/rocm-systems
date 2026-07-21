# shared/lttng/ — LTTng-UST integration for ROCm tracepoints

This directory holds the **shared CMake glue** that ROCm components use to
consume **LTTng-UST** when emitting LTTng tracepoints (today: the HIP and HSA
runtimes' curated per-API tracepoint providers).

LTTng-UST and its `userspace-rcu` dependency are **not vendored here**. They are
provided as **TheRock bundled sysdeps** and consumed via `pkg-config`, matching
how TheRock handles other system libraries (numactl, libdrm, libnl, …). See
`ROCm/TheRock` → `third-party/sysdeps/linux/{lttng-ust,userspace-rcu}` and
`docs/development/dependencies.md`.

## Layout

```
shared/lttng/
├── README.md               ← this file
└── cmake/
    └── RocmLttng.cmake      ← pkg-config lookup → PkgConfig::LTTNG_UST
```

## How a component consumes it

Gate on your project's own option, then include the helper and link the target:

```cmake
if(ROCR_ENABLE_LTTNG_UST)      # or HIP_ENABLE_LTTNG_UST
    include(${CMAKE_CURRENT_SOURCE_DIR}/../../../../shared/lttng/cmake/RocmLttng.cmake)
    target_link_libraries(<your_target> PRIVATE PkgConfig::LTTNG_UST)
endif()
```

`RocmLttng.cmake` runs `pkg_check_modules(... lttng-ust>=2.13)` and defines the
imported target `PkgConfig::LTTNG_UST`. If LTTng-UST is not found it fails with
guidance (enable it in TheRock, install the dev package, or turn the option
off). It does **not** modify `PKG_CONFIG_PATH`: discovery relies on
`CMAKE_PREFIX_PATH` / `PKG_CONFIG_PATH` as provided by the TheRock super-build
(the sysdep) or the system.

## Where LTTng-UST comes from

| Build context | Provider |
|---------------|----------|
| TheRock super-build | `lttng-ust` sysdep, enabled with `-DTHEROCK_ENABLE_LTTNG=ON` (off by default). Installs privately under `lib/rocm_sysdeps/`. |
| Standalone project build | System install, e.g. `liblttng-ust-dev` (Debian/Ubuntu) / `lttng-ust-devel` (RHEL). |

## Building without LTTng

If LTTng-UST is unavailable (and not wanted), disable it per project:
`-DROCR_ENABLE_LTTNG_UST=OFF` (HSA) or `-DHIP_ENABLE_LTTNG_UST=OFF` (HIP). All
other functionality of those projects works without LTTng; the tracepoint
provider libraries simply are not built.

## Minimum version

LTTng-UST **>= 2.13** (the API/schema the curated providers are written
against). The TheRock sysdep ships 2.13.7; most current distributions satisfy
this too.
