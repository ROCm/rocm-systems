# shared/lttng/ — Vendored LTTng-UST + userspace-rcu

This directory holds the canonical, single-source vendored copy of
**LTTng-UST** and **userspace-rcu** used by ROCm components that emit
LTTng tracepoints, plus the shared CMake helpers that build them.

It exists so that multiple consumers (today: `rocr-runtime`,
`clr/hipamd`) reference one set of submodule SHAs and one CMake module,
instead of carrying duplicate vendored trees and duplicate CMake.

## Layout

```
shared/lttng/
├── README.md                   ← this file
├── lttng-ust/                  ← submodule, pinned to v2.13.7
├── userspace-rcu/              ← submodule, pinned to v0.14.0
└── cmake/
    ├── VendorLttng.cmake       ← ExternalProject build of both libs
    └── VendorLttngRpath.cmake  ← post-install RPATH→$ORIGIN rewrite
```

## Consumers

- `projects/rocr-runtime/runtime/hsa-runtime/CMakeLists.txt` (gated on
  `ROCR_ENABLE_LTTNG_UST=ON`, the default).
- `projects/clr/hipamd/src/CMakeLists.txt` (gated on
  `HIP_ENABLE_LTTNG_UST=ON`, the default).

This pattern matches `shared/ctest/` (consumed by
`projects/rocprofiler-compute/CMakeLists.txt`).

## How to consume from a project's CMakeLists.txt

Each consumer still produces its **own** private vendored build of
LTTng-UST and userspace-rcu under its build tree (the source tree is
shared, the build artifacts are not). This keeps standalone-project
builds self-contained and avoids cross-project ordering constraints.

```cmake
if(MY_PROJECT_ENABLE_LTTNG_UST)
    # 4× ".." from <project>/<sub>/<sub>/CMakeLists.txt to repo root.
    # Adjust to your CMakeLists.txt location.
    set(LTTNG_VENDORED_ROOT
        "${CMAKE_CURRENT_SOURCE_DIR}/../../../../shared/lttng"
        CACHE PATH "Path to shared vendored LTTng tree (shared/lttng/)")
    if(NOT EXISTS "${LTTNG_VENDORED_ROOT}/cmake/VendorLttng.cmake")
        message(FATAL_ERROR
            "MY_PROJECT_ENABLE_LTTNG_UST=ON but shared/lttng/ was not "
            "found at ${LTTNG_VENDORED_ROOT}. Either checkout the "
            "rocm-systems super-repo (which provides shared/lttng/), "
            "set -DLTTNG_VENDORED_ROOT=<path>, or disable with "
            "-DMY_PROJECT_ENABLE_LTTNG_UST=OFF.")
    endif()
    set(LTTNG_VENDORED_URCU_SRC "${LTTNG_VENDORED_ROOT}/userspace-rcu")
    set(LTTNG_VENDORED_UST_SRC  "${LTTNG_VENDORED_ROOT}/lttng-ust")
    include(${LTTNG_VENDORED_ROOT}/cmake/VendorLttng.cmake)
endif()
```

`VendorLttng.cmake` defines the following cache variables for the
consumer to use:

| Variable                   | Meaning                                              |
|----------------------------|------------------------------------------------------|
| `LTTNG_VENDORED_PREFIX`    | install prefix in the build tree                     |
| `LTTNG_VENDORED_INCLUDE_DIR` | header include path                                |
| `LTTNG_VENDORED_LIB_DIR`   | runtime `.so` directory                              |
| `LTTNG_VENDORED_PKGCONFIG` | directory containing the generated `.pc` files       |

…and these CMake targets:

| Target               | Meaning                                              |
|----------------------|------------------------------------------------------|
| `urcu_vendored`      | userspace-rcu install                                |
| `lttng_ust_vendored` | lttng-ust install (depends on `urcu_vendored`)       |

Consumers should `add_dependencies(<their target> lttng_ust_vendored)`
to guarantee the vendored libs are built before linking.

## Build dependencies (host packages)

Required at CMake configure / build time:

- `autoconf`
- `automake`
- `libtool` and `libtool-bin`
- `pkg-config` (or `pkgconf`)
- `patchelf` (hard requirement — used to rewrite installed `.so` RPATH
  to `$ORIGIN` so the vendored libs resolve transitively at runtime)

Debian/Ubuntu: `apt-get install autoconf automake libtool libtool-bin
pkg-config patchelf`. RHEL-family: `dnf install autoconf automake libtool
pkgconf-pkg-config patchelf`.

## Build order between consumers

When both consumers are built in the same tree, **HSA (rocr-runtime)
should always build before HIP (clr/hipamd)** — the runtime library
order at install time matters for the link-against-installed pattern HIP
uses. They are typically built one at a time, never concurrently.

The vendored `lttng-ust` / `userspace-rcu` source tree is shared and
read-only during builds, so concurrent CMake configures / builds of the
two consumers are safe in principle, but each consumer will produce its
own private install of the vendored libraries under its own
`<build>/_deps/lttng-prefix/` — there is no shared install tree.

## Standalone-project caveat

If you check out a single consumer (e.g. just `rocr-runtime/`) outside
of the rocm-systems super-repo, you must also check out `shared/lttng/`
alongside it (matching the relative path the consumer's CMakeLists.txt
expects, four directories above its location), or pass
`-DLTTNG_VENDORED_ROOT=<absolute path>` at CMake configure time.

If neither is available, disable LTTng with
`-DROCR_ENABLE_LTTNG_UST=OFF` (HSA) or `-DHIP_ENABLE_LTTNG_UST=OFF`
(HIP). All other functionality of those projects works without LTTng.

## Submodule pins

| Submodule       | Branch       | Pinned commit                              | Tag    |
|-----------------|--------------|--------------------------------------------|--------|
| `lttng-ust`     | `stable-2.13`| `04b0e69420e865e56dba55bd09621cb6dc61ec78` | v2.13.7|
| `userspace-rcu` | `stable-0.14`| `d1252aeb22cc8dbd9b1806afd66ffaf9ad6d7d07` | v0.14.0|

To upgrade, bump the submodule SHA and re-run a full build of both
consumers; verify there are no API breaks in `lttng-ust`'s
`tracepoint.h` or `lttng/ust-events.h` that affect the in-tree
tracepoint provider sources under `<project>/lttng/`.
