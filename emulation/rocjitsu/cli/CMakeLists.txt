# rocjitsu — top-level CMake wrapper around cargo.
#
# This is a thin orchestration layer; the real build is done by
# cargo. It exists so rocjitsu can be (a) configured with the same
# `cmake -S . -B build && cmake --build build` recipe used by the
# rest of the monorepo and (b) installed and tested via `ctest`.
#
# Targets:
#   * `rocjitsu`            — runs `cargo build --release`, output is
#                           `target/release/rocjitsu`
#   * `install` (default) — copies the release binary to
#                           `${CMAKE_INSTALL_PREFIX}/bin/rocjitsu`
#   * `ctest`             — runs the full Rust test suite via
#                           `cargo test --release --workspace`. The
#                           end-to-end suites bring up real sessions on
#                           the `rocjitsu` backend and skip when its
#                           runtime library is not discoverable. It also
#                           runs `cargo clippy` and `cargo fmt --check`,
#                           so a lint or formatting regression is a test
#                           failure like any other.
#   * `rocjitsu_lint`       — just those two checks, without building or
#                           running the suite. Not part of `all`.
#
# Options:
#   * `ROCJITSU_ALLOW_TEST_SKIP` (OFF)  — let the end-to-end suites report
#                                       success when the `rocjitsu`
#                                       runtime is missing and every
#                                       session test therefore skipped.
#                                       Off by default: a suite that
#                                       silently proves nothing is worse
#                                       than a red one.
#   * `ROCJITSU_LINT_TESTS`     (ON)    — register the clippy and rustfmt
#                                       checks as ctests.
#   * `ROCJITSU_CARGO_PROFILE`  (release)
#   * `ROCJITSU_CARGO`          (auto)  — override cargo binary
#
cmake_minimum_required(VERSION 3.20)
project(rocjitsu NONE)

# HotSwap is a full LLVM + COMGR + ROCR source build, so it is opt-in. When ON,
# cmake/Hotswap.cmake builds and stages the HotSwap artifacts rocjitsu discovers.
option(
    ROCJITSU_BUILD_HOTSWAP
    "Build HotSwap (COMGR + patched ROCR + intercept) from source"
    OFF
)
option(
    ROCJITSU_ALLOW_TEST_SKIP
    "Pass ROCJITSU_E2E_ALLOW_SKIP=1 so the e2e suites may skip without a \
rocjitsu runtime"
    OFF
)
option(
    ROCJITSU_LINT_TESTS
    "Register `cargo clippy` and `cargo fmt --check` as ctest cases"
    ON
)
set(ROCJITSU_CARGO_PROFILE "release" CACHE STRING "Cargo build profile")
set(ROCJITSU_CARGO
    ""
    CACHE FILEPATH
    "Path to cargo binary (auto-detect if empty)"
)

# Whether this directory is being configured on its own or as part of a
# larger project (`add_subdirectory` from emulation/rocjitsu). The
# difference is what a missing cargo means: configuring *this* project
# without cargo is a mistake worth failing on, since there is nothing
# else here to build, while the same absence inside the rocjitsu build
# should leave the C++ tree buildable and skip what needs the CLI.
set(_rocjitsu_standalone TRUE)
if(NOT CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
    set(_rocjitsu_standalone FALSE)
endif()

if(NOT ROCJITSU_CARGO)
    find_program(ROCJITSU_CARGO_FOUND cargo)
    set(ROCJITSU_CARGO "${ROCJITSU_CARGO_FOUND}")
endif()
if(ROCJITSU_CARGO)
    message(STATUS "rocjitsu: using cargo at ${ROCJITSU_CARGO}")
elseif(_rocjitsu_standalone)
    message(FATAL_ERROR "rocjitsu: cargo not found, and it is the whole build")
else()
    message(
        STATUS
        "rocjitsu: cargo not found; the CLI and everything that runs through it \
will not be built"
    )
    # Nothing below can be defined without cargo, and the enclosing build
    # detects the absence by `RJ_CLI_BIN` never being set.
    return()
endif()

set(_rocjitsu_env)

# Each end-to-end suite has one test that fails when every *other* test in
# it skipped for a missing emulator runtime, so a build without rocjitsu
# cannot go green while proving nothing. That is the right default, but it
# makes `ctest` red on the rocjitsu-only build documented in
# docs/building.md — so the acknowledgement has to be reachable from here
# and not only from a hand-exported environment variable.
if(ROCJITSU_ALLOW_TEST_SKIP)
    list(APPEND _rocjitsu_env "ROCJITSU_E2E_ALLOW_SKIP=1")
    message(
        STATUS
        "rocjitsu: end-to-end suites may skip (ROCJITSU_E2E_ALLOW_SKIP=1)"
    )
endif()

# Cargo features. The only optional things left are the emulator
# backends, and the default (`rocjitsu`) is what the suites run on, so a
# plain build selects nothing here. Enable another backend with
# `-DROCJITSU_CARGO_FEATURES=hotswap`.
set(ROCJITSU_CARGO_FEATURES
    ""
    CACHE STRING
    "Extra cargo features (comma-separated)"
)

set(_rocjitsu_features)
if(ROCJITSU_CARGO_FEATURES)
    set(_rocjitsu_features "--features" "${ROCJITSU_CARGO_FEATURES}")
    message(STATUS "rocjitsu: cargo features: ${ROCJITSU_CARGO_FEATURES}")
endif()

# The end-to-end suites run their sessions on the default `rocjitsu`
# backend and skip, loudly, when its runtime library is absent — so they
# need no extra features beyond whatever the build already selected.
set(_rocjitsu_test_features ${_rocjitsu_features})

# Cargo's own name for the profile's output directory is not always the
# profile's name: `dev` builds into `debug`. Anything else is a custom
# profile, which cargo needs told about by name.
set(_rocjitsu_profile_flag "")
if(ROCJITSU_CARGO_PROFILE STREQUAL "release")
    set(_rocjitsu_profile_flag "--release")
    set(_rocjitsu_profile_dir "release")
elseif(
    ROCJITSU_CARGO_PROFILE STREQUAL "dev"
    OR ROCJITSU_CARGO_PROFILE STREQUAL "debug"
)
    set(_rocjitsu_profile_dir "debug")
else()
    set(_rocjitsu_profile_flag "--profile" "${ROCJITSU_CARGO_PROFILE}")
    set(_rocjitsu_profile_dir "${ROCJITSU_CARGO_PROFILE}")
endif()

# Where cargo will put the binary.
#
# Asked of cargo rather than assumed to be `<source>/target`, because it
# is not always: `$CARGO_TARGET_DIR` and a `build.target-dir` in any
# `.cargo/config.toml` on the way up the tree both move it, and CI images
# and container builds routinely set one to share a build cache. Assuming
# the default produced a configure that succeeded, a build that succeeded,
# and a `${RJ_CLI_BIN}` naming a file that was never written — so every
# test that runs the CLI failed to start, pointing at a path nothing had
# put anything at.
set(_rocjitsu_target_root "${CMAKE_CURRENT_SOURCE_DIR}/target")
execute_process(
    COMMAND ${ROCJITSU_CARGO} metadata --format-version 1 --no-deps --offline
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    OUTPUT_VARIABLE _rocjitsu_metadata
    ERROR_QUIET
    RESULT_VARIABLE _rocjitsu_metadata_result
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
if(_rocjitsu_metadata_result EQUAL 0)
    string(
        JSON _rocjitsu_metadata_target_dir
        ERROR_VARIABLE _rocjitsu_metadata_error
        GET "${_rocjitsu_metadata}"
        target_directory
    )
    if(_rocjitsu_metadata_target_dir AND NOT _rocjitsu_metadata_error)
        set(_rocjitsu_target_root "${_rocjitsu_metadata_target_dir}")
    endif()
endif()
set(_rocjitsu_target_dir "${_rocjitsu_target_root}/${_rocjitsu_profile_dir}")
message(STATUS "rocjitsu: cargo builds into ${_rocjitsu_target_dir}")

set(_rocjitsu_bin "${_rocjitsu_target_dir}/rocjitsu")

# What an enclosing build needs to run the CLI: where the binary lands,
# and the target that puts it there. Cargo is not a CMake target, so
# there is no `$<TARGET_FILE:…>` to ask — a test that runs the CLI names
# the path and depends on the target, and these two names are how.
if(NOT _rocjitsu_standalone)
    set(RJ_CLI_BIN "${_rocjitsu_bin}" PARENT_SCOPE)
    set(RJ_CLI_TARGET rocjitsu_build PARENT_SCOPE)
endif()

# The same path in the cache, for callers that are not cmake. Where cargo
# writes the binary depends on `$CARGO_TARGET_DIR` and any
# `.cargo/config.toml` on the way up the tree, which is why it is asked
# for above rather than assumed — so a CI job or packaging script that
# needs the binary should ask the configured build too, not reproduce
# cargo's layout rules and drift from them.
set(RJ_CLI_BIN
    "${_rocjitsu_bin}"
    CACHE FILEPATH
    "Path to the rocjitsu CLI this build produces"
    FORCE
)

add_custom_target(
    rocjitsu_build
    ALL
    COMMAND
        ${CMAKE_COMMAND} -E env ${_rocjitsu_env} ${ROCJITSU_CARGO} build
        ${_rocjitsu_profile_flag} ${_rocjitsu_features}
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    BYPRODUCTS ${_rocjitsu_bin}
    COMMENT "Building rocjitsu with cargo (${ROCJITSU_CARGO_PROFILE})"
    VERBATIM
)

# HotSwap source build (opt-in). Kept in its own module since it pulls in
# ExternalProject and a multi-stage LLVM/COMGR/ROCR build.
if(ROCJITSU_BUILD_HOTSWAP)
    include(${CMAKE_CURRENT_SOURCE_DIR}/hotswap/cmake/Hotswap.cmake)
endif()

# `mirage` is installed as a symlink beside `rocjitsu`. That is the name
# this CLI shipped under before the two were merged, and it is in scripts,
# CI jobs, shell history and documentation outside this repository; a
# symlink costs nothing and the binary answers to either. Relative, so the
# pair survives being moved, and relocated with `DESTDIR` staging intact.
install(PROGRAMS ${_rocjitsu_bin} DESTINATION bin)
install(
    CODE
        "
    set(_bin \"\$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}/bin\")
    file(CREATE_LINK rocjitsu \"\${_bin}/mirage\" SYMBOLIC)
    message(STATUS \"Installing: \${_bin}/mirage -> rocjitsu\")
"
)

# `BUILD_TESTING` is the enclosing build's switch and is undefined when
# this project is configured on its own, where testing is always on.
if(DEFINED BUILD_TESTING AND NOT BUILD_TESTING)
    return()
endif()

enable_testing()

# Full Rust workspace test suite: every crate's unit tests plus the
# integration suites under `tests/` (e2e, container_e2e, matrix_e2e,
# strain, the emulator benchmark) and `supervisor/tests/`. `--workspace`
# picks up each member's integration targets, so a new `tests/*.rs` file
# needs no change here. Session tests skip without a rocjitsu runtime;
# see ROCJITSU_ALLOW_TEST_SKIP above for what that means for `ctest`.
#
# Nested in the rocjitsu build, the runtime under test is the one this
# build just produced, and it is named outright: left to discovery the
# suites search the ROCm install locations, where an unrelated
# librocjitsu.so answers and a green result says nothing about this tree.
# The same two variables the C++ tests are given, for the same reason.
set(_rocjitsu_test_env ${_rocjitsu_env})
if(NOT _rocjitsu_standalone AND TARGET rocjitsu_shared)
    list(
        APPEND _rocjitsu_test_env
        "ROCJITSU_LIB=$<TARGET_FILE:rocjitsu_shared>"
    )
endif()
if(NOT _rocjitsu_standalone AND TARGET rocjitsu_hooks)
    list(
        APPEND _rocjitsu_test_env
        "ROCJITSU_HOOKS_LIB=$<TARGET_FILE:rocjitsu_hooks>"
    )
endif()

add_test(
    NAME cargo_test
    COMMAND
        ${CMAKE_COMMAND} -E env ${_rocjitsu_test_env} ${ROCJITSU_CARGO} test
        ${_rocjitsu_profile_flag} --workspace ${_rocjitsu_test_features}
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
set_tests_properties(cargo_test PROPERTIES TIMEOUT 1800)

# The workspace lint policy (Cargo.toml, `[workspace.lints]`) is written at
# deny level, but cargo only applies the clippy half when clippy is the
# thing being run: `cargo build` and `cargo test` never see it. Left to
# `ctest` alone the policy is advisory, and the way that fails is quiet —
# a `#[allow]` widened to cover a whole module, or an `unwrap` added to
# shipping code, lands green.
#
# Both flags on the clippy line are load-bearing. `--all-targets` reaches
# the test and bench targets, where the unwrap/expect opt-outs live and
# could otherwise spread unnoticed; `--all-features` reaches the
# `hotswap` backend, which the default feature set leaves uncompiled.
# `-D warnings` is what promotes the warn-level entries in the policy
# (`unreachable_pub`, `unused_qualifications`,
# `missing_debug_implementations`) into failures — without it they scroll
# past and ctest still reports success.
#
# These are separate ctests rather than one, so a red result says which
# of the two gates broke without reading the log.
set(_rocjitsu_clippy_cmd
    ${CMAKE_COMMAND}
    -E
    env
    ${_rocjitsu_env}
    ${ROCJITSU_CARGO}
    clippy
    ${_rocjitsu_profile_flag}
    --workspace
    --all-targets
    --all-features
    --
    -D
    warnings
)
set(_rocjitsu_fmt_cmd
    ${ROCJITSU_CARGO}
    fmt
    --all
    --
    --check
)

if(ROCJITSU_LINT_TESTS)
    add_test(
        NAME cargo_clippy
        COMMAND ${_rocjitsu_clippy_cmd}
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    )
    set_tests_properties(cargo_clippy PROPERTIES TIMEOUT 1800)

    add_test(
        NAME cargo_fmt
        COMMAND ${_rocjitsu_fmt_cmd}
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    )
    set_tests_properties(cargo_fmt PROPERTIES TIMEOUT 300)
endif()

# The same two checks as a target, for the common case of wanting them
# without a 30-minute test run. Deliberately not in `all`: a formatting
# nit should not stop you building the binary you are debugging.
add_custom_target(
    rocjitsu_lint
    COMMAND ${_rocjitsu_fmt_cmd}
    COMMAND ${_rocjitsu_clippy_cmd}
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Checking rocjitsu formatting and lints"
    VERBATIM
)

# rocjitsu-corpus gfx1250 regression, run through rocjitsu + an emulator
# (HotSwap by default). Opt-in via ROCJITSU_RUN_CORPUS; tests SKIP when the
# corpus, IREE tools, or emulator install are unavailable.
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/RocjitsuCorpus.cmake)
