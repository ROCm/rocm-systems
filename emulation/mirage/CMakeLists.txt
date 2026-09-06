# mirage — top-level CMake wrapper around cargo.
#
# This is a thin orchestration layer; the real build is done by
# cargo. It exists so mirage can be (a) configured with the same
# `cmake -S . -B build && cmake --build build` recipe used by the
# rest of the monorepo and (b) installed and tested via `ctest`.
#
# Targets:
#   * `mirage`            — runs `cargo build --release`, output is
#                           `target/release/mirage`
#   * `install` (default) — copies the release binary to
#                           `${CMAKE_INSTALL_PREFIX}/bin/mirage`
#   * `ctest`             — runs the full Rust test suite via
#                           `cargo test --release --workspace`. The
#                           end-to-end suites bring up real sessions on
#                           the `rocjitsu` backend and skip when its
#                           runtime library is not discoverable. It also
#                           runs `cargo clippy` and `cargo fmt --check`,
#                           so a lint or formatting regression is a test
#                           failure like any other.
#   * `mirage_lint`       — just those two checks, without building or
#                           running the suite. Not part of `all`.
#
# Options:
#   * `MIRAGE_ALLOW_TEST_SKIP` (OFF)  — let the end-to-end suites report
#                                       success when the `rocjitsu`
#                                       runtime is missing and every
#                                       session test therefore skipped.
#                                       Off by default: a suite that
#                                       silently proves nothing is worse
#                                       than a red one.
#   * `MIRAGE_LINT_TESTS`     (ON)    — register the clippy and rustfmt
#                                       checks as ctests.
#   * `MIRAGE_CARGO_PROFILE`  (release)
#   * `MIRAGE_CARGO`          (auto)  — override cargo binary
#

# Temporary no-op change on mirage to exercise TheRock Multi-arch CI

cmake_minimum_required(VERSION 3.20)
project(mirage NONE)

# HotSwap is a full LLVM + COMGR + ROCR source build, so it is opt-in. When ON,
# cmake/Hotswap.cmake builds and stages the HotSwap artifacts mirage discovers.
option(
    MIRAGE_BUILD_HOTSWAP
    "Build HotSwap (COMGR + patched ROCR + intercept) from source"
    OFF
)
option(
    MIRAGE_ALLOW_TEST_SKIP
    "Pass MIRAGE_E2E_ALLOW_SKIP=1 so the e2e suites may skip without a \
rocjitsu runtime"
    OFF
)
option(
    MIRAGE_LINT_TESTS
    "Register `cargo clippy` and `cargo fmt --check` as ctest cases"
    ON
)
set(MIRAGE_CARGO_PROFILE "release" CACHE STRING "Cargo build profile")
set(MIRAGE_CARGO
    ""
    CACHE FILEPATH
    "Path to cargo binary (auto-detect if empty)"
)

if(NOT MIRAGE_CARGO)
    find_program(MIRAGE_CARGO_FOUND cargo REQUIRED)
    set(MIRAGE_CARGO "${MIRAGE_CARGO_FOUND}")
endif()
message(STATUS "mirage: using cargo at ${MIRAGE_CARGO}")

set(_mirage_env)

# Each end-to-end suite has one test that fails when every *other* test in
# it skipped for a missing emulator runtime, so a build without rocjitsu
# cannot go green while proving nothing. That is the right default, but it
# makes `ctest` red on the mirage-only build documented in
# docs/building.md — so the acknowledgement has to be reachable from here
# and not only from a hand-exported environment variable.
if(MIRAGE_ALLOW_TEST_SKIP)
    list(APPEND _mirage_env "MIRAGE_E2E_ALLOW_SKIP=1")
    message(
        STATUS
        "mirage: end-to-end suites may skip (MIRAGE_E2E_ALLOW_SKIP=1)"
    )
endif()

# Cargo features. The only optional things left are the emulator
# backends, and the default (`rocjitsu`) is what the suites run on, so a
# plain build selects nothing here. Enable another backend with
# `-DMIRAGE_CARGO_FEATURES=hotswap`.
set(MIRAGE_CARGO_FEATURES
    ""
    CACHE STRING
    "Extra cargo features (comma-separated)"
)

set(_mirage_features)
if(MIRAGE_CARGO_FEATURES)
    set(_mirage_features "--features" "${MIRAGE_CARGO_FEATURES}")
    message(STATUS "mirage: cargo features: ${MIRAGE_CARGO_FEATURES}")
endif()

# The end-to-end suites run their sessions on the default `rocjitsu`
# backend and skip, loudly, when its runtime library is absent — so they
# need no extra features beyond whatever the build already selected.
set(_mirage_test_features ${_mirage_features})

set(_mirage_profile_flag "")
if(MIRAGE_CARGO_PROFILE STREQUAL "release")
    set(_mirage_profile_flag "--release")
    set(_mirage_target_dir "${CMAKE_CURRENT_SOURCE_DIR}/target/release")
else()
    set(_mirage_target_dir
        "${CMAKE_CURRENT_SOURCE_DIR}/target/${MIRAGE_CARGO_PROFILE}"
    )
endif()

set(_mirage_bin "${_mirage_target_dir}/mirage")

add_custom_target(
    mirage_build
    ALL
    COMMAND
        ${CMAKE_COMMAND} -E env ${_mirage_env} ${MIRAGE_CARGO} build
        ${_mirage_profile_flag} ${_mirage_features}
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    BYPRODUCTS ${_mirage_bin}
    COMMENT "Building mirage with cargo (${MIRAGE_CARGO_PROFILE})"
    VERBATIM
)

# HotSwap source build (opt-in). Kept in its own module since it pulls in
# ExternalProject and a multi-stage LLVM/COMGR/ROCR build.
if(MIRAGE_BUILD_HOTSWAP)
    include(${CMAKE_CURRENT_SOURCE_DIR}/hotswap/cmake/Hotswap.cmake)
endif()

install(PROGRAMS ${_mirage_bin} DESTINATION bin RENAME mirage)

enable_testing()

# Full Rust workspace test suite: every crate's unit tests plus the
# integration suites under `tests/` (e2e, container_e2e, matrix_e2e,
# strain, the emulator benchmark) and `supervisor/tests/`. `--workspace`
# picks up each member's integration targets, so a new `tests/*.rs` file
# needs no change here. Session tests skip without a rocjitsu runtime;
# see MIRAGE_ALLOW_TEST_SKIP above for what that means for `ctest`.
add_test(
    NAME cargo_test
    COMMAND
        ${CMAKE_COMMAND} -E env ${_mirage_env} ${MIRAGE_CARGO} test
        ${_mirage_profile_flag} --workspace ${_mirage_test_features}
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
set(_mirage_clippy_cmd
    ${CMAKE_COMMAND}
    -E
    env
    ${_mirage_env}
    ${MIRAGE_CARGO}
    clippy
    ${_mirage_profile_flag}
    --workspace
    --all-targets
    --all-features
    --
    -D
    warnings
)
set(_mirage_fmt_cmd
    ${MIRAGE_CARGO}
    fmt
    --all
    --
    --check
)

if(MIRAGE_LINT_TESTS)
    add_test(
        NAME cargo_clippy
        COMMAND ${_mirage_clippy_cmd}
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    )
    set_tests_properties(cargo_clippy PROPERTIES TIMEOUT 1800)

    add_test(
        NAME cargo_fmt
        COMMAND ${_mirage_fmt_cmd}
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    )
    set_tests_properties(cargo_fmt PROPERTIES TIMEOUT 300)
endif()

# The same two checks as a target, for the common case of wanting them
# without a 30-minute test run. Deliberately not in `all`: a formatting
# nit should not stop you building the binary you are debugging.
add_custom_target(
    mirage_lint
    COMMAND ${_mirage_fmt_cmd}
    COMMAND ${_mirage_clippy_cmd}
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Checking mirage formatting and lints"
    VERBATIM
)

# rocjitsu-corpus gfx1250 regression, run through mirage + an emulator
# (HotSwap by default). Opt-in via MIRAGE_RUN_CORPUS; tests SKIP when the
# corpus, IREE tools, or emulator install are unavailable.
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/RocjitsuCorpus.cmake)
