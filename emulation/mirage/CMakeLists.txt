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
#                           runtime library is not discoverable.
#
# Options:
#   * `MIRAGE_ALLOW_TEST_SKIP` (OFF)  — let the end-to-end suites report
#                                       success when the `rocjitsu`
#                                       runtime is missing and every
#                                       session test therefore skipped.
#                                       Off by default: a suite that
#                                       silently proves nothing is worse
#                                       than a red one.
#   * `MIRAGE_CARGO_PROFILE`  (release)
#   * `MIRAGE_CARGO`          (auto)  — override cargo binary
#
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

# rocjitsu-corpus gfx1250 regression, run through mirage + an emulator
# (HotSwap by default). Opt-in via MIRAGE_RUN_CORPUS; tests SKIP when the
# corpus, IREE tools, or emulator install are unavailable.
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/RocjitsuCorpus.cmake)
