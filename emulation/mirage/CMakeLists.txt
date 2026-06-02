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
#                           `cargo test --release --workspace`
#
# Options:
#   * `MIRAGE_BUILD_DASHBOARD` (ON)   — set OFF (or define the env
#                                       var MIRAGE_DASHBOARD_SKIP_NPM_CI=1)
#                                       to skip the SPA build
#   * `MIRAGE_CARGO_PROFILE`  (release)
#   * `MIRAGE_CARGO`          (auto)  — override cargo binary
#
cmake_minimum_required(VERSION 3.20)
project(mirage NONE)

option(MIRAGE_BUILD_DASHBOARD "Build the web dashboard SPA" ON)
set(MIRAGE_CARGO_PROFILE "release" CACHE STRING "Cargo build profile")
set(MIRAGE_CARGO "" CACHE FILEPATH "Path to cargo binary (auto-detect if empty)")

if(NOT MIRAGE_CARGO)
    find_program(MIRAGE_CARGO_FOUND cargo REQUIRED)
    set(MIRAGE_CARGO "${MIRAGE_CARGO_FOUND}")
endif()
message(STATUS "mirage: using cargo at ${MIRAGE_CARGO}")

set(_mirage_env)
if(NOT MIRAGE_BUILD_DASHBOARD)
    list(APPEND _mirage_env "MIRAGE_DASHBOARD_SKIP_NPM_CI=1")
    message(STATUS "mirage: dashboard build skipped (MIRAGE_BUILD_DASHBOARD=OFF)")
endif()

set(_mirage_profile_flag "")
if(MIRAGE_CARGO_PROFILE STREQUAL "release")
    set(_mirage_profile_flag "--release")
    set(_mirage_target_dir "${CMAKE_CURRENT_SOURCE_DIR}/target/release")
else()
    set(_mirage_target_dir "${CMAKE_CURRENT_SOURCE_DIR}/target/${MIRAGE_CARGO_PROFILE}")
endif()

set(_mirage_bin "${_mirage_target_dir}/mirage")

add_custom_target(mirage_build ALL
    COMMAND ${CMAKE_COMMAND} -E env ${_mirage_env}
            ${MIRAGE_CARGO} build ${_mirage_profile_flag}
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    BYPRODUCTS ${_mirage_bin}
    COMMENT "Building mirage with cargo (${MIRAGE_CARGO_PROFILE})"
    VERBATIM
)

install(PROGRAMS ${_mirage_bin}
        DESTINATION bin
        RENAME mirage)

enable_testing()

# Full Rust workspace test suite. Picks up: core unit tests, daemon
# e2e, lifecycle e2e, and the env-gated ml_scenarios tests (which
# auto-skip when their prerequisites are missing).
add_test(NAME cargo_test
    COMMAND ${CMAKE_COMMAND} -E env ${_mirage_env}
            ${MIRAGE_CARGO} test ${_mirage_profile_flag} --workspace
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
set_tests_properties(cargo_test PROPERTIES TIMEOUT 1800)
