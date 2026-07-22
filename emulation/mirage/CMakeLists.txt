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
#   * `MIRAGE_ENABLE_DAEMON`  (OFF)   — compile in the `mirage webui`
#                                       HTTP/WS API server (cargo
#                                       feature `daemon`)
#   * `MIRAGE_ENABLE_WEBUI`   (OFF)   — compile in the full web UI:
#                                       the daemon plus the bundled
#                                       dashboard SPA (cargo feature
#                                       `webui`; requires the Node.js
#                                       toolchain). Implies the daemon.
#   * `MIRAGE_CARGO_PROFILE`  (release)
#   * `MIRAGE_CARGO`          (auto)  — override cargo binary
#
cmake_minimum_required(VERSION 3.20)
project(mirage NONE)

# Web UI is opt-in and off by default: a plain build/test never needs
# Node.js or builds the daemon/dashboard crates.
option(MIRAGE_ENABLE_DAEMON "Build the mirage webui HTTP/WS API server" OFF)
option(MIRAGE_ENABLE_WEBUI "Build the full web UI (daemon + dashboard SPA; needs Node.js)" OFF)
# HotSwap is a full LLVM + COMGR + ROCR source build, so it is opt-in. When ON,
# cmake/Hotswap.cmake builds and stages the HotSwap artifacts mirage discovers.
option(MIRAGE_BUILD_HOTSWAP "Build HotSwap (COMGR + patched ROCR + intercept) from source" OFF)
set(MIRAGE_CARGO_PROFILE "release" CACHE STRING "Cargo build profile")
set(MIRAGE_CARGO "" CACHE FILEPATH "Path to cargo binary (auto-detect if empty)")

if(NOT MIRAGE_CARGO)
    find_program(MIRAGE_CARGO_FOUND cargo REQUIRED)
    set(MIRAGE_CARGO "${MIRAGE_CARGO_FOUND}")
endif()
message(STATUS "mirage: using cargo at ${MIRAGE_CARGO}")

set(_mirage_env)
# Select the cargo feature flags for the (optional) web UI. `webui`
# pulls in the daemon plus the bundled SPA; `daemon` is just the API
# server. Both are off by default.
set(_mirage_features)
if(MIRAGE_ENABLE_WEBUI)
    list(APPEND _mirage_features "--features" "webui")
    message(STATUS "mirage: web UI enabled (cargo --features webui)")
elseif(MIRAGE_ENABLE_DAEMON)
    list(APPEND _mirage_features "--features" "daemon")
    message(STATUS "mirage: daemon enabled (cargo --features daemon)")
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
            ${MIRAGE_CARGO} build ${_mirage_profile_flag} ${_mirage_features}
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

install(PROGRAMS ${_mirage_bin}
        DESTINATION bin
        RENAME mirage)

enable_testing()

# Full Rust workspace test suite. Picks up: core unit tests, daemon
# e2e, lifecycle e2e, and the env-gated ml_scenarios tests (which
# auto-skip when their prerequisites are missing).
add_test(NAME cargo_test
    COMMAND ${CMAKE_COMMAND} -E env ${_mirage_env}
            ${MIRAGE_CARGO} test ${_mirage_profile_flag} --workspace ${_mirage_features}
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
set_tests_properties(cargo_test PROPERTIES TIMEOUT 1800)

# rocjitsu-corpus gfx1250 regression, run through mirage + an emulator
# (HotSwap by default). Opt-in via MIRAGE_RUN_CORPUS; tests SKIP when the
# corpus, IREE tools, or emulator install are unavailable.
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/RocjitsuCorpus.cmake)
