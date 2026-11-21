# remapper — top-level CMake wrapper around cargo.
#
# The real build is done by cargo; this lets remapper be configured, installed and tested with the same
# `cmake -S . -B build && cmake --build build && ctest` recipe as the rest of the monorepo.
#
# Targets:
#   * `remapper_build` (all) — `cargo build`, producing `libremapper.so`
#   * `install`              — `lib/libremapper.so`, `include/remapper/remapper.h`, and the ATT merge scripts
#                              under `libexec/remapper`
#   * `ctest`                — `cargo test --workspace`, plus `cargo clippy` and `cargo fmt --check`
#   * `remapper_lint`        — only the clippy and rustfmt checks; not part of `all`
#
# Options:
#   * `REMAPPER_LINT_TESTS`    (ON)      — register the clippy and rustfmt checks as ctests
#   * `REMAPPER_CARGO_PROFILE` (release)
#   * `REMAPPER_CARGO`         (auto)    — override the cargo binary
#   * `ROCM_PATH`              (/opt/rocm) — where the HSA runtime and the ROCm LLVM tools are found
#
cmake_minimum_required(VERSION 3.20)
project(remapper NONE)

include(GNUInstallDirs)

option(
    REMAPPER_LINT_TESTS
    "Register `cargo clippy` and `cargo fmt --check` as ctest cases"
    ON
)
set(REMAPPER_CARGO_PROFILE "release" CACHE STRING "Cargo build profile")
set(REMAPPER_CARGO
    ""
    CACHE FILEPATH
    "Path to cargo binary (auto-detect if empty)"
)
set(ROCM_PATH "/opt/rocm" CACHE PATH "ROCm install prefix")

if(NOT REMAPPER_CARGO)
    find_program(REMAPPER_CARGO_FOUND cargo REQUIRED)
    set(REMAPPER_CARGO "${REMAPPER_CARGO_FOUND}")
endif()
message(STATUS "remapper: using cargo at ${REMAPPER_CARGO}")

set(_remapper_target_root "${CMAKE_CURRENT_BINARY_DIR}/cargo-target")
set(_remapper_env
    "ROCM_PATH=${ROCM_PATH}"
    "CARGO_TARGET_DIR=${_remapper_target_root}"
)

if(REMAPPER_CARGO_PROFILE STREQUAL "release")
    set(_remapper_profile_flag "--release")
elseif(REMAPPER_CARGO_PROFILE STREQUAL "dev")
    set(_remapper_profile_flag "")
else()
    set(_remapper_profile_flag "--profile" "${REMAPPER_CARGO_PROFILE}")
endif()

if(REMAPPER_CARGO_PROFILE STREQUAL "dev")
    set(_remapper_target_dir "${_remapper_target_root}/debug")
else()
    set(_remapper_target_dir
        "${_remapper_target_root}/${REMAPPER_CARGO_PROFILE}"
    )
endif()

set(_remapper_lib "${_remapper_target_dir}/libremapper.so")

add_custom_target(
    remapper_build
    ALL
    COMMAND
        ${CMAKE_COMMAND} -E env ${_remapper_env} ${REMAPPER_CARGO} build
        ${_remapper_profile_flag} --locked -p remapper-ffi
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    BYPRODUCTS ${_remapper_lib}
    COMMENT "Building remapper with cargo (${REMAPPER_CARGO_PROFILE})"
    VERBATIM
)

install(FILES ${_remapper_lib} DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(
    FILES ${CMAKE_CURRENT_SOURCE_DIR}/include/remapper.h
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/remapper
)
install(
    DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/scripts/att_merge
    DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/remapper
    PATTERN "__pycache__" EXCLUDE
)

enable_testing()

add_test(
    NAME remapper_cargo_test
    COMMAND
        ${CMAKE_COMMAND} -E env ${_remapper_env} ${REMAPPER_CARGO} test
        ${_remapper_profile_flag} --locked --workspace
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
set_tests_properties(remapper_cargo_test PROPERTIES TIMEOUT 900)

set(_remapper_clippy_cmd
    ${CMAKE_COMMAND}
    -E
    env
    ${_remapper_env}
    ${REMAPPER_CARGO}
    clippy
    ${_remapper_profile_flag}
    --locked
    --workspace
    --all-targets
    --
    -D
    warnings
)
set(_remapper_fmt_cmd
    ${REMAPPER_CARGO}
    fmt
    --all
    --
    --check
)

if(REMAPPER_LINT_TESTS)
    add_test(
        NAME remapper_cargo_clippy
        COMMAND ${_remapper_clippy_cmd}
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    )
    set_tests_properties(remapper_cargo_clippy PROPERTIES TIMEOUT 900)

    add_test(
        NAME remapper_cargo_fmt
        COMMAND ${_remapper_fmt_cmd}
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    )
    set_tests_properties(remapper_cargo_fmt PROPERTIES TIMEOUT 300)
endif()

find_package(Python3 COMPONENTS Interpreter)
if(Python3_FOUND)
    execute_process(
        COMMAND ${Python3_EXECUTABLE} -c "import pytest, numpy"
        RESULT_VARIABLE _remapper_pytest
        OUTPUT_QUIET
        ERROR_QUIET
    )
    if(_remapper_pytest EQUAL 0)
        add_test(
            NAME remapper_merge_scripts
            COMMAND
                ${Python3_EXECUTABLE} -m pytest -q -p no:cacheprovider
                scripts/tests
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        )
    else()
        message(
            STATUS
            "remapper: pytest or numpy not found; skipping the merge script tests"
        )
    endif()
endif()

add_custom_target(
    remapper_lint
    COMMAND ${_remapper_fmt_cmd}
    COMMAND ${_remapper_clippy_cmd}
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Checking remapper formatting and lints"
    VERBATIM
)
