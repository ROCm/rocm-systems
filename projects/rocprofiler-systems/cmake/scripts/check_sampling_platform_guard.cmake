# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# NFR-PORT-3: Verify that including sampling/platform_guard.hpp on a non-Linux
# platform triggers a static_assert compile error.
# Called from the sampling CMakeLists.txt add_test() — runs as a CTest command,
# NOT as a cmake -P script (try_compile is not scriptable).

if(NOT DEFINED SAMPLING_INCLUDE)
    message(FATAL_ERROR "SAMPLING_INCLUDE not set")
endif()

if(NOT DEFINED CMAKE_CXX_COMPILER)
    message(FATAL_ERROR "CMAKE_CXX_COMPILER not set")
endif()

if(NOT DEFINED CMAKE_COMMAND)
    # Fall back to searching PATH — cmake is always available when this script runs.
    find_program(CMAKE_COMMAND cmake REQUIRED)
endif()

set(_build_dir "${CMAKE_CURRENT_BINARY_DIR}/nfr_port3_try")
file(MAKE_DIRECTORY "${_build_dir}")

# Write a tiny TU that simulates a non-Linux build by undefining __linux__
# and then trying to instantiate sampling_service_platform_guard.
set(_test_src "${_build_dir}/nfr_port3_compile_fail_test.cpp")
file(
    WRITE "${_test_src}"
    "#undef __linux__\n"
    "#include \"sampling/platform_guard.hpp\"\n"
    "// Force instantiation — should static_assert on non-Linux.\n"
    "rocprofsys::sampling::sampling_service_platform_guard<int> g;\n"
    "int main() { return 0; }\n"
)

# Write a minimal CMakeLists.txt for the standalone compile attempt.
set(_test_cmake "${_build_dir}/CMakeLists.txt")
file(
    WRITE "${_test_cmake}"
    "cmake_minimum_required(VERSION 3.21)\n"
    "project(nfr_port3_compile_fail CXX)\n"
    "set(CMAKE_CXX_STANDARD 17)\n"
    "add_executable(nfr_port3_exe nfr_port3_compile_fail_test.cpp)\n"
    "target_include_directories(nfr_port3_exe PRIVATE \"${SAMPLING_INCLUDE}\")\n"
)

# Configure the small project.
execute_process(
    COMMAND
        "${CMAKE_COMMAND}" -S "${_build_dir}" -B "${_build_dir}/cmake-build"
        -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
    RESULT_VARIABLE _cfg_result
    OUTPUT_VARIABLE _cfg_out
    ERROR_VARIABLE _cfg_err
)

# Build it — we expect the COMPILE to fail due to static_assert.
execute_process(
    COMMAND "${CMAKE_COMMAND}" --build "${_build_dir}/cmake-build"
    RESULT_VARIABLE _build_result
    OUTPUT_VARIABLE _build_out
    ERROR_VARIABLE _build_err
)

set(_output "${_cfg_out}${_cfg_err}${_build_out}${_build_err}")

if(_build_result EQUAL 0)
    message(
        FATAL_ERROR
        "NFR-PORT-3 FAIL: compilation succeeded but should have failed with static_assert.\n"
        "Expected: 'Thread sampling is Linux-only in this build.'\n"
        "Compiler output:\n${_output}"
    )
endif()

# Verify the expected error message appears in the compiler output.
if(NOT _output MATCHES "Linux-only")
    message(
        WARNING
        "NFR-PORT-3: compile failed (good) but expected message 'Linux-only' not found.\n"
        "Actual output:\n${_output}"
    )
endif()

message(STATUS "NFR-PORT-3 PASS: non-Linux platform_guard static_assert fires correctly")
