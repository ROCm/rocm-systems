# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# rocprof-sys-sampling-library: new thread sampling subsystem.
# code-writer owns this CMakeLists.txt for production sources.
# code-tester owns tests/CMakeLists.txt.
#
# Layering rule: this library has ZERO dependencies on main rocprof-sys library
# symbols (perf.hpp, tracing.hpp, trace_cache, timemory). Those deps live in
# library/sampling/default_policies.hpp and library/services_accessor.cpp.

cmake_minimum_required(VERSION 3.21)

# ── Main object library ───────────────────────────────────────────────────────
add_library(rocprof-sys-sampling-library OBJECT)

target_include_directories(
    rocprof-sys-sampling-library
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        # Expose the library/ parent so tests can #include "sampling/src/..." headers.
        # sampling_service.hpp itself uses "sampling/src/sampling_service_impl.hpp",
        # which resolves from library/ as sampling/src/ (not from sampling/ root).
        ${CMAKE_CURRENT_SOURCE_DIR}/..
)

# ── Platform-neutral sources ──────────────────────────────────────────────────
# Sources are added incrementally as each class is implemented (TDD).
# Uncomment each line when the corresponding .cpp file exists.
# target_sources(rocprof-sys-sampling-library PRIVATE
#     src/sampling_service.cpp
#     src/thread_sampler_state.cpp
#     src/thread_sampler_state_registry.cpp
#     src/sampler_allocator_pool.cpp
#     src/sample_parser.cpp)

# ── Linux-only impls (code-writer fills in) ───────────────────────────────────
if(LINUX)
    target_compile_definitions(
        rocprof-sys-sampling-library
        PRIVATE ROCPROFSYS_LINUX_SAMPLING=1
    )

    # target_sources(rocprof-sys-sampling-library PRIVATE
    #     src/linux/libunwind_unwinder.cpp
    #     src/linux/real_timer_trigger.cpp
    #     src/linux/real_signal_dispatcher.cpp
    #     src/linux/steady_clock.cpp)
    # Uncomment lines above as corresponding .cpp files are implemented.
endif()

# ── Link dependencies ──────────────────────────────────────────────────────────
# NO main-lib deps here. Heavy policy deps live in library/services_accessor.cpp.
target_link_libraries(
    rocprof-sys-sampling-library
    PUBLIC rocprofiler-systems-logger rocprofiler-systems-interface-library
)

if(LINUX)
    # libunwind is a hard requirement on Linux — the unwinder policy seam,
    # the in-handler unwind in services_accessor.cpp, and the symbol-resolution
    # cascade in real_production_hooks all depend on it. Fail the configure step
    # immediately if libunwind is missing rather than emitting a confusing link
    # error deep in the sampling object library.
    if(NOT TARGET LibUnwind::LibUnwind)
        message(
            FATAL_ERROR
            "rocprof-sys-sampling-library requires libunwind on Linux but "
            "LibUnwind::LibUnwind was not found. Install libunwind-dev (or set "
            "LibUnwind_ROOT) and reconfigure, or build with "
            "ROCPROFSYS_BUILD_LIBUNWIND=ON to use the bundled submodule."
        )
    endif()
    target_link_libraries(rocprof-sys-sampling-library PRIVATE LibUnwind::LibUnwind)
endif()

if(TARGET LibDW::LibDW)
    target_link_libraries(rocprof-sys-sampling-library PRIVATE LibDW::LibDW)
endif()
# PAPI::PAPI -- already in rocprofiler-systems-interface-library

# ── Inject objects into the main library ─────────────────────────────────────
# The sampling library objects are injected into the main library's object set.
# services_accessor.cpp (with heavy policy deps) is added directly in library/CMakeLists.txt.
target_sources(
    rocprofiler-systems-object-library
    PRIVATE $<TARGET_OBJECTS:rocprof-sys-sampling-library>
)

# ── NFR-PORT-1: public-header OS-include grep gate ───────────────────────────
add_test(
    NAME sampling-public-headers-platform-neutral
    COMMAND
        ${CMAKE_COMMAND} -DSAMPLING_INCLUDE=${CMAKE_CURRENT_SOURCE_DIR}/include -P
        ${CMAKE_SOURCE_DIR}/cmake/scripts/check_sampling_public_headers.cmake
)

# ── NFR-PORT-3: platform_guard compile-fail test ─────────────────────────────
# Runs as cmake -P (script mode). The script uses execute_process to drive a
# sub-cmake build — try_compile is not available in script mode.
add_test(
    NAME sampling-platform-guard-non-linux-compile-fail
    COMMAND
        ${CMAKE_COMMAND} -DSAMPLING_INCLUDE=${CMAKE_CURRENT_SOURCE_DIR}/include
        -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER} -DCMAKE_COMMAND=${CMAKE_COMMAND} -P
        ${CMAKE_SOURCE_DIR}/cmake/scripts/check_sampling_platform_guard.cmake
)

# ── Tests ─────────────────────────────────────────────────────────────────────
if(BUILD_TESTING)
    add_subdirectory(tests)
endif()
