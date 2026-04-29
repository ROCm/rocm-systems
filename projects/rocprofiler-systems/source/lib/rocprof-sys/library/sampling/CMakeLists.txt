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
# Curated INTERFACE bundle of project sub-libraries that the sampling lib's
# public surface (sampling/include/sampling/* + sampling/src/sampling_service_impl.hpp)
# transitively requires. Replaces the previous PUBLIC link to the umbrella
# rocprofiler-systems-interface-library — that pulled 9 unused sub-INTERFACEs
# (mpi, ucx, libva, bfd, elfutils-as-include, sanitizer, lto, static-libgcc,
# static-libstdcxx) onto every consumer of the sampling lib (test object lib,
# integration test executables, the rocprof-sys object lib).
#
# Each entry below is the ONLY thing in the curated bundle because it is reached
# either by a public sampling header or by sampling_service_impl.hpp (the
# folded production_hooks lifecycle).
add_library(rocprof-sys-sampling-deps INTERFACE)
target_link_libraries(
    rocprof-sys-sampling-deps
    INTERFACE
        # -I source/lib + source/lib/rocprof-sys, so #include "core/..." and
        # #include "library/thread_info.hpp" resolve in consumer TUs.
        rocprofiler-systems-headers
        # -lpthread. sampling owns thread_sampler_state per tid + uses
        # std::mutex/std::lock_guard in the service.
        rocprofiler-systems-threading
        # Project compile flags (warnings, -std=c++17). Required so consumer TUs
        # match the sampling lib's ABI/flag set.
        rocprofiler-systems-compile-options
        rocprofiler-systems-compile-definitions
        # Project common/ macros + ROCPROFSYS_MAX_THREADS. core/common.hpp
        # (pulled by sampling_service_impl.hpp via the threading namespace
        # alias) needs the configure-time generated common/defines.h.
        rocprofiler-systems-common-library
        # spdlog + the bundled fmt headers. Used by:
        #   • logger/debug.hpp (LOG_DEBUG / LOG_INFO macros across the lib)
        #   • real_fatal_error_policy (fmt::format for fatal messages)
        #   • stack_frame_json (fmt::format for line info)
        rocprofiler-systems-logger
        # Perfetto static archive + headers. Reached via cache_manager.hpp ->
        # buffer_storage which stores backtrace_region_sample / pmc_event_with_sample
        # in the perfetto stream. real_perfetto_sink directly emits track events.
        rocprofiler-systems-perfetto
        # SQLite headers. Reached via cache_manager.hpp -> rocpd writer; the
        # sampling lifecycle's do_emit_resolved pushes samples through trace_cache
        # which lands in the rocpd .db.
        rocprofiler-systems-sqlite3
        # nlohmann/json header-only. Used by stack_frame_json.hpp to emit the
        # per-frame call_stack/line_info JSON blobs that store_timer/store_overflow
        # attach to backtrace_region_sample records.
        rocprofiler-systems-json
        # timemory headers + libtimemory-* archives. Reached via:
        #   • core/config.hpp     — settings registry, get_sampling_*_signal/freq/...
        #   • core/state.hpp      — ThreadState enum + get_thread_state()
        #   • core/common.hpp     — `namespace threading = tim::threading;`
        #                            (gives threading::get_sys_tid portably)
        #   • library/thread_info.hpp — thread_info::get + index_data::system_value
        #   • library/pmc/sampler.hpp — pmc::postfork_parent_reinit / _child_cleanup
        # The sampling lifecycle calls these from check_thread_guards,
        # do_setup_wiring, do_postfork_*.
        rocprofiler-systems-timemory
        # ROCm SDK headers. Pulled for rocprofiler-systems/categories.h, which
        # supplies ROCPROFSYS_CATEGORY_TIMER_SAMPLING / OVERFLOW_SAMPLING /
        # THREAD_CPU_TIME / etc. used by real_trace_cache_sink.
        rocprofiler-systems-rocm
)

# Sampling lib's PUBLIC surface = the curated bundle. PRIVATE link to the
# umbrella interface library covers the remaining build-time options the
# sampling lib's own .cpp files inherit (sanitizer/LTO/static-libgcc flags
# when they are explicitly enabled at configure time).
target_link_libraries(
    rocprof-sys-sampling-library
    PUBLIC rocprof-sys-sampling-deps
    PRIVATE rocprofiler-systems-interface-library
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
