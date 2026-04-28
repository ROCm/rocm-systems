# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Tests for the new sampling subsystem.
# Requires BUILD_TESTING=ON.

cmake_minimum_required(VERSION 3.21)

# ── Test doubles INTERFACE library ────────────────────────────────────────────
# Exposes tests/ so consumers can #include "doubles/..." with subdirectory prefix.
add_library(rocprof-sys-sampling-test-doubles INTERFACE)
target_include_directories(
    rocprof-sys-sampling-test-doubles
    INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(
    rocprof-sys-sampling-test-doubles
    INTERFACE rocprof-sys-sampling-library
)

# ── Sampling OBJECT library (unit + integration sources) ──────────────────────
# Compiled as an OBJECT library so sources can be folded into the main
# rocprof-sys-unit-tests binary without a separate executable.
set(_sampling_test_sources
    # Test-infrastructure stubs (config functions unavailable outside the full library)
    unit/config_stubs.cpp
    unit/sampling_service_setup_test.cpp
    unit/sampling_service_shutdown_test.cpp
    unit/sampling_service_pause_resume_test.cpp
    unit/sample_ring_buffer_test.cpp
    unit/sample_parser_test.cpp
    unit/sampling_duration_controller_test.cpp
    unit/fatal_error_policy_test.cpp
    # unit/native_report_writer_test.cpp — retired: tested old CSV stub; replaced by
    # integration/native_report_writer_tsv_test.cpp for the new TSV interface (Phase H).
    unit/header_policy_test.cpp
    unit/pause_interval_registry_test.cpp
    unit/signal_set_test.cpp
    unit/signal_mask_guard_test.cpp
    unit/thread_sampler_state_test.cpp
    # Integration sources (exercise full pipeline via test_sampling_policies)
    integration/full_pipeline_test.cpp
    # Signal handler invariants: re-entry guard, ring-full drop, in-flight counter (Phase G)
    unit/sampling_signal_handler_test.cpp
    # TSV writer correctness: header, pct_self, flat_scope, multi-thread (Phase H TDD red)
    integration/native_report_writer_tsv_test.cpp
    # Real-filesystem integration test: verify set_streams() + ofstream actually creates files
    # on disk (catches silent open-failure bugs in open_report_writer_streams).
    integration/native_report_writer_realfs_test.cpp
    # TDD red tests for R-1: cpu_clock always empty (signal handler + parser + writer fix).
    unit/metrics_population_test.cpp
    # TDD red tests for AC-2: realtime+cputime trigger mutual-exclusion (else-if → if fix).
    unit/trigger_mutual_exclusion_test.cpp
    # TDD tests for trace_cache_offload_adapter (EmitterPolicy production type, Task #31).
    unit/trace_cache_offload_adapter_test.cpp
    # TDD tests for emit_resolved_to_trace_cache hook + erase() (Variant 2, Task #30).
    unit/emit_resolved_hook_test.cpp
    # TDD tests for ROCPROFSYS_SAMPLING_TRACE_LEGACY config accessor (Task #26).
    unit/sampling_trace_legacy_test.cpp
    # NFR-T-7: golden-file byte-exact diff for native_report_writer TSV output (Task #14).
    unit/native_report_writer_golden_test.cpp
)

# Linux-only smoke tests for real OS-backed implementations.
# Gated at CMake level — no runtime GTEST_SKIP inside the file.
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    list(APPEND _sampling_test_sources unit/linux_impl_smoke_test.cpp)
    # Fork integration tests: AC-17 (postfork_parent_reinit / postfork_child_cleanup)
    # and NFR-TS-5 (signals blocked before state torn down in child).
    # Uses real fork() — Linux-only.
    list(APPEND _sampling_test_sources integration/fork_integration_test.cpp)
    # NOTE: real_overflow_trigger_test.cpp is NOT in the OBJECT library.
    # It links perf::perf_event symbols from rocprofiler-systems-object-library
    # which the unit test binary must not link (TSan deadlock on atexit).
    # It lives in its own standalone binary below.
endif()

# ── Async-signal-safety regression test ──────────────────────────────────────
# Grep the signal handler TU for forbidden async-signal-unsafe symbols.
# Fails if any banned token appears outside an explicit ASYNC-SAFE-EXEMPT annotation.
# This is a static-analysis regression net — not a runtime test.
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set(_handler_tu "${CMAKE_CURRENT_SOURCE_DIR}/../sampling_production_policies.hpp")

    # Tokens banned inside the signal handler (not async-signal-safe).
    # Each must be on a line that does NOT also contain ASYNC-SAFE-EXEMPT.
    set(_banned_patterns
        "malloc("
        " new "
        "std::mutex"
        "LOG_INFO"
        "LOG_DEBUG"
        "LOG_WARNING"
        "LOG_CRITICAL"
        "std::string("
        "printf("
        "fmt::"
        "spdlog::"
    )

    foreach(_token ${_banned_patterns})
        # Escape token for grep -F (fixed string, not regex).
        add_test(
            NAME "async_signal_safety_check_${_token}"
            COMMAND
                ${CMAKE_COMMAND} -E env bash -c
                "! grep -n -F '${_token}' '${_handler_tu}' | grep -v 'ASYNC-SAFE-EXEMPT'"
        )
        set_tests_properties(
            "async_signal_safety_check_${_token}"
            PROPERTIES LABELS "sampling;static;async-signal-safety" TIMEOUT 10
        )
    endforeach()
endif()

add_library(rocprof-sys-sampling-tests-objs OBJECT ${_sampling_test_sources})

# PUBLIC visibility so whoever links this OBJECT lib gets includes + link deps
# automatically — the central CMakeLists needs only one glue line.
target_compile_definitions(
    rocprof-sys-sampling-tests-objs
    PUBLIC
        ROCPROFSYS_SAMPLING_INCLUDE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../include/sampling"
)

target_link_libraries(
    rocprof-sys-sampling-tests-objs
    PUBLIC
        rocprof-sys-sampling-library
        rocprof-sys-sampling-test-doubles
        rocprofiler-systems-googletest-library
        rocprofiler-systems-interface-library
)

target_include_directories(
    rocprof-sys-sampling-tests-objs
    PUBLIC ${PROJECT_SOURCE_DIR}/source/lib ${PROJECT_SOURCE_DIR}/source/lib/rocprof-sys
)

# ── R-3 integration test: real config per-TID signal filtering ───────────────
# This test calls the REAL get_sampling_signals() from core/config.cpp, so it
# must be a separate executable that does NOT include config_stubs.cpp.
# It links rocprofiler-systems-core-library directly (available in the build tree
# because this CMakeLists is included from a subdirectory of the project root).
if(TARGET rocprofiler-systems-core-library)
    add_executable(
        rocprof-sys-sampling-config-integration-tests
        integration/get_sampling_signals_integration_test.cpp
    )
    target_link_libraries(
        rocprof-sys-sampling-config-integration-tests
        PRIVATE
            rocprof-sys-sampling-library
            rocprofiler-systems-core-library
            rocprofiler-systems::rocprofiler-systems-interface-library
            GTest::gtest
    )
    target_include_directories(
        rocprof-sys-sampling-config-integration-tests
        PRIVATE
            ${PROJECT_SOURCE_DIR}/source/lib
            ${PROJECT_SOURCE_DIR}/source/lib/rocprof-sys
    )
    add_test(
        NAME rocprof-sys-sampling-config-integration-tests
        COMMAND rocprof-sys-sampling-config-integration-tests
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
    set_tests_properties(
        rocprof-sys-sampling-config-integration-tests
        PROPERTIES LABELS "sampling;integration;config;r3" TIMEOUT 60
    )
endif()

# ── I-15 production hooks integration test (Task #17) ────────────────────────
# Tests the observable outcome of shutdown_production_wiring (TLS cleared).
# Links librocprofiler-systems-static (contains thread_info, tracing, perf, TLS state)
# and real config, so it must NOT include config_stubs.cpp.
# Uses configure_settings(false) to avoid recursive_init_error.
if(TARGET rocprofiler-systems-core-library AND CMAKE_SYSTEM_NAME STREQUAL "Linux")
    add_executable(
        rocprof-sys-sampling-production-hooks-tests
        integration/sampling_service_production_hooks_test.cpp
    )
    target_link_libraries(
        rocprof-sys-sampling-production-hooks-tests
        PRIVATE
            rocprof-sys-sampling-library
            rocprofiler-systems::librocprofiler-systems-static
            rocprofiler-systems::rocprofiler-systems-interface-library
            GTest::gtest
    )
    target_include_directories(
        rocprof-sys-sampling-production-hooks-tests
        PRIVATE
            ${PROJECT_SOURCE_DIR}/source/lib
            ${PROJECT_SOURCE_DIR}/source/lib/rocprof-sys
    )
    add_test(
        NAME rocprof-sys-sampling-production-hooks-tests
        COMMAND rocprof-sys-sampling-production-hooks-tests
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
    set_tests_properties(
        rocprof-sys-sampling-production-hooks-tests
        PROPERTIES LABELS "sampling;integration;production-hooks;i15" TIMEOUT 60
    )
endif()

# ── I-15 real_perfetto_sink integration test (Task #17) ──────────────────────
# Tests the empty-sample guard and perfetto-disabled early-return path.
# Requires rocprofiler-systems-binary (real_perfetto_sink, perfetto headers).
if(TARGET rocprofiler-systems-core-library AND CMAKE_SYSTEM_NAME STREQUAL "Linux")
    add_executable(
        rocprof-sys-sampling-perfetto-sink-tests
        integration/real_perfetto_sink_test.cpp
    )
    target_link_libraries(
        rocprof-sys-sampling-perfetto-sink-tests
        PRIVATE
            rocprof-sys-sampling-library
            rocprofiler-systems::librocprofiler-systems-static
            rocprofiler-systems::rocprofiler-systems-interface-library
            GTest::gtest
    )
    target_include_directories(
        rocprof-sys-sampling-perfetto-sink-tests
        PRIVATE
            ${PROJECT_SOURCE_DIR}/source/lib
            ${PROJECT_SOURCE_DIR}/source/lib/rocprof-sys
    )
    add_test(
        NAME rocprof-sys-sampling-perfetto-sink-tests
        COMMAND rocprof-sys-sampling-perfetto-sink-tests
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
    set_tests_properties(
        rocprof-sys-sampling-perfetto-sink-tests
        PROPERTIES LABELS "sampling;integration;perfetto-sink;i15" TIMEOUT 60
    )
endif()

# ── I-15 real_overflow_trigger smoke test (Task #17) ─────────────────────────
# Separate binary: real_overflow_trigger.hpp instantiates perf::perf_event inline
# methods (start/stop/close/is_open/~perf_event) that are defined in library/perf.cpp
# inside rocprofiler-systems-object-library.  The main unit test binary must NOT link
# that library (TSan deadlock on atexit), so this lives in its own binary.
if(TARGET rocprofiler-systems-core-library AND CMAKE_SYSTEM_NAME STREQUAL "Linux")
    add_executable(
        rocprof-sys-sampling-overflow-trigger-tests
        unit/real_overflow_trigger_test.cpp
    )
    target_link_libraries(
        rocprof-sys-sampling-overflow-trigger-tests
        PRIVATE
            rocprof-sys-sampling-library
            rocprofiler-systems::librocprofiler-systems-static
            rocprofiler-systems::rocprofiler-systems-interface-library
            GTest::gtest
            GTest::gtest_main
    )
    target_include_directories(
        rocprof-sys-sampling-overflow-trigger-tests
        PRIVATE
            ${PROJECT_SOURCE_DIR}/source/lib
            ${PROJECT_SOURCE_DIR}/source/lib/rocprof-sys
    )
    add_test(
        NAME rocprof-sys-sampling-overflow-trigger-tests
        COMMAND rocprof-sys-sampling-overflow-trigger-tests
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
    set_tests_properties(
        rocprof-sys-sampling-overflow-trigger-tests
        PROPERTIES LABELS "sampling;unit;overflow-trigger;i15" TIMEOUT 30
    )
endif()

# ── Stress test (TSAN, Linux only) ───────────────────────────────────────────
# Kept as a separate binary: TSAN link flags are incompatible with the main binary.
if(ROCPROFSYS_BUILD_SAMPLING_TSAN AND CMAKE_SYSTEM_NAME STREQUAL "Linux")
    add_executable(
        rocprof-sys-sampling-stress-tests
        stress/signal_handler_stress_test.cpp
        unit/config_stubs.cpp
    )
    target_link_libraries(
        rocprof-sys-sampling-stress-tests
        PRIVATE
            rocprof-sys-sampling-library
            rocprof-sys-sampling-test-doubles
            GTest::gtest
            GTest::gtest_main
    )
    target_compile_options(rocprof-sys-sampling-stress-tests PRIVATE -fsanitize=thread)
    target_link_options(rocprof-sys-sampling-stress-tests PRIVATE -fsanitize=thread)

    add_test(
        NAME rocprof-sys-sampling-stress-tests
        COMMAND rocprof-sys-sampling-stress-tests
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )

    set_tests_properties(
        rocprof-sys-sampling-stress-tests
        PROPERTIES LABELS "sampling;stress;tsan" TIMEOUT 120
    )
endif()

# ── Benchmark (no TSAN, separate runner) ──────────────────────────────────────
if(ROCPROFSYS_BUILD_SAMPLING_BENCH)
    find_package(benchmark REQUIRED)

    add_executable(rocprof-sys-sampling-bench bench/bench_signal_path.cpp)
    target_link_libraries(
        rocprof-sys-sampling-bench
        PRIVATE
            rocprof-sys-sampling-library
            benchmark::benchmark
            benchmark::benchmark_main
    )

    add_test(
        NAME rocprof-sys-sampling-bench
        COMMAND
            rocprof-sys-sampling-bench --benchmark_format=json
            --benchmark_out=${CMAKE_CURRENT_BINARY_DIR}/bench-results.json
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )

    set_tests_properties(
        rocprof-sys-sampling-bench
        PROPERTIES LABELS "sampling;bench" TIMEOUT 120
    )
endif()
