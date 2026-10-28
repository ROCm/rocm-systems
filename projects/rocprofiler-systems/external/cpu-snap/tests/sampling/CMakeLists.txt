# Bulk: frame-pointer tests + general sampling tests merged into one executable.
# dl linked for test_stress_streaming; harmless for the others.
add_executable(sampling_tests
    test_sampler.cpp
    test_sampler_streaming.cpp
    test_kernel_drain.cpp
    test_perf_sample_decoder.cpp
    test_atfork_child_handler.cpp
    test_stress.cpp
    test_stress_streaming.cpp
    test_frame_trie.cpp
    test_temporal_diff_filter.cpp
    test_streaming_pipeline.cpp
    test_perf_ring.cpp
    test_dl_module_watcher.cpp
    test_flat_tid_map.cpp
    test_fp_copy_unwinder.cpp
    test_kernel_callchain_unwinder.cpp
    test_sampler_lifecycle.cpp
    test_backend_selection.cpp
    test_thread_comm.cpp
    test_logger.cpp
    test_clock_option.cpp
    test_sampler_plan.cpp
    test_fork_registry.cpp
    test_thread_tracker.cpp)
target_link_libraries(sampling_tests PRIVATE
    cpu_snap::sampling cpu_snap::processing cpu_snap_test_support dl
    GTest::gtest GTest::gtest_main)
target_compile_options(sampling_tests PRIVATE -Wall -Wextra -Wpedantic -O2 -g -fno-omit-frame-pointer)
gtest_discover_tests(sampling_tests PROPERTIES TIMEOUT 60)

# Signal/timer backend: -no-pie so the known-workload function addresses
# are stable against the leaf-IP range check.
# -fno-optimize-sibling-calls keeps busy_outer on the stack for signal_sampler.
add_executable(sampling_no_pie_tests
    test_signal_sampler.cpp)
target_link_libraries(sampling_no_pie_tests PRIVATE
    cpu_snap::sampling cpu_snap::processing dl
    GTest::gtest GTest::gtest_main)
target_compile_options(sampling_no_pie_tests PRIVATE
    -Wall -Wextra -Wpedantic -O2 -g
    -fno-omit-frame-pointer -fno-optimize-sibling-calls -no-pie)
target_link_options(sampling_no_pie_tests PRIVATE -no-pie)
# RUN_SERIAL: RLIMIT_SIGPENDING is counted per UID, so signals queued by tests
# running in parallel count against the limit the lifecycle tests lower.
gtest_discover_tests(sampling_no_pie_tests PROPERTIES TIMEOUT 30 RUN_SERIAL TRUE)

# Streaming self-consistency (two streaming runs of the same workload
# compared statistically): needs perfetto_writer_demo + -no-pie.
add_executable(sampling_perfetto_tests test_streaming_perfetto_equivalence.cpp)
target_link_libraries(sampling_perfetto_tests PRIVATE
    cpu_snap::sampling perfetto_writer_demo
    GTest::gtest GTest::gtest_main)
target_compile_options(sampling_perfetto_tests PRIVATE
    -Wall -Wextra -Wpedantic -O2 -g
    -fno-omit-frame-pointer -fno-optimize-sibling-calls -no-pie)
target_link_options(sampling_perfetto_tests PRIVATE -no-pie)
gtest_discover_tests(sampling_perfetto_tests PROPERTIES TIMEOUT 60)

add_custom_target(run_perfetto_equiv
    COMMAND $<TARGET_FILE:sampling_perfetto_tests> --gtest_color=yes
    DEPENDS sampling_perfetto_tests
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Running perfetto-equivalence test; pftrace outputs land in /tmp/callstack_pr10_equiv/"
    VERBATIM USES_TERMINAL)

# DWARF unwinder: -fomit-frame-pointer so DWARF is the only unwind path.
# Incompatible with the -O2 used elsewhere; built as a separate target.
add_library(dwarf_refresh_fixture SHARED fixtures/dwarf_refresh_fixture.cpp)
target_compile_options(dwarf_refresh_fixture PRIVATE
    -O0 -g -fomit-frame-pointer -fno-optimize-sibling-calls -fasynchronous-unwind-tables)
add_library(dwarf_refresh_fixture_b SHARED fixtures/dwarf_refresh_fixture.cpp)
target_compile_options(dwarf_refresh_fixture_b PRIVATE
    -O0 -g -fomit-frame-pointer -fno-optimize-sibling-calls -fasynchronous-unwind-tables)
add_executable(sampling_dwarf_tests test_dwarf_unwinder.cpp)
target_compile_definitions(sampling_dwarf_tests PRIVATE
    DWARF_REFRESH_FIXTURE_PATH="$<TARGET_FILE:dwarf_refresh_fixture>"
    DWARF_REFRESH_FIXTURE_B_PATH="$<TARGET_FILE:dwarf_refresh_fixture_b>")
add_dependencies(sampling_dwarf_tests dwarf_refresh_fixture dwarf_refresh_fixture_b)
target_link_libraries(sampling_dwarf_tests PRIVATE
    cpu_snap::sampling
    GTest::gtest GTest::gtest_main)
target_compile_options(sampling_dwarf_tests PRIVATE
    -Wall -Wextra -Wpedantic -g
    -O0 -fomit-frame-pointer -fno-optimize-sibling-calls -fasynchronous-unwind-tables)
gtest_discover_tests(sampling_dwarf_tests PROPERTIES TIMEOUT 30)

if(CPU_SNAP_BUILD_BENCH)
    add_executable(bench_streaming_smoke_test bench_streaming_smoke_test.cpp)
    target_link_libraries(bench_streaming_smoke_test PRIVATE GTest::gtest GTest::gtest_main)
    target_compile_options(bench_streaming_smoke_test PRIVATE -Wall -Wextra -Wpedantic -O2 -g)
    target_compile_definitions(bench_streaming_smoke_test PRIVATE
        SAMPLING_BENCH_STREAMING_PATH="$<TARGET_FILE:bench_streaming>")
    add_dependencies(bench_streaming_smoke_test bench_streaming)
    gtest_discover_tests(bench_streaming_smoke_test PROPERTIES TIMEOUT 60)
endif()
