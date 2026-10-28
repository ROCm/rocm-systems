# Microbenchmarks. bench_decode is standalone (rebuilds perf record
# bytes in-memory, no cpu_snap link). bench_overhead + bench_streaming
# drive cpu_snap::sampling end-to-end.

if(CPU_SNAP_BUILD_SAMPLING)
    # The benches consume the header-only workload generator from
    # examples/common (cpu_snap_workload). When examples are not built
    # (e.g. a bench-only configure) that INTERFACE target is absent, so pull
    # in just that header-only directory. It compiles no example sources.
    if(NOT TARGET cpu_snap_workload)
        add_subdirectory(${CMAKE_SOURCE_DIR}/examples/common
                         ${CMAKE_BINARY_DIR}/examples/common)
    endif()

    add_executable(bench_overhead bench_overhead.cpp)
    target_link_libraries(bench_overhead PRIVATE cpu_snap::sampling cpu_snap_workload)
    target_compile_options(bench_overhead PRIVATE
        -Wall -Wextra -Wpedantic -O2 -fno-omit-frame-pointer
        -fno-optimize-sibling-calls -g -no-pie)
    target_link_options(bench_overhead PRIVATE -no-pie)

    add_executable(bench_streaming bench_streaming.cpp)
    target_link_libraries(bench_streaming PRIVATE cpu_snap::sampling cpu_snap_workload)
    target_compile_options(bench_streaming PRIVATE
        -Wall -Wextra -Wpedantic -O2 -fno-omit-frame-pointer
        -fno-optimize-sibling-calls -g -no-pie)
    target_link_options(bench_streaming PRIVATE -no-pie)
endif()

add_executable(bench_decode bench_decode.cpp)
target_compile_features(bench_decode PRIVATE cxx_std_20)
target_compile_options(bench_decode PRIVATE
    -Wall -Wextra -Wpedantic -O3 -fno-omit-frame-pointer -g)
