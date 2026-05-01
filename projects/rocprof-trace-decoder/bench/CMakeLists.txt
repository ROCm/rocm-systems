cmake_minimum_required(VERSION 3.16.0)

# att-bench: standalone single-thread decode benchmark for .att files.
# Built only when -DBUILD_BENCH=ON. Links the static archive so it can
# call internal helpers from source/trace_parser.{cpp,hpp} that are not
# part of the public C API.

option(BENCH_PROFILE "Build att-bench with -pg (gprof) instrumentation" OFF)

add_executable(att-bench bench.cpp)

target_link_libraries(att-bench PRIVATE rocprof-trace-decoder-static pthread)

# Reach internal headers (trace_parser.hpp, stitch.hpp, gfx*/) and the
# public include/ directory.
target_include_directories(
    att-bench
    PRIVATE ${CMAKE_SOURCE_DIR}/source
            ${CMAKE_SOURCE_DIR}/include)

# Always -O3 regardless of CMAKE_BUILD_TYPE (the bench measures decode
# throughput, so an unoptimised build would be misleading).
target_compile_options(att-bench PRIVATE -O3)

if(BENCH_PROFILE)
    target_compile_options(att-bench PRIVATE -pg)
    target_link_options(att-bench PRIVATE -pg)
endif()

set_target_properties(att-bench PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bench")
