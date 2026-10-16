# Build the CSV microbenchmark and its sampling/output regression checks.

add_executable(rj_arith_benchmark benchmark.cpp sampling.cpp workloads.cpp wmma.cpp dot.cpp fp.cpp math.cpp)
target_include_directories(rj_arith_benchmark PRIVATE ${rj_arith_source_dir}/tests)
target_link_libraries(rj_arith_benchmark PRIVATE rj_arith)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(rj_arith_benchmark PRIVATE -Wall -Wextra -Werror -pedantic)
endif()

# Smoke checks have no minimum duration and use fixed repetitions, never asserting speed.
add_test(NAME "rj_arith.benchmark" COMMAND rj_arith_benchmark
  --min-ms 0 --repetitions 5)
add_test(NAME "rj_arith.benchmark_output"
  COMMAND ${CMAKE_COMMAND} -DBENCHMARK=$<TARGET_FILE:rj_arith_benchmark>
    -DRJ_ARITH_INCLUDE=${rj_arith_source_dir}/include/rj_arith/detail
    -P ${CMAKE_CURRENT_SOURCE_DIR}/check_benchmark_output.cmake)

add_executable(rj_arith_benchmark_sampling_test sampling_test.cpp sampling.cpp workloads.cpp)
target_include_directories(rj_arith_benchmark_sampling_test PRIVATE ${rj_arith_source_dir}/tests)
target_link_libraries(rj_arith_benchmark_sampling_test PRIVATE GTest::gtest_main rj_arith)
gtest_discover_tests(rj_arith_benchmark_sampling_test TEST_PREFIX "rj_arith.")
