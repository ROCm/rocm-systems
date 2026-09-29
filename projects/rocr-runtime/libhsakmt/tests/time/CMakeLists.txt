cmake_minimum_required(VERSION 3.16)
project(hsakmt-clock-counters-test LANGUAGES C)

enable_testing()
add_executable(clock-counters-test clock_counters.c ../../src/time.c)
target_include_directories(clock-counters-test PRIVATE
    ../../src ../../include ../../include/hsakmt/linux)
target_compile_options(clock-counters-test PRIVATE -Wall -Wextra -Werror)
add_test(NAME clock-counters COMMAND clock-counters-test)
