cmake_minimum_required(VERSION 3.16)
project(waits_stalls_test)

set_source_files_properties(waits_stalls.cpp PROPERTIES LANGUAGE HIP)
add_executable(waits_stalls waits_stalls.cpp)
target_link_libraries(waits_stalls PRIVATE test_common)
set_target_properties(waits_stalls PROPERTIES
    HIP_STANDARD 17
    HIP_STANDARD_REQUIRED ON
)
