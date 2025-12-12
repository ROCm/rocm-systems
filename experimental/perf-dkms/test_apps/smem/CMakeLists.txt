cmake_minimum_required(VERSION 3.16)
project(smem_test)

set_source_files_properties(smem.cpp PROPERTIES LANGUAGE HIP)
add_executable(smem smem.cpp)
target_link_libraries(smem PRIVATE test_common)
set_target_properties(smem PROPERTIES
    HIP_STANDARD 17
    HIP_STANDARD_REQUIRED ON
)
