cmake_minimum_required(VERSION 3.16)
project(mem_sizes_test)

set_source_files_properties(mem_sizes.cpp PROPERTIES LANGUAGE HIP)
add_executable(mem_sizes mem_sizes.cpp)
target_link_libraries(mem_sizes PRIVATE test_common)
set_target_properties(mem_sizes PROPERTIES
    HIP_STANDARD 17
    HIP_STANDARD_REQUIRED ON
)
