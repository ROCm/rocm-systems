cmake_minimum_required(VERSION 3.16)
project(flat_memory_test)

set_source_files_properties(flat_memory.cpp PROPERTIES LANGUAGE HIP)
add_executable(flat_memory flat_memory.cpp)
target_link_libraries(flat_memory PRIVATE test_common)
set_target_properties(flat_memory PROPERTIES
    HIP_STANDARD 17
    HIP_STANDARD_REQUIRED ON
)
