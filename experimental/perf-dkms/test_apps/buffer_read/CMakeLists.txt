cmake_minimum_required(VERSION 3.16)
project(buffer_read_test)

set_source_files_properties(buffer_read.cpp PROPERTIES LANGUAGE HIP)
add_executable(buffer_read buffer_read.cpp)
target_link_libraries(buffer_read PRIVATE test_common)
set_target_properties(buffer_read PROPERTIES
    HIP_STANDARD 17
    HIP_STANDARD_REQUIRED ON
)
