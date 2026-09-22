cmake_minimum_required(VERSION 3.16)
project(buffer_write_test)

set_source_files_properties(buffer_write.cpp PROPERTIES LANGUAGE HIP)
add_executable(buffer_write buffer_write.cpp)
target_link_libraries(buffer_write PRIVATE test_common)
set_target_properties(buffer_write PROPERTIES
    HIP_STANDARD 17
    HIP_STANDARD_REQUIRED ON
)
