cmake_minimum_required(VERSION 3.16)
project(salu_heavy_test)

set_source_files_properties(salu_heavy.cpp PROPERTIES LANGUAGE HIP)
add_executable(salu_heavy salu_heavy.cpp)
target_link_libraries(salu_heavy PRIVATE test_common)
set_target_properties(salu_heavy PROPERTIES
    HIP_STANDARD 17
    HIP_STANDARD_REQUIRED ON
)
