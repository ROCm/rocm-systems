cmake_minimum_required(VERSION 3.16)
project(valu_heavy_test)

set_source_files_properties(valu_heavy.cpp PROPERTIES LANGUAGE HIP)
add_executable(valu_heavy valu_heavy.cpp)
target_link_libraries(valu_heavy PRIVATE test_common)
set_target_properties(valu_heavy PROPERTIES
    HIP_STANDARD 17
    HIP_STANDARD_REQUIRED ON
)
