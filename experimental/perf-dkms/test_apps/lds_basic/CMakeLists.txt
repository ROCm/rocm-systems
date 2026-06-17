cmake_minimum_required(VERSION 3.16)
project(lds_basic_test)

set_source_files_properties(lds_basic.cpp PROPERTIES LANGUAGE HIP)
add_executable(lds_basic lds_basic.cpp)
target_link_libraries(lds_basic PRIVATE test_common)
set_target_properties(lds_basic PROPERTIES
    HIP_STANDARD 17
    HIP_STANDARD_REQUIRED ON
)
