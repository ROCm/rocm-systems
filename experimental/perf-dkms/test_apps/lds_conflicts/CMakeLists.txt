cmake_minimum_required(VERSION 3.16)
project(lds_conflicts_test)

set_source_files_properties(lds_conflicts.cpp PROPERTIES LANGUAGE HIP)
add_executable(lds_conflicts lds_conflicts.cpp)
target_link_libraries(lds_conflicts PRIVATE test_common)
set_target_properties(lds_conflicts PROPERTIES
    HIP_STANDARD 17
    HIP_STANDARD_REQUIRED ON
)
