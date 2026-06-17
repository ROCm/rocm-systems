cmake_minimum_required(VERSION 3.16)
project(basic_activity_test)

set_source_files_properties(basic_activity.cpp PROPERTIES LANGUAGE HIP)
add_executable(basic_activity basic_activity.cpp)
target_link_libraries(basic_activity PRIVATE test_common)
set_target_properties(basic_activity PROPERTIES
    HIP_STANDARD 17
    HIP_STANDARD_REQUIRED ON
)
