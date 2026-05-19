cmake_minimum_required(VERSION 3.16)
project(cache_hit_test)

set_source_files_properties(cache_hit.cpp PROPERTIES LANGUAGE HIP)
add_executable(cache_hit cache_hit.cpp)
target_link_libraries(cache_hit PRIVATE test_common)
set_target_properties(cache_hit PROPERTIES
    HIP_STANDARD 17
    HIP_STANDARD_REQUIRED ON
)
