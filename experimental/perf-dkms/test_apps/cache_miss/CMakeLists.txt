cmake_minimum_required(VERSION 3.16)
project(cache_miss_test)

set_source_files_properties(cache_miss.cpp PROPERTIES LANGUAGE HIP)
add_executable(cache_miss cache_miss.cpp)
target_link_libraries(cache_miss PRIVATE test_common)
set_target_properties(cache_miss PROPERTIES
    HIP_STANDARD 17
    HIP_STANDARD_REQUIRED ON
)
