cmake_minimum_required(VERSION 3.16)
project(wave_modes_test)

set_source_files_properties(wave_modes.cpp PROPERTIES LANGUAGE HIP)
add_executable(wave_modes wave_modes.cpp)
target_link_libraries(wave_modes PRIVATE test_common)
set_target_properties(wave_modes PROPERTIES
    HIP_STANDARD 17
    HIP_STANDARD_REQUIRED ON
)
