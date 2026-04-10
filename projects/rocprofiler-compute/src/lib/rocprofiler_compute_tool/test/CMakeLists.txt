set(target test_rocprofiler_compute)
add_executable(${target}
    test_rocprofiler_compute_tool.cpp
)

target_link_libraries(${target} PRIVATE 
    rocprofiler-compute-tool
    gtest_main
)

