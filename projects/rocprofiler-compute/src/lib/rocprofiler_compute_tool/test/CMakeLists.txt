set(target test_rocprofiler_compute)

add_executable(${target}
    test_rocprofiler_compute_tool.h
    test_rocprofiler_compute_tool.cpp
    mocks.h
    mocks.cpp
    main.cpp
)

target_link_libraries(${target} PRIVATE 
    rocprofiler-compute-tool
    gtest_main
    gmock
)

