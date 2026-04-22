#
set(ROCPROFILER_LIB_KERNEL_DISPATCH_SOURCES kernel_dispatch.cpp profiling_time.cpp
                                            tracing.cpp aqlmon_receiver.cpp)
set(ROCPROFILER_LIB_KERNEL_DISPATCH_HEADERS kernel_dispatch.hpp profiling_time.hpp
                                            tracing.hpp aqlmon_receiver.hpp)

target_sources(
    rocprofiler-sdk-object-library PRIVATE ${ROCPROFILER_LIB_KERNEL_DISPATCH_SOURCES}
                                           ${ROCPROFILER_LIB_KERNEL_DISPATCH_HEADERS})

target_include_directories(
    rocprofiler-sdk-object-library
    PRIVATE ${PROJECT_SOURCE_DIR}/../../experimental/aql-shm-monitor/source/include)
