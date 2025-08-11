#
set(ROCPROFILER_LIB_KERNEL_DISPATCH_SOURCES kernel_dispatch.cpp profiling_time.cpp
                                            tracing.cpp)
set(ROCPROFILER_LIB_KERNEL_DISPATCH_HEADERS kernel_dispatch.hpp profiling_time.hpp
                                            tracing.hpp)

target_sources(
    rocprofiler-sdk-object-library PRIVATE ${ROCPROFILER_LIB_KERNEL_DISPATCH_SOURCES}
                                           ${ROCPROFILER_LIB_KERNEL_DISPATCH_HEADERS})
