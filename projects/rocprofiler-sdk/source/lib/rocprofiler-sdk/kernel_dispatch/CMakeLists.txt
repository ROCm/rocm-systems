#
set(ROCPROFILER_LIB_KERNEL_DISPATCH_SOURCES kernel_dispatch.cpp tracing.cpp)
set(ROCPROFILER_LIB_KERNEL_DISPATCH_HEADERS kernel_dispatch.hpp tracing.hpp)

target_sources(
    rocprofiler-object-library PRIVATE ${ROCPROFILER_LIB_KERNEL_DISPATCH_SOURCES}
                                       ${ROCPROFILER_LIB_KERNEL_DISPATCH_HEADERS})
