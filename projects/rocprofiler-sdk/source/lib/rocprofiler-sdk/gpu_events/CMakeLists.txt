#
set(ROCPROFILER_LIB_GPU_EVENTS_SOURCES gpu_events.cpp hip_interception.cpp tracing.cpp)
set(ROCPROFILER_LIB_GPU_EVENTS_HEADERS gpu_events.hpp hip_interception.hpp tracing.hpp)

target_sources(
    rocprofiler-sdk-object-library PRIVATE ${ROCPROFILER_LIB_GPU_EVENTS_SOURCES}
                                           ${ROCPROFILER_LIB_GPU_EVENTS_HEADERS})
