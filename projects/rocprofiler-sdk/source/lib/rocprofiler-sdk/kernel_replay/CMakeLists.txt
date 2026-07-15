#
set(ROCPROFILER_LIB_KERNEL_REPLAY_SOURCES kernel_replay.cpp)
set(ROCPROFILER_LIB_KERNEL_REPLAY_HEADERS kernel_replay.hpp)

target_sources(
    rocprofiler-sdk-object-library PRIVATE ${ROCPROFILER_LIB_KERNEL_REPLAY_SOURCES}
                                           ${ROCPROFILER_LIB_KERNEL_REPLAY_HEADERS})
