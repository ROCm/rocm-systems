#
set(ROCPROFILER_LIB_KERNEL_REPLAY_HEADERS kernel_replay.hpp memory_snapshot.hpp
                                          memory_tracker.hpp replay_callbacks.hpp)
set(ROCPROFILER_LIB_KERNEL_REPLAY_SOURCES kernel_replay.cpp memory_snapshot.cpp
                                          memory_tracker.cpp replay_callbacks.cpp)

target_sources(
    rocprofiler-sdk-object-library PRIVATE ${ROCPROFILER_LIB_KERNEL_REPLAY_SOURCES}
                                           ${ROCPROFILER_LIB_KERNEL_REPLAY_HEADERS})

if(ROCPROFILER_BUILD_TESTS)
    add_subdirectory(tests)
endif()
