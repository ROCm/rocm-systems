#
set(ROCPROFILER_LIB_KERNEL_REPLAY_HEADERS
    kernel_replay.hpp
    local_context.hpp
    memory_snapshot.hpp
    memory_tracker.hpp
    queue_hooks.hpp
    replay_callbacks.hpp
    replay_diagnostics.hpp
    utils.hpp)
set(ROCPROFILER_LIB_KERNEL_REPLAY_SOURCES
    kernel_replay.cpp
    local_context.cpp
    memory_snapshot.cpp
    memory_tracker.cpp
    queue_hooks.cpp
    replay_callbacks.cpp
    replay_diagnostics.cpp
    utils.cpp)

target_sources(
    rocprofiler-sdk-object-library PRIVATE ${ROCPROFILER_LIB_KERNEL_REPLAY_SOURCES}
                                           ${ROCPROFILER_LIB_KERNEL_REPLAY_HEADERS})

if(ROCPROFILER_BUILD_TESTS)
    add_subdirectory(tests)
endif()
