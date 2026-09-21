#
set(ROCPROFILER_LIB_KERNEL_REPLAY_HEADERS
    kernel_replay.hpp local_context.hpp memory_snapshot.hpp memory_tracker.hpp
    replay_callbacks.hpp utils.hpp)
if(WIN32)
    # Replay rewrites AQL dispatch packets, which the ETW tracing path cannot do;
    # windows_stubs.cpp supplies the stand-in consumer query.
    set(ROCPROFILER_LIB_KERNEL_REPLAY_SOURCES windows_stubs.cpp)
else()
    set(ROCPROFILER_LIB_KERNEL_REPLAY_SOURCES
        kernel_replay.cpp local_context.cpp memory_snapshot.cpp memory_tracker.cpp
        replay_callbacks.cpp utils.cpp)
endif()

target_sources(
    rocprofiler-sdk-object-library PRIVATE ${ROCPROFILER_LIB_KERNEL_REPLAY_SOURCES}
                                           ${ROCPROFILER_LIB_KERNEL_REPLAY_HEADERS})

if(ROCPROFILER_BUILD_TESTS)
    add_subdirectory(tests)
endif()
