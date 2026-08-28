#
set(ROCPROFILER_LIB_RANGE_REPLAY_HEADERS digest.hpp executor.hpp range_replay.hpp
                                         range_state.hpp replay_callbacks.hpp)
set(ROCPROFILER_LIB_RANGE_REPLAY_SOURCES api.cpp digest.cpp executor.cpp range_replay.cpp
                                         range_state.cpp replay_callbacks.cpp)

target_sources(
    rocprofiler-sdk-object-library PRIVATE ${ROCPROFILER_LIB_RANGE_REPLAY_SOURCES}
                                           ${ROCPROFILER_LIB_RANGE_REPLAY_HEADERS})

if(ROCPROFILER_BUILD_TESTS)
    add_subdirectory(tests)
endif()
