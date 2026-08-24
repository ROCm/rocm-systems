#
#
set(ROCPROFILER_LIB_KFD_EVENT_SOURCES abi.cpp dlog_abi.cpp kfd.cpp doorbell_map.cpp
                                      kfd_correlation.cpp)
set(ROCPROFILER_LIB_KFD_EVENT_HEADERS
    defines.hpp
    kfd.hpp
    utils.hpp
    kfd_dlog_uapi.h
    dlog_drain.hpp
    record_pipe.hpp
    stream_geometry.hpp
    correlation_types.hpp
    doorbell_map.hpp
    kfd_correlation.hpp)

target_sources(
    rocprofiler-sdk-object-library PRIVATE ${ROCPROFILER_LIB_KFD_EVENT_SOURCES}
                                           ${ROCPROFILER_LIB_KFD_EVENT_HEADERS})

if(ROCPROFILER_BUILD_TESTS)
    add_subdirectory(tests)
endif()
