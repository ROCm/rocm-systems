#
#
set(ROCPROFILER_LIB_KFD_EVENT_SOURCES
    abi.cpp kfd.cpp kfd_profiler.cpp doorbell_map.cpp correlation_table.cpp
    results_map.cpp kfd_correlation.cpp kfd_reader.cpp)
set(ROCPROFILER_LIB_KFD_EVENT_HEADERS
    defines.hpp
    kfd.hpp
    utils.hpp
    kfd_profiler.hpp
    kfd_dlog_uapi.h
    correlation_types.hpp
    doorbell_map.hpp
    correlation_table.hpp
    results_map.hpp
    kfd_correlation.hpp
    kfd_reader.hpp)

target_sources(
    rocprofiler-sdk-object-library PRIVATE ${ROCPROFILER_LIB_KFD_EVENT_SOURCES}
                                           ${ROCPROFILER_LIB_KFD_EVENT_HEADERS})

if(ROCPROFILER_BUILD_TESTS)
    add_subdirectory(tests)
endif()
