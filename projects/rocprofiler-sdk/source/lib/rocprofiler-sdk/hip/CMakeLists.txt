set(ROCPROFILER_LIB_HIP_SOURCES hip.cpp)
set(ROCPROFILER_LIB_HIP_HEADERS defines.hpp hip.hpp types.hpp utils.hpp)

target_sources(rocprofiler-object-library PRIVATE ${ROCPROFILER_LIB_HIP_SOURCES}
                                                  ${ROCPROFILER_LIB_HIP_HEADERS})

add_subdirectory(details)
