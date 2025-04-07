set(ROCPROFILER_LIB_HIP_SOURCES abi.cpp hip.cpp stream.cpp)
set(ROCPROFILER_LIB_HIP_HEADERS defines.hpp hip.hpp stream.hpp utils.hpp)

target_sources(rocprofiler-sdk-object-library PRIVATE ${ROCPROFILER_LIB_HIP_SOURCES}
                                                      ${ROCPROFILER_LIB_HIP_HEADERS})

add_subdirectory(details)
