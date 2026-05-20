set(ROCPROFILER_LIB_SPM_SOURCES interface.cpp)
set(ROCPROFILER_LIB_SPM_HEADERS interface.hpp)
target_sources(rocprofiler-sdk-object-library PRIVATE ${ROCPROFILER_LIB_SPM_SOURCES}
                                                      ${ROCPROFILER_LIB_SPM_HEADERS})
