#
# registration library sources
#
set(ROCPROFILER_LIB_REGISTRATION_SOURCES iterate.cpp late.cpp)
set(ROCPROFILER_LIB_REGISTRATION_HEADERS iterate.hpp late.hpp)

target_sources(
    rocprofiler-sdk-object-library PRIVATE ${ROCPROFILER_LIB_REGISTRATION_SOURCES}
                                           ${ROCPROFILER_LIB_REGISTRATION_HEADERS})
