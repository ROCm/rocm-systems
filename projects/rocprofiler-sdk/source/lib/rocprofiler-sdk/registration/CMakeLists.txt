#
# registration library sources
#
if(WIN32)
    # iterate.cpp and late.cpp both drive rocprofiler-register, which is not part of the
    # Windows ETW tracing path; windows_stubs.cpp supplies the stand-in symbols.
    set(ROCPROFILER_LIB_REGISTRATION_SOURCES windows_stubs.cpp)
else()
    set(ROCPROFILER_LIB_REGISTRATION_SOURCES iterate.cpp late.cpp)
endif()
set(ROCPROFILER_LIB_REGISTRATION_HEADERS iterate.hpp late.hpp)

target_sources(
    rocprofiler-sdk-object-library PRIVATE ${ROCPROFILER_LIB_REGISTRATION_SOURCES}
                                           ${ROCPROFILER_LIB_REGISTRATION_HEADERS})
