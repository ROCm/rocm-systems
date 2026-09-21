if(WIN32)
    # SPM streams counter data out of aqlprofile-built AQL packets, which the ETW tracing
    # path cannot emit; windows_stubs.cpp supplies the stand-in symbols.
    set(ROCPROFILER_LIB_SPM_SOURCES windows_stubs.cpp)
else()
    set(ROCPROFILER_LIB_SPM_SOURCES core.cpp service.cpp decode.cpp interface.cpp
                                    dispatch_handlers.cpp)
endif()
set(ROCPROFILER_LIB_SPM_HEADERS core.hpp interface.hpp decode.hpp dispatch_handlers.hpp)
target_sources(rocprofiler-sdk-object-library PRIVATE ${ROCPROFILER_LIB_SPM_SOURCES}
                                                      ${ROCPROFILER_LIB_SPM_HEADERS})
if(ROCPROFILER_BUILD_TESTS)
    add_subdirectory(tests)
endif()
