if(WIN32)
    # Both sources call into aqlprofile to translate counter metrics into AQL packets;
    # aqlprofile is not built on Windows, so windows_stubs.cpp stands in for them.
    set(ROCPROFILER_LIB_AQL_SOURCES windows_stubs.cpp)
else()
    set(ROCPROFILER_LIB_AQL_SOURCES helpers.cpp packet_construct.cpp)
endif()
set(ROCPROFILER_LIB_AQL_HEADERS helpers.hpp packet_construct.hpp)

target_sources(rocprofiler-sdk-object-library PRIVATE ${ROCPROFILER_LIB_AQL_SOURCES}
                                                      ${ROCPROFILER_LIB_AQL_HEADERS})

if(ROCPROFILER_BUILD_TESTS)
    add_subdirectory(tests)
endif()
