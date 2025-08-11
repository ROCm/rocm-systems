set(ROCPROFILER_LIB_AQL_SOURCES helpers.cpp packet_construct.cpp)
set(ROCPROFILER_LIB_AQL_HEADERS helpers.hpp packet_construct.hpp aql_profile_v2.h)

target_sources(rocprofiler-sdk-object-library PRIVATE ${ROCPROFILER_LIB_AQL_SOURCES}
                                                      ${ROCPROFILER_LIB_AQL_HEADERS})

if(ROCPROFILER_BUILD_TESTS)
    add_subdirectory(tests)
endif()
