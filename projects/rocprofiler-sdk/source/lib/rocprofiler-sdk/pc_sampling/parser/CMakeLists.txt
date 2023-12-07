set(ROCPROFILER_LIB_PC_SAMPLING_PARSER_SOURCES pc_record_interface.cpp)
file(GLOB ROCPROFILER_LIB_PC_SAMPLING_PARSER_HEADERS *.h *.hpp)

target_sources(
    rocprofiler-object-library PRIVATE ${ROCPROFILER_LIB_PC_SAMPLING_PARSER_SOURCES}
                                       ${ROCPROFILER_LIB_PC_SAMPLING_PARSER_HEADERS})

if(ROCPROFILER_BUILD_TESTS)
    add_subdirectory(tests)
endif()
