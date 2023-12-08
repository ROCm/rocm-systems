set(ROCPROFILER_LIB_PC_SAMPLING_PARSER_SOURCES pc_record_interface.cpp)
set(ROCPROFILER_LIB_PC_SAMPLING_PARSER_HEADERS
    correlation.hpp gfx9.hpp gfx11.hpp gfx_unknown.hpp parser_types.h pc_record_interface.hpp
    rocr.h translation.hpp)

target_sources(
    rocprofiler-object-library PRIVATE ${ROCPROFILER_LIB_PC_SAMPLING_PARSER_SOURCES}
                                       ${ROCPROFILER_LIB_PC_SAMPLING_PARSER_HEADERS})

if(ROCPROFILER_BUILD_TESTS)
    add_subdirectory(tests)
endif()
