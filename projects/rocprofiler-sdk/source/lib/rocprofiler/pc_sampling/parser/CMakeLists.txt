set(ROCPROFILER_LIB_PC_SAMPLING_PARSER_SOURCES pc_record_interface.cpp correlation.cpp
                                               translation.cpp)
set(ROCPROFILER_LIB_PC_SAMPLING_PARSER_HEADERS
    correlation.hpp gfx9.hpp gfx11.hpp gfx_unknown.hpp parser_types.hpp pc_record_interface.hpp
    rocr.hpp translation.hpp)

target_sources(
    rocprofiler-object-library PRIVATE ${ROCPROFILER_LIB_PC_SAMPLING_PARSER_SOURCES}
                                       ${ROCPROFILER_LIB_PC_SAMPLING_PARSER_HEADERS})

if(ROCPROFILER_BUILD_TESTS)
    add_subdirectory(tests)
endif()
