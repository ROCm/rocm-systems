set(ROCPROFILER_LIB_PC_SAMPLING_PARSER_SOURCES pc_record_interface.cpp)
set(ROCPROFILER_LIB_PC_SAMPLING_PARSER_HEADERS
    correlation.hpp
    gfx9.hpp
    gfx11.hpp
    parser_types.hpp
    pc_record_interface.hpp
    rocr.h
    translation.hpp
    gfx950.hpp
    gfx12.hpp)

target_sources(
    rocprofiler-sdk-object-library PRIVATE ${ROCPROFILER_LIB_PC_SAMPLING_PARSER_SOURCES}
                                           ${ROCPROFILER_LIB_PC_SAMPLING_PARSER_HEADERS})

if(ROCPROFILER_BUILD_TESTS)
    add_subdirectory(tests)
endif()
