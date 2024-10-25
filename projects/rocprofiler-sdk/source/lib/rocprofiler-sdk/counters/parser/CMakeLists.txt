rocprofiler_deactivate_clang_tidy()

set(expr_parser_sources parser.cpp parser.h scanner.cpp raw_ast.hpp reader.hpp)

add_library(rocprofiler-sdk-expr-parser OBJECT)

if(ROCPROFILER_REGENERATE_COUNTERS_PARSER)
    find_package(FLEX REQUIRED)
    find_package(BISON REQUIRED)

    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/parser.h
                   ${CMAKE_CURRENT_BINARY_DIR}/parser.h COPYONLY)

    bison_target(
        ExprBison parser.y ${CMAKE_CURRENT_BINARY_DIR}/parser.cpp
        COMPILE_FLAGS "-t"
        DEFINES_FILE ${CMAKE_CURRENT_BINARY_DIR}/parser.h)
    flex_target(ExprFlex scanner.l ${CMAKE_CURRENT_BINARY_DIR}/scanner.cpp)
    add_flex_bison_dependency(ExprFlex ExprBison)

    set_source_files_properties(${expr_parser_sources} PROPERTIES COMPILE_DEFINITIONS
                                                                  YYDEBUG=1)

    add_custom_target(
        rocprofiler-sdk-expr-parser-patch
        COMMAND
            ${CMAKE_COMMAND} -DPROJECT_SRC_DIR=${PROJECT_SOURCE_DIR}
            -DPROJECT_BLD_DIR=${PROJECT_BINARY_DIR}
            -DSOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}
            -DBINARY_DIR=${CMAKE_CURRENT_BINARY_DIR}
            -DFORMAT_EXE=${ROCPROFILER_CLANG_FORMAT_EXE} -P
            ${PROJECT_SOURCE_DIR}/source/scripts/patch-parser.cmake
        DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/parser.cpp
                ${CMAKE_CURRENT_BINARY_DIR}/scanner.cpp
        VERBATIM)

    # ensure gets applied when rocprofiler-sdk-expr-parser is built
    add_dependencies(rocprofiler-sdk-expr-parser rocprofiler-sdk-expr-parser-patch)
endif()

target_sources(rocprofiler-sdk-expr-parser PRIVATE ${expr_parser_sources})
target_include_directories(rocprofiler-sdk-expr-parser
                           PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(rocprofiler-sdk-expr-parser PRIVATE YY_NO_INPUT)
target_link_libraries(rocprofiler-sdk-expr-parser
                      PRIVATE rocprofiler-sdk::rocprofiler-sdk-common-library)
target_sources(rocprofiler-sdk-object-library
               PUBLIC $<TARGET_OBJECTS:rocprofiler-sdk-expr-parser>)

if(ROCPROFILER_BUILD_TESTS)
    add_subdirectory(tests)
endif()
