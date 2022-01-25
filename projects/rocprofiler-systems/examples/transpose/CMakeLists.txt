cmake_minimum_required(VERSION 3.15 FATAL_ERROR)

project(omnitrace-transpose LANGUAGES CXX)

find_program(HIPCC_EXECUTABLE NAMES hipcc)

if(NOT HIPCC_EXECUTABLE)
    message(AUTHOR_WARNING "hipcc could not be found. Cannot build transpose target")
    return()
endif()

option(TRANSPOSE_USE_MPI "Enable MPI support in transpose exe" ${TIMEMORY_USE_MPI})

if(TRANSPOSE_USE_MPI)
    find_package(MPI REQUIRED)
endif()

add_executable(transpose transpose.cpp)

if(CMAKE_CXX_COMPILER_IS_CLANG AND TARGET omnitrace::omnitrace-compile-options)
    target_link_libraries(transpose PRIVATE omnitrace::omnitrace-compile-options)
else()
    target_compile_options(transpose PRIVATE -W -Wall)
endif()

if(TRANSPOSE_USE_MPI)
    target_compile_definitions(transpose PRIVATE USE_MPI)
    target_link_libraries(transpose PRIVATE MPI::MPI_C)
endif()

if(NOT CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    set_target_properties(transpose PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                                               ${CMAKE_BINARY_DIR})

    if(NOT CMAKE_CXX_COMPILER_IS_HIPCC AND HIPCC_EXECUTABLE)
        # defined in MacroUtilities.cmake
        omnitrace_custom_compilation(COMPILER ${HIPCC_EXECUTABLE} TARGET transpose)
    endif()
endif()
