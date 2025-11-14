cmake_minimum_required(VERSION 3.21 FATAL_ERROR)

project(rocprofiler-systems-hip-quickstart-examples LANGUAGES CXX)

if(ROCPROFSYS_DISABLE_EXAMPLES)
    get_filename_component(_DIR ${CMAKE_CURRENT_LIST_DIR} NAME)
    if(${PROJECT_NAME} IN_LIST ROCPROFSYS_DISABLE_EXAMPLES
       OR ${_DIR} IN_LIST ROCPROFSYS_DISABLE_EXAMPLES)
        return()
    endif()
endif()

find_program(
    HIPCC_EXECUTABLE
    NAMES hipcc
    HINTS ${ROCmVersion_DIR} ${ROCM_PATH} ENV ROCM_PATH /opt/rocm
    PATHS ${ROCmVersion_DIR} ${ROCM_PATH} ENV ROCM_PATH /opt/rocm
    NO_CACHE)

mark_as_advanced(HIPCC_EXECUTABLE)

if(NOT HIPCC_EXECUTABLE)
    message(STATUS "hipcc not found. Skipping HIP quickstart examples.")
    return()
endif()

message(STATUS "Building HIP quickstart examples with hipcc: ${HIPCC_EXECUTABLE}")

if(NOT CMAKE_CXX_COMPILER_IS_HIPCC AND HIPCC_EXECUTABLE)
    if(CMAKE_CXX_COMPILER STREQUAL HIPCC_EXECUTABLE
       OR "${CMAKE_CXX_COMPILER}" MATCHES "hipcc")
        set(CMAKE_CXX_COMPILER_IS_HIPCC 1 CACHE BOOL "HIP compiler")
    endif()
endif()

function(add_hip_example target_name source_file)
    add_executable(${target_name} ${source_file})
    
    target_compile_options(${target_name} PRIVATE -W -Wall)
    
    if("${CMAKE_BUILD_TYPE}" MATCHES "Release")
        target_compile_options(${target_name} PRIVATE -g1)
    endif()
    
    if(NOT CMAKE_CXX_COMPILER_IS_HIPCC AND HIPCC_EXECUTABLE)
        if(COMMAND rocprofiler_systems_custom_compilation)
            rocprofiler_systems_custom_compilation(
                COMPILER ${HIPCC_EXECUTABLE} 
                TARGET ${target_name})
        endif()
    endif()
    
    if(ROCPROFSYS_INSTALL_EXAMPLES)
        install(TARGETS ${target_name} 
                DESTINATION bin 
                COMPONENT rocprofiler-systems-examples)
    endif()
endfunction()

add_hip_example(vector_add vector_add.cpp)
add_hip_example(matrix_multiply matrix_multiply.cpp)
add_hip_example(streams streams.cpp)


message(STATUS "HIP quickstart examples configured:")
message(STATUS "  - vector_add: Basic vector addition")
message(STATUS "  - matrix_multiply: Naive vs. optimized matrix multiplication")
message(STATUS "  - streams: Concurrent kernel execution with HIP streams")
message(STATUS "")
message(STATUS "After building, profile with:")
message(STATUS "  rocprof-sys-sample --quick --hip-trace -- ./vector_add")
message(STATUS "  rocprof-sys-sample --quick --hip-trace -- ./matrix_multiply")
message(STATUS "  rocprof-sys-sample --quick --hip-trace -- ./streams")

