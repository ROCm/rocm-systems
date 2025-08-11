cmake_minimum_required(VERSION 3.21.0 FATAL_ERROR)

if(NOT CMAKE_HIP_COMPILER)
    find_program(
        amdclangpp_EXECUTABLE
        NAMES amdclang++
        HINTS ${ROCM_PATH} ENV ROCM_PATH /opt/rocm
        PATHS ${ROCM_PATH} ENV ROCM_PATH /opt/rocm
        PATH_SUFFIXES bin llvm/bin NO_CACHE)
    mark_as_advanced(amdclangpp_EXECUTABLE)

    if(amdclangpp_EXECUTABLE)
        set(CMAKE_HIP_COMPILER "${amdclangpp_EXECUTABLE}")
    endif()
endif()

project(rocprofiler-sdk-benchmark-bin-mandelbrot LANGUAGES CXX HIP)

foreach(_TYPE DEBUG MINSIZEREL RELEASE RELWITHDEBINFO)
    if("${CMAKE_HIP_FLAGS_${_TYPE}}" STREQUAL "")
        set(CMAKE_HIP_FLAGS_${_TYPE} "${CMAKE_CXX_FLAGS_${_TYPE}}")
    endif()
endforeach()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_HIP_STANDARD 17)
set(CMAKE_HIP_EXTENSIONS OFF)
set(CMAKE_HIP_STANDARD_REQUIRED ON)

set_source_files_properties(mandelbrot.cpp PROPERTIES LANGUAGE HIP)
set_source_files_properties(utils.cpp PROPERTIES LANGUAGE HIP)

add_executable(mandelbrot)
target_sources(mandelbrot PRIVATE mandelbrot.cpp utils.cpp)
target_compile_options(mandelbrot PRIVATE -W -Wall -Wextra -Wpedantic -Werror
                                          -ffp-contract=fast)
target_include_directories(mandelbrot PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

install(
    TARGETS mandelbrot
    DESTINATION ${CMAKE_INSTALL_BINDIR}
    COMPONENT benchmark)
