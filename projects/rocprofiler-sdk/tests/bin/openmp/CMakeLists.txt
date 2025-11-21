#
#
#
cmake_minimum_required(VERSION 3.21.0 FATAL_ERROR)

if(NOT OMP_TARGET_COMPILER)
    find_program(
        amdclangpp_EXECUTABLE
        NAMES amdclang++
        HINTS ${ROCM_PATH} ENV ROCM_PATH /opt/rocm
        PATHS ${ROCM_PATH} ENV ROCM_PATH /opt/rocm
        PATH_SUFFIXES bin llvm/bin NO_CACHE)
    mark_as_advanced(amdclangpp_EXECUTABLE)

    if(amdclangpp_EXECUTABLE)
        set(OMP_TARGET_COMPILER
            "${amdclangpp_EXECUTABLE}"
            CACHE FILEPATH "")
    endif()
endif()

project(rocprofiler-sdk-tests-bin-openmp LANGUAGES CXX)

find_package(rocprofiler-sdk REQUIRED)

set(DEFAULT_GPU_TARGETS "gfx906" "gfx908" "gfx90a" "gfx942" "gfx950" "gfx1100" "gfx1101"
                        "gfx1102")

set(OPENMP_GPU_TARGETS
    "${DEFAULT_GPU_TARGETS}"
    CACHE STRING "GPU targets to compile for")

add_subdirectory(target)
