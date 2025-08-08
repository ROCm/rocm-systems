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

project(rocprofiler-sdk-samples-openmp-target LANGUAGES CXX)

find_package(rocprofiler-sdk REQUIRED)

add_library(openmp-target-sample-client SHARED)
target_sources(openmp-target-sample-client PRIVATE client.cpp client.hpp)
target_link_libraries(
    openmp-target-sample-client
    PRIVATE rocprofiler-sdk::rocprofiler-sdk rocprofiler-sdk::samples-build-flags
            rocprofiler-sdk::samples-common-library)

set(DEFAULT_GPU_TARGETS ${ROCPROFILER_GPU_TARGETS})

set(OPENMP_GPU_TARGETS
    "${DEFAULT_GPU_TARGETS}"
    CACHE STRING "GPU targets to compile for")

set(ROCPROFILER_MEMCHECK_TYPES "ThreadSanitizer" "AddressSanitizer"
                               "UndefinedBehaviorSanitizer")

if(ROCPROFILER_MEMCHECK AND ROCPROFILER_MEMCHECK IN_LIST ROCPROFILER_MEMCHECK_TYPES)
    set(IS_DISABLED ON)
else()
    set(IS_DISABLED OFF)
endif()

find_package(Threads REQUIRED)
find_package(rocprofiler-sdk-roctx REQUIRED)

# disable when GPU-0 is navi2, navi3, and navi4
list(GET rocprofiler-sdk-samples-gfx-info 0 openmp-tools-gpu-0-gfx-info)
if("${openmp-tools-gpu-0-gfx-info}" MATCHES "^gfx(10|11|12)[0-9][0-9]$")
    set(IS_DISABLED ON)
endif()

add_executable(openmp-target-sample)
target_sources(openmp-target-sample PRIVATE main.cpp)
target_link_libraries(
    openmp-target-sample PRIVATE Threads::Threads
                                 rocprofiler-sdk-roctx::rocprofiler-sdk-roctx)
target_compile_options(openmp-target-sample PRIVATE -fopenmp)
target_link_options(openmp-target-sample PRIVATE -fopenmp)

foreach(_TARGET ${OPENMP_GPU_TARGETS})
    target_compile_options(openmp-target-sample PRIVATE --offload-arch=${_TARGET})
    target_link_options(openmp-target-sample PRIVATE --offload-arch=${_TARGET})
endforeach()

include(rocprofiler-sdk-custom-compilation)
rocprofiler_sdk_custom_compilation(TARGET openmp-target-sample
                                   COMPILER ${OMP_TARGET_COMPILER})

rocprofiler_samples_get_preload_env(PRELOAD_ENV openmp-target-sample-client)
rocprofiler_samples_get_ld_library_path_env(
    LIBRARY_PATH_ENV rocprofiler-sdk-roctx::rocprofiler-sdk-roctx-shared-library)

set(openmp-target-sample-env
    ${PRELOAD_ENV} ${LIBRARY_PATH_ENV} "OMP_NUM_THREADS=2" "OMP_TARGET_OFFLOAD=mandatory"
    "OMP_DISPLAY_ENV=1" "ROCR_VISIBLE_DEVICES=0")

add_test(NAME openmp-target-sample COMMAND $<TARGET_FILE:openmp-target-sample>)

set_tests_properties(
    openmp-target-sample
    PROPERTIES TIMEOUT
               45
               LABELS
               "samples;openmp-target"
               ENVIRONMENT
               "${openmp-target-sample-env}"
               FAIL_REGULAR_EXPRESSION
               "${ROCPROFILER_DEFAULT_FAIL_REGEX}"
               DISABLED
               "${IS_DISABLED}")
