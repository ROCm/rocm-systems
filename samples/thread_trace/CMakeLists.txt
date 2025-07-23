#
#
#
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

project(rocprofiler-sdk-samples-thread-trace LANGUAGES CXX HIP)

find_package(rocprofiler-sdk REQUIRED)
find_package(rocprofiler-sdk-roctx REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_HIP_STANDARD 17)
set(CMAKE_HIP_EXTENSIONS OFF)
set(CMAKE_HIP_STANDARD_REQUIRED ON)

foreach(_TYPE DEBUG MINSIZEREL RELEASE RELWITHDEBINFO)
    if("${CMAKE_HIP_FLAGS_${_TYPE}}" STREQUAL "")
        set(CMAKE_HIP_FLAGS_${_TYPE} "${CMAKE_CXX_FLAGS_${_TYPE}}")
    endif()
endforeach()

set_source_files_properties(
    main.cpp PROPERTIES LANGUAGE HIP COMPILE_FLAGS "${CMAKE_HIP_FLAGS_RELWITHDEBINFO}")

add_executable(thread-trace-sample)
target_sources(thread-trace-sample PRIVATE main.cpp agent.cpp)

target_link_libraries(
    thread-trace-sample
    PUBLIC rocprofiler-sdk::samples-build-flags
    PRIVATE rocprofiler-sdk::rocprofiler-sdk
            rocprofiler-sdk::samples-common-library
            rocprofiler-sdk::rocprofiler-sdk-dw
            rocprofiler-sdk-roctx::rocprofiler-sdk-roctx
            rocprofiler-sdk::rocprofiler-sdk-amd-comgr)

add_test(NAME thread-trace-sample COMMAND $<TARGET_FILE:thread-trace-sample>)

find_library(
    attdecoder_LIBRARY
    NAMES rocprof-trace-decoder
    HINTS ${ROCM_PATH}
    PATHS ${ROCM_PATH}
    PATH_SUFFIXES lib)

if(attdecoder_LIBRARY)
    cmake_path(GET attdecoder_LIBRARY PARENT_PATH attdecoder_LIB_DIR)
endif()

find_package_handle_standard_args(attdecoder REQUIRED_VARS attdecoder_LIB_DIR
                                                           attdecoder_LIBRARY)

set(IS_DISABLED ON)
if(attdecoder_FOUND)
    set(IS_DISABLED OFF)
endif()

rocprofiler_samples_get_preload_env(PRELOAD_ENV)
list(APPEND PRELOAD_ENV "ROCPROFILER_TRACE_DECODER_LIB_PATH=${attdecoder_LIB_DIR}")

set_tests_properties(
    thread-trace-sample PROPERTIES TIMEOUT 60 ENVIRONMENT ${PRELOAD_ENV} LABELS
                                   "samples;thread-trace" DISABLED ${IS_DISABLED})
