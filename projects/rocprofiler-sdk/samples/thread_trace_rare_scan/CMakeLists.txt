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

project(rocprofiler-sdk-samples-thread-trace-rare-scan LANGUAGES CXX HIP)

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

add_executable(thread-trace-rare-scan-sample)
target_sources(thread-trace-rare-scan-sample PRIVATE main.cpp agent.cpp)

# Find the rocprof-trace-decoder library — we link directly against it to call
# rocprof_trace_decoder_rare_scan_gfx9 (extern "C" forward-declared in agent.cpp).
#
# The rare_scan export was added very recently in rocprof-trace-decoder, so the
# system-installed copy under /opt/rocm/lib likely does NOT contain the symbol
# yet. Prefer (in order):
#   1. -DROCPROF_TRACE_DECODER_LIB_DIR=<dir>   (cache var)
#   2. ROCPROF_TRACE_DECODER_LIB_DIR env var
#   3. The sibling ../rocprof-trace-decoder/build/lib (parallel-source layout)
#   4. ${ROCM_PATH}/lib (system fallback)
set(_RS_HINTS "")
if(DEFINED ROCPROF_TRACE_DECODER_LIB_DIR)
    list(APPEND _RS_HINTS "${ROCPROF_TRACE_DECODER_LIB_DIR}")
endif()
if(DEFINED ENV{ROCPROF_TRACE_DECODER_LIB_DIR})
    list(APPEND _RS_HINTS "$ENV{ROCPROF_TRACE_DECODER_LIB_DIR}")
endif()
list(APPEND _RS_HINTS
    "${CMAKE_CURRENT_SOURCE_DIR}/../../../rocprof-trace-decoder/build/lib"
    "${ROCM_PATH}/lib")

find_library(
    attdecoder_LIBRARY
    NAMES rocprof-trace-decoder
    HINTS ${_RS_HINTS}
    NO_DEFAULT_PATH)

if(NOT attdecoder_LIBRARY)
    find_library(
        attdecoder_LIBRARY
        NAMES rocprof-trace-decoder
        HINTS ${ROCM_PATH}
        PATH_SUFFIXES lib)
endif()

if(attdecoder_LIBRARY)
    cmake_path(GET attdecoder_LIBRARY PARENT_PATH attdecoder_LIB_DIR)
    message(STATUS "thread-trace-rare-scan: using decoder ${attdecoder_LIBRARY}")
endif()

find_package_handle_standard_args(attdecoder REQUIRED_VARS attdecoder_LIB_DIR
                                                           attdecoder_LIBRARY)

target_link_libraries(
    thread-trace-rare-scan-sample
    PUBLIC rocprofiler-sdk::samples-build-flags
    PRIVATE rocprofiler-sdk::rocprofiler-sdk rocprofiler-sdk::samples-common-library
            rocprofiler-sdk-roctx::rocprofiler-sdk-roctx ${attdecoder_LIBRARY})

add_test(NAME thread-trace-rare-scan-sample
         COMMAND $<TARGET_FILE:thread-trace-rare-scan-sample>)

set(IS_DISABLED ON)
if(attdecoder_FOUND)
    rocprofiler_sdk_sqtt_triple_buffer_disabled(_TB_DISABLED)
    if(NOT _TB_DISABLED)
        set(IS_DISABLED OFF)
    endif()
endif()

rocprofiler_samples_get_preload_env(PRELOAD_ENV)
list(APPEND PRELOAD_ENV "ROCPROFILER_TRACE_DECODER_LIB_PATH=${attdecoder_LIB_DIR}")
# Keep the test fast by default; the sample itself supports RUN_SECONDS=N.
list(APPEND PRELOAD_ENV "RUN_SECONDS=3")

set_tests_properties(
    thread-trace-rare-scan-sample
    PROPERTIES TIMEOUT 60 ENVIRONMENT "${PRELOAD_ENV}" LABELS
               "samples;thread-trace;triple-buffer" DISABLED ${IS_DISABLED})
