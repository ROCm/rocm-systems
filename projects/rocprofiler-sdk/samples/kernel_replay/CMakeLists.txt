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

project(rocprofiler-sdk-samples-kernel-replay LANGUAGES CXX HIP)

foreach(_TYPE DEBUG MINSIZEREL RELEASE RELWITHDEBINFO)
    if("${CMAKE_HIP_FLAGS_${_TYPE}}" STREQUAL "")
        set(CMAKE_HIP_FLAGS_${_TYPE} "${CMAKE_CXX_FLAGS_${_TYPE}}")
    endif()
endforeach()

find_package(rocprofiler-sdk REQUIRED)
find_package(Threads REQUIRED)

set_source_files_properties(main.cpp PROPERTIES LANGUAGE HIP)

rocprofiler_samples_get_ld_library_path_env(LIBRARY_PATH_ENV)
rocprofiler_sdk_pc_sampling_disabled(IS_PC_SAMPLING_DISABLED)
rocprofiler_sdk_spm_disabled(IS_SPM_DISABLED)

# Configuring the thread trace service succeeds without the trace decoder, but reading the
# collected data back needs it: aqlprofile_att_iterate_data fails and the SDK reports that
# as a CI error, which aborts the sample. Locate the decoder the way the thread_trace
# sample does so the ATT samples can be handed its path, and skip them when it is absent.
include(FindPackageHandleStandardArgs)

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

# PC sampling is agent-wide and does not consult localized pass overrides yet. Samples
# that assign counters and PC sampling to different replay passes still run both on every
# pass and can hit the MI2xx/MI3xx clock-gating conflict. Keep those ctests disabled until
# the service honors local_context_override() at collection time.
set(KR_PC_SAMPLING_PASS_PARTITION_UNSUPPORTED ON)

function(add_kernel_replay_sample NAME CLIENT_SRC)
    cmake_parse_arguments(arg "PC_SAMPLING;SPM;ATT" "" "ENVIRONMENT" ${ARGN})

    add_library(${NAME}-client SHARED)
    target_sources(${NAME}-client PRIVATE ${CLIENT_SRC} client.hpp)
    target_link_libraries(
        ${NAME}-client
        PRIVATE rocprofiler-sdk::rocprofiler-sdk rocprofiler-sdk::samples-build-flags
                rocprofiler-sdk::samples-common-library)

    add_executable(${NAME})
    target_sources(${NAME} PRIVATE main.cpp)
    # Preload the tool; do not link it into the HIP executable. Linking the client (and
    # thus the SDK) into the device binary makes HIP report hipErrorInvalidDeviceFunction
    # on the first kernel launch on gfx942.
    target_link_libraries(${NAME} PRIVATE Threads::Threads
                                          rocprofiler-sdk::samples-build-flags)
    add_dependencies(${NAME} ${NAME}-client)

    rocprofiler_samples_get_preload_env(PRELOAD_ENV ${NAME}-client)
    rocprofiler_samples_get_ld_library_path_env(LIBRARY_PATH_ENV ${NAME}-client)
    set(_env ${PRELOAD_ENV} ${LIBRARY_PATH_ENV} ${arg_ENVIRONMENT})
    if(arg_SPM)
        list(APPEND _env "ROCPROFILER_SPM_BETA_ENABLED=True")
    endif()
    if(arg_ATT)
        list(APPEND _env "ROCPROFILER_TRACE_DECODER_LIB_PATH=${attdecoder_LIB_DIR}")
    endif()

    add_test(NAME ${NAME} COMMAND $<TARGET_FILE:${NAME}>)
    # These are the kernel-replay feature samples; do not hide them behind the repo-wide
    # unstable gate. PC sampling / SPM still drop out when unavailable.
    set(_disabled OFF)
    if(arg_PC_SAMPLING)
        if(KR_PC_SAMPLING_PASS_PARTITION_UNSUPPORTED)
            set(_disabled ON)
        else()
            set(_disabled "${IS_PC_SAMPLING_DISABLED}")
        endif()
    endif()
    if(arg_SPM AND IS_SPM_DISABLED)
        set(_disabled ON)
    endif()
    if(arg_ATT AND NOT attdecoder_FOUND)
        set(_disabled ON)
    endif()
    set_tests_properties(
        ${NAME}
        PROPERTIES TIMEOUT
                   120
                   LABELS
                   "samples;kernel-replay"
                   ENVIRONMENT
                   "${_env}"
                   FAIL_REGULAR_EXPRESSION
                   "${ROCPROFILER_DEFAULT_FAIL_REGEX}"
                   SKIP_REGULAR_EXPRESSION
                   "PC sampling unavailable|SPM unavailable|ATT unavailable"
                   DISABLED
                   "${_disabled}")
endfunction()

add_kernel_replay_sample(kernel-replay-basic basic_client.cpp)
add_kernel_replay_sample(kernel-replay-basic-user-data basic_client_with_user_data.cpp)
add_kernel_replay_sample(kernel-replay-counters counters_client.cpp)
add_kernel_replay_sample(kernel-replay-counters-then-pc-sampling
                         counters_then_pc_sampling_client.cpp PC_SAMPLING)
add_kernel_replay_sample(kernel-replay-att att_client.cpp ATT)
add_kernel_replay_sample(kernel-replay-spm spm_client.cpp SPM)
add_kernel_replay_sample(
    kernel-replay-services-first service_sequence_client.cpp PC_SAMPLING SPM ATT
    ENVIRONMENT "KR_SERVICE_ORDER=services-first")
add_kernel_replay_sample(kernel-replay-services-last service_sequence_client.cpp
                         PC_SAMPLING SPM ATT ENVIRONMENT "KR_SERVICE_ORDER=services-last")
add_kernel_replay_sample(kernel-replay-opt-out opt_out_client.cpp)
add_kernel_replay_sample(kernel-replay-early-exit early_exit_client.cpp)
