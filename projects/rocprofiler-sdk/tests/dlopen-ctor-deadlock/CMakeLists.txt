#
# Regression test: per-thread profiler state first used inside a HIP/HSA call, or on a ROCr thread
# the GPU is waiting on, must not take glibc's dl_load_lock, which dlopen holds while it runs a
# library's constructors. See main.cpp for what each mode exercises.
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

project(rocprofiler-sdk-tests-dlopen-ctor-deadlock LANGUAGES CXX HIP)

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

find_package(rocprofiler-sdk REQUIRED)
find_package(Threads REQUIRED)

set_source_files_properties(main.cpp ctor-plugin.cpp PROPERTIES LANGUAGE HIP)

# library the app dlopens; its constructor calls back into the app
add_library(dlopen-ctor-deadlock-plugin MODULE)
target_sources(dlopen-ctor-deadlock-plugin PRIVATE ctor-plugin.cpp)
target_link_libraries(dlopen-ctor-deadlock-plugin
                      PRIVATE ${CMAKE_DL_LIBS} rocprofiler-sdk::tests-build-flags)

add_executable(dlopen-ctor-deadlock)
target_sources(dlopen-ctor-deadlock PRIVATE main.cpp)
# the plugin finds the app's hook, and rocprofiler-register its rocprofiler_configure,
# with dlsym
set_target_properties(dlopen-ctor-deadlock PROPERTIES ENABLE_EXPORTS ON)
target_link_libraries(
    dlopen-ctor-deadlock PRIVATE rocprofiler-sdk::rocprofiler-sdk Threads::Threads
                                 ${CMAKE_DL_LIBS} rocprofiler-sdk::tests-build-flags)
add_dependencies(dlopen-ctor-deadlock dlopen-ctor-deadlock-plugin)

# ROCM-34578: code-object-load with only the app's own code-object tracing context, as
# with PyTorch's Kineto. Deferred loading keeps the app's code object unloaded until the
# worker first uses it.
rocprofiler_add_integration_execute_test(
    dlopen-ctor-deadlock
    COMMAND $<TARGET_FILE:dlopen-ctor-deadlock> code-object-load
            $<TARGET_FILE:dlopen-ctor-deadlock-plugin>
    DEPENDS dlopen-ctor-deadlock
    TIMEOUT 90
    LABELS "integration-tests"
    ENVIRONMENT "HIP_ENABLE_DEFERRED_LOADING=1"
    DISABLED_NO_KFD
    PASS_REGULAR_EXPRESSION "PASSED: code-object-load"
    FAIL_REGULAR_EXPRESSION "${ROCPROFILER_DEFAULT_FAIL_REGEX}|DEADLOCK")

string(REPLACE "LD_PRELOAD=" "ROCPROF_PRELOAD=" PRELOAD_ENV
               "${ROCPROFILER_MEMCHECK_PRELOAD_ENV}")

# The other modes run under rocprofv3 with the services whose per-thread state they reach
# from inside the runtime:
#
# * kernel-dispatch with --kernel-trace: the correlation-id stack
# * kernel-dispatch with --sys-trace: the HIP graph launch stack, the HIP event staged
#   waits and the tool's kernel-rename stack
# * scratch with --scratch-memory-trace: the scratch-memory callback data
function(rocprofiler_dlopen_ctor_deadlock_test _NAME _MODE)
    rocprofiler_add_integration_execute_test(
        rocprofv3-test-dlopen-ctor-deadlock-${_NAME}
        COMMAND
            $<TARGET_FILE:rocprofiler-sdk::rocprofv3> ${ARGN} --output-format csv -d
            ${CMAKE_CURRENT_BINARY_DIR}/${_NAME} -o out --
            $<TARGET_FILE:dlopen-ctor-deadlock> ${_MODE}
            $<TARGET_FILE:dlopen-ctor-deadlock-plugin>
        DEPENDS dlopen-ctor-deadlock
        TIMEOUT 90
        LABELS "integration-tests"
        PRELOAD "${PRELOAD_ENV}"
        DISABLED_NO_KFD
        PASS_REGULAR_EXPRESSION "PASSED: ${_MODE}"
        FAIL_REGULAR_EXPRESSION "${ROCPROFILER_DEFAULT_FAIL_REGEX}|DEADLOCK")
endfunction()

rocprofiler_dlopen_ctor_deadlock_test(kernel-dispatch-kernel-trace kernel-dispatch
                                      --kernel-trace)
rocprofiler_dlopen_ctor_deadlock_test(kernel-dispatch-sys-trace kernel-dispatch
                                      --sys-trace)
rocprofiler_dlopen_ctor_deadlock_test(scratch scratch --scratch-memory-trace)
