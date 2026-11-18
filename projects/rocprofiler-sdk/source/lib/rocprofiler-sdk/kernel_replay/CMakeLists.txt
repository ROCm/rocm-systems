#
set(ROCPROFILER_LIB_KERNEL_REPLAY_HEADERS
    blit-copy-kernel.hpp
    blit-copy.hpp
    kernel_replay.hpp
    local_context.hpp
    memory_snapshot.hpp
    memory_tracker.hpp
    replay_callbacks.hpp
    restore_policy.hpp
    snapshot-plan.hpp
    snapshot-storage.hpp
    untracked_memory.hpp
    utils.hpp)
set(ROCPROFILER_LIB_KERNEL_REPLAY_SOURCES
    blit-copy.cpp
    kernel_replay.cpp
    local_context.cpp
    memory_snapshot.cpp
    memory_tracker.cpp
    replay_callbacks.cpp
    restore_policy.cpp
    untracked_memory.cpp
    utils.cpp)

target_sources(
    rocprofiler-sdk-object-library PRIVATE ${ROCPROFILER_LIB_KERNEL_REPLAY_SOURCES}
                                           ${ROCPROFILER_LIB_KERNEL_REPLAY_HEADERS})

find_program(
    kernel_replay_hipcc
    NAMES hipcc
    HINTS ${ROCM_PATH} ENV ROCM_PATH /opt/rocm
    PATHS ${ROCM_PATH} ENV ROCM_PATH /opt/rocm
    PATH_SUFFIXES bin llvm/bin REQUIRED)

# GPU_TARGETS is left to HIP's detection unless it is given, and detection finds nothing
# on a build machine without GPUs, so the blit kernels are also built for this list.
set(ROCPROFILER_KERNEL_REPLAY_BLIT_DEFAULT_TARGETS
    "gfx908;gfx90a;gfx942;gfx950;gfx1030;gfx1031;gfx1032;gfx1100;gfx1101;gfx1102;gfx1150;gfx1151;gfx1152;gfx1153;gfx1200;gfx1201;gfx1250"
    CACHE STRING "GPUs the kernel replay blit kernels are built for besides GPU_TARGETS")
mark_as_advanced(ROCPROFILER_KERNEL_REPLAY_BLIT_DEFAULT_TARGETS)

# A code object built without xnack or sramecc settings loads whichever setting the GPU
# uses, so a target such as gfx90a:xnack- gets the gfx90a one.
set(kernel_replay_blit_candidates ${GPU_TARGETS}
                                  ${ROCPROFILER_KERNEL_REPLAY_BLIT_DEFAULT_TARGETS})
list(TRANSFORM kernel_replay_blit_candidates REPLACE ":.*" "")
list(REMOVE_DUPLICATES kernel_replay_blit_candidates)

# Older compilers do not know the newest GPUs, which are left out rather than failing the
# build. Checking a GPU compiles a kernel, so like CMake's own checks, a GPU that passes
# is remembered in the cache.
set(kernel_replay_blit_flags -mcode-object-version=5 -O3)
set(kernel_replay_blit_check_source ${CMAKE_CURRENT_BINARY_DIR}/blit-target-check.hip)
file(WRITE ${kernel_replay_blit_check_source} "__global__ void check() {}\n")
set(kernel_replay_blit_targets)
set(kernel_replay_blit_skipped)
foreach(gpu_name IN LISTS kernel_replay_blit_candidates)
    string(MAKE_C_IDENTIFIER "ROCPROFILER_KERNEL_REPLAY_BLIT_CHECKED_${gpu_name}" checked)
    if(NOT ${checked})
        execute_process(
            COMMAND
                ${kernel_replay_hipcc} --genco --offload-arch=${gpu_name}
                ${kernel_replay_blit_flags} -o /dev/null
                ${kernel_replay_blit_check_source}
            RESULT_VARIABLE check_result
            OUTPUT_QUIET ERROR_QUIET)
        if(check_result EQUAL 0)
            set(${checked}
                ON
                CACHE INTERNAL "")
        endif()
    endif()
    if(${checked})
        list(APPEND kernel_replay_blit_targets ${gpu_name})
    else()
        list(APPEND kernel_replay_blit_skipped ${gpu_name})
    endif()
endforeach()

list(JOIN kernel_replay_blit_targets ", " kernel_replay_blit_message)
message(STATUS "Kernel replay blit kernels: ${kernel_replay_blit_message}")
if(kernel_replay_blit_skipped)
    list(JOIN kernel_replay_blit_skipped ", " kernel_replay_blit_message)
    message(
        STATUS
            "Kernel replay blit kernels: ${kernel_replay_hipcc} does not support ${kernel_replay_blit_message}"
        )
endif()
if(NOT kernel_replay_blit_targets)
    message(
        WARNING
            "Kernel replay blit kernels: ${kernel_replay_hipcc} supports none of their GPUs, so device-only snapshots are unavailable and restores are slower"
        )
endif()

# The SDK looks for the code objects in share/rocprofiler-sdk/kernel-replay beside its lib
# directory, a layout the build tree mirrors.
set(kernel_replay_blit_dir ${PROJECT_BINARY_DIR}/share/rocprofiler-sdk/kernel-replay)
file(MAKE_DIRECTORY ${kernel_replay_blit_dir})
set(kernel_replay_blit_code_objects)
foreach(gpu_name IN LISTS kernel_replay_blit_targets)
    set(output_file ${kernel_replay_blit_dir}/${gpu_name}_kernel_replay_blit.hsaco)
    add_custom_command(
        OUTPUT ${output_file}
        COMMAND
            ${kernel_replay_hipcc} --genco --no-gpu-bundle-output
            --offload-arch=${gpu_name} ${kernel_replay_blit_flags} -Xoffload-linker
            --build-id=sha1 -o ${output_file} ${CMAKE_CURRENT_SOURCE_DIR}/blit-copy.hip
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/blit-copy.hip
                ${CMAKE_CURRENT_SOURCE_DIR}/blit-copy-kernel.hpp
        COMMENT "Building ${gpu_name} kernel replay blit code object")
    list(APPEND kernel_replay_blit_code_objects ${output_file})
endforeach()

add_custom_target(kernel-replay-blit-code-objects
                  DEPENDS ${kernel_replay_blit_code_objects})
add_dependencies(rocprofiler-sdk-object-library kernel-replay-blit-code-objects)

install(
    FILES ${kernel_replay_blit_code_objects}
    DESTINATION share/rocprofiler-sdk/kernel-replay
    COMPONENT core)

if(ROCPROFILER_BUILD_TESTS)
    add_subdirectory(tests)
endif()
