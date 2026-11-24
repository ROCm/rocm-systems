#
set(ROCPROFILER_LIB_KERNEL_REPLAY_HEADERS
    blit-code-objects.hpp
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
    blit-code-objects.cpp
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

# The GPUs this compiler can build the blit kernels for. Generic targets need code object
# version 6, and gfx6 and gfx7 lack the cross-lane permutes the kernels use.
set(kernel_replay_blit_flags -mcode-object-version=5 -O3)
execute_process(
    COMMAND ${kernel_replay_hipcc} --target=amdgcn-amd-amdhsa --print-supported-cpus
    OUTPUT_VARIABLE kernel_replay_blit_cpus
    ERROR_VARIABLE kernel_replay_blit_cpus)
string(REGEX MATCHALL "\n[ \t]+gfx[0-9a-z-]+" kernel_replay_blit_supported
             "${kernel_replay_blit_cpus}")
list(TRANSFORM kernel_replay_blit_supported STRIP)
list(FILTER kernel_replay_blit_supported EXCLUDE REGEX "-generic$|^gfx[67][0-9][0-9]$")

# GPU_TARGETS, given or detected by HIP's package, picks the GPUs. It is empty on a build
# machine without GPUs when not given, and then the blit kernels are built for every GPU
# the compiler supports. A code object built without xnack or sramecc settings loads
# whichever setting the GPU uses, so a target such as gfx90a:xnack- gets the gfx90a one.
if(GPU_TARGETS)
    set(kernel_replay_blit_candidates ${GPU_TARGETS})
    list(TRANSFORM kernel_replay_blit_candidates REPLACE ":.*" "")
    list(REMOVE_DUPLICATES kernel_replay_blit_candidates)
else()
    set(kernel_replay_blit_candidates ${kernel_replay_blit_supported})
endif()

set(kernel_replay_blit_targets)
set(kernel_replay_blit_skipped)
foreach(gpu_name IN LISTS kernel_replay_blit_candidates)
    if(gpu_name IN_LIST kernel_replay_blit_supported)
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
        STATUS "Kernel replay blit kernels: not built for ${kernel_replay_blit_message}")
endif()
if(NOT kernel_replay_blit_targets)
    message(
        WARNING
            "Kernel replay blit kernels: built for no GPU, so device-only snapshots are unavailable and restores are slower"
        )
endif()

set(kernel_replay_blit_dir ${CMAKE_CURRENT_BINARY_DIR})
set(kernel_replay_blit_code_objects)
set(kernel_replay_blit_table)
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
    string(MAKE_C_IDENTIFIER "${gpu_name}" symbol)
    string(
        APPEND
        kernel_replay_blit_table
        "ROCPROFILER_KERNEL_REPLAY_BLIT_TABLE_ENTRY(${symbol}, \"${gpu_name}\", \"${output_file}\")\n"
        )
endforeach()

add_custom_target(kernel-replay-blit-code-objects
                  DEPENDS ${kernel_replay_blit_code_objects})
add_dependencies(rocprofiler-sdk-object-library kernel-replay-blit-code-objects)

# blit-code-objects.cpp assembles the code objects into the library, so it is rebuilt
# whenever one of them is.
set(kernel_replay_blit_table_file ${CMAKE_CURRENT_BINARY_DIR}/blit-code-objects.inc)
file(CONFIGURE OUTPUT ${kernel_replay_blit_table_file} CONTENT
     "${kernel_replay_blit_table}" @ONLY)
set_source_files_properties(
    blit-code-objects.cpp TARGET_DIRECTORY rocprofiler-sdk-object-library
    PROPERTIES
        OBJECT_DEPENDS "${kernel_replay_blit_code_objects}"
        COMPILE_DEFINITIONS
        ROCPROFILER_KERNEL_REPLAY_BLIT_CODE_OBJECTS="${kernel_replay_blit_table_file}")

if(ROCPROFILER_BUILD_TESTS)
    add_subdirectory(tests)
endif()
