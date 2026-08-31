#
set(ROCPROFILER_LIB_KERNEL_REPLAY_HEADERS
    blit-copy-kernel.hpp
    blit-copy.hpp
    kernel_replay.hpp
    local_context.hpp
    memory_snapshot.hpp
    memory_tracker.hpp
    replay_callbacks.hpp
    utils.hpp)
set(ROCPROFILER_LIB_KERNEL_REPLAY_SOURCES
    blit-copy.cpp kernel_replay.cpp local_context.cpp memory_snapshot.cpp
    memory_tracker.cpp replay_callbacks.cpp utils.cpp)

target_sources(
    rocprofiler-sdk-object-library PRIVATE ${ROCPROFILER_LIB_KERNEL_REPLAY_SOURCES}
                                           ${ROCPROFILER_LIB_KERNEL_REPLAY_HEADERS})

find_program(
    kernel_replay_hipcc
    NAMES hipcc
    HINTS ${ROCM_PATH} ENV ROCM_PATH /opt/rocm
    PATHS ${ROCM_PATH} ENV ROCM_PATH /opt/rocm
    PATH_SUFFIXES bin llvm/bin REQUIRED)

set(kernel_replay_blit_targets ${GPU_TARGETS})
list(REMOVE_DUPLICATES kernel_replay_blit_targets)
set(kernel_replay_blit_code_objects)
foreach(target_id IN LISTS kernel_replay_blit_targets)
    set(output_file ${CMAKE_CURRENT_BINARY_DIR}/${target_id}_kernel_replay_blit.hsaco)
    add_custom_command(
        OUTPUT ${output_file}
        COMMAND
            ${kernel_replay_hipcc} --genco --no-gpu-bundle-output
            --offload-arch=${target_id} -mcode-object-version=5 -O3 -Xoffload-linker
            --build-id=sha1 -o ${output_file} ${CMAKE_CURRENT_SOURCE_DIR}/blit-copy.hip
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/blit-copy.hip
                ${CMAKE_CURRENT_SOURCE_DIR}/blit-copy-kernel.hpp
        COMMENT "Building ${target_id} kernel replay blit code object")
    list(APPEND kernel_replay_blit_code_objects ${output_file})
endforeach()

add_custom_target(kernel-replay-blit-code-objects
                  DEPENDS ${kernel_replay_blit_code_objects})
add_dependencies(rocprofiler-sdk-object-library kernel-replay-blit-code-objects)

install(
    FILES ${kernel_replay_blit_code_objects}
    DESTINATION ${CMAKE_INSTALL_DATADIR}/rocprofiler-sdk/kernel-replay
    COMPONENT core)

target_compile_definitions(
    rocprofiler-sdk-object-library
    PRIVATE
        ROCPROFILER_KERNEL_REPLAY_BLIT_BUILD_DIR="${CMAKE_CURRENT_BINARY_DIR}"
        ROCPROFILER_KERNEL_REPLAY_BLIT_INSTALL_DIR="${CMAKE_INSTALL_FULL_DATADIR}/rocprofiler-sdk/kernel-replay"
)

if(ROCPROFILER_BUILD_TESTS)
    add_subdirectory(tests)
endif()
