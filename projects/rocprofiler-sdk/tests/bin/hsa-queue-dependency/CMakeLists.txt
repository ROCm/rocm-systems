#
#
# HSA multi-queue dependency test

cmake_minimum_required(VERSION 3.21.0 FATAL_ERROR)

project(rocprofiler-sdk-tests-bin-hsa-multiqueue-dependency LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_program(
    amdclangpp_EXECUTABLE REQUIRED
    NAMES amdclang++
    HINTS ${ROCM_PATH} ENV ROCM_PATH /opt/rocm
    PATHS ${ROCM_PATH} ENV ROCM_PATH /opt/rocm
    PATH_SUFFIXES bin llvm/bin NO_CACHE)

function(generate_hsaco TARGET_ID INPUT_FILE OUTPUT_FILE)
    separate_arguments(
        CLANG_ARG_LIST
        UNIX_COMMAND
        "-O2 -x cl -Xclang -finclude-default-header -cl-denorms-are-zero -cl-std=CL2.0 -Wl,--build-id=sha1
    -target amdgcn-amd-amdhsa -mcpu=${TARGET_ID} -o ${OUTPUT_FILE} ${INPUT_FILE}")
    add_custom_command(
        OUTPUT ${PROJECT_BINARY_DIR}/${OUTPUT_FILE}
        COMMAND ${amdclangpp_EXECUTABLE} ${CLANG_ARG_LIST}
        COMMAND ${CMAKE_COMMAND} -E copy ${PROJECT_BINARY_DIR}/${OUTPUT_FILE}
                ${CMAKE_BINARY_DIR}/tests/rocprofv3/hsa-queue-dependency/${OUTPUT_FILE}
        OUTPUT ${CMAKE_BINARY_DIR}/tests/rocprofv3/hsa-queue-dependency/${OUTPUT_FILE}
        COMMAND
            ${CMAKE_COMMAND} -E copy
            ${CMAKE_BINARY_DIR}/tests/rocprofv3/hsa-queue-dependency/${OUTPUT_FILE}
            ${CMAKE_BINARY_DIR}/rocprofv3/hsa-queue-dependency/${OUTPUT_FILE}
        COMMENT "Building ${OUTPUT_FILE}...")
    set(HSACO_TARGET_LIST
        ${HSACO_TARGET_LIST} ${PROJECT_BINARY_DIR}/${OUTPUT_FILE}
        PARENT_SCOPE)
endfunction(generate_hsaco)

foreach(target_id ${GPU_TARGETS})
    # generate kernel bitcodes
    generate_hsaco(${target_id} ${CMAKE_CURRENT_SOURCE_DIR}/copy.cl
                   ${target_id}_copy.hsaco)
endforeach()

add_custom_target(generate_hsaco_targets DEPENDS ${HSACO_TARGET_LIST})

add_executable(multiqueue_testapp)
target_sources(multiqueue_testapp PRIVATE multiqueue_app.cpp)
target_compile_options(multiqueue_testapp PRIVATE -W -Wall -Wextra -Wshadow -Werror)

find_package(Threads REQUIRED)
target_link_libraries(multiqueue_testapp PRIVATE stdc++fs Threads::Threads)

find_package(rocprofiler-sdk REQUIRED)
target_link_libraries(multiqueue_testapp PRIVATE rocprofiler-sdk::rocprofiler-sdk
                                                 rocprofiler-sdk::tests-common-library)

find_package(
    hsa-runtime64
    REQUIRED
    CONFIG
    HINTS
    ${rocm_version_DIR}
    ${ROCM_PATH}
    PATHS
    ${rocm_version_DIR}
    ${ROCM_PATH})

target_link_libraries(multiqueue_testapp PRIVATE hsa-runtime64::hsa-runtime64)

add_dependencies(multiqueue_testapp generate_hsaco_targets)
