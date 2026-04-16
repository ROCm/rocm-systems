cmake_minimum_required(VERSION 3.24 FATAL_ERROR)

project(rocprofiler-sdk-shim-concept LANGUAGES C CXX)

include(GNUInstallDirs)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(Threads REQUIRED)

set(SHIM_CONCEPT_ROOT "${CMAKE_CURRENT_SOURCE_DIR}")
set(SHIM_CONCEPT_ROCM_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../..")
set(SHIM_CONCEPT_INCLUDE_DIRS
    "${SHIM_CONCEPT_ROOT}/include"
    "${SHIM_CONCEPT_ROOT}/src"
    "${SHIM_CONCEPT_ROCM_ROOT}/projects/rocprofiler-register/source/include"
    "${SHIM_CONCEPT_ROCM_ROOT}/projects/rocprofiler-sdk/source/include"
    "${SHIM_CONCEPT_ROCM_ROOT}/projects/hip/include"
    "${SHIM_CONCEPT_ROCM_ROOT}/projects/clr/hipamd/include"
    "${SHIM_CONCEPT_ROCM_ROOT}/projects/rocr-runtime/runtime/hsa-runtime/inc"
    "/opt/rocm/include"
)

find_library(SHIM_CONCEPT_HIP_LIB NAMES amdhip64 HINTS /opt/rocm/lib /opt/rocm/lib64)
find_library(SHIM_CONCEPT_HSA_LIB NAMES hsa-runtime64 HINTS /opt/rocm/lib /opt/rocm/lib64)
find_library(SHIM_CONCEPT_ROCPROFILER_SDK_LIB NAMES rocprofiler-sdk HINTS /opt/rocm/lib /opt/rocm/lib64)

if(NOT SHIM_CONCEPT_ROCPROFILER_SDK_LIB)
    message(FATAL_ERROR "Unable to find librocprofiler-sdk.so in /opt/rocm/lib or /opt/rocm/lib64")
endif()
add_library(rocprofiler_sdk_shim_concept SHARED
    src/shim_core.c
    src/shim_ipc.c
)
target_include_directories(rocprofiler_sdk_shim_concept PRIVATE ${SHIM_CONCEPT_INCLUDE_DIRS})
target_compile_definitions(rocprofiler_sdk_shim_concept PRIVATE __HIP_PLATFORM_AMD__=1)
target_compile_options(rocprofiler_sdk_shim_concept PRIVATE -O2 -Wall -Wextra)
target_link_libraries(rocprofiler_sdk_shim_concept PRIVATE Threads::Threads ${CMAKE_DL_LIBS} ${SHIM_CONCEPT_ROCPROFILER_SDK_LIB})
set_target_properties(rocprofiler_sdk_shim_concept PROPERTIES
    OUTPUT_NAME "rocprofiler-sdk-shim-concept"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_BINDIR}"
)

add_executable(shim_consumer_test
    src/shim_consumer_test.c
    src/shim_ipc.c
)
target_include_directories(shim_consumer_test PRIVATE ${SHIM_CONCEPT_INCLUDE_DIRS})
target_compile_definitions(shim_consumer_test PRIVATE __HIP_PLATFORM_AMD__=1)
target_compile_options(shim_consumer_test PRIVATE -O2 -Wall -Wextra)
target_link_libraries(shim_consumer_test PRIVATE Threads::Threads ${CMAKE_DL_LIBS} ${SHIM_CONCEPT_ROCPROFILER_SDK_LIB})
set_target_properties(shim_consumer_test PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_BINDIR}"
)

if(SHIM_CONCEPT_HIP_LIB)
    add_executable(real_hip_probe_test src/real_hip_probe_test.cpp)
    target_include_directories(real_hip_probe_test PRIVATE ${SHIM_CONCEPT_INCLUDE_DIRS})
    target_compile_definitions(real_hip_probe_test PRIVATE __HIP_PLATFORM_AMD__=1)
    target_compile_options(real_hip_probe_test PRIVATE -O2 -Wall -Wextra)
    target_link_libraries(real_hip_probe_test PRIVATE ${SHIM_CONCEPT_HIP_LIB})
    set_target_properties(real_hip_probe_test PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_BINDIR}"
    )
endif()

if(SHIM_CONCEPT_HSA_LIB)
    add_executable(real_hsa_probe_test src/real_hsa_probe_test.c)
    target_include_directories(real_hsa_probe_test PRIVATE ${SHIM_CONCEPT_INCLUDE_DIRS})
    target_compile_options(real_hsa_probe_test PRIVATE -O2 -Wall -Wextra)
    target_link_libraries(real_hsa_probe_test PRIVATE ${SHIM_CONCEPT_HSA_LIB} ${CMAKE_DL_LIBS})
    set_target_properties(real_hsa_probe_test PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_BINDIR}"
    )
endif()

message(STATUS "[rocprofiler-sdk-shim-concept] HIP runtime library: ${SHIM_CONCEPT_HIP_LIB}")
message(STATUS "[rocprofiler-sdk-shim-concept] HSA runtime library: ${SHIM_CONCEPT_HSA_LIB}")
message(STATUS "[rocprofiler-sdk-shim-concept] rocprofiler-sdk library: ${SHIM_CONCEPT_ROCPROFILER_SDK_LIB}")
