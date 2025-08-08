set(CMAKE_HIP_COMPILER
    "amdclang++"
    CACHE STRING "desired c++ compiler" FORCE)
if(CMAKE_HIP_COMPILER_ID STREQUAL "Clang")
    message(STATUS "Using ${CMAKE_HIP_COMPILER} to build for amdgpu backend")
else()
    message(
        FATAL_ERROR
            "'amdclang++' compiler required to compile test binaries for ROCm platform.")
endif()

set(VCOPY_SOURCES ../sample/vcopy.cpp)
set_source_files_properties(${VCOPY_SOURCES} PROPERTIES LANGUAGE HIP)
add_executable(vcopy ${VCOPY_SOURCES})
set_target_properties(vcopy PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)

set(VMEM_SOURCES ../sample/vmem.hip)
set_source_files_properties(${VMEM_SOURCES} PROPERTIES LANGUAGE HIP)
add_executable(vmem ${VMEM_SOURCES})
set_target_properties(vmem PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)

set(VSEQ_SOURCES ../sample/vsequential_access.cpp)
set_source_files_properties(${VSEQ_SOURCES} PROPERTIES LANGUAGE HIP)
add_executable(vsequential_access ${VSEQ_SOURCES})
set_target_properties(vsequential_access PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                                                    ${CMAKE_SOURCE_DIR}/tests)

set(VRAND_SOURCES ../sample/vrandom_access.cpp)
set_source_files_properties(${VRAND_SOURCES} PROPERTIES LANGUAGE HIP)
add_executable(vrandom_access ${VRAND_SOURCES})
set_target_properties(vrandom_access PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                                                ${CMAKE_SOURCE_DIR}/tests)

set(DYNAMIC_SHARED_SOURCES ../sample/dynamic_shared/dynamic_shared.hip)
set_source_files_properties(${DYNAMIC_SHARED_SOURCES} PROPERTIES LANGUAGE HIP)
add_executable(hip_dynamic_shared ${DYNAMIC_SHARED_SOURCES})
target_include_directories(hip_dynamic_shared PRIVATE ../sample/dynamic_shared)
set_target_properties(hip_dynamic_shared PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                                                ${CMAKE_SOURCE_DIR}/tests)
