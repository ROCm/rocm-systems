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
