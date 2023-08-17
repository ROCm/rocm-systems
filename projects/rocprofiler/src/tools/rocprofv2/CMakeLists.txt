# Setting Default Binary output directory
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})

# fmt library for formatting
include(FetchContent)
fetchcontent_declare(
    fmt
    GIT_REPOSITORY https://github.com/fmtlib/fmt.git
    GIT_TAG master)
fetchcontent_makeavailable(fmt)

# Getting HSA Include Directory
get_property(
    HSA_RUNTIME_INCLUDE_DIRECTORIES
    TARGET hsa-runtime64::hsa-runtime64
    PROPERTY INTERFACE_INCLUDE_DIRECTORIES)

find_library(
    AQLPROFILE_LIB "libhsa-amd-aqlprofile64.so"
    HINTS ${CMAKE_PREFIX_PATH}
    PATHS ${ROCM_PATH}
    PATH_SUFFIXES lib)
if(NOT AQLPROFILE_LIB)
    message(FATAL_ERROR "AQL_PROFILE not installed. Please install hsa-amd-aqlprofile!")
endif()

file(GLOB ROCPROFV2_SRC_FILES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

add_executable(rocprofv2_bin ${ROCPROFV2_SRC_FILES})
target_compile_definitions(
    rocprofv2_bin
    PUBLIC AMD_INTERNAL_BUILD
    PRIVATE PROF_API_IMPL)
target_include_directories(
    rocprofv2_bin
    PUBLIC ${HSA_RUNTIME_INCLUDE_DIRECTORIES}
           $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
           $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/inc>
    PRIVATE ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/inc)
target_link_libraries(
    rocprofv2_bin PRIVATE ${AQLPROFILE_LIB} hsa-runtime64::hsa-runtime64 stdc++fs
                          Threads::Threads atomic -ldl fmt::fmt)

install(TARGETS rocprofv2_bin RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
                                      COMPONENT runtime)
