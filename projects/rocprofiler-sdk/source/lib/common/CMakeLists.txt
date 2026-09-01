#
#   Builds common utilities into a static library
#
rocprofiler_activate_clang_tidy()

set(common_sources
    demangle.cpp
    dl.cpp
    elf_utils.cpp
    environment.cpp
    logging.cpp
    md5sum.cpp
    regex.cpp
    simple_timer.cpp
    static_object.cpp
    static_tl_object.cpp
    string_entry.cpp
    utility.cpp
    uuid_v7.cpp)
set(common_headers
    abi.hpp
    defines.hpp
    demangle.hpp
    dl.hpp
    elf_utils.hpp
    environment.hpp
    filesystem.hpp
    hasher.hpp
    logging.hpp
    md5sum.hpp
    mpl.hpp
    regex.hpp
    scope_destructor.hpp
    sha256.hpp
    simple_timer.hpp
    static_object.hpp
    static_tl_object.hpp
    string_entry.hpp
    stringize_arg.hpp
    synchronized.hpp
    units.hpp
    utility.hpp
    uuid_v7.hpp)

add_library(rocprofiler-sdk-common-library STATIC)
add_library(rocprofiler-sdk::rocprofiler-sdk-common-library ALIAS
            rocprofiler-sdk-common-library)

# SHA-256 comes from shared/sha256; sha256.hpp here is a shim over it. The cache variable
# is project-prefixed so it cannot collide with the equivalent entry in another
# rocm-systems project configured in the same build.
set(ROCPROFILER_SDK_ROCM_SHA256_DIR
    "${PROJECT_SOURCE_DIR}/../../shared/sha256"
    CACHE PATH "Path to the shared SHA-256 component")
if(NOT TARGET rocm-sha256)
    if(NOT EXISTS "${ROCPROFILER_SDK_ROCM_SHA256_DIR}/CMakeLists.txt")
        message(
            FATAL_ERROR
                "Shared SHA-256 not found at ${ROCPROFILER_SDK_ROCM_SHA256_DIR}. Set "
                "ROCPROFILER_SDK_ROCM_SHA256_DIR when building rocprofiler-sdk outside "
                "the rocm-systems tree.")
    endif()
    add_subdirectory(${ROCPROFILER_SDK_ROCM_SHA256_DIR}
                     ${CMAKE_CURRENT_BINARY_DIR}/rocm-sha256)
    # sha256.cpp got these through common_sources. Applied only when this project
    # created the target, so another consumer is not given our flags.
    target_link_libraries(
        rocm-sha256
        PRIVATE $<BUILD_INTERFACE:rocprofiler-sdk::rocprofiler-sdk-build-flags>
                $<BUILD_INTERFACE:rocprofiler-sdk::rocprofiler-sdk-sanitizer>)
endif()

add_subdirectory(container)
add_subdirectory(memory)

target_sources(rocprofiler-sdk-common-library PRIVATE ${common_sources} ${common_headers})
target_include_directories(rocprofiler-sdk-common-library
                           PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/source>)

target_link_libraries(
    rocprofiler-sdk-common-library
    PUBLIC $<BUILD_INTERFACE:rocm-sha256>
           $<BUILD_INTERFACE:rocprofiler-sdk::rocprofiler-sdk-headers>
           $<BUILD_INTERFACE:rocprofiler-sdk::rocprofiler-sdk-build-flags>
           $<BUILD_INTERFACE:rocprofiler-sdk::rocprofiler-sdk-threading>
           $<BUILD_INTERFACE:rocprofiler-sdk::rocprofiler-sdk-sanitizer>
           $<BUILD_INTERFACE:rocprofiler-sdk::rocprofiler-sdk-cxx-filesystem>
           $<BUILD_INTERFACE:rocprofiler-sdk::rocprofiler-sdk-abseil>
           $<BUILD_INTERFACE:rocprofiler-sdk::rocprofiler-sdk-fmt>
           $<BUILD_INTERFACE:rocprofiler-sdk::rocprofiler-sdk-yaml-cpp>
           $<BUILD_INTERFACE:rocprofiler-sdk::rocprofiler-sdk-dl>
           $<BUILD_INTERFACE:rocprofiler-sdk::rocprofiler-sdk-ptl>
           $<BUILD_INTERFACE:rocprofiler-sdk::rocprofiler-sdk-hsakmt-nolink>
           $<BUILD_INTERFACE:rocprofiler-sdk::rocprofiler-sdk-elfio>
           $<BUILD_INTERFACE:rocprofiler-sdk::rocprofiler-sdk-dw>)

set_target_properties(rocprofiler-sdk-common-library PROPERTIES OUTPUT_NAME
                                                                rocprofiler-sdk-common)

add_subdirectory(details)
