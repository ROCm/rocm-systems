# ROCm Kpack Runtime Library

# Library sources
set(KPACK_SOURCES
    src/kpack.cpp
    src/archive.cpp
    src/compression.cpp
    src/toc_parser.cpp
    src/path_resolution.cpp
    src/isa_target_match.cpp
    src/loader.cpp
)

# Library headers
set(KPACK_HEADERS
    include/rocm_kpack/kpack.h
    include/rocm_kpack/kpack_types.h
    include/rocm_kpack/kpack_export.h
)

# Create library (static or shared based on BUILD_SHARED_LIBS)
add_library(rocm_kpack ${KPACK_SOURCES} ${KPACK_HEADERS})

# Include directories
target_include_directories(rocm_kpack
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

# Link dependencies - PRIVATE since they're not exposed in public API
target_link_libraries(rocm_kpack
    PRIVATE
        msgpack-cxx
        zstd::libzstd
)

# Symbol visibility control
set_target_properties(rocm_kpack PROPERTIES
    C_VISIBILITY_PRESET hidden
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
)

# AddressSanitizer support
option(ENABLE_ASAN "Enable AddressSanitizer for memory error detection" OFF)
if(ENABLE_ASAN)
    target_compile_options(rocm_kpack PRIVATE
        -fsanitize=address
        -fno-omit-frame-pointer
    )
    target_link_options(rocm_kpack PRIVATE
        -fsanitize=address
    )
endif()

# Define export macros based on build type
get_target_property(KPACK_TYPE rocm_kpack TYPE)
if(KPACK_TYPE STREQUAL "SHARED_LIBRARY")
    # Building shared library: define ROCM_KPACK_EXPORTS
    target_compile_definitions(rocm_kpack PRIVATE ROCM_KPACK_EXPORTS)
else()
    # Building static library: define ROCM_KPACK_STATIC for consumers
    target_compile_definitions(rocm_kpack PUBLIC ROCM_KPACK_STATIC)
endif()

# Additional library properties
set_target_properties(rocm_kpack PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
    PUBLIC_HEADER "${KPACK_HEADERS}"
    POSITION_INDEPENDENT_CODE ON
)

# Installation
install(TARGETS rocm_kpack
    EXPORT rocm-kpack-targets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    PUBLIC_HEADER DESTINATION include/rocm_kpack
)

install(EXPORT rocm-kpack-targets
    FILE rocm-kpack-targets.cmake
    NAMESPACE rocm::
    DESTINATION lib/cmake/rocm-kpack
)

# Generate and install CMake package config files
include(CMakePackageConfigHelpers)

configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/rocm-kpack-config.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/rocm-kpack-config.cmake"
    INSTALL_DESTINATION lib/cmake/rocm-kpack
)

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/rocm-kpack-config-version.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/rocm-kpack-config.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/rocm-kpack-config-version.cmake"
    DESTINATION lib/cmake/rocm-kpack
)

# Tests
if(BUILD_TESTING)
    add_subdirectory(tests)
endif()
