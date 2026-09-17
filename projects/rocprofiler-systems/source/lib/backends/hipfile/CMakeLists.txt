# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

if(ROCPROFSYS_BUILD_TESTING)
    add_subdirectory(tests)
endif()

add_library(rocprofiler-systems-hipfile-backend INTERFACE)
add_library(
    rocprofiler-systems::rocprofiler-systems-hipfile-backend
    ALIAS rocprofiler-systems-hipfile-backend
)

# Take hipFile's headers but NOT its library: wrapper.hpp resolves the two entry
# points with dlsym at first sample.
target_link_libraries(
    rocprofiler-systems-hipfile-backend
    INTERFACE
        rocprofiler-systems::rocprofiler-systems-headers
        rocprofiler-systems::rocprofiler-systems-compile-definitions
        rocprofiler-systems::rocprofiler-systems-rocm
        rocprofiler-systems-logger
        ${CMAKE_DL_LIBS}
)

target_include_directories(
    rocprofiler-systems-hipfile-backend
    SYSTEM
    INTERFACE $<TARGET_PROPERTY:hip::hipfile,INTERFACE_INCLUDE_DIRECTORIES>
)

# The wrapper re-checks the minimum version at compile time (against hipfile.h) and at
# run time (against the loaded libhipfile), so the number find_package() enforces is
# pushed down here instead of being restated in the header, where the two could drift.
string(REPLACE "." ";" _hipfile_min_version "${ROCPROFSYS_HIPFILE_MIN_VERSION}")
list(GET _hipfile_min_version 0 _hipfile_min_version_major)
list(GET _hipfile_min_version 1 _hipfile_min_version_minor)
list(GET _hipfile_min_version 2 _hipfile_min_version_patch)

# ROCPROFSYS_BUILD_HIPFILE is also exported globally on
# rocprofiler-systems-compile-definitions (cmake/Packages.cmake) so core can
# guard setting registration.
target_compile_definitions(
    rocprofiler-systems-hipfile-backend
    INTERFACE
        ROCPROFSYS_BUILD_HIPFILE=1
        ROCPROFSYS_HIPFILE_SONAME="libhipfile.so.${_hipfile_min_version_major}"
        ROCPROFSYS_HIPFILE_MIN_VERSION_MAJOR=${_hipfile_min_version_major}
        ROCPROFSYS_HIPFILE_MIN_VERSION_MINOR=${_hipfile_min_version_minor}
        ROCPROFSYS_HIPFILE_MIN_VERSION_PATCH=${_hipfile_min_version_patch}
)
