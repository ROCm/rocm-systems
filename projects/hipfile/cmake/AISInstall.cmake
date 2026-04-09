# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT

# Install hipFile

# From the rocm-cmake repo
include(ROCMInstallTargets)
include(ROCMCreatePackage)

# Install the library
rocm_install(TARGETS hipfile)

# Install the headers
rocm_install(
    DIRECTORY ${HIPFILE_ROOT_PATH}/include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# Install AIS tools
install(PROGRAMS tools/ais-check/ais-check DESTINATION bin)

# Install example code
# Since the input DIRECTORY is `examples` don't include it in
# the DESTINATION path or you'll get `examples/examples/*` output
if(AIS_INSTALL_EXAMPLES)
    install(
        DIRECTORY ${HIPFILE_ROOT_PATH}/examples
        DESTINATION share/doc/${CMAKE_PROJECT_NAME}
        FILES_MATCHING
            PATTERN "*.cpp"
            PATTERN "*.h"
            PATTERN "README.md"
    )

    # Create the installed CMakeLists.txt files from the templates.
    # The CMakeLists.txt files in the examples tree use our build
    # flags and infrastructure so we can ensure they are well-vetted
    # and these can't be installed.

    # aiscp
    configure_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/examples/aiscp/CMakeLists.install.cmake"
        "examples/aiscp/CMakeLists.txt"
    )
    install(FILES
        "${CMAKE_CURRENT_BINARY_DIR}/examples/aiscp/CMakeLists.txt"
        DESTINATION "share/doc/${CMAKE_PROJECT_NAME}/examples/aiscp/"
    )
endif()

# When we have RELEASE/DEV builds set up, we can split
# where these dependencies are added.
# For now, just add both runtime & dev deps.

# AMD Runtime Dependencies
if(CMAKE_HIP_PLATFORM STREQUAL "amd")
    rocm_package_add_dependencies(DEPENDS hip-runtime-amd) # Need minimum version
    # Suppressing libmount RELEASE dependency for now for the following:
    # 1) We currently default to DEV builds.
    # 2) libmount naming convention is not consistent across distros.
    #    We need to figure out the ideal method for handling this case.
    #    openSUSE (RPM): libmount1
    #    rocky    (RPM): libmount
    #    Ubuntu   (DEB): libmount1
    #rocm_package_add_dependencies(DEPENDS libmount)
endif()

# AMD Development Dependencies
if(CMAKE_HIP_PLATFORM STREQUAL "amd")
    rocm_package_add_deb_dependencies(DEPENDS hip-dev) # Need minimum version
    rocm_package_add_rpm_dependencies(DEPENDS hip-devel)

    rocm_package_add_deb_dependencies(DEPENDS libmount-dev)
    rocm_package_add_rpm_dependencies(DEPENDS libmount-devel)
endif()

# NVIDIA Runtime Dependencies
if(CMAKE_HIP_PLATFORM STREQUAL "nvidia")
    rocm_package_add_dependencies(DEPENDS libcufile)
endif()

# NVIDIA Development Dependencies
if(CMAKE_HIP_PLATFORM STREQUAL "nvidia")
    rocm_package_add_deb_dependencies(DEPENDS libcufile-dev)
    rocm_package_add_rpm_dependencies(DEPENDS libcufile-devel)
endif()

# Export the hipfile target
set(target_list hip::hipfile)

# Kitware recommends making the namespace match the package name
# to work with the Common Package Specification, but since the
# rest of ROCm doesn't do that, we'll stick with hip::
rocm_export_targets(
    TARGETS ${target_list}
    NAMESPACE hip::
)

# CPack license setup
set(CPACK_RESOURCE_FILE_LICENSE "${HIPFILE_ROOT_PATH}/LICENSE.md")
set(CPACK_RPM_PACKAGE_LICENSE "MIT")

# rocm-cmake sets CPACK_SET_DESTDIR on Linux, which conflicts with
# asking for relocatable packages
set(CPACK_PACKAGE_RELOCATABLE OFF)
set(CPACK_RPM_PACKAGE_RELOCATABLE OFF)
set(CPACK_DEB_PACKAGE_RELOCATABLE OFF)

# Some RPMs set dependencies on a library automatically instead
# of a manually defined package name via the AutoRequires &
# AutoProvides RPM features.
# ROCmCreatePackage however disables AUTOREQPROV by default.
# The RPM spec does not clearly state how setting AUTOREPROV &
# AUTOREQ / AUTOPROV to different values behaves. On observation,
# AUTOREQPROV supersedes the others and continues the build.
# Ref: https://ftp.rpm.org/max-rpm/s1-rpm-specref-preamble.html#S3-RPM-SPECREF-AUTOREQPROV
set(CPACK_RPM_PACKAGE_AUTOREQPROV "") # Intentionally set to an empty value so ROCmCreatePackage treats it as defined and does not override it.
set(CPACK_RPM_PACKAGE_AUTOPROV ON)
set(CPACK_RPM_PACKAGE_AUTOREQ OFF)
# Alternatively, we could set CPACK_RPM_PACKAGE_PROVIDES,
# but then we would have to maintain it...

# Set DEB/RPM Release Information
# rocm_create_package checks if following variables are defined in the environment:
#   - CPACK_DEBIAN_PACKAGE_RELEASE
#   - CPACK_RPM_PACKAGE_RELEASE
#   - ROCM_LIBPATCH_VERSION
# If ENV{CPACK_*_PACKAGE_RELEASE} is not defined, it will fallback to use ${PROJECT_VERSION_TWEAK}.
# See https://github.com/ROCm/ROCm/blob/8aa43d132f0d541eb5303dc532f5931cb80ad87a/tools/rocm-build/envsetup.sh#L77-L78
# If ENV{ROCM_LIBPATCH_VERSION} is not defined, the ROCm version will not be appended to the project version.
# ROCM_LIBPATCH_VERSION is ROCM_VERSION with '.' replaced by '0'.
# See: https://github.com/ROCm/ROCm/blob/8aa43d132f0d541eb5303dc532f5931cb80ad87a/tools/rocm-build/envsetup.sh#L94

if(NOT DEFINED ENV{CPACK_DEBIAN_PACKAGE_RELEASE} AND NOT DEFINED ENV{CPACK_RPM_PACKAGE_RELEASE})
    set(PROJECT_VERSION_TWEAK "local")
endif()

# Prefer to include the ROCm libpatch version even on a non-release build.
if(NOT DEFINED ENV{ROCM_LIBPATCH_VERSION})
    string(REPLACE "." "0" _rocm_libpatch_version "${ROCM_VERSION}")
    set(ENV{ROCM_LIBPATCH_VERSION} "${_rocm_libpatch_version}")
endif()

# Create the package
#
# Note that you will just get a dev package if you have BUILD_SHARED_LIBS
# set to off (since there are no shared libraries to install).
rocm_create_package(
    NAME "hipFile"
    DESCRIPTION "The hipFile library"
    MAINTAINER "hipfile-maintainer@amd.com"
)
