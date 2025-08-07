# configure packaging

# Add packaging directives
set(CPACK_PACKAGE_NAME ${PROJECT_NAME})
set(CPACK_PACKAGE_VENDOR "Advanced Micro Devices, Inc.")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "${PROJECT_DESCRIPTION}")
set(CPACK_PACKAGE_VERSION_MAJOR "${PROJECT_VERSION_MAJOR}")
set(CPACK_PACKAGE_VERSION_MINOR "${PROJECT_VERSION_MINOR}")
set(CPACK_PACKAGE_VERSION_PATCH "${PROJECT_VERSION_PATCH}")
set(CPACK_PACKAGE_CONTACT "ROCm Profiler Support <dl.ROCm-Profiler.support@amd.com>")
set(CPACK_RESOURCE_FILE_LICENSE "${PROJECT_SOURCE_DIR}/LICENSE")
set(CPACK_INCLUDE_TOPLEVEL_DIRECTORY OFF)
set(ROCPROFILER_REGISTER_CPACK_SYSTEM_NAME
    "${CMAKE_SYSTEM_NAME}"
    CACHE STRING "System name, e.g. Linux or <OS>-<VERSION>")
set(ROCPROFILER_REGISTER_CPACK_PACKAGE_SUFFIX
    ""
    CACHE STRING "Package suffix")

set(CPACK_PACKAGE_FILE_NAME
    "${CPACK_PACKAGE_NAME}-${ROCPROFILER_REGISTER_VERSION}-${ROCPROFILER_REGISTER_CPACK_SYSTEM_NAME}${ROCPROFILER_REGISTER_CPACK_PACKAGE_SUFFIX}"
    )
if(DEFINED ENV{CPACK_PACKAGE_FILE_NAME})
    set(CPACK_PACKAGE_FILE_NAME $ENV{CPACK_PACKAGE_FILE_NAME})
endif()

set(ROCPROFILER_REGISTER_PACKAGE_FILE_NAME
    ${CPACK_PACKAGE_NAME}-${ROCPROFILER_REGISTER_VERSION}-${ROCPROFILER_REGISTER_CPACK_SYSTEM_NAME}${ROCPROFILER_REGISTER_CPACK_PACKAGE_SUFFIX}
    )
rocprofiler_register_add_feature(ROCPROFILER_REGISTER_PACKAGE_FILE_NAME "CPack filename")

get_cmake_property(ROCPROFILER_REGISTER_PACKAGING_COMPONENTS COMPONENTS)

rocprofiler_register_add_feature(ROCPROFILER_REGISTER_PACKAGING_COMPONENTS
                                 "Packaging components")
list(REMOVE_ITEM ROCPROFILER_REGISTER_PACKAGING_COMPONENTS "Development" "Unspecified"
     "external" "fmt-core")
list(LENGTH ROCPROFILER_REGISTER_PACKAGING_COMPONENTS
     NUM_ROCPROFILER_REGISTER_PACKAGING_COMPONENTS)

# the packages we will generate
set(ROCPROFILER_REGISTER_COMPONENT_GROUPS "core" "tests")

set(COMPONENT_GROUP_core_COMPONENTS "core" "Development" "Unspecified" "external")
set(COMPONENT_GROUP_tests_COMPONENTS "tests")

# variables for each component group. Note: eventually we will probably want to separate
# the core to just be the runtime libraries, development to be the headers and cmake
# files, the samples to just be the samples, and tools just be the tool files but right
# now we are just combining core, development, samples, and tools into one package
set(COMPONENT_NAME_core "")
set(COMPONENT_NAME_tests "tests")

set(COMPONENT_SEP_core "")
set(COMPONENT_SEP_tests "-")

set(COMPONENT_DEP_core "")
set(COMPONENT_DEP_tests "${PROJECT_NAME}")

set(COMPONENT_DESC_core "rocprofiler-register libraries and headers")
set(COMPONENT_DESC_tests "rocprofiler-register tests")

if(NOT NUM_ROCPROFILER_REGISTER_PACKAGING_COMPONENTS EQUAL 2)
    message(
        FATAL_ERROR
            "Error new install component needs COMPONENT_NAME_* and COMPONENT_SEP_* entries. Expected 2, found ${NUM_ROCPROFILER_REGISTER_PACKAGING_COMPONENTS}: ${ROCPROFILER_REGISTER_PACKAGING_COMPONENTS}"
        )
endif()

if(ROCM_DEP_ROCMCORE OR ROCPROFILER_REGISTER_DEP_ROCMCORE)
    set(CPACK_DEBIAN_PACKAGE_DEPENDS "rocm-core")
    set(CPACK_RPM_PACKAGE_REQUIRES "rocm-core")
else()
    set(CPACK_DEBIAN_PACKAGE_DEPENDS "")
    set(CPACK_RPM_PACKAGE_REQUIRES "")
endif()

foreach(COMPONENT_GROUP ${ROCPROFILER_REGISTER_COMPONENT_GROUPS})
    set(_SEP "${COMPONENT_SEP_${COMPONENT_GROUP}}")
    set(_DEP "${COMPONENT_DEP_${COMPONENT_GROUP}}")
    set(_NAME "${COMPONENT_NAME_${COMPONENT_GROUP}}")
    set(_DESC "${COMPONENT_DESC_${COMPONENT_GROUP}}")

    cpack_add_component_group(
        ${COMPONENT_GROUP}
        DISPLAY_NAME "${_NAME}"
        DESCRIPTION "${_DESC}")

    if(ROCM_DEP_ROCMCORE OR ROCPROFILER_REGISTER_DEP_ROCMCORE)
        list(INSERT _DEP 0 "rocm-core")
    endif()
    # Dependency list should be separated by comma
    string(REPLACE ";" ", " _DEP "${_DEP}")

    string(TOUPPER "${COMPONENT_GROUP}" UCOMPONENT)
    set(CPACK_DEBIAN_${UCOMPONENT}_PACKAGE_NAME "${PROJECT_NAME}${_SEP}${_NAME}")
    set(CPACK_DEBIAN_${UCOMPONENT}_PACKAGE_DEPENDS "${_DEP}")
    set(CPACK_RPM_${UCOMPONENT}_PACKAGE_NAME "${PROJECT_NAME}${_SEP}${_NAME}")
    set(CPACK_RPM_${UCOMPONENT}_PACKAGE_REQUIRES "${_DEP}")

    foreach(COMPONENT ${COMPONENT_GROUP_${COMPONENT_GROUP}_COMPONENTS})
        cpack_add_component(${COMPONENT} REQUIRED GROUP "${COMPONENT_GROUP}")
    endforeach()
endforeach()

# -------------------------------------------------------------------------------------- #
#
# Debian package specific variables
#
# -------------------------------------------------------------------------------------- #

set(CPACK_DEBIAN_PACKAGE_EPOCH 0)
set(CPACK_DEB_COMPONENT_INSTALL ON)
set(CPACK_DEBIAN_PACKAGE_SHLIBDEPS ON) # auto-generate deps based on shared libs
set(CPACK_DEBIAN_PACKAGE_GENERATE_SHLIBS ON) # generate list of shared libs provided by
                                             # package
set(CPACK_DEBIAN_TESTS_PACKAGE_SHLIBDEPS OFF) # disable for tests package
set(CPACK_DEBIAN_TESTS_PACKAGE_GENERATE_SHLIBS OFF) # disable for tests package
set(CPACK_DEBIAN_PACKAGE_HOMEPAGE "${PROJECT_HOMEPAGE_URL}")
set(CPACK_DEBIAN_PACKAGE_RELEASE
    "${ROCPROFILER_REGISTER_CPACK_SYSTEM_NAME}${ROCPROFILER_REGISTER_CPACK_PACKAGE_SUFFIX}"
    )
string(REGEX REPLACE "([a-zA-Z])-([0-9])" "\\1\\2" CPACK_DEBIAN_PACKAGE_RELEASE
                     "${CPACK_DEBIAN_PACKAGE_RELEASE}")
string(REPLACE "-" "~" CPACK_DEBIAN_PACKAGE_RELEASE "${CPACK_DEBIAN_PACKAGE_RELEASE}")
if(DEFINED ENV{CPACK_DEBIAN_PACKAGE_RELEASE})
    set(CPACK_DEBIAN_PACKAGE_RELEASE $ENV{CPACK_DEBIAN_PACKAGE_RELEASE})
endif()

set(_DEBIAN_PACKAGE_DEPENDS "rocm-core")
if(rocm_version_FOUND)
    set(_ROCPROFILER_REGISTER_SUFFIX " (>= 1.0.0.${rocm_version_NUMERIC_VERSION})")
    set(_ROCTRACER_SUFFIX " (>= 1.0.0.${rocm_version_NUMERIC_VERSION})")
    set(_ROCM_SMI_SUFFIX
        " (>= ${rocm_version_MAJOR_VERSION}.0.0.${rocm_version_NUMERIC_VERSION})")
endif()
string(REPLACE ";" ", " _DEBIAN_PACKAGE_DEPENDS "${_DEBIAN_PACKAGE_DEPENDS}")
set(CPACK_DEBIAN_PACKAGE_DEPENDS
    "${_DEBIAN_PACKAGE_DEPENDS}"
    CACHE STRING "Debian package dependencies" FORCE)
rocprofiler_register_add_feature(CPACK_DEBIAN_PACKAGE_DEPENDS
                                 "Debian package dependencies")
set(CPACK_DEBIAN_FILE_NAME "DEB-DEFAULT")

# -------------------------------------------------------------------------------------- #
#
# RPM package specific variables
#
# -------------------------------------------------------------------------------------- #

if(DEFINED CPACK_PACKAGING_INSTALL_PREFIX)
    set(CPACK_RPM_EXCLUDE_FROM_AUTO_FILELIST_ADDITION "${CPACK_PACKAGING_INSTALL_PREFIX}")
endif()

set(CPACK_RPM_PACKAGE_EPOCH 0)
set(CPACK_RPM_COMPONENT_INSTALL ON)
set(CPACK_RPM_PACKAGE_AUTOREQ ON) # auto-generate deps based on shared libs
set(CPACK_RPM_PACKAGE_AUTOPROV ON) # generate list of shared libs provided by package
set(CPACK_RPM_TESTS_PACKAGE_AUTOREQ OFF) # disable for tests package
set(CPACK_RPM_TESTS_PACKAGE_AUTOPROV OFF) # disable for tests package
set(CPACK_RPM_PACKAGE_RELEASE
    "${ROCPROFILER_REGISTER_CPACK_SYSTEM_NAME}${ROCPROFILER_REGISTER_CPACK_PACKAGE_SUFFIX}"
    )
string(REGEX REPLACE "([a-zA-Z])-([0-9])" "\\1\\2" CPACK_RPM_PACKAGE_RELEASE
                     "${CPACK_RPM_PACKAGE_RELEASE}")
string(REPLACE "-" "~" CPACK_RPM_PACKAGE_RELEASE "${CPACK_RPM_PACKAGE_RELEASE}")
if(DEFINED ENV{CPACK_RPM_PACKAGE_RELEASE})
    set(CPACK_RPM_PACKAGE_RELEASE $ENV{CPACK_RPM_PACKAGE_RELEASE})
endif()

set(_RPM_PACKAGE_DEPENDS "rocm-core")
string(REPLACE ";" ", " _RPM_PACKAGE_DEPENDS "${_RPM_PACKAGE_DEPENDS}")
set(CPACK_RPM_PACKAGE_REQUIRES
    "${_RPM_PACKAGE_DEPENDS}"
    CACHE STRING "RPM package dependencies" FORCE)
rocprofiler_register_add_feature(CPACK_RPM_PACKAGE_REQUIRES "RPM package dependencies")

# Get rpm distro
if(CPACK_RPM_PACKAGE_RELEASE)
    set(CPACK_RPM_PACKAGE_RELEASE_DIST ON)
endif()
set(CPACK_RPM_FILE_NAME "RPM-DEFAULT")
set(CPACK_RPM_PACKAGE_LICENSE "MIT")

# -------------------------------------------------------------------------------------- #
#
# Prepare final version for the CPACK use
#
# -------------------------------------------------------------------------------------- #

set(CPACK_PACKAGE_VERSION
    "${CPACK_PACKAGE_VERSION_MAJOR}.${CPACK_PACKAGE_VERSION_MINOR}.${CPACK_PACKAGE_VERSION_PATCH}"
    )
if(DEFINED ENV{ROCM_LIBPATCH_VERSION})
    set(CPACK_PACKAGE_VERSION "${CPACK_PACKAGE_VERSION}.$ENV{ROCM_LIBPATCH_VERSION}")
    message("Using CPACK_PACKAGE_VERSION ${CPACK_PACKAGE_VERSION}")
endif()

include(CPack)
