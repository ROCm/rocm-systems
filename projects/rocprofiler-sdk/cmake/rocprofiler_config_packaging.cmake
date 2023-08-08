# configure packaging

function(rocprofiler_parse_release)
    if(EXISTS /etc/lsb-release AND NOT IS_DIRECTORY /etc/lsb-release)
        file(READ /etc/lsb-release _LSB_RELEASE)
        if(_LSB_RELEASE)
            string(REGEX
                   REPLACE "DISTRIB_ID=(.*)\nDISTRIB_RELEASE=(.*)\nDISTRIB_CODENAME=.*"
                           "\\1-\\2" _SYSTEM_NAME "${_LSB_RELEASE}")
        endif()
    elseif(EXISTS /etc/os-release AND NOT IS_DIRECTORY /etc/os-release)
        file(READ /etc/os-release _OS_RELEASE)
        if(_OS_RELEASE)
            string(REPLACE "\"" "" _OS_RELEASE "${_OS_RELEASE}")
            string(REPLACE "-" " " _OS_RELEASE "${_OS_RELEASE}")
            string(REGEX REPLACE "NAME=.*\nVERSION=([0-9\.]+).*\nID=([a-z]+).*" "\\2-\\1"
                                 _SYSTEM_NAME "${_OS_RELEASE}")
        endif()
    endif()
    string(TOLOWER "${_SYSTEM_NAME}" _SYSTEM_NAME)
    if(NOT _SYSTEM_NAME)
        set(_SYSTEM_NAME "${CMAKE_SYSTEM_NAME}")
    endif()
    set(_SYSTEM_NAME
        "${_SYSTEM_NAME}"
        PARENT_SCOPE)
endfunction()

# parse either /etc/lsb-release or /etc/os-release
rocprofiler_parse_release()

if(NOT _SYSTEM_NAME)
    set(_SYSTEM_NAME "${CMAKE_SYSTEM_NAME}")
endif()

# Add packaging directives
set(CPACK_PACKAGE_NAME ${PROJECT_NAME})
set(CPACK_PACKAGE_VENDOR "Advanced Micro Devices, Inc.")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "${PROJECT_DESCRIPTION}")
set(CPACK_PACKAGE_VERSION_MAJOR "${PROJECT_VERSION_MAJOR}")
set(CPACK_PACKAGE_VERSION_MINOR "${PROJECT_VERSION_MINOR}")
set(CPACK_PACKAGE_VERSION_PATCH "${PROJECT_VERSION_PATCH}")
set(CPACK_PACKAGE_CONTACT "jonathan.madsen@amd.com")
set(CPACK_RESOURCE_FILE_LICENSE "${PROJECT_SOURCE_DIR}/LICENSE")
set(CPACK_INCLUDE_TOPLEVEL_DIRECTORY OFF)
set(ROCPROFILER_CPACK_SYSTEM_NAME
    "${_SYSTEM_NAME}"
    CACHE STRING "System name, e.g. Linux or Ubuntu-18.04")
set(ROCPROFILER_CPACK_PACKAGE_SUFFIX "")

set(CPACK_PACKAGE_FILE_NAME
    "${CPACK_PACKAGE_NAME}-${ROCPROFILER_VERSION}-${ROCPROFILER_CPACK_SYSTEM_NAME}${ROCPROFILER_CPACK_PACKAGE_SUFFIX}"
    )
if(DEFINED ENV{CPACK_PACKAGE_FILE_NAME})
    set(CPACK_PACKAGE_FILE_NAME $ENV{CPACK_PACKAGE_FILE_NAME})
endif()

set(ROCPROFILER_PACKAGE_FILE_NAME
    ${CPACK_PACKAGE_NAME}-${ROCPROFILER_VERSION}-${ROCPROFILER_CPACK_SYSTEM_NAME}${ROCPROFILER_CPACK_PACKAGE_SUFFIX}
    )
rocprofiler_add_feature(ROCPROFILER_PACKAGE_FILE_NAME "CPack filename")

# -------------------------------------------------------------------------------------- #
#
# Debian package specific variables
#
# -------------------------------------------------------------------------------------- #

set(CPACK_DEBIAN_PACKAGE_HOMEPAGE "${PROJECT_HOMEPAGE_URL}")
set(CPACK_DEBIAN_PACKAGE_RELEASE
    "${ROCPROFILER_CPACK_SYSTEM_NAME}${ROCPROFILER_CPACK_PACKAGE_SUFFIX}")
string(REGEX REPLACE "([a-zA-Z])-([0-9])" "\\1\\2" CPACK_DEBIAN_PACKAGE_RELEASE
                     "${CPACK_DEBIAN_PACKAGE_RELEASE}")
string(REPLACE "-" "~" CPACK_DEBIAN_PACKAGE_RELEASE "${CPACK_DEBIAN_PACKAGE_RELEASE}")
if(DEFINED ENV{CPACK_DEBIAN_PACKAGE_RELEASE})
    set(CPACK_DEBIAN_PACKAGE_RELEASE $ENV{CPACK_DEBIAN_PACKAGE_RELEASE})
endif()

set(_DEBIAN_PACKAGE_DEPENDS "")
if(rocm_version_FOUND)
    set(_ROCPROFILER_SUFFIX " (>= 1.0.0.${rocm_version_NUMERIC_VERSION})")
    set(_ROCTRACER_SUFFIX " (>= 1.0.0.${rocm_version_NUMERIC_VERSION})")
    set(_ROCM_SMI_SUFFIX
        " (>= ${rocm_version_MAJOR_VERSION}.0.0.${rocm_version_NUMERIC_VERSION})")
endif()
string(REPLACE ";" ", " _DEBIAN_PACKAGE_DEPENDS "${_DEBIAN_PACKAGE_DEPENDS}")
set(CPACK_DEBIAN_PACKAGE_DEPENDS
    "${_DEBIAN_PACKAGE_DEPENDS}"
    CACHE STRING "Debian package dependencies" FORCE)
rocprofiler_add_feature(CPACK_DEBIAN_PACKAGE_DEPENDS "Debian package dependencies")
set(CPACK_DEBIAN_FILE_NAME "DEB-DEFAULT")

# -------------------------------------------------------------------------------------- #
#
# RPM package specific variables
#
# -------------------------------------------------------------------------------------- #

if(DEFINED CPACK_PACKAGING_INSTALL_PREFIX)
    set(CPACK_RPM_EXCLUDE_FROM_AUTO_FILELIST_ADDITION "${CPACK_PACKAGING_INSTALL_PREFIX}")
endif()

set(CPACK_RPM_PACKAGE_RELEASE
    "${ROCPROFILER_CPACK_SYSTEM_NAME}${ROCPROFILER_CPACK_PACKAGE_SUFFIX}")
string(REGEX REPLACE "([a-zA-Z])-([0-9])" "\\1\\2" CPACK_RPM_PACKAGE_RELEASE
                     "${CPACK_RPM_PACKAGE_RELEASE}")
string(REPLACE "-" "~" CPACK_RPM_PACKAGE_RELEASE "${CPACK_RPM_PACKAGE_RELEASE}")
if(DEFINED ENV{CPACK_RPM_PACKAGE_RELEASE})
    set(CPACK_RPM_PACKAGE_RELEASE $ENV{CPACK_RPM_PACKAGE_RELEASE})
endif()

# Get rpm distro
if(CPACK_RPM_PACKAGE_RELEASE)
    set(CPACK_RPM_PACKAGE_RELEASE_DIST ON)
endif()
set(CPACK_RPM_FILE_NAME "RPM-DEFAULT")

# -------------------------------------------------------------------------------------- #
#
# Prepare final version for the CPACK use
#
# -------------------------------------------------------------------------------------- #

set(CPACK_PACKAGE_VERSION
    "${CPACK_PACKAGE_VERSION_MAJOR}.${CPACK_PACKAGE_VERSION_MINOR}.${CPACK_PACKAGE_VERSION_PATCH}"
    )

include(CPack)
