if(DYNINST_BUILD_ELFUTILS AND DYNINST_ELFUTILS_DOWNLOAD_VERSION)
    omnitrace_add_feature(DYNINST_ELFUTILS_DOWNLOAD_VERSION "ElfUtils download version")
    foreach(_LIB dw elf)
        install(
            PROGRAMS
                ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}/dyninst-tpls/lib/lib${_LIB}${CMAKE_SHARED_LIBRARY_SUFFIX}
                ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}/dyninst-tpls/lib/lib${_LIB}${CMAKE_SHARED_LIBRARY_SUFFIX}.1
                ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}/dyninst-tpls/lib/lib${_LIB}-${DYNINST_ELFUTILS_DOWNLOAD_VERSION}${CMAKE_SHARED_LIBRARY_SUFFIX}
            DESTINATION ${CMAKE_INSTALL_LIBDIR}/dyninst-tpls/lib
            OPTIONAL)
    endforeach()
endif()

if(EXISTS /etc/lsb-release AND NOT IS_DIRECTORY /etc/lsb-release)
    file(READ /etc/lsb-release _LSB_RELEASE)
    if(_LSB_RELEASE)
        string(REGEX REPLACE "DISTRIB_ID=(.*)\nDISTRIB_RELEASE=(.*)\nDISTRIB_CODENAME=.*"
                             "\\1-\\2" _SYSTEM_NAME "${_LSB_RELEASE}")
    endif()
endif()

if(NOT _SYSTEM_NAME)
    set(_SYSTEM_NAME "${CMAKE_SYSTEM_NAME}")
endif()

# Add packaging directives
set(CPACK_PACKAGE_NAME ${PROJECT_NAME})
set(CPACK_PACKAGE_VENDOR "Advanced Micro Devices, Inc.")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY
    "Runtime instrumentation and binary rewriting for Perfetto via Dyninst")
set(CPACK_PACKAGE_VERSION_MAJOR "${PROJECT_VERSION_MAJOR}")
set(CPACK_PACKAGE_VERSION_MINOR "${PROJECT_VERSION_MINOR}")
set(CPACK_PACKAGE_VERSION_PATCH "${PROJECT_VERSION_PATCH}")
set(CPACK_PACKAGE_CONTACT "jonathan.madsen@amd.com")
set(CPACK_RESOURCE_FILE_LICENSE "${PROJECT_SOURCE_DIR}/LICENSE")
set(CPACK_INCLUDE_TOPLEVEL_DIRECTORY OFF)
set(OMNITRACE_CPACK_SYSTEM_NAME
    "${_SYSTEM_NAME}"
    CACHE STRING "System name, e.g. Linux or Ubuntu-18.04")
set(OMNITRACE_CPACK_PACKAGE_SUFFIX "")

if(OMNITRACE_USE_HIP
   OR OMNITRACE_USE_ROCTRACER
   OR OMNITRACE_USE_ROCM_SMI)
    set(OMNITRACE_CPACK_PACKAGE_SUFFIX
        "${OMNITRACE_CPACK_PACKAGE_SUFFIX}-ROCm-${ROCmVersion_NUMERIC_VERSION}")
endif()

if(OMNITRACE_USE_PAPI)
    set(OMNITRACE_CPACK_PACKAGE_SUFFIX "${OMNITRACE_CPACK_PACKAGE_SUFFIX}-PAPI")
endif()

if(OMNITRACE_USE_OMPT)
    set(OMNITRACE_CPACK_PACKAGE_SUFFIX "${OMNITRACE_CPACK_PACKAGE_SUFFIX}-OMPT")
endif()

if(OMNITRACE_USE_MPI)
    set(VALID_MPI_IMPLS "mpich" "openmpi")
    if("${MPI_C_COMPILER_INCLUDE_DIRS}" MATCHES "openmpi")
        set(OMNITRACE_MPI_IMPL "openmpi")
    elseif("${MPI_C_COMPILER_INCLUDE_DIRS}" MATCHES "mpich")
        set(OMNITRACE_MPI_IMPL "mpich")
    else()
        message(
            WARNING
                "MPI implementation could not be determined. Please set OMNITRACE_MPI_IMPL to one of the following for CPack: ${VALID_MPI_IMPLS}"
            )
    endif()
    if(OMNITRACE_MPI_IMPL AND NOT "${OMNITRACE_MPI_IMPL}" IN_LIST VALID_MPI_IMPLS)
        message(
            SEND_ERROR
                "Invalid OMNITRACE_MPI_IMPL (${OMNITRACE_MPI_IMPL}). Should be one of: ${VALID_MPI_IMPLS}"
            )
    else()
        omnitrace_add_feature(OMNITRACE_MPI_IMPL
                              "MPI implementation for CPack DEBIAN depends")
    endif()

    if("${OMNITRACE_MPI_IMPL}" STREQUAL "openmpi")
        set(OMNITRACE_MPI_IMPL_UPPER "OpenMPI")
    elseif("${OMNITRACE_MPI_IMPL}" STREQUAL "mpich")
        set(OMNITRACE_MPI_IMPL_UPPER "MPICH")
    endif()
    set(OMNITRACE_CPACK_PACKAGE_SUFFIX
        "${OMNITRACE_CPACK_PACKAGE_SUFFIX}-${OMNITRACE_MPI_IMPL_UPPER}")
endif()

if(OMNITRACE_USE_PYTHON)
    string(REPLACE "." "" OMNITRACE_CPACK_PYTHON_VERSION "PY${OMNITRACE_PYTHON_VERSION}")
    set(OMNITRACE_CPACK_PACKAGE_SUFFIX
        "${OMNITRACE_CPACK_PACKAGE_SUFFIX}-${OMNITRACE_CPACK_PYTHON_VERSION}")
endif()

set(CPACK_PACKAGE_FILE_NAME
    "${CPACK_PACKAGE_NAME}-${OMNITRACE_VERSION}-${OMNITRACE_CPACK_SYSTEM_NAME}${OMNITRACE_CPACK_PACKAGE_SUFFIX}"
    )
if(DEFINED ENV{CPACK_PACKAGE_FILE_NAME})
    set(CPACK_PACKAGE_FILE_NAME $ENV{CPACK_PACKAGE_FILE_NAME})
endif()

set(OMNITRACE_PACKAGE_FILE_NAME
    ${CPACK_PACKAGE_NAME}-${OMNITRACE_VERSION}-${OMNITRACE_CPACK_SYSTEM_NAME}${OMNITRACE_CPACK_PACKAGE_SUFFIX}
    )
omnitrace_add_feature(OMNITRACE_PACKAGE_FILE_NAME "CPack filename")

# -------------------------------------------------------------------------------------- #
#
# Debian package specific variables
#
# -------------------------------------------------------------------------------------- #

set(CPACK_DEBIAN_PACKAGE_HOMEPAGE "https://github.com/AMDResearch/omnitrace")
set(CPACK_DEBIAN_PACKAGE_RELEASE
    "${OMNITRACE_CPACK_SYSTEM_NAME}${OMNITRACE_CPACK_PACKAGE_SUFFIX}")
string(REGEX REPLACE "([a-zA-Z])-([0-9])" "\\1\\2" CPACK_DEBIAN_PACKAGE_RELEASE
                     "${CPACK_DEBIAN_PACKAGE_RELEASE}")
string(REPLACE "-" "~" CPACK_DEBIAN_PACKAGE_RELEASE "${CPACK_DEBIAN_PACKAGE_RELEASE}")
if(DEFINED ENV{CPACK_DEBIAN_PACKAGE_RELEASE})
    set(CPACK_DEBIAN_PACKAGE_RELEASE $ENV{CPACK_DEBIAN_PACKAGE_RELEASE})
endif()

set(_DEBIAN_PACKAGE_DEPENDS "")
if(DYNINST_USE_OpenMP)
    list(APPEND _DEBIAN_PACKAGE_DEPENDS libgomp1)
endif()
if(TIMEMORY_USE_PAPI)
    list(APPEND _DEBIAN_PACKAGE_DEPENDS libpapi-dev libpfm4)
endif()
if(NOT OMNITRACE_BUILD_DYNINST)
    if(NOT DYNINST_BUILD_BOOST)
        foreach(_BOOST_COMPONENT atomic system thread date-time filesystem)
            list(APPEND _DEBIAN_PACKAGE_DEPENDS
                 "libboost-${_BOOST_COMPONENT}-dev (>= 1.67.0)")
        endforeach()
    endif()
    if(NOT DYNINST_BUILD_TBB)
        list(APPEND _DEBIAN_PACKAGE_DEPENDS "libtbb-dev (>= 2018.6)")
    endif()
    if(NOT DYNINST_BUILD_LIBIBERTY)
        list(APPEND _DEBIAN_PACKAGE_DEPENDS "libiberty-dev (>= 20170913)")
    endif()
endif()
if(ROCmVersion_FOUND)
    set(_ROCTRACER_SUFFIX " (>= 1.0.0.${ROCmVersion_NUMERIC_VERSION})")
    set(_ROCM_SMI_SUFFIX
        " (>= ${ROCmVersion_MAJOR_VERSION}.0.0.${ROCmVersion_NUMERIC_VERSION})")
endif()
if(OMNITRACE_USE_ROCM_SMI)
    list(APPEND _DEBIAN_PACKAGE_DEPENDS "rocm-smi-lib${_ROCM_SMI_SUFFIX}")
endif()
if(OMNITRACE_USE_ROCTRACER)
    list(APPEND _DEBIAN_PACKAGE_DEPENDS "roctracer-dev${_ROCTRACER_SUFFIX}")
endif()
if(OMNITRACE_USE_MPI)
    if("${OMNITRACE_MPI_IMPL}" STREQUAL "openmpi")
        list(APPEND _DEBIAN_PACKAGE_DEPENDS "libopenmpi-dev")
    elseif("${OMNITRACE_MPI_IMPL}" STREQUAL "mpich")
        list(APPEND _DEBIAN_PACKAGE_DEPENDS "libmpich-dev")
    endif()
endif()
string(REPLACE ";" ", " _DEBIAN_PACKAGE_DEPENDS "${_DEBIAN_PACKAGE_DEPENDS}")
set(CPACK_DEBIAN_PACKAGE_DEPENDS
    "${_DEBIAN_PACKAGE_DEPENDS}"
    CACHE STRING "Debian package dependencies")
omnitrace_add_feature(CPACK_DEBIAN_PACKAGE_DEPENDS "Debian package dependencies")
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
    "${OMNITRACE_CPACK_SYSTEM_NAME}${OMNITRACE_CPACK_PACKAGE_SUFFIX}")
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
