# Add packaging directives for timemory
set(CPACK_PACKAGE_NAME ${PROJECT_NAME})
set(CPACK_PACKAGE_VENDOR "Advanced Micro Devices, Inc.")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY
    "Runtime instrumentation and binary rewriting for Perfetto via Dyninst")
set(CPACK_PACKAGE_VERSION_MAJOR "${PROJECT_VERSION_MAJOR}")
set(CPACK_PACKAGE_VERSION_MINOR "${PROJECT_VERSION_MINOR}")
set(CPACK_PACKAGE_VERSION_PATCH "${PROJECT_VERSION_PATCH}")
set(CPACK_PACKAGE_CONTACT "jonathan.madsen@amd.com")
set(CPACK_RESOURCE_FILE_LICENSE "${PROJECT_SOURCE_DIR}/LICENSE")

if(DYNINST_BUILD_BOOST
   OR DYNINST_BUILD_TBB
   OR DYNINST_BUILD_ELFUTILS
   OR DYNINST_BUILD_LIBIBERTY)
    set(CPACK_INSTALLED_DIRECTORIES
        "${CMAKE_INSTALL_PREFIX}/lib/dyninst-tpls/include" "lib/dyninst-tpls/include"
        "${CMAKE_INSTALL_PREFIX}/lib/dyninst-tpls/lib" "lib/dyninst-tpls/lib")
endif()

if(DYNINST_BUILD_ELFUTILS)
    list(APPEND CPACK_INSTALLED_DIRECTORIES
         "${CMAKE_INSTALL_PREFIX}/lib/dyninst-tpls/bin" "lib/dyninst-tpls/bin"
         "${CMAKE_INSTALL_PREFIX}/lib/dyninst-tpls/share" "lib/dyninst-tpls/share")
endif()

# Debian package specific variables
set(CPACK_DEBIAN_PACKAGE_HOMEPAGE "https://github.com/AARInternal/omnitrace-dyninst")
if(DEFINED ENV{CPACK_DEBIAN_PACKAGE_RELEASE})
    set(CPACK_DEBIAN_PACKAGE_RELEASE $ENV{CPACK_DEBIAN_PACKAGE_RELEASE})
else()
    set(CPACK_DEBIAN_PACKAGE_RELEASE "local")
endif()

# RPM package specific variables
if(DEFINED CPACK_PACKAGING_INSTALL_PREFIX)
    set(CPACK_RPM_EXCLUDE_FROM_AUTO_FILELIST_ADDITION "${CPACK_PACKAGING_INSTALL_PREFIX}")
endif()

if(DEFINED ENV{CPACK_RPM_PACKAGE_RELEASE})
    set(CPACK_RPM_PACKAGE_RELEASE $ENV{CPACK_RPM_PACKAGE_RELEASE})
else()
    set(CPACK_RPM_PACKAGE_RELEASE "local")
endif()

# Get rpm distro
if(CPACK_RPM_PACKAGE_RELEASE)
    set(CPACK_RPM_PACKAGE_RELEASE_DIST ON)
endif()

# Prepare final version for the CPACK use
set(CPACK_PACKAGE_VERSION
    "${CPACK_PACKAGE_VERSION_MAJOR}.${CPACK_PACKAGE_VERSION_MINOR}.${CPACK_PACKAGE_VERSION_PATCH}"
    )

# Set the names now using CPACK utility
set(CPACK_DEBIAN_FILE_NAME "DEB-DEFAULT")
set(CPACK_RPM_FILE_NAME "RPM-DEFAULT")

include(CPack)
