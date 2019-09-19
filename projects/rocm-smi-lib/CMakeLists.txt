#
# Minimum version of cmake required
#
cmake_minimum_required(VERSION 3.5.0)

## Set default module path if not already set
if(NOT DEFINED CMAKE_MODULE_PATH)
    set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake_modules/")
endif()
## Include common cmake modules
include(utils)

set(ROCM_SMI "rocm_smi")
set(ROCM_SMI_COMPONENT "lib${ROCM_SMI}")
set(ROCM_SMI_TARGET "${ROCM_SMI}64")

# The following default version values should be updated as appropriate for
# ABI breaks (update MAJOR and MINOR), and ABI/API additions (update MINOR).
# Until ABI stabilizes VERSION_MAJOR will be 0. This should be over-ridden
# by git tags (through "git describe") when they are present.
set(PKG_VERSION_MAJOR 1)
set(PKG_VERSION_MINOR 0)
set(PKG_VERSION_PATCH 0)
set(PKG_VERSION_NUM_COMMIT 0)

################# Determine the library version #########################
## Setup the package version based on git tags.
set(PKG_VERSION_GIT_TAG_PREFIX "rsmi_pkg_ver")
set(SO_VERSION_GIT_TAG_PREFIX "rsmi_so_ver")

# provide git to utilities
find_program (GIT NAMES git)

get_package_version_number("1.0.0" ${PKG_VERSION_GIT_TAG_PREFIX} GIT)
# VERSION_* variables should be set by get_version_from_tag
message("Package version: ${PKG_VERSION_STR}")

# Debian package specific variables
# Set a default value for the package version
get_version_from_tag("1.0.0.0" ${SO_VERSION_GIT_TAG_PREFIX} GIT)

# VERSION_* variables should be set by get_version_from_tag
if ( ${ROCM_PATCH_VERSION} )
    set ( VERSION_PATCH ${ROCM_PATCH_VERSION})
    set(SO_VERSION_STRING "${VERSION_MAJOR}.${VERSION_MINOR}.${VERSION_PATCH}")
else()
    set(SO_VERSION_STRING "${VERSION_MAJOR}.${VERSION_MINOR}")
endif ()
set(${ROCM_SMI}_VERSION_MAJOR "${VERSION_MAJOR}")
set(${ROCM_SMI}_VERSION_MINOR "${VERSION_MINOR}")
set(${ROCM_SMI}_VERSION_PATCH "0")
set(${ROCM_SMI}_VERSION_BUILD "0")
message("SOVERSION: ${SO_VERSION_STRING}")

## Define default variable and variables for the optional build target
##  rocm_smi_lib-dev
set(SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}
                              CACHE STRING "Location of rocm_smi source code.")
set(CMAKE_INSTALL_PREFIX "/opt/rocm"
                                CACHE STRING "Default installation directory.")
set(CPACK_PACKAGING_INSTALL_PREFIX "/opt/rocm"
                                      CACHE STRING "Default packaging prefix.")
set(CPACK_GENERATOR "DEB;RPM"  CACHE STRING "Default packaging generators.")

project(${ROCM_SMI_TARGET})

# Create a configure file to get version info from within library
configure_file(
  "${PROJECT_SOURCE_DIR}/src/${ROCM_SMI_TARGET}Config.in"
  "${PROJECT_SOURCE_DIR}/include/rocm_smi/${ROCM_SMI_TARGET}Config.h")

if (NOT DEFINED CPACK_PACKAGE_VENDOR)
    set(CPACK_PACKAGE_VENDOR "AMD")
endif()

if (NOT DEFINED CPACK_PACKAGE_CONTACT)
    set(CPACK_PACKAGE_CONTACT "Advanced Micro Devices Inc.")
endif()

if (NOT DEFINED CPACK_PACKAGE_DESCRIPTION_SUMMARY)
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY
                 "ROCm System Management Interface library")
endif()

if (NOT ROCM_SMI_PACKAGE)
   set(ROCM_SMI_PACKAGE rocm_smi_lib64)
endif()

set(CPACK_PACKAGE_FILE_NAME "${ROCM_SMI_PACKAGE}-${PKG_VERSION_STR}")
## Verbose output.
set(CMAKE_VERBOSE_MAKEFILE on)

## Compiler flags
set(CMAKE_CXX_FLAGS
 "${CMAKE_CXX_FLAGS} -Wall -Wextra -fno-rtti -m64 -msse -msse2 -std=c++11 ")
# Security options
set(CMAKE_CXX_FLAGS
 "${CMAKE_CXX_FLAGS} -Wconversion -Wcast-align ")
set(CMAKE_CXX_FLAGS
 "${CMAKE_CXX_FLAGS} -Wformat=2 -fno-common -Wstrict-overflow -Wtrampolines ")
 # Intentionally leave out -Wsign-promo. It causes spurious warnings.
set(CMAKE_CXX_FLAGS
 "${CMAKE_CXX_FLAGS}  -Woverloaded-virtual -Wreorder ")

## Security breach mitigation flags
set(CMAKE_CXX_FLAGS
 "${CMAKE_CXX_FLAGS} -DFORTIFY_SOURCE=2 -fstack-protector-all -Wcast-align")
set(CMAKE_CXX_FLAGS
 "${CMAKE_CXX_FLAGS} -Wl,-z,noexecstack -Wl,-znoexecheap -Wl,-z,relro ")
set(CMAKE_CXX_FLAGS
 "${CMAKE_CXX_FLAGS} -Wl,-z,now -fPIE")

# Use this instead of above for 32 bit
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -m32")

if ("${CMAKE_BUILD_TYPE}" STREQUAL Release)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2")
else ()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ggdb -O0 -DDEBUG")
endif ()

set(SRC_DIR "src")
set(INC_DIR "include/rocm_smi")

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include
                     ${CMAKE_CURRENT_SOURCE_DIR}/src/shared_mutex)
set(SMI_SRC_LIST "${SRC_DIR}/rocm_smi_device.cc")
set(SMI_SRC_LIST ${SMI_SRC_LIST} "${SRC_DIR}/rocm_smi_main.cc")
set(SMI_SRC_LIST ${SMI_SRC_LIST} "${SRC_DIR}/rocm_smi_monitor.cc")
set(SMI_SRC_LIST ${SMI_SRC_LIST} "${SRC_DIR}/rocm_smi.cc")
set(SMI_SRC_LIST ${SMI_SRC_LIST} "${SRC_DIR}/rocm_smi_power_mon.cc")
set(SMI_SRC_LIST ${SMI_SRC_LIST} "${SRC_DIR}/rocm_smi_utils.cc")
set(SMI_SRC_LIST ${SMI_SRC_LIST} "${SRC_DIR}/rocm_smi_counters.cc")
set(SMI_SRC_LIST ${SMI_SRC_LIST} "${SRC_DIR}/rocm_smi_kfd.cc")
set(SMI_SRC_LIST ${SMI_SRC_LIST} "${SRC_DIR}/shared_mutex/shared_mutex.c")

set(SMI_INC_LIST "${INC_DIR}/rocm_smi_device.h")
set(SMI_INC_LIST ${SMI_INC_LIST} "${INC_DIR}/rocm_smi_main.h")
set(SMI_INC_LIST ${SMI_INC_LIST} "${INC_DIR}/rocm_smi_monitor.h")
set(SMI_INC_LIST ${SMI_INC_LIST} "${INC_DIR}/rocm_smi_power_mon.h")
set(SMI_INC_LIST ${SMI_INC_LIST} "${INC_DIR}/rocm_smi_utils.h")
set(SMI_INC_LIST ${SMI_INC_LIST} "${INC_DIR}/rocm_smi_common.h")
set(SMI_INC_LIST ${SMI_INC_LIST} "${INC_DIR}/rocm_smi_exception.h")
set(SMI_INC_LIST ${SMI_INC_LIST} "${INC_DIR}/rocm_smi_counters.h")
set(SMI_INC_LIST ${SMI_INC_LIST} "${INC_DIR}/rocm_smi_kfd.h")
set(SMI_INC_LIST ${SMI_INC_LIST} "${SRC_DIR}/shared_mutex/shared_mutex.h")

set(SMI_EXAMPLE_EXE "rocm_smi_ex")

add_executable(${SMI_EXAMPLE_EXE} "example/rocm_smi_example.cc")
target_link_libraries(${SMI_EXAMPLE_EXE} ${ROCM_SMI_TARGET})
add_library(${ROCM_SMI_TARGET} SHARED ${SMI_SRC_LIST} ${SMI_INC_LIST})
target_link_libraries(${ROCM_SMI_TARGET} pthread rt)


## Set the VERSION and SOVERSION values
set_property(TARGET ${ROCM_SMI_TARGET} PROPERTY
             SOVERSION "${VERSION_MAJOR}")
set_property(TARGET ${ROCM_SMI_TARGET} PROPERTY
             VERSION "${SO_VERSION_STRING}")

## If the library is a release, strip the target library
if ("${CMAKE_BUILD_TYPE}" STREQUAL Release)
    add_custom_command(
          TARGET ${ROCM_SMI_TARGET}
          POST_BUILD COMMAND ${CMAKE_STRIP} lib${ROCM_SMI_TARGET}.so)
endif ()

## Add the install directives for the runtime library.
install(TARGETS ${ROCM_SMI_TARGET}
          LIBRARY DESTINATION ${ROCM_SMI}/lib COMPONENT ${ROCM_SMI_COMPONENT})
install(FILES ${SOURCE_DIR}/include/rocm_smi/rocm_smi.h
                                        DESTINATION rocm_smi/include/rocm_smi)

## Add the packaging directives for the runtime library.

# TODO set(CPACK_DEBIAN_PACKAGE_HOMEPAGE <GITHUB URL> )
set(CPACK_DEBIAN_PACKAGE_CONTROL_EXTRA
    "${CMAKE_CURRENT_SOURCE_DIR}/DEBIAN/postinst;
                                    ${CMAKE_CURRENT_SOURCE_DIR}/DEBIAN/prerm")
set (CPACK_DEBIAN_PACKAGE_NAME  ${ROCM_SMI_PACKAGE})
set (CPACK_DEBIAN_PACKAGE_VERSION ${PKG_VERSION_STR})

# RPM package specific variables
set(CPACK_RPM_PRE_INSTALL_SCRIPT_FILE
                                   "${CMAKE_CURRENT_SOURCE_DIR}/RPM/rpm_post")
set(CPACK_RPM_POST_UNINSTALL_SCRIPT_FILE
                                 "${CMAKE_CURRENT_SOURCE_DIR}/RPM/rpm_postun")
set (CPACK_RPM_PACKAGE_NAME  ${ROCM_SMI_PACKAGE})
set (CPACK_RPM_PACKAGE_VERSION ${PKG_VERSION_STR})

include (CPack)

# Generate Doxygen documentation
find_package(Doxygen)
if (DOXYGEN_FOUND)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/docs/rsmi_doxygen.cfg
                                   ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile @ONLY)
  add_custom_target(doc
         ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
         WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
                  COMMENT "Generating API documentation with Doxygen" VERBATIM)
else()
  message("Doxygen is not found. Will not generate documents.")
endif(DOXYGEN_FOUND)

