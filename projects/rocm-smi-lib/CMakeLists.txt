################################################################################
##
## Copyright (c) 2018 Advanced Micro Devices, Inc. All rights reserved.
##
## MIT LICENSE:
## Permission is hereby granted, free of charge, to any person obtaining a copy of
## this software and associated documentation files (the "Software"), to deal in
## the Software without restriction, including without limitation the rights to
## use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
## of the Software, and to permit persons to whom the Software is furnished to do
## so, subject to the following conditions:
##
## The above copyright notice and this permission notice shall be included in all
## copies or substantial portions of the Software.
##
## THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
## IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
## FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
## AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
## LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
## OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
## SOFTWARE.
##
################################################################################

#
# Minimum version of cmake required
#
cmake_minimum_required(VERSION 3.5.0)

set(ROCM_SMI "rocm_smi")
set(ROCM_SMI_COMPONENT "lib${ROCM_SMI}")
set(ROCM_SMI_TARGET "${ROCM_SMI}64")

project(${ROCM_SMI_TARGET})

## Set default module path if not already set
if(NOT DEFINED CMAKE_MODULE_PATH)
    set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake_modules/")
endif()

## Include common cmake modules
include(utils)

## Setup the package version.
get_version ("1.0.0")

if (NOT DEFINED CPACK_PACKAGE_VENDOR)
    set(CPACK_PACKAGE_VENDOR "AMD")
endif()

if (NOT DEFINED CPACK_PACKAGE_VERSION_MAJOR)
    set(CPACK_PACKAGE_VERSION_MAJOR "1")
endif()

if (NOT DEFINED CPACK_PACKAGE_VERSION_MINOR)
    set(CPACK_PACKAGE_VERSION_MINOR "0")
endif()

if (NOT DEFINED CPACK_PACKAGE_VERSION_PATCH)
    set(CPACK_PACKAGE_VERSION_PATCH "0")
endif()

if (NOT DEFINED CPACK_PACKAGE_CONTACT)
    set(CPACK_PACKAGE_CONTACT "Advanced Micro Devices Inc.")
endif()

if (NOT DEFINED CPACK_PACKAGE_DESCRIPTION_SUMMARY)
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY
                 "ROCm System Management Interface library")
endif()

set(LIB_VERSION_STRING
        "${CPACK_PACKAGE_VERSION_MAJOR}.${CPACK_PACKAGE_VERSION_MINOR}.${CPACK_PACKAGE_VERSION_PATCH}")
# Debian package specific variables

set(BUILD_VERSION_STRING
        "${CPACK_PACKAGE_VERSION_MAJOR}.${CPACK_PACKAGE_VERSION_MAJOR}.${CPACK_PACKAGE_VERSION_PATCH}")

set(CPACK_PACKAGE_FILE_NAME "rocm_smi_lib64-${LIB_VERSION_STRING}")
## Verbose output.
set(CMAKE_VERBOSE_MAKEFILE on)

## Compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -fno-rtti -m64")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse -msse2 -std=c++11 ")
# Use this instead of above for 32 bit
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -m32")

if ("${CMAKE_BUILD_TYPE}" STREQUAL Release)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2")
else ()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ggdb -O0 -DDEBUG")
endif ()

set(SRC_DIR "src")
set(INC_DIR "include/rocm_smi")

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
set(SMI_SRC_LIST "${SRC_DIR}/rocm_smi_device.cc")
set(SMI_SRC_LIST ${SMI_SRC_LIST} "${SRC_DIR}/rocm_smi_main.cc")
set(SMI_SRC_LIST ${SMI_SRC_LIST} "${SRC_DIR}/rocm_smi_monitor.cc")
set(SMI_SRC_LIST ${SMI_SRC_LIST} "${SRC_DIR}/rocm_smi.cc")
set(SMI_SRC_LIST ${SMI_SRC_LIST} "${SRC_DIR}/rocm_smi_power_mon.cc")
set(SMI_SRC_LIST ${SMI_SRC_LIST} "${SRC_DIR}/rocm_smi_utils.cc")

set(SMI_INC_LIST "${INC_DIR}/rocm_smi_device.h")
set(SMI_INC_LIST ${SMI_INC_LIST} "${INC_DIR}/rocm_smi_main.h")
set(SMI_INC_LIST ${SMI_INC_LIST} "${INC_DIR}/rocm_smi_monitor.h")
set(SMI_INC_LIST ${SMI_INC_LIST} "${INC_DIR}/rocm_smi_power_mon.h")
set(SMI_INC_LIST ${SMI_INC_LIST} "${INC_DIR}/rocm_smi_utils.h")
set(SMI_INC_LIST ${SMI_INC_LIST} "${INC_DIR}/rocm_smi_common.h")


# rocm_smi_device.h 
#      rocm_smi.h rocm_smi_main.h rocm_smi_monitor.h")

set(SMI_EXAMPLE_EXE "rocm_smi_ex")

add_executable(${SMI_EXAMPLE_EXE} "example/rocm_smi_example.cc")
target_link_libraries(${SMI_EXAMPLE_EXE} ${ROCM_SMI_TARGET})
add_library(${ROCM_SMI_TARGET} SHARED ${SMI_SRC_LIST} ${SMI_INC_LIST})

## Set the VERSION and SOVERSION values
set_property(TARGET ${ROCM_SMI_TARGET} PROPERTY VERSION "${LIB_VERSION_STRING}")
set_property(TARGET ${ROCM_SMI_TARGET}
             PROPERTY SOVERSION "${CPACK_PACKAGE_VERSION_MAJOR}")


## If the library is a release, strip the target library
if ("${CMAKE_BUILD_TYPE}" STREQUAL Release)
    add_custom_command(
          TARGET ${ROCM_SMI_TARGET}
          POST_BUILD COMMAND ${CMAKE_STRIP} lib${ROCM_SMI_TARGET}.so)
endif ()

## Define default variable and variables for the optional build target
##  rocm_smi_lib-dev
set(SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}
                              CACHE STRING "Location of rocm_smi source code.")
set(CMAKE_INSTALL_PREFIX "/opt/rocm"
                                CACHE STRING "Default installation directory.")
set(CPACK_PACKAGING_INSTALL_PREFIX "/opt/rocm"
                                      CACHE STRING "Default packaging prefix.")
set(CPACK_GENERATOR "DEB;RPM"  CACHE STRING "Default packaging generators.")

## Add the install directives for the runtime library.
install(TARGETS ${ROCM_SMI_TARGET}
          LIBRARY DESTINATION ${ROCM_SMI}/lib COMPONENT ${ROCM_SMI_COMPONENT})
install(FILES ${SOURCE_DIR}/include/rocm_smi/rocm_smi.h DESTINATION rocm_smi/include/rocm_smi)

## Add the packaging directives for the runtime library.

# TODO set(CPACK_DEBIAN_PACKAGE_HOMEPAGE <GITHUB URL> )
set(CPACK_DEBIAN_PACKAGE_CONTROL_EXTRA
    "${CMAKE_CURRENT_SOURCE_DIR}/DEBIAN/postinst;
                                    ${CMAKE_CURRENT_SOURCE_DIR}/DEBIAN/prerm")

# RPM package specific variables
set(CPACK_RPM_PRE_INSTALL_SCRIPT_FILE
                                   "${CMAKE_CURRENT_SOURCE_DIR}/RPM/rpm_post")
set(CPACK_RPM_POST_UNINSTALL_SCRIPT_FILE
                                 "${CMAKE_CURRENT_SOURCE_DIR}/RPM/rpm_postun")
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


