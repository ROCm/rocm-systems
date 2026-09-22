################################################################################
##
## The University of Illinois/NCSA
## Open Source License (NCSA)
##
## Copyright (c) 2014-2017, Advanced Micro Devices, Inc. All rights reserved.
##
## Developed by:
##
##                 AMD Research and AMD HSA Software Development
##
##                 Advanced Micro Devices, Inc.
##
##                 www.amd.com
##
## Permission is hereby granted, free of charge, to any person obtaining a copy
## of this software and associated documentation files (the "Software"), to
## deal with the Software without restriction, including without limitation
## the rights to use, copy, modify, merge, publish, distribute, sublicense,
## and#or sell copies of the Software, and to permit persons to whom the
## Software is furnished to do so, subject to the following conditions:
##
##  - Redistributions of source code must retain the above copyright notice,
##    this list of conditions and the following disclaimers.
##  - Redistributions in binary form must reproduce the above copyright
##    notice, this list of conditions and the following disclaimers in
##    the documentation and#or other materials provided with the distribution.
##  - Neither the names of Advanced Micro Devices, Inc,
##    nor the names of its contributors may be used to endorse or promote
##    products derived from this Software without specific prior written
##    permission.
##
## THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
## IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
## FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
## THE CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
## OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
## ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
## DEALINGS WITH THE SOFTWARE.
##
################################################################################

cmake_minimum_required ( VERSION 3.5.0 )

## Set the name and project name.
set ( PROJECT_STRING hsa-ext-rocr-dev )
project ( ${PROJECT_STRING} )

add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../../hsa-ext-image" "../hsa-ext-image")

## Include the cmake_modules utils.cmake
list ( APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../cmake_modules" )
include ( utils )

## Get the package version.
get_version ( "1.1.9" )
if ( ${ROCM_PATCH_VERSION} )
    set ( VERSION_PATCH ${ROCM_PATCH_VERSION})
endif ()

set( PACKAGE_VERSION_STRING "${VERSION_MAJOR}.${VERSION_MINOR}.${VERSION_PATCH}.${VERSION_COMMIT_COUNT}-${VERSION_JOB}-${VERSION_HASH}" )

## Packaging directives
set ( CPACK_GENERATOR "DEB;RPM" CACHE STRING "Package types to build")

set ( CPACK_PACKAGE_NAME ${PROJECT_NAME} )
set ( CPACK_PACKAGE_VENDOR "Advanced Micro Devices, Inc." )
set ( CPACK_PACKAGE_VERSION ${PACKAGE_VERSION_STRING} )
set ( CPACK_PACKAGE_CONTACT "Advanced Micro Devices, Inc." )
set ( CPACK_PACKAGE_DESCRIPTION_SUMMARY "AMD Heterogeneous System Architecture HSA - Linux HSA Runtime extensions for ROCm platforms" )
set ( CPACK_PACKAGE_DESCRIPTION_FILE "${CMAKE_CURRENT_SOURCE_DIR}/description" )
set ( CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/copyright" )

# Debian package specific variables
set ( CPACK_DEBIAN_FILE_NAME "DEB-DEFAULT" )
set ( CPACK_DEBIAN_PACKAGE_DEPENDS "hsakmt-roct, hsa-rocr-dev" )
set ( CPACK_DEBIAN_PACKAGE_HOMEPAGE "https://github.com/RadeonOpenCompute/ROCR-Runtime" )
set ( CPACK_DEBIAN_PACKAGE_CONTROL_EXTRA "${CMAKE_CURRENT_SOURCE_DIR}/postinst;${CMAKE_CURRENT_SOURCE_DIR}/prerm" )

# RPM package specific variables
set ( CPACK_RPM_FILE_NAME "RPM-DEFAULT" )
set ( CPACK_RPM_PACKAGE_DEPENDS "hsakmt-roct, hsa-rocr-dev" )
set ( CPACK_RPM_POST_INSTALL_SCRIPT_FILE "${CMAKE_CURRENT_SOURCE_DIR}/rpm_post" )
set ( CPACK_RPM_POST_UNINSTALL_SCRIPT_FILE "${CMAKE_CURRENT_SOURCE_DIR}/rpm_postun" )

include ( CPack )
