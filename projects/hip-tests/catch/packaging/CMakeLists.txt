
# Copyright (c) 2016 - 2023 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
cmake_minimum_required(VERSION 3.16.8)
# vc 19.31.31107.0 gives issue while packaging using makefile
# To avoid error NMAKE : fatal error U1065: invalid option 'w'
# Windows to use Ninja generator like other projects
# to skip the simple compiler test
set(CMAKE_C_COMPILER_WORKS 1)
set(CMAKE_CXX_COMPILER_WORKS 1)
include(GNUInstallDirs)

#############################
# Packaging steps
#############################
if(NOT DEFINED CPACK_INSTALL_PREFIX)
  set(CPACK_INSTALL_PREFIX "/opt/rocm/")
endif()
if(BUILD_SHARED_LIBS)
  set(PKG_NAME hip-catch-${HIP_PLATFORM})
else()
  set(PKG_NAME hip-catch-${HIP_PLATFORM}-static)
endif()
set(CPACK_PACKAGE_NAME ${PKG_NAME})
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "HIP: Heterogenous-computing Interface for Portability [CATCH TESTS]")
set(CPACK_PACKAGE_DESCRIPTION "HIP:
          Heterogenous-computing Interface for Portability [CATCH TESTS]")
set(CPACK_PACKAGE_VENDOR "Advanced Micro Devices, Inc.")
set(CPACK_PACKAGE_CONTACT "HIP Support <hip.support@amd.com>")
set(CPACK_PACKAGE_VERSION ${HIP_VERSION_MAJOR}.${HIP_VERSION_MINOR}.${HIP_PACKAGING_VERSION_PATCH})
# by default rpm tries to reduce the file size by stripping the executable
# To avoid build/packaging failures SWDEV-375603 and skip stripping
set(CPACK_RPM_SPEC_MORE_DEFINE "%global __os_install_post %{nil}")

set (CPACK_RPM_PACKAGE_AUTOREQ 0)
# to remove hip-catch-* package during uninstallation of rocm

if(BUILD_SHARED_LIBS)
  set (CPACK_DEBIAN_PACKAGE_DEPENDS "rocm-core, hip-runtime-amd")
  set (CPACK_RPM_PACKAGE_REQUIRES "rocm-core, hip-runtime-amd")
else()
  set (CPACK_DEBIAN_PACKAGE_DEPENDS "rocm-core, hip-static-dev")
  set (CPACK_RPM_PACKAGE_REQUIRES "rocm-core, hip-static-devel")
endif()

# Get all the files that needs to be installed in the global property
get_property(INSTALL_CTEST_INCLUDE_FILES GLOBAL PROPERTY G_INSTALL_CTEST_INCLUDE_FILES)
get_property(INSTALL_EXE_TARGETS GLOBAL PROPERTY G_INSTALL_EXE_TARGETS)
get_property(INSTALL_CUSTOM_TARGETS GLOBAL PROPERTY G_INSTALL_CUSTOM_TARGETS)
get_property(INSTALL_SRC_FILES GLOBAL PROPERTY G_INSTALL_SRC_FILES)
get_property(INSTALL_HEADER_FILES GLOBAL PROPERTY G_INSTALL_HEADER_FILES)
# Create top level CTestTestfile.cmake
file(WRITE ${PROJECT_BINARY_DIR}/${CATCH_BUILD_DIR}/script/CTestTestfile.cmake ${_autogen})
foreach(CTEST_INCLUDE_FILE ${INSTALL_CTEST_INCLUDE_FILES})
  get_filename_component(INCLUDE_FILE_NAME ${CTEST_INCLUDE_FILE} NAME)
  file(APPEND ${PROJECT_BINARY_DIR}/${CATCH_BUILD_DIR}/script/CTestTestfile.cmake
       "include(script/${INCLUDE_FILE_NAME})\n")
endforeach()
file(WRITE ${PROJECT_BINARY_DIR}/${CATCH_BUILD_DIR}/script/windows/CTestTestfile.cmake
     "${_autogen}\nsubdirs(${CATCH_BUILD_DIR})")

if(NOT WIN32)
set(INSTALL_DIR ${CMAKE_INSTALL_DATADIR}/hip)
set(CPACK_GENERATOR "TGZ;DEB;RPM" CACHE STRING "Linux package types for catch tests")
set(CPACK_BINARY_DEB "ON")
set(CPACK_DEBIAN_FILE_NAME "DEB-DEFAULT")
set(CPACK_DEBIAN_PACKAGE_PROVIDES "catch")
set(CPACK_BINARY_RPM "ON")
set(CPACK_RPM_FILE_NAME "RPM-DEFAULT")
set(CPACK_RPM_EXCLUDE_FROM_AUTO_FILELIST_ADDITION "/opt")
if (CPACK_PACKAGE_VERSION MATCHES "local" )
  #If building locally default value will cause build failure
  #DEBUG SYMBOL pacaking require SOURCE_DIR to be small
  set(CPACK_RPM_BUILD_SOURCE_DIRS_PREFIX ${CPACK_INSTALL_PREFIX})
endif()

if (DEFINED ENV{CPACK_DEBIAN_PACKAGE_RELEASE})
  set(CPACK_DEBIAN_PACKAGE_RELEASE $ENV{CPACK_DEBIAN_PACKAGE_RELEASE})
else()
  set(CPACK_DEBIAN_PACKAGE_RELEASE "local")
endif()

if(DEFINED ENV{CPACK_RPM_PACKAGE_RELEASE})
  set(CPACK_RPM_PACKAGE_RELEASE $ENV{CPACK_RPM_PACKAGE_RELEASE})
else()
  set(CPACK_RPM_PACKAGE_RELEASE "local")
endif()

execute_process( COMMAND rpm --eval %{?dist}
                 RESULT_VARIABLE PROC_RESULT
                 OUTPUT_VARIABLE EVAL_RESULT
                 OUTPUT_STRIP_TRAILING_WHITESPACE )
# Add os distribution tag to rpm package name . For deb package its set from build env
if ( PROC_RESULT EQUAL "0" AND NOT EVAL_RESULT STREQUAL "" )
  string(APPEND CPACK_RPM_PACKAGE_RELEASE "%{?dist}")
endif()

set(CPACK_SOURCE_GENERATOR "TGZ")
# Install license file

set ( CPACK_RESOURCE_FILE_LICENSE "${CMAKE_SOURCE_DIR}/../LICENSE.md" )
install(FILES ${CPACK_RESOURCE_FILE_LICENSE} DESTINATION ${INSTALL_DIR})
set(CPACK_RPM_PACKAGE_LICENSE "MIT")

else()
# windows packaging
set(INSTALL_DIR .)
set(CPACK_INSTALL_PREFIX "")
set(CPACK_SYSTEM_NAME "")
set(CPACK_GENERATOR "ZIP" CACHE STRING "Windows package types for catch tests")
set(CPACK_TEST_ZIP "ON")
set(CPACK_ZIP_TEST_PACKAGE_NAME "catch")
endif()

set(INSTALL_DIR_TESTS ${INSTALL_DIR}/${CATCH_BUILD_DIR})
set(INSTALL_DIR_SCRIPT ${INSTALL_DIR}/${CATCH_BUILD_DIR}/script)
set(INSTALL_DIR_SRC ${INSTALL_DIR}/${CATCH_BUILD_DIR}/src)
set(INSTALL_DIR_HEADERS ${INSTALL_DIR}/${CATCH_BUILD_DIR}/headers)
# install catch scripts
install(FILES
        ${ADD_SCRIPT_PATH}
        ${CATCH_INCLUDE_PATH}
        DESTINATION ${INSTALL_DIR_SCRIPT})
# install cmake include files
foreach(CTEST_INCLUDE_FILE ${INSTALL_CTEST_INCLUDE_FILES})
  install(FILES ${CTEST_INCLUDE_FILE} DESTINATION ${INSTALL_DIR_SCRIPT})
endforeach()
# install source files
foreach(SRC_FILE ${INSTALL_SRC_FILES})
  install(FILES ${SRC_FILE} DESTINATION ${INSTALL_DIR_SRC})
endforeach()
# install header files
foreach(HEADER_FILES ${INSTALL_HEADER_FILES})
  install(FILES ${HEADER_FILES} DESTINATION ${INSTALL_DIR_HEADERS})
endforeach()
# install executables
foreach(EXE_TARGET ${INSTALL_EXE_TARGETS})
  install(TARGETS ${EXE_TARGET}
          RUNTIME DESTINATION ${INSTALL_DIR_TESTS} OPTIONAL)
endforeach()
# install custom targets
foreach(CUSTOM_TARGET ${INSTALL_CUSTOM_TARGETS})
  install(FILES ${CUSTOM_TARGET} DESTINATION ${INSTALL_DIR_TESTS})
endforeach()
# install ctest file
install(FILES
        ${PROJECT_BINARY_DIR}/${CATCH_BUILD_DIR}/script/CTestTestfile.cmake
        DESTINATION ${INSTALL_DIR_TESTS})
install(FILES
        ${PROJECT_BINARY_DIR}/${CATCH_BUILD_DIR}/script/windows/CTestTestfile.cmake
        DESTINATION ${INSTALL_DIR})
# install catchinfo
install(FILES ${PROJECT_BINARY_DIR}/${CATCH_BUILD_DIR}/catchInfo.txt DESTINATION ${INSTALL_DIR})
# install hip test config
install(DIRECTORY
        ${HIP_TEST_CONFIG_BINARY_DIR}
        DESTINATION ${INSTALL_DIR_TESTS})
include(CPack)
