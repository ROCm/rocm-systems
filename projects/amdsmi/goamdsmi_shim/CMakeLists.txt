# SPDX-License-Identifier: MIT
# Copyright (c) 2024, Advanced Micro Devices, Inc.

#
# Minimum version of cmake required
#
cmake_minimum_required(VERSION 3.5.0)

message("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
message("                  CMake AMD goamdsmi_shim Library                  ")
message("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")

set(AMDSMI_DIR "" CACHE PATH "path to amdsmi installation")

if(ENABLE_DEBUG_LEVEL)
add_definitions(-DENABLE_DEBUG_LEVEL=${ENABLE_DEBUG_LEVEL})
message("**** Enabling Debug Level=${ENABLE_DEBUG_LEVEL} ****")
else()
add_definitions(-DENABLE_DEBUG_LEVEL=0)
endif()

set(GOAMDSMI_SHIM "goamdsmi_shim")
set(GOAMDSMI_SHIM_LIB "goamdsmi")
set(GOAMDSMI_SHIM_COMPONENT "lib${GOAMDSMI_SHIM}")
set(GOAMDSMI_SHIM_TARGET "${GOAMDSMI_SHIM}64")

# The following default version values should be updated as appropriate for
# ABI breaks (update MAJOR and MINOR), and ABI/API additions (update MINOR).
# Until ABI stabilizes VERSION_MAJOR will be 0. This should be over-ridden
# by git tags (through "git describe") when they are present.
set(VERSION_MAJOR 1)
set(VERSION_MINOR 0)
set(VERSION_PATCH 0)
set(VERSION_NUM_COMMIT 0)

set(SO_VERSION_STRING "${VERSION_MAJOR}.${VERSION_MINOR}")

set(${GOAMDSMI_SHIM}_VERSION_MAJOR "${VERSION_MAJOR}")
set(${GOAMDSMI_SHIM}_VERSION_MINOR "${VERSION_MINOR}")
set(${GOAMDSMI_SHIM}_VERSION_PATCH "0")
set(${GOAMDSMI_SHIM}_VERSION_BUILD "0")
message("SOVERSION: ${SO_VERSION_STRING}")

project(${GOAMDSMI_SHIM_TARGET})

if (NOT DEFINED CPACK_PACKAGE_VENDOR)
    set(CPACK_PACKAGE_VENDOR "AMD")
endif()

if (NOT DEFINED CPACK_PACKAGE_CONTACT)
    set(CPACK_PACKAGE_CONTACT "Advanced Micro Devices Inc.")
endif()

if (NOT DEFINED CPACK_PACKAGE_DESCRIPTION_SUMMARY)
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY
	"AMD CGO wrapper")
endif()

if (NOT GOAMDSMI_SHIM_PACKAGE)
	set(GOAMDSMI_SHIM_PACKAGE goamdsmi_shim_lib64)
endif()

set(CPACK_PACKAGE_FILE_NAME "${GOAMDSMI_SHIM_PACKAGE}-${SO_VERSION_STRING}")
## Verbose output.
set(CMAKE_VERBOSE_MAKEFILE on)

## Compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -fpic -fno-rtti -m64")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse -msse2 -std=c++11 ")
# Use this instead of above for 32 bit
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -m32")

if ("${CMAKE_BUILD_TYPE}" STREQUAL Release)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2")
else ()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ggdb -O0 -DDEBUG")
endif ()

set(go_amd_smi_headers)

set(go_amd_smi_sources)

set(go_amd_smi_includes)


add_subdirectory(smiwrapper)
list(APPEND go_amd_smi_headers smiwrapper/goamdsmi.h ${go_amd_smi_headers})
list(APPEND go_amd_smi_headers smiwrapper/amdsmi_go_shim.h ${go_amd_smi_headers})
list(APPEND go_amd_smi_sources smiwrapper/amdsmi_go_shim.c)
list(APPEND go_amd_smi_includes ${CMAKE_CURRENT_SOURCE_DIR}/smiwrapper)

add_library(${GOAMDSMI_SHIM_TARGET} SHARED
	    ${go_amd_smi_sources}
	    ${go_amd_smi_headers}
	    ${go_amd_smi_includes})

target_link_libraries(${GOAMDSMI_SHIM_TARGET} pthread rt m)

target_link_libraries(${GOAMDSMI_SHIM_TARGET} amd_smi)
target_link_libraries(${GOAMDSMI_SHIM_TARGET} -L${AMDSMI_DIR}/lib)
target_link_libraries(${GOAMDSMI_SHIM_TARGET} -L${AMDSMI_DIR}/lib64)

## Set the VERSION and SOVERSION values
set_property(TARGET ${GOAMDSMI_SHIM_TARGET} PROPERTY
                SOVERSION "${VERSION_MAJOR}")
set_property(TARGET ${GOAMDSMI_SHIM_TARGET} PROPERTY
                VERSION "${SO_VERSION_STRING}")

## If the library is a release, strip the target library
if ("${CMAKE_BUILD_TYPE}" STREQUAL Release)
    add_custom_command(
	  TARGET ${GOAMDSMI_SHIM_TARGET}
	  POST_BUILD COMMAND ${CMAKE_STRIP} lib${GOAMDSMI_SHIM_TARGET}.so)
endif ()

set(go_amd_smi_install_headers
    smiwrapper/goamdsmi.h
    smiwrapper/amdsmi_go_shim.h
)

## Add the install directives for the runtime library.
install(TARGETS ${GOAMDSMI_SHIM_TARGET}
	LIBRARY DESTINATION lib COMPONENT ${GOAMDSMI_SHIM_COMPONENT})
install(FILES ${go_amd_smi_install_headers} 
	DESTINATION include)

include_directories(${go_amd_smi_includes})
