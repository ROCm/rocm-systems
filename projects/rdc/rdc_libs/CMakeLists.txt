# Copyright (c) 2020 - present Advanced Micro Devices, Inc. All rights reserved.
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

message("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
message("                       Cmake RDC Lib                               ")
message("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")

# Required Defines first:

message("")
message("Build Configuration:")
message("-----------BuildType: " ${CMAKE_BUILD_TYPE})
message("------------Compiler: " ${CMAKE_CXX_COMPILER})
message("-------------Version: " ${CMAKE_CXX_COMPILER_VERSION})
message("--------Proj Src Dir: " ${PROJECT_SOURCE_DIR})
message("--------Proj Bld Dir: " ${PROJECT_BINARY_DIR})
message("--------Proj Lib Dir: " ${PROJECT_BINARY_DIR}/lib)
message("--------Proj Exe Dir: " ${PROJECT_BINARY_DIR}/bin)
message("---------SMI Lib Dir: " ${AMD_SMI_LIB_DIR})
message("---------SMI Inc Dir: " ${AMD_SMI_INCLUDE_DIR})
message("")

set(COMMON_DIR "${PROJECT_SOURCE_DIR}/common")
set(SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(INC_DIR "${PROJECT_SOURCE_DIR}/include/rdc_lib")
set(RDC_LIB_INC_DIR "${INC_DIR}")

set(BOOTSTRAP_LIB "rdc_bootstrap")
set(RDC_LIB "rdc")
set(RDC_ROCR_LIB "rdc_rocr")
set(RDC_ROCP_LIB "rdc_rocp")
set(RDC_RVS_LIB "rdc_rvs")
set(RDCCLIENT_LIB "rdc_client")

################# Determine the library version #########################
## Setup the SO version based on git tags.
set(SO_VERSION_GIT_TAG_PREFIX "rdc_so_ver")

# provide git to utilities
find_program(GIT NAMES git)

# Debian package specific variables
# Set a default value for the package version
get_version_from_tag("1.1.0.0" ${SO_VERSION_GIT_TAG_PREFIX} GIT)

# VERSION_* variables should be set by get_version_from_tag
set(SO_VERSION_STRING "${VERSION_MAJOR}.${VERSION_MINOR}")
message("SOVERSION: ${SO_VERSION_STRING}")

set(CPACK_PACKAGE_FILE_NAME "${RDC_PACKAGE}-${VERSION_STRING}")
set(CPACK_DEBIAN_PACKAGE_DEPENDS "rocm-core")
set(CPACK_RPM_PACKAGE_REQUIRES "rocm-core")

# link SMI
link_directories(${AMD_SMI_LIB_DIR})

# add librdc_bootstrap.so
add_subdirectory(bootstrap)

# add librdc.so
add_subdirectory(rdc)

# add librdc_rocr.so to RDC_LIB_MODULES
add_subdirectory(rdc_modules/rdc_rocr)

# add librdc_rocp.so to RDC_LIB_MODULES
add_subdirectory(rdc_modules/rdc_rocp)

# add librdc_rvs.so to RDC_LIB_MODULES
add_subdirectory(rdc_modules/rdc_rvs)

# Create cmake target
# Add all targets to the build-tree export set
export(TARGETS ${BOOTSTRAP_LIB} ${RDC_LIB} FILE "${PROJECT_BINARY_DIR}/rdc_libs.cmake")

## Add the install directives for the runtime library.
install(
    TARGETS ${BOOTSTRAP_LIB} ${RDC_LIB}
    EXPORT rdcTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT ${CLIENT_COMPONENT}
)

if(BUILD_STANDALONE)
    # add librdc_client.so
    add_subdirectory(rdc_client)
    install(
        TARGETS ${RDCCLIENT_LIB}
        EXPORT rdcTargets
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT ${CLIENT_COMPONENT}
    )
    export(TARGETS ${RDCCLIENT_LIB} APPEND FILE "${PROJECT_BINARY_DIR}/rdc_libs.cmake")
endif()

# Add module directives if those exist
if(RDC_LIB_MODULES)
    install(
        TARGETS ${RDC_LIB_MODULES}
        EXPORT rdcTargets
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}/${RDC} COMPONENT ${CLIENT_COMPONENT}
    )
    set_target_properties(${RDC_LIB_MODULES} PROPERTIES INSTALL_RPATH "\$ORIGIN:\$ORIGIN/..")
endif()

install(
    FILES ${PROJECT_SOURCE_DIR}/include/rdc/rdc.h
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${RDC}
    COMPONENT ${CLIENT_COMPONENT}
)

# Don't print kernel install because it floods the terminal
set(OLD_CMAKE_INSTALL_MESSAGE ${CMAKE_INSTALL_MESSAGE})
set(CMAKE_INSTALL_MESSAGE NEVER)
# Install the kernel files
install(
    DIRECTORY ${SRC_DIR}/rdc_modules/kernels/hsaco
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/${RDC}
    COMPONENT ${CLIENT_COMPONENT}
)
# Restore printing verbosity
set(CMAKE_INSTALL_MESSAGE ${OLD_CMAKE_INSTALL_MESSAGE})
unset(OLD_CMAKE_INSTALL_MESSAGE)

message("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
message("                    Finished Cmake RDC Lib                      ")
message("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
