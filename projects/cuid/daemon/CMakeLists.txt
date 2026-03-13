# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
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
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

message("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
message("                           Cmake amdcuid_daemon                           ")
message("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")

message("")
message("Build Configuration:")
message("-----------BuildType: " ${CMAKE_BUILD_TYPE})
message("------------Compiler: " ${CMAKE_CXX_COMPILER})
message("-------------Version: " ${CMAKE_CXX_COMPILER_VERSION})
message("------Install Prefix: " ${CMAKE_INSTALL_PREFIX})
message("-Pkg.-Install Prefix: " ${CPACK_PACKAGING_INSTALL_PREFIX})
message("-CMake inst. Bindir : " ${CMAKE_INSTALL_BINDIR})
message("--------Proj Src Dir: " ${PROJECT_SOURCE_DIR})
message("--------Proj Bld Dir: " ${PROJECT_BINARY_DIR})
message("--------Proj Lib Dir: " ${PROJECT_BINARY_DIR}/lib)
message("--------Proj Exe Dir: " ${PROJECT_BINARY_DIR}/bin)
message("")

set(INC_DIR ${CMAKE_SOURCE_DIR}/lib)
set(AMDCUID_DAEMON_EXECUTABLE amdcuid_daemon)
set(SRC_LIST amdcuid_daemon.cc)

# Add the daemon executable
add_executable(${AMDCUID_DAEMON_EXECUTABLE} ${SRC_LIST})

# Link against the CUID library
target_link_libraries(${AMDCUID_DAEMON_EXECUTABLE} amdcuid_static OpenSSL::Crypto)

# Include directories
target_include_directories(${AMDCUID_DAEMON_EXECUTABLE} PRIVATE ${INC_DIR})
if(WIN32)
    set(CONFIG_INSTALL_DIR "CONFIG")  # Relative to prefix
else()
    set(CONFIG_INSTALL_DIR "${CMAKE_INSTALL_DATADIR}/${AMDCUID}/config")
endif()

# Pass install paths to the daemon at compile time
target_compile_definitions(${AMDCUID_DAEMON_EXECUTABLE} PRIVATE
    AMDCUID_CONFIG_DIR="${CMAKE_INSTALL_PREFIX}/${CONFIG_INSTALL_DIR}")

# Install the daemon
install(
    TARGETS ${AMDCUID_DAEMON_EXECUTABLE}
    PERMISSIONS
        OWNER_EXECUTE
        OWNER_READ
        OWNER_WRITE
        GROUP_READ
        GROUP_EXECUTE
        WORLD_READ
        WORLD_EXECUTE
    DESTINATION bin
    COMPONENT ${DAEMON_COMPONENT})

set(DAEMON_CONFIG_FILE amdcuid_daemon.conf)

configure_file(${DAEMON_CONFIG_FILE} ${CONFIG_INSTALL_DIR}/${DAEMON_CONFIG_FILE} COPYONLY)

install(
    FILES ${DAEMON_CONFIG_FILE}
    DESTINATION ${CONFIG_INSTALL_DIR}
    COMPONENT ${DAEMON_COMPONENT}
)

# Post-installation script for udev rules / cron job setup.
# The choice between udev (daemonize=true) and cron (daemonize=false) is made
# at install time by reading the deployed config file, so it correctly reflects
# any user edits made after the build.
if(UNIX AND NOT APPLE)
    # Generate the post-install shell script from a template, substituting the
    # install prefix and config directory baked-in at cmake configure time.
    configure_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/postinst.in
        ${CMAKE_CURRENT_BINARY_DIR}/postinst
        @ONLY)

    # Ensure the generated script is executable (required by DEB packaging).
    execute_process(COMMAND chmod +x ${CMAKE_CURRENT_BINARY_DIR}/postinst)

    # Hook into CPack DEB packages (postinst is run by dpkg after installation).
    set(CPACK_DEBIAN_PACKAGE_CONTROL_EXTRA
        "${CMAKE_CURRENT_BINARY_DIR}/postinst"
        PARENT_SCOPE)

    # Hook into CPack RPM packages (%post scriptlet).
    set(CPACK_RPM_POST_INSTALL_SCRIPT_FILE
        "${CMAKE_CURRENT_BINARY_DIR}/postinst"
        PARENT_SCOPE)

    # Also install the script so it can be re-run manually after a cmake --install.
    install(
        PROGRAMS ${CMAKE_CURRENT_BINARY_DIR}/postinst
        DESTINATION ${CMAKE_INSTALL_DATADIR}/${AMDCUID}
        RENAME amdcuid_postinst.sh
        COMPONENT ${DAEMON_COMPONENT})
endif()

message("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
message("                   Finished Cmake amdcuid_daemon                   ")
message("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")