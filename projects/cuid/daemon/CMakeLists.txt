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

message(
    "&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&"
)
message(
    "                           Cmake amdcuid_daemon                           "
)
message(
    "&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&"
)

message("")
message("Build Configuration:")
message("-----------BuildType: " ${CMAKE_BUILD_TYPE})
message("------------Compiler: " ${CMAKE_CXX_COMPILER})
message("-------------Version: " ${CMAKE_CXX_COMPILER_VERSION})
message("------Install Prefix: " ${CMAKE_INSTALL_PREFIX})
message("-CMake inst. Bindir : " ${CMAKE_INSTALL_BINDIR})
message("--------Proj Src Dir: " ${PROJECT_SOURCE_DIR})
message("--------Proj Bld Dir: " ${PROJECT_BINARY_DIR})
message("--------Proj Lib Dir: " ${PROJECT_BINARY_DIR}/lib)
message("--------Proj Exe Dir: " ${PROJECT_BINARY_DIR}/bin)
message("")

set(INC_DIR ${CMAKE_SOURCE_DIR}/lib)
set(AMDCUID_DAEMON_EXECUTABLE amdcuid_daemon)
set(SRC_LIST amdcuid_daemon.cc)
if(WIN32)
    set(CONFIG_INSTALL_DIR "CONFIG") # Relative to prefix
else()
    set(CONFIG_INSTALL_DIR "${CMAKE_INSTALL_DATADIR}/${AMDCUID}/config")
endif()

find_package(Threads REQUIRED)

# Add the daemon executable
add_executable(${AMDCUID_DAEMON_EXECUTABLE} ${SRC_LIST})

# Include directories
target_include_directories(${AMDCUID_DAEMON_EXECUTABLE} PRIVATE ${INC_DIR})

# Pass the resolved config install directory to the daemon at compile time so
# the HMAC key and daemon configuration paths are always built from a concrete
# location.
target_compile_definitions(
    ${AMDCUID_DAEMON_EXECUTABLE}
    PRIVATE AMDCUID_CONFIG_DIR="${AMDCUID_SHARED_CONFIG_DIR}"
)

# Link against the CUID library
target_link_libraries(
    ${AMDCUID_DAEMON_EXECUTABLE}
    amdcuid_static
    Threads::Threads
)

# Install the daemon
install(
    TARGETS ${AMDCUID_DAEMON_EXECUTABLE}
    DESTINATION bin
    COMPONENT ${DAEMON_COMPONENT}
)

set(DAEMON_CONFIG_FILE amdcuid_daemon.conf)
install(
    FILES ${CMAKE_CURRENT_SOURCE_DIR}/${DAEMON_CONFIG_FILE}
    DESTINATION ${CMAKE_INSTALL_SYSCONFDIR}
    COMPONENT ${DAEMON_COMPONENT}
)

install(
    FILES ${DAEMON_CONFIG_FILE}
    DESTINATION ${CONFIG_INSTALL_DIR}
    COMPONENT ${DAEMON_COMPONENT}
)

# Install the systemd unit file.
# The unit file itself is a static file — deploying it is an install-step
# concern.  Enabling/starting the service is done in the post-install script
# because it requires running systemctl against the live system.
if(UNIX AND NOT APPLE)
    # Allow callers to override the systemd unit directory.  The default is
    # the conventional system-wide path that systemd always scans, which is
    # intentionally outside CMAKE_INSTALL_PREFIX because systemd does not
    # discover units inside arbitrary prefixes (e.g. /opt/rocm/core) without
    # extra configuration.

    configure_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/amdcuid_daemon.service.in
        ${CMAKE_CURRENT_BINARY_DIR}/amdcuid_daemon.service
        @ONLY
    )

    install(
        FILES ${CMAKE_CURRENT_BINARY_DIR}/amdcuid_daemon.service
        DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/${AMDCUID}
        COMPONENT ${DAEMON_COMPONENT}
    )

    # Post-installation script for udev rules / service registration.
    # The choice between a persistent service with udev rules (daemonize=true)
    # or a one-shot boot run (daemonize=false) is made at install time by
    # reading the deployed config file. To apply subsequent config changes,
    # edit the config file then rerun the post-install script to have the udev
    # rules and service symlink updated to match the new config and restart the
    # service using `sudo systemctl restart amdcuid_daemon`.

    # Generate the post-install shell script from a template, substituting the
    # install prefix and config directory baked-in at cmake configure time.
    configure_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/postinst.in
        ${CMAKE_CURRENT_BINARY_DIR}/postinst
        @ONLY
    )

    # Configure the pre-removal script (mirrors postinst: stops service, removes
    # udev rules, cron job, and systemd service symlink).
    configure_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/prerm.in
        ${CMAKE_CURRENT_BINARY_DIR}/prerm
        @ONLY
    )

    # Also install both scripts so they can be re-run manually after a cmake --install.
    install(
        PROGRAMS ${CMAKE_CURRENT_BINARY_DIR}/postinst
        DESTINATION ${CMAKE_INSTALL_DATADIR}/${AMDCUID}
        RENAME amdcuid_postinst.sh
        COMPONENT ${DAEMON_COMPONENT}
    )

    install(
        PROGRAMS ${CMAKE_CURRENT_BINARY_DIR}/prerm
        DESTINATION ${CMAKE_INSTALL_DATADIR}/${AMDCUID}
        RENAME amdcuid_prerm.sh
        COMPONENT ${DAEMON_COMPONENT}
    )
endif()

message("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
message("                   Finished Cmake amdcuid_daemon                   ")
message("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
