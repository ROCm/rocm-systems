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

project(amdcuid_daemon)

set(CMAKE_CXX_STANDARD 14)

set(INC_DIR ${CMAKE_SOURCE_DIR}/lib)
set(AMDCUID_DAEMON_EXECUTABLE amdcuid_daemon)
set(SRC_LIST amdcuid_daemon.cc)

find_package(Threads REQUIRED)

# Add the tool executable
add_executable(amdcuid_daemon amdcuid_daemon.cc)

# Include directories
target_include_directories(${AMDCUID_DAEMON_EXECUTABLE} PRIVATE ${INC_DIR})

# Pass the resolved config install directory to the daemon at compile time so
# the HMAC key and daemon configuration paths are always built from a concrete
# location.
target_compile_definitions(
    ${AMDCUID_DAEMON_EXECUTABLE}
    PRIVATE AMDCUID_CONFIG_DIR="${CMAKE_INSTALL_SYSCONFDIR}"
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

# udev rules for Linux; TODO: need to figure out Windows service/driver registration for device event handling
if(UNIX AND NOT APPLE)
    # if conf file states daemon mode, install udev rules to trigger daemon on device events
    # Read default from config file
    file(READ ${CMAKE_CURRENT_SOURCE_DIR}/amdcuid_daemon.conf CONFIG_CONTENTS)
    string(REGEX MATCH "daemonize=([a-zA-Z]+)" _ "${CONFIG_CONTENTS}")
    set(CONFIG_DAEMONIZE_DEFAULT "${CMAKE_MATCH_1}")

    # Create CMake option with config file default
    option(
        ENABLE_DAEMONIZE
        "Enable daemon mode with udev rules"
        ${CONFIG_DAEMONIZE_DEFAULT}
    )

    if(ENABLE_DAEMONIZE)
        file(
            WRITE ${CMAKE_CURRENT_BINARY_DIR}/90-amdcuid.rules
            "# AMD CUID daemon udev rules
        SUBSYSTEM==\"drm\", ACTION==\"add|remove|change\", RUN+=\"${CMAKE_INSTALL_PREFIX}/bin/amdcuid_tool --notify-daemon\"
        SUBSYSTEM==\"net\", ACTION==\"add|remove|change\", RUN+=\"${CMAKE_INSTALL_PREFIX}/bin/amdcuid_tool --notify-daemon\"
        "
        )
        install(
            FILES ${CMAKE_CURRENT_BINARY_DIR}/90-amdcuid.rules
            DESTINATION /etc/udev/rules.d
            COMPONENT daemon
        )

        # Custom target to reload rules (must run as root)
        add_custom_target(
            reload-udev
            COMMAND udevadm control --reload-rules
            COMMAND udevadm trigger
            COMMENT "Reloading udev rules..."
            VERBATIM
        )
    else()
        # if not in daemon mode, create a cron job instead to start daemon at reboot
        file(
            WRITE ${CMAKE_CURRENT_BINARY_DIR}/amdcuid_cron
            "@reboot root ${CMAKE_INSTALL_PREFIX}/bin/amdcuid_daemon"
        )
        install(
            FILES ${CMAKE_CURRENT_BINARY_DIR}/amdcuid_cron
            DESTINATION /etc/cron.d
            COMPONENT daemon
        )
    endif()
endif()
