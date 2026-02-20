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

cmake_minimum_required(VERSION 3.10)
project(amdcuid_daemon)

set(CMAKE_CXX_STANDARD 14)

# Find OpenSSL
find_package(OpenSSL REQUIRED)

# Add the tool executable
add_executable(amdcuid_daemon amdcuid_daemon.cc)

# Link against the CUID library
target_link_libraries(amdcuid_daemon amdcuid_static OpenSSL::Crypto)

# Include directories
target_include_directories(amdcuid_daemon PRIVATE ${CMAKE_SOURCE_DIR}/lib)

# Install the tool
install(TARGETS amdcuid_daemon DESTINATION bin COMPONENT daemon)

# Platform-specific config directory
if(WIN32)
    set(CONFIG_INSTALL_DIR "etc")  # Relative to prefix
else()
    set(CONFIG_INSTALL_DIR "${CMAKE_INSTALL_SYSCONFDIR}")
endif()

install(FILES amdcuid_daemon.conf DESTINATION ${CONFIG_INSTALL_DIR} COMPONENT daemon)

# udev rules for Linux; TODO: need to figure out Windows service/driver registration for device event handling
if(UNIX AND NOT APPLE)
    # if conf file states daemon mode, install udev rules to trigger daemon on device events
    # Read default from config file
    file(READ ${CMAKE_CURRENT_SOURCE_DIR}/amdcuid_daemon.conf CONFIG_CONTENTS)
    string(REGEX MATCH "daemonize=([a-zA-Z]+)" _ "${CONFIG_CONTENTS}")
    set(CONFIG_DAEMONIZE_DEFAULT "${CMAKE_MATCH_1}")

    # Create CMake option with config file default
    option(ENABLE_DAEMONIZE "Enable daemon mode with udev rules" ${CONFIG_DAEMONIZE_DEFAULT})

    if (ENABLE_DAEMONIZE)
        file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/90-amdcuid.rules
        "# AMD CUID daemon udev rules
        SUBSYSTEM==\"drm\", ACTION==\"add|remove|change\", RUN+=\"${CMAKE_INSTALL_PREFIX}/bin/amdcuid_tool --notify-daemon\"
        SUBSYSTEM==\"net\", ACTION==\"add|remove|change\", RUN+=\"${CMAKE_INSTALL_PREFIX}/bin/amdcuid_tool --notify-daemon\"
        ")
        install(FILES ${CMAKE_CURRENT_BINARY_DIR}/90-amdcuid.rules
        DESTINATION /etc/udev/rules.d
        COMPONENT daemon)

        # Custom target to reload rules (must run as root)
        add_custom_target(reload-udev
        COMMAND udevadm control --reload-rules
        COMMAND udevadm trigger
        COMMENT "Reloading udev rules..."
        VERBATIM)
    else()
        # if not in daemon mode, create a cron job instead to start daemon at reboot
        file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/amdcuid_cron
        "@reboot root ${CMAKE_INSTALL_PREFIX}/bin/amdcuid_daemon" )
        install(FILES ${CMAKE_CURRENT_BINARY_DIR}/amdcuid_cron
        DESTINATION /etc/cron.d
        COMPONENT daemon)
    endif()
endif()