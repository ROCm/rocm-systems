#!/bin/bash

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

# Install script for amdcuid project

# create directories
mkdir -p /opt/cuid/bin
mkdir -p /opt/cuid/lib
mkdir -p /opt/cuid/include
mkdir -p /opt/cuid/tests

# copy files, these will likely change on actual inclusion in rocm-systems package
cp ../daemon/amdcuid_daemon.conf /opt/cuid/amdcuid_daemon.conf
cp ../build/daemon/amdcuid_daemon /opt/cuid/bin/amdcuid_daemon
cp ../build/lib/libamdcuid_shared.so /opt/cuid/lib/libamdcuid_shared.so
cp ../lib/include/amd_cuid.h /opt/cuid/include/amd_cuid.h
cp ../build/cli/amdcuid_tool /opt/cuid/bin/amdcuid_tool
cp ../build/tests/amdcuid_test /opt/cuid/tests/amdcuid_test

# if config files specifies daemon mode, set up udev rules
DAEMONIZE=$(grep "daemonize" /opt/cuid/amdcuid_daemon.conf | cut -d'=' -f2)
if [ "$DAEMONIZE" = "true" ]; then
    echo "Setting up udev rules for amdcuid daemon..."
    # create udev rule file and copy to /etc/udev/rules.d
    touch 90-amdcuid_daemon.rules
    # add rules for drm and net subsystem device events, this can change once we actually implement the functionality in the CLI tool
    echo 'SUBSYSTEM=="drm", ACTION=="add|remove|change", RUN+="/opt/cuid/bin/amdcuid_tool --notify-daemon"' >> 90-amdcuid_daemon.rules
    echo 'SUBSYSTEM=="net", ACTION=="add|remove|change", RUN+="/opt/cuid/bin/amdcuid_tool --notify-daemon"' >> 90-amdcuid_daemon.rules
    cp 90-amdcuid_daemon.rules /etc/udev/rules.d/90-amdcuid_daemon.rules
    sudo udevadm control --reload-rules
    sudo udevadm trigger
    sudo systemctl restart systemd-udevd
else
    echo "Daemon mode not enabled; creating boot up cron job for amdcuid daemon..."
    # create cron job to run daemon on boot up
    (crontab -l 2>/dev/null; echo "@reboot /opt/cuid/amdcuid_daemon") | crontab -
fi