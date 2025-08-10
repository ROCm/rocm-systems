#!/bin/bash
# Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
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

function download_and_extract_deb()
{
    # Download 6.4.2 hip runtime deb package
    wget -O hip-runtime-amd.deb $1

    # Extract libamdhip64.so.6.4.60402
    dpkg-deb --fsys-tarfile hip-runtime-amd.deb | tar xf - ./opt/rocm-6.4.2/lib/libamdhip64.so.6.4.60402

    # Rename to libamdhip64.so.6
    mv ./opt/rocm-6.4.2/lib/libamdhip64.so.6.4.60402 libamdhip64.so.6

    # Clean up
    rm -r ./opt; rm hip-runtime-amd.deb
}

function download_and_extract_rpm()
{
    # Download 6.4.2 hip runtime rpm package
    wget -O hip-runtime-amd.rpm $1

    # Extract libamdhip64.so.6.4.60402
    rpm2cpio hip-runtime-amd.rpm | cpio -idm

    # Rename to libamdhip64.so.6
    mv ./opt/rocm-6.4.2/lib/libamdhip64.so.6.4.60402 libamdhip64.so.6

    # Clean up
    rm -r ./opt; rm hip-runtime-amd.rpm
}

# Detect OS
ID=$(sed -n 's/^ID=//p' /etc/os-release)
VERSION_ID=$(sed -n 's/^VERSION_ID=//p' /etc/os-release)
PLATFORM_ID=$(sed -n 's/^PLATFORM_ID=//p' /etc/os-release)
echo "ID=$ID"
echo "VERSION_ID=$VERSION_ID"
echo "PLATFORM_ID=$PLATFORM_ID"

# Download and extract libamdhip64.so.6.4.60402 based on OS
case "$ID" in
    ubuntu)
        # Ubuntu
        if [ "$VERSION_ID" == "\"22.04\"" ]; then
            # Ubuntu 22.04
            download_link="https://repo.radeon.com/rocm/apt/6.4.2/pool/main/h/hip-runtime-amd/hip-runtime-amd_6.4.43484.60402-120~22.04_amd64.deb"
            download_and_extract_deb $download_link
        elif [ "$VERSION_ID" == "\"24.04\"" ]; then
            # Ubuntu 24.04
            download_link="https://repo.radeon.com/rocm/apt/6.4.2/pool/main/h/hip-runtime-amd/hip-runtime-amd_6.4.43484.60402-120~24.04_amd64.deb"
            download_and_extract_deb $download_link
        else
            # Unknown Ubuntu version
            echo "Unknown Ubuntu OS"
        fi
        ;;
    azurelinux)
        # AzureLinux
        if [ "$VERSION_ID" == "\"3.0\"" ]; then
            # AzureLinux 3
            download_link="https://repo.radeon.com/rocm/azurelinux3/6.4.2/main/hip-runtime-amd-6.4.43484.60402-108.azl3.x86_64.rpm"
            download_and_extract_rpm $download_link
        else
            # Unknown AzureLinux version
            echo "Unknown AzureLinux"
        fi
        ;;
    \"rhel\")
        # RHEL
        if [ "$PLATFORM_ID" == "\"platform:el9\"" ]; then
            # RHEL 9
            download_link="https://repo.radeon.com/rocm/rhel9/6.4.2/main/hip-runtime-amd6.4.2-6.4.43484.60402-120.el9.x86_64.rpm"
            download_and_extract_rpm $download_link
        else
            # Unknown RHEL version
            echo "Unknown RHEL OS"
        fi
        ;;
    \"sles\")
        # SLES
        if [ "$VERSION_ID" == "\"15.6\"" ]; then
            # SLES 15SP6
            download_link="https://repo.radeon.com/rocm/zyp/6.4.2/main/hip-runtime-amd-6.4.43484.60402-sles156.120.x86_64.rpm"
            download_and_extract_rpm $download_link
        else
            # Unknown SLES version
            echo "Unknown SLES OS"
        fi
        ;;
    *)
        echo "Unknown OS"
        ;;
esac
