# Copyright (c) 2024 - 2026 Advanced Micro Devices, Inc. All rights reserved.
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

import os
import sys
import argparse
import platform
import traceback
if sys.version_info[0] < 3:
    import commands
else:
    import subprocess

__copyright__ = "Copyright (c) 2024 - 2026, AMD ROCm rocJPEG"
__version__ = "2.4.0"
__email__ = "mivisionx.support@amd.com"
__status__ = "Shipping"

# error check calls
def ERROR_CHECK(waitval):
    if(waitval != 0): # return code and signal flags
        print('ERROR_CHECK failed with status:'+str(waitval))
        traceback.print_stack()
        status = ((waitval >> 8) | waitval) & 255 # combine exit code and wait flags into single non-zero byte
        exit(status)

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--rocm_path', 	type=str, default='/opt/rocm',
                    help='ROCm Installation Path - optional (default:/opt/rocm) - ROCm Installation Required')
parser.add_argument('--runtime', 	type=str, default='ON',
                    help='Install RunTime Dependencies - optional (default:ON) [options:ON/OFF]')

args = parser.parse_args()
runtimeInstall = args.runtime.upper()

ROCM_PATH = args.rocm_path

if "ROCM_PATH" in os.environ:
    ROCM_PATH = os.environ.get('ROCM_PATH')
print("\nROCm PATH set to -- "+ROCM_PATH+"\n")

# check ROCm installation
if os.path.exists(ROCM_PATH):
    print("\nROCm Installation Found -- "+ROCM_PATH+"\n")
    os.system('echo ROCm Info -- && '+ROCM_PATH+'/bin/rocminfo')
else:
    print(
        "WARNING: If ROCm installed, set ROCm Path with \"--rocm_path\" option for full installation [Default:/opt/rocm]\n")
    print("ERROR: rocJPEG Setup requires ROCm install\n")
    exit(-1)

if runtimeInstall not in ('OFF', 'ON'):
    print(
        "ERROR: Runtime Option Not Supported - [Supported Options: OFF or ON]\n")
    exit()

# get platfrom info
platfromInfo = platform.platform()

# sudo requirement check
sudoLocation = ''
userName = ''
if sys.version_info[0] < 3:
    status, sudoLocation = commands.getstatusoutput("which sudo")
    if sudoLocation != '/usr/bin/sudo':
        status, userName = commands.getstatusoutput("whoami")
else:
    status, sudoLocation = subprocess.getstatusoutput("which sudo")
    if sudoLocation != '/usr/bin/sudo':
        status, userName = subprocess.getstatusoutput("whoami")

# check os version
os_info_data = 'NOT Supported'
if os.path.exists('/etc/os-release'):
    with open('/etc/os-release', 'r') as os_file:
        os_info_data = os_file.read().replace('\n', ' ')
        os_info_data = os_info_data.replace('"', '')

# setup for Linux
linuxSystemInstall = ''
linuxCMake = 'cmake'
linuxSystemInstall_check = ''
linuxFlag = ''
sudoValidateOption= '-v'
osUpdate = ''
if "centos" in os_info_data or "redhat" in os_info_data:
    linuxSystemInstall = 'yum -y'
    linuxSystemInstall_check = '--nogpgcheck'
    osUpdate = 'makecache'
    if "VERSION_ID=7" in os_info_data:
        linuxCMake = 'cmake3'
        platfromInfo = platfromInfo+'-redhat-7'
    elif "VERSION_ID=8" in os_info_data:
        platfromInfo = platfromInfo+'-redhat-8'
    elif "VERSION_ID=9" in os_info_data:
        platfromInfo = platfromInfo+'-redhat-9'
    else:
        platfromInfo = platfromInfo+'-redhat-centos-undefined-version'
elif "Ubuntu" in os_info_data:
    linuxSystemInstall = 'apt-get -y'
    linuxSystemInstall_check = '--allow-unauthenticated'
    linuxFlag = '-S'
    osUpdate = 'update'
    if "VERSION_ID=20" in os_info_data:
        platfromInfo = platfromInfo+'-Ubuntu-20'
    elif "VERSION_ID=22" in os_info_data:
        platfromInfo = platfromInfo+'-Ubuntu-22'
    elif "VERSION_ID=24" in os_info_data:
        platfromInfo = platfromInfo+'-Ubuntu-24'
    else:
        platfromInfo = platfromInfo+'-Ubuntu-undefined-version'
elif "SLES" in os_info_data:
    linuxSystemInstall = 'zypper -n'
    linuxSystemInstall_check = '--no-gpg-checks'
    platfromInfo = platfromInfo+'-SLES'
    osUpdate = 'refresh'
elif "Mariner" in os_info_data:
    linuxSystemInstall = 'tdnf -y'
    linuxSystemInstall_check = '--nogpgcheck'
    platfromInfo = platfromInfo+'-Mariner'
    runtimeInstall = 'OFF'
    osUpdate = 'makecache'
else:
    print("\nrocJPEG Setup on "+platfromInfo+" is unsupported\n")
    print("\nrocJPEG Setup Supported on: Ubuntu 20/22/24, RedHat 8/9, & SLES 15\n")
    exit(-1)

# rocJPEG Setup
print("\nrocJPEG Setup on: "+platfromInfo+"\n")
print("\nrocJPEG Dependencies Installation with rocJPEG-setup.py V-"+__version__+"\n")

if userName == 'root':
    ERROR_CHECK(os.system(linuxSystemInstall+' '+osUpdate))
    ERROR_CHECK(os.system(linuxSystemInstall+' install sudo'))

# source install - common package dependencies
commonPackages = [
    'cmake',
    'hip-runtime-amd'
]

# Debian packages
coreDebianPackages = [
    'libva-dev',
    'hip-dev'
]
coreDebianU22Packages = [
    'libva-amdgpu-dev',
    'hip-dev',
    'libstdc++-12-dev'
]
runtimeDebianPackages = [
    'libva-drm2',
    'mesa-amdgpu-va-drivers',
    'vainfo'
]
runtimeDebianU22Packages = [
    'libva-amdgpu-drm2',
    'mesa-amdgpu-va-drivers',
    'vainfo'
]

# RPM Packages
if "centos" in os_info_data or "redhat" in os_info_data:
    coreRPMPackages = [
        'libva-devel',
        'hip-devel'
    ]
    runtimeRPMPackages = [
        'libva',
        'mesa-amdgpu-va-drivers',
        'libva-utils'
    ]
else:
    coreRPMPackages = [
        'libva-devel',
        'hip-devel'
    ]
    runtimeRPMPackages = [
        'libva-drm2',
        'mesa-amdgpu-va-drivers',
        'libva-utils'
    ]

# update
ERROR_CHECK(os.system('sudo '+linuxFlag+' '+linuxSystemInstall +' '+linuxSystemInstall_check+' '+osUpdate))

# common packages
ERROR_CHECK(os.system('sudo '+sudoValidateOption))
for i in range(len(commonPackages)):
    ERROR_CHECK(os.system('sudo '+linuxFlag+' '+linuxSystemInstall +
            ' '+linuxSystemInstall_check+' install '+ commonPackages[i]))

# rocJPEG Core - Requirements
ERROR_CHECK(os.system('sudo '+sudoValidateOption))
if "Ubuntu" in platfromInfo:
    if "VERSION_ID=22" in os_info_data:
        for i in range(len(coreDebianU22Packages)):
            ERROR_CHECK(os.system('sudo '+linuxFlag+' '+linuxSystemInstall +
                    ' '+linuxSystemInstall_check+' install '+ coreDebianU22Packages[i]))
    else:
        for i in range(len(coreDebianPackages)):
            ERROR_CHECK(os.system('sudo '+linuxFlag+' '+linuxSystemInstall +
                    ' '+linuxSystemInstall_check+' install '+ coreDebianPackages[i]))
else:
    for i in range(len(coreRPMPackages)):
        ERROR_CHECK(os.system('sudo '+linuxFlag+' '+linuxSystemInstall +
                ' '+linuxSystemInstall_check+' install '+ coreRPMPackages[i]))

# rocJPEG runTime - Requirements
ERROR_CHECK(os.system('sudo '+sudoValidateOption))
if runtimeInstall == 'ON':
    if "Ubuntu" in platfromInfo:
        if "VERSION_ID=22" in os_info_data:
            for i in range(len(runtimeDebianU22Packages)):
                ERROR_CHECK(os.system('sudo '+linuxFlag+' '+linuxSystemInstall +
                    ' '+linuxSystemInstall_check+' install '+ runtimeDebianU22Packages[i]))
        else:
            for i in range(len(runtimeDebianPackages)):
                ERROR_CHECK(os.system('sudo '+linuxFlag+' '+linuxSystemInstall +
                    ' '+linuxSystemInstall_check+' install '+ runtimeDebianPackages[i]))
    else:
        for i in range(len(runtimeRPMPackages)):
            ERROR_CHECK(os.system('sudo '+linuxFlag+' '+linuxSystemInstall +
                ' '+linuxSystemInstall_check+' install '+ runtimeRPMPackages[i]))

print("\nrocJPEG Dependencies Installed with rocJPEG-setup.py V-"+__version__+"\n")
