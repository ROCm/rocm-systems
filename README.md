# AMD ROCDXG Libary
A user-mode library that enables ROCm functionality on Windows Subsystem for Linux (WSL). This library allows users to run GPU-accelerated Linux workloads under WSL, supporting AI, HPC, and other experimental use cases.

## Prerequisites
- Download the compatible Windows driver from [AMD Drivers](https://www.amd.com/en/support/download/drivers.html)
- Download and install the latest stable version of WSL2 [WSL Install](https://learn.microsoft.com/en-us/windows/wsl/install)
- The following tools are required to build librocdxg:
  - CMake >= 3.15
  - GCC >= 11.4

## Quickstart

### 1. Install Windows SDK

Download and install the Windows SDK from [windows SDK](https://developer.microsoft.com/en-us/windows/downloads/windows-sdk/)

### 2. Install AMD ROCm package

#### Install AMD Unified Driver Package Repositories and Installer Script

Select the applicable Ubuntu® version to download and install the amdgpu-install script on the system.

#### Ubuntu® 24.04

Enter the following commands to install the installer script for Ubuntu® version 24.04:

```bash
sudo apt update
wget https://repo.radeon.com/amdgpu-install/7.1.0/ubuntu/noble/amdgpu-install_7.1.0.xxxxx-1_all.deb
sudo apt install ./amdgpu-install_7.1.0.xxxxx-1_all.deb
```

#### Install AMD ROCm package

Run the installer script with appropriate ***--usecase*** parameters to install the components once the Unified Driver Deb Package repositories are installed.

#### Set up ROCm usecase
The ***--no-dkms*** parameter must be passed, as amdgpu kernel driver is not required within WSL2.

Run the following command to install ROCm:

```bash
amdgpu-install -y --usecase=rocm --no-dkms
```

> ***Note***
> The ***-y*** option installs non-interactively. This step may take several minutes, depending on internet connection and system speed.
> Look out for output warning or errors that indicate an unsuccessful installation.

See [Using the amdgpu-install script](https://amdgpu-install.readthedocs.io/en/latest/install-script.html) for more information.

### 3. Build librocdxg
Run the following commands in your WSL console:
 
1. Clone librocdxg repository to your local WSL.

```bash
git clone https://github.com/[...]/librocdxg.git
cd librocdxg
```

2. Verify that ROCm has been successfully installed.

```bash
tree -L 1 /opt
```

Expected result:

```bash
/opt/
├── [...]
├── rocm -> /etc/alternatives/rocm
├── rocm-7.1.0
└── [...]
```

3. Build the librocdxg.

```bash
# Set the Windows SDK path (adjust version number if different)
export win_sdk='/mnt/c/Program Files (x86)/Windows Kits/10/Include/10.0.26100.0/'
 
# Build the library
mkdir -p build
cd build
cmake .. -DWIN_SDK="${win_sdk}/shared"
make
sudo make install
```

> ***Note***
> - The Windows SDK path may vary depending on the version you installed. Common locations include:
>   - C:\Program Files (x86)\Windows Kits\10\Include\10.0.26100.0\
> - Ensure you have the necessary permissions to access the Windows SDK directory from WSL

### 4. Load the AMD ROCDXG libary

Set the environment variable HSA_ENABLE_DXG_DETECTION=1 to load librocdxg.so.

```bash
export HSA_ENABLE_DXG_DETECTION=1
```

### 5. Post-install verification checks
Run these post-installation checks to verify that the installation is complete.

Check if the GPU is listed as an agent:

```bash
rocminfo
```

Expected result:

```bash
[...]
*******
Agent 2
*******
  Name:                    gfx1100
  Marketing Name:          Radeon RX 7900 XTX
  Vendor Name:             AMD
  [...]
[...]

```

See [Installing the all open use case](https://amdgpu-install.readthedocs.io/en/latest/install-installing.html#installing-the-all-open-use-case) for additional troubleshooting tips.

## WSL Compatiblity Matrix
- Windows 11
- Ubuntu 24.04 LTS / Ubuntu 22.04 LTS
- The AMD ROCDXG library utilizes a ROCm runtime feature introduced in ROCm 7.1, which loads ***librocdxg*** to enable ROCm functionality within the WSL environment. This design keeps the ***librocdxg*** solution loosely coupled with both AMD ROCm release and Windows display driver. As a result, the AMD ROCDXG library can evolve independently, following its own development schedule without impacting the existing ROCm solution.

| AMD Rocdxg Lib Version | AMD ROCm Version | AMD Windows Driver Version | Supported AMD GPU Products |
| ---------------------- | ---------------- | -------------------------- | -------------------------- |
| 1.0.0                  | 7.1              | AMD Windows x86 drivers can be directly downloaded from [AMD Driver](https://www.amd.com/en/support/download/drivers.html) | ***Radeon***<br><br>AMD Radeon RX 9070<br>AMD Radeon RX 9070 XT<br>AMD Radeon RX 9070 GRE<br>AMD Radeon AI PRO R9700<br>AMD Radeon RX 9060<br>AMD Radeon RX 9060 XT<br>AMD Radeon RX 7900 XTX<br>AMD Radeon RX 7900 XT<br>AMD Radeon RX 7900 GRE<br>AMD Radeon PRO W7900<br>AMD Radeon PRO W7900 Dual Slot<br>AMD Radeon PRO W7800<br>AMD Radeon PRO W7800 48GB<br>AMD Radeon RX 7800 XT<br>AMD Radeon PRO W7700<br><br>***Ryzen***<br><br>AMD Ryzen AI Max+ 395<br>AMD Ryzen AI Max 390<br>AMD Ryzen AI Max 385<br>AMD Ryzen AI 9 HX 375<br>AMD Ryzen AI 9 HX 370<br>AMD Ryzen AI 9 365 |

## Documentation
For detailed documentation including installation guides, configuration options, and metric descriptions, please refer to [Use ROCm on Radeon and Ryzen](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/index.html#)