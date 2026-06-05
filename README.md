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

please install the ROCm package by following the official ROCm Installation Quick Guide:

[ROCm Installation Quick Start](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/quick-start.html)

> ***Note***
> - The ROCm 7.13.0 technology preview release documentation is available at [ROCm Preview documentation](https://rocm.docs.amd.com/en/7.13.0-preview/). For production use, continue to use ROCm 7.2.3 documentation.
> - This step may take several minutes, depending on internet connection and system speed.
> - Follow the quick-start guide for package repository setup and ROCm package installation.
> - **Important**: Post-installation validation of ROCm (Step 5) must only be performed after the successful completion of **Step 3** and **Step 4**. Executing the validation prior to this will lead to failure.

### 3. Build librocdxg
Run the following commands in your WSL console:

1. Clone librocdxg repository to your local WSL.

```bash
git clone https://github.com/ROCm/librocdxg.git
cd librocdxg
```

2. Build the librocdxg.

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

For legacy ROCm releases, HSA_ENABLE_DXG_DETECTION=1 MUST be set; this requirement is removed starting with the ROCk release 7.13.

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

### 6. Container Launch – WSL-Specific Flags

When you launch the container, add these WSL-specific arguments (they do not replace the native-Linux GPU flags):

| Flag | Purpose |
| ---- | ------- |
| `--device /dev/dxg` | Pass the `/dev/dxg` device node into the container so applications inside the container can access the GPU. |
| `-v /usr/lib/wsl/lib/libdxcore.so:/usr/lib/libdxcore.so`<br>`-v /opt/rocm/lib/librocdxg.so:/usr/lib/librocdxg.so`<br>`-v /opt/rocm/share/rocdxg/dids.conf:/usr/share/rocdxg/dids.conf` | Make the AMD ROCDXG and Microsoft DXCore libraries available inside the container so that ROCm/HIP applications can route their GPU compute calls through ROCDXG and DXCore to communicate with the GPU. |
| `-e HSA_ENABLE_DXG_DETECTION=1` | For legacy ROCm releases, HSA_ENABLE_DXG_DETECTION=1 MUST be set; this requirement is removed starting with the ROCk release 7.13. |

Example docker run command:

```bash
docker run -it  \
    -v /usr/lib/wsl/lib/libdxcore.so:/usr/lib/libdxcore.so \
    -v /opt/rocm/lib/librocdxg.so:/usr/lib/librocdxg.so \
    -v /opt/rocm/share/rocdxg/dids.conf:/usr/share/rocdxg/dids.conf \
    --device=/dev/dxg \
    --cap-add=SYS_PTRACE \
    --security-opt seccomp=unconfined \
    --ipc=host \
    --shm-size 8G \
    rocm/pytorch:latest
```

> ***Note***
> - For ROCm releases prior to 7.13, pass `-e HSA_ENABLE_DXG_DETECTION=1` to the `docker run` command:
>
>   ```bash
>   docker run -it  \
>       -v /usr/lib/wsl/lib/libdxcore.so:/usr/lib/libdxcore.so \
>       -v /opt/rocm/lib/librocdxg.so:/usr/lib/librocdxg.so \
>       -v /opt/rocm/share/rocdxg/dids.conf:/usr/share/rocdxg/dids.conf \
>       -e HSA_ENABLE_DXG_DETECTION=1 \
>       --device=/dev/dxg \
>       --cap-add=SYS_PTRACE \
>       --security-opt seccomp=unconfined \
>       --ipc=host \
>       --shm-size 8G \
>       rocm/pytorch:latest
>   ```

## 7. Known Issues and Limitations

- JAX is supported from version 0.9.1 onwards.
- AMD-SMI currently provides a limited set of features on WSL2. The source code is available in the develop branch, and a formal release plan is under development.
- Debugging/Profiling: `ROCm-profiler`, `Debugger` are not supported.

## WSL Compatibility Matrix
- Windows 11
- Ubuntu 26.04 LTS / Ubuntu 24.04 LTS / Ubuntu 22.04 LTS
- The AMD ROCDXG library utilizes a ROCm runtime feature introduced in ROCm 7.1, which loads ***librocdxg*** to enable ROCm functionality within the WSL environment. This design keeps the ***librocdxg*** solution loosely coupled with both AMD ROCm release and Windows display driver. As a result, the AMD ROCDXG library can evolve independently, following its own development schedule without impacting the existing ROCm solution.

| AMD Rocdxg Lib Version | AMD ROCm Version | AMD Windows Driver Version | Supported AMD GPU Products |
| ---------------------- | ---------------- | -------------------------- | -------------------------- |
| 1.2.0                  | 7.13             | AMD Windows x86 drivers<br>can be directly downloaded<br>from [AMD Driver](https://www.amd.com/en/support/download/drivers.html) | Additional AMD GPU support includes all ASICs for previous versions, plus the following:<br><br>***Ryzen***<br><br>AMD Ryzen AI 7 PRO 360<br>AMD Ryzen AI 7 PRO 350<br>AMD Ryzen AI 5 PRO 340<br>AMD Ryzen AI 7 350<br>AMD Ryzen AI 7 345<br>AMD Ryzen AI 5 340<br>AMD Ryzen AI 5 330 |
| 1.2.0                  | 7.2.x            | AMD Windows x86 drivers<br>can be directly downloaded<br>from [AMD Driver](https://www.amd.com/en/support/download/drivers.html) | ***Radeon***<br><br>AMD Radeon RX 9070<br>AMD Radeon RX 9070 XT<br>AMD Radeon RX 9070 GRE<br>AMD Radeon AI PRO R9700<br>AMD Radeon RX 9060<br>AMD Radeon RX 9060 XT<br>AMD Radeon RX 7900 XTX<br>AMD Radeon RX 7900 XT<br>AMD Radeon RX 7900 GRE<br>AMD Radeon PRO W7900<br>AMD Radeon PRO W7900 Dual Slot<br>AMD Radeon PRO W7800<br>AMD Radeon PRO W7800 48GB<br>AMD Radeon RX 7800 XT<br>AMD Radeon PRO W7700<br><br>***Ryzen***<br><br>AMD Ryzen AI Max+ 395<br>AMD Ryzen AI Max 390<br>AMD Ryzen AI Max 385<br>AMD Ryzen AI 9 HX 375<br>AMD Ryzen AI 9 HX 370<br>AMD Ryzen AI 9 365 |


## Documentation

For detailed documentation—including ROCm installation guides, configuration options, and metric descriptions—see "[Use ROCm on Radeon and Ryzen](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/index.html#)".

## Contributing
See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on setting up your WSL environment, building, and submitting pull requests.