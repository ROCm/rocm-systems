[![MIT licensed](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)

<p align="center"><img width="70%" src="docs/data/AMD_rocDecode_Logo.png" /></p>

rocDecode is a high-performance video decode SDK for AMD GPUs. Using the rocDecode API, you can
access the video decoding features available on your GPU.

> [!NOTE]
> The published documentation is available at [rocDecode](https://rocm.docs.amd.com/projects/rocDecode/en/latest/index.html) in an organized, easy-to-read format, with search and a table of contents. The documentation source files reside in the `rocDecode/docs` folder of this repository. As with all ROCm projects, the documentation is open source. For more information on contributing to the documentation, see [Contribute to ROCm documentation](https://rocm.docs.amd.com/en/latest/contribute/contributing.html).

## Supported codecs
* H.265 (HEVC) - 8 bit, and 10 bit
* H.264 (AVC) - 8 bit
* AV1 - 8 bit, and 10 bit
* VP9 - 8 bit, and 10 bit

## Prerequisites

### Operating Systems
* Linux
  * Ubuntu - `22.04` / `24.04`
  * RedHat - `8` / `9`
  * SLES - `15-SP5`

### Hardware
* **GPU**: [AMD Radeon&trade; Graphics](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html) / [AMD Instinct&trade; Accelerators](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html)

> [!IMPORTANT] 
> * `gfx908` or higher GPU required

* Install ROCm `6.3.0` or later with `amdgpu-install` and usecase:`rocm`
> [!IMPORTANT]
> `sudo amdgpu-install --usecase=rocm`

### Compiler
* AMD Clang++ Version 18.0.0 or later - installed with ROCm

### Libraries
* CMake Version `3.10` or later
  ```shell
  sudo apt install cmake
  ```

* [Video Acceleration API](https://en.wikipedia.org/wiki/Video_Acceleration_API) - `libva-amdgpu-dev` is an AMD implementation for VA-API
  ```shell
  sudo apt install libva-amdgpu-dev
  ```
> [!NOTE]
> * RPM Packages for `RHEL`/`SLES` - `libva-amdgpu-devel`
> * `libva-amdgpu` is strongly recommended over system `libva` as it is used for building mesa-amdgpu-va-driver

* AMD VA Drivers
  ```shell
  sudo apt install libva2-amdgpu libva-amdgpu-drm2 libva-amdgpu-wayland2 libva-amdgpu-x11-2 mesa-amdgpu-va-drivers
  ```
> [!NOTE]
> RPM Packages for `RHEL`/`SLES` - `libva-amdgpu mesa-amdgpu-va-drivers`

* HIP
  ```shell
  sudo apt install hip-dev
  ```

* [pkg-config](https://en.wikipedia.org/wiki/Pkg-config)
  ```shell
  sudo apt install pkg-config
  ```

* [FFmpeg](https://github.com/FFmpeg/FFmpeg) libraries and headers:
  * `libavcodec` - provides implementation of a wider range of codecs.
  * `libavformat` - implements streaming protocols, container formats and basic I/O access.
  * `libavutil` - includes hashers, decompressors and miscellaneous utility functions.

> [!NOTE]
> FFmpeg libraries are used in samples and tests

  ```shell
  sudo apt install libavcodec-dev libavformat-dev libavutil-dev
  ```

> [!IMPORTANT]
> * Required compiler support
>   * C++17
>   * Threads
> * On Ubuntu 22.04 - Additional package required: libstdc++-12-dev
>  ```shell
>  sudo apt install libstdc++-12-dev
>  ```

>[!NOTE]
> * All package installs are shown with the `apt` package manager. Use the appropriate package manager for your operating system.

### Prerequisites setup script

For your convenience, we provide the setup script, [rocDecode-setup.py](https://github.com/ROCm/rocDecode/blob/develop/rocDecode-setup.py), which installs all required dependencies. Run this script only once.

```shell
python3 rocDecode-setup.py  --rocm_path [ ROCm Installation Path  - optional (default:/opt/rocm)]
                            --runtime   [ Setup runtime requirements - optional (default:ON) [options:ON/OFF]]
                            --developer [ Setup Developer Options - optional (default:OFF) [options:ON/OFF]]
```

## Installation instructions

The installation process uses the following steps:

* [ROCm-supported hardware](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html) install verification

* Install ROCm `6.3.0` or later with `amdgpu-install` and `--usecase=rocm`:

  ```
  sudo amdgpu-install --usecase=rocm
  ```

>[!IMPORTANT]
> Use **either** [package install](#package-install) **or** [source install](#source-install) as described below.

### Package install

To install rocDecode runtime, development, and test packages, run the line of code for your operating
system.

* Runtime package - `rocdecode` only provides the rocdecode library `librocdecode.so`
* Development package - `rocdecode-dev`/`rocdecode-devel` provides the library, header files, and samples
* Host package - `rocdecode-host` provides rocdecode host decode
* Test package - `rocdecode-test` provides CTest to verify installation

#### Ubuntu

  ```shell
  sudo apt install rocdecode rocdecode-dev rocdecode-test rocdecode-host
  ```

#### RHEL

  ```shell
  sudo yum install rocdecode rocdecode-devel rocdecode-test rocdecode-host
  ```

#### SLES

  ```shell
  sudo zypper install rocdecode rocdecode-devel rocdecode-test rocdecode-host
  ```

>[!NOTE]
> * Package install auto installs all dependencies.
> * `rocdecode-host` package is optional package use in host video decode

> [!IMPORTANT] 
> `RHEL`/`SLES` package install requires manual `FFMPEG` dev install

### Source install

To build rocDecode from source and install, run:

```shell
git clone https://github.com/ROCm/rocDecode.git
cd rocDecode
python3 rocDecode-setup.py
mkdir build && cd build
cmake ../
make -j8
sudo make install
```

#### Run tests

  ```shell
  make test
  ```
  > [!IMPORTANT] 
  > make test requires `FFMPEG` dev install

  >[!NOTE]
  > To run tests with verbose option, use `make test ARGS="-VV"`.

#### Make package

  ```shell
  sudo make package
  ```

## Verify installation

The installer copies:

* Libraries into `/opt/rocm/lib`
* Header files into `/opt/rocm/include/rocdecode`
* Samples folder into `/opt/rocm/share/rocdecode`
* Documents folder into `/opt/rocm/share/doc/rocdecode`

### Using sample application

To verify your installation using a sample application, run:

  ```shell
  mkdir rocdecode-sample && cd rocdecode-sample
  cmake /opt/rocm/share/rocdecode/samples/videoDecode/
  make -j8
  ./videodecode -i /opt/rocm/share/rocdecode/video/AMD_driving_virtual_20-H265.mp4
  ```

### Using test package

To verify your installation using the `rocdecode-test` package, run:

  ```shell
  mkdir rocdecode-test && cd rocdecode-test
  cmake /opt/rocm/share/rocdecode/test/
  ctest -VV
  ```

## Samples

You can access samples to decode your videos in our
[GitHub repository](https://github.com/ROCm/rocDecode/tree/develop/samples). Refer to the
individual folders to build and run the samples.

[FFmpeg](https://ffmpeg.org/about.html) is required for sample applications and `make test`. To install
FFmpeg, refer to the instructions listed for your operating system:

* Ubuntu:

  ```shell
  sudo apt install libavcodec-dev libavformat-dev libavutil-dev
  ```

* RHEL/SLES:

  Install ffmpeg development packages manually or use `rocDecode-setup.py` script

## Docker

You can find rocDecode Docker containers in our
[GitHub repository](https://github.com/ROCm/rocDecode/tree/develop/docker).

## Tested configurations

* Linux
  * Ubuntu - `22.04` / `24.04`
  * RHEL - `8` / `9`
  * SLES - `15 SP7`
* ROCm: `7.0.0`
* libva-amdgpu-dev - `2.16.0`
* mesa-amdgpu-va-drivers - `1:24.3.0`
* FFmpeg - `4.4.2` / `6.1.1`
* rocDecode Setup Script - `V2.5.0`
