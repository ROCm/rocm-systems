# rocJPEG API Negative tests

This test suite is designed to perform negative testing on all rocJPEG APIs. The purpose of these tests is to validate the robustness and error-handling mechanisms of the rocJPEG APIs 
by providing invalid inputs, unexpected scenarios, or edge cases to ensure the APIs respond with appropriate error messages or behaviors.

## Prerequisites:

* Install [rocJPEG](https://rocm.docs.amd.com/projects/rocJPEG/en/latest/install/rocjpeg-build-and-install.html)

## Build

**Linux:**

```shell
mkdir build && cd build
cmake ../
make -j
```

**Windows:**

```bat
mkdir build && cd build
set ROCM_PATH=<path-to-TheRock-build>
cmake ..
cmake --build . --config Release
```

## Run

**Linux:**

```shell
./rocjpegnegativetest
```

**Windows:**

```bat
set PATH=%ROCM_PATH%\bin;%PATH%
Release\rocjpegnegativetest.exe
```

On Windows, keep `ROCM_PATH` set at run time; libva uses it to locate the VA-API driver.

> [!NOTE]
> Hardware JPEG decoding is not yet available on Windows. When `rocJpegCreate` returns
> `ROCJPEG_STATUS_HW_JPEG_DECODER_NOT_SUPPORTED`, the test skips the cases that need a decoder handle and runs the
> rest.