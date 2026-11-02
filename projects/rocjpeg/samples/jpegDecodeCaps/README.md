# JPEG decode capabilities sample

This sample queries and prints the JPEG decode capabilities of an AMD GPU using the `rocJpegGetDecoderCaps` API from the rocJPEG library.

The sample creates a rocJPEG handle for the selected device and backend, and reports the fields exposed by the `RocJpegDecodeCaps` struct:

- supported JPEG coding processes (only baseline is supported by the VCN JPEG decoder)
- number of VCN JPEG cores available to the handle (on partitioned GPUs, the cores of the selected compute partition)
- minimum and maximum image width x height
- maximum number of color components
- ROI (crop) decode support, done by the hardware or in software
- RGB color conversion, done by the hardware or in software (hardware conversion depends on the `ROCJPEG_ENABLE_VCN_HW_CSC` environment variable)
- a table with one row per rocJPEG output format (`RocJpegOutputFormat`) and, for each supported chroma subsampling, the pixel format of the decoded image and its number of planes, followed by a description of each pixel format. The `ROCJPEG_OUTPUT_NATIVE` pixel format is the surface format written by the VCN JPEG decoder, verified against the VA-API driver

Unlike the other samples, this app does not decode an image, so it needs no input file.

## Prerequisites

* Install [rocJPEG](https://rocm.docs.amd.com/projects/rocJPEG/en/latest/install/rocjpeg-build-and-install.html)

## Build

```shell
mkdir jpeg_decode_caps_sample && cd jpeg_decode_caps_sample
cmake ../
make -j
```

## Run

```shell
./jpegdecodecaps -d  <[device id] - specify the GPU device id for the desired device (use 0 for the first device, 1 for the second device, and so on) [optional - default: 0]>
                 -be <[backend] - select rocJPEG backend (0 for hardware-accelerated JPEG decoding using VCN,
                                                         1 for hybrid JPEG decoding using CPU and GPU HIP kernels (currently not supported)) [optional - default: 0]>
```
