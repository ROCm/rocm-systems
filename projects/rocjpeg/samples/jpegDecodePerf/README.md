# JPEG decode perf sample

The jpeg decode perf sample illustrates decoding JPEG images by batches of specified size with multiple threads using rocJPEG library to achieve optimal performance. The individual decoded images can be retrieved in one of the supported output format (i.e., native, yuv, y, rgb, rgb_planar). This sample can be configured with a device ID and optionally able to dump the output to a file.

## Prerequisites:

* Install [rocJPEG](../../README.md#build-and-install-instructions)

## Build

```shell
mkdir jpeg_decode_perf_sample && cd jpeg_decode_perf_sample
cmake ../
make -j
```

## Run

```shell
./jpegdecodeperf         -i     <[input path] - input path to a single JPEG image or a directory containing JPEG images - [required]>
                         -be    <[backend] - select rocJPEG backend (0 for hardware-accelerated JPEG decoding using VCN,
                                                                     1 for hybrid JPEG decoding using CPU and GPU HIP kernels (currently not supported)) [optional - default: 0]>
                         -fmt   <[output format] - select rocJPEG output format for decoding, one of the [native, yuv_planar, y, rgb, rgb_planar] [optional - default: native]>
                         -o     <[output path] - path to an output file or a path to a directory - write decoded images to a file or directory based on selected output format [optional]>
                         -d     <[device id] - specify the GPU device id for the desired device (use 0 for the first device, 1 for the second device, and so on) [optional - default: 0]>
                         -crop  <[crop rectangle] - crop rectangle for output in a comma-separated format: left,top,right,bottom - [optional]>
                         -t     <[threads] - number of threads for parallel JPEG decoding [optional - default: 1]>
                         -b     <[batch_size] - decode images from input by batches of a specified size - [optional - default: 1]>
```