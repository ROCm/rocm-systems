# JPEG decode batched async sample

The jpeg decode batched async sample illustrates asynchronous batch decoding of JPEG images using the rocJPEG library to get the decoded images in one of the supported output formats (i.e., native, yuv, y, rgb, rgb_planar). It uses a threaded pipeline with `rocJpegDecodeBatchedAsync` and `rocJpegDecodeBatchedSync` APIs for improved throughput by overlapping batch submission with result retrieval. This sample can be configured with a device ID, a batch size, and can optionally dump the output to a file.

## Prerequisites:

* Install [rocJPEG](https://rocm.docs.amd.com/projects/rocJPEG/en/latest/install/rocjpeg-build-and-install.html)

## Build

```shell
mkdir jpeg_decode_batched_async_sample && cd jpeg_decode_batched_async_sample
cmake ../
make -j
```

## Run

```shell
./jpegdecodebatchedasync -i        <[input path] - input path to a single JPEG image or a directory containing JPEG images - [required]>
                         -be       <[backend] - select rocJPEG backend (0 for hardware-accelerated JPEG decoding using VCN,
                                                                        1 for hybrid JPEG decoding using CPU and GPU HIP kernels (currently not supported)) [optional - default: 0]>
                         -fmt      <[output format] - select rocJPEG output format for decoding, one of the [native, yuv_planar, y, rgb, rgb_planar] [optional - default: native]>
                         -o        <[output path] - path to an output file or a path to a directory - write decoded images to a file or directory based on selected output format [optional]>
                         -d        <[device id] - specify the GPU device id for the desired device (use 0 for the first device, 1 for the second device, and so on) [optional - default: 0]>
                         -b        <[batch size] - number of images to decode per batch [optional - default: 2]>
                         -crop     <[crop rectangle] - crop rectangle for output in a comma-separated format: left,top,right,bottom - [optional]>
```
