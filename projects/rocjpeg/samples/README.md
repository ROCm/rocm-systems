# Samples overview

rocJPEG samples

## Windows setup

On Windows, every sample needs the following before you configure it with CMake and when you run it.

* Set `ROCM_PATH` to the ROCm installation prefix and keep it set at run time; libva uses it to locate the
  VA-API driver.
* Add the rocJPEG DLL directory to `PATH`:

  ```bat
  set ROCM_PATH=<path-to-TheRock-build>
  set PATH=%ROCM_PATH%\bin;%PATH%
  ```

The samples do not need FFmpeg. rocJPEG loads the VA-API DLLs from `%ROCM_PATH%\lib\rocm_sysdeps\bin` itself, so
that directory does not need to be on `PATH`.

Build a sample with `cmake --build . --config Release` and run it from the `Release` directory, for example:

```bat
mkdir rocjpeg-sample && cd rocjpeg-sample
cmake "%ROCM_PATH%\share\rocjpeg\samples\jpegDecode"
cmake --build . --config Release
Release\jpegdecode.exe -i "%ROCM_PATH%\share\rocjpeg\images"
```

> [!NOTE]
> Hardware JPEG decoding is not yet available on Windows. The VA-API on D3D12 driver does not expose the JPEG baseline
> profile, so `rocJpegCreate` with `ROCJPEG_BACKEND_HARDWARE` returns `ROCJPEG_STATUS_HW_JPEG_DECODER_NOT_SUPPORTED`
> and the decode samples cannot decode images. Run `jpegDecodeCaps` to check whether JPEG decoding is available on
> your system.

## [JPEG decode](jpegDecode)

The jpeg decode sample illustrates decoding a JPEG images using rocJPEG library to get the individual decoded images in one of the supported output format (i.e., native, yuv, y, rgb, rgb_planar). This sample can be configured with a device ID and optionally able to dump the output to a file.

## [JPEG decode batched](jpegDecodeBatched)

The jpeg decode bacthed sample illustrates decoding JPEG images by batches of specified size using rocJPEG library to get the individual decoded images in one of the supported output format (i.e., native, yuv, y, rgb, rgb_planar). This sample can be configured with a device ID and optionally able to dump the output to a file.

## [JPEG decode perf](jpegDecodePerf)

The jpeg decode perf sample illustrates decoding JPEG images by batches of specified size with multiple threads using rocJPEG library to achieve optimal performance. The individual decoded images can be retrieved in one of the supported output format (i.e., native, yuv, y, rgb, rgb_planar). This sample can be configured with a device ID and optionally able to dump the output to a file.

## [JPEG decode capabilities](jpegDecodeCaps)

The jpeg decode capabilities sample illustrates querying the JPEG decode capabilities of a rocJPEG handle using the `rocJpegGetDecoderCaps` API, such as the number of JPEG cores, the supported image resolutions, chroma subsamplings, and output formats. This sample can be configured with a device ID and backend, and does not require an input image.
