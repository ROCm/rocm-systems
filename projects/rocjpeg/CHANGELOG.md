# rocJPEG changelog

Documentation for rocJPEG is available at
[https://rocm.docs.amd.com/projects/rocJPEG/en/latest/](https://rocm.docs.amd.com/projects/rocJPEG/en/latest/)

## (unreleased) rocJPEG 1.14.0

### Resolved issues

* Fixed the RGB, RGB planar, Y, and YUV planar outputs writing past the image when its width is not a multiple of 8 or its height is odd. With output buffers sized exactly to the image, this overwrote memory after the buffer and made the right and bottom edges of the image nondeterministic.

## rocJPEG 1.13.0 for ROCm 10.2.0

### Added

* Added the `rocJpegGetDecoderCaps` API and the `RocJpegDecodeCaps` structure to query the decode capabilities of a rocJPEG handle, including the number of JPEG cores, the supported JPEG coding processes, image resolutions, chroma subsamplings, and output formats, the native output surface format of each chroma subsampling as verified with the VA-API driver, and the hardware ROI decode and color conversion support.
* Added the `RocJpegSurfaceFormat` enum to describe the layout of the decoded image produced by the VCN JPEG decoder, and the `RocJpegCodingProcess` enum to report the supported JPEG coding processes.
* Added the `jpegDecodeCaps` sample to demonstrate the `rocJpegGetDecoderCaps` API.
* Added HIP-based color conversion kernels to the batched decode path, improving batched decoding performance.

### Changed

* The decode APIs (`rocJpegDecode`, `rocJpegDecodeAsync`, `rocJpegDecodeBatched`, and `rocJpegDecodeBatchedAsync`) now return `ROCJPEG_STATUS_INVALID_PARAMETER` when the crop rectangle in `RocJpegDecodeParams` does not lie within the image.

## rocJPEG 1.10.0 for ROCm 10.1.0

### Added

* Added rocJpegDecodeBatchedAsync and rocJpegDecodeBatchedSync APIs to support asynchronous batched JPEG decoding, and the `jpegDecodeBatchedAsync` sample.
* Added support for explicitly loading librocm_sysdeps_va via dlopen, ensuring complete isolation from the system libva library.
* `BUILD_TESTS` CMake option (default OFF) to build and install the sample and negative API test binaries, including under TheRock.
* CTests for the `jpegDecodeAsync` and `jpegDecodeBatchedAsync` samples.
* `rocjpeg_benchmark.py` script to build `jpegDecodePerf` and run decode benchmarks across image resolutions, batch sizes, and thread counts.

### Changed

* Changed the HIP YUV-to-RGB color conversion kernels from the BT.709 standard to BT.601 for color space conversion.
* The VA-API and DRM CMake find modules now search only the TheRock sysdeps location (`${ROCM_PATH}/lib/rocm_sysdeps`) instead of falling back to system paths.
* The `jpegDecodePerf` sample allocates and frees the output buffers outside the timed decode loop, so multi-threaded runs no longer block on `hipFree`.

### Resolved issues

* Fixed link failures (undefined `pthread_create`) in the `jpegDecodeAsync` and `jpegDecodeBatchedAsync` samples.

## rocJPEG 1.7.0 for ROCm 10.0.0

### Added

* Added rocJpegDecodeAsync and rocJpegDecodeSync APIs to support asynchronous single-image JPEG decoding, allowing decode submission and completion to be separated across threads for improved pipeline throughput, and the `jpegDecodeAsync` sample.

### Changed

* Improved VAAPI batch grouping by using reusable surface bucket dimensions to reduce decode batch fragmentation.
* The `jpegDecodePerf` sample pre-allocates GPU output memory instead of allocating and freeing it for every image.
* Info level logs list each GPU UUID and PCI bus/device/function to render node mapping and the selected render node.

### Resolved issues

* Fixed decoding always running on the first GPU (`renderD128`) on GPUs whose HIP device UUID doesn't match the amdgpu sysfs `unique_id` (for example, MI355X). The decoder now binds to the requested device by PCI bus/device/function.
* Fixed integer overflows in the SOF marker parser and in channel copies for very large image dimensions. Images with a width or height of 0 or greater than 65500 are now rejected.

## rocJPEG 1.6.0 for ROCm 7.14.1

### Added

* Added a logging mechanism for core APIs that can be controlled by setting the `ROCJPEG_LOG_LEVEL` environment variable.
* Debug level logs for each JPEG marker (SOF, DQT, DHT, SOS, EOI) with its stream offset, and clearer error messages for unsupported JPEG streams. Progressive JPEG streams are explicitly rejected.
* Info level logs include the VA-API version and driver, the DRM render node, and the GPU UUID mapping.
* A Python test runner is installed with the rocJPEG tests.
* Support for gfx1250.

### Changed

* Added DRM render node offset handling for MI308 compute partitions.

## rocJPEG 1.5.0 for ROCm 7.13

### Added

* The `jpegDecodePerf` sample reports the total wall time of the decode.
* Support for gfx1150, gfx1152, and gfx1153.

### Changed

* rocJPEG is now delivered as part of [TheRock](https://github.com/ROCm/TheRock). All core dependencies are provided by the TheRock build.
* Removed CPack packaging (DEB/RPM/NSIS/TGZ/ZIP generation and all related CPACK variables).
* Removed `rocJPEG-setup.py` dependency installer script.
* Removed Docker files.
* Removed package install documentation; updated all documentation to reference TheRock for installation.
* Simplified libva version check (single `>= 1.22` requirement).
* Cleaned up CMake error messages.

### Resolved issues

* Fixed the sample CMake files skipping their `Threads_FOUND` checks.

## rocJPEG 1.4.0 for ROCm 7.2.4

### Changed

* Bug fixes and performance improvements
* GitHub repository moved to [https://github.com/ROCm/rocm-systems/tree/develop/projects/rocjpeg](https://github.com/ROCm/rocm-systems/tree/develop/projects/rocjpeg)

## rocJPEG 1.3.0 for ROCm 7.2.0

## Changed
* Updated libdrm path configuration and libva version requirements for ROCm and TheRock platforms
* RHEL now uses `libva-devel` instead of `libva-amdgpu`/`libva-amdgpu-devel`
* Use ROCm clang++ from `${ROCM_PATH}/lib/llvm/bin` location

## rocjpeg 1.2.0 for ROCm 7.1.0

## Changed
* HIP meta package changed - Use hip-dev/devel to bring required hip dev deps

### Resolved issues
* Fixed an issue where extra padding was incorrectly included when saving decoded JPEG images to files.
* Resolved a memory leak in the jpegDecode application.

## rocjpeg 1.1.0 for ROCm 7.0.0

## Added
* cmake config files
* CTEST - New tests were introduced for JPEG batch decoding using various output formats, such as yuv_planar, y, rgb, and rgb_planar, both with and without region-of-interest (ROI).

## Changed
* Readme - cleanup and updates to pre-reqs
* The `decode_params` argument of the `rocJpegDecodeBatched` API is now an array of `RocJpegDecodeParams` structs representing the decode parameters for the batch of JPEG images.
* `libdrm_amdgpu` is now explicitly linked with rocjpeg.

## Removed
* Dev Package - No longer installs pkg-config

### Resolved issues
* Fixed a bug that prevented copying the decoded image into the output buffer when the output buffer is larger than the input image.
* Resolved an issue with resizing the internal memory pool by utilizing the explicit constructor of the vector's type during the resizing process.
* Addressed and resolved CMake configuration warnings.

## rocJPEG 0.8.0 for ROCm 6.4

### Changed

* AMD Clang++ is now the default CXX compiler.
* The jpegDecodeMultiThreads sample has been renamed to jpegDecodePerf, and batch decoding has been added to this sample instead of single image decoding for improved performance.

## rocJPEG 0.6.0 for ROCm 6.3.0

### Changes

* Supported initial enablement of the rocJPEG library
* Supported JPEG chroma subsampling:
  * YUV 4:4:4
  * YUV 4:4:0
  * YUV 4:2:2
  * YUV 4:2:0
  * YUV 4:0:0
* Supported various output image format:
  * Native (i.e., native unchanged output from VCN Hardware, it can be either packed YUV or planar YUV)​
  * YUV planar​
  * Y only​
  * RGB​
  * RGB planar
* Supported single-image and batch-image decoding​
* Supported Different partition modes on MI300 series​
* Supported region-of-interest (ROI) decoding​
* Supported color space conversion from YUV to RGB
