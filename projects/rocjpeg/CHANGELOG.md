# rocJPEG changelog

Documentation for rocJPEG is available at
[https://rocm.docs.amd.com/projects/rocJPEG/en/latest/](https://rocm.docs.amd.com/projects/rocJPEG/en/latest/)

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