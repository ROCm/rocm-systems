# Samples overview

rocJPEG samples

## [JPEG decode](jpegDecode)

The jpeg decode sample illustrates decoding a JPEG images using rocJPEG library to get the individual decoded images in one of the supported output format (i.e., native, yuv, y, rgb, rgb_planar). This sample can be configured with a device ID and optionally able to dump the output to a file.

## [JPEG decode batched](jpegDecodeBatched)

The jpeg decode bacthed sample illustrates decoding JPEG images by batches of specified size using rocJPEG library to get the individual decoded images in one of the supported output format (i.e., native, yuv, y, rgb, rgb_planar). This sample can be configured with a device ID and optionally able to dump the output to a file.

## [JPEG decode perf](jpegDecodePerf)

The jpeg decode perf sample illustrates decoding JPEG images by batches of specified size with multiple threads using rocJPEG library to achieve optimal performance. The individual decoded images can be retrieved in one of the supported output format (i.e., native, yuv, y, rgb, rgb_planar). This sample can be configured with a device ID and optionally able to dump the output to a file.