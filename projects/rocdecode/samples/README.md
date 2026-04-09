# Samples overview

rocDecode samples

## [Video decode](videoDecode)

The video decode sample illustrates decoding a single packetized video stream using FFMPEG demuxer, video parser, and rocDecoder to get the individual decoded frames in YUV format. This sample can be configured with a device ID and optionally able to dump the output to a file. This sample uses the high-level RocVideoDecoder class which connects both the video parser and Rocdecoder. This process repeats in a loop until all frames have been decoded.

## [Video decode batch sample](videoDecodeBatch)

This sample decodes multiple files using multiple threads, using the rocDecode library. The input is a directory of files and an input number of threads. The maximum number of threads is capped to 64.
If the number of files is higher than the number of threads requested by the user, the files are distributed to the threads in a round robin fashion. 
If the number of files is lesser than the number of threads requested by the user, the number of threads created will be equal to the number of files.

## [Video decode memory](videoDecodeMem)

The video decode memory sample illustrates a way to pass the data chunk-by-chunk sequentially to the FFMPEG demuxer which is then decoded on AMD hardware using rocDecode library.

The sample provides a user class `FileStreamProvider` derived from the existing `VideoDemuxer::StreamProvider` to read a video file and fill the buffer owned by the demuxer. It then takes frames from this buffer for further parsing and decoding.

## [Video decode multi files](videoDecodeMultiFiles)

The video decodes multiple files sample illustrates the use of providing a list of files as input to showcase the reconfigure option in the rocDecode library. The input video files have to be of the same codec type to use the reconfigure option but can have different resolutions or resize parameters.

The reconfigure option can be disabled by the user if needed. The input file is parsed line by line and data is stored in a queue. The individual video files are demuxed and decoded one after the other in a loop. Output for each input file can also be stored if needed.

## [Video decode performance](videoDecodePerf)

This sample illustrates the FFMPEG demuxer to get the individual frames which are then decoded on AMD hardware using rocDecode library.

This sample uses multiple threads to decode the same input video parallelly.

## [Video decode RGB](videoDecodeRGB)

This sample illustrates the FFMPEG demuxer to get the individual frames which are then decoded using rocDecode API and optionally color-converted using custom HIP kernels on AMD hardware. This sample converts decoded YUV output to one of the RGB or BGR formats(24bit, 32bit, 464bit) in a separate thread allowing it to run both VCN hardware and compute engine in parallel.

This sample uses HIP kernels to showcase the color conversion.  Whenever a frame is ready after decoding, the `ColorSpaceConversionThread` is notified and can be used for post-processing.