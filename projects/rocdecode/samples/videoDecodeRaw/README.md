# Video decode sample

The video decode raw sample illustrates decoding a single packetized video stream using the built-in bitstream reader, video parser, and rocDecoder to get the individual decoded frames in YUV format. This sample can be configured with a device ID and optionally able to dump the output to a file. This sample uses the high-level RocVideoDecoder class which connects both the video parser and Rocdecoder. This process repeats in a loop until all frames have been decoded.

## Prerequisites:

* Install [rocDecode](../../README.md#build-and-install-instructions)

## Build

```shell
mkdir video_decode_raw_sample && cd video_decode_raw_sample
cmake ../
make -j
```

## Run

```shell
./videodecoderaw -i <input video file [required]>
              -o <output path to save decoded YUV frames [optional]> 
              -d <GPU device ID - 0:device 0 / 1:device 1/ ... [optional - default:0]>
              -f <Number of decoded frames - specify the number of pictures to be decoded [optional]>
              -z <force_zero_latency - Decoded frames will be flushed out for display immediately [optional]>
              -disp_delay <display delay - specify the number of frames to be delayed for display [optional - default: 1]>
              -sei <extract SEI messages [optional]>
              -crop <crop rectangle for output (not used when using interopped decoded frame) [optional - default: 0,0,0,0]>
              -m <output_surface_memory_type - decoded surface memory [optional - default: 0][0 : OUT_SURFACE_MEM_DEV_INTERNAL/ 1 : OUT_SURFACE_MEM_DEV_COPIED/ 2 : OUT_SURFACE_MEM_HOST_COPIED/3 : OUT_SURFACE_MEM_NOT_MAPPED]>
```