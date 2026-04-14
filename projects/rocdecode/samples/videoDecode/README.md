# Video decode sample

The video decode sample illustrates decoding a single packetized video stream using FFMPEG demuxer, video parser, and rocDecoder to get the individual decoded frames in YUV format. This sample can be configured with a device ID and optionally able to dump the output to a file. This sample uses the high-level RocVideoDecoder class which connects both the video parser and Rocdecoder. This process repeats in a loop until all frames have been decoded.

## Prerequisites:

* Install [rocDecode](../../README.md#build-and-install-instructions)

* [FFMPEG](https://ffmpeg.org/about.html)

    * On `Ubuntu`

  ```shell
  sudo apt install libavcodec-dev libavformat-dev libavutil-dev
  ```
  
    * On `RHEL`/`SLES` - install ffmpeg development packages manually or use [rocDecode-setup.py](../../rocDecode-setup.py) script

## Build

```shell
mkdir video_decode_sample && cd video_decode_sample
cmake ../
make -j
```

## Run

```shell
./videodecode -i <input video file [required]> 
              -o <output path to save decoded YUV frames [optional]> 
              -d <GPU device ID - 0:device 0 / 1:device 1/ ... [optional - default:0]>
              -f <Number of decoded frames - specify the number of pictures to be decoded [optional]>
              -z <force_zero_latency - Decoded frames will be flushed out for display immediately [optional]>
              -disp_delay <display delay - specify the number of frames to be delayed for display [optional - default: 1]>
              -sei <extract SEI messages [optional]>
              -md5 <generate MD5 message digest on the decoded YUV image sequence [optional]>
              -md5_check MD5_File_Path <generate MD5 message digest on the decoded YUV image sequence and compare to the reference MD5 string in a file [optional]>
              -crop <crop rectangle for output (not used when using interopped decoded frame) [optional - default: 0,0,0,0]>
              -m <output_surface_memory_type - decoded surface memory [optional - default: 0][0 : OUT_SURFACE_MEM_DEV_INTERNAL/ 1 : OUT_SURFACE_MEM_DEV_COPIED/ 2 : OUT_SURFACE_MEM_HOST_COPIED/3 : OUT_SURFACE_MEM_NOT_MAPPED]>
              -no_ffmpeg_demux <use the built-in bitstream reader instead of FFMPEG demuxer to obtain picture data [optional]>
```