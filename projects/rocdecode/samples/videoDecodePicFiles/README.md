# Video decode picture files sample

The video decode picture files sample illustrates decoding an elementary video stream which is stored in multiple files with each file containing bitstream data of a coded picutre. This sample can be configured with a device ID and optionally able to dump the output to a file. This sample uses the high-level RocVideoDecoder class which connects both the video parser and Rocdecoder. This process repeats in a loop until all frames have been decoded.

## Prerequisites:

* Install [rocDecode](../../README.md#build-and-install-instructions)

## Build

```shell
mkdir video_decode_pic_files && cd video_decode_pic_files
cmake ../
make -j
```

## Run

```shell
./videodecodepicfiles -i <Input picture files [required]>
                    -codec <Codec type (0: HEVC, 1: AVC; 2: AV1; 3: VP9) - [required]>
                    -l <Number of iterations [optional - default: 1]>
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
```