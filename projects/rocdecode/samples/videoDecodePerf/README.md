# Video decode performance sample

This sample illustrates the FFMPEG demuxer to get the individual frames which are then decoded on AMD hardware using rocDecode library.

This sample uses multiple threads to decode the same input video parallelly.

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
mkdir video_decode_perf_sample && cd video_decode_perf_sample
cmake ../
make -j
```

## Run

```shell
./videodecodeperf -i <input video file [required]> 
                  -t <number of threads [optional - default:1]>
                  -f <Number of decoded frames - specify the number of pictures to be decoded [optional]>
                  -disp_delay <display delay - specify the number of frames to be delayed for display [optional]>
                  -d <Device ID (>= 0) [optional - default:0]>
                  -z <force_zero_latency - Decoded frames will be flushed out for display immediately [optional]>
                  -m <Memory type (integer values between 0 to 3: specifies where to store the decoded output:
                                                                  0 = decoded output will be in internal interopped memory,
                                                                  1 = decoded output will be copied to a separate device memory
                                                                  2 = decoded output will be copied to a separate host memory
                                                                  3 = decoded output will not be available (decode only)) [optional; default: 3]>
```