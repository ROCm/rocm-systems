# Video decode batch sample

This sample decodes multiple files using multiple threads, using the rocDecode library. The input is a directory of files and an input number of threads. The maximum number of threads is capped to 64.
If the number of files is higher than the number of threads requested by the user, the files are distributed to the threads in a round robin fashion. 
If the number of files is lesser than the number of threads requested by the user, the number of threads created will be equal to the number of files.

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
mkdir video_decode_batch && cd video_decode_batch
cmake ../
make -j
```

## Run

```shell
./videodecodebatch -i <directory containing input video files [required]> 
                                   -t <number of threads [optional - default:4]>
                                   -d <Device ID (>= 0) [optional - default:0]>
                                   -o Directory for output YUV files - optional
                                   -m output_surface_memory_type - decoded surface memory; optional; default - 3 [0 : OUT_SURFACE_MEM_DEV_INTERNAL/ 1 : OUT_SURFACE_MEM_DEV_COPIED/ 2 : OUT_SURFACE_MEM_HOST_COPIED/ 3 : OUT_SURFACE_MEM_NOT_MAPPED]
                                   -disp_delay -specify the number of frames to be delayed for display; optional; default: 1
```