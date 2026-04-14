# Video decode multi files sample

The video decodes multiple files sample illustrates the use of providing a list of files as input to showcase the reconfigure option in the rocDecode library. The input video files have to be of the same codec type to use the reconfigure option but can have different resolutions or resize parameters.

The reconfigure option can be disabled by the user if needed. The input file is parsed line by line and data is stored in a queue. The individual video files are demuxed and decoded one after the other in a loop. Output for each input file can also be stored if needed.

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
mkdir video_decode_multi_files_sample && cd video_decode_multi_files_sample
cmake ../
make -j
```

## Run

```shell
./videodecodemultifiles -i <input file list[required - example.txt]>
                        -d <GPU device ID - 0:device 0 / 1:device 1/ ... [optional - default:0]>
                        -use_reconfigure <flag (bool - 0/1) [optional - default: 1] set 0 to disable reconfigure api for decoding multiple files. Only resolution changes between files are supported when reconfigure is enabled. The codec, bit_depth, and the chroma_format must be the same between files>
```
### Note: Example input file list - example.txt

```shell
infile input1.[mp4/mov...] [required]
outfile output1.yuv [optional]
z 0 [optional]
sei 0 [optional]
crop l,t,r,b [optional]
m  0 [optional] [0 : OUT_SURFACE_MEM_DEV_INTERNAL/ 1 : OUT_SURFACE_MEM_DEV_COPIED/ 2 : OUT_SURFACE_MEM_HOST_COPIED/ 3 : OUT_SURFACE_MEM_NOT_MAPPED]
infile input2.[mp4/mov...] [optional]
outfile output2.yuv [optional]
...
...
```