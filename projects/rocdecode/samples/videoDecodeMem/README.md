# Video decode memory sample

The video decode memory sample illustrates a way to pass the data chunk-by-chunk sequentially to the FFMPEG demuxer which is then decoded on AMD hardware using rocDecode library.

The sample provides a user class `FileStreamProvider` derived from the existing `VideoDemuxer::StreamProvider` to read a video file and fill the buffer owned by the demuxer. It then takes frames from this buffer for further parsing and decoding.

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
mkdir video_decode_mem_sample && cd video_decode_mem_sample
cmake ../
make -j
```

## Run

```shell
./videodecodemem  -i <input video file [required]> 
                  -o <output path to save decoded YUV frames [optional]> 
                  -d <GPU device ID - 0:device 0 / 1:device 1/ ... [optional - default:0]>
                  -z <force_zero_latency - Decoded frames will be flushed out for display immediately [optional]>
                  -sei <extract SEI messages [optional]>
                  -crop <crop rectangle for output (not used when using interopped decoded frame) [optional - default: 0,0,0,0]>
                  -m <output_surface_memory_type - decoded surface memory [optional - default: 0][0 : OUT_SURFACE_MEM_DEV_INTERNAL/ 1 : OUT_SURFACE_MEM_DEV_COPIED/ 2 : OUT_SURFACE_MEM_HOST_COPIED/ 3 : OUT_SURFACE_MEM_NOT_MAPPED]>
```