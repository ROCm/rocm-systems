# Video decode RGB sample

This sample illustrates the FFMPEG demuxer to get the individual frames which are then decoded using rocDecode API and optionally color-converted using custom HIP kernels on AMD hardware. This sample converts decoded YUV output to one of the RGB or BGR formats(24bit, 32bit, 464bit) in a separate thread allowing it to run both VCN hardware and compute engine in parallel.

This sample uses HIP kernels to showcase the color conversion.  Whenever a frame is ready after decoding, the `ColorSpaceConversionThread` is notified and can be used for post-processing.

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
mkdir video_decode_rgb_sample && cd video_decode_rgb_sample
cmake ../
make -j
```

## Run

```shell
./videodecodergb    -i <input video file - required> 
                    -o <optional; output path to save decoded YUV frames>
                    -d <GPU device ID, 0 for the first device, 1 for the second device, etc> 
                    -of <optional: output format bgr, bgra, bgr48, bgr64 etc>
```