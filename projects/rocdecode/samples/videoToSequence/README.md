# Video decode sample

The VideoToSequence sample illustrates decoding a single packetized video stream using FFMPEG demuxer and splitting it into multiple video sequences. This uses seek functionality to seek to a random position and extract a batch of video sequences in YUV format with step and stride. This sample can be configured with a device ID and optionally able to dump the output to a file. This sample uses the high-level RocVideoDecoder class which connects both the video parser and Rocdecoder. This process repeats until a batch of sequences are extracted or EOS is reached.

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
./videotosequence -i <Input file/folder Path [required]> 
              -o <Output folder to dump sequences - dumps output if requested [optional]>
              -d <GPU device ID - 0:device 0 / 1:device 1/ ... [optional - default:0]>
              -b <batch_size - specify the number of sequences to be decoded [optional - default:1]>
              -step <frame interval between each sequence [optional - default:1]> 
              -stride <distance between consective frames in a sequence [optional - default:1]>
              -l <Number of frames in each sequence [optional - default:1]>
              -crop <crop rectangle for output (not used when using interopped decoded frame) [optional - default:1]>
              -seek_mode <option for seeking (0: no seek 1: seek to prev key frame) [optional - default: 0]>
              -crop <crop rectangle for output (not used when using interopped decoded frame) [optional - default: 0,0,0,0]>
              -m <output_surface_memory_type - decoded surface memory [optional - default: 0][0 : OUT_SURFACE_MEM_DEV_INTERNAL/ 1 : OUT_SURFACE_MEM_DEV_COPIED/ 2 : OUT_SURFACE_MEM_HOST_COPIED/3 : OUT_SURFACE_MEM_NOT_MAPPED]>
```