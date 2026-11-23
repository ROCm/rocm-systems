# Video decode async sample

The video decode async sample decodes a video stream with a producer thread and a consumer thread. The producer demuxes the stream with the FFMPEG demuxer, parses it, and submits pictures to the decoder; its display callback hands each decoded frame to the consumer through a bounded queue. The consumer hashes the frames in display order. The sample uses the rocDecode parser and decoder APIs directly, not the RocVideoDecoder class, because it exists to compare ways of handing frames from the display callback to another thread.

How a frame is handed over is set by the display policy:

* `sync-copy` (default): the display callback gets the frame with `rocDecGetVideoFrame`, copies it into a buffer owned by the sample and waits for the copy before queuing the buffer. This is safe, because the decoder may reuse the decoded surface as soon as the display callback returns.
* `no-copy`: the display callback queues the pointer returned by `rocDecGetVideoFrame` without copying, and the consumer reads it later. That pointer is an *alias* of the decoded surface: a second reference to the same memory, not a copy. This policy is intentionally unsafe. By the time the consumer reads the frame, the parser might have reused the surface for another picture, so the output doesn't match the reference MD5. It serves as a control that shows a test can detect this reuse.

The consumer workload sets how long each frame is held before it is hashed. `lag:<L>` handles frame n only after frame n+L has been displayed, which holds frames for a fixed number of pictures; `hold:<min>-<max>` waits a random time instead.

## Prerequisites:

* Install [rocDecode](https://rocm.docs.amd.com/projects/rocDecode/en/latest/install/rocDecode-build-and-install.html)

* [FFMPEG](https://ffmpeg.org/about.html)

  ```shell
  sudo apt install libavcodec-dev libavformat-dev libavutil-dev
  ```

## Build

**Linux:**

```shell
mkdir video_decode_async_sample && cd video_decode_async_sample
cmake ../
make -j
```

**Windows:**

```bat
mkdir video_decode_async_sample && cd video_decode_async_sample
set ROCM_PATH=<path-to-TheRock-build>
cmake ..
cmake --build . --config Release
```

> [!NOTE]
> On Windows, set up `ROCM_PATH` and `PATH` before configuring and running; see
> [Windows setup](../README.md#windows-setup).

## Run

```shell
./videodecodeasync -i <input video file [required]>
                   -d <GPU device ID - 0:device 0 / 1:device 1/ ... [optional - default:0]>
                   -p <display policy: sync-copy | no-copy [optional - default: sync-copy]>
                   -w <consumer workload: md5 | lag:<L> (L 1-255) | hold:<min>-<max> [optional - default: md5]>
                   -dd <display delay passed to the parser [optional - default: 1]>
                   -ds <max number of decode surfaces passed to the parser [optional - default: 1, the parser decides]>
                   -q <queue depth between producer and consumer, 1-256 [optional - default: 4, raised to L+1 for lag:<L>]>
                   -md5 <generate MD5 message digest on the output frames [optional]>
                   -md5_check MD5_File_Path <generate MD5 message digest and compare to the reference MD5 string in a file [optional]>
                   -md5_frames File_Path <write one MD5 digest per output frame [optional]>
                   -expect_mismatch <succeed only if the -md5_check digest does NOT match; for control runs [optional]>
```

The MD5 digest is computed the same way as `videoDecode -md5` in its default output mode, so `-md5_check` can compare against the same reference files; the first token in the file must be a 32-digit hexadecimal MD5 digest. MD5 hashing (`-md5`, `-md5_check`, `-md5_frames`) supports only 4:2:0 streams; for other chroma formats the sample stops with an error. The sample prints the decode FPS, the end-to-end FPS and the time the producer spent inside the display callback.

## Running the no-copy control

The examples use the test clip and reference MD5 installed with rocDecode:

```shell
V=$ROCM_PATH/share/rocdecode/video
M=$ROCM_PATH/share/rocdecode/md5
```

A plain `no-copy` run, where the consumer reads each frame as soon as it is dequeued:

```shell
./videodecodeasync -i $V/AMD_driving_virtual_20-H265.mp4 -p no-copy -md5_check $M/AMD_driving_virtual_20-H265_yuv420.md5
```

This prints `MD5 digest does not match the reference MD5 digest` and exits with 1. Even with no hold, the decoder has already reused many of the surfaces by the time the consumer reads them; a smaller queue (`-q 1`) reduces the damage but does not prevent it.

The following command runs the control the same way as the `video_decodeAsync-no-copy-control` CTest. Each frame is held until 8 more frames have been displayed (`-w lag:8`). Because this is a fixed number of pictures, the result doesn't depend on timing. The `-dd 0` option makes the decoder reuse surfaces as soon as possible:

```shell
./videodecodeasync -i $V/AMD_driving_virtual_20-H265.mp4 -p no-copy -w lag:8 -dd 0 \
                   -md5_check $M/AMD_driving_virtual_20-H265_yuv420.md5 -expect_mismatch
```

This prints `-expect_mismatch: PASSED, the mismatch was detected` and exits with 0. To run it through CTest from a rocDecode build directory:

```shell
ctest -R no-copy-control --output-on-failure
```

To see which frames were affected, write one MD5 per frame (`<frame index> <md5>` per line) for both policies and compare them:

```shell
./videodecodeasync -i $V/AMD_driving_virtual_20-H265.mp4 -p sync-copy -md5_frames sync.txt
./videodecodeasync -i $V/AMD_driving_virtual_20-H265.mp4 -p no-copy -w lag:8 -dd 0 -md5_frames nocopy.txt
paste -d' ' sync.txt nocopy.txt | awk '$2 != $4 {n++} END {print n + 0 " frames differ"}'
```

With `lag:8` nearly every frame differs. The last few frames match, because they are released at the end of the stream before any further decode can reuse their surfaces.
