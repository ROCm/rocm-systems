# Video decode capabilities sample

This sample queries and prints the hardware video decode capabilities of an AMD GPU using the `rocDecGetDecoderCaps` API from the rocDecode library.

For the selected device, the sample iterates over every video codec (`MPEG1`, `MPEG2`, `MPEG4`, `AVC/H.264`, `HEVC/H.265`, `AV1`, `VP8`, `VP9`), chroma format (Monochrome, 4:2:0, 4:2:2, 4:4:4), and bit depth (8/10/12-bit). For every supported combination it reports the fields exposed by the `RocdecDecodeCaps` struct:

- number of decoders that support the parameters
- minimum coded width x height
- maximum coded width x height
- supported output surface formats (decoded from `output_format_mask`)

Unlike the other samples, this app does not decode a stream, so it needs no input file and does not depend on FFmpeg.

## Prerequisites

* Install [rocDecode](https://rocm.docs.amd.com/projects/rocDecode/en/latest/install/rocDecode-build-and-install.html)

## Build

**Linux:**

```shell
mkdir build && cd build
cmake ../
make -j
```

**Windows:**

```bat
mkdir build && cd build
cmake .. -DROCM_PATH=<path-to-TheRock-build>
cmake --build . --config Release
```

> [!NOTE]
> Before running, add the rocDecode DLL directory to your PATH:
> ```bat
> set PATH=%ROCM_PATH%\bin;%PATH%
> ```

## Run

```shell
./videodecodecaps -d <GPU device ID - 0:default>
```

### Options

- `-d`: GPU device ID (0 for the first device, 1 for the second, etc.); optional; default: 0
