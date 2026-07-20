# jpegDecodeAsync

This sample demonstrates the asynchronous JPEG decoding API using `rocJpegDecodeAsync` and `rocJpegDecodeSync`. It uses a threaded pipeline where decode submissions and completions are separated across threads for improved throughput.

## Usage

```bash
./jpegdecodeasync -i <input_path> [-o <output_path>] [-be <backend>] [-d <device_id>] [-fmt <format>] [-crop <rect>] [-n <iterations>]
```

## Options

- `-i [input path]` - Input path to a single JPEG image (required)
- `-o [output path]` - Path to save the decoded output image (optional)
- `-be [backend]` - Select rocJPEG backend: 0 for hardware-accelerated JPEG decoding using VCN (default: 0)
- `-d [device_id]` - Specify the GPU device ID (default: 0)
- `-fmt [output format]` - Output format: native, yuv_planar, y, rgb, rgb_planar (optional, default: native)
- `-crop [crop rectangle]` - Crop rectangle: left,top,right,bottom (optional)
- `-n [iterations]` - Number of decode iterations per image (optional, default: 1)

## Pipeline Architecture

The sample uses a producer-consumer pipeline:
- **Main thread (producer)**: Reads JPEG files and submits decode operations via `rocJpegDecodeAsync`
- **Sync thread (consumer)**: Waits for decode completions via `rocJpegDecodeSync`

This enables true cross-image async pipelining — submitting image N+1 while image N is still being synchronized.
