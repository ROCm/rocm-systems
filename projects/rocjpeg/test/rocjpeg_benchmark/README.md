# rocJPEG Benchmark

A Python harness that builds the `jpegdecodeperf` sample application, optionally prepares a large image dataset, runs the decoder multiple times, and writes results to a timestamped CSV file.

---

## Requirements

- Python 3.6+
- ROCm installed (e.g. `<path_to_rocm_installation>`)
- `cmake` and `make` available in `PATH`
- JPEG images to benchmark

---

## Quick start

```bash
python3 rocjpeg_benchmark.py --rocm_path <path_to_rocm_installation> -i <path_to_jpeg_images>
```

rocJPEG installs this script and README in `<path_to_rocm_installation>/share/rocjpeg/test/rocjpeg_benchmark/`.

---

## Arguments

### Required

| Argument | Short | Description |
|---|---|---|
| `--rocm_path PATH` | | Path to the ROCm installation directory (e.g. `<path_to_rocm_installation>`). Used to locate the `jpegDecodePerf` CMake source and passed as `ROCM_PATH` to the build and runtime environment. |
| `--images_path PATH` | `-i` | Path to the JPEG images to benchmark. Accepts four layouts (see [Input path layouts](#input-path-layouts) below). |

### Benchmark tuning

| Argument | Short | Default | Description |
|---|---|---|---|
| `--num_images N` | `-n` | `1` | Number of copies of each source image to place in the benchmark working directory. When set to `1` (default) no copying is performed and `--images_path` is used directly. Increase this value to ensure the decoder has enough data to produce stable throughput numbers (e.g. `-n 100`). |
| `--num_iterations N` | `-iter` | `1` | Number of times `jpegdecodeperf` decodes each image in a run, passed via its `-n` flag. Like `--num_images`, it lengthens runs for stable throughput numbers, but without copying images to disk. Requires rocJPEG 1.14.0 or later; the script stops with an error if `jpegdecodeperf` decodes each image only once. |
| `--batch_size N` | `-b` | `32` | Batch size passed to `jpegdecodeperf` via its `-b` flag. |
| `--num_threads N` | `-t` | `1` | Number of decoder threads passed to `jpegdecodeperf` via its `-t` flag. |
| `--device_id N` | `-d` | `0` | GPU device ID passed to `jpegdecodeperf` via its `-d` flag. |
| `--benchmark_mode N` | `-m` | `0` | Controls which resolutions are exercised:<br>`0` — light mode: `1920x1080`, `3840x2160`<br>`1` — full mode: `640x480`, `800x600`, `1920x1080`, `3840x2160`, `3991x2661`, `7680x4320`, `16384x16384` |

### Decoder options (passed directly to `jpegdecodeperf`)

| Argument | Short | Default | Description |
|---|---|---|---|
| `--output_format FMT` | `-fmt` | `native` | Output pixel format for decoding. One of:<br>`native` — keep the format the JPEG was encoded in (default)<br>`yuv_planar` — planar YUV (Y, U, V planes)<br>`y` — luminance (Y plane) only<br>`rgb` — packed RGB<br>`rgb_planar` — planar RGB (R, G, B planes) |
| `--crop L,T,R,B` | `-crop` | *(none)* | Crop rectangle applied to each decoded image, specified as four comma-separated integers: `left,top,right,bottom`. Example: `-crop 0,0,1280,720`. Images that the rectangle does not fit in are decoded in full. |
| `--output_path PATH` | `-o` | *(none)* | Existing directory in which `jpegdecodeperf` saves each decoded image, as `<name>_<width>x<height>_<format>.<ext>`. The script stops with an error if the directory does not exist. When omitted, `jpegdecodeperf` does not write output files. |

---

## Input path layouts

The script auto-detects the layout of `--images_path` and handles it appropriately.

| Layout | Description | Example |
|---|---|---|
| **single** | A single `.jpg` file | `<path_to_jpeg_file>` |
| **flat** | A directory containing `.jpg` files with no resolution subdirectories | `<path_to_jpeg_images>/` |
| **category** | A directory whose immediate subdirectories are resolution folders named `<width>_<height>` | `<path_to_jpeg_images>/` containing `1920_1080/`, `3840_2160/`, … |
| **parent** | A directory whose immediate subdirectories are category folders, each containing resolution folders | `<path_to_jpeg_images>/` containing `<category_1>/1920_1080/`, `<category_2>/3840_2160/`, … |

Resolution folder names must follow the pattern `<width>_<height>` (e.g. `1920_1080`) to be recognized.

---

## Output

Results are written to `~/jpeg_decoding_benchmark_results/` as a timestamped CSV:

```
benchmark_results_<YYYYMMDD_HHMMSS>_batch_size_<B>_num_threads_<T>_device_id_<D>.csv
```

When `--num_iterations` is greater than `1`, the file name also records it:

```
benchmark_results_<YYYYMMDD_HHMMSS>_batch_size_<B>_num_threads_<T>_device_id_<D>_num_iterations_<N>.csv
```

### CSV columns

| Column | Description |
|---|---|
| Image Category | Category or folder name |
| Decode Type | Output format used for decoding — reflects the `-fmt` argument (e.g. `rgb`, `yuv_planar`); defaults to `native` when `-fmt` is not specified |
| Resolution | Image resolution string (e.g. `1920x1080` or `mixed` for flat/single layouts) |
| Number of Images | Number of images decoded, as reported by `jpegdecodeperf`. Each image counts once per `--num_iterations`; images that `jpegdecodeperf` skips do not count. |
| Processing Time Per Image (ms) | Average decode time per image in milliseconds |
| Images Per Second (IPS) | Decoder throughput |
| MPixels/sec | Megapixels per second |
| Total Wall Time (sec) | End-to-end wall-clock time for the benchmark run |

Each directory is benchmarked three times and the best result (highest IPS) is recorded.

---

## Examples

### Minimal — run directly on existing images

```bash
python3 rocjpeg_benchmark.py \
    --rocm_path <path_to_rocm_installation> \
    -i <path_to_jpeg_images>
```

### Full benchmark with 100 image copies, 4 threads, batch size 64

```bash
python3 rocjpeg_benchmark.py \
    --rocm_path <path_to_rocm_installation> \
    -i <path_to_jpeg_images> \
    -n 100 \
    -b 64 \
    -t 4 \
    -m 1
```

### Decode each image 50 times without copying images

```bash
python3 rocjpeg_benchmark.py \
    --rocm_path <path_to_rocm_installation> \
    -i <path_to_jpeg_images> \
    -iter 50
```

### Decode to RGB format with a crop region

```bash
python3 rocjpeg_benchmark.py \
    --rocm_path <path_to_rocm_installation> \
    -i <path_to_jpeg_images> \
    -fmt rgb \
    -crop 0,0,1280,720
```

### Save decoded output images to a directory

```bash
python3 rocjpeg_benchmark.py \
    --rocm_path <path_to_rocm_installation> \
    -i <path_to_jpeg_images> \
    -fmt yuv_planar \
    -o <path_to_output_directory>
```

### Use a specific GPU and decode only the Y (luminance) channel

```bash
python3 rocjpeg_benchmark.py \
    --rocm_path <path_to_rocm_installation> \
    -i <path_to_jpeg_file> \
    -d 1 \
    -fmt y
```

---

## Performance tuning suggestions

For achieving optimal JPEG decoding performance:

- **Choose a batch size that matches the available JPEG decode cores.** The GPU exposes a fixed number of hardware JPEG decode cores. Setting `-b` to match that count keeps all cores busy without over-subscribing them. Exceeding the core count adds scheduling overhead with no throughput gain.

- **Use multiple host threads to further scale performance.** Once the optimal batch size has been selected, increase the thread count with `-t` to overlap host-side work (I/O, memory transfers) with GPU decoding.

```bash
# Example: 32 batch size with 4 threads
python3 rocjpeg_benchmark.py \
    --rocm_path <path_to_rocm_installation> \
    -i <path_to_jpeg_images> \
    -b 32 \
    -t 4
```

---

## How it works

1. **Image preparation** — when `-n` is greater than 1, source images are copied into `~/jpeg_images/` and duplicated so each resolution folder contains at least `-n` images. This step is skipped when `-n 1` (default).
2. **Build** — the `jpegDecodePerf` CMake project under `<rocm_path>/share/rocjpeg/samples/jpegDecodePerf/` is configured and built into `~/jpegdecodeperf-test/`. Subsequent runs reuse the existing binary; delete `~/jpegdecodeperf-test/` to rebuild it, for example after updating ROCm.
3. **Benchmark** — each relevant directory is decoded three times, and each run decodes every image `--num_iterations` times; the run with the highest IPS is kept.
4. **Results** — metrics are written to a timestamped CSV in `~/jpeg_decoding_benchmark_results/`.
