
import os
import re
import shutil
import subprocess
import argparse
from datetime import datetime
import csv

# ── Helpers ───────────────────────────────────────────────────────────────────

def is_resolution_dir(name):
    # Resolution folders follow the pattern <width>_<height> (e.g. 1920_1080)
    return bool(re.match(r'^\d+_\d+$', name))

def detect_path_mode(path):
    """
    Inspects the given path and returns one of four modes:
      'single'   - path points to a single .jpg file
      'category' - path is a folder whose subdirs are resolution folders (e.g. jpeg_images/flowers)
      'parent'   - path is a folder whose subdirs are category folders, each containing resolution folders (e.g. jpeg_images/)
      'flat'     - path is a folder with arbitrary images or subfolders, no resolution structure (e.g. jpeg_images/foo)
    """
    if os.path.isfile(path):
        return 'single'
    subdirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    if any(is_resolution_dir(d) for d in subdirs):
        return 'category'
    if any(
        is_resolution_dir(sub)
        for d in subdirs
        for sub in os.listdir(os.path.join(path, d))
        if os.path.isdir(os.path.join(path, d, sub))
    ):
        return 'parent'
    return 'flat'

# ── Image preparation ─────────────────────────────────────────────────────────

def copy_images(images_path, num_images, benchmark_mode):
    """
    Copies images from images_path into ~/jpeg_images/, expanding each image
    to num_images copies so the benchmark has enough data to work with.
    Handles four input layouts: single file, flat folder, category folder,
    and parent folder containing multiple categories.
    """
    home_dir = os.path.expanduser('~')
    dst_base = os.path.join(home_dir, 'jpeg_images')
    mode = detect_path_mode(images_path)
    resolutions = ['1920_1080', '3840_2160'] if benchmark_mode == 0 else \
                  ['640_480', '800_600', '1920_1080', '3840_2160', '3991_2661', '7680_4320', '16384_16384']

    if mode == 'single':
        # Create ~/jpeg_images/<image_stem>/ and fill it with num_images copies of the file
        base_name = os.path.splitext(os.path.basename(images_path))[0]
        dst_dir = os.path.join(dst_base, base_name)
        existing = len([f for f in os.listdir(dst_dir) if f.endswith('.jpg')]) if os.path.exists(dst_dir) else 0
        if existing >= num_images:
            print(f"Skipping copy: {dst_dir} already has {existing} images.")
            return
        os.makedirs(dst_dir, exist_ok=True)
        print(f"Copying single image {images_path} to {dst_dir} ({num_images} times)")
        for i in range(num_images):
            shutil.copy(images_path, os.path.join(dst_dir, f'{base_name}_{i}.jpg'))
        return

    if mode == 'flat':
        # Copy the source folder into ~/jpeg_images/<folder_name>/ then duplicate
        # each image so the total per-image count reaches num_images
        folder_name = os.path.basename(images_path.rstrip('/'))
        dst_dir = os.path.join(dst_base, folder_name)
        src_images = [f for f in os.listdir(images_path) if f.endswith('.jpg')]
        expected = len(src_images) * num_images
        existing = len([f for f in os.listdir(dst_dir) if f.endswith('.jpg')]) if os.path.exists(dst_dir) else 0
        if existing >= expected:
            print(f"Skipping copy: {dst_dir} already has {existing} images.")
            return
        os.makedirs(dst_dir, exist_ok=True)
        print(f"Copying flat images from {images_path} to {dst_dir}")
        # Copy originals first
        for img in src_images:
            shutil.copy(os.path.join(images_path, img), os.path.join(dst_dir, img))
        # Add num_images - 1 duplicates per image (original already counts as 1)
        for img in src_images:
            base_stem = os.path.splitext(img)[0]
            src_img = os.path.join(images_path, img)
            for i in range(num_images - 1):
                shutil.copy(src_img, os.path.join(dst_dir, f'{base_stem}_{i}.jpg'))
        return

    # parent mode: iterate each category subdir; category mode: treat images_path itself as the category
    if mode == 'parent':
        categories = [d for d in os.listdir(images_path) if os.path.isdir(os.path.join(images_path, d))]
    else:  # category
        categories = [None]  # None sentinel means images_path itself is the category dir

    for category in categories:
        src_cat_dir = os.path.join(images_path, category) if category else images_path
        dst_cat_dir = os.path.join(dst_base, category if category else os.path.basename(images_path.rstrip(os.sep)))

        for resolution in resolutions:
            src_dir = os.path.join(src_cat_dir, resolution)
            dst_dir = os.path.join(dst_cat_dir, resolution)
            if not os.path.exists(src_dir):
                continue
            src_images = [f for f in os.listdir(src_dir) if f.endswith('.jpg')]
            expected = len(src_images) * num_images
            existing = len([f for f in os.listdir(dst_dir) if f.endswith('.jpg')]) if os.path.exists(dst_dir) else 0
            if existing >= expected:
                print(f"Skipping copy: {dst_dir} already has {existing} images.")
                continue
            os.makedirs(dst_dir, exist_ok=True)
            print(f"Copying images from {src_dir} to {dst_dir}")
            # Copy originals first
            for img in src_images:
                shutil.copy(os.path.join(src_dir, img), os.path.join(dst_dir, img))
            # Add num_images - 1 duplicates per source image (original counts as 1)
            for img in src_images:
                base_stem = os.path.splitext(img)[0]
                src_img = os.path.join(src_dir, img)
                for i in range(num_images - 1):
                    shutil.copy(src_img, os.path.join(dst_dir, f'{base_stem}_{i}.jpg'))

# ── Build ─────────────────────────────────────────────────────────────────────

def build_jpegdecodeperf(rocm_path):
    """
    Builds the jpegdecodeperf sample app from source using cmake and make.
    Skips the build if the binary already exists in ~/jpegdecodeperf-test/.
    If the directory exists but the binary is missing, it is wiped first to
    ensure a clean build.
    """
    home_dir = os.path.expanduser('~')
    build_dir = os.path.join(home_dir, 'jpegdecodeperf-test')
    if os.path.exists(build_dir) and os.path.exists(os.path.join(build_dir, 'jpegdecodeperf')):
        print("jpegdecodeperf is already built.")
    else:
        # Clean any stale build artifacts before starting fresh
        if os.path.exists(build_dir):
            shutil.rmtree(build_dir)
        os.makedirs(build_dir)
        cmake_source = os.path.join(rocm_path, 'share/rocjpeg/samples/jpegDecodePerf/')
        env = {**os.environ, 'ROCM_PATH': rocm_path}
        subprocess.run(['cmake', cmake_source], check=True, env=env, cwd=build_dir)
        subprocess.run(['make', '-j8'], check=True, env=env, cwd=build_dir)

# ── Benchmark execution ───────────────────────────────────────────────────────

def run_one(dir_path, batch_size, num_threads, device_id, rocm_path, output_format=None, crop=None, output_path=None):
    """
    Runs jpegdecodeperf on dir_path three times and returns the metrics from
    the best run (highest IPS). Parsed from app stdout:
      Images/Sec, Mpixels/Sec, Average processing time per image, Total wall time.
    """
    home_dir = os.path.expanduser('~')
    exe = os.path.join(home_dir, 'jpegdecodeperf-test/jpegdecodeperf')
    env = {**os.environ, 'ROCM_PATH': rocm_path}
    best_ips, best_mps, best_proc_time, best_wall_time, total_images = 0, 0, 0, 0, 0

    base_cmd = [exe, '-i', dir_path, '-b', str(batch_size), '-t', str(num_threads), '-d', str(device_id)]
    if output_format:
        base_cmd += ['-fmt', output_format]
    if crop:
        base_cmd += ['-crop', crop]
    if output_path:
        base_cmd += ['-o', output_path]

    for _ in range(3):
        print(f"Decoding images in {dir_path}")
        result = subprocess.run(
            base_cmd,
            capture_output=True, text=True, env=env
        )
        if result.returncode != 0:
            print(result.stderr)
            raise RuntimeError(
                f"jpegdecodeperf failed (exit {result.returncode}) on {dir_path}"
            )
        print(result.stdout)

        ips, mps, proc_time, wall_time, images = 0, 0, 0, 0, 0
        for line in result.stdout.split('\n'):
            if 'Images/Sec' in line:
                ips = float(line.split(':')[-1].strip())
            if 'Mpixels/Sec' in line:
                mps = float(line.split(':')[-1].strip())
            if 'Average processing time per image' in line:
                proc_time = float(line.split(':')[-1].strip())
            if 'Total wall time' in line:
                wall_time = float(line.split(':')[-1].strip())
            if 'Total Images' in line or 'Number of Images' in line:
                try:
                    images = int(line.split(':')[-1].strip())
                except ValueError:
                    pass

        # Keep only the metrics from the best-performing run
        if ips > best_ips:
            best_ips = ips
            best_mps = mps
            best_proc_time = proc_time
            best_wall_time = wall_time
            total_images = images

    return best_ips, best_mps, best_proc_time, best_wall_time, total_images

def run_benchmark(images_path, batch_size, num_threads, num_images, device_id, benchmark_mode, rocm_path, direct=False, output_format=None, crop=None, output_path=None):
    """
    Drives run_one() across all relevant directories and collects results for
    every category/resolution combination.
    When direct=True (i.e. -n 1), images_path is passed straight to the app
    without any remapping to ~/jpeg_images/.
    """
    home_dir = os.path.expanduser('~')
    dst_base = os.path.join(home_dir, 'jpeg_images')
    mode = detect_path_mode(images_path)
    resolutions = ['1920_1080', '3840_2160'] if benchmark_mode == 0 else \
                  ['640_480', '800_600', '1920_1080', '3840_2160', '3991_2661', '7680_4320', '16384_16384']
    results = []
    decode_type = output_format if output_format else 'native'

    if mode == 'parent':
        # Benchmark every category found under the parent folder
        categories = sorted(d for d in os.listdir(images_path) if os.path.isdir(os.path.join(images_path, d)))
        for category in categories:
            cat_dir = os.path.join(images_path, category) if direct else os.path.join(dst_base, category)
            for resolution in resolutions:
                dir_path = os.path.join(cat_dir, resolution)
                if not os.path.exists(dir_path):
                    continue
                total = sum(1 for f in os.listdir(dir_path) if f.endswith('.jpg'))
                ips, mps, proc_time, wall_time, _ = run_one(dir_path, batch_size, num_threads, device_id, rocm_path, output_format, crop, output_path)
                results.append([category, decode_type, resolution.replace('_', 'x'), total, proc_time, ips, mps, wall_time])

    elif mode == 'category':
        # Benchmark each resolution subfolder within the single category
        category = os.path.basename(images_path.rstrip('/'))
        cat_dir = images_path if direct else os.path.join(dst_base, category)
        for resolution in resolutions:
            dir_path = os.path.join(cat_dir, resolution)
            if not os.path.exists(dir_path):
                continue
            total = sum(1 for f in os.listdir(dir_path) if f.endswith('.jpg'))
            ips, mps, proc_time, wall_time, _ = run_one(dir_path, batch_size, num_threads, device_id, rocm_path, output_format, crop, output_path)
            results.append([category, decode_type, resolution.replace('_', 'x'), total, proc_time, ips, mps, wall_time])

    else:  # flat or single — benchmark the whole folder (or file's parent folder) as one unit
        if direct:
            # Use images_path directly: resolve to containing folder if it's a single file
            dir_path = os.path.dirname(images_path) if mode == 'single' else images_path.rstrip('/')
            label = os.path.basename(dir_path)
        else:
            label = os.path.splitext(os.path.basename(images_path))[0] if mode == 'single' \
                    else os.path.basename(images_path.rstrip('/'))
            dir_path = os.path.join(dst_base, label)
        # Count files directly since the app output doesn't reliably report the total
        total = sum(1 for f in os.listdir(dir_path) if f.endswith('.jpg'))
        ips, mps, proc_time, wall_time, _ = run_one(dir_path, batch_size, num_threads, device_id, rocm_path, output_format, crop, output_path)
        results.append([label, decode_type, 'mixed', total, proc_time, ips, mps, wall_time])

    return results

# ── Results persistence ───────────────────────────────────────────────────────

def save_results(results, batch_size, num_threads, device_id):
    """
    Writes benchmark results to a timestamped CSV file inside
    ~/jpeg_decoding_benchmark_results/. The directory is created on first run.
    Returns the full path to the saved file.
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = os.path.join(os.path.expanduser('~'), 'jpeg_decoding_benchmark_results')
    os.makedirs(results_dir, exist_ok=True)
    csv_filename = os.path.join(
        results_dir,
        f'benchmark_results_{timestamp}_batch_size_{batch_size}_num_threads_{num_threads}_device_id_{device_id}.csv'
    )
    with open(csv_filename, 'w', newline='') as csvfile:
        fieldnames = [
            'Image Category', 'Decode Type', 'Resolution', 'Number of Images',
            'Processing Time Per Image (ms)', 'Images Per Second (IPS)',
            'MPixels/sec', 'Total Wall Time (sec)'
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow({
                'Image Category':                result[0],
                'Decode Type':                   result[1],
                'Resolution':                    result[2],
                'Number of Images':              result[3],
                'Processing Time Per Image (ms)':result[4],
                'Images Per Second (IPS)':      result[5],
                'MPixels/sec':                   result[6],
                'Total Wall Time (sec)':         result[7],
            })
    return csv_filename

# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_images_path = os.path.join(script_dir, 'jpeg_images', 'flowers')

    parser = argparse.ArgumentParser(description='Benchmark JPEG decoding using AMD ROCm rocJPEG APIs.')
    parser.add_argument('--rocm_path', type=str, required=True,
                        help='Path to the ROCm installation (e.g. /opt/rocm)')
    parser.add_argument('-i', '--images_path', type=str, default=default_images_path,
                        help='Path to images to benchmark. Can be a single .jpg file, a category folder '
                             '(e.g. jpeg_images/flowers), a parent folder (e.g. jpeg_images/), or a flat '
                             f'folder of images. (default: {default_images_path})')
    parser.add_argument('-n', '--num_images', type=int, default=1,
                        help='Number of images per source image to copy into the benchmark folder (default: 1)')
    parser.add_argument('-b', '--batch_size', type=int, default=32,
                        help='Batch size for benchmarking (default: 32)')
    parser.add_argument('-t', '--num_threads', type=int, default=1,
                        help='Number of threads for benchmarking (default: 1)')
    parser.add_argument('-d', '--device_id', type=int, default=0,
                        help='Device id for the GPU for benchmarking (default: 0)')
    parser.add_argument('-m', '--benchmark_mode', type=int, default=0,
                        help='Benchmark mode: 0 for light (2 resolutions), 1 for full (7 resolutions) (default: 0)')
    parser.add_argument('-fmt', '--output_format', type=str, default=None,
                        choices=['native', 'yuv_planar', 'y', 'rgb', 'rgb_planar'],
                        help='rocJPEG output format for decoding: native, yuv_planar, y, rgb, rgb_planar (default: native)')
    parser.add_argument('-crop', '--crop', type=str, default=None,
                        help='Crop rectangle for output in comma-separated format: left,top,right,bottom (optional)')
    parser.add_argument('-o', '--output_path', type=str, default=None,
                        help='Path to an output file or directory to write decoded images (optional)')
    args = parser.parse_args()

    if args.num_images == 1:
        # Skip copying entirely and benchmark the source images_path directly
        print("num_images=1: running benchmark directly on --images_path, skipping copy.")
        direct = True
    else:
        copy_images(args.images_path, args.num_images, args.benchmark_mode)
        direct = False
    build_jpegdecodeperf(args.rocm_path)
    results = run_benchmark(args.images_path, args.batch_size, args.num_threads, args.num_images, args.device_id, args.benchmark_mode, args.rocm_path, direct=direct, output_format=args.output_format, crop=args.crop, output_path=args.output_path)
    csv_path = save_results(results, args.batch_size, args.num_threads, args.device_id)
    print(f"Results saved to: {csv_path}")
    print("Benchmarking completed!")

if __name__ == '__main__':
    main()
