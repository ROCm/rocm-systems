# Copyright (c) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import argparse
import numpy as np
import os
import subprocess
import sys
import json

__license__ = "MIT"
__version__ = "1.0"
__status__ = "Shipping"

# Color standard mapping: rocDecode flag -> FFmpeg in_color_matrix
COLOR_STANDARDS = {
    "bt709":     "bt709",
    "fcc":       "fcc",
    "bt470":     "bt470bg",
    "bt601":     "smpte170m",
    "smpte240m": "smpte240m",
    "bt2020":    "bt2020ncl",
    "bt2020c":   "bt2020cl",
}

def get_video_info(input_file):
    """Get video width, height, and number of frames using ffprobe."""
    cmd = [
        "ffprobe", "-v", "quiet", "-print_format", "json",
        "-show_streams", "-select_streams", "v:0", input_file
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ERROR: ffprobe failed on {input_file}")
        return None
    info = json.loads(result.stdout)
    streams = info.get("streams")
    if not isinstance(streams, list) or not streams:
        print(f"ERROR: ffprobe found no video stream in {input_file}")
        return None
    stream = streams[0]
    width = int(stream["width"])
    height = int(stream["height"])
    try:
        nb_frames = int(stream.get("nb_frames", 0))
    except (ValueError, TypeError):
        nb_frames = 0
    # Determine bit depth: check bits_per_raw_sample first, then infer from pix_fmt
    try:
        bit_depth = int(stream.get("bits_per_raw_sample", 0))
    except (ValueError, TypeError):
        bit_depth = 0
    if bit_depth == 0:
        pix_fmt = stream.get("pix_fmt", "")
        if "10" in pix_fmt:
            bit_depth = 10
        elif "12" in pix_fmt:
            bit_depth = 12
        else:
            bit_depth = 8
    return width, height, nb_frames, bit_depth

def run_rocdecode(exe_path, input_file, output_file, color_standard, num_frames, device_id):
    """Run videodecodergb to produce raw RGB output."""
    cmd = [
        exe_path,
        "-i", input_file,
        "-of", "rgb",
        "-o", output_file,
        "-cs", color_standard,
        "-f", str(num_frames),
        "-d", str(device_id)
    ]
    print(f"  rocDecode cmd: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=False)
    return result.returncode == 0

def run_ffmpeg(input_file, output_file, ffmpeg_matrix, num_frames):
    """Run FFmpeg to produce raw RGB output with matching colorspace."""
    cmd = [
        "ffmpeg", "-y", "-sws_flags", "neighbor",
        "-i", input_file,
        "-vf", f"scale=in_color_matrix={ffmpeg_matrix}:in_range=tv:out_range=pc",
        "-pix_fmt", "rgb24",
        "-f", "rawvideo",
        "-frames:v", str(num_frames),
        output_file
    ]
    print(f"  FFmpeg cmd: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  FFmpeg stderr: {result.stderr.strip()}")
    return result.returncode == 0

def compare_rgb_files(rocdec_file, ffmpeg_file, width, height, num_frames, tolerance):
    """Compare two raw RGB24 files pixel-by-pixel with tolerance.

    rocDecode pads rgb_width to even: (width + 1) & ~1
    FFmpeg outputs exact width.
    """
    rocdec_stride = ((width + 1) & ~1) * 3  # rocDecode padded stride
    ffmpeg_stride = width * 3                # FFmpeg exact stride

    rocdec_frame_size = rocdec_stride * height
    ffmpeg_frame_size = ffmpeg_stride * height

    rocdec_data = np.fromfile(rocdec_file, dtype=np.uint8)
    ffmpeg_data = np.fromfile(ffmpeg_file, dtype=np.uint8)

    rocdec_expected_size = rocdec_frame_size * num_frames
    ffmpeg_expected_size = ffmpeg_frame_size * num_frames

    actual_frames_rocdec = len(rocdec_data) // rocdec_frame_size
    actual_frames_ffmpeg = len(ffmpeg_data) // ffmpeg_frame_size
    actual_frames = min(actual_frames_rocdec, actual_frames_ffmpeg)

    if actual_frames == 0:
        return False, 0, 0.0, 0

    max_diff = 0
    total_diff = 0
    total_pixels = 0

    for f in range(actual_frames):
        rocdec_frame = rocdec_data[f * rocdec_frame_size : (f + 1) * rocdec_frame_size]
        ffmpeg_frame = ffmpeg_data[f * ffmpeg_frame_size : (f + 1) * ffmpeg_frame_size]

        rocdec_frame = rocdec_frame.reshape(height, rocdec_stride)
        ffmpeg_frame = ffmpeg_frame.reshape(height, ffmpeg_stride)

        # Compare only valid pixels (width * 3 bytes per row)
        valid_bytes = width * 3
        rocdec_valid = rocdec_frame[:, :valid_bytes].astype(np.int16)
        ffmpeg_valid = ffmpeg_frame[:, :valid_bytes].astype(np.int16)

        diff = np.abs(rocdec_valid - ffmpeg_valid)
        frame_max = int(diff.max())
        max_diff = max(max_diff, frame_max)
        total_diff += diff.sum()
        total_pixels += diff.size

    mean_diff = float(total_diff) / total_pixels if total_pixels > 0 else 0.0
    passed = max_diff <= tolerance

    return passed, max_diff, mean_diff, actual_frames

def main():
    parser = argparse.ArgumentParser(description="rocDecode Color Conversion Test: compare GPU vs FFmpeg CPU colorspace conversion")
    parser.add_argument('--rocdecode_exe', type=str, required=True,
                        help='Path to the videodecodergb executable')
    parser.add_argument('--input_file', type=str, required=True,
                        help='Path to the input video file')
    parser.add_argument('--gpu_device_id', type=int, default=0,
                        help='GPU device ID (default: 0)')
    parser.add_argument('--num_frames', type=int, default=3,
                        help='Number of frames to decode and compare (default: 3)')
    parser.add_argument('--tolerance', type=int, default=2,
                        help='Maximum per-channel pixel difference tolerance (default: 2)')
    parser.add_argument('--results_directory', type=str, default='',
                        help='Directory to store results (default: script directory)')
    parser.add_argument('--color_standards', type=str, nargs='*', default=None,
                        help='Color standards to test (default: all). Options: ' + ', '.join(COLOR_STANDARDS.keys()))

    args = parser.parse_args()

    # Validate inputs
    if not os.path.isfile(args.rocdecode_exe):
        print(f"ERROR: videodecodergb executable not found: {args.rocdecode_exe}")
        sys.exit(1)
    if not os.path.isfile(args.input_file):
        print(f"ERROR: Input video file not found: {args.input_file}")
        sys.exit(1)

    # Set up results directory
    script_path = os.path.dirname(os.path.realpath(__file__))
    results_path = args.results_directory if args.results_directory else os.path.join(script_path, 'rocDecode_colorConvert_results')
    os.makedirs(results_path, exist_ok=True)

    # Get video info
    video_info = get_video_info(args.input_file)
    if video_info is None:
        sys.exit(1)
    width, height, total_frames, bit_depth = video_info
    num_frames = min(args.num_frames, total_frames) if total_frames > 0 else args.num_frames
    print(f"Video: {args.input_file} ({width}x{height}, {bit_depth}-bit, {total_frames} frames)")
    print(f"Testing {num_frames} frame(s) with tolerance +/-{args.tolerance}\n")

    # Determine which standards to test
    standards_to_test = args.color_standards if args.color_standards else list(COLOR_STANDARDS.keys())
    for cs in standards_to_test:
        if cs not in COLOR_STANDARDS:
            print(f"ERROR: Unknown color standard '{cs}'. Options: {', '.join(COLOR_STANDARDS.keys())}")
            sys.exit(1)

    # Run tests
    pass_count = 0
    fail_count = 0
    skip_count = 0
    results = []

    print("Starting color conversion tests .....................................\n")

    # BT.2020 uses 10-bit range values; skip for 8-bit streams
    bt2020_standards = {"bt2020", "bt2020c"}

    for cs_name in standards_to_test:
        ffmpeg_matrix = COLOR_STANDARDS[cs_name]
        print(f"--- Testing color standard: {cs_name} (FFmpeg matrix: {ffmpeg_matrix}) ---")

        if cs_name in bt2020_standards and bit_depth <= 8:
            print(f"  SKIP: {cs_name} requires 10-bit content (input is {bit_depth}-bit)\n")
            skip_count += 1
            results.append((cs_name, "SKIP", f"requires 10-bit (input is {bit_depth}-bit)", 0, 0.0))
            continue
        if cs_name not in bt2020_standards and bit_depth > 8:
            print(f"  SKIP: {cs_name} requires 8-bit content (input is {bit_depth}-bit)\n")
            skip_count += 1
            results.append((cs_name, "SKIP", f"requires 8-bit (input is {bit_depth}-bit)", 0, 0.0))
            continue

        rocdec_output = os.path.join(results_path, f"rocdec_{cs_name}.rgb")
        ffmpeg_output = os.path.join(results_path, f"ffmpeg_{cs_name}.rgb")

        # Run rocDecode
        if not run_rocdecode(args.rocdecode_exe, args.input_file, rocdec_output, cs_name, num_frames, args.gpu_device_id):
            print(f"  SKIP: rocDecode failed for {cs_name}\n")
            skip_count += 1
            results.append((cs_name, "SKIP", "rocDecode failed", 0, 0.0))
            continue

        # Run FFmpeg
        if not run_ffmpeg(args.input_file, ffmpeg_output, ffmpeg_matrix, num_frames):
            print(f"  SKIP: FFmpeg failed for {cs_name}\n")
            skip_count += 1
            results.append((cs_name, "SKIP", "FFmpeg failed", 0, 0.0))
            continue

        # Compare outputs
        passed, max_diff, mean_diff, compared_frames = compare_rgb_files(
            rocdec_output, ffmpeg_output, width, height, num_frames, args.tolerance
        )

        if passed:
            print(f"  PASS: max_diff={max_diff}, mean_diff={mean_diff:.4f}, frames={compared_frames}")
            pass_count += 1
            results.append((cs_name, "PASS", "", max_diff, mean_diff))
        else:
            print(f"  FAIL: max_diff={max_diff} (tolerance={args.tolerance}), mean_diff={mean_diff:.4f}, frames={compared_frames}")
            fail_count += 1
            results.append((cs_name, "FAIL", f"max_diff={max_diff}", max_diff, mean_diff))
        print()

    # Summary
    print("=====================================================")
    print("Color Conversion Test Summary")
    print("=====================================================")
    print(f"{'Standard':<12} {'Result':<8} {'Max Diff':<10} {'Mean Diff':<12} {'Details'}")
    print("-" * 60)
    for cs_name, result, details, max_diff, mean_diff in results:
        print(f"{cs_name:<12} {result:<8} {max_diff:<10} {mean_diff:<12.4f} {details}")
    print("-" * 60)
    print(f"Passed: {pass_count}, Failed: {fail_count}, Skipped: {skip_count}")
    print(f"Results saved to: {results_path}")

    # Write results log
    with open(os.path.join(results_path, 'colorConvert_results.log'), 'w') as f:
        f.write("Color Conversion Test Results\n")
        f.write("=============================\n")
        f.write(f"Input: {args.input_file}\n")
        f.write(f"Resolution: {width}x{height}\n")
        f.write(f"Frames tested: {num_frames}\n")
        f.write(f"Tolerance: +/-{args.tolerance}\n\n")
        for cs_name, result, details, max_diff, mean_diff in results:
            f.write(f"{cs_name}: {result} (max_diff={max_diff}, mean_diff={mean_diff:.4f}) {details}\n")
        f.write(f"\nPassed: {pass_count}, Failed: {fail_count}, Skipped: {skip_count}\n")

    sys.exit(1 if fail_count > 0 else 0)

if __name__ == "__main__":
    main()
