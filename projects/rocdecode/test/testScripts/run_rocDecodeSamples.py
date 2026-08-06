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

from datetime import datetime
from subprocess import Popen, PIPE, run
import argparse
import os
import sys
import platform
import pandas as pd
from pathlib import Path

__license__ = "MIT"
__version__ = "1.0"
__status__ = "Shipping"


def run_command(command):
    try:
        result = run(command, stdout=PIPE, stderr=PIPE, text=True, check=False)
    except OSError as error:
        return str(error)
    return result.stdout.rstrip('\n')


def run_and_tee(command, log_path):
    with open(log_path, 'a') as outputLog:
        with Popen(command, stdout=PIPE, text=True) as process:
            for line in process.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                outputLog.write(line)
        return process.returncode


def write_formatted(output, f):
    f.write("````\n")
    f.write("%s\n\n" % output)
    f.write("````\n")


def strip_libtree_addresses(lib_tree):
    return lib_tree

def iter_files(path):
    file_list = path.rglob('*')
    sorted_file_list = sorted(file_list)
    for item in sorted_file_list:
        if item.is_file():
            yield item

def print_bitrate(current_file):
    file_name = str(current_file)
    mbps_pos = file_name.find("mbps")
    if (mbps_pos != -1):
        fps_pos = file_name.find("fps_")
        if (fps_pos != -1):
            frame_rate = file_name[fps_pos - 2 : fps_pos]
            bit_rate = file_name[fps_pos + 4 : mbps_pos]
        else:
            frame_rate = "n/a"
            bit_rate = "n/a"
    else:
        frame_rate = "n/a"
        bit_rate = "n/a"
    with open(outputLogPath, 'a') as outputLog:
        print("Framerate: ", frame_rate, file=outputLog)
        print("Bitrate: ", bit_rate, file=outputLog)

# Import arguments
parser = argparse.ArgumentParser()
parser.add_argument('--rocDecode_directory',   type=str, default='',
                    help='The rocDecode Directory - required')
parser.add_argument('--videodecode_exe',   type=str, default='',
                    help='Video decode sample app exe - optional')
parser.add_argument('--gpu_device_id',      type=int, default=0,
                    help='The GPU device ID that will be used to run the test on it - optional (default:0 [range:0 - N-1] N = total number of available GPUs on a machine)')
parser.add_argument('--files_directory',    type=str, default='',
                    help='The path to a directory containing one or more supported files for decoding (e.g., mp4, mov, etc.) - required')
parser.add_argument('--sample_mode',          type=int, default=0,
                    help='The sample to run - optional (default:0 [range:0-1] 0: videoDecode, 1: videoDecodePerf)')
parser.add_argument('--num_threads',          type=int, default=1,
                    help='The number of threads is only for the videoDecodePerf sample (sample_mode = 1) - optional (default:1)')
parser.add_argument('--max_num_decoded_frames',          type=int, default=0,
                    help='The max number of decoded frames. Useful for partial decoding of a long stream. - optional (default:0, meaning no limit)')
parser.add_argument('--results_directory',    type=str, default='',
                    help='The path to a directory to store results - optional')
parser.add_argument('--check_decode_status',          type=int, default=0,
                    help='Report the number of streams that have completed decoding without abortion. For decoder stability check. - optional (default:0, meaning normal performance report)')
parser.add_argument('--use_ffmpeg_demuxer',   type=int, default=1,
                    help='Indicator to use FFMPEG demuxer - optional (default:1). If set to 0, built-in bitstream reader is used.')

args = parser.parse_args()

rocDecodeDirectory = args.rocDecode_directory
gpuDeviceID = args.gpu_device_id
filesDir = args.files_directory
filesDirPath = Path(filesDir)
videoDecodeEXE = args.videodecode_exe
resultsDir = args.results_directory
sampleMode = args.sample_mode
numThreads = args.num_threads
maxNumFrames = args.max_num_decoded_frames
checkDecStatus = args.check_decode_status
useFFDemuxer = args.use_ffmpeg_demuxer
if checkDecStatus == 1:
    sampleMode = 0

if useFFDemuxer == 1:
    bsReaderOption = ''
else:
    bsReaderOption = '-no_ffmpeg_demux'

print("\nrunrocDecodeTests V"+__version__+"\n")

# rocDecode Application
scriptPath = os.path.dirname(os.path.realpath(__file__))
if videoDecodeEXE == '':
    if sampleMode == 0:
        rocDecode_exe = rocDecodeDirectory+'/samples/videoDecode/build/videodecode'
    elif sampleMode == 1:
        rocDecode_exe = rocDecodeDirectory+'/samples/videoDecodePerf/build/videodecodeperf'
else:
    rocDecode_exe = videoDecodeEXE
if resultsDir == '':
    if sampleMode == 0:
        resultsPath = scriptPath+'/rocDecode_videoDecode_results'
    elif sampleMode == 1:
        resultsPath = scriptPath+'/rocDecode_videoDecodePerf_results'
else:
    if sampleMode == 0:
        resultsPath = resultsDir+'/rocDecode_videoDecode_results'
    elif sampleMode == 1:
        resultsPath = resultsDir+'/rocDecode_videoDecodePerf_results'

run_rocDecode_app = os.path.abspath(rocDecode_exe)
os.makedirs(resultsPath, exist_ok=True)
if(os.path.isfile(run_rocDecode_app)):
    print("STATUS: rocDecode path - "+run_rocDecode_app+"\n")
else:
    print("\nERROR: rocDecode Executable Not Found\n")
    exit()

if os.path.exists(filesDir) and not os.path.isfile(filesDir):
    # Checking if the directory is empty or not
    if not os.listdir(filesDir):
        print("\nERROR: Empty directory - no videos to decode")
        exit()
else:
    print("\nERROR: The input directory path is either for a file or directory does not exist!")
    exit()

outputLogPath = os.path.join(resultsPath, 'rocDecode_output.log')
resultsCsvPath = os.path.join(resultsPath, 'rocDecode_test_results.csv')
if os.path.exists(outputLogPath):
    os.remove(outputLogPath)

if os.path.exists(resultsCsvPath):
    os.remove(resultsCsvPath)

if sampleMode == 0:
    for current_file in iter_files(filesDirPath):
        print_bitrate(current_file)

        command = [
            run_rocDecode_app,
            '-i',
            str(current_file),
            '-d',
            str(gpuDeviceID),
            '-f',
            str(maxNumFrames),
        ]
        if bsReaderOption:
            command.append(bsReaderOption)
        returnCode = run_and_tee(command, outputLogPath)
        if returnCode != 0:
            sys.exit(returnCode)
        print("\n\n")

    if checkDecStatus == 0:
        echo_1 = 'File Name, Codec, Video Size, Bit Depth, Frame rate, Bit rate (Mb/s), Total Frames, Average decoding time per frame (ms), Avg FPS'
        with open(resultsCsvPath, 'a') as resultsFile:
            print(echo_1, file=resultsFile)

        awkProgram = r'''/Framerate: / {frameRate=$2; next}
                            /Bitrate: / {bitRate=$2; next}
                            /info: Input file: / {filename=$4; next}
                            /info: Using GPU device 0 - AMD Radeon Graphics[gfx1030] on PCI bus 0d:00.0/{next}
                            /info: decoding started, please wait!/{next}
                            /Input Video Information/{next}
                            /\tCodec        : / {codec=$3; next}
                            /\tSequence     : /{next}
                            /\tCoded size   : /{next}
                            /\tDisplay area : /{next}
                            /\tChroma       : /{next}
                            /\tBit depth    : / {bitDepth=$4; next}
                            /Video Decoding Params:/{next}
                            /\tNum Surfaces : /{next}
                            /\tCrop         : /{next}
                            /\tResize       : /{videoSize=$3; next}
                            /^$/{next}
                            /info: Total pictures decoded: / {totalFrames=$5; next}
                            /info: avg decoding time per picture: /{timePerFrame=$7; next}
                            /info: avg decode FPS: / { printf("%s, %s, %s, %d, %s, %s, %d, %f, %f\n", filename, codec, videoSize, bitDepth, frameRate, bitRate, totalFrames, timePerFrame, $5) }'''
        with open(resultsCsvPath, 'a') as resultsFile:
            awkResult = run(
                ['awk', awkProgram, outputLogPath],
                stdout=resultsFile,
                check=False,
            )
        if awkResult.returncode != 0:
            sys.exit(awkResult.returncode)
elif sampleMode == 1:
    for current_file in iter_files(filesDirPath):
        print_bitrate(current_file)

        command = [
            run_rocDecode_app,
            '-i',
            str(current_file),
            '-t',
            str(numThreads),
            '-f',
            str(maxNumFrames),
        ]
        returnCode = run_and_tee(command, outputLogPath)
        if returnCode != 0:
            sys.exit(returnCode)
        print("\n\n")

    if checkDecStatus == 0:
        echo_1 = 'File Name, Num Threads, Codec, Video Size, Bit Depth, Frame rate, Bit rate (Mb/s), Total Frames, Average decoding time per frame (ms), Avg FPS'
        with open(resultsCsvPath, 'a') as resultsFile:
            print(echo_1, file=resultsFile)

        awkProgram = r'''/Framerate: / {frameRate=$2; next}
                            /Bitrate: / {bitRate=$2; next}
                            /info: Input file: / {filename=$4; next}
                            /info: Number of threads: / {numThreads=$5; next}
                            /info: Using GPU device 0 - AMD Radeon Graphics[gfx1030] on PCI bus 0d:00.0/{next}
                            /info: decoding started, please wait!/{next}
                            /Input Video Information/{next}
                            /\tCodec        : / {codec=$3; next}
                            /\tSequence     : /{next}
                            /\tCoded size   : /{next}
                            /\tDisplay area : /{next}
                            /\tChroma       : /{next}
                            /\tBit depth    : / {bitDepth=$4; next}
                            /Video Decoding Params:/{next}
                            /\tNum Surfaces : /{next}
                            /\tCrop         : /{next}
                            /\tResize       : /{videoSize=$3; next}
                            /^$/{next}
                            /info: Total pictures decoded: / {totalFrames=$5; next}
                            /info: avg decoding time per picture: /{timePerFrame=$7; next}
                            /info: avg decode FPS: / { printf("%s, %d, %s, %s, %d, %s, %s, %d, %f, %f\n", filename, numThreads, codec, videoSize, bitDepth, frameRate, bitRate, totalFrames, timePerFrame, $5) }'''
        with open(resultsCsvPath, 'a') as resultsFile:
            awkResult = run(
                ['awk', awkProgram, outputLogPath],
                stdout=resultsFile,
                check=False,
            )
        if awkResult.returncode != 0:
            sys.exit(awkResult.returncode)

# get data
if checkDecStatus == 0:
    platform_name = platform.platform()
    platform_name_fq = run_command(['hostname', '--all-fqdns'])
    platform_ip = run_command(['hostname', '-I']).rstrip()

    file_dtstr = datetime.now().strftime("%Y%m%d")
    reportFilename = 'rocDecode_report_%s_%s.md' % (platform_name, file_dtstr)
    report_dtstr = datetime.now().strftime("%Y-%m-%d %H:%M:%S %Z")
    sys_info = run_command(['inxi', '-c0', '-S'])
    cpu_info = run_command(['inxi', '-c0', '-C'])
    gpu_info = run_command(['inxi', '-c0', '-G'])
    memory_info = run_command(['inxi', '-c', '0', '-m'])
    board_info = run_command(['inxi', '-c0', '-M'])

    lib_tree = run_command(['ldd', run_rocDecode_app])
    lib_tree = strip_libtree_addresses(lib_tree)

    # Load the data
    df = pd.read_csv(resultsCsvPath)
    # Generate the markdown table
    print(df.to_markdown(index=False))

    # Write Report
    with open(reportFilename, 'w') as f:
        f.write("rocDecode app report\n")
        f.write("================================\n")
        f.write("\n")

        f.write("Generated: %s\n" % report_dtstr)
        f.write("\n")

        f.write("Platform: %s (%s)\n" % (platform_name_fq, platform_ip))
        f.write("--------\n")
        f.write("\n")

        write_formatted(sys_info, f)
        write_formatted(cpu_info, f)
        write_formatted(gpu_info, f)
        write_formatted(board_info, f)
        write_formatted(memory_info, f)

        f.write("\n\nBenchmark Report\n")
        f.write("--------\n")
        f.write("\n")
        f.write("\n")
        f.write(df.to_markdown(index=False))
        f.write("\n")
        f.write("\n")
        f.write("Dynamic Libraries Report\n")
        f.write("-----------------\n")
        f.write("\n")
        write_formatted(lib_tree, f)
        f.write("\n")

        f.write(
            "\n\n---\n**Copyright (c) 2023 - 2026 AMD ROCm rocDecode app -- run_rocDecode_tests.py V-"+__version__+"**\n")
        f.write("\n")
        # report file
        reportFileDir = os.path.abspath(reportFilename)
        print("\nSTATUS: Output Report File - "+reportFileDir)

    print("\nrun_rocDecode_tests.py completed - V"+__version__+"\n")
else:
    fileString = 'info: Input file:'
    decodeEndString = 'info: Total pictures decoded:'
    numFiles = 0
    numDecodedStreams = 0
    with open(outputLogPath, 'r') as logFile:
        line = logFile.readline()
        while line:
            if line.find(fileString) != -1:
                numFiles += 1
            if line.find(decodeEndString) != -1:
                numDecodedStreams += 1
            line = logFile.readline()

        print("Decode status report of the", numFiles, "streams:")
        print("     - The number of completely decoded streams is", numDecodedStreams)
        print("     - The number of streams that did not finish decoding is " + str(numFiles - numDecodedStreams))
    logFile.close()
    if numFiles != numDecodedStreams:
        sys.exit(-1)
    else:
        sys.exit(0)
