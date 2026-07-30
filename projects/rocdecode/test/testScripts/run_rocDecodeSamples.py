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
from subprocess import Popen, PIPE
import argparse
import os
import shutil
import sys
import platform
import glob
import pandas as pd
from pathlib import Path

__license__ = "MIT"
__version__ = "1.0"
__status__ = "Shipping"


def shell(cmd):
    p = Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE)
    output = p.communicate()[0][0:-1]
    return output


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
    orig_stdout = sys.stdout
    sys.stdout = open(resultsPath+'/rocDecode_output.log', 'a')
    print("Framerate: ", frame_rate)
    print("Bitrate: ", bit_rate)
    sys.stdout = orig_stdout

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
        if platform.system() == 'Windows':
            rocDecode_exe = rocDecodeDirectory+'/samples/videoDecode/build/Release/videodecode.exe'
        else:
            rocDecode_exe = rocDecodeDirectory+'/samples/videoDecode/build/videodecode'
    elif sampleMode == 1:
        if platform.system() == 'Windows':
            rocDecode_exe = rocDecodeDirectory+'/samples/videoDecodePerf/build/Release/videodecodeperf.exe'
        else:
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

# Get cwd
cwd = os.getcwd()
if os.path.exists(resultsPath+'/rocDecode_output.log'):
    os.remove(resultsPath+'/rocDecode_output.log')

if os.path.exists(resultsPath+'/rocDecode_test_results.csv'):
    os.remove(resultsPath+'/rocDecode_test_results.csv')

if sampleMode == 0:
    for current_file in iter_files(filesDirPath):
        print_bitrate(current_file)

        cmd = run_rocDecode_app + ' -i ' + str(current_file) + ' -d ' + str(gpuDeviceID) + ' -f ' + str(maxNumFrames) + ' ' + str(bsReaderOption)
        logFilePath = resultsPath+'/rocDecode_output.log'
        process = Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE, text=True)
        stdout, stderr = process.communicate()
        print(stdout)
        if stderr:
            print(stderr, file=sys.stderr)
        with open(logFilePath, 'a') as logf:
            logf.write(stdout)
            if stderr:
                logf.write(stderr)
        print("\n\n")

    if checkDecStatus == 0:
        orig_stdout = sys.stdout
        sys.stdout = open(resultsPath+'/rocDecode_test_results.csv', 'a')
        echo_1 = 'File Name, Codec, Video Size, Bit Depth, Frame rate, Bit rate (Mb/s), Total Frames, Average decoding time per frame (ms), Avg FPS'
        print(echo_1)
        sys.stdout = orig_stdout

        with open(resultsPath+'/rocDecode_output.log', 'r') as lf:
            frameRate = bitRate = filename = codec = videoSize = bitDepth = totalFrames = timePerFrame = 'n/a'
            csvf = open(resultsPath+'/rocDecode_test_results.csv', 'a')
            for line in lf:
                if 'Framerate: ' in line:
                    frameRate = line.split()[1] if len(line.split()) > 1 else 'n/a'
                elif 'Bitrate: ' in line:
                    bitRate = line.split()[1] if len(line.split()) > 1 else 'n/a'
                elif 'info: Input file: ' in line:
                    filename = line.split()[3] if len(line.split()) > 3 else 'n/a'
                elif '\tCodec        : ' in line:
                    codec = line.split()[2] if len(line.split()) > 2 else 'n/a'
                elif '\tBit depth    : ' in line:
                    bitDepth = line.split()[3] if len(line.split()) > 3 else 'n/a'
                elif '\tResize       : ' in line:
                    videoSize = line.split()[2] if len(line.split()) > 2 else 'n/a'
                elif 'info: Total pictures decoded: ' in line:
                    totalFrames = line.split()[4] if len(line.split()) > 4 else 'n/a'
                elif 'info: avg decoding time per picture: ' in line:
                    timePerFrame = line.split()[6] if len(line.split()) > 6 else 'n/a'
                elif 'info: avg decode FPS: ' in line:
                    avgFPS = line.split()[4] if len(line.split()) > 4 else 'n/a'
                    csvf.write('%s, %s, %s, %s, %s, %s, %s, %s, %s\n' % (filename, codec, videoSize, bitDepth, frameRate, bitRate, totalFrames, timePerFrame, avgFPS))
            csvf.close()
elif sampleMode == 1:
    for current_file in iter_files(filesDirPath):
        print_bitrate(current_file)

        cmd = run_rocDecode_app+' -i '+str(current_file)+' -t '+str(numThreads)+' -f '+str(maxNumFrames)
        logFilePath = resultsPath+'/rocDecode_output.log'
        process = Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE, text=True)
        stdout, stderr = process.communicate()
        print(stdout)
        if stderr:
            print(stderr, file=sys.stderr)
        with open(logFilePath, 'a') as logf:
            logf.write(stdout)
            if stderr:
                logf.write(stderr)
        print("\n\n")

    if checkDecStatus == 0:
        orig_stdout = sys.stdout
        sys.stdout = open(resultsPath+'/rocDecode_test_results.csv', 'a')
        echo_1 = 'File Name, Num Threads, Codec, Video Size, Bit Depth, Frame rate, Bit rate (Mb/s), Total Frames, Average decoding time per frame (ms), Avg FPS'
        print(echo_1)
        sys.stdout = orig_stdout

        with open(resultsPath+'/rocDecode_output.log', 'r') as lf:
            frameRate = bitRate = filename = codec = videoSize = bitDepth = totalFrames = timePerFrame = numThr = 'n/a'
            csvf = open(resultsPath+'/rocDecode_test_results.csv', 'a')
            for line in lf:
                if 'Framerate: ' in line:
                    frameRate = line.split()[1] if len(line.split()) > 1 else 'n/a'
                elif 'Bitrate: ' in line:
                    bitRate = line.split()[1] if len(line.split()) > 1 else 'n/a'
                elif 'info: Input file: ' in line:
                    filename = line.split()[3] if len(line.split()) > 3 else 'n/a'
                elif 'info: Number of threads: ' in line:
                    numThr = line.split()[4] if len(line.split()) > 4 else 'n/a'
                elif '\tCodec        : ' in line:
                    codec = line.split()[2] if len(line.split()) > 2 else 'n/a'
                elif '\tBit depth    : ' in line:
                    bitDepth = line.split()[3] if len(line.split()) > 3 else 'n/a'
                elif '\tResize       : ' in line:
                    videoSize = line.split()[2] if len(line.split()) > 2 else 'n/a'
                elif 'info: Total pictures decoded: ' in line:
                    totalFrames = line.split()[4] if len(line.split()) > 4 else 'n/a'
                elif 'info: avg decoding time per picture: ' in line:
                    timePerFrame = line.split()[6] if len(line.split()) > 6 else 'n/a'
                elif 'info: avg decode FPS: ' in line:
                    avgFPS = line.split()[4] if len(line.split()) > 4 else 'n/a'
                    csvf.write('%s, %s, %s, %s, %s, %s, %s, %s, %s, %s\n' % (filename, numThr, codec, videoSize, bitDepth, frameRate, bitRate, totalFrames, timePerFrame, avgFPS))
            csvf.close()

# get data
if checkDecStatus == 0:
    platform_name = platform.platform()
    if platform.system() == 'Windows':
        platform_name_fq = shell('hostname')
        platform_ip = b'N/A'
    else:
        platform_name_fq = shell('hostname --all-fqdns')
        platform_ip = shell('hostname -I')[0:-1]  # extra trailing space

    file_dtstr = datetime.now().strftime("%Y%m%d")
    reportFilename = 'rocDecode_report_%s_%s.md' % (platform_name, file_dtstr)
    report_dtstr = datetime.now().strftime("%Y-%m-%d %H:%M:%S %Z")
    if platform.system() == 'Windows':
        sys_info = shell('systeminfo')
        cpu_info = shell('wmic cpu get Name')
        gpu_info = shell('wmic path win32_VideoController get Name')
        memory_info = shell('wmic ComputerSystem get TotalPhysicalMemory')
        board_info = shell('wmic baseboard get product,manufacturer')
        lib_tree = b'N/A (use dumpbin /dependents on Windows)'
    else:
        sys_info = shell('inxi -c0 -S')
        cpu_info = shell('inxi -c0 -C')
        gpu_info = shell('inxi -c0 -G')
        memory_info = shell('inxi -c 0 -m')
        board_info = shell('inxi -c0 -M')
        lib_tree = shell('ldd '+run_rocDecode_app)
    lib_tree = strip_libtree_addresses(lib_tree)

    # Load the data
    df = pd.read_csv(resultsPath+'/rocDecode_test_results.csv')
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
    with open(resultsPath + '/rocDecode_output.log', 'r') as logFile:
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
