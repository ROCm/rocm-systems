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

from subprocess import Popen, PIPE, STDOUT
import argparse
import os
import sys
import platform

__license__ = "MIT"
__version__ = "1.0"
__status__ = "Shipping"


def run_and_log(cmd, logFilePath):
    # Portable replacement for '| tee -a': stream the output line by line so
    # progress stays visible during long runs, while appending it to the log.
    # cmd is an argument list launched without a shell, so paths holding spaces
    # or shell metacharacters reach the sample unmangled.
    # universal_newlines (not the 3.7+ 'text' alias) keeps this working on the
    # python3 3.6 that ships with SLES 15 and RHEL 8.
    with open(logFilePath, 'a') as logf:
        p = Popen(cmd, shell=False, stdout=PIPE, stderr=STDOUT, universal_newlines=True)
        for line in p.stdout:
            print(line, end='')
            logf.write(line)
        p.wait()
    return p.returncode


# Import arguments
parser = argparse.ArgumentParser()
parser.add_argument('--rocDecode_directory',   type=str, default='',
                    help='The rocDecode Directory - required')
parser.add_argument('--videodecode_exe',   type=str, default='',
                    help='Video decode sample app exe - optional. If omitted, the sample is looked up under <rocDecode_directory>/samples/videoDecode/build (on Windows, the Release config). Pass this explicitly to run a Debug build or an exe from another location.')
parser.add_argument('--gpu_device_id',      type=int, default=0,
                    help='The GPU device ID that will be used to run the test on it - optional (default:0 [range:0 - N-1] N = total number of available GPUs on a machine)')
parser.add_argument('--files_directory',    type=str, default='',
                    help='The path to a directory containing one or more supported files for decoding (e.g., mp4, mov, etc.) and their corresponding reference MD5 digests - required')
parser.add_argument('--results_directory',    type=str, default='',
                    help='The path to a directory to store results - optional')
parser.add_argument('--use_ffmpeg_demuxer',   type=int, default=1,
                    help='Indicator to use FFMPEG demuxer - optional (default:1). If set to 0, built-in bitstream reader is used.')

args = parser.parse_args()

rocDecodeDirectory = args.rocDecode_directory
gpuDeviceID = args.gpu_device_id
filesDir = args.files_directory
videoDecodeEXE = args.videodecode_exe
resultsDir = args.results_directory
useFFDemuxer = args.use_ffmpeg_demuxer

if useFFDemuxer == 1:
    bsReaderOption = ''
else:
    bsReaderOption = '-no_ffmpeg_demux'

print("\nrunrocDecodeTests V"+__version__+"\n")

# rocDecode Application
scriptPath = os.path.dirname(os.path.realpath(__file__))
if videoDecodeEXE == '':
    if platform.system() != 'Windows':
        rocDecode_exe = rocDecodeDirectory+'/samples/videoDecode/build/videodecode'
    else:
        rocDecode_exe = rocDecodeDirectory+'/samples/videoDecode/build/Release/videodecode.exe'
else:
    rocDecode_exe = videoDecodeEXE
if resultsDir == '':
    resultsPath = scriptPath+'/rocDecode_videoDecode_results'
else:
    resultsPath = resultsDir+'/rocDecode_videoDecode_results'

run_rocDecode_app = os.path.abspath(rocDecode_exe)
os.makedirs(resultsPath, exist_ok=True)
if(os.path.isfile(run_rocDecode_app)):
    print("STATUS: rocDecode path - "+run_rocDecode_app+"\n")
else:
    print("\nERROR: rocDecode Executable Not Found - "+run_rocDecode_app)
    print("Build the sample first, or pass --videodecode_exe to point at it.\n")
    sys.exit(1)

if os.path.exists(filesDir) and not os.path.isfile(filesDir):
    # Checking if the directory is empty or not
    if not os.listdir(filesDir):
        print("\nERROR: Empty directory - no videos to decode")
        sys.exit(1)
else:
    print("\nERROR: The input directory path is either for a file or directory does not exist!")
    sys.exit(1)

if os.path.exists(resultsPath+'/rocDecode_output.log'):
    os.remove(resultsPath+'/rocDecode_output.log')

print("Starting conformance test .....................................\n")
streamFileDir = filesDir + '/Streams/'
streamFileList = os.listdir(streamFileDir)
streamFileList.sort(key=str.lower)
streamListSize = len(streamFileList)

md5FileDir = filesDir + '/MD5/'
md5FileList = os.listdir(md5FileDir)
md5FileList.sort(key=str.lower)

if streamListSize == 0:
    print("Error: Empty stream file folder\n")
    sys.exit(1)

# Pair each stream with the digest of the same name, rather than with whichever digest happens to
# land at the same index. The two directories are listed and sorted independently, so one extra,
# missing or differently-named file shifts the lists against each other and every stream from that
# point on is checked against another stream's reference digest -- which still reports a tidy pass
# or failure, just about the wrong pairing. Names are matched without extension and case-folded,
# since a digest is conventionally named for its stream but not with the same suffix.
md5ByStem = {}
duplicateStems = []
for md5FileName in md5FileList:
    stem = os.path.splitext(md5FileName)[0].lower()
    if stem in md5ByStem:
        duplicateStems.append(stem)
    md5ByStem[stem] = md5FileDir + md5FileName
if duplicateStems:
    print("Error: More than one MD5 file for: " + ', '.join(sorted(set(duplicateStems))))
    print("       MD5 file names must be unique ignoring extension and case.\n")
    sys.exit(1)

streamPairs = []
matchedStems = set()
unmatchedStreams = []
for streamFileName in streamFileList:
    stem = os.path.splitext(streamFileName)[0].lower()
    if stem in md5ByStem:
        streamPairs.append((streamFileDir + streamFileName, md5ByStem[stem]))
        matchedStems.add(stem)
    else:
        unmatchedStreams.append(streamFileName)
if unmatchedStreams:
    print("Error: No MD5 file found for bit stream: " + ', '.join(unmatchedStreams) + "\n")
    sys.exit(1)

# An MD5 with no stream means the corpus is incomplete: the run would quietly cover fewer streams
# than it is meant to, and report a clean pass for doing so.
unmatchedMd5 = sorted(set(md5ByStem) - matchedStems)
if unmatchedMd5:
    print("Error: No bit stream found for MD5 file: " + ', '.join(unmatchedMd5) + "\n")
    sys.exit(1)

for streamFilePath, md5FilePath in streamPairs:
    cmd = [run_rocDecode_app, '-i', streamFilePath]
    if bsReaderOption:
        cmd.append(bsReaderOption)
    cmd += ['-md5_check', md5FilePath, '-d', str(gpuDeviceID)]
    logFilePath = resultsPath + '/rocDecode_output.log'
    run_and_log(cmd, logFilePath)
    print("======================================================================================\n")

fileString = 'Input file'
md5String = 'MD5 message digest'
matchString = 'MD5 digest matches the reference MD5 digest'
mismatchString = 'MD5 digest does not match the reference MD5 digest'
passNum = 0
failNum = 0
with open(resultsPath + '/rocDecode_output.log', 'r') as logFile:
    resultFile = open(resultsPath + '/rocDecode_conformance.log', 'w')
    resultFile.write("=========================\n")
    resultFile.write("Conformance test results\n")
    resultFile.write("=========================\n")

    line = logFile.readline()
    while line:
        if line.find(fileString) != -1:
            resultFile.write(line)
        if line.find(md5String) != -1:
            resultFile.write(line)
        if line.find(matchString) != -1:
            resultFile.write(line)
            passNum += 1
        if line.find(mismatchString) != -1:
            resultFile.write(line)
            failNum += 1
        line = logFile.readline()

    unfinishedNum = streamListSize - passNum - failNum
    print("Conformance test completed on the", streamListSize, "streams:")
    print("     - The number of passing streams is", passNum)
    print("     - The number of failing streams is", failNum)
    print("     - The number of streams that did not finish decoding is " + str(unfinishedNum))
    resultFile.write("\n===================================================\n")
    resultFile.write("Conformance test result summary on the " + str(streamListSize) + " streams:\n")
    resultFile.write("===================================================")
    resultFile.write("\n     - The number of passing streams is " + str(passNum))
    resultFile.write("\n     - The number of failing streams is " + str(failNum))
    resultFile.write("\n     - The number of streams that did not finish decoding is " + str(unfinishedNum))
    resultFile.close()
logFile.close()

# Report the outcome through the exit code so callers -- CI jobs in particular --
# can tell a clean sweep from a failing one without scraping stdout.
#
# A pass count above the stream count cannot come from decoding this run's streams: it means the
# tally was taken over output that is not all from this run. That happens when another run shares
# the results directory, because run_and_log appends to rocDecode_output.log. Treat it as a failure
# rather than letting it through as a negative unfinishedNum, which reads as a pass.
if passNum > streamListSize:
    print("\nERROR: reported " + str(passNum) + " passing streams out of " + str(streamListSize) + ".")
    print("       The tally is inconsistent, so these results are not usable. Pass a dedicated")
    print("       --results_directory and re-run with nothing else decoding on this machine.")
    sys.exit(1)

if failNum > 0 or unfinishedNum > 0:
    print("\nConformance test FAILED")
    sys.exit(1)

print("\nConformance test PASSED")
