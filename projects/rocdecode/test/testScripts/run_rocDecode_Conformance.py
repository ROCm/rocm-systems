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

# Emitted by the videoDecode sample when the GPU's VCN cannot handle the stream's
# codec -- AV1 on gfx90a, say. That is a property of the hardware, not a decoder
# fault, so these streams are counted separately from failures.
#
# The sample signals this two ways and both are checked. The exit code is the
# primary signal: videodecode.cpp documents 77 as the conventional skip code and
# this script launches the executable directly, so it sees the true status.
# test/CMakeLists.txt has to match on the message instead, because
# `ctest --build-and-test` flattens every non-zero code to 1; the message is kept
# here as a second opinion so the two stay in agreement.
UNSUPPORTED_CODEC_MESSAGE = 'Codec not supported on GPU, skipping!'
UNSUPPORTED_CODEC_EXIT_CODE = 77

# Lines the sample prints per stream. The first two are transcribed into the
# conformance log for a human to read; the last two are what classify the stream.
# 'does not match' does not contain 'matches', so the two verdicts are disjoint.
FILE_STRING = 'Input file'
MD5_STRING = 'MD5 message digest'
MATCH_STRING = 'MD5 digest matches the reference MD5 digest'
MISMATCH_STRING = 'MD5 digest does not match the reference MD5 digest'

# The whole tally on one line, so a CI log holding several codecs' sweeps can be
# grepped for a per-codec summary without reading back the surrounding output.
COUNTS_PREFIX = 'CONFORMANCE_COUNTS'


def run_and_log(cmd, logFilePath, resultFilePath):
    # Portable replacement for '| tee -a': stream the output line by line so
    # progress stays visible during long runs, while appending it to the log.
    # cmd is an argument list launched without a shell, so paths holding spaces
    # or shell metacharacters reach the sample unmangled.
    # universal_newlines (not the 3.7+ 'text' alias) keeps this working on the
    # python3 3.6 that ships with SLES 15 and RHEL 8.
    #
    # Every fact the caller needs about a stream is collected here, while its own
    # output is in hand: the exit status, whether the codec was reported
    # unsupported, and which digest verdict -- if any -- the sample reached. That
    # lets the caller give each stream exactly one classification. Deriving the
    # tallies instead from a scan of the shared rocDecode_output.log, as this
    # script used to, counted whatever else had appended to that file.
    sawUnsupportedCodec = False
    sawMatch = False
    sawMismatch = False
    with open(logFilePath, 'a') as logf, open(resultFilePath, 'a') as resultf:
        p = Popen(cmd, shell=False, stdout=PIPE, stderr=STDOUT, universal_newlines=True)
        for line in p.stdout:
            print(line, end='')
            logf.write(line)
            if UNSUPPORTED_CODEC_MESSAGE in line:
                sawUnsupportedCodec = True
            if FILE_STRING in line or MD5_STRING in line:
                resultf.write(line)
            if MATCH_STRING in line:
                sawMatch = True
                resultf.write(line)
            elif MISMATCH_STRING in line:
                sawMismatch = True
                resultf.write(line)
        p.wait()
    return p.returncode, sawUnsupportedCodec, sawMatch, sawMismatch


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
parser.add_argument('--codec_name',    type=str, default='',
                    help='The codec this stream set belongs to, e.g. AVC - optional. Named in the summary and verdict lines so a sweep that runs several codecs in turn can be read back per codec.')

args = parser.parse_args()

rocDecodeDirectory = args.rocDecode_directory
gpuDeviceID = args.gpu_device_id
filesDir = args.files_directory
videoDecodeEXE = args.videodecode_exe
resultsDir = args.results_directory
useFFDemuxer = args.use_ffmpeg_demuxer
codecName = args.codec_name

# Every line that reports an outcome is prefixed with this, so a log holding more
# than one codec's sweep says which codec each verdict belongs to. Run standalone
# without --codec_name it stays the original wording.
testLabel = 'Conformance test for ' + codecName if codecName else 'Conformance test'

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
md5ListSize = len(md5FileList)

if streamListSize == 0:
    print("Error: Empty stream file folder\n")
    sys.exit(1)
if streamListSize != md5ListSize:
    print("Error: Bit stream file number and MD5 file number do not match\n")
    sys.exit(1)

# Each stream lands in exactly one of these, decided from its own output while it
# is still in hand. Together they partition the stream list, so the five numbers
# in the summary sum to the stream count and none of them overlaps another.
#
#   passed            digest matched and the process exited cleanly
#   digestFailures    digest mismatched -- the decode was wrong
#   teardownFailures  digest matched but the process still exited non-zero, e.g.
#                     a throw while destroying the demuxer or decoder. Without
#                     this bucket such a stream would be counted a pass purely on
#                     the strength of its own output.
#   skipped           this GPU's VCN does not support the codec (not a fault)
#   unfinished        the run ended before any digest verdict was printed -- a
#                     caps query failure, an unreadable MD5 file, a decode error
#                     or an exception
#
# An earlier version counted "decoder exited non-zero" as a sixth total. It was
# not a sixth bucket but a column cutting across the other five, and because the
# sample returns 1 for an ordinary digest mismatch it was dominated by exactly the
# case it was not written for: every mismatch was reported twice, once as a
# failure and again as a decoder error. The exit status is now a detail recorded
# on each entry rather than a total competing with them.
passed = []
digestFailures = []
teardownFailures = []
skipped = []
unfinished = []

resultFilePath = resultsPath + '/rocDecode_conformance.log'
with open(resultFilePath, 'w') as resultFile:
    resultHeading = testLabel + " results"
    resultFile.write("=" * len(resultHeading) + "\n")
    resultFile.write(resultHeading + "\n")
    resultFile.write("=" * len(resultHeading) + "\n")

for i in range(streamListSize):
    streamFilePath = streamFileDir + streamFileList[i]
    md5FilePath = md5FileDir + md5FileList[i]
    cmd = [run_rocDecode_app, '-i', streamFilePath]
    if bsReaderOption:
        cmd.append(bsReaderOption)
    cmd += ['-md5_check', md5FilePath, '-d', str(gpuDeviceID)]
    logFilePath = resultsPath + '/rocDecode_output.log'
    returnCode, unsupportedCodec, matched, mismatched = run_and_log(
        cmd, logFilePath, resultFilePath)
    streamName = streamFileList[i]
    if unsupportedCodec or returnCode == UNSUPPORTED_CODEC_EXIT_CODE:
        skipped.append(streamName)
    elif mismatched:
        digestFailures.append((streamName, returnCode))
    elif matched and returnCode == 0:
        passed.append(streamName)
    elif matched:
        teardownFailures.append((streamName, returnCode))
    else:
        unfinished.append((streamName, returnCode))
    print("======================================================================================\n")

passNum = len(passed)
failNum = len(digestFailures)
teardownNum = len(teardownFailures)
skipNum = len(skipped)
unfinishedNum = len(unfinished)

summaryLines = [
    "     - The number of passing streams is " + str(passNum),
    "     - The number of streams whose digest did not match is " + str(failNum),
    "     - The number of streams that decoded but exited non-zero is " + str(teardownNum),
    "     - The number of streams skipped as unsupported by this GPU is " + str(skipNum),
    "     - The number of streams that did not finish decoding is " + str(unfinishedNum),
]

print(testLabel + " completed on the", streamListSize, "streams:")
for summaryLine in summaryLines:
    print(summaryLine)

summaryHeading = testLabel + " result summary on the " + str(streamListSize) + " streams:"
with open(resultFilePath, 'a') as resultFile:
    resultFile.write("\n" + "=" * len(summaryHeading) + "\n")
    resultFile.write(summaryHeading + "\n")
    resultFile.write("=" * len(summaryHeading) + "\n")
    for summaryLine in summaryLines:
        resultFile.write(summaryLine + "\n")
    for streamName, returnCode in digestFailures:
        resultFile.write("         " + streamName + " digest mismatch (exited " + str(returnCode) + ")\n")
    for streamName, returnCode in teardownFailures:
        resultFile.write("         " + streamName + " digest matched but exited " + str(returnCode) + "\n")
    for streamName, returnCode in unfinished:
        resultFile.write("         " + streamName + " did not finish (exited " + str(returnCode) + ")\n")
    for streamName in skipped:
        resultFile.write("         " + streamName + " skipped: codec unsupported\n")

print(COUNTS_PREFIX
      + " total=" + str(streamListSize)
      + " passed=" + str(passNum)
      + " digest_failures=" + str(failNum)
      + " teardown_failures=" + str(teardownNum)
      + " unfinished=" + str(unfinishedNum)
      + " skipped=" + str(skipNum))

# Name every failing stream on stdout, not just in rocDecode_conformance.log.
# That file is written beside the results and CI keeps no artifact of it, so on a
# red run it dies with the runner -- leaving the counts above as the only record
# and a triager to read back through a few hundred interleaved streams. All three
# loops are empty on a clean run, so this is silent unless something failed.
for streamName, returnCode in digestFailures:
    print("\nFAILED: " + streamName + " digest did not match its reference (exited "
          + str(returnCode) + ")")

for streamName, returnCode in unfinished:
    print("\nFAILED: " + streamName + " did not finish decoding (exited "
          + str(returnCode) + ")")

for streamName, returnCode in teardownFailures:
    print("\nWARNING: " + streamName + " matched its reference digest but the decoder "
          "exited " + str(returnCode) + " (failure during teardown)")

if skipped:
    print("\n" + str(skipNum) + " stream(s) were skipped: this GPU does not support their codec.")

# Report the outcome through the exit code so callers -- CI jobs in particular --
# can tell a clean sweep from a failing one without scraping stdout.
#
# Failures are checked before the all-skip case. The other order lets a run exit 0
# reporting SKIPPED while holding real failures.
if failNum > 0 or teardownNum > 0 or unfinishedNum > 0:
    print("\n" + testLabel + " FAILED")
    sys.exit(1)

# Nothing was decoded, but nothing was wrong either -- this GPU simply cannot play
# this codec. Report it as a skip rather than claiming a pass for work not done.
if skipNum == streamListSize:
    print("\n" + testLabel + " SKIPPED: no stream in this set is supported by this GPU")
    sys.exit(0)

if skipNum > 0:
    print("\n" + testLabel + " PASSED on " + str(passNum) + " stream(s), "
          + str(skipNum) + " skipped as unsupported by this GPU")
else:
    print("\n" + testLabel + " PASSED")
