# Copyright (c) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
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
            
# Import arguments
parser = argparse.ArgumentParser()
parser.add_argument('--rocDecode_directory',   type=str, default='',
                    help='The rocDecode Directory - required')
parser.add_argument('--videodecode_exe',   type=str, default='',
                    help='Video decode sample app exe - optional')
parser.add_argument('--gpu_device_id',      type=int, default=0,
                    help='The GPU device ID that will be used to run the test on it - optional (default:0 [range:0 - N-1] N = total number of available GPUs on a machine)')
parser.add_argument('--files_directory',    type=str, default='',
                    help='The path to a dirctory containing one or more supported files for decoding (e.g., mp4, mov, etc.) and their corresponding reference MD5 digests - required')
parser.add_argument('--results_directory',    type=str, default='',
                    help='The path to a dirctory to store results - optional')
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
    rocDecode_exe = rocDecodeDirectory+'/samples/videoDecode/build/videodecode'
else:
    rocDecode_exe = videoDecodeEXE
if resultsDir == '':
    resultsPath = scriptPath+'/rocDecode_videoDecode_results'
else:
    resultsPath = resultsDir+'/rocDecode_videoDecode_results'

run_rocDecode_app = os.path.abspath(rocDecode_exe)
os.system('(mkdir -p ' +  resultsPath + ')')
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
    exit()
if streamListSize != md5ListSize:
    print("Error: Bit stream file number and MD5 file number do not match\n")
    exit()

for i in range(streamListSize):
    streamFilePath = streamFileDir + streamFileList[i]
    md5FilePath = md5FileDir + md5FileList[i]
    os.system(run_rocDecode_app +' -i ' + streamFilePath + ' ' + bsReaderOption + ' -md5_check ' + md5FilePath + ' -d ' + str(gpuDeviceID) + ' | tee -a ' + resultsPath + '/rocDecode_output.log')
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

    print("Conformance test completed on the", streamListSize, "streams:")
    print("     - The number of passing streams is", passNum)
    print("     - The number of failing streams is", failNum)
    print("     - The number of streams that did not finish decoding is " + str(streamListSize - passNum - failNum))
    resultFile.write("\n===================================================\n")
    resultFile.write("Conformance test result summary on the " + str(streamListSize) + " streams:\n")
    resultFile.write("===================================================")
    resultFile.write("\n     - The number of passing streams is " + str(passNum))
    resultFile.write("\n     - The number of failing streams is " + str(failNum))
    resultFile.write("\n     - The number of streams that did not finish decoding is " + str(streamListSize - passNum - failNum))
    resultFile.close()
logFile.close()