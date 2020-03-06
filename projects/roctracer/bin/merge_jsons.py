#!/usr/bin/python

################################################################################
# Copyright (c) 2018 Advanced Micro Devices, Inc. All rights reserved.
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
################################################################################

import os, sys, re
import argparse

shift = 100

def parse_json(jsonfile, fo):
  if not re.search(r'\.json$', jsonfile):
    raise Exception('wrong input file type: "' + jsonfile + '"' )

  with open(jsonfile) as fp:
    for line in fp:
      ms = re.match(r'^{ "traceEvents":\[{}\n|^\]}\n',line)
      if ms:
        continue
      mo = re.match(r'(.*"pid"\s*:\s*)(\d+)(.*)',line)
      mp = re.match(r'(.*)"name"\s*:\s*"([\w,\s]+)"(.*)"pid"\s*:\s*(\d+)(.*)',line)
      mp2 = re.match(r'(.*"pid"\s*:\s*")(\d+)(".*)',line)
      if mp:
        laneName = mp.group(2) + ' from ' + jsonfile
        mpid = int(str(mp.group(4)))
        mpid = mpid + shift
        fo.write(mp.group(1) + '"name":"' + laneName + '"' + mp.group(3) + '"pid":' + str(mpid) + mp.group(5) + '\n') 
      elif mo:
        mpid = int(str(mo.group(2)))
        mpid = mpid + shift
        fo.write(mo.group(1) + str(mpid) + mo.group(3) + '\n')
      elif mp2:
        mpid = int(str(mp2.group(2)))
        mpid = mpid + shift
        fo.write(mp2.group(1) + str(mpid) + mp2.group(3) + '\n')
      else:
        fo.write(line)

def merge_jsons(jsons,outfile):
  ljsons = jsons.split(',')
  fo = open(outfile, mode='w')
  fo.write('{ "traceEvents":[{}\n')
  for i in range(0, len(ljsons)):
    parse_json(ljsons[i],fo)
  fo.write(']}\n')
  fo.close()

parser = argparse.ArgumentParser(description='merge_jsons.py: merges list of jsons into one json file.')
requiredNamed=parser.add_argument_group('Required arguments')
requiredNamed.add_argument('-in','--in', help='comma separated list of json files', required=True)
requiredNamed.add_argument('-out','--out', help='Output file (.json)', required=True)

args = vars(parser.parse_args())

if __name__ == '__main__':
    merge_jsons(args['in'],args['out'])

