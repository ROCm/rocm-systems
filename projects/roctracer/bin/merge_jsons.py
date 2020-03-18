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

def parse_json(jsonfile, fo, index, lastjson):
  if not re.search(r'\.json$', jsonfile):
    raise Exception('wrong input file type: "' + jsonfile + '"' )
  metadata = ''
  with open(jsonfile) as fp:
    for line in fp:
      ms = re.match(r'^{ "traceEvents":\[{}\n|^\]}\n|^\],\n',line)
      if ms:
        continue
      md = re.match(r'.*otherData.*',line)
      if md:
        # collect metadata in variable 'metadata'
        metadata = ' '  
        continue
      if metadata != '':
        minfo = re.match(r'(.*)"(.*)":(.*)',line)
        # minfo catch pattern like: "version": "my app V1.0" 
        if minfo:
          metadata = metadata + minfo.group(1) + '"' + minfo.group(2) + '(' + os.path.splitext(jsonfile)[0] + ')":' + minfo.group(3) + '\n'
        else:
          metadata = metadata + line 
        continue
      mo = re.match(r'(.*"pid"\s*:\s*)(\d+)(.*)',line)
      # mo catch pattern like: ,{"ts":4258451581657,"ph":"s","cat":"DataFlow","id":0,"pid":2,"tid":83583,"name":"dep"}
      # grp2 is pid number that needs shifting, grp1 is what comes before pid number and grp3 is what comes after
      mp = re.match(r'(.*)"name"\s*:\s*"([\w,\s]+)"(.*)"pid"\s*:\s*(\d+)(.*)',line)
      # mp catch pattern like: ,{"args":{"name":"0 CPU HIP API"},"ph":"M","pid":2,"name":"process_name"} 
      # Grp 1.	0-10	,{"args":{
      # Grp 2.	18-31	0 CPU HIP API
      # Grp 3.	32-43	},"ph":"M",
      # Grp 4.	49-50	2
      # Grp 5.	50-73	,"name":"process_name"}
      mp2 = re.match(r'(.*"pid"\s*:\s*")(\d+)(".*)',line)
      # mo2 catch pattern like:     "pid":"3",
      # where grp2 is the pid number to shift
      if mp:
        laneName = mp.group(2) + '(' + os.path.splitext(jsonfile)[0] + ')'
        mpid = int(str(mp.group(4)))
        mpid = mpid + index*shift
        fo.write(mp.group(1) + '"name":"' + laneName + '"' + mp.group(3) + '"pid":' + str(mpid) + mp.group(5) + '\n') 
      elif mo:
        mpid = int(str(mo.group(2)))
        mpid = mpid + index*shift
        fo.write(mo.group(1) + str(mpid) + mo.group(3) + '\n')
      elif mp2:
        mpid = int(str(mp2.group(2)))
        mpid = mpid + index*shift
        fo.write(mp2.group(1) + str(mpid) + mp2.group(3) + '\n')
      else:
        fo.write(line)

  metadata = metadata[:metadata.rfind('\n')]
  metadata = metadata[:metadata.rfind('\n')]
  if lastjson == 0:
    metadata = metadata + ',\n'
  else:
    metadata = metadata + '\n'
  return metadata

def merge_jsons(jsons,outfile):
  ljsons = jsons.split(',')
  fo = open(outfile, mode='w')
  fo.write('{ "traceEvents":[{}\n')
  metadata = ''
  res=''
  # res will contain all metadata for all jsons files provided as input
  for i in range(0, len(ljsons)):
    if i == len(ljsons)-1:
      res=res+parse_json(ljsons[i],fo,i,1)
    else:
      res=res+parse_json(ljsons[i],fo,i,0)
  fo.write('],\n')
  # write metadata at the end of output json file
  fo.write('  "otherData": {\n')
  fo.write(res)
  fo.write('  }\n}\n')
  fo.close()

parser = argparse.ArgumentParser(description='merge_jsons.py: merges list of jsons into one json file.')
requiredNamed=parser.add_argument_group('Required arguments')
requiredNamed.add_argument('-in','--in', help='comma separated list of json files', required=True)
requiredNamed.add_argument('-out','--out', help='Output file (.json)', required=True)

args = vars(parser.parse_args())

if __name__ == '__main__':
    merge_jsons(args['in'],args['out'])

