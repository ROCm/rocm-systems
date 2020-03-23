#!/usr/bin/python

#Copyright (c) 2015-present Advanced Micro Devices, Inc. All rights reserved.
#
#Permission is hereby granted, free of charge, to any person obtaining a copy
#of this software and associated documentation files (the "Software"), to deal
#in the Software without restriction, including without limitation the rights
#to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#copies of the Software, and to permit persons to whom the Software is
#furnished to do so, subject to the following conditions:
#
#The above copyright notice and this permission notice shall be included in
#all copies or substantial portions of the Software.
#
#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
#AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
#THE SOFTWARE.

import os, re
import filecmp
import argparse

events_count = {}
events_order = {} 
trace2level = {}
trace2level_filename = 'tests_trace_cmp_levels.txt'

def parse_trace_levels(filename):
    f = open(filename)
    trace2level = {}
    for line in f:
        item = line.split(' ', 1)
        trace2level[item[0]] = eval(item[1])
    return trace2level

# check trace againt golden reference and returns 0 for match, 1 for mismatch 
def check_trace_status(tracename):
  trace2level = parse_trace_levels(trace2level_filename)
  trace = tracename + '.txt'
  rtrace = tracename + '_r.txt'
  if os.path.basename(tracename) in trace2level:
    trace_level = trace2level[os.path.basename(tracename)]
    print 'Trace comparison for ' + os.path.basename(tracename) + ' is at level ' + str(trace_level) + '\n'
  else:
    print 'Trace ' + os.path.basename(tracename) + ' not found in ' + trace2level_filename + ', defaulting to level 0\n'
    return 0
  
  if trace_level == 1:
    cnt_r = gen_events_info(rtrace,'cnt')
    cnt = gen_events_info(trace,'cnt')
    if cnt_r == cnt:
      return 0
    else:
      return 1
  elif trace_level == 2:
    cnt_r = gen_events_info(rtrace,'or')
    cnt = gen_events_info(trace,'or')
    if cnt_r == cnt:
      return 0
    else:
      return 1
  elif trace_level == 3:
   if filecmp.cmp(trace,rtrace):
     return 0
   else:
     return 1

#Parses roctracer trace file for regression purpose 
#and generates events count per event (when cnt is on) or events order per tid (when order is on) 
def gen_events_info(tracefile, metric):
  res=''
  with open(tracefile) as f: 
    for line in f: 
      event_pattern = re.compile(r'<(\w+)\s+id\(\d+\)\s+.*tid\((\d+)\)>')
      # event_pattern extracts event(grp1) and tid (grp2) from a line like this:
      # <hsaKmtGetVersion id(2) correlation_id(0) on-enter pid(26224) tid(26224)>
      m = event_pattern.match(line)
      if m:
        event = m.group(1)
        tid = m.group(2)
      event_pattern2 = re.compile(r'\d+:\d+\s+\d+:(\d+)\s+(\w+)')
      # event_pattern2 extracts tid (grp1) and event (grp2) from a line like this:
      # 1822810364769411:1822810364771941 116477:116477 hsa_agent_get_info(<agent 0x8990e0>, 17, 0x7ffeac015fec) = 0
      m2 = event_pattern2.match(line)
      if m2:
        event = m2.group(2)
        tid = m2.group(1)
      if metric == 'cnt' and (m or m2):
        if event in events_count:
          events_count[event] = events_count[event] + 1
        else:
          events_count[event] = 1
      if metric == 'or' and (m or m2):
        if tid in events_order.keys():
          events_order[tid].append(event)
        else:
          events_order[tid] = [event]
  if metric == 'cnt':
    for event,count in events_count.items():
      res = res + event + " : count " + str(count) + '\n'
  if metric == 'or':
    for tid in sorted (events_order.keys()) :
      res = res + str(events_order[tid]) 

return res


parser = argparse.ArgumentParser(description='check_trace.py: check a trace aainst golden ref. Returns 0 for success, 1 for failure')
requiredNamed = parser.add_argument_group('Required arguments')
requiredNamed.add_argument('-in', metavar='file', help='Name of trace to be checked', required=True)
args = vars(parser.parse_args())

if __name__ == '__main__':
    check_trace_status(args['in'])

