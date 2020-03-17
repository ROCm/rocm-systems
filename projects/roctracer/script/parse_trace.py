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

import os, sys, re
import argparse

events_count = {}
events_order = {} 

#Parses roctracer trace file for regression purpose 
#and generates events count per event (when cnt is on) or events order per tid (when order is on) 
def gen_events_info(tracefile,cnt,order):
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
      if cnt and (m or m2):
        if event in events_count:
          events_count[event] = events_count[event] + 1
        else:
          events_count[event] = 1
      if order and (m or m2):
        if tid in events_order.keys():
          events_order[tid].append(event)
        else:
          events_order[tid] = [event]
  if cnt==1:
    for event,count in events_count.items():
      res = res + event + ": count " + str(count)
  if order==1:
    for tid in sorted (events_order.keys()) :
      res = res + str(events_order[tid])
  return res

