#!/usr/bin/python

import os, sys, re
import argparse

events_count = {}
events_order = {} 

def parse_trace(tracefile,cnt,order):
  with open(tracefile) as f: 
    for line in f: 
      event_pattern = re.compile(r'<(\w+)\s+id\(\d+\)\s+.*tid\((\d+)\)>')
      m = event_pattern.match(line)
      if m:
        event = m.group(1)
        tid = m.group(2)
      event_pattern2 = re.compile(r'\d+:\d+\s+\d+:(\d+)\s+(\w+)')
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
  if cnt:
    for event,count in events_count.items():
      print event + ": count " + str(count)
  if order:
    for tid in sorted (events_order.keys()) :
      print str(events_order[tid])

parser = argparse.ArgumentParser(description='parse_trace.py: reads roctracer trace file and parses it.')
parser.add_argument('-or', action='store_true',help='Generates ordered events')
parser.add_argument('-cn', action='store_true',help='Generates events count')
requiredNamed = parser.add_argument_group('Required arguments')
requiredNamed.add_argument('-in', metavar='file', help='Trace file', required=True)

args = vars(parser.parse_args())

if __name__ == '__main__':
    parse_trace(args['in'],args['cn'],args['or'])

