#!/usr/bin/python

import os, sys, re
import argparse

events_count = {}
events_order = []

def parse_trace(tracefile,cnt,order):
  with open(tracefile) as f: 
    for line in f: 
      event_pattern = re.compile(r'<(\w+)\s+id\(\d+\)\s+.*|\d+:\d+\s+\d+:\d+\s+(\w+).*')
      m = event_pattern.match(line)
      if m:
        event = m.group(1)
        if not event:
          event = m.group(2)
        if cnt:
          if event in events_count:
            events_count[event] = events_count[event] + 1
          else:
            events_count[event] = 1
        if order:
            events_order.append(event)
  if cnt:
    for event,count in events_count.items():
      print event + ": count " + str(count)
  if order:
    print events_order

parser = argparse.ArgumentParser(description='parse_trace.py: reads roctracer trace file and parses it.')
parser.add_argument('-cn', action='store_true',help='Generates events count')
parser.add_argument('-or', action='store_true',help='Generates ordered events')
requiredNamed = parser.add_argument_group('Required arguments')
requiredNamed.add_argument('-in', metavar='file', help='Trace file', required=True)

args = vars(parser.parse_args())

if __name__ == '__main__':
    parse_trace(args['in'],args['cn'],args['or'])

