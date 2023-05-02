#!/usr/bin/env python3

################################################################################
# Copyright (c) 2018-2022 Advanced Micro Devices, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to
# deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
# sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
# IN THE SOFTWARE.
################################################################################

import sys, os, re
import filecmp
import argparse

events_count = {}
events_order = {}
events_order_r = {}
trace2info = {}
trace2info_filename = 'test/golden_traces/tests_trace_cmp_levels.txt'

# Parses trace comparison config file and stores the info in a dictionary
def parse_trace_levels(trace_config_filename, check_trace_flag):
    status = 0
    f = open(trace_config_filename)
    trace2info = {}
    for line in f:
        if check_trace_flag == 0:
          return (trace2info, status)
        if (check_trace_flag == None) and re.match('^# dummy',line):
          return (trace2info, status)
        status = 1
        lis = line.split(' ')
        trace_name = lis[0]
        comp_level = lis[1]
        no_events_cnt = ''
        events2ignore = ''
        events2chkcnt = ''
        events2chkord = ''
        events2ch = ''
        for l in lis:
          if no_events_cnt == ' ':
            no_events_cnt = l
          if events2ignore == ' ':
            events2ignore = l
          if events2chkcnt == ' ':
            events2chkcnt = l
          if events2chkord == ' ':
            events2chkord = l
          if events2ch == ' ':
            events2ch = l
            events2chkcnt = l
            no_events_cnt = l
          if l == '--ignore-count':
            no_events_cnt = ' '
          if l == '--ignore-event':
            events2ignore = ' '
          if l == '--check-count':
            events2chkcnt = ' '
          if l == '--check-order':
            events2chkord = ' '
          if l == '--check-events':
            events2ch = ' '

        trace2info[trace_name] = (comp_level,no_events_cnt,events2ignore,events2chkcnt,events2chkord,events2ch)

    return (trace2info, status)

# diff multi lines strings to show events differences
def diff_strings(cnt_r, cnt, metric):
  global events_order_r
  global events_order

  print ("\nDiffs (if any):\n")
  if metric == 'cnt':
    evt_ptrn = re.compile(r'(\w+).*$')
    #cnt_ptrn = re.compile(r'(\w+): count (\d+)$')
    for evt in cnt_r.split('\n'):
      mevt_ptrn = evt_ptrn.match(evt)
      #mcnt_ptrn = cnt_ptrn.match(evt)
      if mevt_ptrn:
        if not re.search(mevt_ptrn.group(1), cnt):
          print ('+ ' + evt)
        elif not re.search(evt, cnt):
          print ('>D< ' + evt)

    for evt in cnt.split('\n'):
      mevt_ptrn = evt_ptrn.match(evt)
      #mcnt_ptrn = cnt_ptrn.match(evt)
      if mevt_ptrn:
        if not re.search(mevt_ptrn.group(1), cnt_r):
          print ('- ' + evt)
  if metric == 'or':
    cnt_tid_r = 0
    for tid_r in sorted (events_order_r.keys()):
      if len(events_order) == 0:
        print ("+ " + str(events_order_r[tid_r]) + "\n\n")
        continue
      cnt_tid = 0
      for tid in sorted (events_order.keys()):
        if cnt_tid == cnt_tid_r:
          if events_order_r[tid_r] != events_order[tid]:
            #print (">D< " + str(events_order_r[tid_r]) + "\n")
            #print (">D< " + str(events_order[tid]) + "\n\n")
            diff_cnt_r = 0
            found_diff_evt = 0
            for evt in events_order_r[tid_r]:
              diff_cnt = 0
              for evt2 in events_order[tid]:
                if diff_cnt == diff_cnt_r:
                  if evt != evt2:
                    print (">I< Difference starts at tid rank: " + str(cnt_tid) + " event index: " + str(diff_cnt_r) + ", tid_r " + str(tid_r) + ", tid " + str(tid) + ", with evts " + evt + " and " + evt2 + "\n")
                    found_diff_evt = 1
                    break
                diff_cnt += 1
              diff_cnt_r += 1
              if found_diff_evt: break
            if len(events_order_r[tid_r]) != len(events_order[tid]) and found_diff_evt == 0:
              print (">I< Difference starts at tid rank: " + str(cnt_tid) + " event index: " + str(min(len(events_order_r[tid_r]), len(events_order[tid])))  + ", with missing evts\n")
          break
        cnt_tid += 1
      cnt_tid_r += 1
    if len(events_order_r) == 0:
      for tid in sorted (events_order.keys()):
        print ("- " + str(events_order[tid]) + "\n")

# check trace againt golden reference and returns 0 for pass, 1 for fail
def check_trace_status(tracename, verbose, check_trace_flag):
  global events_order_r
  global events_order

  (trace2info, status) = parse_trace_levels(trace2info_filename, check_trace_flag)

  if len(trace2info) == 0:
    if status == 1:
      print ("Error: no trace comparison info found in config file " + trace2info_filename + "\n")
      print('FAILED!')
      return 1
    if status == 0:
      print('PASSED!')
      return 0

  trace = 'test/golden_traces/' + tracename + '.txt'
  rtrace = '/tmp/test/out/' + tracename + '.out'
  if os.path.basename(tracename) in trace2info.keys():
    (trace_level, no_events_cnt, events2ignore, events2chkcnt, events2chkord, events2ch) = trace2info[os.path.basename(tracename)]
    trace_level = trace_level.rstrip('\n')
    no_events_cnt = no_events_cnt.rstrip('\n')
    events2ignore = events2ignore.rstrip('\n')
    events2chkcnt = events2chkcnt.rstrip('\n')
    events2chkord = events2chkord.rstrip('\n')
    events2ch = events2ch.rstrip('\n')
  else:
    print('Trace ' + os.path.basename(tracename) + ' not found in ' + trace2info_filename)
    print('FAILED!')
    return 1

  if no_events_cnt == '':
    no_events_cnt = 'empty-regex'
  if events2ignore == '':
    events2ignore = 'empty-regex'
  if events2chkcnt == '':
    events2chkcnt = ''
  if events2chkord == '':
    events2chkord = ''

  if trace_level == '--check-none':
    print('PASSED!')
    return 0

  if trace_level == '--check-diff':
    if filecmp.cmp(trace,rtrace):
      print('PASSED!')
      return 0
    else:
      print('FAILED!')
      os.system('/usr/bin/diff --brief ' + trace + ' ' + rtrace)
      return 1

  metric = ''
  if trace_level == '--check-count' or trace_level == '--check-events':
    metric = 'cnt'
  if trace_level == '--check-order':
    metric = 'or'

  cnt_r = gen_events_info(rtrace,trace_level,no_events_cnt,events2ignore,events2chkcnt,events2chkord,verbose)
  events_order_r = {}
  for tid in sorted (events_order.keys()) :
    events_order_r[tid] = events_order[tid]
  cnt = gen_events_info(trace,trace_level,no_events_cnt,events2ignore,events2chkcnt,events2chkord,verbose)
  if verbose:
      print ('\n' + rtrace + ':\n')
      print (cnt_r)
      print ('\n' + trace + ':\n')
      print (cnt)
      diff_strings(cnt_r, cnt, metric)

  if cnt_r == cnt:
    print('PASSED!')
    return 0
  else:
    print('FAILED!')
    return 1

# Parses roctracer trace file for regression purpose
# and generates events count per event (when cnt is on) or events order per tid (when order is on)
def gen_events_info(tracefile, trace_level, no_events_cnt, events2ignore, events2chkcnt, events2chkord, verbose):
  global events_order
  metric = ''
  if trace_level == '--check-count' or trace_level == '--check-events':
    metric = 'cnt'
  if trace_level == '--check-order':
    metric = 'or'

  events_count = {}
  events_order = {}
  res = ''
  re_no_events_cnt = r'{}'.format(no_events_cnt)
  re_events2ignore = r'{}'.format(events2ignore)
  re_events2chkcnt = r'{}'.format(events2chkcnt)
  re_events2chkord = r'{}'.format(events2chkord)

  test_act_pattern = re.compile(r'\s*(\w+)\s+.*_id\((\d+)\)$')
  #'       hipSetDevice    correlation_id(1) time_ns(1548622357525055:1548622357542015) process_id(126283) thread_id(126283)'
  #'       hcCommandKernel correlation_id(6) time_ns(1548622661443020:1548622662666935) device_id(0) queue_id(0)'
  test_api_cb_pattern = re.compile(r'.*<(\w+)\s+.*tid\((\d+)\)>')
  # <hsaKmtGetVersion id(2) correlation_id(0) on-enter pid(26224) tid(26224)>
  # below is roctx pattern
  # <hipLaunchKernel pid(123) tid(123)>
  tool_record = re.compile(r'\d+:\d+\s+\d+:(\d+)\s+(\w+)')
  # tool_api_record
  # 1822810364769411:1822810364771941 116477:116477 hsa_agent_get_info(<agent 0x8990e0>, 17, 0x7ffeac015fec) = 0
  # tool_gpu_act_record
  # 3632773658039902:3632773658046462 0:0 hcCommandMarker:273
  roctx_record = re.compile(r'\d+\s\d+:(\d)+\s(\d):\d+:\".*\"')

  with open(tracefile) as f:
    for line in f:
      if re.search("before", line) or re.search("after",line):#roctx before/after not real events
        continue
      line=line.rstrip('\n')
      event = ''
      test_act_pattern_match = test_act_pattern.match(line)
      if test_act_pattern_match:
        event = test_act_pattern_match.group(1)
        tid = int(test_act_pattern_match.group(2))
      test_api_cb_pattern_match = test_api_cb_pattern.match(line)
      if test_api_cb_pattern_match:
        event = test_api_cb_pattern_match.group(1)
        tid = int(test_api_cb_pattern_match.group(2))
      tool_record_match = tool_record.match(line)
      if tool_record_match:
        event = tool_record_match.group(2)
        tid = int(tool_record_match.group(1))
      roctx_record_match = roctx_record.match(line)
      if roctx_record_match:
        event = roctx_record_match.group(2)
        tid = int(roctx_record_match.group(1))
      if event == '' or event == '(null)': #some traces has these null events
        continue

      if re.search(re_events2ignore,event):
        continue

      if metric == 'cnt' and re.search(re_events2chkcnt,event):
        if event in events_count:
          events_count[event] = events_count[event] + 1
        else:
          if not re.search(re_no_events_cnt,event):
            events_count[event] = 1

      if metric == 'or' and re.search(re_events2chkord,event):
        if tid in events_order.keys():
          if re.search(re_no_events_cnt,event):
            if event != events_order[tid][-1]: #Add event only if it is not last event in the list
              events_order[tid].append(event)
          else:
            events_order[tid].append(event)
        else:
          events_order[tid] = [event]
  if metric == 'cnt':
    for event,count in events_count.items():
      if re.search(re_no_events_cnt,event):
        res = res + event + '\n'
      else:
        res = res + event + " : count " + str(count) + '\n'
  if metric == 'or':
    for tid in sorted (events_order.keys()) :
      res = res + str(events_order[tid])
  if metric == 'cnt':
    newres = res.split('\n')
    newres = sorted(newres)
    res = str(newres)
  return res

parser = argparse.ArgumentParser(description='check_trace.py: check a trace aainst golden ref. Returns 0 for success, 1 for failure')
requiredNamed = parser.add_argument_group('Required arguments')
requiredNamed.add_argument('-in', metavar='file', help='Name of trace to be checked', required=True)
requiredNamed.add_argument('-v', action='store_true', help='debug info', required=False)
requiredNamed.add_argument('-ck', metavar='N', type=int, help='check trace 0|1', required=False)

args = vars(parser.parse_args())

if __name__ == '__main__':
    sys.exit(check_trace_status(args['in'],args['v'],args['ck']))


