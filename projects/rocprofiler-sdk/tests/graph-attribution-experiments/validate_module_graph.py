import json
import sys
from collections import defaultdict
p=json.load(open(sys.argv[1]))['rocprofiler-sdk-json-tool']
names=next(x['operations'] for x in p['buffer_records']['names'] if x['kind']=='HIP_RUNTIME_API')
active=False
calls=[]
for api in sorted(p['buffer_records']['hip_api_traces'],key=lambda x:x['start_timestamp']):
    name=names[api['operation']]
    if name=='hipStreamBeginCapture': active=True
    elif name=='hipStreamEndCapture': active=False
    elif active and name in ('hipLaunchKernel','hipModuleLaunchKernel','hipExtModuleLaunchKernel'):
        calls.append(api['correlation_id']['internal'])
assert len(calls)==6,calls
groups=defaultdict(list)
for rec in p['buffer_records']['kernel_dispatch']:
    if rec['dispatch_info']['graph_exec_id']['handle']:
        groups[rec['correlation_id']['internal']].append(rec)
assert len(groups)==3
for group in groups.values():
    group.sort(key=lambda x:x['dispatch_info']['dispatch_id'])
    got=[x['dispatch_info']['capture_correlation_id'] for x in group]
    assert got==calls,(got,calls)
print('PASS: 18/18 exact capture IDs, including repeated module names and ext-module launch')
