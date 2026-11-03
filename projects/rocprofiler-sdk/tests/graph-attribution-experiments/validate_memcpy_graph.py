import json
import sys
from collections import defaultdict
p=json.load(open(sys.argv[1]))['rocprofiler-sdk-json-tool']
names=next(x['operations'] for x in p['buffer_records']['names'] if x['kind']=='HIP_RUNTIME_API')
captures=[]
active=None
for api in sorted(p['buffer_records']['hip_api_traces'],key=lambda x:x['start_timestamp']):
    name=names[api['operation']]
    if name=='hipStreamBeginCapture': active=[]
    elif name=='hipStreamEndCapture': captures.append(active);active=None
    elif active is not None and name=='hipMemcpyAsync': active.append(api['correlation_id']['internal'])
assert list(map(len,captures))==[2,2]
groups=defaultdict(lambda: defaultdict(list))
for rec in p['buffer_records']['kernel_dispatch']:
    info=rec['dispatch_info']
    if info['graph_exec_id']['handle']:
        groups[info['graph_exec_id']['handle']][rec['correlation_id']['internal']].append(info)
assert len(groups)==3
original,fresh,duplicate=[groups[x] for x in sorted(groups)]
assert [len(x) for x in (original,fresh,duplicate)]==[6,3,3]
for gi,group in enumerate((original,fresh,duplicate)):
    for i,cid in enumerate(sorted(group)):
        actual=[r['capture_correlation_id'] for r in sorted(group[cid],key=lambda r:r['dispatch_id'])]
        expected=captures[0] if (gi==0 and i<3) or gi==1 else [0,0]
        assert actual==expected,(gi,i,actual,expected)
print('PASS: exact memcpy IDs, mutation invalidation, fresh recovery, duplicate refusal (24 dispatches)')
