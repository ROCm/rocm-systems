"""Strict native rocprofv3 JSON validator for graph_memset_probe.hip (stdlib only)."""
import argparse
import collections
import json
import pathlib
p = argparse.ArgumentParser()
p.add_argument('trace')
p.add_argument('--disabled', action='store_true')
a = p.parse_args()
raw = json.loads(pathlib.Path(a.trace).read_text())['rocprofiler-sdk-tool']
assert len(raw) == 1
d = raw[0]
r = d['buffer_records']
assert d['metadata']['fini_time']

def name(x):
    return d['strings']['buffer_records'][x['kind']]['operations'][x['operation']]

def cid(x):
    return x['correlation_id']['internal']
api = r['hip_api']
assert len({cid(x) for x in api}) == len(api)
apis = {cid(x): x for x in api}
caps = []
active = None
for x in sorted(api, key=lambda x: x['start_timestamp']):
    n = name(x)
    if n == 'hipStreamBeginCapture':
        assert active is None
        active = []
    elif n == 'hipStreamEndCapture':
        assert active is not None
        caps.append(active)
        active = None
    elif active is not None and n in ('hipMemsetD8Async', 'hipMemsetD16Async', 'hipMemsetD32Async', 'hipMemset2DAsync'):
        active.append(cid(x))
assert list(map(len, caps)) == [3, 2, 2, 1, 2, 7], list(map(len, caps))
launches = sorted(r['hip_graph'], key=lambda x: x['start_timestamp'])
assert len(launches) == 36, len(launches)
# Launch sequence is fixed by the probe; source IDs come from actual capture APIs.
expected = (
    [caps[0]] * 6       # original, including a failed exec mutation
    + [[0] * 3] * 3    # successful exec mutation invalidates the plan
    + [caps[0]] * 3    # a fresh exec from unchanged source recovers attribution
    + [caps[0]] * 3    # failed driver API mutation preserves the plan
    + [[0] * 3] * 3    # successful driver API mutation invalidates it
    + [[0] * 3] * 3    # source parameters changed since capture
    + [[0] * 2] * 3    # identical captured fills are ambiguous
    + [caps[2]] * 3    # same address and length, distinct values
    + [[0] * 2] * 3    # unrecorded manually added fill
    + [[0] * 2] * 3    # unsupported captured 2D fill
    + [caps[5]] * 3    # tiny, unaligned, large and typed patterns
)
assert len(expected) == 36
ks = [x for x in r['kernel_dispatch'] if x['graph_exec_id']]
assert len(ks) == 108, len(ks)
groups = collections.defaultdict(list)
for x in ks:
    i = x['dispatch_info']
    assert i['graph_exec_id']['handle'] == x['graph_exec_id']
    assert i['graph_node_id']['handle'] == x['graph_node_id']
    groups[x['graph_exec_id'], cid(x)].append(x)
resolved = 0
for (launch, want) in zip(launches, expected):
    assert name(apis[cid(launch)]) == 'hipGraphLaunch'
    got = groups.pop((launch['graph_exec_id']['handle'], cid(launch)))
    assert len(got) == launch['kernel_dispatch_count'] == len(want)
    assert all(x['dispatch_info']['agent_id'] == launch['agent_id'] for x in got)
    assert len({x['graph_node_id'] for x in got}) == len(got)
    ids = [x['dispatch_info']['capture_correlation_id'] for x in got]
    target = [0] * len(want) if a.disabled else want
    assert collections.Counter(ids) == collections.Counter(target), (cid(launch), ids, target)
    for c in filter(None, ids):
        assert c in apis and apis[c]['end_timestamp'] < launch['start_timestamp']
        resolved += 1
assert not groups
assert resolved == (0 if a.disabled else 63), resolved
print(json.dumps({'status': 'PASS', 'graph_launches': 36, 'graph_dispatches': 108, 'resolved': resolved, 'expected_refused': 108 - resolved, 'flag_disabled': a.disabled}))
