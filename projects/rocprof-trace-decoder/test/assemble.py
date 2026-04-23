import json
import glob
import sys

target = sys.argv[1]

att_files = [att.split('/')[-1] for att in glob.glob(target+'/*.att')]
codeobj = [c.split('/')[-1] for c in glob.glob(target+'/*.out')]
codeobj = {int(c.split('_')[-1].split('.out')[0]): c for c in codeobj}
maxobj = max(codeobj.keys())

obj = ["" for k in range(1+maxobj)]
code_objects = [{"uri": ""} for k in range(1+maxobj)]

for k, v in codeobj.items():
    obj[k] = v
    code_objects[k] = {"code_object_id": k, 'load_delta': k*0x1000000, 'load_size': 0x1000000, "uri": "test"}

fs = {}
fs['strings'] = {}
fs['strings']['att_filenames'] = att_files
fs['strings']['code_object_snapshot_filenames'] = obj
fs['code_objects'] = code_objects

out = {}
out['rocprofiler-sdk-tool'] = []
out['rocprofiler-sdk-tool'].append(fs)

with open(target+'/1_results.json', 'w') as f:
    json.dump(out, f)
