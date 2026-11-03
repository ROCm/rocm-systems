import csv
import json
from collections import Counter
from pathlib import Path
from inspect_graph import analyze
root=Path(__import__("sys").argv[1])
request=json.loads((root/"512k-run-1.json").read_text())
assert request["meta_info"]["prompt_tokens"]==524288
assert request["meta_info"]["completion_tokens"]==128
window=(request["clocks_start_ns"]["7"],request["clocks_end_ns"]["7"])
processes=json.loads((root/"processes.json").read_text())
ranks=[p for p in processes if p["name"].startswith("sglang::scheduler_TP")]
assert len(ranks)==4
summary={"request":request,"clock":"CLOCK_BOOTTIME","request_window_ns":window,"ranks":[]}
for proc in ranks:
    path=root/f"trace-{proc['pid']}.json"
    doc=json.loads(path.read_text())
    rows,stats=analyze(doc,window)
    stats.update(pid=proc["pid"],rank=proc["name"],trace=path.name)
    stats["kernel_counts"]=dict(Counter(r["kernel_name"] for r in rows))
    stats["trace_finalized_marker"]=(root/f"finalized-{proc['pid']}").exists()
    stats["total_trace_dispatches"]=len(doc["rocprofiler-sdk-json-tool"]["buffer_records"]["kernel_dispatch"])
    output=root/f"graph-{proc['name'].split('_')[-1]}.csv"
    if rows:
        with output.open("w") as f:
            writer=csv.DictWriter(f,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)
    summary["ranks"].append(stats)
    print(proc["name"],stats["graph_dispatches"],stats["capture_resolved_dispatches"],stats["missing_kernel_names"],flush=True)
(root/"graph-summary.json").write_text(json.dumps(summary,indent=2))
