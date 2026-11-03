import time
import os
import json
import psutil
import runpy
import requests
from pathlib import Path

root = Path(os.environ["GRAPH_TRACE_DIR"])
for attempt in range(120):
    try:
        response = requests.get(
            os.environ.get("SGLANG_BASE_URL", "http://127.0.0.1:31241") + "/health",
            timeout=5,
        )
        if response.status_code == 200:
            break
    except requests.RequestException:
        pass
    time.sleep(10)
else:
    raise RuntimeError("Server did not become ready within 20 minutes")
print("Server healthy; starting requests", flush=True)
server = psutil.Process(int(os.environ["SGLANG_SERVER_PID"]))
rows = [server.as_dict(attrs=["pid", "ppid", "name", "cmdline"])]
rows.extend(
    p.as_dict(attrs=["pid", "ppid", "name", "cmdline"])
    for p in server.children(recursive=True)
)
(root / "processes.json").write_text(json.dumps(rows, indent=2))
runpy.run_path(str(Path(__file__).with_name("bench.py")), run_name="__main__")
root.joinpath("finalize").write_text("request completed; flush idle workers\n")
print("Request complete; requested trace finalization", flush=True)
