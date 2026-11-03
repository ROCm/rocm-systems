import json
import time
import os
from pathlib import Path
import requests
from transformers import AutoTokenizer

BASE = os.environ.get("SGLANG_BASE_URL", "http://127.0.0.1:31241")
OUT = Path(os.environ["GRAPH_TRACE_DIR"])
tok = AutoTokenizer.from_pretrained(os.environ["MODEL_PATH"], trust_remote_code=True)
prefix = tok.encode(
    "Read the following technical notes and write a detailed summary.\n",
    add_special_tokens=False,
)
unit = tok.encode(
    "The inference system processes requests in order. Each worker stores model weights and a key value cache. Long documents require careful memory management. Performance measurements distinguish prompt processing from autoregressive token generation.\n",
    add_special_tokens=False,
)
end = tok.encode(
    "\nSummarize the notes above and explain the practical engineering implications in detail.\n",
    add_special_tokens=False,
)


def run(n, out_tokens, label):
    requests.post(BASE + "/flush_cache", timeout=120).raise_for_status()
    ids = prefix + (unit * ((n // len(unit)) + 1))[: n - len(prefix) - len(end)] + end
    assert len(ids) == n
    payload = {
        "input_ids": ids,
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": out_tokens,
            "ignore_eos": True,
        },
        "stream": True,
    }
    clocks_start = {
        str(c): time.clock_gettime_ns(c)
        for c in (time.CLOCK_MONOTONIC, time.CLOCK_MONOTONIC_RAW, time.CLOCK_BOOTTIME)
    }
    started = time.perf_counter()
    first = None
    events = []
    last = {}
    with requests.post(
        BASE + "/generate", json=payload, stream=True, timeout=(30, 7200)
    ) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines(chunk_size=1):
            if not line or not line.startswith(b"data: "):
                continue
            raw = line[6:]
            if raw == b"[DONE]":
                break
            obj = json.loads(raw)
            now = time.perf_counter()
            meta = obj.get("meta_info", {})
            count = meta.get("completion_tokens", 0)
            if obj.get("error"):
                raise RuntimeError(obj)
            if count and (not events or count > events[-1]["tokens"]):
                if first is None:
                    first = now
                events.append({"elapsed_s": now - started, "tokens": count})
            last = obj
    ended = time.perf_counter()
    count = last["meta_info"]["completion_tokens"]
    assert count == out_tokens, (count, last)
    elapsed = events[-1]["elapsed_s"] - events[0]["elapsed_s"]
    decoded = count - events[0]["tokens"]
    result = {
        "clocks_start_ns": clocks_start,
        "clocks_end_ns": {
            str(c): time.clock_gettime_ns(c)
            for c in (time.CLOCK_MONOTONIC, time.CLOCK_MONOTONIC_RAW, time.CLOCK_BOOTTIME)
        },
        "label": label,
        "input_tokens_requested": n,
        "output_tokens_requested": out_tokens,
        "ttft_s": first - started,
        "e2e_s": ended - started,
        "decode_tokens_per_s": decoded / elapsed,
        "output_tokens_per_e2e_s": count / (ended - started),
        "meta_info": last["meta_info"],
        "events": events,
        "text": last.get("text", ""),
    }
    (OUT / (label + ".json")).write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(
        json.dumps(
            {k: v for k, v in result.items() if k not in ["events", "text"]},
            ensure_ascii=False,
        ),
        flush=True,
    )


run(4096, 128, "warmup")
run(524288, 128, "512k-run-1")
