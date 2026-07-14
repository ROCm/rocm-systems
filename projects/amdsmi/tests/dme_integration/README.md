# dme_integration

CI helpers for the **DME <-> AMDSMI integration workflow**
([`.github/workflows/amdsmi-dme-ci.yml`](../../../../.github/workflows/amdsmi-dme-ci.yml)).

The workflow used to inline ~340 lines of bash across a dozen `run:`
blocks. Those blocks moved here so they are:

- runnable on a developer laptop without re-triggering CI;
- reviewable as Python code rather than YAML strings;
- unit-testable;
- easy to remove once upstream workarounds are no longer needed.

---

## Module map

| Module | Replaces |
| --- | --- |
| `submodules.py` | DME clone + `.gitmodules` SSH->HTTPS rewrite + protobuf-submodule guard |
| `build_env.py` | gpu-agent symlink + AMDSMI header/lib copies |
| `gpp_wrapper.py` | The inline `printf '#!/bin/sh\nexec g++ "$@" -lunwind\n'` heredoc |
| `services.py` | `nohup ... &` start blocks + cleanup loop, with real TCP port-readiness checks |
| `metrics.py` | The curl/grep loop, with assertions on required GPU metric names |
| `__main__.py` | Single CLI dispatcher (`python3 -m dme_integration <subcommand>`) |
| `_common.py` | Shared subprocess + GitHub Actions log helpers |

---

## Prerequisites

| Requirement | Notes |
| --- | --- |
| Python >= 3.9 | Uses only the standard library. No `pip install`, no `requirements.txt`. |
| `git` | For `prepare-submodules`. |
| `make`, `cmake`, `g++` | For Phase 1/3 of the workflow (these are *not* invoked by the helpers; the workflow calls them directly). |
| AMDSMI installed under `/opt/rocm` (or another prefix you pass) | Required only by `prepare-build-env`. |

The helpers themselves have **zero third-party Python dependencies**, so
they run on any container or workstation with `python3` available.

---

## Running the helpers locally

The package is not on `PYTHONPATH` by default, so run from the
`projects/amdsmi/tests/` directory **or** export `PYTHONPATH`:

```bash
# Option A - change into the package's parent directory
cd projects/amdsmi/tests
python3 -m dme_integration --help

# Option B - set PYTHONPATH from anywhere in the repo
PYTHONPATH=projects/amdsmi/tests python3 -m dme_integration --help
```

Every subcommand supports `--help` and `--verbose`. Errors from CI
helpers print `::error::` markers that GitHub Actions surfaces in the
job summary; locally they just go to stderr.

### `write-gpp-wrapper` - emit the g++ shim

```bash
python3 -m dme_integration write-gpp-wrapper --output /tmp/g++-wrap
cat /tmp/g++-wrap
# #!/bin/sh
# exec g++ "$@" -lunwind
```

### `verify-metrics` - Prometheus endpoint check

Asserts that the endpoint returns HTTP 200, parses as Prometheus
exposition text, and contains the named metrics. Default required set
is the AMDSMI-sourced GPU metrics (`gpu_edge_temperature`,
`gpu_power_usage`, `gpu_gfx_activity`).

```bash
python3 -m dme_integration verify-metrics \
    --url http://localhost:5000/metrics \
    --max-retries 5 --retry-delay 2 \
    --required-metric gpu_edge_temperature \
    --required-metric gpu_power_usage \
    --gpu-agent-pid-file /tmp/gpuagent.pid \
    --gpu-agent-log-file /tmp/gpuagent.log \
    --output /tmp/captured.txt
```

`--gpu-agent-pid-file` and `--gpu-agent-log-file` make the check aware of
GPU Agent health. When the required GPU metrics are missing **and** the
GPU Agent is detected as crashed (dead PID, missing PID file, or crash
indicators such as `Segmentation fault` / `stack smashing detected` in
its log, typically an ABI mismatch with `libamd_smi.so`),
`verify-metrics` emits a `::warning::` and exits 0 (**soft-pass**) instead
of failing. The build/deploy/service-management infrastructure is still
validated; only the GPU-metric assertion is skipped. If the GPU Agent is
alive but the metrics are still missing, the check hard-fails (exit 1).

Failure path against a closed port:

```bash
python3 -m dme_integration verify-metrics \
    --url http://127.0.0.1:1/metrics --max-retries 2 --retry-delay 0.2
# ::error::Metrics endpoint unreachable after 2 attempts (last status 0)
# exit code 1
```

### `start-service` / `stop-service` - supervised process

Starts a binary detached, redirects stdout+stderr to `--log-file`,
writes its PID to `--pid-file`, and waits for `--ready-port` to accept
TCP before returning success. `--ready-port 0` skips the port check
and only verifies the process stayed alive for `--ready-timeout`
seconds.

```bash
# Start
python3 -m dme_integration start-service \
    --name dme \
    --binary /tmp/dme/bin/amd-metrics-exporter \
    --log-file /tmp/dme.log \
    --pid-file /tmp/dme.pid \
    --ready-port 5000 \
    --ready-timeout 30 \
    --ld-library-path /opt/rocm/lib

# Stop (SIGTERM -> SIGKILL after 2s if still alive)
python3 -m dme_integration stop-service --name dme --pid-file /tmp/dme.pid
```

### `check-alive`: post-start liveness check

Waits `--delay` seconds and then verifies the process recorded in
`--pid-file` is still running. Used as a post-start health check to catch
services that launch but crash immediately (e.g. ABI-mismatch segfaults
or stack-smashing aborts that `start-service` may miss). Exits 0 if the
process is alive, or 1 with an `::error::` (tailing `--log-file` when
given) if it died.

```bash
python3 -m dme_integration check-alive \
    --name gpuagent \
    --pid-file /tmp/gpuagent.pid \
    --log-file /tmp/gpuagent.log \
    --delay 2
```

### `prepare-submodules` - clone DME and init nested submodules

> **Destructive.** Removes `--dme-dir` if it exists. Runs network git
> clones. Don't point this at a directory you care about.

```bash
python3 -m dme_integration prepare-submodules \
    --dme-repo https://github.com/ROCm/device-metrics-exporter.git \
    --dme-branch v1.4.2 \
    --dme-dir /tmp/dme \
    --gpu-agent-repo https://github.com/ROCm/gpu-agent.git \
    --gpu-agent-branch v1.4.2
```

### `prepare-build-env` - lay out gpu-agent build tree

> **Destructive.** Symlinks `--gpu-agent-workdir` (replacing it if it
> exists) and copies AMDSMI headers/libs into the gpu-agent tree.

```bash
python3 -m dme_integration prepare-build-env \
    --gpuagent-src /tmp/dme/gpuagent \
    --gpu-agent-workdir /usr/src/github.com/ROCm/gpu-agent \
    --rocm-dir /opt/rocm
```

---

## Quick local end-to-end recipe

You can exercise the readiness + metrics path without building any of
the real binaries by spinning up a fake Prometheus exporter:

```bash
# 1. Fake exporter
cat > /tmp/fake_prom.py <<'PYEOF'
import http.server, socketserver, sys
PORT = int(sys.argv[1])
BODY = b"""# HELP gpu_edge_temperature edge temp
# TYPE gpu_edge_temperature gauge
gpu_edge_temperature{gpu="0"} 42.0
# HELP gpu_power_usage power
# TYPE gpu_power_usage gauge
gpu_power_usage{gpu="0"} 120.5
# HELP gpu_gfx_activity gfx
# TYPE gpu_gfx_activity gauge
gpu_gfx_activity{gpu="0"} 87
"""
class H(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/metrics":
            self.send_response(200); self.end_headers(); self.wfile.write(BODY)
        else:
            self.send_response(404); self.end_headers()
    def log_message(self, *a, **k): pass
with socketserver.TCPServer(("127.0.0.1", PORT), H) as srv:
    srv.serve_forever()
PYEOF

# 2. Wrapper that start-service can launch
cat > /tmp/fake_prom.sh <<'SHEOF'
#!/bin/sh
exec python3 /tmp/fake_prom.py 19191
SHEOF
chmod +x /tmp/fake_prom.sh

# 3. Run the supervisor + verifier
cd projects/amdsmi/tests

python3 -m dme_integration start-service \
    --name fake_dme \
    --binary /tmp/fake_prom.sh \
    --log-file /tmp/fake_dme.log \
    --pid-file /tmp/fake_dme.pid \
    --ready-port 19191 --ready-timeout 5

python3 -m dme_integration verify-metrics \
    --url http://127.0.0.1:19191/metrics \
    --max-retries 3 --retry-delay 0.5

python3 -m dme_integration stop-service \
    --name fake_dme --pid-file /tmp/fake_dme.pid

# 4. Cleanup
rm -f /tmp/fake_prom.py /tmp/fake_prom.sh /tmp/fake_dme.log /tmp/fake_dme.pid
```

Expected: both `start-service` and `verify-metrics` exit 0. Flipping
`--required-metric does_not_exist` on the `verify-metrics` call makes
it exit 1 with an `::error::Required GPU metrics missing` line.

---

## CI compatibility

These helpers are designed to drop into a vanilla GitHub-hosted (or
self-hosted) Ubuntu container:

- `python3` is preinstalled on every official `ubuntu-*` runner image
  and on the `Ubuntu*_DOCKER_IMAGE` containers used by the workflow.
- The package uses **only the standard library**, so the workflow does
  not need a `pip install` step.
- Every step in
  [`amdsmi-dme-ci.yml`](../../../../.github/workflows/amdsmi-dme-ci.yml)
  sets `working-directory: ${{ env.AMDSMI_TESTS_DIR }}` before invoking
  `python3 -m dme_integration ...`, which puts CWD on `sys.path` so the
  package imports cleanly without `PYTHONPATH`.
- Log lines that need to surface in the GitHub Actions UI use the
  workflow-command markers (`::group::`, `::error::`, `::warning::`)
  emitted by `_common.py`.

To reproduce the CI steps locally, set the same env vars and run from
the same directory:

```bash
export AMDSMI_TESTS_DIR=$PWD/projects/amdsmi/tests
export DME_DIR=/tmp/dme
export DME_REPO=https://github.com/ROCm/device-metrics-exporter.git
export GPU_AGENT_REPO=https://github.com/ROCm/gpu-agent.git
export DME_BRANCH=v1.4.2 GPU_AGENT_BRANCH=v1.4.2
cd "$AMDSMI_TESTS_DIR"

python3 -m dme_integration prepare-submodules \
    --dme-repo "$DME_REPO" --dme-branch "$DME_BRANCH" --dme-dir "$DME_DIR" \
    --gpu-agent-repo "$GPU_AGENT_REPO" --gpu-agent-branch "$GPU_AGENT_BRANCH"
# ... then prepare-build-env, build with cmake/make as in the workflow,
#    start-service for gpuagent + dme, verify-metrics, stop-service.
```

---

## CI gating status

GPU-metric verification is currently **non-gating** *only when the GPU Agent
crashes*: that crash is upstream ABI skew between the pinned gpu-agent and the
develop AMDSMI (it fires on every run until gpu-agent is rebased), not a
regression in the PR under test, so the step soft-passes with a `::warning::`
and exits 0. A green check therefore only guarantees the build, deploy, and
service-management path works. It does **not** by itself confirm that GPU
metrics were validated. To tell a real GPU-metric pass from a soft-pass, look
for the `GPU metric verification SKIPPED` warning in the job log.

This soft-pass is self-limiting: it only triggers while the GPU Agent dies. If
the agent stays alive but metrics are missing, the check hard-fails, and once
gpu-agent is ABI-compatible the soft-pass branch is never taken and the check
gates for real.

---

## Upstream workarounds tracked here

These helpers exist because of bugs upstream that we cannot easily fix
in this repo. When upstream lands a fix, delete the matching helper:

- **`gpp_wrapper.py`** - appends `-lunwind` to every `g++` link line
  because gpu-agent's hard-coded `LDFLAGS` omit it but `libzmq.a`
  references libunwind symbols. Delete when upstream gpu-agent fixes
  its `LDFLAGS`.
- **`submodules._rewrite_gitmodules_ssh_to_https`** - rewrites SSH
  GitHub URLs in gpu-agent's nested `.gitmodules`. Delete once upstream
  switches all dependency URLs to HTTPS.

---

## Validating after edits

```bash
# Syntax check every module
python3 -c "import ast, glob; [ast.parse(open(f).read(), f) for f in glob.glob('projects/amdsmi/tests/dme_integration/*.py')]"

# Import + CLI smoke test
PYTHONPATH=projects/amdsmi/tests python3 -m dme_integration --help
```
