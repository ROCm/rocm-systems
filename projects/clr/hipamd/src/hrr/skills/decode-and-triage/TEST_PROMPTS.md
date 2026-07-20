# HRR decode-and-triage — test prompts

Copy-paste into a **new Cursor chat** (fresh session). The agent should load
`hrr-decode-and-triage` and run `scripts/triage_archive.sh`.

Symlink (optional): `~/.cursor/skills/hrr-decode-and-triage` → this directory.

---

## 1. MAF / read-only page fault (docker replay)

Large crash capture (~13M events). **Requires docker** — host `/opt/rocm` usually
does not match the capture stack.

```
Use the hrr-decode-and-triage skill. Triage with docker replay:
/var/lib/rancher/maf-repro/runs/fresh-v4-maf-20260716T102834Z/capture.hrr/pid-138

Docker image used for capture:
rocm/vllm:rocm7.13.0_gfx950-dcgpu_ubuntu24.04_py3.13_pytorch_2.10.0_vllm_0.19.1
```

**Expect:** outcome `MAF`, fault class `read_only_page_fault`, StreamK
`Cijk_..._MT128x192x128_..._SK3`, docker replay ~8 min. Legacy manifest (no
`metadata`) — preflight skipped.

Optional if GPU 0 is busy:

```
export GPU=1
```

---

## 2. Simple GEMM — native replay (small archive)

25 events, 4 kernels, clean shutdown. Best with **native** replay and a built
`hrr-playback` from the same CLR tree.

```
Use the hrr-decode-and-triage skill. Decode and triage this archive:
/home/amd/hrr/hrr-demo-capture/out/demo-gemm-native.hrr/pid-762659

Print the finding summary in your reply.
```

**Host env** (if native replay fails on library path):

```bash
export CLR_BUILD=/home/amd/hrr/rocm-systems/projects/clr/build-hrr713
export GPU=1
```

**Expect:** outcome `PASS`, fault class `replay_pass`, ~25 events, Complete YES.

---

## 3. Simple GEMM — metadata + preflight (#8680)

Exercises manifest `metadata` and `check_replay_compat.py` before replay.

```
Use the hrr-decode-and-triage skill. Decode and triage this archive:
/home/amd/hrr/hrr-demo-capture/out/demo-gemm-metadata-docker-20260717T131000Z.hrr/capture.hrr/pid-1

Docker image used for capture:
rocm/vllm:rocm7.13.0_gfx950-dcgpu_ubuntu24.04_py3.13_pytorch_2.10.0_vllm_0.19.1

Use docker replay. Print the finding summary in your reply.
```

**Expect:** preflight prints capture HIP/comgr + GPU count; finding includes
capture runtime metadata fields; replay PASS.

---

## 4. Metadata-only (no GPU replay)

Fast sanity check — parser + `--info` only.

```
Use the hrr-decode-and-triage skill. Metadata-only triage (no replay):
/home/amd/hrr/hrr-demo-capture/out/demo-gemm-native.hrr/pid-762659
```

**Expect:** no replay log; archive stats from `hrr-playback --info`; unit-test
policy satisfied by existing `test_*.py` in scripts/.

---

## Recreate GEMM captures

```bash
cd /home/amd/hrr/hrr-demo-capture
./build.sh && ./build-hrr.sh
GPU=1 ./capture.sh                    # -> out/demo-gemm-<ts>.hrr/pid-*
GPU=1 ./capture-docker.sh             # docker inject + metadata (develop)
GEMM_DEMO_CRASH=host GPU=1 ./capture-docker.sh   # incomplete manifest test
```

Replace `pid-*` paths in prompts with the new directory names.
