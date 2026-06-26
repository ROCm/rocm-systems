# find-amdsmi-dependents

Find all external projects that depend on or use amdsmi (AMD System Management Interface), excluding ROCm itself.

## Instructions

Search GitHub and other sources for projects that import, link against, or otherwise depend on `amdsmi` but are NOT part of the ROCm organization or the main amdsmi repository.

### Step 1: Search GitHub for amdsmi usage

```bash
# Python imports
gh search code "import amdsmi" --limit 100 --json repository,path,url

# requirements.txt / pip dependencies
gh search code "amdsmi" --extension requirements.txt --limit 100 --json repository,path,url

# pyproject.toml dependencies
gh search code "amdsmi" --extension toml --limit 100 --json repository,path,url

# C/C++ includes
gh search code "include amdsmi" --limit 100 --json repository,path,url

# CMake linkage
gh search code "amdsmi" --extension cmake --limit 100 --json repository,path,url

# pkg-config or linker flags
gh search code "libamd_smi" --limit 100 --json repository,path,url
```

### Step 2: Check PyPI reverse dependencies

Visit or fetch: https://pypi.org/project/amdsmi/

Also search for packages that list amdsmi as a dependency:
```bash
# Use pip-api or pypinfo to find reverse deps
curl -s "https://pypistats.org/api/packages/amdsmi/recent" | python3 -m json.tool
```

### Step 3: Filter out ROCm/AMD-owned repositories

Exclude results from:
- `ROCm/amdsmi` (source repo)
- `ROCm/rocm-systems`
- `ROCm/device-metrics-exporter`
- `amd/MxGPU-Virtualization`
- Any repo under `ROCm/` or `amd/` orgs that are direct ROCm components

### Step 4: Categorize findings

Group external projects by use case:

| Category | Examples |
|----------|---------|
| Monitoring/Observability | Prometheus exporters, Grafana plugins, node exporters |
| ML/AI Frameworks | Training frameworks, inference engines using GPU metrics |
| Cluster/HPC Management | Slurm plugins, PBS plugins, MPI wrappers |
| Kubernetes/Cloud | Device plugins, operators, autoscalers |
| Benchmarking | GPU benchmark suites using power/thermal data |
| Developer Tools | Profilers, debuggers, IDEs |

### Step 5: Check known ecosystem projects explicitly

- `prometheus-community` GPU exporters
- Kubernetes GPU device plugins
- `wandb`, `mlflow`, `clearml` — ML experiment tracking
- `nvitop` or similar cross-vendor GPU monitoring CLIs
- Slurm `gres` plugins
- OpenStack / cloud-init GPU configuration

### Step 6: Output report

Produce a summary table:

| Project | Category | Integration | Proof Link |
|---------|----------|-------------|------------|
| ... | ... | Python API / C lib / pkg dep | https://github.com/... |

**Integration Type** options:
- `Python API` — `import amdsmi` in Python code
- `C/C++ library` — links against `libamd_smi.so`
- `CLI wrapper` — shells out to `amd-smi` command
- `pkg dependency` — lists amdsmi in requirements/pyproject/cmake

**Activity**: check latest commit/release date to mark Active / Stale / Archived.

---

## Last Run Results (2026-06-26)

| Project | Category | Integration | Proof Link |
|---------|----------|-------------|------------|
| pytorch/pytorch | ML Framework | Python API + C lib | [memory.py](https://github.com/pytorch/pytorch/blob/68c05c738d9eb3e80eda31f882e9128a7a770488/torch/cuda/memory.py) · [CMakeLists.txt](https://github.com/pytorch/pytorch/blob/main/caffe2/CMakeLists.txt) |
| vllm-project/vllm | LLM Inference | Python API | [platforms/__init__.py](https://github.com/vllm-project/vllm/blob/c6554f321ce4c7563290d02eec323f262fc43fef/vllm/platforms/__init__.py) |
| huggingface/accelerate | ML Framework | Python API | [environment.py](https://github.com/huggingface/accelerate/blob/26d16dea23ca7d71cd5ab6f6a7a71e5ee9b162bd/src/accelerate/utils/environment.py) |
| huggingface/optimum-benchmark | Benchmarking | Python API | [system_utils.py](https://github.com/huggingface/optimum-benchmark/blob/702a15ec4a239bd5d594601dbd03bd645cc66674/optimum_benchmark/system_utils.py) |
| ml-energy/zeus | Energy Monitoring | Python API + pkg dep | [gpu/amd.py](https://github.com/ml-energy/zeus/blob/main/zeus/device/gpu/amd.py) · [pyproject.toml](https://github.com/ml-energy/zeus/blob/main/pyproject.toml) |
| mlco2/codecarbon | Carbon Tracking | Python API | [amdsmi_demo.py](https://github.com/mlco2/codecarbon/blob/11374f42ab9ddbdd0c47b180dc9507ff4258a01a/examples/slurm_rocm/amdsmi_demo.py) |
| microsoft/lisa | Cloud Testing | Python API | [gpu_drivers.py](https://github.com/microsoft/lisa/blob/f1ba590d2a41987150e13b70ff5aaf62e579b7bd/lisa/tools/gpu_drivers.py) |
| interTwin-eu/itwinai | HPC/AI | Python API + pkg dep | [monitoring/backend.py](https://github.com/interTwin-eu/itwinai/blob/c175f9ada2624e902dc9de3221753ff85b314ef1/src/itwinai/torch/monitoring/backend.py) · [pyproject.toml](https://github.com/interTwin-eu/itwinai/blob/main/pyproject.toml) |
| ORNL/flowcept | HPC Monitoring | Python API + pkg dep | [olcf_frontier/README.md](https://github.com/ORNL/flowcept/blob/dc556a5efc5fdd661c52ded21479faa0d815e27f/examples/olcf_frontier/README.md) · [pyproject.toml](https://github.com/ORNL/flowcept/blob/main/pyproject.toml) |
| ai-dynamo/aiperf | AI Perf | Python API + C lib | [amdsmi_collector.py](https://github.com/ai-dynamo/aiperf/blob/ab7d3ad86bbcc211ba1f703a6891f92bf8888f9e/src/aiperf/gpu_telemetry/amdsmi_collector.py) |
| openlit/openlit | Observability | Python API | [gpu/__init__.py](https://github.com/openlit/openlit/blob/main/sdk/python/src/openlit/instrumentation/gpu/__init__.py) |
| KernelTuner/kernel_tuner | Benchmarking | Python API | [observers/amd.py](https://github.com/KernelTuner/kernel_tuner/blob/main/kernel_tuner/observers/amd.py) |
| Helmholtz-AI-Energy/perun | Energy Monitoring | pkg dep | [pyproject.toml](https://github.com/Helmholtz-AI-Energy/perun/blob/main/pyproject.toml) |
| Mandark-droid/genai_otel_instrument | Observability | Python API + pkg dep | [gpu_metrics.py](https://github.com/Mandark-droid/genai_otel_instrument/blob/main/genai_otel/gpu_metrics.py) · [pyproject.toml](https://github.com/Mandark-droid/genai_otel_instrument/blob/main/pyproject.toml) |
| HPCI-Lab/yProv4ML | ML Provenance | pkg dep | [pyproject.toml](https://github.com/HPCI-Lab/yProv4ML/blob/main/pyproject.toml) |
| last9/gpu-telemetry | Observability | pkg dep | [pyproject.toml](https://github.com/last9/gpu-telemetry/blob/main/pyproject.toml) |
| Tencent-Hunyuan/HY-WorldPlay | LLM | Python API | [wan/platforms/__init__.py](https://github.com/Tencent-Hunyuan/HY-WorldPlay/blob/1588e1336e842b03b0a7860c654ebd7c46bb065e/wan/platforms/__init__.py) |
| yichao-yuan-99/vllm-otel | Observability | Python API + pkg dep | [amdsmi_power_service.py](https://github.com/yichao-yuan-99/vllm-otel/blob/main/amd-power-reader/deamon/amdsmi_power_service.py) |
| gpustack/gpustack-operator | Kubernetes/Cloud | C lib (Go binding) | [binding/amdsmi/library.go](https://github.com/gpustack/gpustack-operator/blob/main/binding/amdsmi/library.go) |
| alumet-dev/amd-smi-wrapper | Rust Wrapper | C lib | [amd-smi-wrapper-sys/src/lib.rs](https://github.com/alumet-dev/amd-smi-wrapper/blob/main/amd-smi-wrapper-sys/src/lib.rs) |
| sustainai-energy/amdsmi-wrapper | Rust Wrapper | C lib | [amdsmi-wrapper-sys/build.rs](https://github.com/sustainai-energy/amdsmi-wrapper/blob/main/amdsmi-wrapper-sys/build.rs) |
| FrancoisGib/amdsmi-wrapper | Rust Wrapper | C lib | [amdsmi/src/utils.rs](https://github.com/FrancoisGib/amdsmi-wrapper/blob/main/amdsmi/src/utils.rs) |
| adam-mcdaniel/scorep-amdsmi-plugin | HPC Profiling | C lib | [apapi.c](https://github.com/adam-mcdaniel/scorep-amdsmi-plugin/blob/main/apapi.c) |
| passlab/pinsight | HPC Profiling | C lib | [src/energy_amd_smi.c](https://github.com/passlab/pinsight/blob/main/src/energy_amd_smi.c) |
| icl-utk-edu/papi | HPC Counters | C lib | [src/components/amd_smi/README.md](https://github.com/icl-utk-edu/papi/blob/master/src/components/amd_smi/README.md) |
| Osmanyasal/OPTKIT | Perf Toolkit | C lib | [examples/gpm_workload/Makefile](https://github.com/Osmanyasal/OPTKIT/blob/main/examples/gpm_workload/Makefile) |
| Netflix-Skunkworks/atlas-system-agent | Infrastructure | C lib | [conanfile.py](https://github.com/Netflix-Skunkworks/atlas-system-agent/blob/main/conanfile.py) |
| zml/zml | ML Framework | C lib | [bin/zml-smi/BUILD.bazel](https://github.com/zml/zml/blob/main/bin/zml-smi/BUILD.bazel) |
| AMDResearch/intellikit | AMD Dev Tools | Python API + pkg dep | [rocm_mcp/pyproject.toml](https://github.com/AMDResearch/intellikit/blob/main/rocm_mcp/pyproject.toml) |
| microsoft/azurelinux | Linux Distro | pkg | [specs/a/amdsmi/amdsmi.spec](https://github.com/microsoft/azurelinux/blob/main/specs/a/amdsmi/amdsmi.spec) |
| gentoo/gentoo | Linux Distro | pkg | [dev-util/amdsmi/amdsmi-7.1.1.ebuild](https://github.com/gentoo/gentoo/blob/master/dev-util/amdsmi/amdsmi-7.1.1.ebuild) |
| easybuilders/easybuild-easyconfigs | HPC Build System | pkg | [amdsmi-25.4.2 patch](https://github.com/easybuilders/easybuild-easyconfigs/blob/develop/easybuild/easyconfigs/a/amdsmi/amdsmi-25.4.2_handle-non-standard-rocm-paths.patch) |
