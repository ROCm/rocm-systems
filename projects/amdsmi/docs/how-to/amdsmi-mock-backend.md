# Test an AMD SMI consumer without GPU hardware

The CPU-only mock backend is a separate, opt-in test library for applications
that consume AMD SMI. It exposes the public C API from this source tree and
returns controlled identity, telemetry, and ECC values. No AMD GPU, kernel
driver, ROCm installation, root access, or Kubernetes cluster is required.

The backend is built independently from production AMD SMI. It has no install
rule and does not change the production library or enable a mock at runtime in
a production build. On Linux it produces `libamd_smi.so.<ABI-major>` with the
`AMDSMI_1` symbol version; the ABI major comes from the public header.

## Build and test

Requirements: CMake 3.20+, a C compiler, a C++17 compiler, and Python 3.9+.
Run these commands from the `rocm-systems` repository root:

```bash
cmake -S projects/amdsmi/tests/mock_backend -B /tmp/amdsmi-mock \
  -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/amdsmi-mock --parallel
ctest --test-dir /tmp/amdsmi-mock --output-on-failure
```

The contract tests compile against the public header and check discovery,
count-query/buffer handling, initialization reference counting, invalid
handles, API units, per-device isolation, concurrent reads, malformed/missing
state, and explicit unsupported results. Python integration tests load the
shared library through `ctypes`, run the control tool in another process, and
verify live changes without reinitializing the consumer. They also check that
all public functions in the preprocessed GPU-enabled header are exported.

The included C example uses only normal AMD SMI API calls. It has no dependency
on the control tool or mock-specific APIs. Linux AMD64 and ARM64 CI builds and
tests the backend on hosted CPU-only runners and checks the ELF SONAME and
symbol version. The standalone sources can also be built on macOS for local
contract tests; macOS does not provide Linux ELF compatibility.

## Observe telemetry and ECC changes

Create a new fixture directory; `init` refuses to overwrite a nonempty one:

```bash
export AMDSMI_MOCK_STATE_DIR="$(mktemp -d)"
python3 projects/amdsmi/tests/mock_backend/control.py \
  --state-dir "$AMDSMI_MOCK_STATE_DIR" init --gpus 2
/tmp/amdsmi-mock/mock_backend_probe
```

The initial output is:

```text
gpu=0 temperature_c=45 utilization_percent=0 power_w=100 vram_used_mib=0 ecc_correctable=0 ecc_uncorrectable=0
gpu=1 temperature_c=45 utilization_percent=0 power_w=100 vram_used_mib=0 ecc_correctable=0 ecc_uncorrectable=0
```

Set a high temperature, full utilization, and ECC counts on GPU 0:

```bash
python3 projects/amdsmi/tests/mock_backend/control.py \
  --state-dir "$AMDSMI_MOCK_STATE_DIR" set --gpu 0 \
  --temperature-c 95 --utilization 100 --correctable 3 --uncorrectable 7
/tmp/amdsmi-mock/mock_backend_probe
```

```text
gpu=0 temperature_c=95 utilization_percent=100 power_w=100 vram_used_mib=0 ecc_correctable=3 ecc_uncorrectable=7
gpu=1 temperature_c=45 utilization_percent=0 power_w=100 vram_used_mib=0 ecc_correctable=0 ecc_uncorrectable=0
```

These values travel through the AMD SMI C ABI into the C example. The control
tool only writes fixture files; it does not rewrite the consumer's output.
The live-library integration test additionally verifies updates within one
initialized process and confirms that the second GPU remains unchanged.

Restore the selected GPU's baseline telemetry while preserving its identity
and total memory:

```bash
python3 projects/amdsmi/tests/mock_backend/control.py \
  --state-dir "$AMDSMI_MOCK_STATE_DIR" reset --gpu 0
python3 projects/amdsmi/tests/mock_backend/control.py \
  --state-dir "$AMDSMI_MOCK_STATE_DIR" show
```

The ECC values are absolute counters, not increments. `reset` resets the
fixture; it does not implement `amdsmi_reset_gpu()`, which returns
`AMDSMI_STATUS_NOT_SUPPORTED`.

## Use another consumer

For a Linux application dynamically linked to the same AMD SMI ABI, select the
mock directory for that process:

```bash
AMDSMI_MOCK_STATE_DIR="$AMDSMI_MOCK_STATE_DIR" \
LD_LIBRARY_PATH="/tmp/amdsmi-mock${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  ./your-amdsmi-consumer
```

The application must initialize with `AMDSMI_INIT_AMD_GPUS`. Other initialization
flags, CPU/NIC discovery, and ESMI APIs are outside this first backend. A consumer
that loads an absolute library path or a bundled library needs its own library
selection mechanism. The backend does not support statically linked consumers.

Exporting a function does not mean its behavior is implemented. Check the
consumer's required APIs against the supported subset below. Compatibility with
the complete AMD CLI, GPU Agent, or device metrics exporter is not asserted by
these tests.

## Supported behavior

| API | Modeled result |
| --- | --- |
| `amdsmi_init`, `amdsmi_shut_down` | GPU-only discovery, serialized reference-counted lifecycle |
| `amdsmi_get_lib_version` | Version from this tree's header; build string identifies the mock |
| `amdsmi_get_socket_handles`, `amdsmi_get_socket_info` | One mock socket when GPUs are configured; none for zero GPUs |
| `amdsmi_get_processor_handles`, `amdsmi_get_processor_type` | Up to `AMDSMI_MAX_DEVICES` AMD GPU handles |
| `amdsmi_get_gpu_device_uuid`, `amdsmi_get_gpu_device_bdf` | Fixture identity |
| `amdsmi_get_gpu_asic_info` | Configured market name and AMD vendor identity; unmodeled numeric fields unavailable |
| `amdsmi_get_temp_metric` | Current edge temperature in Celsius |
| `amdsmi_get_gpu_activity` | GFX utilization in percent; MM/UMC activity unavailable |
| `amdsmi_get_power_info` | Socket power and power limit in watts; voltage fields unavailable |
| `amdsmi_get_gpu_memory_total`, `amdsmi_get_gpu_memory_usage` | VRAM only, in bytes |
| `amdsmi_get_gpu_vram_usage` | VRAM total/used in MiB, matching the implementation's 1024-based conversion |
| `amdsmi_get_gpu_total_ecc_count`, `amdsmi_get_gpu_ecc_count` | Correctable/uncorrectable counters; per-block queries support UMC only |

All remaining declarations in the preprocessed header have generated stubs
returning `AMDSMI_STATUS_NOT_SUPPORTED` without changing output buffers. CPU-only
APIs hidden behind `ENABLE_ESMI_LIB` are excluded. There is no successful no-op
implementation of reset, partition changes, or other hardware controls.

This backend simulates management observations. It does not execute GPU kernels,
create device nodes, implement SR-IOV or memory isolation, generate kernel RAS
events, or force scheduler actions. A higher temperature alone does not simulate
throttling; an ECC counter alone does not deliver an event notification.

## Fixture format and lifetime

Set `AMDSMI_MOCK_STATE_DIR` explicitly before the first initialization; there is
no default path. Initialization validates every record. The directory and GPU
count remain fixed until the final shutdown. Keep UUID, BDF, market name, and
total memory fixed while a consumer is initialized. Telemetry records are read
fresh for each supported API call.

The `count` file contains a decimal GPU count. Each `gpuN` file contains:

```text
AMDSMI_MOCK_V1
45 100 750 0 0 196608 0 0
AMD Instinct MI300X (mock)
00000000-0000-0000-0000-000000000000
0000:01:00.0
```

The numeric fields are: temperature Celsius, power watts, power cap watts,
utilization percent, used MiB, total MiB, correctable ECC, uncorrectable ECC.
The final three lines contain market name, UUID, and PCI BDF. The default
MI300X label and memory size are fixture values, not a complete hardware model.

The control tool validates updates and atomically replaces a single record.
Use one writer per state directory. Each API call observes one complete record;
multiple calls or GPUs do not form a transactional snapshot. Missing records
return `NOT_FOUND`; malformed records return `INVAL`. A failed update through
the tool preserves the previous file.

Stop consumers before deleting the temporary fixture directory. Handles are
valid only during an initialized session and must not be reused after the last
shutdown, matching the public API's lifetime rule.
