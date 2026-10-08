# QEMU VFIO-user compute

This guide describes the QEMU/VFIO-user path that rocJITsu supports today.
The supported device profile is a single gfx1250 compute GPU using
`configs/gfx1250_mi455x.json`. QEMU owns the guest, while rocJITsu supplies
the emulated PCI device and the shared GPU execution models.

The repository provides the VFIO-user server and
`scripts/run-vfio-guest.py`. It does not build or distribute a guest kernel,
initramfs, AMDGPU package, ROCm runtime, or rocBLAS payload. Those inputs must
come from one mutually compatible public release or nightly. Record their
versions and SHA-256 digests when publishing a result.

## Host prerequisites

- Linux on x86-64.
- CMake 3.28 or newer for a VFIO-enabled build.
- Linux UAPI headers containing the VFIO migration and DMA logging interfaces
  used by libvfio-user (Linux 6.1 or newer is a practical baseline).
- Ninja and a supported C++20 compiler.
- Python 3.10 or newer for the launcher.
- QEMU with the `vfio-user-pci` device.
- Internet access during the first CMake configure, unless the fetched
  libvfio-user and json-c dependencies are already available locally.
- Optional access to `/dev/kvm`. The launcher falls back to TCG.

Check the QEMU capabilities before building a guest:

```bash
qemu-system-x86_64 -accel help
qemu-system-x86_64 -device help | rg vfio-user-pci
```

## Build the VFIO-user server

Run these commands from the rocjitsu source directory:

```bash
cmake -S . -B build-vfio -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DROCJITSU_ENABLE_VFIO=ON
cmake --build build-vfio --target rocjitsu_bin rj_ip_discovery
```

The VFIO dependency gate runs during configuration and rejects CMake older than
3.28 while leaving the ordinary build's 3.22 minimum unchanged.

The resulting server is:

```text
build-vfio/tools/rocjitsu/rocjitsu
```

The build also produces the profile-specific IP-discovery generator at:

```text
build-vfio/tools/rj-ip-discovery/rj-ip-discovery
```

## Prepare compatible guest artifacts

The kernel, AMDGPU module, firmware, ROCm user-mode stack, and rocBLAS payload
must be mutually compatible. A current ROCm nightly is suitable when it has
gfx1250 support and its AMDGPU driver contains commit `4e07da515d1c`
(`drm/amdgpu: enumerate UMSCH HW IP`) or an equivalent backport. That fix
allows discovery of this compute-only device without a VCN instance.

The gfx1250 profile intentionally omits display and media IP such as JPEG,
VCN, and UVD. It retains the management-processor discovery records required
by the driver; those records do not imply a complete firmware-managed hardware
model. This does not mean that the Linux driver never opens firmware files.
With direct firmware loading, current AMDGPU code still parses firmware for
the enabled compute, MES, SDMA, and IMU paths. Package the firmware files that
match the exact AMDGPU module in the initramfs. The authoritative inventory
for a built module is:

```bash
modinfo -F firmware /path/to/amdgpu.ko | sort -u
```

For the current gfx1250 path, requests can include
`amdgpu/gc_12_1_0_imu.bin`, `amdgpu/gc_12_1_0_mec.bin`,
`amdgpu/gc_12_1_0_rlc_1.bin`, a matching gfx1250 MES image, and
`amdgpu/sdma_7_1_0.bin`. Use files from the same public driver/firmware
release. Do not add JPEG or VCN firmware, because those IP blocks are absent.

The launch command below selects file-based IP discovery with
`amdgpu.discovery=2`. Its `amdgpu/ip_discovery.bin` is not a vendor firmware
blob and must not be copied from an unrelated package. Generate it from the
same rocJITsu gfx1250 profile used by the emulated device:

```bash
mkdir -p /path/to/staging-root/lib/firmware/amdgpu
build-vfio/tools/rj-ip-discovery/rj-ip-discovery \
  gfx1250 \
  /path/to/staging-root/lib/firmware/amdgpu/ip_discovery.bin
```

Regenerate the table whenever the rocJITsu device profile changes, then pack
that staging tree into the initramfs.

The initramfs must contain:

- an executable `/init`;
- `/dev`, `/proc`, and `/sys` mount points;
- the guest kernel's AMDGPU module and all of its module dependencies;
- every firmware file requested by the selected gfx1250 driver path;
- the dynamic loader and transitive libraries needed by the workload;
- a gfx1250-capable ROCr/HIP runtime, rocBLAS, its gfx1250 Tensile metadata and
  code object, and a GEMM validation executable;
- a shutdown command such as BusyBox `poweroff`.

The init process must:

1. Mount devtmpfs, procfs, and sysfs.
2. Load the AMDGPU module and its dependencies.
3. Fail if AMDGPU did not bind the `0x1002:0x75c1` PCI function.
4. Fail if `/dev/kfd` is absent.
5. Run the GEMM validator and propagate its result.
6. Print stable success or failure markers.
7. Power the guest off. QEMU runs with `-no-reboot`, so poweroff is the
   normal successful exit path.

Use a fresh staging directory and an ordinary initramfs builder, or pack a
reviewed staging tree directly:

```bash
(
  cd /path/to/staging-root
  find . -print0 | cpio --null --create --format=newc
) | gzip -9 > /path/to/gfx1250-compute-initramfs.gz
```

Do not copy an arbitrary host `/opt/rocm` tree. Copy the selected package
payloads and all transitive ELF dependencies, preserve symlinks, and verify
that the rocBLAS library data contains a gfx1250 code object. A missing loader,
library, Tensile data file, or firmware file normally appears as a probe or
workload failure rather than as a host-launch failure.

## GEMM qualification contract

A useful qualification workload calls `rocblas_sgemm` and compares every
output element against a CPU reference. The reference qualification used an
FP32, column-major, non-transposed 128 x 128 x 128 multiply with alpha 1 and
beta 0. Its inputs were deterministic. A validator can expose this stable log
contract:

```text
guest: /dev/kfd present
sgemm-validator: api=rocblas_sgemm ... m=128 n=128 k=128 alpha=1 beta=0
sgemm-validator: max_absolute_error=... failures=0
sgemm-validator: PASS
guest: done
```

Do not qualify the device merely because the kernel probed or a kernel was
submitted. Require numerical validation, a zero-status QEMU exit, and orderly
rocJITsu shutdown. The packaged rocBLAS installation must select a supported
gfx1250 backend and find its matching library data without runtime environment
overrides.

## Launch the guest

Choose a short output path because Unix-domain socket paths are limited. The
output directory must not already exist.

```bash
QEMU=/path/to/qemu-system-x86_64
KERNEL=/path/to/vmlinuz
INITRAMFS=/path/to/gfx1250-compute-initramfs.gz
ROCJITSU=build-vfio/tools/rocjitsu/rocjitsu
OUTPUT=/tmp/rocjitsu-gemm-run

python3 scripts/run-vfio-guest.py \
  --qemu "${QEMU}" \
  --kernel "${KERNEL}" \
  --initramfs "${INITRAMFS}" \
  --rocjitsu "${ROCJITSU}" \
  --config configs/gfx1250_mi455x.json \
  --output "${OUTPUT}" \
  --accel auto \
  --memory 4G \
  --probe-timeout 10 \
  --startup-timeout 60 \
  --append 'console=ttyS0 rdinit=/init panic=-1 amdgpu.discovery=2 amdgpu.emu_mode=1 amdgpu.fw_load_type=0 amdgpu.vm_update_mode=3 amdgpu.gpu_recovery=0 amdgpu.vramlimit=1024 amdgpu.ip_block_mask=0x3f rocjitsu.workload=sgemm' \
  --expect-log 'guest: /dev/kfd present' \
  --expect-log 'sgemm-validator: PASS' \
  --reject-log 'sgemm-validator: FAIL' \
  --reject-log 'guest: FATAL:'
```

The 1 GiB VRAM limit leaves enough guest VRAM for ROCr to provision queue
scratch at the device's advertised occupancy. A smaller limit can be adequate
for scratch-free kernels but may make a private-segment dispatch wait
indefinitely for scratch allocation.

The repository does not provide the validator binary, so make the guest
validator emit the example markers above or replace the `--expect-log` and
`--reject-log` values with its stable public output.

`--accel auto` selects KVM when QEMU advertises it and `/dev/kvm` is
usable; otherwise it selects TCG. KVM and TCG both expose the same VFIO-user
device. TCG is slower but avoids a host KVM dependency.

Run the same command with `--dry-run` first. Dry-run validates all input paths
and executable permissions, confirms that QEMU provides `vfio-user-pci`, asks
the selected rocJITsu binary to confirm that it was built with VFIO-user
support, selects the accelerator, prints the exact server and QEMU argument
vectors, and does not create the output directory.

Each QEMU or rocJITsu capability probe is limited to `--probe-timeout`
seconds (10 by default). SIGTERM or SIGINT sent to the launcher while a probe
is running terminates and reaps that probe before the launcher exits.

The launcher waits at most `--startup-timeout` seconds (60 by default) for the
server to create and explicitly announce its listening socket. It then creates
shared memfd-backed guest RAM and starts:

```text
rocjitsu --config configs/gfx1250_mi455x.json \
  --vfio-socket <output>/vfio-user.sock
```

and connects QEMU's `vfio-user-pci` device to that socket. Shared guest RAM
is required so the VFIO-user server can map DMA windows.

## Results and reruns

The launcher writes:

```text
<output>/guest.log
<output>/server.log
```

Success requires all requested log markers, no rejected marker, QEMU status
zero, and cooperative rocJITsu termination. The launcher sends SIGTERM to the
server after QEMU exits and waits up to five seconds for it to stop cleanly.
After that grace period it sends SIGKILL, reaps the server, and fails the run.
Any server exit while QEMU is still running fails the run, regardless of exit
status. An orderly guest disconnect does not terminate the server; it remains
alive until the launcher observes QEMU exit and requests shutdown.

Each run needs a new output directory. Preserve the logs together with:

- the rocJITsu Git commit;
- the QEMU version;
- the accelerator selected by dry-run;
- kernel, initramfs, AMDGPU package, firmware package, and ROCm package
  versions and SHA-256 digests;
- the exact kernel command line;
- the GEMM dimensions, input definition, reference calculation, and
  acceptance tolerance.

## Troubleshooting

### QEMU does not recognize `vfio-user-pci`

Use a QEMU build with VFIO-user PCI support. This is distinct from the
host-kernel VFIO passthrough devices.

### The server does not report readiness

Inspect `server.log`. Also keep the output path short and ensure that no
stale path already exists. The socket must be directly inside the output
directory. The launcher waits on a pipe inherited by rocJITsu, which reports
readiness after configuration parsing, topology construction, engine startup,
and the final socket bind. Process exit or a closed pipe reports startup
failure immediately; a live server that does not report readiness receives
SIGTERM after `--startup-timeout` seconds and is force-killed if it cannot
complete the bounded shutdown handshake.

### AMDGPU reports a missing firmware file

Copy the named file from the firmware package matching the guest AMDGPU module,
then rebuild the initramfs. Absence of JPEG/VCN hardware only removes media
firmware requirements; it does not remove compute, MES, SDMA, IMU, or
IP-discovery parser inputs.

### AMDGPU fails while discovering multimedia IP

Use a driver containing `4e07da515d1c` or its backport. Do not work around
this by adding a fake JPEG or VCN device to rocJITsu.

### `/dev/kfd` is absent

Check `guest.log` for the first AMDGPU probe failure. Verify the PCI ID,
module/kernel match, firmware inventory, kernel configuration, and the
compute-only IP-block mask before debugging ROCm user space.

### rocBLAS cannot find a kernel

Verify that the guest contains the gfx1250 Tensile metadata and code object
from the same rocBLAS package as `librocblas.so`. Do not substitute files
from another GPU target.

### The run stops making progress

The launcher does not impose a runtime deadline. It waits for QEMU to exit and
fails immediately if rocJITsu exits first. A normal guest disconnect leaves the
server alive until the launcher observes QEMU's exit and requests shutdown, so
this ordering needs no grace period. Use the partial `guest.log` and `server.log`
to distinguish slow progress from a deadlock. CI systems can enforce a job-level
budget without changing launcher behavior. If that budget sends SIGTERM or
SIGINT to the launcher, it terminates and reaps both children and removes the
owned VFIO socket before exiting. SIGKILL cannot provide cooperative cleanup.
