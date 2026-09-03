# Hermetic vfio-user guest

This directory contains the guest-side init used by
`tools/vfio_guest_image.py`. Prepare a staging root from explicitly selected
driver and ROCm artifacts, then use `tools/vfio_guest_lock.py` to snapshot every
regular file and its hash. The image builder consumes that reviewed JSON lock;
it never discovers the running host's kernel, modules, firmware, or ROCm
installation.

The lock has this shape:

```json
{
  "schema_version": 2,
  "guest": {
    "distribution": "<distribution and release>",
    "kernel_release": "<uname release embedded in the modules>",
    "role": "m2-workload",
    "policy": {"ip_block_mask": "0x3f"},
    "kernel": {
      "source": "<path to vmlinuz>",
      "sha256": "<64 lowercase hexadecimal digits>"
    }
  },
  "amdgpu": {
    "repository": "https://github.com/ROCm/amdgpu.git",
    "branch": "roc-7.1.x",
    "build_id": "<module package/build identity>",
    "commit": "01cee31f91e55dc913933217d10341360f4a6f9e"
  },
  "rocm": {
    "packages": [
      {
        "name": "rocblas",
        "version": "<full Debian version including epoch>",
        "architecture": "amd64",
        "filename": "<repository artifact filename>",
        "sha256": "<computed artifact SHA-256>",
        "source": "<path to the reviewed .deb>"
      }
    ]
  },
  "build_tools": {
    "busybox": "<pinned version>",
    "python": "<pinned version>"
  },
  "files": [
    {
      "source": "init",
      "destination": "init",
      "mode": "0755",
      "sha256": "<sha256 of init>"
    },
    {
      "source": "<path to a static busybox>",
      "destination": "bin/busybox",
      "mode": "0755",
      "sha256": "<sha256 of busybox>"
    }
  ]
}
```

The `amdgpu` record accepts exactly one provenance form. Source builds use the
`branch` and 40-digit `commit` fields shown above. A release or nightly DKMS
artifact is recorded without inventing a source commit:

```json
{
  "amdgpu": {
    "repository": "https://repo.radeon.com/amdgpu/<channel>/ubuntu",
    "compatibility": {
      "no_vcn_discovery_guard": true,
      "zero_instance_partition_guard": true
    },
    "module": {
      "destination": "modules/amdgpu.ko",
      "version": "<modinfo version>",
      "srcversion": "<modinfo srcversion>",
      "vermagic": "<modinfo vermagic>",
      "sha256": "<computed module SHA-256>"
    },
    "build_attestation": {
      "source": "<path to reviewed JSON attestation>",
      "sha256": "<attestation SHA-256>",
      "record": "<embedded attestation object>"
    },
    "package": {
      "name": "amdgpu-dkms",
      "version": "<full Debian version including epoch>",
      "architecture": "all",
      "filename": "<repository artifact filename>",
      "sha256": "<computed artifact SHA-256>",
      "source": "<path to the reviewed .deb>"
    }
  }
}
```

Nightly is a source channel, not a floating input. Pass the actual `.deb` to
`vfio_guest_lock.py`; the tool extracts its package metadata, preserves a
Debian epoch, computes its digest, verifies the compute-only no-VCN discovery
and zero-instance partition fixes, and binds it to the staged module's version,
`srcversion`, `vermagic`, and digest. The image builder repeats those checks.
For M2, `--amdgpu-build-attestation` is also mandatory. Its JSON record binds
the package digest, kernel release and image digest, module digest and all three
`modinfo` identities, and nonempty build command, input, and toolchain records.
Every recorded build input and toolchain binary has an explicit path and digest
that are rechecked while creating both the lock and image.
This is an explicit reproducibility attestation, not a substitute for a future
mechanical rebuild gate.
Assertion-only package names, versions, and hashes are not accepted.
There is no hard-coded driver version: a newer release or nightly is acceptable
when the exact pinned artifact contains those guards and matches the staged
module.

Every module, library, executable, configuration file, and workload is another
explicit `files` entry. The load order is an ordinary locked input installed as
`modules/load-order`; the builder does not call `modprobe` or inspect the host.
The lock generator refuses symlinks, special files, and undeclared AMD firmware
in the staging root.

The `m2-workload` role additionally requires a generated SGEMM inventory. The
inventory verifies the pinned package artifacts, follows the lazy Tensile
metadata to the FP32 no-transpose library, resolves the shipping fallback's
nearest candidates for the exact 128 x 128 x 128 non-batched request, proves
that they name one gfx1250 kernel, and records that kernel's resources and
instruction set. It also requires the source, compiled workload, direct code
object, and both metadata files to be present at their declared guest paths with
exact hashes. The inventory also binds the complete repository-owned `init` by
digest, so reordered, unreachable, or later-overridden environment assignments
are rejected rather than inferred from matching shell text. Every required
path below `opt/rocm` is resolved from the payload of one of the pinned Debian
artifacts; package symlinks are followed within the archive namespace and the
resolved bytes must match the staged regular file.

An AMD firmware destination is accepted only when its entry declares:

```json
{
  "kind": "generated-fixture",
  "generator": "<repository-owned generator>",
  "generator_revision": "<Git revision>"
}
```

The source must not reside under either host AMD firmware directory. The
builder creates `firmware-SHA256SUMS`, scans the staged root and packed
initramfs with `vfio_guest_provenance.py`, and publishes hashes in
`manifest.json`.

The compute-capable M2 guest needs the stock driver's firmware parsers to
complete even though rocjitsu does not run a firmware processor. Generate the
small, deterministic, non-AMD compatibility blobs with:

```console
python3 tools/vfio_guest_firmware.py --output tmp/m2-firmware
```

Stage those five files under `lib/firmware/amdgpu/`, and generate
`lib/firmware/amdgpu/ip_discovery.bin` with a pinned `rj-ip-discovery gfx1250`
binary. M2 accepts exactly those six paths. Pass the binary with
`--ip-discovery-generator`; the lock generator and image builder independently
regenerate and byte-compare every fixture, and record both generator hashes and
argument vectors. Their payloads are parser inputs, not executable AMD
microcode.

Build and inventory the pinned workload before locking the staging root:

```console
/opt/rocm/core-7.14/bin/hipcc -std=c++20 \
  -I/opt/rocm/core-7.14/include \
  tools/vfio_guest/rocblas_sgemm.cpp \
  -L/opt/rocm/core-7.14/lib \
  -Wl,-rpath,/opt/rocm/core-7.14/lib -lrocblas \
  -o tmp/m2-rocblas-sgemm

python3 tools/vfio_guest_sgemm_inventory.py \
  --contract tools/vfio_guest/rocblas_sgemm_contract.json \
  --workload-binary tmp/m2-rocblas-sgemm \
  --code-object tmp/m2-gfx1250-package-root/opt/rocm/core-7.14/lib/rocblas/library/gfx1250/TensileLibrary_Type_SS_Contraction_l_Ailk_Bljk_Cijk_Dijk_fallback_gfx1250.hsaco \
  --metadata tmp/m2-gfx1250-package-root/opt/rocm/core-7.14/lib/rocblas/library/gfx1250/TensileLibrary_Type_SS_Contraction_l_Ailk_Bljk_Cijk_Dijk_fallback_gfx1250.dat \
  --lazy-metadata tmp/m2-gfx1250-package-root/opt/rocm/core-7.14/lib/rocblas/library/gfx1250/TensileLibrary_lazy_gfx1250.dat \
  --llvm-readobj /opt/rocm/core-7.14/lib/llvm/bin/llvm-readobj \
  --llvm-nm /opt/rocm/core-7.14/lib/llvm/bin/llvm-nm \
  --llvm-objdump /opt/rocm/core-7.14/lib/llvm/bin/llvm-objdump \
  --output tmp/m2-sgemm-inventory.json
```

Stage the files named by `required_guest_files`, the complete contents of the
pinned gfx1250 BLAS package needed by the rocBLAS/Tensile path, and the pinned
runtime libraries and loader dependencies. Do not substitute or rename payloads
from another target. Then pass
`--role m2-workload --workload-inventory tmp/m2-sgemm-inventory.json`, the
actual `--amdgpu-package <path.deb>`, and one `--rocm-package <path.deb>` for
every package in the workload contract, plus `--amdgpu-build-attestation` and
`--ip-discovery-generator`, to `vfio_guest_lock.py`. The workload
calls exactly `rocblas_sgemm` with FP32
column-major 128-cube inputs, no transposes, alpha 1, beta 0, deterministic
multiples of 1/16, and a long-double CPU reference. Its error threshold is fixed
in the contract before any emulated GPU result is observed.

The gfx1250 compute profile advertises zero JPEG and VCN instances. M2 requires
an amdgpu package whose discovery path accepts an absent VCN IP and whose
partition code accepts zero media resources. Rocjitsu does not provide a JPEG
or VCN block model, startup-ring shim, media queue, media register surface, or
firmware fallback.

The guest explicitly sets `ROCBLAS_USE_HIPBLASLT=0`, selecting the shipping
rocBLAS Tensile backend pinned by the workload contract. The public API remains
`rocblas_sgemm`, and the selected gfx1250 Tensile metadata and code object remain
fully inventoried. This avoids hipBLASLt's separate eager device allocation,
which is unrelated to the SGEMM under test and does not fit within the guest's
deliberately bounded 256 MiB allocation heap.

Build into a new directory under `tmp/`:

```console
python3 tools/vfio_guest_image.py \
  --lock tmp/vfio-guest-lock.json \
  --output tmp/vfio-guest-image
```

Launch through the tracked runner. M2 locks the compute-only IP-block mask to
`0x3f` in both the image lock and manifest, so a run cannot silently enable PSP,
SMU, display, or media blocks:

```console
python3 tools/vfio_guest_run.py \
  --image tmp/vfio-guest-image \
  --qemu tmp/qemu-install/bin/qemu-system-x86_64 \
  --rocjitsu tmp/build-vfu/tools/rocjitsu/rocjitsu \
  --config configs/gfx1250_mi455x.json \
  --memory 4G \
  --expect-log '<named bounded frontier>' \
  --output tmp/vfio-guest-run
```

The runner uses KVM only when both QEMU and `/dev/kvm` support it, otherwise it
uses TCG. Both paths explicitly expose the hypervisor CPUID bit, use shared
memfd-backed guest RAM, rerun the firmware/ROM provenance gate, and save the
exact argument vectors plus input hashes in `run-manifest.json`.
M2 workload images default to 4 GiB of guest RAM because their locked initramfs
contains the ROCm runtime and rocBLAS/Tensile payload. Smaller driver-discovery
images retain the 2 GiB default. An explicit `--memory` value overrides either
role-specific default.
For an image whose locked role is `m2-workload`, the runner also selects
`rocjitsu.workload=sgemm`; callers cannot override that workload through
`--append`. A caller-supplied `--ip-block-mask` is accepted only when it is
numerically identical to the locked M2 value. The runner also rejects any VCN
or JPEG evidence in the guest log as an intrinsic M2 assertion; callers do not
need to remember an optional `--reject-log` argument for the no-media guarantee.
Discovery-only images continue to require an explicit mask.

The launch policy also pins the unmodified driver's supported
`amdgpu.vramlimit=256` parameter. The emulated product retains its configured
physical-memory identity, while the guest allocation heap and the kernel's HMM
page metadata stay bounded independently of the hermetic guest's RAM size.
