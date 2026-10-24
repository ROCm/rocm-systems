## Introduction
Managing hardware devices (GPUs, CPUs, NICs) in data centers often requires multiple tools, each with its own device identification method. This inconsistency complicates device correlation and management, especially across different vendors. The **Component Unified ID (CUID)** solves this by providing a deterministic, unique identifier for each device type, enabling consistent referencing and easier integration.

### What is CUID?

A primary CUID combines device-specific data into a 122-bit payload `P` of 16
octets. Payload bit `b` is bit `b % 8` of `P[b // 8]`; multi-octet fields are
little-endian. This release does not produce primary CUIDs: every CUID it
returns is a temporary CUID (below), in which only bit 117, the version and the
variant are meaningful and the other payload bits are hash output.

| Bits | Name | Comments |
|------|------|----------|
| 0:63 | Serial | Stage 2. The CUID specification lists the sources, and the firmware UUIDs (such as the SMBIOS system UUID) adopted whole as a primary. A serial that reads as zero is absent, and the component keeps its temporary CUID |
| 64:71 | UnitID (part 1) | UnitID bits 0:7 |
| 72:79 | RevisionID | PCI revision; for a CPU the CPUID stepping |
| 80:95 | DeviceID | PCI device ID; for a CPU `(family << 8) \| model` |
| 96:111 | VendorID | PCI vendor ID; `0x1022` or `0x8086` for a CPU |
| 112:116 | UnitID (part 2) | UnitID bits 8:12 |
| 117 | Temporary marker | 0 in every primary and derived CUID; set only in a temporary CUID; not part of UnitID |
| 118:121 | Component Type | See `amdcuid_device_type_t` in amd_cuid.h, which uses the on-wire values directly |

Component Type uses these values: `0x0` Platform, `0x1` CPU, `0x2` GPU, `0x3` NIC, `0x4` NPU, `0x5` Storage, `0x6` Memory, `0x7` generic PCIe device, `0x8` generic component, `0x9` rack tray, `0xA` rack, `0xB`–`0xE` reserved, `0xF` other. The field spans an octet boundary: its low two bits are the top two bits of payload octet 14, and its high two bits are the low two bits of payload octet 15. A producer that drops payload bits 120:121 therefore renders the type modulo 4 and collides an NPU with a Platform.

`amdcuid_device_type_t` names all sixteen of those values, not only the five this library can discover for itself. A Component Type also arrives from identifiers this library did not produce, such as one another producer emitted, so a Storage or Rack component has to decode to a name rather than to an unnameable integer. `0xB`–`0xE` are reserved and have **no enumerator**. Because the valid range is therefore not contiguous, there is no `AMDCUID_DEVICE_TYPE_LAST` sentinel. Test a value with the header-only helper instead:

```c
if (!amdcuid_device_type_is_valid(type)) { /* reserved, or not a Component Type at all */ }
```

`amdcuid_device_type_is_valid()` is true for `0x0`–`0xA` and `0xF`, and false for the reserved `0xB`–`0xE`, for `AMDCUID_DEVICE_TYPE_NONE` (`0xFF`, deliberately outside what a 4-bit field can hold), and for anything else. The library version is 2.2.0 (`AMDCUID_LIB_VERSION_MAJOR`, `_MINOR`, `_PATCH`); code that loops over Component Types uses the helper.

The payload is framed as an RFC 9562 UUIDv8. `U` is the UUID in printed octet order:

```text
U[0..5] = P[0..5]
U[6]    = 0x80 | (P[6] >> 4)
U[7]    = ((P[6] & 0x0f) << 4) | (P[7] >> 4)
U[8]    = 0x80 | ((P[7] & 0x0f) << 2) | (P[8] >> 6)
U[i]    = ((P[i-1] & 0x3f) << 2) | (P[i] >> 6)        for i = 9..14
U[15]   = ((P[14] & 0x3f) << 2) | (P[15] & 0x03)
```

The version (`8`) and variant (`10b`) are **inserted**, not written over payload bits: they displace the payload after them. The framing keeps payload bits 0 through 121 and drops bits 122 through 127, which are always zero, so the payload recovers from the rendered UUID exactly. Payload octet 0 appears at the **front** of the printed string: payload `1032547698badcfe0000a37302108000` renders as `10325476-98ba-8dcf-b800-028dcc084200`.

A **derived CUID** is derived by hashing the primary CUID using HMAC-SHA2-256 (keyed hash) with a 256-bit key. The key is the HMAC key and the 16 packed primary payload octets are the message. The digest folds into the derived payload as `digest[0:63]` → bits 0:63, reserved zero at 64:71, `digest[64:108]` (45 bits) → bits 72:116, zero at bit 117, reserved zero at 118:121.

A **temporary CUID** names a component without a node key, a driver or a serial number. It is not derived from a primary, and there is no temporary primary. It is `HMAC-SHA256(key = S, message = the 16 octets of the CUID application ID deeda863-4023-5bac-865a-9923e61370d7)`, truncated to 16 octets, with the UUID version set to 8, the variant to `10b`, and payload bit 117 set. `S` is 32 octets: a format, the host's machine-id, the component's PCIe routing ID, revision, device and vendor IDs, Component Type, UnitID, and the high 16 bits of its PCIe segment (for a CPU package its package ID and CPUID values; for the platform nothing else). Every input is world-readable, so every caller gets the same value. Bit 117 is the only marker of a temporary CUID; amd-smi reports it as `temporary` (`TEMPORARY: True`).

When an input cannot be read, for example because the host has no machine-id, the library **returns an error** rather than a placeholder. Callers must handle the error rather than assume an identifier always comes back. Derivation of a permanent CUID requires a 256-bit HMAC key (FIPS 198-1). Anyone holding the key can confirm a guessed serial number, so the key carries the same handling requirements as the serial it protects: random, unique per deployment, and readable only by root.

### Temporary CUIDs

This library holds no node key and reads no CUID from a driver, so every CUID
it returns is a temporary CUID, for root and every other caller alike. It
covers:

- AMD GPUs (vendor `0x1002`, class `0x03xx` or `0x12xx`, bound to amdgpu or
  gim) and their partitions
- AMD NPUs bound to amdxdna
- network-class PCI functions of any vendor (class `0x02xx`, `0x0D20`,
  `0x0D21` or `0x0D80`), with or without a network interface
- AMD and Intel CPU packages
- the platform

No other PCI function has a CUID.

A GPU split into `n >= 2` compute partitions has one component per partition.
Partition `k` has UnitID `n*(n-1)/2 + 1 + k` (DPX 2–3, QPX 7–10, CPX on 8 XCCs
29–36); the whole GPU keeps UnitID 0, and a GPU in SPX has no separate
partition. The UnitID comes from the partition's `cuid_unit_id` where amdgpu
publishes one, and otherwise from the KFD topology, so a stock kernel gives
the same values. A partition whose UnitID cannot be determined, such as one
in a container whose device cgroup hides some of the GPU's KFD nodes, has no
CUID.

Temporary CUIDs depend on the host's machine-id, so they are node-local.
See [Temporary CUIDs](docs/conceptual/what_is_cuid.rst) for the construction
and its effect on containers.

### Using the library

`libamdcuid_static.a` is a static library with one header, `amd_cuid.h`
(installed under `include/amdcuid`), and a CMake package that puts that
directory on the include path:

```cmake
find_package(amdcuid CONFIG REQUIRED)
target_link_libraries(my_tool PRIVATE amdcuid::amdcuid)
```

A program that links it carries no runtime dependency on CUID. It discovers
hardware, constructs each component's CUID, and answers queries by handle;
[`example/main.cc`](example/main.cc) walks through the API. amd-smi links the
same library and is the administrator's interface to CUID:
`amd-smi node --cuid` lists every component's CUID.

### Build and Install Instructions

To build and install the library:

```sh
mkdir build
cd build
# Set CMAKE_INSTALL_PREFIX as needed (default is typically /opt/rocm/core)
cmake -DCMAKE_INSTALL_PREFIX=/opt/rocm ..
make
# Use sudo if installing to a system prefix like /opt
sudo make install
```

If users need to uninstall the CUID project, they can do so with the following instructions:

```sh
cd build
sudo make uninstall
```

A source install of 2.0, which shipped the `amdcuid` daemon, leaves the
daemon, `amdcuid_tool`, its configuration and its systemd unit behind when a
later version is installed over it, because this release's install manifest does not list
them. Run `sudo make uninstall` in the old build tree first.

Documentation can be built using the following instructions:

```sh
pip install -r docs/sphinx/requirements.txt
sphinx-build docs docs/_build/html
```

### Supported Operating Systems

#### Minimum Requirements

| Component | Minimum Version | Notes |
|-----------|-----------------|-------|
| **CMake** | 3.20 | Build system requirement |
| **GCC** | 7.0 | C++17 standard required |
| **Kernel** | 2.6+ | Standard sysfs interfaces |
| **Architecture** | x86_64 for CPU CUIDs | Without CPUID, CPU packages have no CUID; other components are unaffected |

#### Linux Distribution Compatibility

| Distribution | Version | Kernel | GLibc | Status |
|--------------|---------|--------|-------|--------|
| **Ubuntu** | 24.04.4 | 6.8 (GA), 6.17(HWE) | 2.39 | ✅ Supported |
| | 22.04.5 | 5.15(GA), 6.8(HWE) | 2.35 | ✅ Supported |
| **RHEL** | 10.1 | 6.12.0-124 | 2.39 | ✅ Supported |
| | 10.0 | 6.12.0-55 | 2.39 | ✅ Supported |
| | 9.7 | 5.14.0-611 | 2.34 | ✅ Supported |
| | 9.6 | 5.14.0-570 | 2.34 | ✅ Supported |
| | 9.4 | 5.14.0-427 | 2.34 | ✅ Supported |
| | 8.10 | 4.18.0-553 | 2.28 | ✅ Supported |
| **SLES** | 15 SP7 | 6.4.0-150700.51 | 2.38 | ✅ Supported |
| **Debian** | 13 | 6.12 | 2.36 | ✅ Supported |
| | 12 | 6.1.0 | 2.36 | ✅ Supported |
| **Rocky Linux** | 9 | 5.14.0-570 | 2.34 | ✅ Supported |
| **Oracle Linux** | 10 | 6.12.0 (UEK) | 2.39 | ✅ Supported |
| | 9 | 5.15.0 (UEK) | 2.34 | ✅ Supported |
| | 8 | 5.15.0 (UEK) | 2.28 | ✅ Supported |

#### Notes

- No third-party crypto dependency: SHA-256 and HMAC-SHA-256 are built from source out of the shared `shared/sha256` component, whose objects are absorbed into `libamdcuid_static.a`, so the archive defines the digest symbols rather than importing them, and the library links no TLS stack
- Linux only: this release does not build on Windows
- CUIDs need no privilege. `AMDCUID_QUERY_PRIMARY_CUID` and `AMDCUID_QUERY_HARDWARE_FINGERPRINT` return `AMDCUID_STATUS_UNSUPPORTED` to every caller, because a temporary CUID has no primary and discloses no serial. Root is needed only for `amdcuid_generate_hash_key()`

## License

Copyright (c) Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
