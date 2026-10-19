<!-- Copyright (c) 2026 Advanced Micro Devices, Inc. -->
<!-- SPDX-License-Identifier: MIT -->

# libamdf

libamdf implements the AMDF C API in Rust over [rocddi](../rocddi/README.md).
`amdf_query_api` negotiates core ABI v5; the GPU extension supports v1. The
XDNA extension is not implemented. The [AMDF headers](../../api-headers/include/amdf/)
define the public contract, including arguments, ownership, and error behavior.
The Rust modules are private implementation details.

## Runtime model

An instance owns a rocddi session. Creating an instance, enumerating endpoints,
and opening an endpoint do not activate a GPU. Device creation activates an
endpoint for execution. CPU memory services work without an available GPU.

The instance's `native_lifetime` selects a process-owned or instance-owned
driver context. On Linux KFD, PROCESS can reuse the primary connection;
INSTANCE GPU activation requires a secondary context supported by KFD UAPI
1.19 or newer.

The CMake build stages `libamdf.a` and a `libamdf.so` alias of the shared image
that also exports HSA. Each ABI owns a separate rocddi session and public
handles; process-lifetime KFD contexts share rocddi's process-local backend
state. A separately linked static archive has its own copy of that state.

## Current support

- Host allocations and registration of caller-owned memory, with or without
  a GPU. GPU SYSTEM and LOCAL profiles are available when the selected device
  supports them. SYSTEM can target multiple devices at one shared GPU address;
  LOCAL peer access requires a supported route.
- DMA-BUF export from supported GPU SYSTEM profiles and conditional same-device
  SYSTEM import. The selected profile must advertise each operation. Foreign
  API, distinct-GPU, and LOCAL imports are not supported.
- PM4, AQL, and SDMA user queues where the discovered family advertises them.
  Selected Linux x86-64 GFX12 targets also support PM4 and SDMA kernel queues.
  Callers build packets and manage publication and completion; the provider
  does not pool queues.

A populated API table describes compiled entry points, not support for every
request. Query endpoint capabilities, memory profiles, and queue families
before using a service. Dynamic AQL scratch growth and live reset or
unplug recovery are not supported.

## Ownership

Endpoints borrow their instance, devices borrow their endpoint, and queues
borrow their device. Memory borrows its scope and requested devices; host
mappings and queue producer mappings borrow their parent objects. Destroy
children before parents. Device destruction does not track dependent memory,
so callers must release that memory first.

Destruction does not wait for device work. Keep memory and submitted command
storage alive until the work has retired; queue consumption alone does not
prove completion. The [headers](../../api-headers/include/amdf/) specify which
errors preserve a handle and which operations consume it.

## Build and verification

See the [workspace build instructions](../../README.md) for CMake targets,
shared-library aliases, and Cargo commands. The implementation currently
builds for Linux x86-64 and AArch64 with a KFD/DRM GPU backend. GPU execution
was previously observed on one x86-64 GFX1201 host. CTest runs Rust tests and
native C ABI checks without a GPU. GPU execution requires separate
qualification on a supported host.
