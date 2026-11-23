# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project status: pre-implementation

rocencode contains **no source code yet**. The only file is `rocEncode-development-plan.md`
(rev 2.2), a scoping document for a hardware video *encode* library for ROCm. There is no
CMake, no build, no tests. The directory is untracked in `rocm-systems`; work happens on
branch `users/rrawther/rocencode_initial`.

Until M1 lands, treat the plan as the spec and the deliverable: most tasks here are edits to
that document (revisions, resolving open questions, re-checking claims against source), not code.
When adding code, follow the layering and naming in §4–§5 of the plan rather than inventing a
structure, and update the plan's revision history (§11) when a decision changes.

## What rocEncode is

A C API over VA-API (AMD's Mesa/radeonsi stack) for H.264 and HEVC encode on VCN engines,
with a zero-copy path for frames already in HIP device memory. It is the encode sibling of
`projects/rocdecode` and deliberately mirrors both rocDecode's internal layering and NVENC's
API shape. Distribution is TheRock only; no Python bindings; elementary streams only (no muxing).

**Support floor: VCN 3.0+, Radeon RDNA2 and later. Instinct MI300-class and later cannot
encode at all** — the silicon is decode-only. This is the headline constraint and belongs in
any external-facing text.

## Facts that are easy to get wrong

These are the traps the plan was written to prevent; re-deriving them costs hours.

- **Never probe encode capability from the VCN encode ring count.** MI300 reports
  `num_enc_rings = 1` while having no encoder (`vcn_v4_0_3.c`). Capability comes from
  `VAEntrypointEncSlice` or `AMDGPU_INFO_VIDEO_CAPS`, never from `AMDGPU_HW_IP_VCN_ENC`,
  and never from a GPU-name or `gfx`-target lookup table.
- **Navi 24 (VCN 3.0.33) has no encoder** despite being RDNA2. It and MI300 are the two
  negative-test targets.
- **Mesa owns the parameter sets.** Since Mesa 24.3 radeonsi emits SPS/PPS/VPS only if the
  application supplies packed headers — and it *parses* those headers into driver state, then
  regenerates its own, overriding some fields. So supplied bytes are state transfer, not
  output. Bit-exact tests of the writer against a reference encoder test something the
  hardware path never produces (§8).
- **Mesa silently ignores unsupported misc parameter buffers and returns success.** A return
  code cannot tell you whether a feature works; gate everything on `vaGetConfigAttributes`.
- **There is no device-side HIP→VA ordering primitive.** libva exposes no in/out fence on
  encode submission, and KFD writes don't feed DRM implicit sync. M1 ships
  `hipStreamSynchronize`; M4 ships a host-mediated event-plus-worker pipeline. The strategy
  stays out of the ABI — that is why `RocencEncodePicParams` carries `input_stream`.
- **Surfaces must be forced linear at creation** via `VASurfaceAttribDRMFormatModifiers`
  with `DRM_FORMAT_MOD_LINEAR`, or the pitch arithmetic on the HIP side yields garbage.
- **A silent zero-copy fallback is a bug, not a graceful degradation.** Every staged-copy
  fallback must be reported through `RocencInputSurface`, and the negative tests exist to
  catch regressions that quietly halve throughput.

## Planned architecture (plan §4.1)

```
Application → rocencode_api.cpp (C ABI) → EncHandle → RocEncoder ──→ HIP
                                                        │  ╲
                                                        │   BitstreamWriter (pure computation)
                                                        ↓
                                              VaapiVideoEncoder ──→ VA-API/Mesa
                                                        ↓
                                                    VaContext ──→ /dev/dri/renderD<N>
```

The invariant worth defending in review: **exactly one layer touches HIP (`RocEncoder`) and
exactly one touches VA-API (`VaapiVideoEncoder`)**; they communicate through a plain descriptor
struct. `BitstreamWriter` touches neither.

`VaContext` is ported from rocDecode (`src/rocdecode/vaapi/`) — its HIP-ordinal-to-render-node
mapping by GPU UUID is the highest-value reuse. Port it with one substitution
(`VAEntrypointEncSlice` for `VAEntrypointVLD`) and one deliberate non-substitution (do **not**
turn `amdgpu_query_hw_ip_count(AMDGPU_HW_IP_VCN_DEC)` into the `_ENC` variant).

The zero-copy bridge is rocDecode's VA→HIP path run in reverse: `vaCreateSurfaces` →
`vaExportSurfaceHandle` (DRM_PRIME_2, **writable** — rocDecode exports read-only) →
`hipImportExternalMemory` → `hipExternalMemoryGetMappedBuffer`, once per surface slot, never
per frame. radeonsi exports NV12 encode surfaces as a single object; guard `num_objects == 1`
and stage otherwise.

## Naming conventions (plan §5)

Mirrors rocDecode, including where rocDecode is inconsistent:

- Functions `rocEncCreateEncoder`; calling-convention macro `ROCENCAPI`.
- Structs `Rocenc…` PascalCase, members snake_case.
- Status codes screaming case `ROCENC_SUCCESS`; domain enums use a camelCase type prefix
  per value (`rocEncVideoCodec_H264`).
- Caps structs carry their own query inputs (device_id, codec, chroma, bit depth) and the
  call fills the outputs — the `rocDecGetDecoderCaps` pattern.

Code conventions follow rocDecode: C++17, `amdclang++`, `-Wall`, `#pragma once`, MIT header
block on every source and CMake file, formatting enforced by the monorepo root
`.pre-commit-config.yaml` (clang-format, gersemi, black).

## Milestones

M0 feasibility spikes (HIP→VA write path, synchronisation, bitstream-writer field map,
capability probe) gate everything else — two of the four can invalidate the architecture.
Then M1 minimum viable H.264, M2 H.264 production, M3 HEVC, M4 performance, M5 release.
Roughly 28–38 weeks with two engineers and an architect. See plan §7.

## Open questions

Plan §10 tracks them in three buckets (Resolved in rev 2.2 / Deferred / Still open). Before
asserting anything about writable DMA-BUF export, input-surface alignment, the HIP→DMA-BUF
route, libva fences, H.264 LTR pass-through, or Mesa 24.1/24.2 packed-header behaviour, check
that list — those are explicitly *unverified*, and the plan marks unverified claims rather
than stating them. Preserve that discipline when editing: rev 2 found several rev-1 claims
that could not be traced to a source.

Primary sources (Mesa, Linux amdgpu, rocDecode files) are listed in Appendix A; verify against
source, not documentation, when the two could disagree.
