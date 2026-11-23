# rocEncode capability mapping

**Status:** Phase 1
**Applies to:** `rocEncGetEncodeCaps()` / `ROCENC_CAPS_*`

Every `ROCENC_CAPS_*` value resolves to exactly one of four sources:

| Source | Meaning |
|---|---|
| **VA** | A VA-API config or surface attribute queried on `VAEntrypointEncSlice`. This is the only source that can establish that a device can encode at all. |
| **KERNEL** | A field of `AMDGPU_INFO_VIDEO_CAPS` (encode sub-query), read through the DRM render node. |
| **0** | Hard zero. There is no VA attribute and no kernel field, so the honest answer is "not supported". |
| **CONST** | A fixed non-zero value that follows from the codec standard or from a documented rocEncode design decision, not from the device. Each one is justified in the Notes column. |

`CONST` is a deliberate fourth category. The alternative was to report 0 for
things like quarter-pel motion compensation, which is mandatory in H.264 and
HEVC — a device that encodes the codec necessarily does it. Reporting 0 there
would be a wrong answer, not a conservative one.

## Rules this table obeys

- **Capability comes from `VAEntrypointEncSlice` or `AMDGPU_INFO_VIDEO_CAPS`.**
  Never from the VCN encode-ring count: MI300 reports `num_enc_rings = 1` and
  has no encoder (`vcn_v4_0_3.c`). Never from a GPU name or a `gfx` target.
- **Mesa silently ignores unsupported misc parameter buffers and returns
  success**, so a VA return code never proves a feature works. Only attribute
  *values* from `vaGetConfigAttributes` are trusted.
- **A codec the device cannot encode answers 0 for every capability**, with
  `ROCENC_SUCCESS`. "No encoder" is an answer, not an error.
- `ROCENC_CAPS_SUPPORTED_RATECONTROL_MODES` is a bitmask of
  `1 << ROCENC_PARAMS_RC_MODE`. NVENC documents only "a bitmask of the values
  in `NV_ENC_PARAMS_RC_MODE`", which is ambiguous because `CONSTQP` is 0;
  rocEncode states the convention explicitly.

## The 56 values

| # | Capability | Source | Resolves to | Notes |
|---|---|---|---|---|
| 1 | `NUM_MAX_BFRAMES` | VA | `VAConfigAttribEncMaxRefFrames`, high 16 bits (L1 count) | A device with no L1 reference slots cannot emit B-frames. VCN 3.0+ reports L1 for H.264. |
| 2 | `SUPPORTED_RATECONTROL_MODES` | VA | `VAConfigAttribRateControl` → `VA_RC_CQP`/`VA_RC_VBR`/`VA_RC_CBR` | Remapped to `1 << ROCENC_PARAMS_RC_MODE`. |
| 3 | `SUPPORT_FIELD_ENCODING` | VA | `VAConfigAttribEncInterlaced` | Reported as 0/1; the "2" (frame and field) case is not distinguishable from this attribute. |
| 4 | `SUPPORT_MONOCHROME` | 0 | — | No VA attribute, no kernel field. |
| 5 | `SUPPORT_FMO` | 0 | — | VCN has no FMO. |
| 6 | `SUPPORT_QPELMV` | CONST | 1 for H.264/HEVC, 0 for AV1 | Quarter-pel MC is mandatory in H.264 and HEVC. AV1 uses eighth-pel, so 0 is correct there. |
| 7 | `SUPPORT_BDIRECT_MODE` | 0 | — | VA exposes no BDirect selection; Mesa chooses internally. |
| 8 | `SUPPORT_CABAC` | CONST | 1 for H.264, 0 otherwise | radeonsi's H.264 encoder emits CABAC; VA has no attribute for entropy coding mode. |
| 9 | `SUPPORT_ADAPTIVE_TRANSFORM` | 0 | — | H.264 8x8 transform exists from VCN 5.0, but nothing in VA or the kernel exposes it, and inferring it from the VCN version is forbidden. |
| 10 | `SUPPORT_STEREO_MVC` | 0 | — | No MVC encode on VCN. |
| 11 | `NUM_MAX_TEMPORAL_LAYERS` | VA | `VAConfigAttribEncRateControlExt`, `max_num_temporal_layers_minus1` + 1 | 0 when the attribute is `VA_ATTRIB_NOT_SUPPORTED`. |
| 12 | `SUPPORT_HIERARCHICAL_PFRAMES` | 0 | — | No VA attribute. |
| 13 | `SUPPORT_HIERARCHICAL_BFRAMES` | 0 | — | No VA attribute. |
| 14 | `LEVEL_MAX` | KERNEL | `drm_amdgpu_info_video_codec_info::max_level` | 0 when the encode sub-query is unavailable. |
| 15 | `LEVEL_MIN` | CONST | `ROCENC_LEVEL_AUTOSELECT` (0) | Neither VA nor the kernel reports a minimum level. |
| 16 | `SEPARATE_COLOUR_PLANE` | 0 | — | Not supported by radeonsi encode. |
| 17 | `WIDTH_MAX` | VA, KERNEL fallback | `VASurfaceAttribMaxWidth`, else `max_width` | VA is authoritative; the kernel value is the fallback when VA does not report the attribute. |
| 18 | `HEIGHT_MAX` | VA, KERNEL fallback | `VASurfaceAttribMaxHeight`, else `max_height` | Same. |
| 19 | `SUPPORT_TEMPORAL_SVC` | VA | `VAConfigAttribEncRateControlExt` > 1 layer | Derived from row 11. |
| 20 | `SUPPORT_DYN_RES_CHANGE` | 0 | — | Depends on `rocEncReconfigureEncoder`, which is Phase 4. |
| 21 | `SUPPORT_DYN_BITRATE_CHANGE` | 0 | — | Same. Mesa accepts a rate-control misc buffer mid-stream and may ignore it, so the VA return code cannot be used as evidence. |
| 22 | `SUPPORT_DYN_FORCE_CONSTQP` | 0 | — | Same. |
| 23 | `SUPPORT_DYN_RCMODE_CHANGE` | 0 | — | Same. |
| 24 | `SUPPORT_SUBFRAME_READBACK` | 0 | — | No slice-granular readback through VA-API. |
| 25 | `SUPPORT_CONSTRAINED_ENCODING` | 0 | — | `VAConfigAttribEncSliceStructure` describes slice partitioning, not slice independence; using it here would overstate the capability. |
| 26 | `SUPPORT_INTRA_REFRESH` | VA | `VAConfigAttribEncIntraRefresh` != 0 | |
| 27 | `SUPPORT_CUSTOM_VBV_BUF_SIZE` | 0 | — | HRD parameters go through a misc buffer that Mesa may ignore silently. Not provable until the encode path exists. |
| 28 | `SUPPORT_DYNAMIC_SLICE_MODE` | VA | `VAConfigAttribEncSliceStructure` != 0 | |
| 29 | `SUPPORT_REF_PIC_INVALIDATION` | 0 | — | No VA mechanism. |
| 30 | `PREPROC_SUPPORT` | 0 | — | rocEncode does no pre-processing. |
| 31 | `ASYNC_ENCODE_SUPPORT` | 0 | — | rocEncode is synchronous-only by design, as NVENC is on Linux. `enableEncodeAsync` must be 0. |
| 32 | `MB_NUM_MAX` | KERNEL, VA fallback | `max_pixels_per_frame` / 256, else `WIDTH_MAX` × `HEIGHT_MAX` / 256 | 256 = 16x16 macroblock. For HEVC/AV1 the unit is nominal. |
| 33 | `MB_PER_SEC_MAX` | 0 | — | Neither VA nor the kernel reports a throughput figure. |
| 34 | `SUPPORT_YUV444_ENCODE` | VA | `VAConfigAttribRTFormat` & `VA_RT_FORMAT_YUV444` | |
| 35 | `SUPPORT_LOSSLESS_ENCODE` | 0 | — | No VA attribute. |
| 36 | `SUPPORT_SAO` | 0 | — | Mesa's HEVC writer overrides the SAO flag from its own state, so an application-visible capability would be misleading. |
| 37 | `SUPPORT_MEONLY_MODE` | 0 | — | No motion-estimation-only mode on VCN. Permanent. |
| 38 | `SUPPORT_LOOKAHEAD` | 0 | — | No VA mechanism; `rocEncLookaheadPicture` is permanently unsupported. |
| 39 | `SUPPORT_TEMPORAL_AQ` | 0 | — | No VA attribute. |
| 40 | `SUPPORT_10BIT_ENCODE` | VA | `VAConfigAttribRTFormat` & `VA_RT_FORMAT_YUV420_10` | Also reflected in `rocEncGetInputFormats` as `ROCENC_BUFFER_FORMAT_YUV420_10BIT`. |
| 41 | `NUM_MAX_LTR_FRAMES` | 0 | — | H.264 LTR pass-through is an open question in the scoping plan (§10). |
| 42 | `SUPPORT_WEIGHTED_PREDICTION` | 0 | — | Mesa's PPS writer forces `weighted_bipred_idc`, so the application cannot control it. |
| 43 | `DYNAMIC_QUERY_ENCODER_CAPACITY` | CONST | 100 | There is no virtualized-capacity query on Linux. NVENC also returns 100 on bare metal, so a port sees the same value. |
| 44 | `SUPPORT_BFRAME_REF_MODE` | 0 | — | B-as-reference is not selectable through VA-API. Distinct from row 1, which is the B-frame count. |
| 45 | `SUPPORT_EMPHASIS_LEVEL_MAP` | 0 | — | `VAConfigAttribEncROI` exists but has different semantics; mapping it here would misreport. |
| 46 | `WIDTH_MIN` | VA | `VASurfaceAttribMinWidth` | |
| 47 | `HEIGHT_MIN` | VA | `VASurfaceAttribMinHeight` | |
| 48 | `SUPPORT_MULTIPLE_REF_FRAMES` | VA | `VAConfigAttribEncMaxRefFrames`, low 16 bits (L0 count) > 1 | |
| 49 | `SUPPORT_ALPHA_LAYER_ENCODING` | 0 | — | No HEVC alpha layer on VCN. |
| 50 | `NUM_ENCODER_ENGINES` | CONST | 1 when the codec is supported, else 0 | **Deliberately not** `amdgpu_query_hw_ip_count(AMDGPU_HW_IP_VCN_ENC)`. That count is 1 on MI300, which has no encoder, so it answers "how many rings", never "can this encode". A true multi-engine count needs a source that does not conflate the two; until one exists this reports presence, not population. See the open question below. |
| 51 | `SINGLE_SLICE_INTRA_REFRESH` | 0 | — | No VA attribute. |
| 52 | `DISABLE_ENC_STATE_ADVANCE` | 0 | — | No VA mechanism. |
| 53 | `OUTPUT_RECON_SURFACE` | 0 | — | Reconstructed output is not exposed by radeonsi encode. |
| 54 | `OUTPUT_BLOCK_STATS` | 0 | — | No per-block stats from VA-API. |
| 55 | `OUTPUT_ROW_STATS` | 0 | — | No per-row stats from VA-API. |
| 56 | `EXPOSED_COUNT` | — | Not queryable | Reserved sentinel. `rocEncGetEncodeCaps` returns `ROCENC_ERR_INVALID_PARAM`, which the negative test pins down. |

## Counts

- VA config/surface attributes: 14 (rows 1, 2, 3, 11, 17, 18, 19, 26, 28, 34, 40, 46, 47, 48)
- `AMDGPU_INFO_VIDEO_CAPS`: 2 primary (rows 14, 32), plus fallback use in rows 17 and 18
- Hard 0: 36
- Fixed non-zero constants: 4 (rows 6, 8, 15, 43) plus row 50
- Not queryable: 1 (row 56)

## Open questions this table raises

1. **`NUM_ENCODER_ENGINES` has no trustworthy source.** The requirement forbids
   the ring count, and nothing else reports VCN encode instance population.
   Reporting 1-or-0 is honest but lossy on parts with two encode-capable VCN
   instances. Needs either a Mesa-side query or a kernel field that
   distinguishes encode-capable instances from encode rings.
2. **`SUPPORT_FIELD_ENCODING` cannot express NVENC's value 2.**
   `VAConfigAttribEncInterlaced` is a flag set, not a tri-state. An application
   that branches on 2 will take the "field only" path.
3. **`MB_NUM_MAX` for HEVC and AV1** divides a pixel budget by 256 even though
   the coding unit is a CTB or super block. NVENC does the same thing for the
   same reason; it is documented here so nobody mistakes it for a bug.
4. **The `AMDGPU_INFO_VIDEO_CAPS` encode sub-query is issued by direct ioctl.**
   libdrm's `amdgpu_query_info()` zeroes the request union, which pins the
   sub-query to `DECODE`, so `drmCommandWrite(DRM_AMDGPU_INFO)` is used
   directly with `video_caps.type = AMDGPU_INFO_VIDEO_CAPS_ENCODE`. This has
   not yet been exercised against a real kernel.
