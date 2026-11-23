# gfx1201 counter definitions

ROCm Compute Profiler supplies its own SDK configuration through
`ROCPROFILER_METRICS_PATH`. The SDK does not merge an existing override file
with its installed definitions. This configuration retains the complete
87-definition gfx1201 contract from the SDK before the architecture cleanup.

The restoration adds 63 missing architecture definitions. Existing definitions
for other architectures and the firmware restrictions remain intact.
`FETCH_SIZE` includes 32-, 64-, 128- and 256-byte packets.

## References

- [SDK contract before #9541](https://github.com/ROCm/rocm-systems/blob/e92aa6ee8c01f991c9da08ea386330a6374e0f4a/projects/rocprofiler-sdk/source/share/rocprofiler-sdk/config.yaml).
- [Architecture cleanup #9541](https://github.com/ROCm/rocm-systems/commit/9d29a1da312470ebfb1cc3d862e607d9a1960559).
- [Navi48 event-ID correction #158](https://github.com/ROCm/rocm-systems/commit/5dbb3b2f8083ef21f1a4a86c4adcfe4222d39076).
- [256-byte FETCH_SIZE correction #8022](https://github.com/ROCm/rocm-systems/commit/e5f2ddbdb0662b45b9f1ea9eada90cbb7984023c).

## Additional experimental coverage

Another 16 raw selectors and their 16 sums have no gfx1201 definition in the
historical SDK snapshot. They cover GL1C, GL2C, SQ/SQC and TCP requests.
The feature continues to use the existing experimental gfx1201 profile path.

[The gfx1201 reference implementation](https://github.com/qiangpan2/rocm-systems/blob/76e2aa6b63ec70675f7e6c2066d80e6e7a27cd4b/projects/rocprofiler-compute/src/rocprof_compute_soc/profile_configs/sdk_config.yaml)
and the [soc24 selector catalog](https://github.com/ROCm/rocm-systems/blob/6f441646398273bb16a065a3b3798d95f7a95563/projects/aqlprofile/linux/soc24_enum.h)
provide the available public references. Seven GL1C/GL2C selectors still need
a public chip-specific IP/RDL reference: GL1C_GL2_REQ_READ_64B,
GL1C_REQ_READ_32B, GL1C_REQ_READ_128B, GL1C_REQ_WRITE_32B,
GL2C_READ_32_REQ, GL2C_READ_64_REQ and GL2C_READ_128_REQ.
