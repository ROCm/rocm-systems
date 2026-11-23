# gfx1201 cache selector catalog

The experimental gfx1201 SDK contract and the unqualified soc24 cache
enumerations disagree on 11 names. The new C-compatible catalog gives these
selectors explicit `PERF_SEL_GFX1201` names. Existing generic names keep their
values, so their current callers remain compatible.

This is a declaration and configuration-consistency change. AQLprofile still
programs the numeric ID supplied by its caller. Adding a declaration does not
prove that the generic enumeration is wrong for every IP version.

## References

- [soc24 baseline](https://github.com/ROCm/rocm-systems/blob/6f441646398273bb16a065a3b3798d95f7a95563/projects/aqlprofile/linux/soc24_enum.h).
- [gfx1201-specific block information](https://github.com/ROCm/rocm-systems/blob/6f441646398273bb16a065a3b3798d95f7a95563/projects/aqlprofile/src/core/gfx12_factory.cpp#L113).
- [Historical gfx1201 SDK contract](https://github.com/ROCm/rocm-systems/blob/e92aa6ee8c01f991c9da08ea386330a6374e0f4a/projects/rocprofiler-sdk/source/share/rocprofiler-sdk/config.yaml).
- [Navi48 correction #158](https://github.com/ROCm/rocm-systems/commit/5dbb3b2f8083ef21f1a4a86c4adcfe4222d39076).
- The companion Compute counter-definition document describes the additional
  experimental selectors and their remaining public IP/RDL reference gaps.


## IP/RDL review scope

The seven selectors without a public chip-specific reference require
maintainer IP/RDL review before a hardware semantic claim. The catalog
describes the experimental configuration contract.
