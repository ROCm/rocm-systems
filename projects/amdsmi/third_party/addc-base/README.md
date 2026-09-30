# addc-base (vendored)

Snapshot of `AMD-DCTOOLS/addc-base` at commit `95a5331` (version 3.3.0), used only by
`libamd_smi` to turn a CPER record into AFIDs and the JSON event report.

Copied from the upstream tree: `CMakeLists.txt`, `cmake/`, `include/`, `LICENSE`, `meson.build` and
`src/{addc-common,addc-cper-parser,addc-cper-pipeline,addc-decoder,addc-mca,api,core,products,tier_api}`.
The tests, fuzz targets, CLI (`src/cli`), tools and manifest are not copied. Nothing here is edited.

`src/CMakeLists.txt` adds this tree with `ADDC_LIB_STATIC=ON`, so addc is built as static archives that end
up inside `libamd_smi`, and adjusts the targets' usage requirements from outside (see the comments there).

To update: run `git -C <addc-base> archive <commit> CMakeLists.txt cmake include LICENSE meson.build src | tar -x -C third_party/addc-base`,
delete `src/cli`, and update the commit above. CMake reads the version from `meson.build`.

Because addc is linked into `libamd_smi`, the shared library exports no addc symbols. `libamd_smi_static`
carries addc's default-visibility C symbols (`addc_*`), so it cannot be linked together with another copy
of addc. The exported static targets are named `amd_smi_addc_*` to avoid clashing with a consumer's own
addc-base targets.
