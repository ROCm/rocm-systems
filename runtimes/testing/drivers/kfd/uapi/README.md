<!-- SPDX-License-Identifier: MIT -->

# Expected KFD userspace interface

This snapshot belongs to the tests, independently of headers used to build any runtime. Its files are byte-identical to the following sources at rocm-systems commit `7e826fcf68aabaf0bbab9478c6492a2b560c1b4b` (the Git blob identities below identify the exact contents):

| Test include path | Source under `projects/rocr-runtime/libhsakmt/include/hsakmt/` | Git blob |
|---|---|---|
| `linux/kfd_ioctl.h` | `linux/kfd_ioctl.h` | `81237c7689c2f86bb88eca17fa609278766d78b1` |
| `hsakmt/drm/drm.h` | `drm/drm.h` | `5e8573d4d909e03d6c16728d596fa7c7a1cbd686` |
| `hsakmt/drm/drm_mode.h` | `drm/drm_mode.h` | `d390011b89b49546bd4ebf47b71567cb1bc958ce` |

The `hsakmt/drm` layout preserves the KFD header's companion includes; it does not introduce a thunk library dependency. General Linux types and architecture ioctl encoding headers come from the native compiler sysroot. Copyright and license notices remain in each header. There are no content adaptations.

The snapshot describes UAPI 1.19. The initial version-query test requires only the original 1.1 contract and accepts later compatible minor versions; matching the snapshot's entire minor version is not an execution prerequisite. This test does not qualify queues, firmware or every ioctl in the snapshot.

To refresh, select and record the source revision, copy the coherent KFD/DRM closure while retaining notices, update this table's blob identities, and rebuild the KFD-only selection without any runtime include root. Run the required suite on its admitted driver environment and review changes to command numbers, layouts and capability prerequisites before accepting the refresh. Runtime header updates alone never refresh this snapshot.
