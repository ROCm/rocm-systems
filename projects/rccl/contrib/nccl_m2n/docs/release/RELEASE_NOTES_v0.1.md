# NCCL M2N v0.1 Release Notes

Initial preview of NCCL M2N reshard functionality.

## New Features

- The initial public C API, `ncclReshardWithWindow`, redistributes a tensor
  between source and destination layouts through a caller-registered
  `ncclWindow_t`.
- `ncclMesh_t` and `ncclDistTensor_t` describe 2-D rank topology, tensor
  layout, and placement for each side of the transfer.
- `ncclM2nInit` and `ncclM2nFinalize` manage M2N runtime state.
- Ring uses hierarchical transfers with intra-NVLink fan-out. Direct uses
  per-rank GIN puts. Both support cross-dimension transpose for 2-D and 3-D
  tensors.
- Calls support `NULL`, `cudaStreamLegacy`, and `cudaStreamPerThread` through
  internal non-blocking streams with ordered completion.
- Three benchmarks cover single-layer, batched, and configuration-driven model
  transfers: `reshard_bench`, `reshard_batch_bench_user_window`, and
  `reshard_model_bench`.

## Fixes

- None. This is the initial release.

## Known Limitations

| Limitation | Description |
|---|---|
| Transfer API | Only `ncclReshardWithWindow` is available. Callers provide the symmetric-memory window. |
| Window contract | Active source and destination pointers must be in the supplied window and use the same offset on a rank. |
| Layout rank | Meshes support up to 2 dimensions. Tensors support up to 3 dimensions. |
