* `NCCL_GIN_ALLGATHER_ENABLE` (default `1`): selects the single-node GIN AllGather on gfx950 and gfx1250 when the buffers are single-segment `NCCL_WIN_COLL_SYMMETRIC` device windows. Set `0` to keep the previous AllGather algorithms.
* `NCCL_GIN_ALLGATHER_MIN_BYTES` (default `0`): per-rank byte floor below which GIN AllGather is not selected.
* `NCCL_GIN_ALLGATHER_SDMA_MIN_BYTES` (default 8 MiB): per-rank byte threshold at or above which GIN AllGather uses SDMA puts instead of LSA copies.
* `NCCL_GIN_ALLGATHER_GFX1250_ENABLE` (default `0`): opt in to GIN AllGather on gfx1250, including DPX. Left off, gfx1250 keeps the pre-existing AllGather selectors.
