# Algorithms

This directory contains RCCL's collective algorithm implementations
(e.g. ring, tree, and CollNet variants) used to schedule and execute
This directory contains RCCL's specialized collective implementations:
`dda/` (direct device access, over IPC and fabric), `gin/` (GPU-initiated
networking AllReduce and AllToAll over SDMA), and `rccl_ep/` (intranode
expert-parallel dispatch and combine). The ring, tree, and CollNet
algorithms live elsewhere, under `src/device/`, `src/graph/`, and
`src/transport/coll_net.cc`.
