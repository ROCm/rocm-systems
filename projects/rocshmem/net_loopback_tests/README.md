# net/ verbs loopback tests (standalone, for reference)

Standalone single-process tests that validated the `src/net/` verbs building
blocks on this node's real RoCE HCA (Broadcom bnxt) during the host-init-verbs
work. They are NOT part of the CMake build — they're hand-compiled references.

They build on each other, simplest first:

| File | Exercises | Deps to compile |
|---|---|---|
| `net_smoke.cpp` | pure C++ helpers: `remote_region`, `lane_map`, `completion`, `addr_exchange` | `-I src` only (g++ fine) |
| `ibv_smoke.cpp` | `net::Ibv` dlopen wrapper: device list/open, PD, CQ, RC QP create | `net/ibv.cpp` `-ldl` |
| `ibctrl_smoke.cpp` | `net::IbCtrl` RC connect: 2 QPs cross-connected → RTS | `net/ibv.cpp net/ib_ctrl.cpp` `-ldl` (uses `-libverbs` only for `ibv_query_qp`/status strings) |
| `rdma_loopback.cpp` | full data plane: `MrRegistry` + a loopback `RDMA_WRITE` w/ completion + data check | `net/{ibv,ib_ctrl,mr_registry}.cpp` `-ldl` |
| `wiv_test.cpp` | `WindowInfoVerbs` + `RmaEngine`: put/get/nbi/quiet + 8B fadd/cas over a 2-QP loopback | `net/{ibv,ib_ctrl,mr_registry,rma_engine,window_info_verbs}.cpp` `-ldl` |

## Build & run the most complete one (`wiv_test`)

From the rocSHMEM project root, after a build exists under `build/` (for the
generated `rocshmem_config.h`):

```sh
CLANG=/opt/rocm-7.2.1/lib/llvm/bin/clang++     # the ROCm clang++ (hipcc works too)
OMPI_INC=/home/mshantha/software/install/ompi/include   # for the transitive <mpi.h>

$CLANG -x hip --offload-arch=gfx950 -std=c++17 \
  -I include -I src -I build/include -I build/include/rocshmem \
  -isystem "$OMPI_INC" \
  net_loopback_tests/wiv_test.cpp \
  src/net/ibv.cpp src/net/ib_ctrl.cpp src/net/mr_registry.cpp \
  src/net/rma_engine.cpp src/net/window_info_verbs.cpp \
  -ldl -o /tmp/wiv_test
/tmp/wiv_test
```

Expected output (real RoCE HCA):
```
[rocSHMEM] verbs: using <hca> port 1 (RoCE)
put_bytes OK: '...'
get_bytes OK: '...'
amo_fadd OK: old=100 new=105
amo_cas OK: prev=105 new=200
put_nbi+quiet OK
WINDOWINFOVERBS ALL OK
```

Notes:
- `wiv_test` uses `NoHdpPolicy` (HDP flush is a no-op in the bnxt build config) and
  host memory as a stand-in symmetric heap, so no GPU is required to run it.
- The simpler tests (`net_smoke`, `ibv_smoke`) need fewer include paths; drop the
  `-I include`/`-isystem` MPI paths for those that don't pull in `window_info.hpp`.
- These are loopback (one process, two cross-connected QPs on the same HCA); a
  true multi-process test would exchange DestInfo/rkeys across ranks.
