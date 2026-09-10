#include "reverse_offload/net/ibv.hpp"
#include "reverse_offload/net/ib_ctrl.hpp"
#include "reverse_offload/net/mr_registry.hpp"
#include "reverse_offload/net/lane_map.hpp"
#include <cstdio>
#include <cstdlib>
#include <cstring>
using namespace rocshmem::net;
int main() {
  Ibv ibv;
  if (!ibv.load()) return 1;
  IbCtrl ctrl;
  if (!ctrl.open(&ibv, nullptr, 1)) return 1;
  LaneMap lm(1, 2);
  if (!ctrl.create_queues(lm, 64, 64)) return 1;
  std::vector<DestInfo> d; ctrl.fill_local_dest(d);
  if (!ctrl.connect_qp(0, d[1]) || !ctrl.connect_qp(1, d[0])) return 1;

  // Host "heap" buffer, page-aligned.
  void* heap = nullptr;
  if (posix_memalign(&heap, 4096, 4096) != 0) return 1;
  memset(heap, 0, 4096);
  MrRegistry mr;
  if (!mr.register_heap(&ibv, const_cast<ibv_pd*>(ctrl.dev().pd), heap, 4096,
                        /*is_device=*/false, /*dmabuf=*/{})) return 1;
  printf("heap lkey=0x%x rkey=0x%x base=%p\n", mr.heap_lkey(), mr.heap_rkey(), heap);

  // Write a pattern into src region [0..64), RDMA_WRITE it to dst region [2048..).
  char* h = static_cast<char*>(heap);
  const char* msg = "ROCSHMEM-VERBS-LOOPBACK-0123456789ABCDEF";
  size_t len = strlen(msg) + 1;
  memcpy(h, msg, len);

  struct ibv_sge sge; memset(&sge,0,sizeof(sge));
  sge.addr = reinterpret_cast<uintptr_t>(h);
  sge.length = len;
  sge.lkey = mr.heap_lkey();
  struct ibv_send_wr wr; memset(&wr,0,sizeof(wr));
  wr.wr_id = 0x1234; wr.sg_list = &sge; wr.num_sge = 1;
  wr.opcode = IBV_WR_RDMA_WRITE; wr.send_flags = IBV_SEND_SIGNALED;
  wr.wr.rdma.remote_addr = reinterpret_cast<uintptr_t>(h) + 2048;
  wr.wr.rdma.rkey = mr.heap_rkey();
  struct ibv_send_wr* bad = nullptr;
  int rc = ibv.post_send(ctrl.qp(0), &wr, &bad);
  if (rc != 0) { printf("post_send FAIL rc=%d\n", rc); return 1; }

  struct ibv_wc wc; int n = 0;
  for (int spins=0; spins<10000000 && n==0; ++spins) n = ibv.poll_cq(ctrl.cq(0), 1, &wc);
  if (n <= 0) { printf("poll_cq timeout\n"); return 1; }
  if (wc.status != IBV_WC_SUCCESS) { printf("WC status=%d (%s)\n", wc.status, ibv_wc_status_str(wc.status)); return 1; }
  printf("completion wr_id=0x%lx status=SUCCESS\n", (unsigned long)wc.wr_id);

  if (memcmp(h + 2048, msg, len) == 0) { printf("RDMA_WRITE LOOPBACK DATA OK: '%s'\n", h + 2048); }
  else { printf("DATA MISMATCH\n"); return 1; }
  return 0;
}
