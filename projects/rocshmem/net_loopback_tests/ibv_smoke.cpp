#include "reverse_offload/net/ibv.hpp"
#include <cstdio>
#include <cstring>
using namespace rocshmem::net;
int main() {
  Ibv ibv;
  if (!ibv.load()) { printf("load FAIL\n"); return 1; }
  printf("dlopen OK\n");
  int n = 0;
  auto **list = ibv.get_device_list(&n);
  if (!list || n == 0) { printf("no devices\n"); return 1; }
  printf("devices=%d first=%s\n", n, ibv.get_device_name(list[0]));
  auto *ctx = ibv.open_device(list[0]);
  if (!ctx) { printf("open_device FAIL\n"); return 1; }
  struct ibv_port_attr pa; std::memset(&pa,0,sizeof(pa));
  ibv.query_port(ctx, 1, &pa);
  printf("port1 state=%d link_layer=%d lid=%u\n", pa.state, pa.link_layer, pa.lid);
  auto *pd = ibv.alloc_pd(ctx);
  if (!pd) { printf("alloc_pd FAIL\n"); return 1; }
  auto *cq = ibv.create_cq(ctx, 64, nullptr, nullptr, 0);
  if (!cq) { printf("create_cq FAIL\n"); return 1; }
  struct ibv_qp_init_attr qa; std::memset(&qa,0,sizeof(qa));
  qa.send_cq = cq; qa.recv_cq = cq; qa.qp_type = IBV_QPT_RC;
  qa.cap.max_send_wr = 64; qa.cap.max_recv_wr = 1;
  qa.cap.max_send_sge = 1; qa.cap.max_recv_sge = 1;
  auto *qp = ibv.create_qp(pd, &qa);
  if (!qp) { printf("create_qp FAIL\n"); return 1; }
  printf("RC QP created qpn=%u\n", qp->qp_num);
  // Exercise the inline op-table pointers exist (don't post; just check non-null).
  if (!qp->context->ops.post_send || !cq->context->ops.poll_cq) {
    printf("ops table NULL\n"); return 1;
  }
  ibv.destroy_qp(qp); ibv.destroy_cq(cq); ibv.dealloc_pd(pd);
  ibv.close_device(ctx); ibv.free_device_list(list);
  printf("IBV WRAPPER OK\n");
  return 0;
}
