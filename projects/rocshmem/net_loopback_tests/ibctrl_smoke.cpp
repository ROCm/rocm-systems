#include "reverse_offload/net/ibv.hpp"
#include "reverse_offload/net/ib_ctrl.hpp"
#include "reverse_offload/net/lane_map.hpp"
#include <cstdio>
using namespace rocshmem::net;
int main() {
  Ibv ibv;
  if (!ibv.load()) { printf("load FAIL\n"); return 1; }
  IbCtrl ctrl;
  if (!ctrl.open(&ibv, nullptr, 1)) { printf("open FAIL\n"); return 1; }
  printf("gid_index=%d mtu=%d\n", ctrl.dev().gid_index, (int)ctrl.dev().mtu);
  LaneMap lm(1 /*ctx*/, 2 /*lanes as pseudo-pes*/);
  if (!ctrl.create_queues(lm, 64, 64)) { printf("create_queues FAIL\n"); return 1; }
  std::vector<DestInfo> local;
  ctrl.fill_local_dest(local);
  printf("qpn[0]=%u qpn[1]=%u\n", local[0].qpn, local[1].qpn);
  // cross-connect: QP0<->QP1 (two distinct QPs, RC loopback)
  if (!ctrl.connect_qp(0, local[1])) { printf("connect0 FAIL\n"); return 1; }
  if (!ctrl.connect_qp(1, local[0])) { printf("connect1 FAIL\n"); return 1; }
  // verify both reached RTS
  struct ibv_qp_attr qa; struct ibv_qp_init_attr qia;
  for (int l=0;l<2;l++){
    ibv_query_qp(ctrl.qp(l), &qa, IBV_QP_STATE, &qia);
    printf("qp[%d] state=%d (RTS=%d)\n", l, qa.qp_state, IBV_QPS_RTS);
    if (qa.qp_state != IBV_QPS_RTS) { printf("NOT RTS\n"); return 1; }
  }
  printf("IB_CTRL RC LOOPBACK OK\n");
  return 0;
}
