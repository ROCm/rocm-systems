#include "net/ibv.hpp"
#include "net/ib_ctrl.hpp"
#include "net/mr_registry.hpp"
#include "net/rma_engine.hpp"
#include "net/window_info_verbs.hpp"
#include "hdp_policy.hpp"
#include <cstdio>
#include <cstdlib>
#include <cstring>
using namespace rocshmem;
using namespace rocshmem::net;

int main() {
  Ibv ibv;
  if (!ibv.load()) { printf("load FAIL\n"); return 1; }
  IbCtrl ctrl;
  if (!ctrl.open(&ibv, nullptr, 1)) { printf("open FAIL\n"); return 1; }
  LaneMap lm(1 /*ctx*/, 2 /*pes*/);
  if (!ctrl.create_queues(lm, 64, 64)) { printf("create_queues FAIL\n"); return 1; }
  // cross-connect the 2 QPs for RC loopback (lane0 <-> lane1)
  std::vector<DestInfo> d; ctrl.fill_local_dest(d);
  if (!ctrl.connect_qp(0, d[1]) || !ctrl.connect_qp(1, d[0])) { printf("connect FAIL\n"); return 1; }

  // host "symmetric heap"
  void* heap=nullptr; if (posix_memalign(&heap,4096,65536)) return 1; memset(heap,0,65536);
  MrRegistry mr;
  if (!mr.register_heap(&ibv, const_cast<ibv_pd*>(ctrl.dev().pd), heap, 65536, false, {})) { printf("reg heap FAIL\n"); return 1; }
  // fake 2-PE allgather: both peers' region = our own heap (loopback)
  auto self_allgather = [&](void* inout, size_t bpp){
    struct HK{ uintptr_t base; uint64_t rkey; };
    HK me{(uintptr_t)heap, mr.heap_rkey()};
    for (int pe=0; pe<2; ++pe) memcpy((char*)inout+pe*bpp, &me, sizeof(HK));
  };
  mr.exchange_heap(2, 0, self_allgather);
  // amo scratch (host, registered)
  void* scratch=nullptr; if (posix_memalign(&scratch,64,64)) return 1; *(uint64_t*)scratch=0;
  uint32_t scr_lkey = mr.register_local(&ibv, const_cast<ibv_pd*>(ctrl.dev().pd), scratch, 64, false, {});

  std::vector<LocalMr> lmrs{
    { (uintptr_t)heap, (uintptr_t)heap+65536, mr.heap_lkey() },
    { (uintptr_t)scratch, (uintptr_t)scratch+64, scr_lkey },
  };
  RmaEngine rma(&ibv, &ctrl, &mr, lm, (uintptr_t)heap, lmrs);
  NoHdpPolicy hdp;
  WindowInfoVerbs win(&rma, &hdp, 0, heap, 65536, scratch, scr_lkey);

  char* h = (char*)heap;
  // 1) put_bytes: src=h+0 -> dst=h+2048 (pe1)
  const char* msg="WINDOWINFOVERBS-PUT-CHECK-0123456789";
  size_t n=strlen(msg)+1; memcpy(h+0, msg, n);
  win.put_bytes(h+2048, h+0, n, 1);
  if (memcmp(h+2048, msg, n)!=0){ printf("PUT FAIL: '%s'\n", h+2048); return 1; }
  printf("put_bytes OK: '%s'\n", h+2048);

  // 2) get_bytes: read remote h+2048 -> local h+4096
  win.get_bytes(h+4096, h+2048, n, 1);
  if (memcmp(h+4096, msg, n)!=0){ printf("GET FAIL\n"); return 1; }
  printf("get_bytes OK: '%s'\n", h+4096);

  // 3) amo_fadd: [h+8192]=100; fadd 5 -> returns 100, becomes 105
  *(uint64_t*)(h+8192)=100;
  uint64_t old = win.amo_fadd(h+8192, 5, 1);
  if (old!=100 || *(uint64_t*)(h+8192)!=105){ printf("FADD FAIL old=%lu val=%lu\n", old, *(uint64_t*)(h+8192)); return 1; }
  printf("amo_fadd OK: old=%lu new=%lu\n", old, *(uint64_t*)(h+8192));

  // 4) amo_cas: cmp 105 swap 200 -> returns 105, becomes 200
  uint64_t prev = win.amo_cas(h+8192, 105, 200, 1);
  if (prev!=105 || *(uint64_t*)(h+8192)!=200){ printf("CAS FAIL prev=%lu val=%lu\n", prev, *(uint64_t*)(h+8192)); return 1; }
  printf("amo_cas OK: prev=%lu new=%lu\n", prev, *(uint64_t*)(h+8192));

  // 5) nbi + quiet
  memcpy(h+0, "NBI", 4);
  win.put_nbi(h+16384, h+0, 4, 1);
  win.quiet();
  if (memcmp(h+16384,"NBI",4)!=0){ printf("PUT_NBI/QUIET FAIL\n"); return 1; }
  printf("put_nbi+quiet OK\n");

  printf("WINDOWINFOVERBS ALL OK\n");
  return 0;
}
