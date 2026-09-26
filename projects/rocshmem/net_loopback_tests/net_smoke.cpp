#include "reverse_offload/net/remote_region.hpp"
#include "reverse_offload/net/lane_map.hpp"
#include "reverse_offload/net/completion.hpp"
#include "reverse_offload/net/addr_exchange.hpp"
#include <cstdio>
#include <cstring>
using namespace rocshmem::net;
int main() {
  // remote_region: offset preservation
  RemoteRegion r{0x9000, 0x11};
  uintptr_t local_base = 0x1000, local_va = 0x1240;
  auto ra = remote_addr(r, local_va, local_base);
  if (ra != 0x9240) { printf("remote_addr FAIL %lx\n", ra); return 1; }

  // lane_map
  LaneMap lm(3 /*ctx*/, 4 /*pes*/);
  if (lm.num_lanes() != 12) { printf("num_lanes FAIL\n"); return 1; }
  if (lm.lane(2, 3) != 11) { printf("lane FAIL %d\n", lm.lane(2,3)); return 1; }
  if (lm.lane_begin(1) != 4 || lm.lane_end(1) != 8) { printf("lane range FAIL\n"); return 1; }

  // completion counters
  CompletionCounters cc(lm.num_lanes());
  cc.on_post(11, 5); cc.on_complete(11, 2);
  if (cc.outstanding(11) != 3 || cc.drained(11)) { printf("cc FAIL\n"); return 1; }
  cc.on_complete(11, 3);
  if (!cc.drained(11)) { printf("cc drain FAIL\n"); return 1; }

  // addr_exchange with a fake 2-PE allgather (simulate peer slot fill)
  struct Blob { uintptr_t base; uint64_t key; };
  int num_pes = 2, my_pe = 0;
  Blob mine{0xABC0, 0x22};
  auto fake_allgather = [&](void* inout, size_t bpp){
    // pretend PE1 published {0xDEF0, 0x33}
    auto* p = static_cast<char*>(inout);
    Blob peer{0xDEF0, 0x33};
    std::memcpy(p + 1*bpp, &peer, sizeof(Blob));
  };
  auto all = allgather_value<Blob>(mine, num_pes, my_pe, fake_allgather);
  if (all[0].base != 0xABC0 || all[1].base != 0xDEF0 || all[1].key != 0x33) {
    printf("allgather FAIL\n"); return 1;
  }
  printf("NET HELPERS OK\n");
  return 0;
}
