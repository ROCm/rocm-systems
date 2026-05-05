#ifndef ROCM_HSA_CLOCK_SYNC_H_
#define ROCM_HSA_CLOCK_SYNC_H_

namespace rocr {
namespace clock_sync {

// Initialize the clock synchronization module. Spawns the background poller
// thread that periodically emits rocm_hsa:clock_sync tracepoints.
void init();

// Shut down the clock synchronization module. Signals the poller thread to
// stop and joins it.
void shutdown();

}  // namespace clock_sync
}  // namespace rocr

#endif  // ROCM_HSA_CLOCK_SYNC_H_
