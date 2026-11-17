#ifndef DDA_ORDER_H_
#define DDA_ORDER_H_

#include "dda_done.h"

#include "checks.h"
#include "comm.h"

#include <hip/hip_ext.h>

// Empty while capturing, where the epilogue still records doneEvent, and when this comm has no signal word.
static inline DdaDoneMark rcclPrepareAddonDone(struct ncclComm* comm) {
  DdaDoneMark mark{nullptr, nullptr, 0};
  rcclDdaDoneState* state = comm->ddaDone;
  if (comm->addonStopEvent == nullptr || state == nullptr || state->seq == nullptr || state->grid == nullptr) {
    return mark;
  }
  state->seqValue += 1;
  mark.seq = state->seq;
  mark.grid = state->grid;
  mark.value = state->seqValue;
  return mark;
}

// Call after a host-staged launch succeeds. Taking the stop event tells the epilogue not to record.
static inline void rcclCommitAddonDone(struct ncclComm* comm, DdaDoneMark mark) {
  if (mark.seq == nullptr || comm->ddaDone == nullptr) return;
  comm->ddaDone->wait = mark.value;
  comm->ddaDone->signaled = true;
  (void)rcclTakeAddonStopEvent(comm);
}

// Wait for the previous collective on this stream. A live signal replaces doneEvent.
inline ncclResult_t rcclDdaWaitPrevious(struct ncclComm* comm, cudaStream_t stream) {
  rcclDdaDoneState* state = comm->ddaDone;
  if (state != nullptr && state->wait != 0 && state->seq != nullptr) {
    CUDACHECK(hipStreamWaitValue64(stream, state->seq, state->wait, hipStreamWaitValueGte, ~0ull));
  } else {
    CUDACHECK(hipStreamWaitEvent(stream, comm->doneEvent, 0));
  }
  return ncclSuccess;
}

// A real doneEvent record covers an earlier signal. A capture record does not, and a launch that
// published the signal is itself the completion.
inline void rcclDdaCoverWait(struct ncclComm* comm, bool capturing, bool eventTaken) {
  rcclDdaDoneState* state = comm->ddaDone;
  if (capturing || state == nullptr) return;
  if (!eventTaken || !state->signaled) state->wait = 0;
}

inline void rcclDdaClearSignaled(struct ncclComm* comm) {
  if (comm->ddaDone != nullptr) comm->ddaDone->signaled = false;
}

#endif
