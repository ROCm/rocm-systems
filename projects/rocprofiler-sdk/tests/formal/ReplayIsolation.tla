---------------------------- MODULE ReplayIsolation ----------------------------
(***************************************************************************)
(* Model of the kernel-replay isolation window for ONE agent.              *)
(*                                                                         *)
(* Source of truth:                                                        *)
(*   source/lib/rocprofiler-sdk/hsa/queue.cpp  (WriteInterceptor)          *)
(*   source/docs/conceptual/kernel_replay/                                 *)
(*       kernel_replay_concurrency_and_isolation.md                        *)
(*                                                                         *)
(* The three isolation layers being modelled:                              *)
(*   1. per-agent shared_mutex: replay takes WRITER, ordinary dispatches    *)
(*      take READER across their submit only.                              *)
(*   2. agent-wide drain: replay waits until no async handler is in flight. *)
(*   3. snapshot/restore bracket the passes.                               *)
(*                                                                         *)
(* The reader lock is released at the end of submit, but the GPU work it    *)
(* launched stays in flight -- which is precisely why layer 2 exists.       *)
(*                                                                         *)
(* BYPASS models the HIP-graph fast path taking neither the reader lock nor *)
(* registering with async_started(), i.e. the hazard the guard in           *)
(* queue.cpp:521 exists to prevent.                                         *)
(***************************************************************************)
EXTENDS Naturals, FiniteSets

CONSTANTS Apps, BYPASS

VARIABLES appPC,      \* app thread -> "idle" | "submit" | "inflight"
          repPC,      \* "idle" | "wantw" | "drain" | "snap" | "pass" | "restore" | "done"
          readers,    \* set of app threads holding the reader lock
          writer      \* TRUE when the replayer holds the writer lock

vars == <<appPC, repPC, readers, writer>>

Init ==
    /\ appPC  = [a \in Apps |-> "idle"]
    /\ repPC  = "idle"
    /\ readers = {}
    /\ writer = FALSE

InFlight == {a \in Apps : appPC[a] = "inflight"}

(* ---------------- ordinary dispatch ---------------- *)

\* Enter submit: needs the reader lock, which the writer excludes.
AppSubmit(a) ==
    /\ appPC[a] = "idle"
    /\ ~writer
    /\ appPC' = [appPC EXCEPT ![a] = "submit"]
    /\ readers' = readers \cup {a}
    /\ UNCHANGED <<repPC, writer>>

\* Submit completes: reader released, but GPU work is now in flight.
AppLaunch(a) ==
    /\ appPC[a] = "submit"
    /\ appPC' = [appPC EXCEPT ![a] = "inflight"]
    /\ readers' = readers \ {a}
    /\ UNCHANGED <<repPC, writer>>

\* The async completion handler retires.
AppRetire(a) ==
    /\ appPC[a] = "inflight"
    /\ appPC' = [appPC EXCEPT ![a] = "idle"]
    /\ UNCHANGED <<repPC, readers, writer>>

\* HIP-graph fast path: submits with NO reader lock and NO async_started()
\* registration, so it is invisible to both the writer lock and the drain.
AppBypass(a) ==
    /\ BYPASS
    /\ appPC[a] = "idle"
    /\ appPC' = [appPC EXCEPT ![a] = "inflight"]
    /\ UNCHANGED <<repPC, readers, writer>>

(* ---------------- replay ---------------- *)

RepWant ==
    /\ repPC = "idle"
    /\ repPC' = "wantw"
    /\ UNCHANGED <<appPC, readers, writer>>

\* Writer lock: granted only once no reader is mid-submit.
RepAcquire ==
    /\ repPC = "wantw"
    /\ readers = {}
    /\ writer' = TRUE
    /\ repPC' = "drain"
    /\ UNCHANGED <<appPC, readers>>

\* Agent-wide drain: wait for every in-flight async handler to retire.
RepDrained ==
    /\ repPC = "drain"
    /\ InFlight = {}
    /\ repPC' = "snap"
    /\ UNCHANGED <<appPC, readers, writer>>

RepPass ==
    /\ repPC = "snap"
    /\ repPC' = "pass"
    /\ UNCHANGED <<appPC, readers, writer>>

RepRestore ==
    /\ repPC = "pass"
    /\ repPC' = "restore"
    /\ UNCHANGED <<appPC, readers, writer>>

\* Either loop to another pass or finish and drop the writer lock.
RepLoop ==
    /\ repPC = "restore"
    /\ repPC' = "pass"
    /\ UNCHANGED <<appPC, readers, writer>>

RepDone ==
    /\ repPC = "restore"
    /\ repPC' = "done"
    /\ writer' = FALSE
    /\ UNCHANGED <<appPC, readers>>

Next ==
    \/ \E a \in Apps : AppSubmit(a) \/ AppLaunch(a) \/ AppRetire(a) \/ AppBypass(a)
    \/ RepWant \/ RepAcquire \/ RepDrained \/ RepPass \/ RepRestore
    \/ RepLoop \/ RepDone

Spec == Init /\ [][Next]_vars /\ WF_vars(Next)

(* ---------------- properties ---------------- *)

\* The window in which device memory must not be mutated by anyone else.
InWindow == repPC \in {"snap", "pass", "restore"}

\* SAFETY: no ordinary dispatch may be in flight or mid-submit inside the window.
\* This is what guarantees a pass observes the snapshotted inputs and that
\* restore() does not revert application writes.
MemoryStable == InWindow => (InFlight = {} /\ readers = {})

\* The writer lock is exclusive against readers.
LockExclusion == writer => (readers = {})

\* LIVENESS: a replay that starts eventually finishes (no deadlock in the window).
ReplayTerminates == (repPC = "wantw") ~> (repPC = "done")
================================================================================
