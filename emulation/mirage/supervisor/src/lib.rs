//! `mirage_supervisor`: the engine a `mirage run` is built out of.
//!
//! # One owner, no exceptions
//!
//! Every process mirage starts is owned by the process that asked for it,
//! and dies with it. `mirage run` owns its session's containers, its
//! emulator daemon and its workload processes; a `mirage exec` in another
//! terminal owns the processes *it* starts. Nothing is detached, nothing
//! is inherited by init, and nothing has to be reclaimed later.
//!
//! That is a deliberate reversal. Mirage previously had a long-lived
//! supervisor daemon owning every session, because sessions outlived the
//! commands that created them; before that, a detached `mirage host` per
//! session, with the filesystem as the channel to it. Both leaked, and
//! for the same underlying reason: a process nobody is responsible for is
//! a process nobody reaps.
//!
//! Three properties fall out of the current design, and each one used to
//! need machinery to approximate:
//!
//! * **Liveness is a fact, not an inference.** "Is this session alive?"
//!   is answered by whether the owning process is alive, so there is no
//!   heartbeat file, no staleness ladder, and no guessing.
//! * **Teardown is closed-loop.** Destroying a session does not return
//!   until every child has been waited on and every container removed.
//! * **Nothing needs a cap.** Finished-exec eviction and bounded output
//!   replay existed because a daemon ran forever; a process that exits
//!   frees its memory by exiting.
//!
//! # Structure
//!
//! * [`Run`] — one session: bring-up, health, teardown.
//! * [`session::Session`] — its profile, emulator runtime, containers and
//!   execs.
//! * [`exec::Exec`] — one command invocation's process grid.
//! * [`spec`] — the shared mapping from a session description plus a
//!   command to the processes that implement it.
//! * [`process`] — spawning, supervising and reliably killing processes.
//! * [`rpc`] — the socket a run serves so `mirage exec` can find it.

pub mod exec;
pub mod output;
pub mod process;
pub mod rpc;
pub mod run;
pub mod session;
pub mod spec;

pub use exec::Exec;
pub use process::{Exit, OutputChunk, SpawnSpec, Spawned, StdioMode};
pub use run::Run;
pub use session::Session;
pub use spec::{CallerStreams, build_specs};
