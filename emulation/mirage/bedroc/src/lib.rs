//! `mirage_bedroc`: a composable correctness strategy for GPU kernels.
//!
//! Bedroc turns a high-level **request** — "here is my kernel source,
//! these are the target GPUs, prove it has no data hazards and produces
//! the correct output" — into an executable **plan and proof**. It does
//! this by combining three inputs:
//!
//! 1. **What the user wants** ([`request::BedrocRequest`]): the kernel
//!    source, the target architectures, and the correctness properties
//!    to establish.
//! 2. **What the environment offers** ([`environment::Environment`]):
//!    the physical GPUs present and which tools are installed. This is
//!    the implicit constraint set of wherever the user is running.
//! 3. **What the tools can do** ([`manifest::ToolManifest`]): a catalogue
//!    of correctness tools, each declaring its preconditions, effects,
//!    cost, and environment requirements. **Tools are pure data** —
//!    JSON manifests, never hardcoded logic — so new tools are added
//!    without touching this crate (see `docs/bedroc.md`).
//!
//! The [`engine::Engine`] compiles these into a propositional planning
//! problem and hands it to [`mirage_solver`], which finds the
//! minimum-cost sequence of tools that establishes the requested
//! properties (sharing work across goals, reusing cached results, and
//! reporting precisely which goals are unsupported in the current
//! environment). The resulting [`proof::Proof`] records exactly what was
//! checked, what was not, and why.
//!
//! Because every tool declares the transformation it performs, the same
//! catalogue can be **differentially fuzzed** ([`fuzz`]): the engine
//! enumerates distinct tool sequences that should establish the same
//! property and confirms they agree, exercising the tools against each
//! other.
//!
//! # Layers
//!
//! ```text
//!   BedrocRequest + Environment + [ToolManifest]
//!                    │
//!            Engine::plan / Engine::prove
//!                    │  (lowers to facts + steps)
//!              mirage_solver::plan
//!                    │
//!                  Proof
//! ```

pub mod engine;
pub mod environment;
pub mod executor;
pub mod fuzz;
pub mod manifest;
pub mod proof;
pub mod request;

pub use engine::{Engine, EngineError};
pub use environment::Environment;
pub use executor::{ExecutionReport, ExecutedStep, Executor};
pub use manifest::{ToolCatalog, ToolManifest};
pub use proof::{GoalOutcome, Proof, ProofStep};
pub use request::{BedrocRequest, GoalKind, ResolvedTarget, SourceKind};
