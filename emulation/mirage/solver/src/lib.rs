//! `mirage_solver`: a small, domain-agnostic correctness **path planner**.
//!
//! The solver knows nothing about GPUs, kernels, or emulators. It works
//! over an abstract world of **facts** and **steps**:
//!
//! * A [`State`] is a monotone set of [`Fact`]s — things currently known
//!   to be true (e.g. `compiled(gfx950)`).
//! * A [`Step`] is a tool/action with **preconditions** ([`Step::requires`])
//!   and **effects** ([`Step::produces`]). A step can run only once all of
//!   its preconditions hold, and running it adds its effects to the state.
//!   Effects are **additive only** — a step never retracts a fact.
//! * A [`Goal`] is the set of facts the caller wants to establish.
//!
//! Given an initial [`State`], a pool of [`Step`]s, and a [`Goal`], the
//! [`plan`] function finds a **minimum-cost** ordered sequence of steps
//! that reaches a state satisfying the goal, or reports exactly which
//! goal facts are unreachable. Because effects are additive, the search
//! space is the lattice of fact-sets and the optimal solution is found
//! with a Dijkstra search over reachable states ([`plan`]). The same
//! machinery enumerates *alternative* plans ([`enumerate_plans`]) which
//! is what makes differential fuzzing of a tool catalogue possible:
//! distinct step sequences that establish the same goal should agree.
//!
//! Idempotent steps can be memoized through a [`Cache`], so a plan that
//! reuses an already-computed result pays a reduced cost and can skip
//! re-execution.
//!
//! This crate has no knowledge of how steps are *executed*; it only
//! plans. Execution (and the mapping from a domain like ROCm kernels to
//! facts and steps) lives in higher layers such as `mirage_bedroc`.

pub mod cache;
pub mod plan;
pub mod state;
pub mod step;

pub use cache::{Cache, CacheKey, CachedResult};
pub use plan::{Plan, PlanError, PlanStep, enumerate_plans, plan, plan_with_cache};
pub use state::{Fact, Goal, State};
pub use step::{Step, StepId};
