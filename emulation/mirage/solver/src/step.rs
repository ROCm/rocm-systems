//! Steps: the tools/actions the planner can sequence.

use serde::{Deserialize, Serialize};

use crate::state::{Fact, State};

/// A stable identifier for a [`Step`] (typically a tool id).
pub type StepId = String;

/// A single action the planner may schedule.
///
/// A step is *applicable* in a [`State`] when all of its [`requires`]
/// facts hold. Applying it yields a new state with every fact in
/// [`produces`] added. Effects are additive: a step never removes a
/// fact, which keeps the planning lattice monotone.
///
/// [`requires`]: Step::requires
/// [`produces`]: Step::produces
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Step {
    /// Stable identity used in plans, caching, and diagnostics.
    pub id: StepId,
    /// Preconditions: every fact here must hold for the step to run.
    #[serde(default)]
    pub requires: Vec<Fact>,
    /// Effects: facts added to the state when the step runs.
    pub produces: Vec<Fact>,
    /// Relative cost of running the step. The planner minimizes the
    /// total cost of the chosen sequence. Units are arbitrary but
    /// should be consistent across a catalogue (e.g. rough wall-clock
    /// milliseconds, or a relative effort score).
    pub cost: u64,
    /// Whether the step is idempotent and its result may be memoized in
    /// a [`crate::Cache`]. Only cacheable steps are ever reused.
    #[serde(default)]
    pub cacheable: bool,
}

impl Step {
    /// Construct a step with the given id, preconditions, effects, and cost.
    pub fn new(
        id: impl Into<StepId>,
        requires: impl IntoIterator<Item = impl Into<Fact>>,
        produces: impl IntoIterator<Item = impl Into<Fact>>,
        cost: u64,
    ) -> Self {
        Self {
            id: id.into(),
            requires: requires.into_iter().map(Into::into).collect(),
            produces: produces.into_iter().map(Into::into).collect(),
            cost,
            cacheable: false,
        }
    }

    /// Builder: mark this step cacheable.
    pub fn cacheable(mut self, yes: bool) -> Self {
        self.cacheable = yes;
        self
    }

    /// Whether every precondition holds in `state`.
    pub fn applicable_in(&self, state: &State) -> bool {
        state.has_all(self.requires.iter())
    }

    /// Whether applying this step would add at least one new fact.
    /// Steps that produce only already-known facts are never useful to
    /// schedule and are pruned by the planner.
    pub fn advances(&self, state: &State) -> bool {
        self.produces.iter().any(|f| !state.has(f))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn applicability_and_advancement() {
        let s = Step::new("compile", ["source"], ["compiled"], 10);
        let empty = State::new();
        assert!(!s.applicable_in(&empty));
        let with_src = State::from_facts(["source"]);
        assert!(s.applicable_in(&with_src));
        assert!(s.advances(&with_src));
        let done = State::from_facts(["source", "compiled"]);
        assert!(!s.advances(&done));
    }

    #[test]
    fn cacheable_builder() {
        let s = Step::new("x", [] as [&str; 0], ["y"], 1).cacheable(true);
        assert!(s.cacheable);
    }
}
