//! Facts, states, and goals: the propositional world the planner reasons over.

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

/// A single proposition known to be true, e.g. `compiled(gfx950)`.
///
/// Facts are opaque strings to the solver. Higher layers impose
/// structure (such as `predicate(argument)`); the planner only ever
/// compares them for equality and set membership.
pub type Fact = String;

/// A monotone set of [`Fact`]s — everything currently known to be true.
///
/// States only ever grow as steps run (effects are additive), which is
/// what makes the planning search a shortest-path problem over a finite
/// lattice rather than a general search with backtracking.
#[derive(Debug, Clone, Default, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct State {
    facts: BTreeSet<Fact>,
}

impl State {
    /// An empty state — nothing is known yet.
    pub fn new() -> Self {
        Self::default()
    }

    /// Build a state from an iterator of facts.
    pub fn from_facts<I, S>(facts: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<Fact>,
    {
        Self {
            facts: facts.into_iter().map(Into::into).collect(),
        }
    }

    /// Whether `fact` currently holds.
    pub fn has(&self, fact: &str) -> bool {
        self.facts.contains(fact)
    }

    /// Whether every fact in `facts` currently holds.
    pub fn has_all<'a, I>(&self, facts: I) -> bool
    where
        I: IntoIterator<Item = &'a Fact>,
    {
        facts.into_iter().all(|f| self.facts.contains(f))
    }

    /// Insert a fact, returning `true` if it was newly added.
    pub fn insert(&mut self, fact: impl Into<Fact>) -> bool {
        self.facts.insert(fact.into())
    }

    /// A new state with all of `facts` added.
    pub fn with_all<'a, I>(&self, facts: I) -> State
    where
        I: IntoIterator<Item = &'a Fact>,
    {
        let mut next = self.clone();
        for f in facts {
            next.facts.insert(f.clone());
        }
        next
    }

    /// Iterate the facts in deterministic (sorted) order.
    pub fn facts(&self) -> impl Iterator<Item = &Fact> {
        self.facts.iter()
    }

    /// Number of facts known.
    pub fn len(&self) -> usize {
        self.facts.len()
    }

    /// Whether the state is empty.
    pub fn is_empty(&self) -> bool {
        self.facts.is_empty()
    }
}

/// The set of facts a caller wants the planner to establish.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Goal {
    required: BTreeSet<Fact>,
}

impl Goal {
    /// A goal requiring every fact in `facts`.
    pub fn new<I, S>(facts: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<Fact>,
    {
        Self {
            required: facts.into_iter().map(Into::into).collect(),
        }
    }

    /// Whether `state` satisfies every required fact.
    pub fn is_satisfied_by(&self, state: &State) -> bool {
        state.has_all(self.required.iter())
    }

    /// The required facts, in deterministic order.
    pub fn required(&self) -> impl Iterator<Item = &Fact> {
        self.required.iter()
    }

    /// Required facts not present in `state`.
    pub fn missing_in(&self, state: &State) -> Vec<Fact> {
        self.required
            .iter()
            .filter(|f| !state.has(f))
            .cloned()
            .collect()
    }

    /// Whether the goal asks for anything at all.
    pub fn is_empty(&self) -> bool {
        self.required.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn state_membership_and_growth() {
        let mut s = State::from_facts(["a", "b"]);
        assert!(s.has("a"));
        assert!(!s.has("c"));
        assert!(s.insert("c"));
        assert!(!s.insert("c"));
        assert!(s.has("c"));
    }

    #[test]
    fn has_all_and_with_all() {
        let s = State::from_facts(["a", "b"]);
        let req: Vec<Fact> = vec!["a".into(), "b".into()];
        assert!(s.has_all(req.iter()));
        let more: Vec<Fact> = vec!["c".into()];
        let s2 = s.with_all(more.iter());
        assert!(s2.has("c"));
        // Original is untouched (immutability of `with_all`).
        assert!(!s.has("c"));
    }

    #[test]
    fn goal_satisfaction_and_missing() {
        let goal = Goal::new(["x", "y"]);
        let s = State::from_facts(["x"]);
        assert!(!goal.is_satisfied_by(&s));
        assert_eq!(goal.missing_in(&s), vec!["y".to_string()]);
        let s2 = State::from_facts(["x", "y", "z"]);
        assert!(goal.is_satisfied_by(&s2));
        assert!(goal.missing_in(&s2).is_empty());
    }
}
