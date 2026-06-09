//! The user-facing request: what to prove, about which kernel, for which targets.

use serde::{Deserialize, Serialize};

/// The kind of kernel source supplied. Determines which compile tool
/// applies and which source-level facts hold.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum SourceKind {
    /// HIP C++ source.
    #[default]
    Hip,
    /// Handwritten AMDGPU assembly.
    Asm,
    /// A prebuilt code object (already compiled).
    CodeObject,
}

impl SourceKind {
    /// The canonical lowercase name.
    pub fn as_str(&self) -> &'static str {
        match self {
            SourceKind::Hip => "hip",
            SourceKind::Asm => "asm",
            SourceKind::CodeObject => "codeobject",
        }
    }

    /// Parse from a user string, accepting common spellings.
    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "hip" | "hip++" | "cpp" | "c++" => Some(SourceKind::Hip),
            "asm" | "assembly" | "s" => Some(SourceKind::Asm),
            "codeobject" | "code-object" | "co" | "hsaco" | "prebuilt" => {
                Some(SourceKind::CodeObject)
            }
            _ => None,
        }
    }

    /// The source-level facts this kind establishes in the initial
    /// state. A prebuilt code object is treated as already `compiled`
    /// for every target, so no compile step is needed.
    pub fn source_facts(&self, targets: &[ResolvedTarget]) -> Vec<String> {
        match self {
            SourceKind::Hip => vec!["source:hip".to_string()],
            SourceKind::Asm => vec!["source:asm".to_string()],
            SourceKind::CodeObject => targets
                .iter()
                .map(|t| format!("compiled:{}", t.gfx))
                .collect(),
        }
    }
}

/// A correctness property the user wants established.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GoalKind {
    /// No data/wait hazards (static Wait Check).
    NoDataHazards,
    /// The kernel produces the correct output (emulation / oracle).
    CorrectOutput,
    /// No shared-memory (LDS) data races (LdsSan / ConSan).
    NoDataRaces,
    /// Floating-point numeric behavior is correct (FPSan).
    FpCorrect,
}

impl GoalKind {
    /// The canonical fact predicate for this goal (before target
    /// substitution).
    pub fn predicate(&self) -> &'static str {
        match self {
            GoalKind::NoDataHazards => "no_hazards",
            GoalKind::CorrectOutput => "correct_output",
            GoalKind::NoDataRaces => "no_races",
            GoalKind::FpCorrect => "fp_correct",
        }
    }

    /// A short human label.
    pub fn label(&self) -> &'static str {
        match self {
            GoalKind::NoDataHazards => "no data hazards",
            GoalKind::CorrectOutput => "correct output",
            GoalKind::NoDataRaces => "no data races",
            GoalKind::FpCorrect => "correct floating-point",
        }
    }

    /// The goal fact for a specific target, e.g. `no_hazards:gfx950`.
    pub fn fact_for(&self, target: &str) -> String {
        format!("{}:{}", self.predicate(), target)
    }

    /// Parse a user string, accepting friendly aliases.
    pub fn parse(s: &str) -> Option<Self> {
        let norm = s.trim().to_ascii_lowercase().replace(['-', ' '], "_");
        match norm.as_str() {
            "no_data_hazards" | "no_hazards" | "hazards" | "waitcheck" | "wait_check" => {
                Some(GoalKind::NoDataHazards)
            }
            "correct_output" | "correct" | "output" | "correctness" => {
                Some(GoalKind::CorrectOutput)
            }
            "no_data_races" | "no_races" | "races" | "race" | "ldssan" | "consan" => {
                Some(GoalKind::NoDataRaces)
            }
            "fp_correct" | "fp" | "fpsan" | "numeric" | "float" => Some(GoalKind::FpCorrect),
            _ => None,
        }
    }
}

/// A requested target, resolved from a possibly-friendly name (e.g.
/// `mi350`) to a canonical gfx architecture (e.g. `gfx950`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResolvedTarget {
    /// The name the user supplied.
    pub requested: String,
    /// The canonical `gfxNNN` architecture used in facts.
    pub gfx: String,
}

/// Resolve a friendly target name to a canonical gfx architecture.
///
/// Recognizes a small, documented table of AMD Instinct marketing names
/// and passes through anything already shaped like `gfxNNN`. Unknown
/// names are passed through unchanged so the catalogue can still be
/// exercised against architectures this table does not yet list.
pub fn resolve_target(name: &str) -> ResolvedTarget {
    let key = name.trim().to_ascii_lowercase();
    let gfx = match key.as_str() {
        // CDNA3
        "mi300" | "mi300x" | "mi300a" | "mi308" | "mi308x" | "mi325" | "mi325x" => "gfx942",
        // CDNA4
        "mi350" | "mi350x" | "mi355" | "mi355x" => "gfx950",
        // CDNA-next / forthcoming
        "mi400" | "mi450" | "mi450x" => "gfx1250",
        other => {
            // Already a gfx target, or unknown: pass through verbatim.
            return ResolvedTarget {
                requested: name.to_string(),
                gfx: other.to_string(),
            };
        }
    };
    ResolvedTarget {
        requested: name.to_string(),
        gfx: gfx.to_string(),
    }
}

/// A complete bedroc request: a kernel, its targets, and the properties
/// to establish.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BedrocRequest {
    /// Path to the kernel source (informational; used for content
    /// fingerprinting during execution).
    pub source: String,
    /// The kind of source supplied.
    pub source_kind: SourceKind,
    /// Resolved target architectures.
    pub targets: Vec<ResolvedTarget>,
    /// Properties to establish.
    pub goals: Vec<GoalKind>,
}

impl BedrocRequest {
    /// Build a request from raw user inputs, resolving target names and
    /// deduplicating targets/goals while preserving order.
    pub fn build(
        source: impl Into<String>,
        source_kind: SourceKind,
        targets: impl IntoIterator<Item = impl AsRef<str>>,
        goals: impl IntoIterator<Item = GoalKind>,
    ) -> Self {
        let mut resolved = Vec::new();
        let mut seen_gfx = std::collections::BTreeSet::new();
        for t in targets {
            let r = resolve_target(t.as_ref());
            if seen_gfx.insert(r.gfx.clone()) {
                resolved.push(r);
            }
        }
        let mut goal_list = Vec::new();
        let mut seen_goal = std::collections::BTreeSet::new();
        for g in goals {
            if seen_goal.insert(g) {
                goal_list.push(g);
            }
        }
        Self {
            source: source.into(),
            source_kind,
            targets: resolved,
            goals: goal_list,
        }
    }

    /// Every (goal, target) pair as a concrete goal fact.
    pub fn goal_facts(&self) -> Vec<String> {
        let mut facts = Vec::new();
        for g in &self.goals {
            for t in &self.targets {
                facts.push(g.fact_for(&t.gfx));
            }
        }
        facts
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolves_marketing_names() {
        assert_eq!(resolve_target("mi350").gfx, "gfx950");
        assert_eq!(resolve_target("MI300X").gfx, "gfx942");
        assert_eq!(resolve_target("mi450").gfx, "gfx1250");
        // Passthrough for gfx and unknowns.
        assert_eq!(resolve_target("gfx942").gfx, "gfx942");
        assert_eq!(resolve_target("gfx1250").gfx, "gfx1250");
    }

    #[test]
    fn dedups_targets_by_gfx() {
        let req = BedrocRequest::build(
            "k.hip",
            SourceKind::Hip,
            ["mi350", "mi355x", "gfx950"], // all gfx950
            [GoalKind::CorrectOutput],
        );
        assert_eq!(req.targets.len(), 1);
        assert_eq!(req.targets[0].gfx, "gfx950");
    }

    #[test]
    fn goal_facts_cover_cross_product() {
        let req = BedrocRequest::build(
            "k.hip",
            SourceKind::Hip,
            ["mi350", "mi450"],
            [GoalKind::NoDataHazards, GoalKind::CorrectOutput],
        );
        let facts = req.goal_facts();
        assert!(facts.contains(&"no_hazards:gfx950".to_string()));
        assert!(facts.contains(&"correct_output:gfx1250".to_string()));
        assert_eq!(facts.len(), 4);
    }

    #[test]
    fn parses_goal_and_source_aliases() {
        assert_eq!(GoalKind::parse("no-data-hazards"), Some(GoalKind::NoDataHazards));
        assert_eq!(GoalKind::parse("FPSan"), Some(GoalKind::FpCorrect));
        assert_eq!(SourceKind::parse("assembly"), Some(SourceKind::Asm));
        assert!(GoalKind::parse("nonsense").is_none());
    }

    #[test]
    fn codeobject_seeds_compiled_facts() {
        let targets = vec![resolve_target("mi350")];
        let facts = SourceKind::CodeObject.source_facts(&targets);
        assert_eq!(facts, vec!["compiled:gfx950".to_string()]);
    }
}
