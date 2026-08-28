//! The workspace lint policy and this crate's copy of it must not drift.
//!
//! `rocjitsu_sys` is the one crate that cannot inherit `[workspace.lints]`
//! with `workspace = true`: the workspace *forbids* `unsafe_code`, and
//! `forbid` cannot be relaxed by a later `allow`, while this crate is
//! where every `unsafe` in mirage lives. So it carries its own table —
//! and a hand-copied table is a list that goes stale. Clippy cannot
//! notice, either: a *missing* lint produces no diagnostic, so the crate
//! holding all the unsafe code would quietly become the least-linted one
//! in the workspace.
//!
//! This test derives the answer instead of trusting the copy. It reads
//! both manifests and asserts that every lint the workspace sets is also
//! set here, at least as strictly, with exactly one exception:
//! `unsafe_code`, which is the whole reason this crate opts out.
//!
//! Both manifests are found from `CARGO_MANIFEST_DIR`, so the test moves
//! with the crate. The scan below is a few lines of string handling
//! rather than a `toml` dependency: this crate's dependency list is part
//! of what makes it auditable — it is the FFI boundary — and one test's
//! convenience is not worth a parser in it.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::collections::BTreeMap;
use std::path::PathBuf;

/// A lint level, ordered from most permissive to most severe so that
/// "the local table may be stricter" is a comparison rather than a
/// special case.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Level {
    Allow,
    Warn,
    Deny,
    Forbid,
}

impl Level {
    fn parse(name: &str) -> Option<Self> {
        match name {
            "allow" => Some(Self::Allow),
            "warn" => Some(Self::Warn),
            "deny" => Some(Self::Deny),
            "forbid" => Some(Self::Forbid),
            _ => None,
        }
    }

    /// The spelling a manifest uses, so a failure message can be pasted
    /// straight into one.
    fn as_str(self) -> &'static str {
        match self {
            Self::Allow => "allow",
            Self::Warn => "warn",
            Self::Deny => "deny",
            Self::Forbid => "forbid",
        }
    }
}

/// The lint that is deliberately not copied, and the only key this test
/// permits the local table to be missing.
const EXEMPT: &str = "unsafe_code";

/// Read the `key = level` pairs of one manifest section.
///
/// Values come in two shapes here — `"deny"` and
/// `{ level = "deny", priority = -1 }` — and both reduce to a level;
/// anything else is a shape this test has not been taught and is
/// reported rather than skipped, because silently ignoring a line is how
/// the drift this test exists to catch would get through it.
fn lint_levels(manifest: &str, section: &str) -> BTreeMap<String, Level> {
    let mut out = BTreeMap::new();
    let mut here = false;
    for line in manifest.lines() {
        let line = line.trim();
        if line.starts_with('#') || line.is_empty() {
            continue;
        }
        if let Some(header) = line.strip_prefix('[').and_then(|l| l.strip_suffix(']')) {
            here = header == section;
            continue;
        }
        if !here {
            continue;
        }
        let Some((key, value)) = line.split_once('=') else {
            continue;
        };
        let key = key.trim();
        let value = value.split('#').next().unwrap_or(value).trim();
        // `key = "deny"`, or an inline table whose `level = "deny"`.
        let quoted = value
            .rsplit_once("level")
            .map_or(value, |(_, rest)| rest)
            .trim_start_matches(['=', ' '])
            .split('"')
            .nth(1);
        let level = quoted
            .and_then(Level::parse)
            .unwrap_or_else(|| panic!("[{section}] {key}: cannot read a lint level from {value}"));
        out.insert(key.to_string(), level);
    }
    out
}

fn manifests() -> (String, String) {
    let crate_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let workspace_dir = crate_dir
        .parent()
        .expect("the crate directory is inside the workspace");
    let read = |dir: &std::path::Path| {
        let path = dir.join("Cargo.toml");
        std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
    };
    (read(workspace_dir), read(&crate_dir))
}

/// Every lint the workspace sets is set here too, at least as strictly.
///
/// The failure this catches is a lint added to `[workspace.lints]` and
/// not to `[lints]` in `rocjitsu_sys/Cargo.toml`, which exempts the one
/// crate that most needs it and produces no diagnostic anywhere.
#[test]
fn the_local_lint_table_covers_the_workspace_policy() {
    let (workspace, local) = manifests();

    for (tool, workspace_section, local_section) in [
        ("rust", "workspace.lints.rust", "lints.rust"),
        ("clippy", "workspace.lints.clippy", "lints.clippy"),
    ] {
        let expected = lint_levels(&workspace, workspace_section);
        let actual = lint_levels(&local, local_section);
        assert!(
            !expected.is_empty(),
            "[{workspace_section}] is empty or was not found; this test is \
             reading the wrong manifest"
        );

        for (lint, level) in &expected {
            if lint == EXEMPT {
                assert!(
                    !actual.contains_key(lint),
                    "`{EXEMPT}` is the one lint this crate opts out of, and \
                     [{local_section}] sets it anyway — every `unsafe` in \
                     mirage is in this crate"
                );
                continue;
            }
            let Some(mine) = actual.get(lint) else {
                panic!(
                    "the workspace sets `{lint} = \"{}\"` for every crate but \
                     [{local_section}] does not, so the one crate holding \
                     mirage's `unsafe` is exempt from it. Copy it into \
                     rocjitsu_sys/Cargo.toml (or, if it must not apply here, \
                     say so next to `{EXEMPT}` and teach this test the second \
                     exception).",
                    level.as_str()
                );
            };
            assert!(
                mine >= level,
                "the workspace sets `{lint}` to {} and [{local_section}] \
                 relaxes it to {}. The local table may be stricter than the \
                 workspace policy, never looser.",
                level.as_str(),
                mine.as_str()
            );
        }

        // The exemption is deliberate, so it has to still be there to be
        // exempted: if the workspace stops forbidding `unsafe_code`, this
        // whole opt-out — and the hand-copied table with it — is dead
        // weight that should be replaced by `workspace = true`.
        if tool == "rust" {
            assert_eq!(
                expected.get(EXEMPT),
                Some(&Level::Forbid),
                "this crate copies the lint table only because the workspace \
                 forbids `{EXEMPT}` and `forbid` cannot be relaxed. If that is \
                 no longer true, inherit the table with `workspace = true` \
                 instead and delete this test."
            );
        }
    }
}

/// The scan itself, on the shapes the manifests actually use — so a
/// failure above is a real drift and not this test misreading a line.
#[test]
fn lint_levels_reads_both_value_shapes() {
    let manifest = "\
[workspace.lints.clippy]
# a comment
all = { level = \"deny\", priority = -1 }
unwrap_used = \"deny\"
undocumented_unsafe_blocks = \"allow\"  # trailing comment

[workspace.dependencies]
unwrap_used = \"1.0\"
";
    let levels = lint_levels(manifest, "workspace.lints.clippy");
    assert_eq!(levels.get("all"), Some(&Level::Deny));
    assert_eq!(levels.get("unwrap_used"), Some(&Level::Deny));
    assert_eq!(
        levels.get("undocumented_unsafe_blocks"),
        Some(&Level::Allow)
    );
    assert_eq!(levels.len(), 3, "a later section leaked into the scan");
    assert!(Level::Warn < Level::Deny && Level::Deny < Level::Forbid);
}
