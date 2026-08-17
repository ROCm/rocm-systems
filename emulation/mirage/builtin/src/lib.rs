//! Built-in agents, topologies and profiles that mirage preloads into
//! `<MIRAGE_CONFIG>/{agent,topology,profile}/`.
//!
//! Historically these shipped as `agents/*.json` files embedded into
//! `mirage_core` at build time and parsed at runtime. They now live
//! here as strongly-typed [`mirage_core::agent::AgentDef`] /
//! [`mirage_core::topology::TopologyDef`] constructors so the data is
//! validated by the compiler instead of by a runtime parse.
//!
//! This crate owns both the builtin *data* and the policy for writing
//! it to disk. It relies on `mirage_core` only for the low-level path
//! resolution ([`mirage_core::paths`]) and JSON serialization
//! ([`mirage_core::state::write_json`]), and registers what it ships with
//! [`mirage_core::store`] so the store can tell a document mirage seeded
//! from one the user wrote.

pub mod agents;
pub mod profiles;
pub mod topologies;

use serde::Serialize;

use mirage_core::error::{MirageError, Result};
use mirage_core::store::{BuiltinDocuments, DocKind, is_pristine_builtin};

pub use agents::{agents, mi300x, mi350x, mi450x};
pub use profiles::profiles;
pub use topologies::{default_topology, topologies};

// Tell `mirage_core` what mirage ships. The core store has to be able to
// answer "did the user write this file, or did we?" — it is the
// difference between a write that destroys somebody's work and one that
// refreshes our own seed. It cannot ask this crate directly (the
// dependency runs the other way), so the answer is registered at link
// time, exactly as emulator backends are.
inventory::submit! {
    BuiltinDocuments { documents: builtin_documents }
}

fn builtin_documents() -> Vec<(DocKind, String, serde_json::Value)> {
    fn collect<T: Serialize>(
        out: &mut Vec<(DocKind, String, serde_json::Value)>,
        kind: DocKind,
        documents: Vec<(&'static str, T)>,
    ) {
        for (name, document) in documents {
            // These are mirage's own structs serialising into a JSON
            // object; the only way `to_value` fails is a type that cannot
            // be represented at all, which none of them is. A builtin
            // that somehow did not serialise is simply not claimed as
            // one, which costs the user nothing but a refusal they would
            // otherwise not have seen.
            if let Ok(value) = serde_json::to_value(&document) {
                out.push((kind, name.to_string(), value));
            }
        }
    }

    let mut out = Vec::new();
    collect(&mut out, DocKind::Agent, agents());
    collect(&mut out, DocKind::Topology, topologies());
    collect(&mut out, DocKind::Profile, profiles());
    out
}

/// Write all builtin agents to disk. See [`ensure`] for what `force`
/// does — and does not — allow.
///
/// # Errors
///
/// Returns an error if a document cannot be written, or if `force` would
/// have had to overwrite an agent the user has edited.
pub fn ensure_agents(force: bool) -> Result<Vec<(String, bool)>> {
    ensure(DocKind::Agent, agents(), force)
}

/// Write all builtin topologies to disk. See [`ensure`].
///
/// # Errors
///
/// Returns an error if a document cannot be written, or if `force` would
/// have had to overwrite a topology the user has edited.
pub fn ensure_topologies(force: bool) -> Result<Vec<(String, bool)>> {
    ensure(DocKind::Topology, topologies(), force)
}

/// Write all builtin profiles to disk. See [`ensure`].
///
/// # Errors
///
/// Returns an error if a document cannot be written, or if `force` would
/// have had to overwrite a profile the user has edited.
pub fn ensure_profiles(force: bool) -> Result<Vec<(String, bool)>> {
    ensure(DocKind::Profile, profiles(), force)
}

/// Materialise one kind of builtin, and report `(name, written)` per
/// document.
///
/// Without `force` — the startup path, run before every command — only
/// missing documents are written, so a fresh config directory fills
/// itself in and an existing one is left exactly as it is.
///
/// With `force` — `mirage state builtins`, which exists so a mirage
/// upgrade can bring its new definitions with it — every document that is
/// missing or still identical to the shipped one is rewritten, and a
/// document the user has *changed* is not. Rewriting that one would
/// discard the only copy of their edits with nothing to say for itself,
/// which is what this used to do. Refusing is loud, names each file, and
/// leaves the user a decision they can act on; the documents that could
/// safely be refreshed are refreshed first, so the upgrade still lands
/// everywhere it can.
fn ensure<T: Serialize>(
    kind: DocKind,
    documents: Vec<(&'static str, T)>,
    force: bool,
) -> Result<Vec<(String, bool)>> {
    let mut report = Vec::new();
    let mut refused = Vec::new();
    for (name, document) in documents {
        let path = kind.path(name);
        if path.exists() {
            if !force {
                report.push((name.to_string(), false));
                continue;
            }
            if !is_pristine_builtin(kind, name) {
                refused.push((name, path));
                report.push((name.to_string(), false));
                continue;
            }
        }
        mirage_core::state::write_json(&path, &document)?;
        report.push((name.to_string(), true));
    }
    if refused.is_empty() {
        return Ok(report);
    }

    let kind = kind.as_str();
    let mut msg = format!(
        "refusing to overwrite {} builtin {kind} document{} you have changed",
        refused.len(),
        if refused.len() == 1 { "" } else { "s" }
    );
    for (name, path) in &refused {
        msg.push_str(&format!("\n  {name} -> {}", path.display()));
    }
    msg.push_str(&format!(
        "\nEach differs from the {kind} mirage ships, so rewriting it would discard \
         your edits. Every other builtin was refreshed. To take the shipped version \
         after all, delete the file and run `mirage state builtins` again."
    ));
    Err(MirageError::other(msg))
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

    use super::*;

    #[test]
    fn ensure_agents_writes_then_skips() {
        let _g = mirage_core::paths::test_env_lock();
        let tmp = tempfile::tempdir().unwrap();
        mirage_core::paths::set_test_root(tmp.path());

        let first = ensure_agents(false).unwrap();
        assert!(!first.is_empty());
        assert!(
            first.iter().all(|(_, w)| *w),
            "first run should write every builtin"
        );

        let mut names: Vec<String> = first.iter().map(|(n, _)| n.clone()).collect();
        names.sort();
        assert_eq!(mirage_core::agent::store::list().unwrap(), names);

        assert!(
            ensure_agents(false).unwrap().iter().all(|(_, w)| !*w),
            "second run should not rewrite existing builtins"
        );
        assert!(
            ensure_agents(true).unwrap().iter().all(|(_, w)| *w),
            "force should rewrite every builtin"
        );

        for name in &names {
            assert!(
                mirage_core::agent::store::get(name).is_ok(),
                "{name} should be readable"
            );
        }
    }

    #[test]
    fn a_forced_rewrite_leaves_an_edited_builtin_alone() {
        // `mirage state builtins` used to overwrite a builtin the user had
        // edited without a word and without a copy — the file was simply
        // gone. It still refreshes everything it safely can; what it will
        // not do any more is discard the one document here that nobody
        // else has a copy of.
        let _g = mirage_core::paths::test_env_lock();
        let tmp = tempfile::tempdir().unwrap();
        mirage_core::paths::set_test_root(tmp.path());

        ensure_profiles(false).unwrap();
        let mut mine = mirage_core::store::profile_get("mi350x").unwrap();
        mine.description = Some("my own mi350x".to_string());
        mirage_core::state::write_json(&mirage_core::paths::profile_path("mi350x"), &mine).unwrap();

        let err = ensure_profiles(true).unwrap_err().to_string();
        assert!(err.contains("mi350x"), "{err}");
        assert!(err.contains("you have changed"), "{err}");
        assert!(err.contains("mirage state builtins"), "{err}");
        assert_eq!(
            mirage_core::store::profile_get("mi350x").unwrap(),
            mine,
            "the user's edits must survive"
        );

        // The builtins that could be refreshed were, before the refusal.
        for name in ["mi300x", "mi450x"] {
            assert!(mirage_core::store::profile_get(name).is_ok(), "{name}");
        }

        // And deleting the edited one restores the shipped version, which
        // is what the refusal tells the user to do.
        std::fs::remove_file(mirage_core::paths::profile_path("mi350x")).unwrap();
        assert!(ensure_profiles(true).unwrap().iter().all(|(_, w)| *w));
        assert_eq!(
            mirage_core::store::profile_get("mi350x")
                .unwrap()
                .description,
            None
        );

        mirage_core::paths::clear_test_root();
    }

    #[test]
    fn ensure_topologies_writes_then_skips() {
        let _g = mirage_core::paths::test_env_lock();
        let tmp = tempfile::tempdir().unwrap();
        mirage_core::paths::set_test_root(tmp.path());

        let first = ensure_topologies(false).unwrap();
        assert!(!first.is_empty());
        assert!(first.iter().all(|(_, w)| *w));
        assert!(first.iter().any(|(n, _)| n == "MI350X-1x1"));

        assert!(
            ensure_topologies(false).unwrap().iter().all(|(_, w)| !*w),
            "second run should not rewrite existing builtins"
        );
    }
}
