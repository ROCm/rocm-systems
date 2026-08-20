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
pub mod presets;
pub mod profiles;
pub mod topologies;

use std::path::PathBuf;

use serde::Serialize;

use mirage_core::error::Result;
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

/// What one pass of `ensure` did to one kind of builtin.
///
/// A report rather than a success-or-error, because the interesting
/// outcome is neither: a builtin the user has edited is left alone, which
/// is a fact about that one document and says nothing about the other
/// forty. Returning it lets the caller finish the other two kinds and
/// then say everything it left alone at once — `mirage state builtins`
/// used to abandon the run at the first one, so repairing three edited
/// builtins took three invocations to even discover.
#[derive(Debug, Default)]
pub struct Ensured {
    /// Every document of this kind, as its name and whether this pass
    /// wrote it.
    pub documents: Vec<(String, bool)>,
    /// The documents that differ from the ones mirage ships and were
    /// therefore left alone, each with the file it lives in. Only ever
    /// non-empty for a forced pass; without `force` an existing document
    /// is left alone whether or not it was edited.
    pub edited: Vec<(String, PathBuf)>,
}

impl Ensured {
    /// Every document this pass considered, as `(name, written)`.
    ///
    /// The report *is* mostly this list — callers that only want to know
    /// which builtins exist should not have to know that it grew a second
    /// field for the ones left alone.
    pub fn iter(&self) -> std::slice::Iter<'_, (String, bool)> {
        self.documents.iter()
    }
}

/// Write all builtin agents to disk. See `ensure` for what `force`
/// does — and does not — allow.
///
/// # Errors
///
/// Returns an error if a document cannot be written. A builtin the user
/// has edited is reported in [`Ensured::edited`], not as an error.
pub fn ensure_agents(force: bool) -> Result<Ensured> {
    ensure(DocKind::Agent, agents(), force)
}

/// Write all builtin topologies to disk. See `ensure`.
///
/// # Errors
///
/// Returns an error if a document cannot be written.
pub fn ensure_topologies(force: bool) -> Result<Ensured> {
    ensure(DocKind::Topology, topologies(), force)
}

/// Write all builtin profiles to disk. See `ensure`.
///
/// # Errors
///
/// Returns an error if a document cannot be written.
pub fn ensure_profiles(force: bool) -> Result<Ensured> {
    ensure(DocKind::Profile, profiles(), force)
}

/// Materialise one kind of builtin, and report what happened to each
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
/// which is what this used to do.
///
/// Leaving one alone is not a failure, and this does not return one. It
/// is the outcome `mirage state builtins --help` describes as ordinary,
/// every other document is still refreshed, and there is nothing for the
/// user to fix unless they want the shipped version back. What they need
/// is to be told which files those are — so they are named in
/// [`Ensured::edited`] and reported by the caller, which is the only
/// place that can name all three kinds in one breath.
fn ensure<T: Serialize>(
    kind: DocKind,
    documents: Vec<(&'static str, T)>,
    force: bool,
) -> Result<Ensured> {
    let mut out = Ensured::default();
    for (name, document) in documents {
        let path = kind.path(name);
        if path.exists() {
            if !force {
                out.documents.push((name.to_string(), false));
                continue;
            }
            if !is_pristine_builtin(kind, name) {
                out.edited.push((name.to_string(), path));
                out.documents.push((name.to_string(), false));
                continue;
            }
        }
        mirage_core::state::write_json(&path, &document)?;
        out.documents.push((name.to_string(), true));
    }
    Ok(out)
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
        assert!(!first.documents.is_empty());
        assert!(
            first.documents.iter().all(|(_, w)| *w),
            "first run should write every builtin"
        );

        let mut names: Vec<String> = first.documents.iter().map(|(n, _)| n.clone()).collect();
        names.sort();
        assert_eq!(mirage_core::agent::store::list().unwrap(), names);

        assert!(
            ensure_agents(false)
                .unwrap()
                .documents
                .iter()
                .all(|(_, w)| !*w),
            "second run should not rewrite existing builtins"
        );
        assert!(
            ensure_agents(true)
                .unwrap()
                .documents
                .iter()
                .all(|(_, w)| *w),
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

        let forced = ensure_profiles(true).unwrap();
        assert_eq!(
            forced
                .edited
                .iter()
                .map(|(n, _)| n.as_str())
                .collect::<Vec<_>>(),
            vec!["mi350x"],
            "the edited builtin must be named"
        );
        assert_eq!(
            mirage_core::store::profile_get("mi350x").unwrap(),
            mine,
            "the user's edits must survive"
        );

        // And it is a report, not a failure: the pass carried on and
        // refreshed everything it safely could. Abandoning the run here
        // was what made repairing three edited builtins take three runs.
        for name in ["mi300x", "mi450x"] {
            assert!(
                forced.documents.contains(&(name.to_string(), true)),
                "{name} should have been refreshed"
            );
        }

        // Deleting the edited one restores the shipped version, which is
        // what the report tells the user to do.
        std::fs::remove_file(mirage_core::paths::profile_path("mi350x")).unwrap();
        let clean = ensure_profiles(true).unwrap();
        assert!(clean.edited.is_empty());
        assert!(clean.documents.iter().all(|(_, w)| *w));
        assert_eq!(
            mirage_core::store::profile_get("mi350x")
                .unwrap()
                .description,
            None
        );

        mirage_core::paths::clear_test_root();
    }

    #[test]
    fn every_edited_builtin_of_a_kind_is_named_by_one_pass() {
        // Only the first offender used to be reported, and the pass then
        // stopped — so the count was wrong as well as short, and each
        // repair revealed the next one.
        let _g = mirage_core::paths::test_env_lock();
        let tmp = tempfile::tempdir().unwrap();
        mirage_core::paths::set_test_root(tmp.path());

        ensure_profiles(false).unwrap();
        for name in ["mi300x", "mi350x", "mi450x"] {
            let mut mine = mirage_core::store::profile_get(name).unwrap();
            mine.description = Some(format!("my own {name}"));
            mirage_core::state::write_json(&mirage_core::paths::profile_path(name), &mine).unwrap();
        }

        let forced = ensure_profiles(true).unwrap();
        let mut named: Vec<&str> = forced.edited.iter().map(|(n, _)| n.as_str()).collect();
        named.sort_unstable();
        assert_eq!(named, vec!["mi300x", "mi350x", "mi450x"]);

        mirage_core::paths::clear_test_root();
    }

    #[test]
    fn ensure_topologies_writes_then_skips() {
        let _g = mirage_core::paths::test_env_lock();
        let tmp = tempfile::tempdir().unwrap();
        mirage_core::paths::set_test_root(tmp.path());

        let first = ensure_topologies(false).unwrap();
        assert!(!first.documents.is_empty());
        assert!(first.documents.iter().all(|(_, w)| *w));
        assert!(first.documents.iter().any(|(n, _)| n == "MI350X-1x1"));

        assert!(
            ensure_topologies(false)
                .unwrap()
                .documents
                .iter()
                .all(|(_, w)| !*w),
            "second run should not rewrite existing builtins"
        );
    }
}
