//! Built-in agents, topologies and profiles that mirage preloads into
//! `<MIRAGE_CONFIG>/{agent,topology,profile}/`.
//!
//! Agent hardware and component topologies are embedded from RocJITsu's
//! configs and validated against [`mirage_core::agent::AgentDef`] at build
//! time. System layouts and profiles remain Mirage-owned constructors.
//!
//! This crate owns both the builtin *data* and the policy for writing
//! it to disk. It relies on `mirage_core` only for the low-level path
//! resolution ([`mirage_core::paths`]) and JSON serialization
//! ([`mirage_core::state::write_json`]), and registers what it ships with
//! [`mirage_core::store`] so the store can tell a document mirage seeded
//! from one the user wrote.

pub mod agents;
mod presets;
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
/// itself in and an existing one is left exactly as it is. The one
/// exception is [`repair_sdma_queue_count`], which fills in a single
/// field an older mirage never wrote and without which rocjitsu refuses
/// to start at all.
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
            repair_sdma_queue_count(kind, &path, &document)?;
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

/// Give an on-disk agent the SDMA queue count rocjitsu now requires.
///
/// rocjitsu refuses a device that has SDMA engines and no queues on them
/// — `num_sdma_queues_per_engine must be nonzero when num_sdma_engines
/// is nonzero`, thrown while the config is loaded, which means the
/// session dies at daemon start. Mirage never wrote the field before
/// this release, so *every* agent file an older mirage left behind is
/// one rocjitsu will now reject.
///
/// Those files cannot simply be replaced with the shipped definition.
/// An older builtin and a builtin the user has edited look alike from
/// here, and the second is theirs to keep — mirage's standing promise
/// (see `is_pristine_builtin`) is that it never overwrites a document
/// somebody changed. So only the one field that makes the file unusable
/// is filled in, from the shipped definition, and only when it is
/// absent: an explicit zero is an answer, and mirage does not argue
/// with it.
fn repair_sdma_queue_count<T: Serialize>(
    kind: DocKind,
    path: &std::path::Path,
    document: &T,
) -> Result<()> {
    if !matches!(kind, DocKind::Agent) {
        return Ok(());
    }
    let Some(queues) = serde_json::to_value(document)
        .ok()
        .as_ref()
        .and_then(|shipped| shipped.pointer("/vm/gpu/device/num_sdma_queues_per_engine"))
        .and_then(serde_json::Value::as_u64)
        .filter(|queues| *queues > 0)
    else {
        return Ok(());
    };
    let Ok(mut stored) = mirage_core::state::read_json::<serde_json::Value>(path) else {
        return Ok(());
    };
    let Some(device) = stored
        .pointer_mut("/vm/gpu/device")
        .and_then(serde_json::Value::as_object_mut)
    else {
        return Ok(());
    };
    // An omitted engine count reads as zero rather than as rocjitsu's
    // schema default of 2, so an agent that names neither field is left
    // exactly as the user has it. Nothing is lost by that: the rocjitsu
    // backend settles the same pair when it synthesises the config, and
    // a document that says nothing about SDMA is not one this repair
    // has a value for — it exists for a stored agent whose engine count
    // an older mirage wrote and whose queue count it did not.
    let engines = device
        .get("num_sdma_engines")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0);
    if engines == 0 || device.contains_key("num_sdma_queues_per_engine") {
        return Ok(());
    }
    device.insert("num_sdma_queues_per_engine".to_string(), queues.into());
    mirage_core::state::write_json(path, &stored)
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

    /// An explicit zero is the user's answer, and mirage does not
    /// second-guess it — even though rocjitsu will refuse the file.
    #[test]
    fn an_explicit_sdma_queue_count_is_left_alone() {
        let _guard = mirage_core::paths::test_env_lock();
        let tmp = tempfile::tempdir().unwrap();
        mirage_core::paths::set_test_root(tmp.path());

        for (name, agent) in agents() {
            let path = mirage_core::paths::agent_path(name);
            let mut edited = serde_json::to_value(&agent).unwrap();
            edited["vm"]["gpu"]["device"]["num_sdma_queues_per_engine"] = 0.into();
            mirage_core::state::write_json(&path, &edited).unwrap();
            for force in [false, true] {
                ensure_agents(force).unwrap();
                assert_eq!(
                    mirage_core::state::read_json::<serde_json::Value>(&path).unwrap(),
                    edited
                );
            }
        }

        mirage_core::paths::clear_test_root();
    }

    /// The repair is one field wide.
    ///
    /// An agent an older mirage wrote is not the shipped one minus a
    /// key — it is a different document, with its own topology and its
    /// own device identity, and it may be one the user has since
    /// edited. Both have to survive; only the field rocjitsu refuses to
    /// start without is filled in.
    #[test]
    fn a_missing_sdma_queue_count_is_filled_in_without_touching_anything_else() {
        let _guard = mirage_core::paths::test_env_lock();
        let tmp = tempfile::tempdir().unwrap();
        mirage_core::paths::set_test_root(tmp.path());

        for (name, agent) in agents() {
            let path = mirage_core::paths::agent_path(name);
            let shipped = serde_json::to_value(&agent).unwrap();
            let mut legacy = shipped.clone();
            let device = legacy["vm"]["gpu"]["device"].as_object_mut().unwrap();
            device.remove("num_sdma_queues_per_engine");
            // Nothing a released mirage wrote looks like the shipped
            // document: the tree and the identity moved too.
            device.insert("marketing_name".to_string(), "custom GPU".into());
            legacy["topology"]["root"]["children"] = serde_json::json!([]);
            mirage_core::state::write_json(&path, &legacy).unwrap();

            ensure_agents(false).unwrap();

            let mut expected = legacy.clone();
            expected["vm"]["gpu"]["device"]["num_sdma_queues_per_engine"] =
                shipped["vm"]["gpu"]["device"]["num_sdma_queues_per_engine"].clone();
            let repaired = mirage_core::state::read_json::<serde_json::Value>(&path).unwrap();
            assert_eq!(repaired, expected, "{name}");
            assert!(
                repaired["vm"]["gpu"]["device"]["num_sdma_queues_per_engine"]
                    .as_u64()
                    .is_some_and(|queues| queues > 0),
                "{name}"
            );

            // Idempotent: a second pass has nothing left to do.
            ensure_agents(false).unwrap();
            assert_eq!(
                mirage_core::state::read_json::<serde_json::Value>(&path).unwrap(),
                expected,
                "{name}"
            );
        }

        mirage_core::paths::clear_test_root();
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
