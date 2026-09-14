//! Built-in agents, topologies and profiles.
//!
//! Everything here comes from the rocjitsu configs mirage ships. Each
//! `rocjitsu/configs/<gpu>.json` is embedded and validated against
//! [`mirage_core::agent::AgentDef`] at build time (see `build.rs`), and
//! becomes one builtin agent ([`mod@agents`]) and one builtin profile
//! pinning it ([`mod@profiles`]). System layouts ([`mod@topologies`])
//! remain mirage-owned constructors over those agent names.
//!
//! # Two kinds of builtin
//!
//! Agents and topologies are **seeded**: mirage writes them into
//! `<MIRAGE_CONFIG>/{agent,topology}/` and writes back any that goes
//! missing, because they are documents a user edits, references by name,
//! and expects to find as files.
//!
//! Builtin profiles are **generated**: nothing is written, and
//! [`mirage_core::store`] derives one from this crate whenever the
//! profile directory has no file of that name. A profile is three fields
//! around an agent reference, so the file said nothing the agent did not
//! — and a stored copy went stale on upgrade, needing `mirage state
//! builtins` to pick up a definition mirage could just as well produce.
//! Writing your own profile under a builtin's name still shadows it, and
//! deleting that file still brings the builtin back.
//!
//! This crate owns both the builtin *data* and the policy for writing
//! the seeded half to disk. It relies on `mirage_core` only for the
//! low-level path resolution ([`mirage_core::paths`]) and JSON
//! serialization ([`mirage_core::state::write_json`]), and registers what
//! it ships with [`mirage_core::store`] so the store can tell a document
//! mirage seeded from one the user wrote, and can serve a profile mirage
//! never wrote at all.

pub mod agents;
mod presets;
pub mod profiles;
pub mod topologies;

use std::path::PathBuf;

use serde::Serialize;

use mirage_core::error::Result;
use mirage_core::profile::ProfileDef;
use mirage_core::store::{BuiltinDocuments, BuiltinProfiles, DocKind, is_pristine_builtin};

pub use agents::{agent, agents};
pub use profiles::{profile_named, profiles};
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

// And what it generates rather than ships. Registered separately because
// the two answer different questions: a *seeded* document on disk may be
// mirage's own untouched copy, and a generated one has no copy on disk at
// all, so every profile file that exists is the user's.
inventory::submit! {
    BuiltinProfiles { profiles: builtin_profiles }
}

/// The builtin profiles, for [`mirage_core::store`] to serve.
fn builtin_profiles() -> Vec<(String, ProfileDef)> {
    profiles()
        .into_iter()
        .map(|(name, profile)| (name.to_string(), profile))
        .collect()
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
    // Not profiles: mirage writes none, so no profile file is ever
    // mirage's own seed and none of the rules this list feeds applies to
    // one. They are registered as [`BuiltinProfiles`] instead.
    out
}

/// What one pass of `ensure` did to one kind of builtin.
///
/// A report rather than a success-or-error, because the interesting
/// outcome is neither: a builtin the user has edited is left alone, which
/// is a fact about that one document and says nothing about the other
/// forty. Returning it lets the caller finish the other two kinds and
/// then say everything it left alone at once — `mirage state builtins`
/// used to abandon the run at the first one, so repairing an edited
/// agent and an edited topology took two invocations to even discover.
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

/// Materialise one kind of builtin, and report what happened to each
/// document.
///
/// Without `force` — the startup path, run before every command — only
/// missing documents are written, so a fresh config directory fills
/// itself in and an existing one is left exactly as it is. The one
/// exception is [`refresh_agent_mirage_wrote`], which replaces an agent
/// file mirage can prove it wrote itself — a document an earlier release
/// seeded — with the one it ships now, because rocjitsu will not load
/// the older ones at all. It touches nothing else.
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
/// place that can name both kinds in one breath.
fn ensure<T: Serialize>(
    kind: DocKind,
    documents: Vec<(&'static str, T)>,
    force: bool,
) -> Result<Ensured> {
    let mut out = Ensured::default();
    for (name, document) in documents {
        let path = kind.path(name);
        if path.exists() {
            refresh_agent_mirage_wrote(kind, name, &path, &document)?;
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

/// The agent documents previous releases of mirage wrote.
///
/// These cannot be derived from anything: they are what the definitions
/// in `builtin/src/agents.rs` serialised to before mirage read its
/// agents out of the rocjitsu configs, and that code is gone. So they
/// are kept verbatim, generated by building the release that wrote them
/// and serialising its own `agents()`. A file that equals one of these
/// is mirage's own work, however little it now resembles what mirage
/// ships, and [`refresh_agent_mirage_wrote`] is what acts on that.
///
/// The table is deliberately incomplete and appending to it is the
/// intended way to extend it. The serialised shape moved more than once
/// before this — `num_gpus` arrived in v0.1.0, `arch` went `gfx1250` to
/// `cdna5` — and a document from one of those older shapes is simply not
/// recognised: it is left alone, named by `mirage state builtins`, and
/// refused with its reason at profile validation, which is the same
/// place a user's own document ends up. Recognising one more release is
/// a matter of dumping its agents and adding the file.
const SUPERSEDED_AGENTS: [(&str, &str); 3] = [
    ("mi300x", include_str!("../legacy/mi300x.json")),
    ("mi350x", include_str!("../legacy/mi350x.json")),
    ("mi450x", include_str!("../legacy/mi450x.json")),
];

/// Bring a stored agent that mirage itself wrote up to what it ships.
///
/// rocjitsu refuses a device that has SDMA engines and no queues on them
/// — `num_sdma_queues_per_engine must be nonzero when num_sdma_engines
/// is nonzero`, thrown while the config is loaded, which means the
/// session dies at daemon start. Mirage never wrote the field before
/// this release, so every agent file an older mirage left behind is one
/// rocjitsu will now reject — including `mi350x`, which is what
/// `--profile` defaults to, so an upgraded machine that does nothing
/// unusual gets a failed session out of the box.
///
/// Which is still not licence to write the field into every file
/// bearing a builtin's name, because a *pristine* builtin already has
/// it: every document this can ever see is one mirage cannot claim from
/// its contents alone. Filling in just this GPU's queue count is the
/// wrong repair — an MI350X seeded before this release has five SDMA
/// engines and four CUs per shader array, and would come out of it with
/// five engines, today's eight queues each, and its own old shader
/// fabric, describing no GPU either party ships.
///
/// So the question asked is whether the stored document is one mirage
/// wrote, and there are exactly two ways for it to be:
///
/// * it is the shipped document with only the queue count missing, which
///   is what an install that last ran *this* release's predecessor by a
///   hair looks like; or
/// * it equals one of [`SUPERSEDED_AGENTS`], the documents earlier
///   releases wrote.
///
/// Either way the answer is the whole shipped document, not a patched
/// copy of the old one — the same file a fresh install gets, which is
/// the only version mirage can vouch for. Anything else is somebody's
/// own document and is left exactly as it is: `mirage state builtins`
/// names it and says how to take the shipped version, and the rocjitsu
/// backend refuses an incomplete pair at profile validation, where the
/// document can still be edited.
fn refresh_agent_mirage_wrote<T: Serialize>(
    kind: DocKind,
    name: &str,
    path: &std::path::Path,
    document: &T,
) -> Result<()> {
    if !matches!(kind, DocKind::Agent) {
        return Ok(());
    }
    let Ok(shipped) = serde_json::to_value(document) else {
        return Ok(());
    };
    let Ok(stored) = mirage_core::state::read_json::<serde_json::Value>(path) else {
        return Ok(());
    };
    if stored == shipped {
        return Ok(());
    }
    if !completes_to_shipped(&stored, &shipped) && !is_superseded_agent(name, &stored) {
        return Ok(());
    }
    mirage_core::state::write_json(path, &shipped)
}

/// Is `stored` the shipped document with only its queue count missing?
fn completes_to_shipped(stored: &serde_json::Value, shipped: &serde_json::Value) -> bool {
    // A shipped agent whose own config omits the field has no value to
    // complete anything with.
    let Some(queues) = shipped
        .pointer("/vm/gpu/device/num_sdma_queues_per_engine")
        .cloned()
    else {
        return false;
    };
    let mut candidate = stored.clone();
    let Some(device) = candidate
        .pointer_mut("/vm/gpu/device")
        .and_then(serde_json::Value::as_object_mut)
    else {
        return false;
    };
    // An explicit zero is an answer, and mirage does not argue with it —
    // rocjitsu will.
    if device.contains_key("num_sdma_queues_per_engine") {
        return false;
    }
    device.insert("num_sdma_queues_per_engine".to_string(), queues);
    candidate == *shipped
}

/// Does `stored` equal the document some earlier release wrote for
/// `name`?
///
/// Compared as parsed JSON rather than as text, so a file that has been
/// reformatted — or written by a serialiser that ordered its keys
/// differently — is still recognised as the document it is.
fn is_superseded_agent(name: &str, stored: &serde_json::Value) -> bool {
    SUPERSEDED_AGENTS
        .iter()
        .filter(|(superseded, _)| superseded.eq_ignore_ascii_case(name))
        .any(|(_, document)| {
            serde_json::from_str::<serde_json::Value>(document)
                .is_ok_and(|superseded| superseded == *stored)
        })
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

    /// The document mirage may complete is the shipped one with that
    /// single key taken out, and completing it produces the shipped one
    /// whole.
    #[test]
    fn a_shipped_agent_missing_only_the_queue_count_is_completed() {
        let _guard = mirage_core::paths::test_env_lock();
        let tmp = tempfile::tempdir().unwrap();
        mirage_core::paths::set_test_root(tmp.path());

        for (name, agent) in agents() {
            let path = mirage_core::paths::agent_path(name);
            let shipped = serde_json::to_value(&agent).unwrap();
            let mut stored = shipped.clone();
            let device = stored["vm"]["gpu"]["device"].as_object_mut().unwrap();
            // An agent whose own rocjitsu config omits the field has
            // nothing to complete and nothing to say here.
            if device.remove("num_sdma_queues_per_engine").is_none() {
                continue;
            }
            mirage_core::state::write_json(&path, &stored).unwrap();

            ensure_agents(false).unwrap();
            let completed = mirage_core::state::read_json::<serde_json::Value>(&path).unwrap();
            assert_eq!(completed, shipped, "{name}");

            // Idempotent: a second pass has nothing left to do.
            ensure_agents(false).unwrap();
            assert_eq!(
                mirage_core::state::read_json::<serde_json::Value>(&path).unwrap(),
                shipped,
                "{name}"
            );
        }

        mirage_core::paths::clear_test_root();
    }

    /// An agent an earlier release wrote becomes the one mirage ships.
    ///
    /// This is the upgrade, and the documents are the real ones: the
    /// previous release's `mi350x` has five SDMA engines, four CUs per
    /// shader array and no queue count at all, against a shipped MI355X
    /// of two engines and eight queues each. Patching the one field into
    /// it would leave a machine neither party ships, and leaving it
    /// alone leaves `--profile`'s own default unable to start a session,
    /// so the whole document is replaced.
    #[test]
    fn an_agent_an_earlier_release_wrote_is_replaced_with_the_shipped_one() {
        let _guard = mirage_core::paths::test_env_lock();
        let tmp = tempfile::tempdir().unwrap();
        mirage_core::paths::set_test_root(tmp.path());

        for (name, superseded) in SUPERSEDED_AGENTS {
            let shipped = agents()
                .into_iter()
                .find(|(shipped, _)| shipped.eq_ignore_ascii_case(name))
                .map(|(_, agent)| serde_json::to_value(&agent).unwrap())
                .unwrap_or_else(|| panic!("mirage still ships an agent called {name}"));
            let superseded: serde_json::Value = serde_json::from_str(superseded).unwrap();
            assert_ne!(
                superseded, shipped,
                "{name}: a fixture equal to the shipped agent tests nothing"
            );

            let path = mirage_core::paths::agent_path(name);
            mirage_core::state::write_json(&path, &superseded).unwrap();

            ensure_agents(false).unwrap();
            assert_eq!(
                mirage_core::state::read_json::<serde_json::Value>(&path).unwrap(),
                shipped,
                "{name}"
            );

            // Idempotent: a second pass has nothing left to do.
            ensure_agents(false).unwrap();
            assert_eq!(
                mirage_core::state::read_json::<serde_json::Value>(&path).unwrap(),
                shipped,
                "{name}"
            );
        }

        mirage_core::paths::clear_test_root();
    }

    /// A document mirage cannot claim is not touched at all.
    ///
    /// Recognising the documents earlier releases wrote is not licence
    /// to guess: one changed field is enough to make a file the user's,
    /// and a file that merely *resembles* an older release is not one.
    #[test]
    fn an_agent_that_is_not_one_mirage_wrote_is_left_alone() {
        let _guard = mirage_core::paths::test_env_lock();
        let tmp = tempfile::tempdir().unwrap();
        mirage_core::paths::set_test_root(tmp.path());

        for (name, agent) in agents() {
            let path = mirage_core::paths::agent_path(name);
            let shipped = serde_json::to_value(&agent).unwrap();

            // The user's own edit of the shipped document.
            let mut edited = shipped.clone();
            let device = edited["vm"]["gpu"]["device"].as_object_mut().unwrap();
            device.remove("num_sdma_queues_per_engine");
            device.insert("marketing_name".to_string(), "custom GPU".into());

            // A document shaped like an older release's but not one of
            // them, which is also what every unrecognised older shape
            // looks like from here.
            let mut invented = shipped.clone();
            let device = invented["vm"]["gpu"]["device"].as_object_mut().unwrap();
            device.remove("num_sdma_queues_per_engine");
            device.insert("num_sdma_engines".to_string(), 5.into());
            device.insert("num_cu_per_sh".to_string(), 4.into());
            invented["topology"]["root"]["children"] = serde_json::json!([]);

            let mut theirs = vec![edited, invented];

            // The previous release's own document with a single field
            // changed: recognition has to be exact, so this is theirs.
            if let Some((_, superseded)) = SUPERSEDED_AGENTS
                .iter()
                .find(|(superseded, _)| superseded.eq_ignore_ascii_case(name))
            {
                let mut nearly: serde_json::Value = serde_json::from_str(superseded).unwrap();
                nearly["vm"]["gpu"]["device"]["num_cp_queues"] = 7.into();
                theirs.push(nearly);
            }

            for document in theirs {
                mirage_core::state::write_json(&path, &document).unwrap();
                for force in [false, true] {
                    ensure_agents(force).unwrap();
                    assert_eq!(
                        mirage_core::state::read_json::<serde_json::Value>(&path).unwrap(),
                        document,
                        "{name} (force={force})"
                    );
                }
            }
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

        ensure_topologies(false).unwrap();
        let mut mine = mirage_core::store::topology_get("MI350X-1x8").unwrap();
        mine.num_nodes = 7;
        mirage_core::state::write_json(&mirage_core::paths::topology_path("MI350X-1x8"), &mine)
            .unwrap();

        let forced = ensure_topologies(true).unwrap();
        assert_eq!(
            forced
                .edited
                .iter()
                .map(|(n, _)| n.as_str())
                .collect::<Vec<_>>(),
            vec!["MI350X-1x8"],
            "the edited builtin must be named"
        );
        assert_eq!(
            mirage_core::store::topology_get("MI350X-1x8").unwrap(),
            mine,
            "the user's edits must survive"
        );

        // And it is a report, not a failure: the pass carried on and
        // refreshed everything it safely could. Abandoning the run here
        // was what made repairing several edited builtins take one run each.
        for name in ["MI350X-1x1", "MI350X-2x8", "MI300X-1x8"] {
            assert!(
                forced.documents.contains(&(name.to_string(), true)),
                "{name} should have been refreshed"
            );
        }

        // Deleting the edited one restores the shipped version, which is
        // what the report tells the user to do.
        std::fs::remove_file(mirage_core::paths::topology_path("MI350X-1x8")).unwrap();
        let clean = ensure_topologies(true).unwrap();
        assert!(clean.edited.is_empty());
        assert!(clean.documents.iter().all(|(_, w)| *w));
        assert_eq!(
            mirage_core::store::topology_get("MI350X-1x8")
                .unwrap()
                .num_nodes,
            1
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

        ensure_topologies(false).unwrap();
        let edited = ["MI300X-1x8", "MI350X-1x1", "MI350X-1x8"];
        for name in edited {
            let mut mine = mirage_core::store::topology_get(name).unwrap();
            mine.num_nodes = 7;
            mirage_core::state::write_json(&mirage_core::paths::topology_path(name), &mine)
                .unwrap();
        }

        let forced = ensure_topologies(true).unwrap();
        let mut named: Vec<&str> = forced.edited.iter().map(|(n, _)| n.as_str()).collect();
        named.sort_unstable();
        assert_eq!(named, edited.to_vec());

        mirage_core::paths::clear_test_root();
    }

    /// A builtin profile is usable without ever being written, and
    /// stays unwritten however many commands run.
    ///
    /// This is the whole point of generating them: `mirage run --profile
    /// mi350x` on a fresh machine, and on one whose config directory
    /// mirage cannot write, has to find `mi350x`. Seeding them meant a
    /// directory that could not be written read as a machine with no
    /// profiles at all — `profile list` printed nothing and exited 0.
    #[test]
    fn builtin_profiles_are_served_without_being_stored() {
        let _g = mirage_core::paths::test_env_lock();
        let tmp = tempfile::tempdir().unwrap();
        mirage_core::paths::set_test_root(tmp.path());

        let shipped: Vec<String> = profiles().iter().map(|(n, _)| (*n).to_string()).collect();
        assert!(shipped.contains(&"mi350x".to_string()));

        // Every startup pass there is, and then the whole set is still
        // readable and still not on disk.
        ensure_agents(false).unwrap();
        ensure_topologies(false).unwrap();

        assert_eq!(mirage_core::store::profile_list().unwrap(), shipped);
        for name in &shipped {
            let profile = mirage_core::store::profile_get(name).unwrap();
            assert_eq!(&profile.name, name);
            assert!(
                !mirage_core::paths::profile_path(name).exists(),
                "{name} must not have been written"
            );
        }
        assert!(!mirage_core::paths::profile_root().exists());

        mirage_core::paths::clear_test_root();
    }

    /// Writing over a builtin's name shadows it; deleting that file
    /// brings the builtin back, with nothing to re-seed.
    #[test]
    fn a_profile_file_shadows_the_builtin_of_that_name() {
        let _g = mirage_core::paths::test_env_lock();
        let tmp = tempfile::tempdir().unwrap();
        mirage_core::paths::set_test_root(tmp.path());

        let mut mine = mirage_core::store::profile_get("mi350x").unwrap();
        mine.description = Some("my own mi350x".to_string());
        mirage_core::state::write_json(&mirage_core::paths::profile_path("mi350x"), &mine).unwrap();

        assert_eq!(mirage_core::store::profile_get("mi350x").unwrap(), mine);
        // Shadowed, not duplicated.
        let listed = mirage_core::store::profile_list().unwrap();
        assert_eq!(listed.iter().filter(|n| *n == "mi350x").count(), 1);

        mirage_core::store::profile_delete("mi350x").unwrap();
        assert_ne!(mirage_core::store::profile_get("mi350x").unwrap(), mine);
        assert!(
            mirage_core::store::profile_get("mi350x")
                .unwrap()
                .description
                .is_some()
        );

        // And the builtin itself cannot be deleted: there is no file, and
        // saying "deleted" would be a lie.
        let error = mirage_core::store::profile_delete("mi350x").unwrap_err();
        assert!(error.to_string().contains("builtin"), "{error}");

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
