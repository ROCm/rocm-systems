//! The on-disk configuration store: profiles, topologies and agents.
//!
//! These are user-authored documents, not session state. They live in the
//! config directory, they outlive every process, and reading or writing
//! one involves nothing but the filesystem.
//!
//! That last point is why this module exists separately from the
//! supervisor. Both the daemon and the CLI answer configuration queries
//! from here directly, so `mirage profile list` does not have to start a
//! background daemon to read a directory. There is no cache to keep
//! coherent: the daemon reads a profile off disk when a session is
//! created, so a profile written by the CLI a moment earlier is already
//! visible to it.

use crate::agent::AgentDef;
use crate::error::{MirageError, Result};
use crate::profile::ProfileDef;
use crate::topology::TopologyDef;

/// List every profile name, sorted.
///
/// # Errors
///
/// Returns an error if the profile directory exists but cannot be read.
pub fn profile_list() -> Result<Vec<String>> {
    list_json_stems(&crate::paths::profile_root())
}

/// Read one profile.
///
/// # Errors
///
/// Returns [`MirageError::ProfileNotFound`] if there is no such profile,
/// or a parse error if it is malformed.
pub fn profile_get(name: &str) -> Result<ProfileDef> {
    let path = crate::paths::profile_path(name);
    if !path.exists() {
        return Err(MirageError::ProfileNotFound(name.to_string()));
    }
    crate::state::read_json(&path)
}

/// Write a profile, overwriting any existing one with the same name.
///
/// # Errors
///
/// Returns an error if the document cannot be written.
pub fn profile_put(profile: &ProfileDef) -> Result<()> {
    // Names are case-insensitive and stored lowercase, so the document
    // agrees with the path it lives at.
    let mut profile = profile.clone();
    profile.name = profile.name.to_lowercase();
    crate::state::write_json(&crate::paths::profile_path(&profile.name), &profile)
}

/// Delete a profile.
///
/// # Errors
///
/// Returns [`MirageError::ProfileNotFound`] if there is no such profile.
pub fn profile_delete(name: &str) -> Result<()> {
    let path = crate::paths::profile_path(name);
    if !path.exists() {
        return Err(MirageError::ProfileNotFound(name.to_string()));
    }
    std::fs::remove_file(&path).map_err(|e| MirageError::io(path, e))
}

/// List every topology name, sorted.
///
/// # Errors
///
/// Returns an error if the topology directory cannot be read.
pub fn topology_list() -> Result<Vec<String>> {
    crate::topology::store::list()
}

/// Read one topology.
///
/// # Errors
///
/// Returns an error if there is no such topology, or it is malformed.
pub fn topology_get(name: &str) -> Result<TopologyDef> {
    crate::topology::store::get(name)
}

/// Write a topology under `name`, overwriting any existing one.
///
/// # Errors
///
/// Returns an error if the document cannot be written.
pub fn topology_put(name: &str, topology: &TopologyDef) -> Result<()> {
    crate::topology::store::put(name, topology).map(|_| ())
}

/// Delete a topology.
///
/// # Errors
///
/// Returns an error if there is no such topology.
pub fn topology_delete(name: &str) -> Result<()> {
    let path = crate::paths::topology_path(name);
    if !path.exists() {
        return Err(MirageError::other(format!("topology not found: {name}")));
    }
    std::fs::remove_file(&path).map_err(|e| MirageError::io(path, e))
}

/// List every agent name, sorted.
///
/// # Errors
///
/// Returns an error if the agent directory cannot be read.
pub fn agent_list() -> Result<Vec<String>> {
    crate::agent::store::list()
}

/// Read one agent.
///
/// # Errors
///
/// Returns an error if there is no such agent, or it is malformed.
pub fn agent_get(name: &str) -> Result<AgentDef> {
    crate::agent::store::get(name)
}

/// Write an agent under `name`, overwriting any existing one.
///
/// # Errors
///
/// Returns an error if the document cannot be written.
pub fn agent_put(name: &str, agent: &AgentDef) -> Result<()> {
    crate::agent::store::put(name, agent).map(|_| ())
}

/// Delete an agent.
///
/// # Errors
///
/// Returns an error if there is no such agent.
pub fn agent_delete(name: &str) -> Result<()> {
    let path = crate::paths::agent_path(name);
    if !path.exists() {
        return Err(MirageError::other(format!("agent not found: {name}")));
    }
    std::fs::remove_file(&path).map_err(|e| MirageError::io(path, e))
}

/// The `.json` stems in a directory, sorted.
///
/// A missing directory reads as empty rather than as an error: it just
/// means nothing of that kind has been written yet, which is the normal
/// state of a fresh machine.
fn list_json_stems(root: &std::path::Path) -> Result<Vec<String>> {
    if !root.exists() {
        return Ok(Vec::new());
    }
    let mut out = Vec::new();
    for entry in std::fs::read_dir(root).map_err(|e| MirageError::io(root, e))? {
        let entry = entry.map_err(|e| MirageError::io(root, e))?;
        let name = entry.file_name().to_string_lossy().into_owned();
        if let Some(stem) = name.strip_suffix(".json") {
            out.push(stem.to_string());
        }
    }
    out.sort();
    Ok(out)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;
    use crate::common::MaybeRef;
    use crate::emulator::{EmulatorDef, ExecMode};

    fn profile(name: &str) -> ProfileDef {
        ProfileDef {
            name: name.to_string(),
            description: None,
            emulator: EmulatorDef {
                emulator: "rocjitsu".to_string(),
                plugins: Default::default(),
                exec_mode: ExecMode::Functional,
                options: Default::default(),
                topology: MaybeRef::Ref("t".to_string()),
            },
            containerize: None,
        }
    }

    #[test]
    fn profiles_round_trip() {
        let _g = crate::paths::test_env_lock();
        let dir = tempfile::tempdir().unwrap();
        crate::paths::set_test_root(dir.path());

        assert!(profile_list().unwrap().is_empty());
        profile_put(&profile("a")).unwrap();
        profile_put(&profile("b")).unwrap();
        assert_eq!(profile_list().unwrap(), vec!["a", "b"]);
        assert_eq!(profile_get("a").unwrap().name, "a");

        profile_delete("a").unwrap();
        assert_eq!(profile_list().unwrap(), vec!["b"]);
        assert!(matches!(
            profile_get("a"),
            Err(MirageError::ProfileNotFound(_))
        ));
        crate::paths::clear_test_root();
    }

    #[test]
    fn profile_names_are_case_insensitive() {
        let _g = crate::paths::test_env_lock();
        let dir = tempfile::tempdir().unwrap();
        crate::paths::set_test_root(dir.path());

        profile_put(&profile("MixedCase")).unwrap();
        // Stored lowercase, so the document and its path agree and a
        // lookup under any casing finds it.
        assert_eq!(profile_list().unwrap(), vec!["mixedcase"]);
        assert_eq!(profile_get("MIXEDCASE").unwrap().name, "mixedcase");
        crate::paths::clear_test_root();
    }

    #[test]
    fn a_missing_directory_lists_as_empty() {
        let _g = crate::paths::test_env_lock();
        let dir = tempfile::tempdir().unwrap();
        crate::paths::set_test_root(dir.path());
        // A fresh machine has written nothing yet; that is not an error.
        assert!(profile_list().unwrap().is_empty());
        assert!(agent_list().unwrap().is_empty());
        crate::paths::clear_test_root();
    }

    #[test]
    fn deleting_something_absent_reports_not_found() {
        let _g = crate::paths::test_env_lock();
        let dir = tempfile::tempdir().unwrap();
        crate::paths::set_test_root(dir.path());
        assert!(profile_delete("ghost").is_err());
        assert!(topology_delete("ghost").is_err());
        assert!(agent_delete("ghost").is_err());
        crate::paths::clear_test_root();
    }
}
