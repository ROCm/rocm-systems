//! Session: the long-lived context within which execs run.
//!
//! A session is owned in memory by the supervisor daemon. The types here
//! are the *wire* shapes clients see — a definition, a health snapshot,
//! and the aggregate [`SessionState`] — plus [`SessionContext`], the
//! resolved bundle an emulator backend needs to do its job.

use std::path::PathBuf;
use std::str::FromStr;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{common::MaybeRef, profile::ProfileDef};

/// A session identifier.
///
/// Mirage uses ids as on-disk directory names; they must be safe across
/// common filesystems and shells. The rules are:
///
/// * length: 1..=64 characters
/// * allowed characters: ascii alphanumeric, `-`, `_`, `.`
/// * may not start with `.`
/// * may not contain `..`
///
/// Use [`SessionId::new`] to validate or [`SessionId::generate`] to
/// auto-generate a timestamp-based id.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub struct SessionId(String);

/// Why a session or exec id was rejected.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum IdError {
    /// The id was the empty string.
    #[error("id may not be empty")]
    Empty,
    /// The id exceeded the 64-character limit.
    #[error("id must be at most 64 characters")]
    Length,
    /// The id contained a character outside `[A-Za-z0-9._-]`.
    #[error("id contains invalid character: {0:?}")]
    Char(char),
    /// The id started with `.`.
    #[error("id may not start with '.'")]
    LeadingDot,
    /// The id contained `..`.
    #[error("id may not contain '..'")]
    DoubleDot,
}

fn validate_id(s: &str) -> Result<(), IdError> {
    if s.is_empty() {
        return Err(IdError::Empty);
    }
    if s.len() > 64 {
        return Err(IdError::Length);
    }
    if s.starts_with('.') {
        return Err(IdError::LeadingDot);
    }
    if s.contains("..") {
        return Err(IdError::DoubleDot);
    }
    for c in s.chars() {
        if !(c.is_ascii_alphanumeric() || c == '-' || c == '_' || c == '.') {
            return Err(IdError::Char(c));
        }
    }
    Ok(())
}

impl SessionId {
    /// Validate `s` as a session id.
    ///
    /// # Errors
    ///
    /// Returns [`IdError`] if `s` violates any of the rules above.
    pub fn new(s: impl Into<String>) -> Result<Self, IdError> {
        let s = s.into();
        validate_id(&s)?;
        Ok(Self(s))
    }

    /// Generate a fresh id like `s-20260530-153012-4d2-1f`.
    ///
    /// The parts are a wall-clock stamp (so ids sort and read
    /// chronologically), this process's pid, and a per-process counter.
    /// Uniqueness rests on the last two and is therefore structural
    /// rather than probabilistic: the counter strictly increases, so no
    /// two ids from one process can collide however fast they are minted,
    /// and the pid separates concurrent processes.
    ///
    /// This matters more than it looks. The timestamp has one-second
    /// resolution, so a loop creating sessions — exactly what the strain
    /// tests do — produces thousands of ids sharing a stamp. An earlier
    /// version hashed the clock into a 16-bit suffix, which collides in
    /// practice at that rate and surfaced as a spurious `SessionExists`.
    #[must_use]
    pub fn generate() -> Self {
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);

        let seq = COUNTER.fetch_add(1, Ordering::Relaxed);
        Self(format!(
            "s-{}-{:x}-{:x}",
            Utc::now().format("%Y%m%d-%H%M%S"),
            std::process::id(),
            seq
        ))
    }

    /// The id as a string slice.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for SessionId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // `pad`, not `write_str`: the CLI prints ids in aligned columns
        // with `{:<32}`, and `write_str` silently ignores the width,
        // producing a ragged table.
        f.pad(&self.0)
    }
}

impl FromStr for SessionId {
    type Err = IdError;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::new(s)
    }
}

impl TryFrom<String> for SessionId {
    type Error = IdError;
    fn try_from(s: String) -> Result<Self, Self::Error> {
        Self::new(s)
    }
}

impl From<SessionId> for String {
    fn from(id: SessionId) -> String {
        id.0
    }
}

/// Health/status snapshot for a session.
///
/// The supervisor holds this in memory and updates it as a session moves
/// through bring-up, so it is always the truth rather than a report that
/// might be stale. The previous file-backed design needed a heartbeat and
/// a staleness ladder (`ready` → `stalled` → `dead`) purely to guess
/// whether the process that wrote the file was still alive; with an
/// in-memory owner that question cannot arise, and those states are gone.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct SessionHealth {
    /// When this snapshot was taken.
    pub timestamp: DateTime<Utc>,

    /// `true` once the session is ready to accept execs.
    pub healthy: bool,

    /// Human-readable lifecycle phase. One of the [`state`] constants:
    /// `"starting"`, `"preparing"`, `"pulling"`, `"building"`,
    /// `"ready"`, `"stopping"`, `"stopped"`, `"failed"`.
    ///
    /// [`state`]: self::state
    pub state: Option<String>,

    /// `true` if the session will never become healthy and must be
    /// discarded.
    pub terminal: bool,

    /// Detail for the current phase: pull progress, the node being
    /// started, or the error that made the session terminal.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
}

/// The lifecycle phases a session reports through
/// [`SessionHealth::state`].
pub mod state {
    /// Registered, nothing done yet.
    pub const STARTING: &str = "starting";
    /// Resolving the profile, emulator injection and container provider.
    pub const PREPARING: &str = "preparing";
    /// Pulling a container image.
    pub const PULLING: &str = "pulling";
    /// Building a derived container image for profile hacks.
    pub const BUILDING: &str = "building";
    /// Accepting execs.
    pub const READY: &str = "ready";
    /// Tearing down.
    pub const STOPPING: &str = "stopping";
    /// Torn down.
    pub const STOPPED: &str = "stopped";
    /// Bring-up failed; the session is terminal.
    pub const FAILED: &str = "failed";

    /// Phases that are externally bounded and must not count against a
    /// readiness timeout: pulling a multi-gigabyte image or building a
    /// derived one can legitimately take minutes, and timing that out
    /// would report a healthy-but-slow session as broken.
    #[must_use]
    pub fn is_externally_bounded(state: Option<&str>) -> bool {
        matches!(state, Some(PULLING | BUILDING))
    }
}

impl SessionHealth {
    /// A snapshot for a non-terminal phase.
    #[must_use]
    pub fn phase(healthy: bool, state: &str, message: Option<String>) -> Self {
        Self {
            timestamp: Utc::now(),
            healthy,
            state: Some(state.to_string()),
            terminal: false,
            message,
        }
    }

    /// A terminal failure snapshot carrying the reason.
    #[must_use]
    pub fn failed(message: impl Into<String>) -> Self {
        Self {
            timestamp: Utc::now(),
            healthy: false,
            state: Some(state::FAILED.to_string()),
            terminal: true,
            message: Some(message.into()),
        }
    }

    /// Whether waiting on this session can still change the answer.
    #[must_use]
    pub fn is_settled(&self) -> bool {
        self.healthy || self.terminal
    }
}

/// Everything an emulator backend needs to act on a session.
///
/// Backends used to receive only a [`SessionId`] and recover the rest by
/// reading the session's `def.json` off disk. Sessions no longer have a
/// `def.json` — and a backend reaching back into mirage's storage to
/// answer a question mirage already knows the answer to was the wrong
/// shape regardless — so the resolved context is passed in instead.
#[derive(Debug, Clone)]
pub struct SessionContext {
    /// The session this context describes.
    pub id: SessionId,
    /// Its fully-resolved profile: every [`MaybeRef`] followed.
    pub profile: ProfileDef,
    /// Directory the backend may materialise runtime assets in (config
    /// files, discovery files, daemon sockets). Created before the
    /// backend is called and removed with the session.
    pub runtime_dir: PathBuf,
    /// Whether the session asked for out-of-process emulator daemon mode.
    pub daemon: bool,
}

impl SessionContext {
    /// The emulator configuration this session runs under.
    #[must_use]
    pub fn emulator(&self) -> &crate::emulator::EmulatorDef {
        &self.profile.emulator
    }
}

/// A session definition: user-facing parameters used to start a session.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SessionDef {
    /// The session's id.
    pub id: SessionId,

    /// Profile (inline or by name) the supervisor builds the emulator
    /// runtime from.
    pub profile: MaybeRef<ProfileDef>,

    /// Working directory used as the default `cwd` for execs.
    pub workdir: String,

    /// Run the emulator in out-of-process *daemon* mode for this session
    /// (e.g. rocjitsu's per-node daemon) instead of the default
    /// in-process (local) emulation. Off by default; opt in with
    /// `mirage run --daemon` / `mirage session start --daemon`.
    #[serde(default)]
    pub daemon: bool,

    /// When this session was created (wall-clock).
    pub created_at: DateTime<Utc>,
}

/// Parameters for creating a session.
#[derive(Debug, Clone)]
pub struct CreateSessionRequest {
    /// Pre-validated id; if `None` mirage generates one.
    pub id: Option<SessionId>,
    /// Inline or by-name profile reference. Containerisation (image,
    /// mounts, provider) travels with the profile via
    /// [`crate::profile::ContainerizedDef`].
    pub profile: MaybeRef<ProfileDef>,
    /// Working directory used as the default cwd for execs.
    pub workdir: String,
    /// Run the emulator in out-of-process daemon mode for this session
    /// instead of in-process emulation.
    pub daemon: bool,
}

/// Aggregate view of a live session: what it was created from, how it is
/// doing, and what it is running on.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SessionState {
    /// The definition the session was created from.
    pub def: SessionDef,
    /// Its current health snapshot.
    pub health: SessionHealth,

    /// Container runtime state for containerised sessions: the
    /// provider, network, and per-node containers the supervisor
    /// launched. `None` for non-containerised sessions, or before
    /// bring-up has started the containers.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub container: Option<crate::container::ContainerState>,
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

    use super::*;

    #[test]
    fn id_valid() {
        SessionId::new("ok_name-123.tag").unwrap();
    }

    #[test]
    fn id_invalid() {
        assert_eq!(SessionId::new("").unwrap_err(), IdError::Empty);
        assert!(matches!(
            SessionId::new("/oops").unwrap_err(),
            IdError::Char('/')
        ));
        assert_eq!(SessionId::new(".hidden").unwrap_err(), IdError::LeadingDot);
        assert_eq!(SessionId::new("a..b").unwrap_err(), IdError::DoubleDot);
        assert_eq!(SessionId::new("x".repeat(65)).unwrap_err(), IdError::Length);
        // 64 is the boundary and must be accepted.
        assert!(SessionId::new("x".repeat(64)).is_ok());
    }

    #[test]
    fn id_generate_is_valid() {
        let id = SessionId::generate();
        validate_id(id.as_str()).unwrap();
    }

    #[test]
    fn generated_ids_are_unique_under_rapid_creation() {
        // The strain tests create sessions as fast as the process can go.
        // Ids embed a one-second-resolution timestamp, so uniqueness rests
        // entirely on the suffix; a clock-derived suffix used to collide
        // here and surface as a spurious `SessionExists`.
        let ids: std::collections::HashSet<String> = (0..10_000)
            .map(|_| SessionId::generate().as_str().to_string())
            .collect();
        assert_eq!(ids.len(), 10_000, "generated ids must never collide");
    }

    #[test]
    fn generated_ids_are_unique_across_threads() {
        let handles: Vec<_> = (0..8)
            .map(|_| {
                std::thread::spawn(|| {
                    (0..1_000)
                        .map(|_| SessionId::generate().as_str().to_string())
                        .collect::<Vec<_>>()
                })
            })
            .collect();
        let mut all = std::collections::HashSet::new();
        let mut total = 0usize;
        for h in handles {
            for id in h.join().unwrap() {
                total += 1;
                all.insert(id);
            }
        }
        assert_eq!(all.len(), total);
    }

    #[test]
    fn health_phase_and_failed_are_distinguishable() {
        let starting = SessionHealth::phase(false, state::STARTING, None);
        assert!(!starting.is_settled());
        assert!(!starting.terminal);

        let ready = SessionHealth::phase(true, state::READY, None);
        assert!(ready.is_settled());
        assert!(ready.healthy);

        let failed = SessionHealth::failed("image pull failed");
        assert!(failed.is_settled());
        assert!(failed.terminal);
        assert!(!failed.healthy);
        assert_eq!(failed.message.as_deref(), Some("image pull failed"));
        assert_eq!(failed.state.as_deref(), Some(state::FAILED));
    }

    #[test]
    fn image_phases_do_not_count_against_a_readiness_timeout() {
        assert!(state::is_externally_bounded(Some(state::PULLING)));
        assert!(state::is_externally_bounded(Some(state::BUILDING)));
        assert!(!state::is_externally_bounded(Some(state::STARTING)));
        assert!(!state::is_externally_bounded(Some(state::READY)));
        assert!(!state::is_externally_bounded(None));
    }

    #[test]
    fn session_id_round_trips_through_serde() {
        let id = SessionId::new("round-trip.1").unwrap();
        let json = serde_json::to_string(&id).unwrap();
        assert_eq!(json, "\"round-trip.1\"");
        let back: SessionId = serde_json::from_str(&json).unwrap();
        assert_eq!(back, id);
        // An invalid id must be rejected at deserialization, not silently
        // accepted and later used as a directory name.
        assert!(serde_json::from_str::<SessionId>("\"../escape\"").is_err());
    }
}
