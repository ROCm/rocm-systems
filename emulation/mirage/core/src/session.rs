//! Session: the long-lived context within which execs run.

use std::str::FromStr;
use std::time::Duration;

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

#[derive(Debug, Error, PartialEq, Eq)]
pub enum IdError {
    #[error("id may not be empty")]
    Empty,
    #[error("id must be at most 64 characters")]
    Length,
    #[error("id contains invalid character: {0:?}")]
    Char(char),
    #[error("id may not start with '.'")]
    LeadingDot,
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
    pub fn new(s: impl Into<String>) -> Result<Self, IdError> {
        let s = s.into();
        validate_id(&s)?;
        Ok(Self(s))
    }

    /// Generate a fresh id like `s-20260530-153012-abcd`.
    pub fn generate() -> Self {
        let now = Utc::now();
        let suffix: u32 = rand_suffix();
        Self(format!(
            "s-{}-{:04x}",
            now.format("%Y%m%d-%H%M%S"),
            suffix & 0xffff
        ))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for SessionId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
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

fn rand_suffix() -> u32 {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.subsec_nanos())
        .unwrap_or(0);
    let pid = std::process::id();
    nanos.wrapping_mul(2_654_435_761).wrapping_add(pid)
}

/// Health/status indicator for a session, written by the host.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct SessionHealth {
    pub timestamp: DateTime<Utc>,

    /// `true` once the host is ready to accept new execs.
    pub healthy: bool,

    /// Human-readable lifecycle phase.
    ///
    /// One of: `"starting"`, `"pulling"`, `"ready"`, `"degraded"`,
    /// `"stalled"`, `"dead"`, `"stopping"`, `"stopped"`, `"error"`.
    pub state: Option<String>,

    /// `true` if the host will never become (re-)healthy and the
    /// session must be discarded.
    pub terminal: bool,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
}

/// How often a running host re-stamps [`SessionHealth::timestamp`] to
/// prove it is still alive ("heartbeat"). Readers compare the recorded
/// timestamp against [`HEALTH_STALLED_AFTER`] / [`HEALTH_DEAD_AFTER`] to
/// detect a host that stopped beating.
pub const HEALTH_HEARTBEAT_INTERVAL: Duration = Duration::from_secs(2);

/// A healthy host whose heartbeat is older than this is reported as
/// `stalled` (not healthy, but not yet terminal): it may still recover.
pub const HEALTH_STALLED_AFTER: Duration = Duration::from_secs(6);

/// A healthy host whose heartbeat is older than this is reported as
/// `dead` (terminal): the host process is assumed to have crashed.
pub const HEALTH_DEAD_AFTER: Duration = Duration::from_secs(20);

impl SessionHealth {
    /// Escalate a previously-healthy report whose heartbeat has gone
    /// stale. A host that reached `ready` re-stamps its timestamp every
    /// [`HEALTH_HEARTBEAT_INTERVAL`]; if that stops (the process crashed,
    /// was killed, or hung), the on-disk record keeps claiming `healthy`
    /// forever. Readers call this so a stale record is surfaced as
    /// `stalled` and then terminally `dead` with an actionable message,
    /// instead of a session that looks ready but never responds.
    ///
    /// Only `healthy`, non-terminal reports are escalated: a host that
    /// cleanly reached `stopped`/`error` (or is still `starting`/
    /// `pulling`, which can legitimately take a while without a
    /// heartbeat) is left untouched.
    pub fn escalate_if_stale(mut self) -> Self {
        if !self.healthy || self.terminal {
            return self;
        }
        let age = Utc::now()
            .signed_duration_since(self.timestamp)
            .to_std()
            .unwrap_or_default();
        if age >= HEALTH_DEAD_AFTER {
            self.healthy = false;
            self.terminal = true;
            self.state = Some("dead".to_string());
            self.message = Some(format!(
                "host unresponsive: no heartbeat for {}s (last update {}). \
                 The host process has likely crashed; destroy and recreate the session.",
                age.as_secs(),
                self.timestamp.to_rfc3339()
            ));
        } else if age >= HEALTH_STALLED_AFTER {
            self.healthy = false;
            self.state = Some("stalled".to_string());
            self.message = Some(format!(
                "host stalled: no heartbeat for {}s (last update {}). \
                 Waiting for the host to recover.",
                age.as_secs(),
                self.timestamp.to_rfc3339()
            ));
        }
        self
    }
}

/// A session definition: user-facing parameters used to start a session.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SessionDef {
    pub id: SessionId,

    /// Profile (inline or by name) used by the host to construct the
    /// emulator runtime.
    pub profile: MaybeRef<ProfileDef>,

    /// Working directory used as the default `cwd` for execs.
    pub workdir: String,

    /// When this session was created (wall-clock).
    pub created_at: DateTime<Utc>,
}

/// Aggregate view returned by `MirageCtl::session_state`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SessionState {
    pub def: SessionDef,
    pub health: SessionHealth,

    /// Container runtime state for containerised sessions: the
    /// provider, network, and per-node containers the host launched.
    /// `None` for non-containerised sessions or before the host has
    /// started the containers.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub container: Option<crate::container::ContainerState>,
}

#[cfg(test)]
mod tests {
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
    }

    #[test]
    fn id_generate_is_valid() {
        let id = SessionId::generate();
        validate_id(id.as_str()).unwrap();
    }

    fn healthy_at(age: Duration) -> SessionHealth {
        SessionHealth {
            timestamp: Utc::now() - chrono::Duration::from_std(age).unwrap(),
            healthy: true,
            state: Some("ready".to_string()),
            terminal: false,
            message: None,
        }
    }

    #[test]
    fn fresh_health_is_unchanged() {
        let h = healthy_at(Duration::from_secs(0)).escalate_if_stale();
        assert!(h.healthy);
        assert!(!h.terminal);
        assert_eq!(h.state.as_deref(), Some("ready"));
    }

    #[test]
    fn stale_health_becomes_stalled() {
        let h = healthy_at(HEALTH_STALLED_AFTER + Duration::from_secs(1)).escalate_if_stale();
        assert!(!h.healthy);
        assert!(!h.terminal);
        assert_eq!(h.state.as_deref(), Some("stalled"));
        assert!(h.message.unwrap().contains("stalled"));
    }

    #[test]
    fn long_stale_health_becomes_dead() {
        let h = healthy_at(HEALTH_DEAD_AFTER + Duration::from_secs(1)).escalate_if_stale();
        assert!(!h.healthy);
        assert!(h.terminal);
        assert_eq!(h.state.as_deref(), Some("dead"));
        assert!(h.message.unwrap().contains("unresponsive"));
    }

    #[test]
    fn unhealthy_record_is_not_escalated() {
        // A cleanly-stopped host (healthy=false) with an old timestamp
        // must not be flipped into a scary "dead" report.
        let h = SessionHealth {
            timestamp: Utc::now() - chrono::Duration::seconds(3600),
            healthy: false,
            state: Some("stopped".to_string()),
            terminal: false,
            message: None,
        }
        .escalate_if_stale();
        assert_eq!(h.state.as_deref(), Some("stopped"));
        assert!(!h.terminal);
    }
}
