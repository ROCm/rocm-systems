//! Emulator backends: the trait every GPU emulator integration
//! implements, and the link-time registry they are discovered through.

use serde::{Deserialize, Serialize};

use crate::{
    common::{MaybeRef, SimpleMap},
    config::OptionDef,
    error::Result,
    exec::InjectionDef,
    plugin::PluginsDef,
    profile::ProfileDef,
    session::{SessionContext, SessionHealth},
    topology::TopologyDef,
};

/// How faithfully the emulator models the device.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum ExecMode {
    /// Correct results, no timing model.
    #[default]
    Functional,
    /// Model device timing as well as results.
    Clocked,
}

/// The canonical lowercase name of an emulator backend.
pub type EmulatorKind = String;

/// The emulator half of a profile: which backend, how it runs, and the
/// system it emulates.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EmulatorDef {
    /// which emulator backend to use, e.g. "rocjitsu"
    pub emulator: EmulatorKind,

    /// plugins to use with the emulator, e.g. "rocjitsu" plugin for AMD GPU simulation
    pub plugins: PluginsDef,

    /// Functional or clocked emulation.
    pub exec_mode: ExecMode,

    /// extra options to configure the emulator, e.g. {"gpu_model": "cdna3"}
    pub options: SimpleMap,

    /// system topology (rack/node/GPU layout + per-GPU agent).
    pub topology: MaybeRef<TopologyDef>,
}

/// Whether the host's hardware/environment can actually run an
/// emulator. This is distinct from [`EmulatorBackend::installed`]:
/// an emulator can be installed yet unsupported (e.g. HotSwap installed
/// on a machine with no compatible physical GPU), or supported yet not
/// installed. Both signals are surfaced so the UX/CLI can explain
/// exactly what a user needs to do.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SupportStatus {
    /// `true` if this host meets the emulator's hardware/environment
    /// requirements.
    pub supported: bool,
    /// Human-readable explanation of the support decision — what was
    /// required and what was found. Always populated so the UX/CLI can
    /// show a reason whether or not the host is supported.
    pub reason: String,
}

impl SupportStatus {
    /// The host meets this emulator's requirements.
    pub fn supported(reason: impl Into<String>) -> Self {
        Self {
            supported: true,
            reason: reason.into(),
        }
    }

    /// The host does not meet this emulator's requirements.
    pub fn unsupported(reason: impl Into<String>) -> Self {
        Self {
            supported: false,
            reason: reason.into(),
        }
    }
}

/// The static identity of an emulator backend: its name, version, a
/// short human-readable blurb, and the schema of options it accepts.
/// Live runtime status (installed / supported) is reported separately
/// through the [`EmulatorBackend`] trait methods.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EmulatorDescription {
    pub name: String,
    pub version: String,
    pub description: String,
    /// Schema of the options this backend accepts (empty when none).
    pub options_schema: Vec<OptionDef>,
}

/// A backend integration mirage can drive to emulate a workload.
///
/// Every emulator (`rocjitsu`, `rocjitsu-dbt`, `hotswap`, …)
/// lives in its own crate and registers a single stateless
/// implementation of this trait into the global registry via
/// [`inventory`]. The core and control-plane crates never name a
/// concrete backend: they look one up by its [`EmulatorKind`] with
/// [`get_emulator_backend`] and dispatch through this trait, so all
/// backend-specific behaviour stays inside the backend's own crate.
///
/// Implementations are stateless singletons (a unit struct is typical);
/// any per-session state is resolved on demand from the on-disk session
/// definition (see [`crate::session::resolve_profile`]).
pub trait EmulatorBackend: Sync + Send + std::fmt::Debug {
    /// Returns a description of the emulator, including its name, version, and a brief description.
    fn description(&self) -> EmulatorDescription;

    /// Bring the emulator up for the given profile.
    ///
    /// # Errors
    ///
    /// Returns a human-readable reason on failure.
    fn boot(&self, def: &ProfileDef) -> std::result::Result<(), String>;

    /// Schema for the options that this emulator supports. Empty when
    /// the emulator takes no options.
    fn options(&self) -> Vec<OptionDef>;

    /// Shut the emulator down for a session and release any resources it
    /// holds. Called exactly once, during session teardown, after every
    /// workload process is gone.
    fn shutdown(&self, ctx: &SessionContext);

    /// Validate that `def` can be used with this emulator before it is
    /// persisted.
    ///
    /// # Errors
    ///
    /// Returns a human-readable reason on rejection.
    fn validate_profile(&self, def: &ProfileDef) -> std::result::Result<(), String>;

    /// Returns true if the emulator is properly installed and can be used.
    fn installed(&self) -> bool;

    /// check if the emulator is supported on this host, i.e. meets the hardware/environment requirements to run. This is a stronger condition than `installed`: an emulator can be installed but unsupported (e.g. HotSwap installed on a machine with no compatible physical GPU), or supported but not installed.
    fn supported(&self) -> SupportStatus;

    /// Discovers available plugins for the emulator.
    fn discover_plugins(&self) -> Vec<PluginsDef>;

    /// Health of the emulator for a session, e.g. whether the underlying
    /// runtime is present and responsive.
    fn health(&self, ctx: &SessionContext) -> SessionHealth;

    /// Compute the env vars / `LD_PRELOAD` / files to inject into a
    /// workload run under this emulator. Returns an error when the
    /// emulator is selected but its runtime library or assets are
    /// missing, so a misconfigured session fails loudly instead of
    /// silently running unemulated.
    ///
    /// `ctx` carries the session's resolved profile and a scratch
    /// directory the backend may materialise runtime assets in.
    ///
    /// # Errors
    ///
    /// Returns an error when the backend is selected but cannot produce a
    /// usable injection — a missing runtime library, an unresolvable
    /// topology or agent reference, an unwritable scratch directory. This
    /// must fail loudly: silently returning an empty injection would run
    /// the workload unemulated on whatever hardware is actually present.
    fn injection_def(&self, ctx: &SessionContext) -> Result<InjectionDef>;

    /// Start a host-side emulator *daemon* for `session`, if this
    /// backend hosts one.
    ///
    /// The supervisor calls this once during session bring-up, before
    /// any exec runs. A backend that emulates the GPU out-of-process
    /// (e.g. rocjitsu's daemon, which owns the simulated device and
    /// serves the workload's KFD ioctls over a Unix socket) stands the
    /// daemon up here and returns a handle the supervisor keeps alive for
    /// the whole session, stopping it during teardown after every
    /// workload process has exited.
    ///
    /// Returns `Ok(None)` — the default — for backends that need no
    /// daemon (`hotswap`, or rocjitsu when its runtime library
    /// is not installed and the exec will fail loudly anyway).
    ///
    /// # Errors
    ///
    /// Returns `Err` only when a daemon was expected but could not be
    /// started.
    fn start_daemon(&self, ctx: &SessionContext) -> Result<Option<Box<dyn EmulatorDaemon>>> {
        let _ = ctx;
        Ok(None)
    }
}

/// A running, host-side emulator daemon owned by a per-node host.
///
/// The host holds the boxed handle for the lifetime of the session and
/// drops it on shutdown. Implementations must tear the daemon down in
/// their [`Drop`] so cleanup happens even if the host panics; [`stop`]
/// is provided for an explicit, ordered shutdown and defaults to simply
/// dropping the handle.
///
/// [`stop`]: EmulatorDaemon::stop
pub trait EmulatorDaemon: Send + std::fmt::Debug {
    /// Stop the daemon and release its resources. Blocking, so the
    /// supervisor calls it from a blocking task rather than inline on the
    /// runtime. The default drops the handle, which must perform the
    /// teardown.
    fn stop(self: Box<Self>) {}
}

/// One registry entry: the canonical [`EmulatorKind`] name plus the
/// backend that handles it. Each backend crate submits exactly one of
/// these via [`inventory::submit!`]; the backend is a `'static`
/// reference to a stateless singleton (typically a unit struct), so the
/// entry is const-constructible and needs no allocation.
#[derive(Debug)]
pub struct EmulatorBackendDef {
    /// Canonical lowercase name the backend registers under (the value
    /// stored in [`EmulatorDef::emulator`]).
    pub kind: &'static str,
    /// The backend implementation.
    pub backend: &'static dyn EmulatorBackend,
}

inventory::collect!(EmulatorBackendDef);

/// Look up the [`EmulatorBackend`] registered for `kind`, or `None` if
/// no backend with that name was compiled into this binary (e.g. its
/// crate's feature is disabled).
#[must_use]
pub fn get_emulator_backend(kind: &str) -> Option<&'static dyn EmulatorBackend> {
    inventory::iter::<EmulatorBackendDef>
        .into_iter()
        .find(|def| def.kind == kind)
        .map(|def| def.backend)
}
